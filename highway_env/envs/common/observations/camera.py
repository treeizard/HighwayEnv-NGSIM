"""Lane-camera and combined lidar-camera observation types."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is optional at runtime
    cKDTree = None

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

from .base import ObservationType, _ObservationProfiler, _ObstacleSpatialIndex
from .lidar import LidarObservation


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


@lru_cache(maxsize=32)
def _upper_triangle_pair_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return stable lexicographic point-pair indices for the small lane fit."""
    return np.triu_indices(int(count), k=1)


def _lane_state_from_camera(
    camera_observation: np.ndarray,
    *,
    maximum_range: float,
) -> tuple[float, float, float]:
    """Estimate local lane state only from the current lane-camera returns.

    The camera rows are normalized ego-frame boundary points.  Fitting one line
    per visible side keeps this actor input causal and sensor-derived: no lane
    object, lane index, route, or future geometry is consulted.
    """
    values = np.asarray(camera_observation, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected lane-camera rows [N,3], got {values.shape}.")
    scale = float(maximum_range)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"maximum_range must be positive and finite, got {scale}.")
    points = values[:, 1:3] * scale
    valid = (
        (values[:, 0] >= 0.5)
        & np.all(np.isfinite(points), axis=1)
        & (points[:, 0] >= 0.0)
        # Use the near field so boundary side can be identified without a
        # privileged lane label even when the road bends farther ahead.
        & (points[:, 0] <= min(scale, 12.0))
    )

    side_points = points[valid]
    candidate_slopes = np.zeros((0,), dtype=float)
    candidate_intercepts = np.zeros((0,), dtype=float)
    candidate_residuals = np.zeros((0, len(side_points)), dtype=float)
    candidate_inliers = np.zeros((0, len(side_points)), dtype=bool)
    candidate_counts = np.zeros((0,), dtype=np.int64)
    if len(side_points) >= 3:
        first, second = _upper_triangle_pair_indices(len(side_points))
        dx = side_points[second, 0] - side_points[first, 0]
        usable = np.abs(dx) >= 0.75
        first = first[usable]
        second = second[usable]
        dx = dx[usable]
        if dx.size:
            slopes = (side_points[second, 1] - side_points[first, 1]) / dx
            usable = np.abs(slopes) <= 1.0
            first = first[usable]
            slopes = slopes[usable]
            if slopes.size:
                intercepts = (
                    side_points[first, 1] - slopes * side_points[first, 0]
                )
                residuals = np.abs(
                    side_points[None, :, 1]
                    - (
                        slopes[:, None] * side_points[None, :, 0]
                        + intercepts[:, None]
                    )
                )
                inliers = residuals <= 0.15
                candidate_slopes = slopes
                candidate_intercepts = intercepts
                candidate_residuals = residuals
                candidate_inliers = inliers
                candidate_counts = np.sum(inliers, axis=1, dtype=np.int64)

    def fit_side(intercept_sign: float) -> tuple[float, float] | None:
        if candidate_slopes.size == 0:
            return None
        eligible = np.flatnonzero(
            (candidate_intercepts * intercept_sign > 0.0)
            & (candidate_counts >= 3)
        )
        if eligible.size == 0:
            return None
        maximum_count = int(np.max(candidate_counts[eligible]))
        eligible = eligible[candidate_counts[eligible] == maximum_count]
        minimum_abs_intercept = float(
            np.min(np.abs(candidate_intercepts[eligible]))
        )
        eligible = eligible[
            np.abs(candidate_intercepts[eligible]) == minimum_abs_intercept
        ]
        if eligible.size > 1:
            eligible_residuals = np.where(
                candidate_inliers[eligible],
                candidate_residuals[eligible],
                np.inf,
            )
            ordered_residuals = np.sort(eligible_residuals, axis=1)
            middle = maximum_count // 2
            if maximum_count % 2:
                median_residuals = ordered_residuals[:, middle]
            else:
                median_residuals = np.mean(
                    ordered_residuals[:, middle - 1 : middle + 1], axis=1
                )
            best_index = int(eligible[int(np.argmin(median_residuals))])
        else:
            best_index = int(eligible[0])
        inlier_points = side_points[candidate_inliers[best_index]]
        x = inlier_points[:, 0]
        y = inlier_points[:, 1]
        design = np.column_stack([x, np.ones_like(x)])
        weights = 1.0 / (1.0 + x / 15.0)
        coefficients, *_ = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]),
            y * np.sqrt(weights),
            rcond=None,
        )
        slope, intercept = (float(value) for value in coefficients)
        if (
            not np.isfinite(slope)
            or not np.isfinite(intercept)
            or intercept * intercept_sign <= 0.0
        ):
            return None
        return float(np.clip(slope, -1.0, 1.0)), float(
            np.clip(intercept, -scale, scale)
        )

    left = fit_side(1.0)
    right = fit_side(-1.0)
    available = [fit for fit in (left, right) if fit is not None]
    if not available:
        return 0.0, 0.0, 0.0
    lane_slope = float(np.mean([fit[0] for fit in available]))
    lane_heading_error = -float(np.arctan(lane_slope))
    if left is None or right is None:
        return 0.0, lane_heading_error, 0.0
    lane_center = 0.5 * (left[1] + right[1])
    lane_width = abs(left[1] - right[1])
    if not 2.4 <= lane_width <= 5.5:
        return 0.0, lane_heading_error, 0.0
    return -float(lane_center), lane_heading_error, float(lane_width)


def _route_free_ego_state(
    env: AbstractEnv,
    vehicle,
    history: dict[int, tuple[int, np.ndarray, float, float, np.ndarray]],
    camera_observation: np.ndarray,
    *,
    camera_maximum_range: float,
) -> np.ndarray:
    """Build causal IMU/odometry plus sensor-estimated local-lane features."""
    key = id(vehicle)
    step = int(getattr(env, "steps", 0))
    position = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=float)
    speed = float(getattr(vehicle, "speed", 0.0))
    heading = float(getattr(vehicle, "heading", 0.0))
    previous = history.get(key)
    if previous is not None and previous[0] == step:
        return previous[4].copy()
    frequency = max(1.0, float(getattr(env, "config", {}).get("policy_frequency", 10.0)))
    dt = 1.0 / frequency
    longitudinal_acceleration = 0.0
    lateral_speed = 0.0
    yaw_rate = 0.0
    if previous is not None and step > previous[0]:
        elapsed = max(dt, float(step - previous[0]) * dt)
        longitudinal_acceleration = (speed - previous[2]) / elapsed
        displacement = position - previous[1]
        lateral_axis = np.asarray([-np.sin(heading), np.cos(heading)], dtype=float)
        lateral_speed = float(displacement.dot(lateral_axis) / elapsed)
        yaw_rate = _wrap_angle(heading - previous[3]) / elapsed

    lane_offset, lane_heading_error, lane_width = _lane_state_from_camera(
        camera_observation,
        maximum_range=camera_maximum_range,
    )
    state = np.asarray(
        [
            float(max(getattr(vehicle, "LENGTH", 0.0), 0.0)),
            float(max(getattr(vehicle, "WIDTH", 0.0), 0.0)),
            speed,
            float(np.clip(longitudinal_acceleration, -20.0, 20.0)),
            float(np.clip(lateral_speed, -20.0, 20.0)),
            float(np.clip(yaw_rate, -4.0, 4.0)),
            float(lane_offset),
            float(np.sin(lane_heading_error)),
            float(np.cos(lane_heading_error)),
            float(max(lane_width, 0.0)),
        ],
        dtype=np.float32,
    )
    history[key] = (step, position.copy(), speed, heading, state.copy())
    return state


class SharedMultiAgentLidarCameraObservations(ObservationType):
    """
    Multi-agent LiDAR + lane-camera observation that shares static road caches.

    Each controlled vehicle still receives its own observation, but the expensive
    lane list / boundary-point structures are built only once for the entire
    multi-agent observer instead of once per controlled vehicle.
    """

    def __init__(
        self,
        env: AbstractEnv,
        lidar: dict | None = None,
        camera: dict | None = None,
        batch_road_edges: bool = False,
        ego_state_version: str = "legacy_v1",
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.lidar_observation = LidarObservation(env, **(lidar or {}))
        self.camera_observation = LaneCameraObservation(env, **(camera or {}))
        self.batch_road_edges = bool(batch_road_edges)
        self.ego_state_version = str(ego_state_version)
        self._ego_history: dict[
            int, tuple[int, np.ndarray, float, float, np.ndarray]
        ] = {}

    def _ego_state_space(self) -> spaces.Box:
        if self.ego_state_version == "route_free_v2":
            return spaces.Box(
                low=np.full(10, -np.inf, dtype=np.float32),
                high=np.full(10, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
        return spaces.Box(
            low=np.array([-np.inf, -np.pi, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.pi, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    def _build_ego_state(
        self, vehicle, camera_observation: np.ndarray | None = None
    ) -> np.ndarray:
        if self.ego_state_version == "route_free_v2":
            if camera_observation is None:
                raise ValueError("route_free_v2 requires a current lane-camera observation.")
            return _route_free_ego_state(
                self.env,
                vehicle,
                self._ego_history,
                camera_observation,
                camera_maximum_range=self.camera_observation.maximum_range,
            )
        return np.array(
            [
                float(getattr(vehicle, "speed", 0.0)),
                float(getattr(vehicle, "heading", 0.0)),
                float(max(getattr(vehicle, "WIDTH", 0.0), 0.0)),
                float(max(getattr(vehicle, "LENGTH", 0.0), 0.0)),
            ],
            dtype=np.float32,
        )

    def space(self) -> spaces.Space:
        single_space = spaces.Tuple(
            [
                self.lidar_observation.space(),
                self.camera_observation.space(),
                self._ego_state_space(),
            ]
        )
        return spaces.Tuple([single_space for _ in self.env.controlled_vehicles])

    def observe(self) -> tuple:
        started = time.perf_counter()
        collect_started = time.perf_counter()
        obstacle_entries = self.lidar_observation.collect_obstacle_entries()
        _ObservationProfiler.record("shared_collect_obstacles", time.perf_counter() - collect_started)
        index_started = time.perf_counter()
        obstacle_index = _ObstacleSpatialIndex(obstacle_entries)
        _ObservationProfiler.record("shared_obstacle_index_build", time.perf_counter() - index_started)
        vehicles = list(self.env.controlled_vehicles)
        origins = np.asarray([vehicle.position for vehicle in vehicles], dtype=float)
        headings = np.asarray(
            [float(getattr(vehicle, "heading", 0.0)) for vehicle in vehicles],
            dtype=float,
        )
        if self.batch_road_edges:
            edge_started = time.perf_counter()
            edge_distances: list[np.ndarray | None] | np.ndarray = (
                self.lidar_observation._distance_to_road_edges_many(
                    origins=origins,
                    directions=np.stack(
                        [
                            self.lidar_observation.directions_for_heading(heading)
                            for heading in headings
                        ],
                        axis=0,
                    ),
                    max_range=self.lidar_observation.maximum_range,
                    coarse_step=self.lidar_observation.coarse_step,
                    refine_iters=self.lidar_observation.refine_iters,
                )
            )
            _ObservationProfiler.record(
                "shared_lidar_road_edge_many", time.perf_counter() - edge_started
            )
        else:
            edge_distances = [None] * len(vehicles)
        query_started = time.perf_counter()
        candidate_entries = obstacle_index.query_many(origins, self.lidar_observation.maximum_range)
        _ObservationProfiler.record("shared_obstacle_candidate_query", time.perf_counter() - query_started)
        observations = []
        for vehicle, vehicle_obstacle_entries, vehicle_edge_distances in zip(
            vehicles, candidate_entries, edge_distances, strict=True
        ):
            self.lidar_observation.observer_vehicle = vehicle
            self.camera_observation.observer_vehicle = vehicle
            camera_observation = self.camera_observation.observe()
            ego_started = time.perf_counter()
            ego_state = self._build_ego_state(vehicle, camera_observation)
            _ObservationProfiler.record(
                "shared_route_free_ego_state", time.perf_counter() - ego_started
            )
            observations.append(
                (
                    self.lidar_observation.observe(
                        obstacle_entries=vehicle_obstacle_entries,
                        edge_dists=vehicle_edge_distances,
                    ),
                    camera_observation,
                    ego_state,
                )
            )
        _ObservationProfiler.record("shared_observation_total", time.perf_counter() - started)
        return tuple(observations)

class LaneCameraObservation(LidarObservation):
    """
    Forward-facing topology camera.

    The sensor ignores dynamic/static obstacles and only returns road/lane boundary
    points that fall within an ego-centric cone. Each row is:
      [presence, x, y]
    where (x, y) is the boundary point expressed in the ego frame.
    """

    PRESENCE = 0
    X = 1
    Y = 2

    def __init__(
        self,
        env,
        cells: int = 21,
        maximum_range: float = 60.0,
        field_of_view: float = np.pi / 2,
        normalize: bool = True,
        longitudinal_resolution: float = 1.0,
        coarse_step: float | None = None,
        refine_iters: int = 8,
        **kwargs,
    ):
        super().__init__(
            env,
            cells=cells,
            maximum_range=maximum_range,
            normalize=normalize,
            edge_as_return=True,
            coarse_step=coarse_step,
            refine_iters=refine_iters,
            **kwargs,
        )
        self.field_of_view = float(field_of_view)
        self.longitudinal_resolution = float(longitudinal_resolution)
        self.grid = np.zeros((self.cells, 3), dtype=np.float32)
        self._boundary_points_cache = self._collect_boundary_points()
        self._boundary_points_tree = self._build_boundary_points_tree()
        self._bin_edges = np.linspace(
            -self.field_of_view / 2.0,
            self.field_of_view / 2.0,
            self.cells + 1,
        )

    def space(self) -> spaces.Space:
        high = 1.0 if self.normalize else self.maximum_range
        low = np.tile(np.array([0.0, -high, -high], dtype=np.float32), (self.cells, 1))
        high_arr = np.tile(np.array([1.0, high, high], dtype=np.float32), (self.cells, 1))
        return spaces.Box(
            shape=(self.cells, 3),
            low=low,
            high=high_arr,
            dtype=np.float32,
        )

    def observe(self) -> np.ndarray:
        vehicle = self.observer_vehicle
        heading = float(getattr(vehicle, "heading", 0.0))
        traced = self.trace_topology(vehicle.position, heading)
        started = time.perf_counter()
        obs = traced.copy()
        if self.normalize:
            obs[:, 1:] /= self.maximum_range
        _ObservationProfiler.record("lane_camera_normalize_copy", time.perf_counter() - started)
        return obs

    def trace_topology(self, origin: np.ndarray, heading: float) -> np.ndarray:
        started = time.perf_counter()
        self.origin = np.array(origin, dtype=float).copy()
        self.grid.fill(0.0)

        if self._lanes_cache is None or getattr(self.env, "road", None) is None:
            self._lanes_cache = self._collect_lanes()
        if self._boundary_points_cache is None:
            self._boundary_points_cache = self._collect_boundary_points()
            self._boundary_points_tree = self._build_boundary_points_tree()

        if self._boundary_points_cache is None or len(self._boundary_points_cache) == 0:
            _ObservationProfiler.record("lane_camera", time.perf_counter() - started)
            return self.grid

        boundary_points = self._boundary_points_cache
        if self._boundary_points_tree is not None:
            query_started = time.perf_counter()
            indices = self._boundary_points_tree.query_ball_point(self.origin, self.maximum_range)
            _ObservationProfiler.record("lane_camera_query", time.perf_counter() - query_started)
            if not indices:
                _ObservationProfiler.record("lane_camera", time.perf_counter() - started)
                return self.grid
            boundary_points = self._boundary_points_cache[np.asarray(sorted(indices), dtype=np.int64)]

        transform_started = time.perf_counter()
        relative_points = boundary_points - self.origin
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        world_to_ego = np.array([[cos_h, sin_h], [-sin_h, cos_h]], dtype=float)
        ego_points = relative_points @ world_to_ego.T

        distances = np.linalg.norm(ego_points, axis=1)
        angles = np.arctan2(ego_points[:, 1], ego_points[:, 0])

        valid = (
            np.isfinite(distances)
            & np.isfinite(angles)
            & (ego_points[:, 0] >= 0.0)
            & (distances <= self.maximum_range)
            & (np.abs(angles) <= self.field_of_view / 2.0)
        )
        if not np.any(valid):
            _ObservationProfiler.record("lane_camera_transform_filter", time.perf_counter() - transform_started)
            _ObservationProfiler.record("lane_camera", time.perf_counter() - started)
            return self.grid

        ego_points = ego_points[valid]
        distances = distances[valid]
        angles = angles[valid]
        _ObservationProfiler.record("lane_camera_transform_filter", time.perf_counter() - transform_started)

        bin_started = time.perf_counter()
        bin_indices = np.digitize(angles, self._bin_edges[1:-1], right=False)
        original_order = np.arange(bin_indices.shape[0], dtype=np.int64)
        order = np.lexsort((original_order, distances, bin_indices))
        sorted_bins = bin_indices[order]
        first_in_bin = np.empty(sorted_bins.shape[0], dtype=bool)
        first_in_bin[0] = True
        first_in_bin[1:] = sorted_bins[1:] != sorted_bins[:-1]
        nearest = order[first_in_bin]
        nearest_bins = bin_indices[nearest]
        self.grid[nearest_bins, self.PRESENCE] = 1.0
        self.grid[nearest_bins, self.X] = ego_points[nearest, 0]
        self.grid[nearest_bins, self.Y] = ego_points[nearest, 1]
        _ObservationProfiler.record("lane_camera_binning", time.perf_counter() - bin_started)

        _ObservationProfiler.record("lane_camera", time.perf_counter() - started)
        return self.grid

    def _build_boundary_points_tree(self):
        if cKDTree is None or self._boundary_points_cache is None:
            return None
        points = np.asarray(self._boundary_points_cache, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            return None
        return cKDTree(points)

    def _collect_boundary_points(self) -> np.ndarray | None:
        if self._lanes_cache is None:
            self._lanes_cache = self._collect_lanes()
        if self._lanes_cache is None:
            return None

        points = []
        seen = set()
        resolution = max(0.5, self.longitudinal_resolution)

        for lane in self._lanes_cache:
            length = float(getattr(lane, "length", 0.0))
            if length <= 0.0 or not np.isfinite(length):
                continue

            longitudinals = np.arange(0.0, length + resolution, resolution, dtype=float)
            for longitudinal in longitudinals:
                width = float(lane.width_at(longitudinal))
                for lateral in (-0.5 * width, 0.5 * width):
                    point = lane.position(longitudinal, lateral)
                    if point is None or not np.all(np.isfinite(point)):
                        continue
                    key = tuple(np.round(point, 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    points.append(np.array(point, dtype=np.float32))

        if not points:
            return None
        return np.vstack(points)

class LidarCameraObservations(ObservationType):
    """
    Composite observation that returns:
      - full LiDAR obstacle/road-edge scan
      - forward-facing topology camera scan
    """

    def __init__(
        self,
        env: AbstractEnv,
        lidar: dict | None = None,
        camera: dict | None = None,
        ego_state_version: str = "legacy_v1",
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.lidar_observation = LidarObservation(env, **(lidar or {}))
        self.camera_observation = LaneCameraObservation(env, **(camera or {}))
        self.ego_state_version = str(ego_state_version)
        self._ego_history: dict[
            int, tuple[int, np.ndarray, float, float, np.ndarray]
        ] = {}

    def _ego_state_space(self) -> spaces.Box:
        if self.ego_state_version == "route_free_v2":
            return spaces.Box(
                low=np.full(10, -np.inf, dtype=np.float32),
                high=np.full(10, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
        return spaces.Box(
            low=np.array([-np.inf, -np.pi, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.pi, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    def _build_ego_state(
        self, vehicle, camera_observation: np.ndarray | None = None
    ) -> np.ndarray:
        if self.ego_state_version == "route_free_v2":
            if camera_observation is None:
                raise ValueError("route_free_v2 requires a current lane-camera observation.")
            return _route_free_ego_state(
                self.env,
                vehicle,
                self._ego_history,
                camera_observation,
                camera_maximum_range=self.camera_observation.maximum_range,
            )
        return np.array(
            [
                float(getattr(vehicle, "speed", 0.0)),
                float(getattr(vehicle, "heading", 0.0)),
                float(max(getattr(vehicle, "WIDTH", 0.0), 0.0)),
                float(max(getattr(vehicle, "LENGTH", 0.0), 0.0)),
            ],
            dtype=np.float32,
        )

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [
                self.lidar_observation.space(),
                self.camera_observation.space(),
                self._ego_state_space(),
            ]
        )

    def observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.lidar_observation.observer_vehicle = self.observer_vehicle
        self.camera_observation.observer_vehicle = self.observer_vehicle
        camera_observation = self.camera_observation.observe()
        ego_state = self._build_ego_state(
            self.observer_vehicle, camera_observation
        )
        return (
            self.lidar_observation.observe(),
            camera_observation,
            ego_state,
        )

# Backward-compatible alias while moving to the clearer name.
LidarCameraObservation = LaneCameraObservation

__all__ = [
    'SharedMultiAgentLidarCameraObservations',
    'LaneCameraObservation',
    'LidarCameraObservations',
    'LidarCameraObservation',
]
