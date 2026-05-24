"""Lidar observation and road-boundary tracing helpers."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.road.lane import AbstractLane, SineLane, StraightLane
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is optional at runtime
    cKDTree = None

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

from .base import (
    ObservationType,
    _LaneBounds,
    _LaneGeometry,
    _ObservationProfiler,
    _ObstacleEntry,
    _ObstacleSpatialIndex,
)

class LidarObservation(ObservationType):
    """
    LiDAR observation with road-edge clamping.

    Key behavior:
      - Each LiDAR beam is truncated at the first point where the ray leaves the road.
      - The returned distance is the minimum of:
            (distance to nearest obstacle intersection) and (distance to road edge).
      - If no obstacle is hit, the beam returns the road edge distance (dense boundary cue).

    Output:
      grid[:, 0] = distance (meters)
      grid[:, 1] = relative speed along beam direction (m/s)
    """

    DISTANCE = 0
    SPEED = 1

    def __init__(
        self,
        env,
        cells: int = 16,
        maximum_range: float = 60.0,
        normalize: bool = True,
        edge_as_return: bool = True,
        coarse_step: float | None = None,
        refine_iters: int = 8,
        use_topology_fast_path: bool = True,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.cells = int(cells)
        self.maximum_range = float(maximum_range)
        self.normalize = bool(normalize)
        self.use_topology_fast_path = bool(use_topology_fast_path)

        # If True: empty beams return road-edge distance; else they return maximum_range.
        self.edge_as_return = bool(edge_as_return)

        # Ray-march parameters
        self.coarse_step = float(coarse_step) if coarse_step is not None else max(0.5, self.maximum_range / 60.0)
        self.refine_iters = int(refine_iters)

        self.angle = 2 * np.pi / self.cells
        self.grid = np.ones((self.cells, 2), dtype=np.float32) * self.maximum_range
        self.origin = None
        self._directions = np.stack(
            [self.index_to_direction(i) for i in range(self.cells)],
            axis=0,
        )

        # Cache lanes for faster on-road checks (safe for typical static road networks)
        self._lanes_cache = self._collect_lanes()
        self._lane_geometry_cache: dict[int, _LaneGeometry | None] = {}
        self._lane_bounds_cache: list[_LaneBounds] | None = self._collect_lane_bounds()
        self._on_road_many_reference_compare = os.environ.get(
            "HIGHWAY_ENV_ON_ROAD_MANY_COMPARE",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self._on_road_many_mismatch: dict | None = None
        self._edge_lane_cache = self._collect_edge_lanes()
        self._edge_bounds_cache = self._collect_edge_bounds()

    # ----------------- Gym space -----------------

    def space(self) -> spaces.Space:
        high = 1.0 if self.normalize else self.maximum_range
        return spaces.Box(
            shape=(self.cells, 2),
            low=-high,
            high=high,
            dtype=np.float32,
        )

    # ----------------- Main API -----------------

    def observe(
        self,
        obstacle_entries: list[_ObstacleEntry] | None = None,
        obstacle_index: _ObstacleSpatialIndex | None = None,
    ) -> np.ndarray:
        traced = self.trace(
            self.observer_vehicle.position,
            self.observer_vehicle.velocity,
            obstacle_entries=obstacle_entries,
            obstacle_index=obstacle_index,
        )
        started = time.perf_counter()
        obs = traced.copy()
        if self.normalize:
            obs /= self.maximum_range
        _ObservationProfiler.record("lidar_normalize_copy", time.perf_counter() - started)
        return obs

    # ----------------- Core LiDAR logic -----------------

    def collect_obstacle_entries(self) -> list[_ObstacleEntry]:
        road = getattr(self.env, "road", None)
        if road is None:
            return []

        entries: list[_ObstacleEntry] = []
        for obstacle in list(road.vehicles) + list(road.objects):
            if not getattr(obstacle, "solid", True):
                continue
            if hasattr(obstacle, "appear") and not getattr(obstacle, "appear", True):
                continue
            if hasattr(obstacle, "visible") and not getattr(obstacle, "visible", True):
                continue

            length = float(getattr(obstacle, "LENGTH", 0.0))
            width = float(getattr(obstacle, "WIDTH", 0.0))
            if length == 0.0 and width == 0.0:
                continue

            position = getattr(obstacle, "position", None)
            if position is None or not np.all(np.isfinite(position)):
                continue
            obstacle_pos = np.array(position, dtype=float)

            obs_vel = getattr(obstacle, "velocity", np.zeros(2, dtype=float))
            if obs_vel is None or not np.all(np.isfinite(obs_vel)):
                obs_vel = np.zeros(2, dtype=float)
            else:
                obs_vel = np.array(obs_vel, dtype=float)

            heading = float(getattr(obstacle, "heading", 0.0))
            corners = np.asarray(
                utils.rect_corners(obstacle_pos, length, width, heading),
                dtype=float,
            )
            axis_u, axis_v = self._rect_axes(corners)
            entries.append(
                _ObstacleEntry(
                    obstacle=obstacle,
                    position=obstacle_pos,
                    velocity=obs_vel,
                    width=width,
                    corners=corners,
                    axis_u=axis_u,
                    axis_v=axis_v,
                )
            )
        return entries

    def trace(
        self,
        origin: np.ndarray,
        origin_velocity: np.ndarray,
        obstacle_entries: list[_ObstacleEntry] | None = None,
        obstacle_index: _ObstacleSpatialIndex | None = None,
    ) -> np.ndarray:
        self.origin = np.array(origin, dtype=float).copy()

        # Ensure velocity is finite
        if origin_velocity is None or not np.all(np.isfinite(origin_velocity)):
            origin_velocity = np.zeros(2, dtype=float)
        else:
            origin_velocity = np.array(origin_velocity, dtype=float)

        # Refresh lane cache if road object changed (rare, but safe)
        if self._lanes_cache is None or getattr(self.env, "road", None) is None:
            self._lanes_cache = self._collect_lanes()

        if obstacle_entries is None:
            started = time.perf_counter()
            obstacle_entries = self.collect_obstacle_entries()
            _ObservationProfiler.record("lidar_collect_obstacles", time.perf_counter() - started)
        if obstacle_index is not None:
            started = time.perf_counter()
            obstacle_entries = obstacle_index.query(self.origin, self.maximum_range)
            _ObservationProfiler.record("lidar_obstacle_candidate_query", time.perf_counter() - started)

        # Precompute per-ray road edge distance
        started = time.perf_counter()
        edge_dists = self._distance_to_road_edges_batch(
            origin=self.origin,
            directions=self._directions,
            max_range=self.maximum_range,
            coarse_step=self.coarse_step,
            refine_iters=self.refine_iters,
        )
        _ObservationProfiler.record("lidar_road_edge", time.perf_counter() - started)

        # Initialize grid distances
        self.grid.fill(0.0)
        if self.edge_as_return:
            self.grid[:, self.DISTANCE] = edge_dists
        else:
            self.grid[:, self.DISTANCE] = self.maximum_range
        self.grid[:, self.SPEED] = 0.0

        # Iterate over road vehicles + static objects
        started = time.perf_counter()
        obstacle_entries = obstacle_entries or []
        mask_started = time.perf_counter()
        if obstacle_entries:
            positions = np.asarray([entry.position for entry in obstacle_entries], dtype=float)
            center_vectors = positions - self.origin.reshape(1, 2)
            center_distances = np.linalg.norm(center_vectors, axis=1)
            candidate_indices = np.flatnonzero(
                np.isfinite(center_distances) & (center_distances <= self.maximum_range)
            )
        else:
            center_distances = np.zeros((0,), dtype=float)
            candidate_indices = np.zeros((0,), dtype=np.int64)
        _ObservationProfiler.record("lidar_obstacle_mask_prep", time.perf_counter() - mask_started)
        for obstacle_entry_index in candidate_indices:
            entry = obstacle_entries[int(obstacle_entry_index)]
            if entry.obstacle is self.observer_vehicle:
                continue
            center_distance = float(center_distances[int(obstacle_entry_index)])

            # Approximate center ray bin
            center_angle = self.position_to_angle(entry.position, self.origin)
            center_index = self.angle_to_index(center_angle)

            # Quick cull: if obstacle center beyond the road edge for its bin plus its half width, likely irrelevant
            if center_distance > float(edge_dists[center_index]) + 0.5 * entry.width:
                # Still might intersect another bin, but this removes many far obstacles cheaply
                pass  # keep conservative; do not continue
            angles = [self.position_to_angle(corner, self.origin) for corner in entry.corners]
            angles = [a for a in angles if np.isfinite(a)]
            if len(angles) == 0:
                continue

            min_angle, max_angle = min(angles), max(angles)

            # Handle wrap-around across -pi / +pi
            if min_angle < -np.pi / 2 < np.pi / 2 < max_angle:
                min_angle, max_angle = max_angle, min_angle + 2 * np.pi

            start = self.angle_to_index(min_angle)
            end = self.angle_to_index(max_angle)

            if start <= end:
                index_ranges = (range(start, end + 1),)
            else:
                index_ranges = (range(start, self.cells), range(0, end + 1))

            # Exact ray-rectangle distance per LiDAR cell, with road-edge clamping
            for indexes in index_ranges:
                for index in indexes:
                    max_t = float(edge_dists[index])
                    if max_t <= 0.0:
                        continue

                    direction = self._directions[int(index)]

                    dist = self._distance_to_rect_precomputed(
                        origin=self.origin,
                        direction=direction,
                        max_t=max_t,
                        entry=entry,
                    )

                    if not np.isfinite(dist):
                        continue

                    dist = float(np.clip(dist, 0.0, max_t))

                    if dist <= float(self.grid[int(index), self.DISTANCE]):
                        rel_vel = float((entry.velocity - origin_velocity).dot(direction))
                        self.grid[int(index), :] = [dist, rel_vel]

        # If we are NOT returning edge as return, still ensure beams do not exceed edge
        if not self.edge_as_return:
            self.grid[:, self.DISTANCE] = np.minimum(self.grid[:, self.DISTANCE], edge_dists)

        _ObservationProfiler.record("lidar_obstacles", time.perf_counter() - started)
        return self.grid

    @staticmethod
    def _rect_axes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b, _c, d = corners
        u = b - a
        v = d - a
        return u / np.linalg.norm(u), v / np.linalg.norm(v)

    @staticmethod
    def _distance_to_rect_precomputed(
        *,
        origin: np.ndarray,
        direction: np.ndarray,
        max_t: float,
        entry: _ObstacleEntry,
    ) -> float:
        eps = 1e-6
        a = entry.corners[0]
        b = entry.corners[1]
        d = entry.corners[3]
        q_minus_r = float(max_t) * direction
        rqu = q_minus_r @ entry.axis_u + eps
        rqv = q_minus_r @ entry.axis_v + eps
        interval_1 = [(a - origin) @ entry.axis_u / rqu, (b - origin) @ entry.axis_u / rqu]
        interval_2 = [(a - origin) @ entry.axis_v / rqv, (d - origin) @ entry.axis_v / rqv]
        if rqu < 0:
            interval_1 = list(reversed(interval_1))
        if rqv < 0:
            interval_2 = list(reversed(interval_2))
        if (
            utils.interval_distance(*interval_1, *interval_2) <= 0
            and utils.interval_distance(0, 1, *interval_1) <= 0
            and utils.interval_distance(0, 1, *interval_2) <= 0
        ):
            return float(max(interval_1[0], interval_2[0]) * np.linalg.norm(q_minus_r))
        return float(np.inf)

    # ----------------- Road boundary helpers -----------------

    def _collect_lanes(self):
        """
        Cache lane objects for on-road tests.
        """
        road = getattr(self.env, "road", None)
        if road is None or getattr(road, "network", None) is None:
            return None

        lanes = []
        try:
            for _from, tos in road.network.graph.items():
                for _to, lane_list in tos.items():
                    for lane in lane_list:
                        lanes.append(lane)
        except Exception:
            return None

        return lanes if lanes else None

    def _collect_edge_lanes(self) -> dict[tuple[str, str], list[AbstractLane]] | None:
        road = getattr(self.env, "road", None)
        if road is None or getattr(road, "network", None) is None:
            return None

        edge_lanes: dict[tuple[str, str], list[AbstractLane]] = {}
        try:
            for src, tos in road.network.graph.items():
                for dst, lane_list in tos.items():
                    edge_lanes[(src, dst)] = list(lane_list)
        except Exception:
            return None
        return edge_lanes if edge_lanes else None

    def _collect_lane_bounds(self) -> list[_LaneBounds] | None:
        if self._lanes_cache is None:
            return None
        bounds: list[_LaneBounds] = []
        for lane in self._lanes_cache:
            bounds.append(self._trusted_lane_bounds(lane))
        return bounds

    def _trusted_lane_bounds(self, lane: AbstractLane) -> _LaneBounds:
        if not isinstance(lane, StraightLane) or isinstance(lane, SineLane):
            return _LaneBounds(lane=lane, always_required=True)
        try:
            geometry = self._lane_geometry(lane)
            if geometry is None:
                return _LaneBounds(lane=lane, always_required=True)
            longitudinal_min = -geometry.vehicle_length
            longitudinal_max = geometry.length + geometry.vehicle_length
            lateral = 0.5 * geometry.width
            corners = np.asarray(
                [
                    geometry.start + longitudinal_min * geometry.direction - lateral * geometry.direction_lateral,
                    geometry.start + longitudinal_min * geometry.direction + lateral * geometry.direction_lateral,
                    geometry.start + longitudinal_max * geometry.direction - lateral * geometry.direction_lateral,
                    geometry.start + longitudinal_max * geometry.direction + lateral * geometry.direction_lateral,
                ],
                dtype=float,
            )
            if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                return _LaneBounds(lane=lane, always_required=True)
            # Expand slightly beyond the exact on_lane support so the bbox can only add candidates, never remove them.
            margin = max(1.0e-9, 1.0e-9 * max(1.0, self.maximum_range, geometry.length))
            return _LaneBounds(
                lane=lane,
                min_x=float(np.min(corners[:, 0]) - margin),
                max_x=float(np.max(corners[:, 0]) + margin),
                min_y=float(np.min(corners[:, 1]) - margin),
                max_y=float(np.max(corners[:, 1]) + margin),
                always_required=False,
            )
        except Exception:
            return _LaneBounds(lane=lane, always_required=True)

    def _lane_bounds(self, lane: AbstractLane) -> tuple[float, float, float, float] | None:
        try:
            length = float(getattr(lane, "length", 0.0))
            if not np.isfinite(length) or length <= 0.0:
                sample_s = np.array([0.0], dtype=float)
            else:
                num = max(3, min(9, int(np.ceil(length / 40.0)) + 1))
                sample_s = np.linspace(0.0, length, num=num, dtype=float)

            points = []
            for longitudinal in sample_s:
                center = lane.position(float(longitudinal), 0.0)
                width = float(lane.width_at(float(longitudinal)))
                if center is None or not np.all(np.isfinite(center)) or not np.isfinite(width):
                    continue
                center = np.asarray(center, dtype=float)
                points.append(center)

                lateral = 0.5 * width
                points.append(np.asarray(lane.position(float(longitudinal), lateral), dtype=float))
                points.append(np.asarray(lane.position(float(longitudinal), -lateral), dtype=float))

            if not points:
                return None

            stacked = np.vstack(points)
            return (
                float(np.min(stacked[:, 0])),
                float(np.max(stacked[:, 0])),
                float(np.min(stacked[:, 1])),
                float(np.max(stacked[:, 1])),
            )
        except Exception:
            return None

    def _collect_edge_bounds(
        self,
    ) -> dict[tuple[str, str], tuple[float, float, float, float]] | None:
        if self._edge_lane_cache is None:
            return None

        edge_bounds: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        for edge, lane_list in self._edge_lane_cache.items():
            lane_bounds = [self._lane_bounds(lane) for lane in lane_list]
            lane_bounds = [bounds for bounds in lane_bounds if bounds is not None]
            if not lane_bounds:
                continue

            edge_bounds[edge] = (
                min(bounds[0] for bounds in lane_bounds),
                max(bounds[1] for bounds in lane_bounds),
                min(bounds[2] for bounds in lane_bounds),
                max(bounds[3] for bounds in lane_bounds),
            )
        return edge_bounds if edge_bounds else None

    def _candidate_lanes_for_point(self, p: np.ndarray) -> list[AbstractLane] | None:
        if not self.use_topology_fast_path:
            return None
        if self._edge_lane_cache is None or self._edge_bounds_cache is None:
            self._edge_lane_cache = self._collect_edge_lanes()
            self._edge_bounds_cache = self._collect_edge_bounds()
            if self._edge_lane_cache is None or self._edge_bounds_cache is None:
                return None

        x = float(p[0])
        y = float(p[1])
        observer_edge = None
        lane_index = getattr(self.observer_vehicle, "lane_index", None)
        if lane_index is not None:
            observer_edge = tuple(lane_index[:2])

        candidates: list[AbstractLane] = []
        seen: set[int] = set()
        bounds_margin = max(2.0, self.coarse_step + 1.0)

        def add_edge(edge: tuple[str, str]) -> None:
            for lane in self._edge_lane_cache.get(edge, []):
                key = id(lane)
                if key not in seen:
                    seen.add(key)
                    candidates.append(lane)

        if observer_edge is not None:
            add_edge(observer_edge)

        for edge, bounds in self._edge_bounds_cache.items():
            min_x, max_x, min_y, max_y = bounds
            if (
                min_x - bounds_margin <= x <= max_x + bounds_margin
                and min_y - bounds_margin <= y <= max_y + bounds_margin
            ):
                add_edge(edge)

        return candidates if candidates else None

    def _on_road_at(self, p: np.ndarray) -> bool:
        """
        Conservative point-on-road test: on-road if it lies on any lane surface.
        """
        if self._lanes_cache is None:
            self._lanes_cache = self._collect_lanes()
            if self._lanes_cache is None:
                # Fallback: assume on-road to avoid over-clamping in pathological cases
                return True

        candidate_lanes = self._candidate_lanes_for_point(p)
        lanes = candidate_lanes if candidate_lanes is not None else self._lanes_cache

        for lane in lanes:
            try:
                if lane.on_lane(p):
                    return True
            except Exception:
                continue
        if candidate_lanes is not None:
            for lane in self._lanes_cache:
                if lane in lanes:
                    continue
                try:
                    if lane.on_lane(p):
                        return True
                except Exception:
                    continue
        return False

    def _lane_on_points_vectorized(self, lane: AbstractLane, points: np.ndarray) -> np.ndarray | None:
        points = np.asarray(points, dtype=float)
        geometry = self._lane_geometry(lane)
        if geometry is None:
            return None
        if geometry.kind == "sine":
            delta = points - geometry.start
            longitudinal = delta @ geometry.direction
            lateral = delta @ geometry.direction_lateral
            lateral = lateral - geometry.amplitude * np.sin(
                geometry.pulsation * longitudinal + geometry.phase
            )
            width = geometry.width
        elif geometry.kind == "straight":
            delta = points - geometry.start
            longitudinal = delta @ geometry.direction
            lateral = delta @ geometry.direction_lateral
            width = geometry.width
        else:
            return None

        return (
            np.abs(lateral) <= width / 2.0
        ) & (
            -geometry.vehicle_length <= longitudinal
        ) & (
            longitudinal < geometry.length + geometry.vehicle_length
        )

    def _lane_geometry(self, lane: AbstractLane) -> _LaneGeometry | None:
        key = id(lane)
        if key in self._lane_geometry_cache:
            return self._lane_geometry_cache[key]
        geometry: _LaneGeometry | None
        if isinstance(lane, SineLane):
            geometry = _LaneGeometry(
                lane=lane,
                kind="sine",
                start=np.asarray(lane.start, dtype=float),
                direction=np.asarray(lane.direction, dtype=float),
                direction_lateral=np.asarray(lane.direction_lateral, dtype=float),
                width=float(lane.width),
                length=float(lane.length),
                vehicle_length=float(lane.VEHICLE_LENGTH),
                amplitude=float(lane.amplitude),
                pulsation=float(lane.pulsation),
                phase=float(lane.phase),
            )
        elif isinstance(lane, StraightLane):
            geometry = _LaneGeometry(
                lane=lane,
                kind="straight",
                start=np.asarray(lane.start, dtype=float),
                direction=np.asarray(lane.direction, dtype=float),
                direction_lateral=np.asarray(lane.direction_lateral, dtype=float),
                width=float(lane.width),
                length=float(lane.length),
                vehicle_length=float(lane.VEHICLE_LENGTH),
            )
        else:
            geometry = None
        self._lane_geometry_cache[key] = geometry
        return geometry

    def _on_road_many(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized equivalent of _on_road_at for batches of points.

        StraightLane and SineLane are handled analytically. Any unsupported lane
        type falls back to its scalar on_lane method, preserving semantics.
        """
        started = time.perf_counter()
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if self._lanes_cache is None:
            self._lanes_cache = self._collect_lanes()
            if self._lanes_cache is None:
                return np.ones((points.shape[0],), dtype=bool)
            self._lane_bounds_cache = self._collect_lane_bounds()

        result = self._on_road_many_bounded(points)
        if self._on_road_many_reference_compare:
            compare_started = time.perf_counter()
            reference = self._on_road_many_full_scan(points)
            _ObservationProfiler.record("_on_road_many_reference_compare", time.perf_counter() - compare_started)
            if not np.array_equal(result, reference):
                mismatch_indices = np.flatnonzero(result != reference)
                mismatch_index = int(mismatch_indices[0]) if mismatch_indices.size else -1
                self._on_road_many_mismatch = {
                    "point_index": mismatch_index,
                    "point": points[mismatch_index].tolist() if mismatch_index >= 0 else None,
                    "optimized": bool(result[mismatch_index]) if mismatch_index >= 0 else None,
                    "reference": bool(reference[mismatch_index]) if mismatch_index >= 0 else None,
                    "points": int(points.shape[0]),
                    "lanes": int(len(self._lanes_cache or [])),
                }
                raise AssertionError(f"_on_road_many mismatch: {self._on_road_many_mismatch}")
        _ObservationProfiler.record("_on_road_many_total", time.perf_counter() - started)
        return result

    def _on_road_many_full_scan(self, points: np.ndarray) -> np.ndarray:
        result = np.zeros((points.shape[0],), dtype=bool)
        for lane in self._lanes_cache:
            try:
                lane_mask = self._lane_on_points_vectorized(lane, points)
            except Exception:
                lane_mask = None
            if lane_mask is None:
                lane_mask = np.asarray(
                    [
                        self._safe_lane_on_point(lane, point)
                        for point in points
                    ],
                    dtype=bool,
                )
            result |= lane_mask
            if bool(result.all()):
                break
        return result

    def _on_road_many_bounded(self, points: np.ndarray) -> np.ndarray:
        if self._lane_bounds_cache is None or len(self._lane_bounds_cache) != len(self._lanes_cache):
            self._lane_bounds_cache = self._collect_lane_bounds()
        if self._lane_bounds_cache is None:
            return self._on_road_many_full_scan(points)

        result = np.zeros((points.shape[0],), dtype=bool)
        point_indices = np.arange(points.shape[0], dtype=np.int64)
        for lane_bounds in self._lane_bounds_cache:
            remaining = ~result
            if not bool(np.any(remaining)):
                break

            bbox_started = time.perf_counter()
            if lane_bounds.always_required:
                candidate_mask = remaining
            else:
                candidate_mask = (
                    remaining
                    & (points[:, 0] >= lane_bounds.min_x)
                    & (points[:, 0] <= lane_bounds.max_x)
                    & (points[:, 1] >= lane_bounds.min_y)
                    & (points[:, 1] <= lane_bounds.max_y)
                )
            candidate_indices = point_indices[candidate_mask]
            _ObservationProfiler.record("_on_road_many_bbox_filter", time.perf_counter() - bbox_started)
            if candidate_indices.size == 0:
                continue

            candidate_points = points[candidate_indices]
            lane_eval_started = time.perf_counter()
            try:
                lane_mask = self._lane_on_points_vectorized(lane_bounds.lane, candidate_points)
            except Exception:
                lane_mask = None

            if lane_mask is None:
                _ObservationProfiler.record("_on_road_many_lane_eval", time.perf_counter() - lane_eval_started)
                fallback_started = time.perf_counter()
                lane_mask = np.asarray(
                    [
                        self._safe_lane_on_point(lane_bounds.lane, point)
                        for point in candidate_points
                    ],
                    dtype=bool,
                )
                _ObservationProfiler.record("_on_road_many_fallback_eval", time.perf_counter() - fallback_started)
            else:
                _ObservationProfiler.record("_on_road_many_lane_eval", time.perf_counter() - lane_eval_started)

            if np.any(lane_mask):
                result[candidate_indices[lane_mask]] = True
        return result

    @staticmethod
    def _safe_lane_on_point(lane: AbstractLane, point: np.ndarray) -> bool:
        try:
            return bool(lane.on_lane(point))
        except Exception:
            return False

    def _distance_to_road_edges_batch(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        max_range: float,
        coarse_step: float,
        refine_iters: int,
    ) -> np.ndarray:
        """
        Batched version of _distance_to_road_edge using the same coarse march
        and binary refinement semantics for all LiDAR beams.
        """
        origin = np.asarray(origin, dtype=float).reshape(2)
        directions = np.asarray(directions, dtype=float).reshape(-1, 2)
        max_range = float(max_range)
        coarse_step = float(coarse_step)
        refine_iters = int(refine_iters)
        if directions.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if not bool(self._on_road_many(origin.reshape(1, 2))[0]):
            return np.zeros((directions.shape[0],), dtype=np.float32)

        norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1.0e-12
        directions = directions / norms
        if coarse_step <= 0.0:
            coarse_step = max_range
        steps = np.arange(coarse_step, max_range + 1.0e-9, coarse_step, dtype=float)
        if steps.size == 0 or steps[-1] < max_range:
            steps = np.concatenate([steps, np.asarray([max_range], dtype=float)])
        else:
            steps[-1] = min(float(steps[-1]), max_range)

        points = origin.reshape(1, 1, 2) + steps.reshape(-1, 1, 1) * directions.reshape(1, -1, 2)
        on_road = self._on_road_many(points.reshape(-1, 2)).reshape(steps.shape[0], directions.shape[0])
        first_off = np.argmax(~on_road, axis=0)
        has_off = np.any(~on_road, axis=0)
        distances = np.full((directions.shape[0],), max_range, dtype=float)
        if not bool(np.any(has_off)):
            return distances.astype(np.float32)

        crossing_indices = np.where(has_off)[0]
        step_indices = first_off[crossing_indices]
        hi = steps[step_indices].astype(float, copy=True)
        lo = np.where(step_indices > 0, steps[step_indices - 1], 0.0).astype(float, copy=True)
        crossing_dirs = directions[crossing_indices]

        for _ in range(refine_iters):
            mid = 0.5 * (lo + hi)
            mid_points = origin.reshape(1, 2) + mid.reshape(-1, 1) * crossing_dirs
            mid_on = self._on_road_many(mid_points)
            lo = np.where(mid_on, mid, lo)
            hi = np.where(mid_on, hi, mid)
        distances[crossing_indices] = lo
        return np.clip(distances, 0.0, max_range).astype(np.float32)

    def _distance_to_road_edge(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_range: float,
        coarse_step: float,
        refine_iters: int,
    ) -> float:
        """
        Distance along ray until it first leaves the road surface.
        Returns in [0, max_range]. If road never ends within max_range, returns max_range.
        """
        d = direction / (np.linalg.norm(direction) + 1e-12)

        # If already off-road, edge is at 0
        if not self._on_road_at(origin):
            return 0.0

        t = 0.0
        last_on = 0.0

        # Coarse march
        while t < max_range:
            t = min(t + coarse_step, max_range)
            p = origin + t * d
            if not self._on_road_at(p):
                # Refine boundary between last_on (on) and t (off)
                lo, hi = last_on, t
                for _ in range(refine_iters):
                    mid = 0.5 * (lo + hi)
                    pm = origin + mid * d
                    if self._on_road_at(pm):
                        lo = mid
                    else:
                        hi = mid
                return lo
            last_on = t

        return max_range

    # ----------------- Helper functions -----------------

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        dx = float(position[0] - origin[0])
        dy = float(position[1] - origin[1])

        if not np.isfinite(dx) or not np.isfinite(dy):
            return 0.0

        ang = float(np.arctan2(dy, dx) + self.angle / 2.0)
        if not np.isfinite(ang):
            return 0.0
        return ang

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        if not np.isfinite(angle):
            return 0
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: int) -> np.ndarray:
        """
        Convert a LiDAR cell index into a unit direction vector in world coordinates.
        The beam is centered in the cell: angle = (index + 0.5) * cell_angle.
        """
        theta = (int(index) + 0.5) * self.angle
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)

__all__ = [
    'LidarObservation',
]
