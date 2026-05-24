"""Lane-camera and combined lidar-camera observation types."""

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

from .base import ObservationType, _ObservationProfiler, _ObstacleSpatialIndex
from .lidar import LidarObservation

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
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.lidar_observation = LidarObservation(env, **(lidar or {}))
        self.camera_observation = LaneCameraObservation(env, **(camera or {}))

    @staticmethod
    def _ego_state_space() -> spaces.Box:
        return spaces.Box(
            low=np.array([-np.inf, -np.pi, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.pi, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def _build_ego_state(vehicle) -> np.ndarray:
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
        query_started = time.perf_counter()
        origins = np.asarray([vehicle.position for vehicle in vehicles], dtype=float)
        candidate_entries = obstacle_index.query_many(origins, self.lidar_observation.maximum_range)
        _ObservationProfiler.record("shared_obstacle_candidate_query", time.perf_counter() - query_started)
        observations = []
        for vehicle, vehicle_obstacle_entries in zip(vehicles, candidate_entries):
            self.lidar_observation.observer_vehicle = vehicle
            self.camera_observation.observer_vehicle = vehicle
            ego_state = self._build_ego_state(vehicle)
            observations.append(
                (
                    self.lidar_observation.observe(
                        obstacle_entries=vehicle_obstacle_entries,
                    ),
                    self.camera_observation.observe(),
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
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.lidar_observation = LidarObservation(env, **(lidar or {}))
        self.camera_observation = LaneCameraObservation(env, **(camera or {}))

    @staticmethod
    def _ego_state_space() -> spaces.Box:
        return spaces.Box(
            low=np.array([-np.inf, -np.pi, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.pi, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def _build_ego_state(vehicle) -> np.ndarray:
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
        ego_state = self._build_ego_state(self.observer_vehicle)
        return (
            self.lidar_observation.observe(),
            self.camera_observation.observe(),
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
