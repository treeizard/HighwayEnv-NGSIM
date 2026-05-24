"""Shared observation primitives and spatial indexes."""

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

@dataclass(frozen=True)
class _ObstacleEntry:
    obstacle: object
    position: np.ndarray
    velocity: np.ndarray
    width: float
    corners: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray

@dataclass(frozen=True)
class _LaneGeometry:
    lane: AbstractLane
    kind: str
    start: np.ndarray | None = None
    direction: np.ndarray | None = None
    direction_lateral: np.ndarray | None = None
    width: float = 0.0
    length: float = 0.0
    vehicle_length: float = 0.0
    amplitude: float = 0.0
    pulsation: float = 0.0
    phase: float = 0.0

@dataclass(frozen=True)
class _LaneBounds:
    lane: AbstractLane
    min_x: float = -np.inf
    max_x: float = np.inf
    min_y: float = -np.inf
    max_y: float = np.inf
    always_required: bool = True

class _ObservationProfiler:
    enabled = os.environ.get("HIGHWAY_ENV_OBS_PROFILE", "").lower() in {"1", "true", "yes", "on"}
    report_every = max(1, int(os.environ.get("HIGHWAY_ENV_OBS_PROFILE_EVERY", "1000")))
    counts: defaultdict[str, int] = defaultdict(int)
    totals: defaultdict[str, float] = defaultdict(float)
    events = 0

    @classmethod
    def record(cls, name: str, elapsed: float) -> None:
        if not cls.enabled:
            return
        cls.counts[name] += 1
        cls.totals[name] += float(elapsed)
        if name == "shared_observation_total":
            cls.events += 1
            if cls.events % cls.report_every == 0:
                cls.report()

    @classmethod
    def report(cls) -> None:
        if not cls.enabled:
            return
        parts = []
        for name in sorted(cls.totals):
            count = max(1, int(cls.counts[name]))
            total_ms = 1000.0 * float(cls.totals[name])
            parts.append(f"{name}={total_ms / count:.3f}ms avg ({total_ms:.1f}ms/{count})")
        print("[observation_profile] " + " ".join(parts), flush=True)

    @classmethod
    def reset(cls) -> None:
        cls.counts.clear()
        cls.totals.clear()
        cls.events = 0

class _ObstacleSpatialIndex:
    def __init__(
        self,
        entries: list[_ObstacleEntry],
    ) -> None:
        self.entries = entries
        self.tree = None
        if cKDTree is None or not entries:
            return
        positions = np.asarray([entry.position for entry in entries], dtype=float)
        if positions.ndim == 2 and positions.shape[1] == 2 and np.all(np.isfinite(positions)):
            self.tree = cKDTree(positions)

    def query(
        self,
        origin: np.ndarray,
        radius: float,
    ) -> list[_ObstacleEntry]:
        if self.tree is None:
            return self.entries
        indices = self.tree.query_ball_point(np.asarray(origin, dtype=float), float(radius))
        return [self.entries[int(index)] for index in sorted(indices)]

    def query_many(
        self,
        origins: np.ndarray,
        radius: float,
    ) -> list[list[_ObstacleEntry]]:
        origins = np.asarray(origins, dtype=float)
        if origins.ndim != 2 or origins.shape[1] != 2 or self.tree is None:
            return [self.entries for _ in range(len(origins))]
        all_indices = self.tree.query_ball_point(origins, float(radius))
        return [[self.entries[int(index)] for index in sorted(indices)] for indices in all_indices]

class ObservationType:
    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle

__all__ = [
    '_ObstacleEntry',
    '_LaneGeometry',
    '_LaneBounds',
    '_ObservationProfiler',
    '_ObstacleSpatialIndex',
    'ObservationType',
]
