from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from typing import Tuple, Optional

from highway_env import utils
from highway_env.vehicle.behavior import IDMVehicle

@dataclass
class DebugState:
    step: int
    veh_id: int
    edge: Tuple[str, str]
    lane_idx: int
    local_s: float
    r: float
    x: float
    y: float
    speed: float
    length: float
    width: float
    crashed: bool
    overtaken: bool

class ReplayVehicle(IDMVehicle):
    """
    Robust NGSIM trajectory replayer with overlap diagnostics.

    Expected traj rows already in METERS: [s_m, r_m, v_mps, lane_id]
    where s is longitudinal along the freeway centerline and r is lateral.
    """

    # IDM tuning (kept modest to avoid aggressive catch-ups on handover)
    ACC_MAX = 4.0
    COMFORT_ACC_MAX = 2.5
    COMFORT_ACC_MIN = -3.0
    DISTANCE_WANTED = 2.0
    TIME_WANTED = 0.7
    DELTA = 4.0

    def __init__(
        self, road, position, heading: float = 0.0, speed: float = 0.0,
        target_lane_index=None, target_speed: Optional[float] = None,
        route=None, enable_lane_change: bool = False, timer: Optional[float] = None,
        *, vehicle_ID: Optional[int] = None, v_length: Optional[float] = None,
        v_width: Optional[float] = None, ngsim_traj=None, debug: bool = True
    ):
        super().__init__(
            road, position, heading=heading, speed=speed,
            target_lane_index=target_lane_index, target_speed=target_speed,
            route=route, enable_lane_change=enable_lane_change, timer=timer
        )
        self.vehicle_ID = int(vehicle_ID) if vehicle_ID is not None else -1
        self.ngsim_traj = np.asarray(ngsim_traj) if ngsim_traj is not None else None
        self.sim_steps = 0
        self.overtaken = False   # False => follow replay; True => use IDM/MOBIL
        self.debug = debug

        # Dimensions
        if v_length is not None: self.LENGTH = float(v_length)
        if v_width is not None:  self.WIDTH  = float(v_width)

        # Histories
        self.debug_history: list[DebugState] = []

    # ---- helpers ----
    def _edge_for_s(self, s: float) -> Tuple[Tuple[str, str], float]:
        # MUST match env's _create_road cutpoints
        cut1 = 560/3.281
        cut2 = (698+578+150)/3.281
        if s <= cut1:
            return ("s1","s2"), 0.0
        elif s <= cut2:
            return ("s2","s3"), cut1
        else:
            return ("s3","s4"), cut2

    def _rect_aabb(self, x: float, y: float, heading: float) -> Tuple[float,float,float,float]:
        """Axis-aligned bbox of oriented rectangle (conservative)."""
        # Over-approximate with a circle radius then AABB (fast + safe)
        r = 0.5 * math.hypot(self.LENGTH, self.WIDTH)
        return (x - r, y - r, x + r, y + r)

    def _log_debug(self, lane_idx_tuple, edge, local_s, r_val):
        if not self.debug: return
        self.debug_history.append(
            DebugState(
                step=self.sim_steps, veh_id=self.vehicle_ID,
                edge=edge, lane_idx=lane_idx_tuple[2] if lane_idx_tuple else -1,
                local_s=float(local_s), r=float(r_val),
                x=float(self.position[0]), y=float(self.position[1]),
                speed=float(self.speed), length=float(self.LENGTH), width=float(self.WIDTH),
                crashed=bool(self.crashed), overtaken=bool(self.overtaken)
            )
        )

    # ---- main logic ----
    @classmethod
    def create(
        cls, road, vehicle_ID: int, position, v_length: float, v_width: float, ngsim_traj,
        heading: float = 0.0, speed: float = 15.0, target_lane_index=None, target_speed=None,
        route=None, enable_lane_change: bool = False, debug: bool = True
    ):
        return cls(
            road, position, heading=heading, speed=float(speed),
            target_lane_index=target_lane_index, target_speed=target_speed, route=route,
            enable_lane_change=enable_lane_change,
            vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width,
            ngsim_traj=ngsim_traj, debug=debug
        )

    def act(self, action=None) -> None:
        # On replay we do nothing; once overtaken we let IDM handle it
        if not self.overtaken or self.crashed:
            return
        super().act(action)

    def _ngsim_step_update(self, dt: float) -> None:
        # Hand off to IDM if replay is exhausted
        if self.ngsim_traj is None or self.sim_steps + 1 >= len(self.ngsim_traj):
            self.overtaken = True
            return

        cur_s, cur_r_abs, cur_v, cur_lane = self.ngsim_traj[self.sim_steps]
        nxt_s, nxt_r_abs, nxt_v, _ = self.ngsim_traj[self.sim_steps + 1]

        # Pick road edge by global s and compute local s
        edge, edge_start = self._edge_for_s(float(cur_s))
        try:
            n_lanes_on_edge = len(self.road.network.graph[edge[0]][edge[1]])
        except Exception:
            n_lanes_on_edge = 1

        # Lane index (NGSIM is 1-based); clamp to edge lane count
        lane_zero_based = int(np.clip((int(cur_lane) if cur_lane else 1) - 1, 0, max(0, n_lanes_on_edge - 1)))
        lane_idx_tuple = (edge[0], edge[1], lane_zero_based)
        lane = self.road.network.get_lane(lane_idx_tuple)

        # Local curvilinear s relative to the chosen edge
        local_s = float(cur_s - edge_start)

        # ---- KEY FIX: convert absolute lateral (r_abs) -> relative to lane center ----
        # lane.position(local_s, 0.0) returns the lane center at that s
        center_x, center_y = lane.position(local_s, 0.0)
        # cur_r_abs is absolute lateral (already meters). Make it relative to lane center:
        cur_r_rel = float(cur_r_abs) - float(center_y)

        # Update pose from replay
        self.position = np.asarray(lane.position(local_s, cur_r_rel), dtype=float)
        self.heading  = float(lane.heading_at(local_s))

        # Prefer finite-difference speed from s; fallback to dataset v
        ds = float(nxt_s - cur_s)
        self.speed = ds / utils.not_zero(dt) if abs(ds) > 1e-6 else float(cur_v)

        # Book-keeping
        self.lane_index = lane_idx_tuple
        self.lane = lane

        # Debug log (store absolute r so you can inspect conversion later)
        self._log_debug(lane_idx_tuple, edge, local_s, cur_r_abs)

    def step(self, dt: float) -> None:
        # Maintain histories
        if getattr(self, "timer", None) is not None:
            self.timer += dt

        if self.crashed:
            super().step(dt)
            return

        # Neighbour check to avoid piling on top of a stopped car
        if self.road is not None:
            lane_idx = getattr(self, "lane_index", None)
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, lane_idx)
        else:
            front_vehicle, rear_vehicle = (None, None)

        gap = self.lane_distance_to(front_vehicle) if front_vehicle is not None else float("inf")
        desired_gap = self.desired_gap(self, front_vehicle) if front_vehicle is not None else 0.0

        can_replay = (not self.overtaken and self.ngsim_traj is not None and self.sim_steps + 1 < len(self.ngsim_traj))
        if can_replay and (gap >= max(desired_gap, 0.5 * self.LENGTH)):
            self._ngsim_step_update(dt)
            self.sim_steps += 1
            return

        # Handover to IDM/MOBIL if replay blocked or exhausted
        self.overtaken = True
        super().step(dt)

    # ---- overlap test (optional, used by env) ----
    def aabb(self) -> Tuple[float, float, float, float]:
        return self._rect_aabb(self.position[0], self.position[1], self.heading)

    def overlaps_aabb(self, other: "ReplayVehicle") -> bool:
        ax1, ay1, ax2, ay2 = self.aabb()
        bx1, by1, bx2, by2 = other.aabb()
        return (ax1 <= bx2) and (ax2 >= bx1) and (ay1 <= by2) and (ay2 >= by1)