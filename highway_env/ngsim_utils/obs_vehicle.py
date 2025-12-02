# Modified by: Yide Tao (yide.tao@monash.edu)
# Reference: @article{huang2021driving,
#   title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
#   author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
#   journal={IEEE Transactions on Intelligent Transportation Systems},
#   year={2021},
#   publisher={IEEE}
# }
# @misc{highway-env,
#   author = {Leurent, Edouard},
#   title = {An Environment for Autonomous Driving Decision-Making},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/eleurent/highway-env}},
# }


from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from typing import Tuple, Optional

from highway_env import utils
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.ngsim_utils.ego_vehicle import ControlledVehicle

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


class NGSIMVehicle(IDMVehicle):
    """
    NGSIM-driven vehicle that replays a human trajectory while it is safe,
    and hands over to IDM/MOBIL when the gap to the front vehicle becomes too small,
    or when the trajectory is exhausted.

    Expected ngsim_traj rows (already converted to METERS):
        [x_m, y_m, v_mps, lane_id]
    """

    # Longitudinal policy parameters (you can retune if you like)
    COLLISIONS_ENABLED  = True
    ACC_MAX = 5.0          # [m/s²]
    COMFORT_ACC_MAX = 3.0  # [m/s²]
    COMFORT_ACC_MIN = -3.0 # [m/s²]
    DISTANCE_WANTED = 1.0  # [m]
    TIME_WANTED = 0.5      # [s]
    DELTA = 4.0

    # Lateral policy parameters [MOBIL]
    POLITENESS = 0.1
    LANE_CHANGE_MIN_ACC_GAIN = 0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0
    LANE_CHANGE_DELAY = 1.0

    # Scenario
    SCENE = "us-101"

    # NGSIM sampling period (seconds)
    DATA_DT = 0.1
    DEFAULT_COLOR = (160, 160, 160)
    def __init__(
        self,
        road,
        position,
        heading: float = 0.0,
        speed: float = 0.0,
        target_lane_index=None,
        target_speed: float = None,
        route=None,
        enable_lane_change: bool = False,
        timer: float = None,
        vehicle_ID=None,
        v_length=None,
        v_width=None,
        ngsim_traj=None,
        color=None
    ):
        super().__init__(
            road,
            position,
            heading=heading,
            speed=speed,
            target_lane_index=target_lane_index,
            target_speed=target_speed,
            route=route,
            enable_lane_change=enable_lane_change,
            timer=timer,
        )

        # Trajectory: [x, y, v, lane_id]
        self.ngsim_traj = (
            np.asarray(ngsim_traj, dtype=float) if ngsim_traj is not None else None
        )
        self.vehicle_ID = vehicle_ID

        # Replay state
        self.sim_steps = 0
        self.overtaken = False       # False => follow NGSIM; True => IDM/MOBIL
        self.appear = bool(self.position[0] != 0)

        # Diagnostics
        self.traj = np.array(self.position, dtype=float)
        self.speed_history = []
        self.heading_history = []
        self.crash_history = []
        self.overtaken_history = []

        # Dimensions with safe fallbacks
        self.LENGTH = float(v_length) if v_length is not None else getattr(self, "LENGTH", 4.5)
        self.WIDTH  = float(v_width)  if v_width  is not None else getattr(self, "WIDTH", 2.0)
        self.diagonal = np.sqrt(self.LENGTH**2 + self.WIDTH**2)
        self.color = color if color is not None else self.DEFAULT_COLOR
    # ---------------- Factory ----------------
    @classmethod
    def create(
        cls,
        road,
        vehicle_ID,
        position,
        v_length,
        v_width,
        ngsim_traj,
        heading: float = 0.0,
        speed: float = 15.0,
        color=None,  
    ):
        return cls(
            road,
            position,
            heading=heading,
            speed=speed,
            vehicle_ID=vehicle_ID,
            v_length=v_length,
            v_width=v_width,
            ngsim_traj=ngsim_traj,
            color=color,
        )

    # ---------------- Behaviour ----------------
    def act(self, action: dict | str = None):
        """
        Only act (IDM/MOBIL) once we are overtaken.
        While in replay mode (not overtaken), we don't call Vehicle.act().
        """
        if self.crashed:
            return
        if not self.overtaken:
            return
        # Reuse IDMVehicle.act for actual control once taken over
        super().act(action)

    def _update_from_trajectory(self):
        """
        Apply one replay step from ngsim_traj[sim_steps] -> [sim_steps+1].
        Sets position, speed, heading, lane_index/lane.
        """
        if self.ngsim_traj is None:
            self.overtaken = True
            return
        if self.sim_steps + 1 >= len(self.ngsim_traj):
            self.overtaken = True
            return

        # Current and next samples: [x, y, v, lane_id]
        cur_x, cur_y, cur_v, cur_lane = self.ngsim_traj[self.sim_steps][:4]
        nxt_x, nxt_y, nxt_v, _        = self.ngsim_traj[self.sim_steps + 1][:4]

        # Position from data
        self.position = np.array([cur_x, cur_y], dtype=float)
        self.appear = bool(cur_x != 0.0)

        # Speed from spatial difference (fallback to recorded v)
        dx = nxt_x - cur_x
        dy = nxt_y - cur_y
        dist = math.hypot(dx, dy)
        data_dt = self.DATA_DT

        speed_est = dist / utils.not_zero(data_dt)
        self.speed = speed_est if speed_est > 1e-3 else float(cur_v)
        self.target_speed = self.speed

        # Heading from motion vector
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            self.heading = math.atan2(dy, dx)

        # Attach to nearest lane
        self.lane_index = self.road.network.get_closest_lane_index(self.position)
        self.lane = self.road.network.get_lane(self.lane_index)

    def _front_gap_logic(self):
        """
        Decide which front vehicle counts for gap checking.

        We only care when the front vehicle is the ego (ControlledVehicle).
        All other fronts (NGSIM or background vehicles) are ignored
        for takeover (treated as "safe" for the replay logic).

        Returns:
            gap: float
            desired_gap: float
            ego_ahead: bool
        """
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # No front vehicle -> treat as safe, no ego ahead
        if front_vehicle is None:
            return 100.0, 50.0, False

        # If the front vehicle is the ego, we care about the real gap
        if isinstance(front_vehicle, ControlledVehicle):
            gap = self.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(self, front_vehicle)
            return gap, desired_gap, True

        # Any other front vehicle (NGSIM or other) is ignored for takeover
        return 100.0, 50.0, False


    def step(self, dt: float):
        """
        Update the state:

        - If we still have NGSIM data and (either no ego ahead, or ego is at a safe gap):
              → follow the recorded NGSIM trajectory.
        - Otherwise (trajectory exhausted OR ego too close ahead):
              → mark as overtaken and use IDM/MOBIL dynamics.
        """
        # No trajectory: behave as a pure IDM vehicle
        if self.ngsim_traj is None or len(self.ngsim_traj) == 0:
            self.overtaken = True
            super().step(dt)
            return

        # Timer / histories
        self.timer += dt
        self.heading_history.append(self.heading)
        self.speed_history.append(self.speed)
        self.crash_history.append(self.crashed)
        self.overtaken_history.append(self.overtaken)

        gap, desired_gap, ego_ahead = self._front_gap_logic()

        # Still have replay data?
        can_replay = (not self.overtaken) and (self.sim_steps + 1 < len(self.ngsim_traj))

        # --- Replay vs takeover logic ---
        #  - If no ego ahead: ignore gaps, just replay while we have data.
        #  - If ego ahead: only replay when gap >= desired_gap.
        if can_replay and (not ego_ahead or gap >= desired_gap):
            # Keep replaying the NGSIM trajectory
            self._update_from_trajectory()
            self.sim_steps += 1
        else:
            # Handover to IDM/MOBIL
            self.overtaken = True
            self.color = (100, 200, 255)

            # Use lane_id + x to pick a target lane index (your existing mapping)
            lane_id = int(
                self.ngsim_traj[min(self.sim_steps, len(self.ngsim_traj) - 1)][3]
            )
            x = self.position[0]
            target_lane_index = None

            if self.SCENE == "us-101":
                if lane_id <= 5:
                    if 0 < x <= 560 / 3.281:
                        target_lane_index = ("s1", "s2", lane_id - 1)
                    elif 560 / 3.281 < x <= (698 + 578 + 150) / 3.281:
                        target_lane_index = ("s2", "s3", lane_id - 1)
                    else:
                        target_lane_index = ("s3", "s4", lane_id - 1)
                elif lane_id == 6:
                    target_lane_index = ("s2", "s3", -1)
                elif lane_id == 7:
                    target_lane_index = ("merge_in", "s2", -1)
                elif lane_id == 8:
                    target_lane_index = ("s3", "merge_out", -1)
            elif self.SCENE == "i-80":
                if lane_id <= 6:
                    if 0 < x <= 600 / 3.281:
                        target_lane_index = ("s1", "s2", lane_id - 1)
                    elif 600 / 3.281 < x <= 700 / 3.281:
                        target_lane_index = ("s2", "s3", lane_id - 1)
                    elif 700 / 3.281 < x <= 900 / 3.281:
                        target_lane_index = ("s3", "s4", lane_id - 1)
                    else:
                        target_lane_index = ("s4", "s5", lane_id - 1)
                elif lane_id == 7:
                    target_lane_index = ("s1", "s2", -1)

            if target_lane_index is not None:
                self.target_lane_index = target_lane_index

            # Now evolve with IDM/MOBIL dynamics
            super().step(dt)

        # Record replayed / simulated position
        self.traj = np.append(self.traj, self.position, axis=0)


    # ---------------- Collision handling ----------------
        # ---------------- Collision handling ----------------
    def handle_collisions(self, other: IDMVehicle, dt: float = 0.0) -> None:
        """
        Collision handling that plugs into RoadObject.handle_collisions, but keeps
        the NGSIM-specific behaviour:

        - Respect COLLISIONS_ENABLED on both sides.
        - Ignore NGSIM-vs-NGSIM collisions while both are still in replay
          (both overtaken == False).
        - Delegate geometry and crash/hit/impact flags to the base implementation.
        - When a *new* crash happens, clamp both speeds to the smaller magnitude
          (your original "safe-ish min speed" rule).
        """

        # 1. Global toggle: if either side disabled, do nothing.
        if not getattr(self, "COLLISIONS_ENABLED", True) or \
           not getattr(other, "COLLISIONS_ENABLED", True):
            return

        # 2. Skip NGSIM-vs-NGSIM collisions while both still in replay mode.
        #    This preserves your original "ignore replay vs replay" logic.
        if isinstance(self, NGSIMVehicle) and isinstance(other, NGSIMVehicle):
            if not self.overtaken and not getattr(other, "overtaken", False):
                return

        # 3. Remember prior crash states so we can detect a *new* crash.
        pre_crash_self = self.crashed
        pre_crash_other = getattr(other, "crashed", False)

        # 4. Delegate the actual geometric test and crash/hit/impact logic
        #    to the parent implementation (Vehicle/RoadObject).
        super().handle_collisions(other, dt)

        # 5. If a new crash has just occurred, and both are now crashed,
        #    apply the "safe-ish" min-speed rule from your legacy code.
        post_crash_self = self.crashed
        post_crash_other = getattr(other, "crashed", False)

        new_crash_happened = (not pre_crash_self or not pre_crash_other) and \
                             (post_crash_self and post_crash_other)

        if new_crash_happened and hasattr(other, "speed"):
            # Use the smaller absolute speed of the two
            min_speed = min(self.speed, other.speed, key=abs)
            self.speed = other.speed = min_speed
