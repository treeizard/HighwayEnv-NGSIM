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
from highway_env.ngsim_utils.constants import FEET_PER_METER
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle
from highway_env.ngsim_utils.helper_ngsim import target_lane_index_from_lane_id
from highway_env.ngsim_utils.trajectory_gen import process_raw_trajectory

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
        scene: str | None = None,
        color=None,
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
        if scene is not None:
            self.SCENE = str(scene)

        # Replay state
        self.sim_steps = 0
        self.overtaken = False
        self.remove_from_road = False

        # ---- Initialise from trajectory instead of dummy (0,0) ----
        if self.ngsim_traj is not None and len(self.ngsim_traj) > 0:
            # Current sample (sim_steps == 0 initially)
            cur_x, cur_y, cur_v, _ = self.ngsim_traj[self.sim_steps][:4]

            # Use data pose as initial pose
            self.position = np.array([cur_x, cur_y], dtype=float)
            self.speed = float(cur_v)
            self.target_speed = self.speed

            # "Appear" is keyed off x == 0.0 (your convention)
            self.appear = bool(cur_x != 0.0)
        else:
            # Fallback if no traj at all
            self.appear = bool(self.position[0] != 0.0)

        # Hook into highway-env's rendering: ghost vehicles are invisible
        self.visible = self.appear

        # Diagnostics
        self.traj = np.array([self.position.copy()], dtype=float)
        self.speed_history = []
        self.heading_history = []
        self.crash_history = []
        self.overtaken_history = []

        # Dimensions with safe fallbacks
        self.real_length = float(v_length) if v_length is not None else getattr(self, "LENGTH", 4.5)
        self.real_width  = float(v_width)  if v_width  is not None else getattr(self, "WIDTH", 2.0)

        if self.appear:
            self.LENGTH = self.real_length
            self.WIDTH  = self.real_width
        else:
            # Ghost: zero footprint
            self.LENGTH = 0.0
            self.WIDTH  = 0.0
        self._update_diagonal()

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
        scene: str | None = None,
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
            scene=scene,
            color=color,
        )

    # ---------------- Behaviour ----------------
    def _update_diagonal(self) -> None:
        self.diagonal = float(np.hypot(self.LENGTH, self.WIDTH))

    def _set_visibility_from_appearance(self, appear: bool) -> None:
        self.appear = bool(appear)
        self.visible = self.appear
        if self.appear:
            self.LENGTH = self.real_length
            self.WIDTH = self.real_width
        else:
            self.LENGTH = 0.0
            self.WIDTH = 0.0
        self._update_diagonal()

    def _record_position(self) -> None:
        self.traj = np.vstack([self.traj, self.position.copy()])

    def _mark_for_removal(self) -> None:
        """Hide the vehicle and flag it for pruning from the road."""
        self.remove_from_road = True
        self.speed = 0.0
        self.target_speed = 0.0
        self._set_visibility_from_appearance(False)

    def _reached_terminal_lane_end(self) -> bool:
        """
        Return True when the vehicle is on a lane with no downstream edge and has
        progressed beyond the usable end of that lane.
        """
        if self.lane is None or self.lane_index is None:
            return False

        lane = self.lane
        lane_index = self.lane_index
        downstream = self.road.network.graph.get(lane_index[1], {})
        if downstream:
            return False

        local_s, _ = lane.local_coordinates(self.position)
        removal_margin = 0.5 * max(float(self.real_length), 0.0)
        return bool(local_s >= lane.length - removal_margin)

    def act(self, action: dict | str = None):
        """
        Only act (IDM/MOBIL) once we are overtaken.
        While in replay mode (not overtaken), we don't call Vehicle.act().
        """
        if self.crashed or self.remove_from_road:
            return
        if not self.overtaken:
            return
        # Reuse IDMVehicle.act for actual control once taken over
        super().act(action)

    # ---------------- collision prevention ----------------

    def _update_from_trajectory(self):
        """
        Apply one replay step from ngsim_traj[sim_steps] -> [sim_steps+1].
        Sets position, speed, heading, lane_index/lane.
        """
        if self.ngsim_traj is None:
            self.overtaken = True
            return

        # Switch to IDM if trajectory is expired
        if self.sim_steps + 1 >= len(self.ngsim_traj):
            self.overtaken = True
            return

        # Current and next samples: [x, y, v, lane_id]
        cur_x, cur_y, cur_v, cur_lane = self.ngsim_traj[self.sim_steps][:4]
        nxt_x, nxt_y, nxt_v, _        = self.ngsim_traj[self.sim_steps + 1][:4]

        # Position from data
        self.position = np.array([cur_x, cur_y], dtype=float)

        # Update appear/visible + footprint for this frame
        self._set_visibility_from_appearance(cur_x != 0.0)

        # Speed from spatial difference (fallback to recorded v)
        dx = nxt_x - cur_x
        dy = nxt_y - cur_y
        dist = math.hypot(dx, dy)
        data_dt = self.DATA_DT

        speed_est = dist / utils.not_zero(data_dt)
        self.speed = speed_est if speed_est > 1e-3 else float(cur_v)
        self.target_speed = self.speed

        # Prefer the recorded lane id during replay; fall back to closest geometry.
        mapped_lane_index = target_lane_index_from_lane_id(
            self.road.network, self.SCENE, self.position[0], cur_lane
        )
        self.lane_index = (
            mapped_lane_index
            if mapped_lane_index is not None
            else self.road.network.get_closest_lane_index(self.position)
        )
        self.lane = self.road.network.get_lane(self.lane_index)

        local_s, _local_r = self.lane.local_coordinates(self.position)
        self.heading = self.lane.heading_at(local_s)

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
        if isinstance(front_vehicle, EgoVehicle):
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
        if self.remove_from_road:
            return

        # No trajectory: behave as a pure IDM vehicle
        if self.ngsim_traj is None or len(self.ngsim_traj) == 0:
            self.overtaken = True
            super().step(dt)
            if self._reached_terminal_lane_end():
                self._mark_for_removal()
            return

        # Timer / histories
        self.timer += dt
        self.heading_history.append(self.heading)
        self.speed_history.append(self.speed)
        self.crash_history.append(self.crashed)
        self.overtaken_history.append(self.overtaken)

        gap, desired_gap, ego_ahead = self._front_gap_logic()

        # Still have replay data?
        replay_exhausted = self.sim_steps + 1 >= len(self.ngsim_traj)
        can_replay = (not self.overtaken) and (not replay_exhausted)

        # --- Replay vs takeover logic ---
        #  - If no ego ahead: ignore gaps, just replay while we have data.
        #  - If ego ahead: only replay when gap >= desired_gap.
        if can_replay and (not ego_ahead or gap >= desired_gap):
            # Keep replaying the NGSIM trajectory
            self._update_from_trajectory()
            if self._reached_terminal_lane_end():
                self._mark_for_removal()
                self._record_position()
                return
            self.sim_steps += 1
        else:
            if not self.overtaken and replay_exhausted:
                self._mark_for_removal()
                self._record_position()
                return

            # Handover to IDM/MOBIL
            self.overtaken = True
            self.color = (100, 200, 255)

            # Use lane_id + x to pick a target lane index (your existing mapping)
            lane_id = int(
                self.ngsim_traj[min(self.sim_steps, len(self.ngsim_traj) - 1)][3]
            )
            x = self.position[0]
            target_lane_index = target_lane_index_from_lane_id(
                self.road.network, self.SCENE, x, lane_id
            )

            if target_lane_index is not None:
                self.target_lane_index = target_lane_index

            # Do not hand ghost vehicles with zero footprint to the IDM bicycle model.
            if not self.appear or self.LENGTH <= 0.0 or self.WIDTH <= 0.0:
                self.speed = 0.0
                self.target_speed = 0.0
                self._record_position()
                return

            # Now evolve with IDM/MOBIL dynamics
            super().step(dt)
            if self._reached_terminal_lane_end():
                self._mark_for_removal()

        # Record replayed / simulated position
        self._record_position()


    # ---------------- Collision handling ----------------
    def handle_collisions(self, other: IDMVehicle, dt: float = 0.0) -> None:
        """
        Collision handling that plugs into RoadObject.handle_collisions, but keeps
        the NGSIM-specific behaviour:

        - Respect COLLISIONS_ENABLED on both sides.
        - Ignore NGSIM-vs-NGSIM collisions while both are still in replay
        (both overtaken == False).
        - Ignore collisions involving vehicles that have not yet appeared
        (ngsim_traj row is [0,0,0,0] → appear == False).
        - Delegate geometry and crash flags to the base implementation.
        - When a new crash happens, clamp both speeds to the smaller magnitude.
        """

        # 1. Global toggle: if either side disabled, do nothing.
        if self.remove_from_road or getattr(other, "remove_from_road", False):
            return
        if not getattr(self, "COLLISIONS_ENABLED", True) or \
        not getattr(other, "COLLISIONS_ENABLED", True):
            return

        # 2. NEW: ghost/“not yet appeared” guard.
        #    If either vehicle is marked as not appearing, skip collisions entirely.
        if hasattr(self, "appear") and not self.appear:
            return
        if hasattr(other, "appear") and not getattr(other, "appear", True):
            return

        # 3. Skip NGSIM-vs-NGSIM collisions while both still in replay mode.
        if isinstance(self, NGSIMVehicle) and isinstance(other, NGSIMVehicle):
            if not self.overtaken and not getattr(other, "overtaken", False):
                return

        # 4. Remember prior crash states to detect a new crash.
        pre_crash_self = self.crashed
        pre_crash_other = getattr(other, "crashed", False)

        # 5. Delegate actual geometry + crash logic.
        super().handle_collisions(other, dt)

        # 6. If a new crash has occurred, clamp speeds.
        post_crash_self = self.crashed
        post_crash_other = getattr(other, "crashed", False)

        new_crash_happened = (not pre_crash_self or not pre_crash_other) and \
                            (post_crash_self and post_crash_other)

        if new_crash_happened and hasattr(other, "speed"):
            min_speed = min(self.speed, other.speed, key=abs)
            self.speed = other.speed = min_speed

    
def spawn_surrounding_vehicles(
    trajectory_set,
    ego_start_index,
    max_surrounding,
    road,
    f2m_conv=FEET_PER_METER,
    scene = "us-101"
):
    """Spawn surrounding vehicles based on the given trajectory set.

    max_surrounding:
        - int > 0: spawn up to that many vehicles
        - None: spawn all available vehicles
    """
    if isinstance(ego_start_index, dict):
        shared_start_index = max(int(idx) for idx in ego_start_index.values()) if ego_start_index else 0
    else:
        shared_start_index = int(ego_start_index)

    spawned = 0
    unlimited = max_surrounding is None

    def _first_valid_index(traj: np.ndarray) -> int | None:
        nonzero = np.any(traj[:, :3] != 0.0, axis=1)
        if not np.any(nonzero):
            return None
        return int(np.argmax(nonzero))

    def _heading_from_traj(traj: np.ndarray, idx: int) -> float:
        """Estimate heading from the current and next non-zero trajectory samples."""
        if len(traj) < 2:
            return 0.0

        cur = traj[idx]
        for next_idx in range(idx + 1, len(traj)):
            nxt = traj[next_idx]
            if np.any(nxt[:2] != 0.0):
                dx = float(nxt[0] - cur[0])
                dy = float(nxt[1] - cur[1])
                if math.hypot(dx, dy) > 1e-3:
                    return float(math.atan2(dy, dx))
        return 0.0

    def _intersects_existing(
        position: np.ndarray,
        heading: float,
        length: float,
        width: float,
    ) -> bool:
        candidate = (position, length, width, heading)
        for other in road.vehicles:
            if not getattr(other, "visible", True):
                continue
            other_length = float(getattr(other, "LENGTH", 0.0))
            other_width = float(getattr(other, "WIDTH", 0.0))
            if other_length <= 0.0 or other_width <= 0.0:
                continue
            other_rect = (
                np.array(other.position, dtype=float),
                other_length,
                other_width,
                float(getattr(other, "heading", 0.0)),
            )
            if utils.rotated_rectangles_intersect(candidate, other_rect):
                return True
        return False

    ego_anchor_positions: list[np.ndarray] = []
    ego_records = trajectory_set.get("ego", {})
    if isinstance(ego_records, dict):
        for ego_id, ego_meta in ego_records.items():
            ego_traj_full = process_raw_trajectory(ego_meta["trajectory"], scene)
            if len(ego_traj_full) <= shared_start_index:
                continue
            ego_traj = ego_traj_full[shared_start_index:]
            ego_first_idx = _first_valid_index(ego_traj)
            if ego_first_idx is None:
                continue
            ego_anchor_positions.append(
                np.array(ego_traj[ego_first_idx][:2], dtype=float)
            )

    candidates = []
    for vid, meta in trajectory_set.items():
        if vid == "ego":
            continue

        traj_full = process_raw_trajectory(meta["trajectory"], scene)
        if len(traj_full) <= shared_start_index:
            continue

        traj = traj_full[shared_start_index:]
        first_idx = _first_valid_index(traj)
        if first_idx is None:
            continue

        if len(traj[first_idx:]) < 2:
            continue

        first_pos = np.array(traj[first_idx][:2], dtype=float)
        if ego_anchor_positions:
            priority = min(
                float(np.linalg.norm(first_pos - ego_pos))
                for ego_pos in ego_anchor_positions
            )
        else:
            priority = float("inf")

        candidates.append((priority, int(vid), meta, traj, first_idx))

    if not unlimited:
        candidates.sort(key=lambda item: (item[0], item[1]))

    for _priority, vid, meta, traj, first_idx in candidates:
        if not unlimited and spawned >= max_surrounding:
            break

        traj = traj[first_idx:]
        if len(traj) < 2:
            continue

        if scene == "us-101":
            v_length = meta["length"] / f2m_conv
            v_width = meta["width"] / f2m_conv
        else:
            v_length = meta["length"]
            v_width = meta["width"]

        spawn_idx = None
        spawn_heading = 0.0
        for idx in range(len(traj) - 1):
            pos = np.array(traj[idx][:2], dtype=float)
            if not np.any(pos != 0.0):
                continue
            heading = _heading_from_traj(traj, idx)
            if not _intersects_existing(pos, heading, v_length, v_width):
                spawn_idx = idx
                spawn_heading = heading
                break

        if spawn_idx is None:
            continue

        traj = traj[spawn_idx:]
        if len(traj) < 2:
            continue
        
        v = NGSIMVehicle.create(
            road=road,
            vehicle_ID=vid,
            position=traj[0][:2],
            v_length=v_length,
            v_width=v_width,
            ngsim_traj=traj,
            scene=scene,
            heading=spawn_heading,
            speed=traj[0][2],
            color=(200, 0, 150),
        )
        road.vehicles.append(v)
        spawned += 1
