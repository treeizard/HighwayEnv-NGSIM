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

import math
import numpy as np

from highway_env import utils
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.ngsim_utils.constants import FEET_PER_METER
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle
from highway_env.ngsim_utils.helper_ngsim import target_lane_index_from_lane_id
from highway_env.ngsim_utils.trajectory_gen import (
    first_valid_index as first_active_index,
    process_raw_trajectory,
    trajectory_row_is_active,
)


def _road_entity_blocks_spawn(entity) -> bool:
    if getattr(entity, "remove_from_road", False):
        return False
    if hasattr(entity, "visible") and not bool(getattr(entity, "visible", True)):
        return False
    if hasattr(entity, "appear") and not bool(getattr(entity, "appear", True)):
        return False
    if hasattr(entity, "scene_collection_is_active") and not bool(
        getattr(entity, "scene_collection_is_active", True)
    ):
        return False
    if float(getattr(entity, "LENGTH", 0.0)) <= 0.0 or float(getattr(entity, "WIDTH", 0.0)) <= 0.0:
        return False
    return True


def _rectangle_polygon(
    position: np.ndarray,
    heading: float,
    length: float,
    width: float,
) -> np.ndarray:
    points = np.array(
        [
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
        ],
        dtype=float,
    ).T
    c, s = np.cos(float(heading)), np.sin(float(heading))
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    polygon = (rotation @ points).T + np.tile(np.asarray(position, dtype=float), (4, 1))
    return np.vstack([polygon, polygon[0:1]])


def road_entity_pose_polygon(
    position: np.ndarray,
    heading: float,
    length: float,
    width: float,
) -> np.ndarray:
    return _rectangle_polygon(position, heading, length, width)


def road_entity_conflicts_at_pose(
    road,
    position,
    *,
    heading: float,
    length: float,
    width: float,
    ignore_entity=None,
) -> bool:
    position_arr = np.asarray(position, dtype=float)
    diagonal = float(np.hypot(length, width))
    candidate_polygon = _rectangle_polygon(position_arr, heading, length, width)
    zero_velocity = np.zeros(2, dtype=float)

    for entity in list(getattr(road, "vehicles", [])) + list(getattr(road, "objects", [])):
        if entity is ignore_entity or not _road_entity_blocks_spawn(entity):
            continue
        entity_position = np.asarray(getattr(entity, "position", None), dtype=float)
        entity_diagonal = float(getattr(entity, "diagonal", np.hypot(entity.LENGTH, entity.WIDTH)))
        if np.linalg.norm(entity_position - position_arr) > 0.5 * (diagonal + entity_diagonal):
            continue
        intersecting, _will_intersect, _transition = utils.are_polygons_intersecting(
            candidate_polygon,
            entity.polygon(),
            zero_velocity,
            zero_velocity,
        )
        if intersecting:
            return True
    return False


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
    HIDDEN_POSITION = np.array([0.0, 0.0], dtype=float)
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
        allow_idm: bool = True,
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
            params_profile=scene,
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
        self._has_appeared_once = False
        self.allow_idm = bool(allow_idm)
        self._debug_handover_logged = False
        self.idm_handover_step: int | None = None
        self.idm_handover_reason: str | None = None

        # ---- Initialise from trajectory instead of dummy (0,0) ----
        if self.ngsim_traj is not None and len(self.ngsim_traj) > 0:
            # Current sample (sim_steps == 0 initially)
            cur_x, cur_y, cur_v, _ = self.ngsim_traj[self.sim_steps][:4]

            # Use data pose as initial pose
            self.position = np.array([cur_x, cur_y], dtype=float)
            self.speed = float(cur_v)
            self.target_speed = self.speed

            self.appear = trajectory_row_is_active(self.ngsim_traj[self.sim_steps])
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
            self.position = self.HIDDEN_POSITION.copy()
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
        allow_idm: bool = True,
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
            allow_idm=allow_idm,
        )

    # ---------------- Behaviour ----------------
    def _update_diagonal(self) -> None:
        self.diagonal = float(np.hypot(self.LENGTH, self.WIDTH))

    def _set_visibility_from_appearance(self, appear: bool) -> None:
        self.appear = bool(appear)
        self.visible = self.appear
        if self.appear:
            self._has_appeared_once = True
            self.LENGTH = self.real_length
            self.WIDTH = self.real_width
        else:
            self.position = self.HIDDEN_POSITION.copy()
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
        self.position = self.HIDDEN_POSITION.copy()
        self._set_visibility_from_appearance(False)

    def _row_heading(self, row: np.ndarray, next_row: np.ndarray | None = None) -> float:
        row_arr = np.asarray(row, dtype=float)
        x, y, _speed, lane_id = row_arr[:4]
        mapped_lane_index = target_lane_index_from_lane_id(
            self.road.network, self.SCENE, float(x), int(lane_id)
        )
        if mapped_lane_index is not None:
            lane = self.road.network.get_lane(mapped_lane_index)
            local_s, _local_r = lane.local_coordinates(np.array([x, y], dtype=float))
            return float(lane.heading_at(local_s))
        if next_row is not None and trajectory_row_is_active(next_row):
            next_arr = np.asarray(next_row, dtype=float)
            dx = float(next_arr[0] - x)
            dy = float(next_arr[1] - y)
            if math.hypot(dx, dy) > 1e-3:
                return float(math.atan2(dy, dx))
        return float(self.heading)

    def _spawn_row_is_clear(self, row: np.ndarray, next_row: np.ndarray | None = None) -> bool:
        row_arr = np.asarray(row, dtype=float)
        return not road_entity_conflicts_at_pose(
            self.road,
            row_arr[:2],
            heading=self._row_heading(row_arr, next_row=next_row),
            length=float(self.real_length),
            width=float(self.real_width),
            ignore_entity=self,
        )

    def _lane_index_contains_current_pose(self, lane_index) -> bool:
        if self.road is None or lane_index is None:
            return False
        try:
            lane = self.road.network.get_lane(lane_index)
        except Exception:
            return False
        local_s, local_r = lane.local_coordinates(self.position)
        return bool(lane.on_lane(self.position, local_s, local_r, margin=0.05))

    def _set_lane_from_recorded_lane_id(self, lane_id: int) -> bool:
        lane_index = target_lane_index_from_lane_id(
            self.road.network,
            self.SCENE,
            float(self.position[0]),
            int(lane_id),
        )
        if lane_index is None or not self._lane_index_contains_current_pose(lane_index):
            return False
        self.lane_index = lane_index
        self.lane = self.road.network.get_lane(lane_index)
        return True

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

    def on_state_update(self) -> None:
        """
        Preserve the current/recorded lane while the pose is still on it.

        NGSIM vehicles often hand over near lane boundaries. The default closest-lane
        update can snap them into an adjacent lane before IDM has had a chance to
        steer or brake, which creates artificial lateral motion into neighbouring
        traffic. Once the vehicle genuinely leaves its current lane footprint, fall
        back to the normal closest-lane update.
        """
        if self.road and self._lane_index_contains_current_pose(self.lane_index):
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))
            return
        super().on_state_update()

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
        self._set_visibility_from_appearance(
            trajectory_row_is_active(self.ngsim_traj[self.sim_steps])
        )

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

    def _front_vehicle_requires_takeover(self, front_vehicle) -> bool:
        """
        Decide whether a detected front vehicle should influence replay handover.

        Pure replay-vs-replay traffic is allowed to ignore each other so recorded
        trajectories can coexist. Once a front actor is no longer a plain replay
        obstacle (ego vehicle, IDM-taken-over replay vehicle, crashed replay
        vehicle, obstacle, etc.), the follower must respect the real gap.
        """
        if front_vehicle is None:
            return False
        if getattr(front_vehicle, "remove_from_road", False):
            return False
        if hasattr(front_vehicle, "appear") and not bool(getattr(front_vehicle, "appear", True)):
            return False
        if float(getattr(front_vehicle, "LENGTH", 0.0)) <= 0.0:
            return False
        if float(getattr(front_vehicle, "WIDTH", 0.0)) <= 0.0:
            return False
        if isinstance(front_vehicle, EgoVehicle):
            return True
        if isinstance(front_vehicle, NGSIMVehicle):
            return bool(
                getattr(front_vehicle, "overtaken", False)
                or getattr(front_vehicle, "crashed", False)
            )
        return True

    def _should_handover_for_front_vehicle(
        self,
        front_vehicle,
        gap: float,
        desired_gap: float,
    ) -> tuple[bool, str]:
        """
        Decide whether a relevant front vehicle should force replay -> IDM handover.

        A small deficit relative to the full IDM desired gap is often not enough
        to justify abandoning replay. We only hand over when the situation
        shows concrete collision risk: strong required braking, a very short
        gap while still closing, or a short time-to-contact with a materially
        compressed headway.
        """
        if front_vehicle is None:
            return False, "no_front_vehicle"

        front_speed = max(float(getattr(front_vehicle, "speed", 0.0)), 0.0)
        ego_speed = max(float(self.speed), 0.0)
        closing_speed = max(0.0, ego_speed - front_speed)
        required_brake = float(self._required_braking_to_avoid_contact(self, front_vehicle, gap))
        gap_deficit = max(0.0, float(desired_gap) - float(gap))
        time_to_contact = float(gap / closing_speed) if closing_speed > 1e-3 else float("inf")

        material_deficit = gap_deficit > max(2.0, 0.1 * float(desired_gap))
        critical_gap = float(gap) < max(6.0, 0.45 * float(desired_gap))
        strong_braking_needed = required_brake < -0.75
        moderate_closing = closing_speed > 1.0
        short_time_to_contact = time_to_contact < 3.0

        trigger_labels: list[str] = []
        if strong_braking_needed:
            trigger_labels.append("required_brake")
        if critical_gap and moderate_closing:
            trigger_labels.append("critical_gap")
        if material_deficit and short_time_to_contact:
            trigger_labels.append("short_ttc")

        should_handover = bool(trigger_labels)
        trigger_text = ",".join(trigger_labels) if trigger_labels else "none"
        reason = (
            f"front_gap trigger={trigger_text} gap={gap:.3f} desired_gap={desired_gap:.3f} "
            f"closing_speed={closing_speed:.3f} required_brake={required_brake:.3f} "
            f"gap_deficit={gap_deficit:.3f} ttc={time_to_contact:.3f}"
        )
        return should_handover, reason

    def _front_gap_logic(self):
        """
        Decide which front vehicle counts for gap checking.

        We always care about the ego. We also care about any front actor that is
        no longer a plain replay vehicle, e.g. a replay vehicle that has already
        switched to IDM/MOBIL or that has crashed. Pure replay-only fronts remain
        ignored so matching recorded traffic can still replay in parallel.

        Returns:
            front_vehicle: object | None
            gap: float
            desired_gap: float
            relevant_front: bool
            handover_needed: bool
            reason: str
        """
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        # No front vehicle -> treat as safe, no relevant front vehicle
        if front_vehicle is None:
            return None, 100.0, 50.0, False, False, "no_front_vehicle"

        # If the front vehicle meaningfully constrains replay, use the real gap.
        if self._front_vehicle_requires_takeover(front_vehicle):
            gap = self.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(self, front_vehicle)
            handover_needed, reason = self._should_handover_for_front_vehicle(
                front_vehicle,
                gap,
                desired_gap,
            )
            return front_vehicle, gap, desired_gap, True, handover_needed, reason

        # A plain replay front is ignored for takeover.
        return front_vehicle, 100.0, 50.0, False, False, "plain_replay_front"


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

        current_row = self.ngsim_traj[self.sim_steps]
        if not trajectory_row_is_active(current_row):
            # Before a replay vehicle appears in the scene, keep advancing its
            # internal replay clock without running interaction or takeover logic.
            if self._has_appeared_once:
                self._mark_for_removal()
                self._record_position()
                return
            self.position = self.HIDDEN_POSITION.copy()
            self.speed = 0.0
            self.target_speed = 0.0
            self._set_visibility_from_appearance(False)
            if self.sim_steps + 1 >= len(self.ngsim_traj):
                self._mark_for_removal()
            else:
                self.sim_steps += 1
            self._record_position()
            return

        # Timer / histories
        self.timer += dt
        self.heading_history.append(self.heading)
        self.speed_history.append(self.speed)
        self.crash_history.append(self.crashed)
        self.overtaken_history.append(self.overtaken)

        front_vehicle, gap, desired_gap, relevant_front, handover_needed, handover_reason = self._front_gap_logic()

        # Still have replay data?
        replay_exhausted = self.sim_steps + 1 >= len(self.ngsim_traj)
        can_replay = (not self.overtaken) and (not replay_exhausted)
        next_row = (
            self.ngsim_traj[self.sim_steps + 1]
            if self.sim_steps + 1 < len(self.ngsim_traj)
            else None
        )

        if not self.allow_idm:
            if replay_exhausted:
                self._mark_for_removal()
                self._record_position()
                return
            if not self.appear and not self._spawn_row_is_clear(current_row, next_row=next_row):
                self.position = self.HIDDEN_POSITION.copy()
                self.speed = 0.0
                self.target_speed = 0.0
                self._set_visibility_from_appearance(False)
                self.sim_steps += 1
                self._record_position()
                return
            self._update_from_trajectory()
            if self._reached_terminal_lane_end():
                self._mark_for_removal()
                self._record_position()
                return
            self.sim_steps += 1
            self._record_position()
            return

        # --- Replay vs takeover logic ---
        #  - If no relevant front vehicle: ignore gaps and keep replaying.
        #  - If a relevant front vehicle is ahead: only replay when gap >= desired_gap.
        if can_replay and (not relevant_front or not handover_needed):
            if not self.appear and not self._spawn_row_is_clear(current_row, next_row=next_row):
                self.position = self.HIDDEN_POSITION.copy()
                self.speed = 0.0
                self.target_speed = 0.0
                self._set_visibility_from_appearance(False)
                self.sim_steps += 1
                self._record_position()
                return
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

            handover_reason = "replay_exhausted" if replay_exhausted else handover_reason

            # Handover to IDM/MOBIL
            self.overtaken = True
            self.idm_handover_step = int(self.sim_steps)
            self.idm_handover_reason = handover_reason
            self.color = (100, 200, 255)

            # Use lane_id + x to pick a target lane index (your existing mapping)
            lane_id = int(
                self.ngsim_traj[min(self.sim_steps, len(self.ngsim_traj) - 1)][3]
            )
            self._set_lane_from_recorded_lane_id(lane_id)
            x = self.position[0]
            target_lane_index = target_lane_index_from_lane_id(
                self.road.network, self.SCENE, x, lane_id
            )

            if target_lane_index is not None and self._handover_target_lane_is_safe(
                target_lane_index
            ):
                self.target_lane_index = target_lane_index
            else:
                self.target_lane_index = self.lane_index

            # After handover, stop using the last replay speed as the free-road target.
            # Otherwise the IDM free-road term can brake on open road simply because
            # the vehicle is a bit faster than the final replay sample.
            self.target_speed = float(max(self.speed, self.params.desired_speed))

            # Do not hand ghost vehicles with zero footprint to the IDM bicycle model.
            if not self.appear or self.LENGTH <= 0.0 or self.WIDTH <= 0.0:
                self.speed = 0.0
                self.target_speed = 0.0
                self._record_position()
                return

            # Road.act() already ran before this replay->IDM transition, so compute
            # the first IDM/MOBIL command immediately instead of stepping once with
            # the stale replay/no-op action.
            self.act()

            if not self._debug_handover_logged and bool(getattr(self.road, "debug_idm_handover", False)):
                debug_ids = getattr(self.road, "debug_idm_handover_ids", None)
                vehicle_id = getattr(self, "vehicle_ID", None)
                if debug_ids is None or vehicle_id in debug_ids:
                    print(
                        "vehicle id changed to idm:",
                        vehicle_id,
                        f"step={int(getattr(self, 'sim_steps', 0))}",
                        f"reason={handover_reason}",
                        f"lane={self.lane_index}",
                        f"target_lane={self.target_lane_index}",
                        flush=True,
                    )
                    self._debug_handover_logged = True

            # Now evolve with IDM/MOBIL dynamics
            super().step(dt)
            if self._reached_terminal_lane_end():
                self._mark_for_removal()

        # Record replayed / simulated position
        self._record_position()

    def _handover_target_lane_is_safe(self, target_lane_index) -> bool:
        """
        Keep replay-derived lane targets only when they are immediately safe.

        Handover may happen because the replay trajectory is already in conflict.
        If the recorded lane id asks for a lateral move into an occupied lane,
        committing to that target bypasses MOBIL's safety checks and can create a
        crash that braking in the current lane would have avoided.
        """
        if target_lane_index is None:
            return False
        if target_lane_index == self.lane_index:
            return True
        try:
            target_lane = self.road.network.get_lane(target_lane_index)
        except Exception:
            return False
        if not target_lane.is_reachable_from(self.position):
            return False

        target_front, _target_rear = self.road.neighbour_vehicles(
            self, target_lane_index
        )
        target_acc = self.acceleration(self, front_vehicle=target_front)
        return self._target_lane_is_safe_for_ego(
            target_lane_index,
            target_front,
            float(target_acc),
        )

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
    scene = "us-101",
    allow_idm: bool = True,
):
    """Spawn surrounding vehicles based on the given trajectory set.

    max_surrounding:
        - int > 0: spawn up to that many vehicles
        - None: spawn all available vehicles
    """
    if isinstance(ego_start_index, dict):
        shared_start_index = (
            max(int(idx) for idx in ego_start_index.values())
            if ego_start_index
            else 0
        )
    else:
        shared_start_index = int(ego_start_index)

    spawned = 0
    unlimited = max_surrounding is None

    def _heading_from_traj(traj: np.ndarray, idx: int) -> float:
        """Estimate heading from the current and next non-zero trajectory samples."""
        if len(traj) < 2:
            return 0.0

        cur = traj[idx]
        for next_idx in range(idx + 1, len(traj)):
            nxt = traj[next_idx]
            if trajectory_row_is_active(nxt):
                dx = float(nxt[0] - cur[0])
                dy = float(nxt[1] - cur[1])
                if math.hypot(dx, dy) > 1e-3:
                    return float(math.atan2(dy, dx))
        return 0.0

    ego_anchor_positions: list[np.ndarray] = []
    ego_records = trajectory_set.get("ego", {})
    if isinstance(ego_records, dict):
        for ego_id, ego_meta in ego_records.items():
            ego_traj_full = process_raw_trajectory(ego_meta["trajectory"], scene)
            if len(ego_traj_full) <= shared_start_index:
                continue
            ego_traj = ego_traj_full[shared_start_index:]
            ego_first_idx = first_active_index(ego_traj)
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
        first_idx = first_active_index(traj)
        if first_idx is None:
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

        if scene == "us-101":
            v_length = meta["length"] / f2m_conv
            v_width = meta["width"] / f2m_conv
        else:
            v_length = meta["length"]
            v_width = meta["width"]
        spawn_heading = _heading_from_traj(traj, first_idx)

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
            allow_idm=allow_idm,
        )
        spawn_row = np.asarray(traj[first_idx], dtype=float)
        next_row = (
            np.asarray(traj[first_idx + 1], dtype=float)
            if first_idx + 1 < len(traj)
            else None
        )
        if not v._spawn_row_is_clear(spawn_row, next_row=next_row):
            continue
        road.vehicles.append(v)
        spawned += 1
