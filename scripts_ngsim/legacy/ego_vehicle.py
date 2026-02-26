from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle

from highway_env.ngsim_utils.gen_road import clamp_location_ngsim


class EgoVehicle(Vehicle):
    """
    Ego vehicle with two control modes:
      - continuous: expects low-level dict action {"steering": float, "acceleration": float}
      - discrete:   expects high-level string action:
          {"FASTER","SLOWER","IDLE","STEER_LEFT","STEER_RIGHT","LANE_LEFT","LANE_RIGHT"}

    Discrete semantics:
      - Speed: discrete ladder (FASTER/SLOWER) + IDLE
      - Steering bias: within-lane lateral offset setpoint (STEER_LEFT/RIGHT)
      - Lane change: switches target lane index (LANE_LEFT/RIGHT)

    NGSIM road considerations (your create_ngsim_101_road):
      - lane count changes by section: ("s1","s2")=5, ("s2","s3")=6, ("s3","s4")=5
      - to be robust, lane changes must be computed on the *current section edge* based on x-position
      - within-lane offsets must be clamped using lane width and vehicle width

    Notes:
      - "LANE_LEFT/RIGHT" below follow highway-env convention: lane_id - 1 is left, lane_id + 1 is right.
        If this appears visually flipped in your renderer, swap the direction signs in _apply_lane_change().
    """

    # --------------------------
    # Controller gains/limits
    # --------------------------
    TAU_ACC = 0.6
    TAU_HEADING = 0.2
    TAU_LATERAL = 0.6

    TAU_PURSUIT = 0.5 * TAU_HEADING
    KP_A = 1.0 / TAU_ACC
    KP_HEADING = 1.0 / TAU_HEADING
    KP_LATERAL = 1.0 / TAU_LATERAL

    MAX_STEERING_ANGLE = np.pi / 4

    # --------------------------
    # Discrete speed grid defaults
    # --------------------------
    DEFAULT_TARGET_SPEEDS = np.arange(0.0, 30.0 + 1e-6, 2.0)

    # --------------------------
    # Discrete steering setpoint defaults
    # --------------------------
    DEFAULT_LATERAL_OFFSET_STEP = 0.30  # [m] per action
    DEFAULT_LATERAL_OFFSET_MAX = 1.50   # [m] absolute max cap (further clamped by lane & vehicle width)
    DEFAULT_OFFSET_MARGIN = 0.10        # [m] keep some clearance to lane boundary

    # Lane change behavior
    DEFAULT_LANE_CHANGE_COOLDOWN_STEPS = 10  # e.g., 1s at 10Hz

    # Fallback lane width if geometry doesn't expose it (should not happen for StraightLane)
    DEFAULT_LANE_WIDTH_FALLBACK = 12 / 3.281  # ~3.66m, matches your road builder

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0.0,
        speed: float = 0.0,
        target_lane_index: LaneIndex | None = None,
        target_speed: float | None = None,
        route: Route | None = None,
        control_mode: str = "discrete",
        target_speeds: Optional[Vector] = None,
        lateral_offset_step: float = DEFAULT_LATERAL_OFFSET_STEP,
        lateral_offset_max: float = DEFAULT_LATERAL_OFFSET_MAX,
        lane_change_cooldown_steps: int = DEFAULT_LANE_CHANGE_COOLDOWN_STEPS,
    ) -> None:
        super().__init__(road, position, heading, speed)

        if control_mode not in ("discrete", "continuous"):
            raise ValueError("control_mode must be 'discrete' or 'continuous'.")

        self.route = route
        self.control_mode = control_mode

        # Target lane index (will be projected to correct NGSIM section on first act())
        self.target_lane_index = self.lane_index

        # Continuous mode memory
        self._last_low_level_action = {"steering": 0.0, "acceleration": 0.0}

        # Discrete speed ladder
        self.target_speeds = (
            np.array(target_speeds, dtype=float)
            if target_speeds is not None
            else np.array(self.DEFAULT_TARGET_SPEEDS, dtype=float)
        )
        if self.target_speeds.ndim != 1 or self.target_speeds.size < 2:
            raise ValueError("target_speeds must be a 1D array with at least 2 elements.")
        if not np.all(np.isfinite(self.target_speeds)):
            raise ValueError("target_speeds contains non-finite values.")
        if np.any(np.diff(self.target_speeds) <= 0):
            raise ValueError("target_speeds must be strictly increasing.")

        init_target_speed = float(target_speed if target_speed is not None else self.speed)
        self.speed_index = int(self.speed_to_index(init_target_speed))
        self.target_speed = float(self.index_to_speed(self.speed_index))

        # Within-lane lateral offset setpoint
        self.lateral_offset = 0.0
        self.lateral_offset_step = float(lateral_offset_step)
        self.lateral_offset_max = float(lateral_offset_max)

        # Lane-change cooldown
        self.lane_change_cooldown_steps = int(max(0, lane_change_cooldown_steps))
        self._lane_change_cooldown = 0

        # Lange-change direction
        self.lane_change_direction = 0

    # --------------------------
    # Utility setters
    # --------------------------
    def set_control_mode(self, mode: str) -> None:
        if mode not in ("discrete", "continuous"):
            raise ValueError("mode must be 'discrete' or 'continuous'.")
        self.control_mode = mode

    def set_ego_dimension(self, width: float, length: float) -> None:
        self.WIDTH = float(width)
        self.LENGTH = float(length)

    # --------------------------
    # Discrete speed ladder helpers
    # --------------------------
    def index_to_speed(self, index: int) -> float:
        index = int(np.clip(int(index), 0, self.target_speeds.size - 1))
        return float(self.target_speeds[index])

    def speed_to_index(self, speed: float) -> int:
        speed = float(speed)
        diffs = np.diff(self.target_speeds)
        uniform = np.allclose(diffs, diffs[0], rtol=1e-4, atol=1e-6)

        if uniform:
            lo = float(self.target_speeds[0])
            hi = float(self.target_speeds[-1])
            if hi <= lo:
                return 0
            x = (speed - lo) / (hi - lo)
            idx = int(np.round(x * (self.target_speeds.size - 1)))
            return int(np.clip(idx, 0, self.target_speeds.size - 1))

        return int(np.argmin(np.abs(self.target_speeds - speed)))

    # --------------------------
    # NGSIM section helpers (consistent with your create_ngsim_101_road)
    # --------------------------
    def _main_edge_from_x(self, x: float) -> tuple[str, str]:
        length = 2150 / 3.281
        ends = [0.0, 560 / 3.281, (698 + 578 + 150) / 3.281, length]

        x_m = float(x)
        if x_m < ends[1]:
            return ("s1", "s2")  # 5 lanes
        if x_m < ends[2]:
            return ("s2", "s3")  # 6 lanes
        return ("s3", "s4")      # 5 lanes

    def _current_edge_lane_index(self) -> LaneIndex:
        """
        Project current target_lane_index to the correct edge for the current x-position.
        This prevents invalid lane ids when the lane-count changes by road section.
        """
        edge = self._main_edge_from_x(self.position[0])

        # Preferred lane id = from target_lane_index if available, else from current lane_index, else 0
        lane_index = self.target_lane_index or self.lane_index
        lane_id_guess = 0
        if lane_index is not None:
            try:
                lane_id_guess = int(lane_index[2])
            except Exception:
                lane_id_guess = 0

        lanes_on_edge = self.road.network.graph[edge[0]][edge[1]]
        n_lanes = len(lanes_on_edge)
        lane_id = int(np.clip(lane_id_guess, 0, n_lanes - 1))
        return (edge[0], edge[1], lane_id)

    # --------------------------
    # Road/lane helpers
    # --------------------------
    def follow_road(self) -> None:
        """
        At end of lane, automatically switch to the next one.
        IMPORTANT: In your segmented NGSIM road, this should operate on a valid edge lane index.
        """
        if self.target_lane_index is None:
            self.target_lane_index = self._current_edge_lane_index()

        try:
            lane = self.road.network.get_lane(self.target_lane_index)
            if lane.after_end(self.position):
                self.target_lane_index = self.road.network.next_lane(
                    self.target_lane_index,
                    route=self.route,
                    position=self.position,
                    np_random=self.road.np_random,
                )
        except Exception:
            # If anything goes wrong (e.g., stale lane index), re-project to the current edge
            self.target_lane_index = self._current_edge_lane_index()

    def _lane_width(self, lane_index: LaneIndex) -> float:
        """
        Best-effort lane width:
          - StraightLane exposes .width, and that matches your road builder.
          - fallback to constant if needed.
        """
        try:
            lane = self.road.network.get_lane(lane_index)
            w = float(getattr(lane, "width", np.nan))
            if np.isfinite(w) and w > 0.5:
                return w
        except Exception:
            pass
        return float(self.DEFAULT_LANE_WIDTH_FALLBACK)

    def _max_safe_lateral_offset(self, lane_index: LaneIndex) -> float:
        """
        Defines the 'invisible wall' for STEER actions.
        
        User Goal: "Threshold = Lane's Edge + Half of Car"
        Interpretation: The vehicle's center can drift up to the lane boundary, 
                        or slightly past it, but cannot drift fully into the next lane
                        without a discrete LANE_CHANGE action.
        """
        lane_w = float(self._lane_width(lane_index))
        if not np.isfinite(lane_w):
            lane_w = 3.7 
        limit = lane_w / 2.0
        return float(limit)

    def _adjacent_lane_index(self, direction: int) -> LaneIndex | None:
        """
        direction: -1 means lane_id-1, +1 means lane_id+1 (highway-env lane id convention).
        Uses section-aware lane counts based on current x-position.
        """
        base = self._current_edge_lane_index()
        _from, _to, _id = base

        try:
            lanes = self.road.network.graph[_from][_to]
            n = len(lanes)
            new_id = int(np.clip(_id + int(direction), 0, n - 1))
            if new_id == _id:
                return None

            new_index = (_from, _to, new_id)

            # Reachability check if available; otherwise accept the index
            try:
                if self.road.network.get_lane(new_index).is_reachable_from(self.position):
                    return new_index
                return None
            except Exception:
                return new_index
        except Exception:
            return None

    # --------------------------
    # Controllers
    # --------------------------
    def steering_control_with_offset(self, target_lane_index: LaneIndex, lateral_offset: float) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        s, ey = target_lane.local_coordinates(self.position)

        # Apply the lateral offset to adjust position
        ey = float(ey + lateral_offset)

        s_next = s + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(s_next)

        lateral_speed_command = -self.KP_LATERAL * ey

        heading_command = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(self.speed), -1.0, 1.0)
        )
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)

        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)

        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1.0,
                1.0,
            )
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        return float(self.KP_A * (float(target_speed) - float(self.speed)))

    # --------------------------
    # Discrete action application helpers
    # --------------------------
    def _apply_speed_action(self, act: str) -> None:
        if act == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
            self.speed_index = int(np.clip(self.speed_index, 0, self.target_speeds.size - 1))
            self.target_speed = self.index_to_speed(self.speed_index)
        elif act == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
            self.speed_index = int(np.clip(self.speed_index, 0, self.target_speeds.size - 1))
            self.target_speed = self.index_to_speed(self.speed_index)

    def _apply_steer_bias_action(self, act: str) -> None:
        # 1. Calculate the Hard Limit for the current lane
        max_off = self._max_safe_lateral_offset(self.target_lane_index)
        
        # 2. Apply Step
        if act == "STEER_LEFT":
            self.lateral_offset += self.lateral_offset_step
        elif act == "STEER_RIGHT":
            self.lateral_offset -= self.lateral_offset_step
        
        # 3. HARD CLAMP (The "Invisible Wall")
        # This forces the agent to switch to "LANE_LEFT" if it wants to move further.
        self.lateral_offset = float(np.clip(self.lateral_offset, -max_off, max_off))

    def _apply_lane_change(self, act: str) -> None:
        """
        Initiates a Ramped Lane Change.
        We do NOT change the lane index yet. We just tell the controller to start
        walking the lateral offset towards the new lane.
        """
        if self._lane_change_cooldown > 0 or self.lane_change_direction != 0:
            return

        # Highway-env convention: -1 is Left (lower ID), +1 is Right (higher ID)
        direction = -1 if act == "LANE_LEFT" else 1
        
        # Verify the target lane actually exists before starting
        if self._adjacent_lane_index(direction) is not None:
            self.lane_change_direction = direction
            self._lane_change_cooldown = self.lane_change_cooldown_steps
            
    # --------------------------
    # Control loop entrypoint
    # --------------------------
    def act(self, action: Union[dict, str, int, None] = None) -> None:
        """
        Supports both:
          - CONTINUOUS: action is {"steering": float, "acceleration": float}
          - DISCRETE: action is a string like "LANE_LEFT" or "FASTER"
        """
        # 1. Update Cooldowns
        if self._lane_change_cooldown > 0:
            self._lane_change_cooldown -= 1

        # ---------------------------------------------------------
        # Path A: CONTINUOUS MODE 
        # ---------------------------------------------------------
        if self.control_mode == "continuous":
            # Sync target lane to current position to keep road-following logic alive
            self.target_lane_index = self._current_edge_lane_index()
            self.follow_road()

            if action is None:
                # Repeat last known low-level command if no new action
                super().act(self._last_low_level_action)
                return

            if isinstance(action, dict):
                steering = float(action.get("steering", 0.0))
                acceleration = float(action.get("acceleration", 0.0))
            else:
                # Fallback for unexpected action types in continuous mode
                steering, acceleration = 0.0, 0.0

            steering = float(np.clip(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))
            self._last_low_level_action = {"steering": steering, "acceleration": acceleration}
            
            super().act(self._last_low_level_action)
            return

        # ---------------------------------------------------------
        # Path B: DISCRETE MODE
        # ---------------------------------------------------------
        # Normalize the discrete action input
        act = "IDLE"
        if isinstance(action, str):
            act = action.upper()

        # ---------------------------------------------------------
        # 1. Handle New Commands
        # ---------------------------------------------------------
        if act in ("LANE_LEFT", "LANE_RIGHT"):
            self._apply_lane_change(act)
        elif act in ("FASTER", "SLOWER"):
            self._apply_speed_action(act)
        # Note: We ignore manual STEER inputs if a lane change is active to prevent fighting
        elif act in ("STEER_LEFT", "STEER_RIGHT") and self.lane_change_direction == 0:
            self._apply_steer_bias_action(act)

        # ---------------------------------------------------------
        # 2. Process Ramped Lane Change (The "Macro")
        # ---------------------------------------------------------
        if self.lane_change_direction != 0:
            # A. Move the target offset incrementally (Same magnitude as manual steer)
            # If moving Left (-1), we decrease offset? 
            # WAIT: Check your coordinate system! 
            # In highway-env, usually +y is "Left" relative to lane center if heading is 0.
            # But let's assume standard: direction -1 (Left) implies we want to go to *that* side.
            # We used direction * lateral_offset_step in steer bias? 
            # Let's align with your _apply_steer_bias_action logic:
            # STEER_LEFT (dir -1?) -> lateral_offset += step (Positive is Left)
            
            step = self.lateral_offset_step
            if self.lane_change_direction == -1: # LEFT
                self.lateral_offset += step
            else: # RIGHT
                self.lateral_offset -= step

            # B. Check for Coordinate Swap (Crossing the line)
            # We swap when we are closer to the new lane than the old one (offset > width/2)
            lane_width = self._lane_width(self.target_lane_index)
            threshold = lane_width / 2.0
            
            # Check Left Crossing (Positive Offset > Width/2)
            if self.lane_change_direction == -1 and self.lateral_offset > threshold:
                new_idx = self._adjacent_lane_index(-1)
                if new_idx:
                    self.target_lane_index = new_idx
                    self.lateral_offset -= lane_width # Wrap coordinate
                else:
                    self.lane_change_direction = 0 # Abort if lane missing

            # Check Right Crossing (Negative Offset < -Width/2)
            elif self.lane_change_direction == 1 and self.lateral_offset < -threshold:
                new_idx = self._adjacent_lane_index(1)
                if new_idx:
                    self.target_lane_index = new_idx
                    self.lateral_offset += lane_width # Wrap coordinate
                else:
                    self.lane_change_direction = 0

            # C. Check for Completion (Centering)
            # If we have wrapped, the offset is now "shrinking" towards 0.
            # If we cross 0 (change sign) or get very close, we are done.
            if abs(self.lateral_offset) < step:
                self.lateral_offset = 0.0
                self.lane_change_direction = 0 # FINISHED

        # ---------------------------------------------------------
        # 3. Standard Constraints (Only clamp if NOT changing lanes)
        # ---------------------------------------------------------
        if self.lane_change_direction == 0:
             # Standard "Invisible Wall" logic applies here
             self.follow_road()
             if self._lane_change_cooldown == 0:
                 self.target_lane_index = self._current_edge_lane_index()
             
             max_off = self._max_safe_lateral_offset(self.target_lane_index)
             self.lateral_offset = float(np.clip(self.lateral_offset, -max_off, max_off))

        # ---------------------------------------------------------
        # 4. Low Level Control
        # ---------------------------------------------------------
        low_level_action = {
            "steering": self.steering_control_with_offset(self.target_lane_index, self.lateral_offset),
            "acceleration": self.speed_control(self.target_speed),
        }
        super().act(low_level_action)

    # --------------------------
    # Routing helpers (unchanged)
    # --------------------------
    def get_routes_at_intersection(self) -> List[Route]:
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [
            self.route[0 : index + 1]
            + [(self.route[index][1], destination, self.route[index][2])]
            for destination in next_destinations_from
        ]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.integers(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        pos_heads = [
            self.road.network.position_heading_along_route(
                route, coordinates[0] + self.speed * t, 0, self.lane_index
            )
            for t in times
        ]
        return tuple(zip(*pos_heads))


def create_ego_vehicle(net, road, ego_traj, ego_len, ego_wid, ego_start_index, config):
    """Create the ego vehicle and set its initial position, speed, and heading."""
    ego_traj = ego_traj[ego_start_index :]
    if len(ego_traj) < 2:
        raise RuntimeError(f"Ego trajectory too short after truncation (len={len(ego_traj)}).")

    x0, y0, ego_speed, lane0 = ego_traj[0]
    ego_xy = np.array([x0, y0], dtype=float)
    # Compute heading from ego_trajectory
    dx0 = ego_traj[1, 0] - ego_traj[0, 0]
    dy0 = ego_traj[1, 1] - ego_traj[0, 1]

    disp = np.hypot(dx0, dy0)

    MIN_DISP = 0.1  # meters (reasonable for NGSIM @ 10Hz)

    if disp >= MIN_DISP:
        heading_raw = np.arctan2(dy0, dx0)
    else:
        heading_raw = 0.0  # safe only for first frame

    ego_lane = clamp_location_ngsim(x0, lane0, net, warning=False)
    target_lane_index = ego_lane.index  # LaneIndex tuple

    expert_mode = bool(config.get("expert_test_mode", False))
    expert_action_mode = str(config.get("expert_action_mode", "continuous"))
    
    if expert_mode and expert_action_mode == "discrete":
            ego_control_mode = "discrete"
    else:
        ego_control_mode = "continuous" if (expert_mode and expert_action_mode == "continuous") else "continuous"

    ego = EgoVehicle(
        road=road,
        position=ego_xy,
        speed=ego_speed,
        heading=heading_raw,
        control_mode=ego_control_mode,
        target_lane_index=target_lane_index,
        target_speeds=np.array(config.get("target_speeds", []), dtype=float) if config.get("target_speeds", None) is not None else None,
        lane_change_cooldown_steps=int(config.get("lane_change_cooldown_steps", 10)),
        lateral_offset_step=float(config.get("lateral_offset_step", EgoVehicle.DEFAULT_LATERAL_OFFSET_STEP)),
        lateral_offset_max=float(config.get("lateral_offset_max", EgoVehicle.DEFAULT_LATERAL_OFFSET_MAX)),
    )
    ego.set_ego_dimension(width=ego_wid, length=ego_len)

    return ego
