from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle

from highway_env.ngsim_utils.helper_ngsim import edge_from_x


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

        # Lane-change direction: -1 left, +1 right, 0 none.
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
        return edge_from_x(self.road.network, x)

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
    
    def _safe_lateral_offset_bounds(self, lane_index: LaneIndex) -> tuple[float, float]:
        """
        Return (min_offset, max_offset) for within-lane STEER actions.

        Convention in this class:
        - positive lateral_offset  => move left
        - negative lateral_offset  => move right

        If there is no adjacent lane on one side (i.e. road edge), keep the vehicle body
        inside the road by subtracting half vehicle width and a small margin from that side.
        """
        lane_w = float(self._lane_width(lane_index))
        half_lane = lane_w / 2.0

        veh_half_w = float(getattr(self, "WIDTH", 2.0)) / 2.0
        margin = float(self.DEFAULT_OFFSET_MARGIN)

        # Absolute cap from config
        abs_cap = float(self.lateral_offset_max)

        # Start from full lane-boundary center limits
        max_left = half_lane
        max_right = half_lane

        # Check whether adjacent lanes exist
        has_left_lane = self._adjacent_lane_index(-1) is not None
        has_right_lane = self._adjacent_lane_index(1) is not None

        # If this side is the road edge, stop center earlier so body stays on-road
        if not has_left_lane:
            max_left = max(0.0, half_lane - veh_half_w - margin)

        if not has_right_lane:
            max_right = max(0.0, half_lane - veh_half_w - margin)

        # Apply global cap
        max_left = min(max_left, abs_cap)
        max_right = min(max_right, abs_cap)

        # Returned as (min, max)
        return (-max_right, max_left)

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
        min_off, max_off = self._safe_lateral_offset_bounds(self.target_lane_index)

        if act == "STEER_LEFT":
            self.lateral_offset += self.lateral_offset_step
        elif act == "STEER_RIGHT":
            self.lateral_offset -= self.lateral_offset_step

        self.lateral_offset = float(np.clip(self.lateral_offset, min_off, max_off))

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
        if self._lane_change_cooldown > 0:
            self._lane_change_cooldown -= 1

        if self.control_mode == "continuous":
            self.target_lane_index = self._current_edge_lane_index()
            self.follow_road()

            if action is not None:
                if isinstance(action, dict):
                    steering = float(action.get("steering", 0.0))
                    acceleration = float(action.get("acceleration", 0.0))
                else:
                    steering, acceleration = 0.0, 0.0

                steering = float(np.clip(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))
                self._last_low_level_action = {"steering": steering, "acceleration": acceleration}

            super().act(self._last_low_level_action)
            return

        # ---------------- discrete ----------------
        act = None
        if isinstance(action, str):
            act = action.upper()

        # Only process NEW discrete commands when explicitly provided
        if act is not None:
            if act in ("LANE_LEFT", "LANE_RIGHT"):
                self._apply_lane_change(act)
            elif act in ("FASTER", "SLOWER"):
                self._apply_speed_action(act)
            elif act in ("STEER_LEFT", "STEER_RIGHT") and self.lane_change_direction == 0:
                self._apply_steer_bias_action(act)

        # Continue internal lane-change controller
        if self.lane_change_direction != 0:
            step = self.lateral_offset_step
            if self.lane_change_direction == -1:
                self.lateral_offset += step
            else:
                self.lateral_offset -= step

            lane_width = self._lane_width(self.target_lane_index)
            threshold = lane_width / 2.0

            if self.lane_change_direction == -1 and self.lateral_offset > threshold:
                new_idx = self._adjacent_lane_index(-1)
                if new_idx:
                    self.target_lane_index = new_idx
                    self.lateral_offset -= lane_width
                else:
                    self.lane_change_direction = 0

            elif self.lane_change_direction == 1 and self.lateral_offset < -threshold:
                new_idx = self._adjacent_lane_index(1)
                if new_idx:
                    self.target_lane_index = new_idx
                    self.lateral_offset += lane_width
                else:
                    self.lane_change_direction = 0

            if abs(self.lateral_offset) < step:
                self.lateral_offset = 0.0
                self.lane_change_direction = 0

        if self.lane_change_direction == 0:
            self.follow_road()
            if self._lane_change_cooldown == 0:
                self.target_lane_index = self._current_edge_lane_index()

            min_off, max_off = self._safe_lateral_offset_bounds(self.target_lane_index)
            self.lateral_offset = float(np.clip(self.lateral_offset, min_off, max_off))

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
