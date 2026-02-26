from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

@dataclass
class IDMParams:
    """
    IDM/MOBIL parameters for a single vehicle.

    Notes:
    - `a_max` is the actuation clamp used by the simulator (what the vehicle can actually do).
    - `a_comf` is the desired comfortable acceleration used by IDM (model aggressiveness).
    - `b_comf` is the desired comfortable deceleration magnitude (>0). Internally we use -b_comf for convenience.
    - `s0_factor_*` define how jam distance scales with vehicle length.
    """
    # Actuation limits
    a_max: float = 10.0                      # [m/s^2] absolute acceleration magnitude clamp
    a_comf: float = 3.0                      # [m/s^2] comfortable acceleration (IDM)
    b_comf: float = 10.0                     # [m/s^2] comfortable deceleration magnitude (IDM), positive

    # IDM headway
    time_headway: float = 1.5                # [s] desired time gap (tau)
    delta: float = 4.0                       # [-] acceleration exponent

    # Jam-distance scaling (geometry-aware)
    s0_factor_ego: float = 0.75               # s0 includes ~1.5 * L_ego  (e.g., L/2 + L)
    s0_factor_front: float = 0.0             # optionally include leader length into jam distance

    # Additional safety buffer used for hard collision avoidance
    contact_buffer_factor: float = 0.5       # bumper buffer ~ 0.5*(L_ego+L_front)

    # MOBIL (lane change) parameters
    politeness: float = 0.0                  # [0,1]
    lane_change_min_acc_gain: float = 0.2    # [m/s^2]
    lane_change_max_braking_imposed: float = 2.0  # [m/s^2]
    lane_change_delay: float = 1.0           # [s]


class IDMVehicle(ControlledVehicle):
    """
    A geometry-aware IDM + MOBIL vehicle with a rear-end collision avoidance guard.

    Design goals:
    - Use per-instance vehicle geometry (length) consistently.
    - Retain classic IDM/MOBIL behavior in nominal conditions.
    - Add a hard safety term (required braking) to reduce rear-end impacts in dense / discrete-time simulation.

    Practical notes:
    - This class is intended to be a drop-in replacement for highway-env IDMVehicle.
    - If you randomize vehicle lengths at runtime, ensure `vehicle.length` is updated on each instance.
    """

    # Default parameters (can be overridden per instance)
    PARAMS = IDMParams()

    # Range for randomizing delta (optional)
    DELTA_RANGE = (3.5, 4.5)

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0.0,
        speed: float = 0.0,
        target_lane_index: LaneIndex | None = None,
        target_speed: float | None = None,
        route: Route | None = None,
        enable_lane_change: bool = True,
        timer: float | None = None,
        params: IDMParams | None = None,
    ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = float(timer) if timer is not None else (np.sum(self.position) * np.pi) % self.PARAMS.lane_change_delay
        self.params: IDMParams = params if params is not None else self.PARAMS

    def randomize_behavior(self) -> None:
        """Optionally randomize IDM aggressiveness by sampling delta."""
        self.params = IDMParams(**{**self.params.__dict__})  # shallow copy
        self.params.delta = float(self.road.np_random.uniform(self.DELTA_RANGE[0], self.DELTA_RANGE[1]))

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle, **kwargs) -> "IDMVehicle":
        """Create a new IDMVehicle from an existing controlled vehicle state."""
        return cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
            timer=getattr(vehicle, "timer", None),
            **kwargs,
        )

    # ----------------------------
    # Main control loop
    # ----------------------------
    def act(self, action: dict | str | None = None) -> None:
        """
        Compute and apply steering + acceleration.

        - Steering: lane following + MOBIL lane change.
        - Acceleration: IDM with geometry-aware desired gap and a hard collision-avoidance guard.
        """
        if self.crashed:
            return

        cmd: dict[str, float] = {}

        # Lateral: follow road and optionally change lane (MOBIL)
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        steering = float(self.steering_control(self.target_lane_index))
        cmd["steering"] = float(np.clip(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))

        # Longitudinal: IDM (consider current lane)
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        acc = self.acceleration(self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)

        # When changing lanes, be conservative: check both lanes and take the min acceleration
        if self.lane_index != self.target_lane_index:
            f2, r2 = self.road.neighbour_vehicles(self, self.target_lane_index)
            acc2 = self.acceleration(self, front_vehicle=f2, rear_vehicle=r2)
            acc = min(float(acc), float(acc2))

        # Actuation clamp
        acc = float(np.clip(acc, -self.params.a_max, self.params.a_max))
        cmd["acceleration"] = acc

        # Skip ControlledVehicle.act(); apply directly to the kinematic model.
        Vehicle.act(self, cmd)

    def step(self, dt: float) -> None:
        """Advance the simulation and update the lane-change timer."""
        self.timer += float(dt)
        super().step(dt)

    # ----------------------------
    # Longitudinal model (IDM + guard)
    # ----------------------------
    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle | None = None,
        rear_vehicle: Vehicle | None = None,
    ) -> float:
        """
        Compute longitudinal acceleration using IDM plus a hard rear-end collision avoidance guard.

        Rear vehicle is intentionally ignored in the longitudinal command here:
        - factoring the rear often reduces braking when you most need it, increasing rear-end impacts.
        - if you later want a "courtesy" term, it should be implemented as a bounded modification.

        Returns:
            acceleration [m/s^2]
        """
        if ego_vehicle is None or not isinstance(ego_vehicle, Vehicle):
            return 0.0

        p = self.params

        # Target speed
        v0 = float(getattr(ego_vehicle, "target_speed", 0.0))
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            v0 = float(np.clip(v0, 0.0, float(ego_vehicle.lane.speed_limit)))
        v0 = abs(utils.not_zero(v0))

        v = max(float(ego_vehicle.speed), 0.0)

        # Free-road term
        a_free = float(p.a_comf * (1.0 - (v / v0) ** float(p.delta)))

        if front_vehicle is None:
            return a_free

        # Distance to leader along lane coordinates
        d = float(ego_vehicle.lane_distance_to(front_vehicle))

        # Interaction term
        s_star = float(self.desired_gap(ego_vehicle, front_vehicle, projected=True))
        a_int = -float(p.a_comf * (s_star / utils.not_zero(d)) ** 2)
        a_idm = a_free + a_int

        # Hard safety guard (geometry-based)
        return float(min(a_idm, self._required_braking_to_avoid_contact(ego_vehicle, front_vehicle, d)))

    def _required_braking_to_avoid_contact(self, ego: Vehicle, front: Vehicle, lane_distance: float) -> float:
        """
        Compute a conservative (<=0) acceleration that prevents overlap with the leader, given current gap.

        Uses a simple constant-deceleration stopping condition on the relative speed:
            dv^2 = 2 * |a| * gap  =>  a_req = -dv^2 / (2*gap)

        Returns:
            a_req (negative or 0). If not closing, returns 0.
        """
        p = self.params

        L_ego = float(getattr(ego, "length", getattr(ego, "LENGTH", self.LENGTH)))
        L_front = float(getattr(front, "length", getattr(front, "LENGTH", self.LENGTH)))
        buffer = float(p.contact_buffer_factor * (L_ego + L_front))

        gap = float(lane_distance - buffer)
        if gap <= 0.1:
            return -float(p.a_max)  # already overlapping/too close: maximum braking

        v_ego = max(float(ego.speed), 0.0)
        v_front = max(float(front.speed), 0.0)
        dv = v_ego - v_front  # positive -> closing

        if dv <= 0.0:
            return 0.0

        a_req = -(dv * dv) / (2.0 * gap)

        # Do not request more than max braking capability
        return float(max(a_req, -float(p.a_max)))

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle | None = None, projected: bool = True) -> float:
        """
        Geometry-aware IDM desired gap s*.

        s* = s0 + v*tau + v*dv/(2*sqrt(a*b))   (with dv projected along lane direction)

        Here:
        - s0 scales with per-instance vehicle length (ego and optional leader).
        - dv term is rectified to be conservative (only when closing).
        """
        p = self.params

        L_ego = float(getattr(ego_vehicle, "length", getattr(ego_vehicle, "LENGTH", self.LENGTH)))
        s0 = float(p.s0_factor_ego * L_ego)

        if front_vehicle is not None and p.s0_factor_front != 0.0:
            L_front = float(getattr(front_vehicle, "length", getattr(front_vehicle, "LENGTH", self.LENGTH)))
            s0 += float(p.s0_factor_front * L_front)

        tau = float(p.time_headway)

        v = max(float(ego_vehicle.speed), 0.0)

        # comfortable deceleration is -b (store as magnitude)
        ab = float(p.a_comf * p.b_comf)  # > 0

        if front_vehicle is None:
            return float(s0 + v * tau)

        if projected:
            dv = float(np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction))
        else:
            dv = float(ego_vehicle.speed - front_vehicle.speed)

        # Conservative: only penalize when closing
        closing = max(0.0, v * dv)
        s_star = s0 + v * tau + closing / (2.0 * np.sqrt(ab))
        return float(s_star)

    # ----------------------------
    # MOBIL lane change policy
    # ----------------------------
    def change_lane_policy(self) -> None:
        """MOBIL lane-change decision at a fixed decision rate."""
        p = self.params

        # Lane change ongoing
        if self.lane_index != self.target_lane_index:
            # Abort conflicting merges into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if (
                        v is not self
                        and isinstance(v, ControlledVehicle)
                        and v.lane_index != self.target_lane_index
                        and v.target_lane_index == self.target_lane_index
                    ):
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v, projected=True)
                        if 0.0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # Decide at fixed frequency
        if not utils.do_every(p.lane_change_delay, self.timer):
            return
        self.timer = 0.0

        # Consider adjacent lanes
        for lane_index in self.road.network.side_lanes(self.lane_index):
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            if abs(self.speed) < 1.0:
                continue
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """MOBIL: decide whether a lane change is beneficial and safe."""
        p = self.params

        # Safety for new follower in target lane
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)

        new_following_a = self.acceleration(new_following, front_vehicle=new_preceding) if new_following else 0.0
        new_following_pred_a = self.acceleration(new_following, front_vehicle=self) if new_following else 0.0

        if new_following_pred_a < -float(p.lane_change_max_braking_imposed):
            return False

        # Route constraint (optional): if route requests a direction, do not go opposite
        old_preceding, old_following = self.road.neighbour_vehicles(self)

        self_pred_a = self.acceleration(self, front_vehicle=new_preceding)

        if self.route and self.route[0][2] is not None:
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            if self_pred_a < -float(p.lane_change_max_braking_imposed):
                return False
        else:
            self_a = self.acceleration(self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(old_following, front_vehicle=self) if old_following else 0.0
            old_following_pred_a = self.acceleration(old_following, front_vehicle=old_preceding) if old_following else 0.0

            jerk = (
                self_pred_a
                - self_a
                + float(p.politeness)
                * ((new_following_pred_a - new_following_a) + (old_following_pred_a - old_following_a))
            )
            if jerk < float(p.lane_change_min_acc_gain):
                return False

        return True

class LinearVehicle(IDMVehicle):
    """A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters."""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [
        ControlledVehicle.KP_HEADING,
        ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL,
    ]

    ACCELERATION_RANGE = np.array(
        [
            0.5 * np.array(ACCELERATION_PARAMETERS),
            1.5 * np.array(ACCELERATION_PARAMETERS),
        ]
    )
    STEERING_RANGE = np.array(
        [
            np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
            np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5]),
        ]
    )

    TIME_WANTED = 2.5

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        data: dict = None,
    ):
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed,
            route,
            enable_lane_change,
            timer,
        )
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: dict | str = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua * (
            self.ACCELERATION_RANGE[1] - self.ACCELERATION_RANGE[0]
        )
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub * (
            self.STEERING_RANGE[1] - self.STEERING_RANGE[0]
        )

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with a Linear Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - reach the speed of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
        - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return float(
            np.dot(
                self.ACCELERATION_PARAMETERS,
                self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle),
            )
        )

    def acceleration_features(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = (
                getattr(ego_vehicle, "target_speed", ego_vehicle.speed)
                - ego_vehicle.speed
            )
            d_safe = (
                self.DISTANCE_WANTED
                + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            )
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Linear controller with respect to parameters.

        Overrides the non-linear controller ControlledVehicle.steering_control()

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        return float(
            np.dot(
                np.array(self.STEERING_PARAMETERS),
                self.steering_features(target_lane_index),
            )
        )

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        A collection of features used to follow a lane

        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array(
            [
                utils.wrap_to_pi(lane_future_heading - self.heading)
                * self.LENGTH
                / utils.not_zero(self.speed),
                -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed) ** 2),
            ]
        )
        return features

    def longitudinal_structure(self):
        # Nominal dynamics: integrate speed
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        # Target speed dynamics
        phi0 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        # Front speed control
        phi1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0]])
        # Front position control
        phi2 = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [-1, 1, -self.TIME_WANTED, 0], [0, 0, 0, 0]]
        )
        # Disable speed control
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # Disable front position control
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([[0, 1], [0, 0]])
        phi0 = np.array([[0, 0], [0, -1]])
        phi1 = np.array([[0, 0], [-1, 0]])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """Store features and outputs for parameter regression."""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        0.5,
    ]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        2.0,
    ]
