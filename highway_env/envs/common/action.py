from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
from gymnasium import spaces

from highway_env import utils
from highway_env.utils import Vector
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle


if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType:
    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        """The action space."""
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable:
        """
        The class of a vehicle able to execute the action.

        Must return a subclass of :py:class:`highway_env.vehicle.kinematics.Vehicle`.
        """
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """
        Execute the action on the ego-vehicle.

        Most of the action mechanics are actually implemented in vehicle.act(action), where
        vehicle is an instance of the specified :py:class:`highway_env.envs.common.action.ActionType.vehicle_class`.
        Must some pre-processing can be applied to the action based on the ActionType configurations.

        :param action: the action to execute
        """
        raise NotImplementedError

    def get_available_actions(self):
        """
        For discrete action space, return the list of available actions.
        """
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        """The vehicle acted upon.

        If not set, the first controlled vehicle is used by default."""
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class ContinuousAction(ActionType):
    """
    An continuous action space for throttle and/or steering angle.

    If both throttle and steering are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    ACCELERATION_RANGE = (-5, 5.0)
    """Acceleration range: [-x, x], in m/s²."""

    STEERING_RANGE = (-np.pi / 4, np.pi / 4)
    """Steering angle range: [-x, x], in rad."""

    def __init__(
        self,
        env: AbstractEnv,
        acceleration_range: tuple[float, float] | None = None,
        steering_range: tuple[float, float] | None = None,
        speed_range: tuple[float, float] | None = None,
        longitudinal: bool = True,
        lateral: bool = True,
        dynamical: bool = False,
        clip: bool = True,
        **kwargs,
    ) -> None:
        """
        Create a continuous action space.

        :param env: the environment
        :param acceleration_range: the range of acceleration values [m/s²]
        :param steering_range: the range of steering values [rad]
        :param speed_range: the range of reachable speeds [m/s]
        :param longitudinal: enable throttle control
        :param lateral: enable steering control
        :param dynamical: whether to simulate dynamics (i.e. friction) rather than kinematics
        :param clip: clip action to the defined range
        """
        super().__init__(env)
        self.acceleration_range = (
            acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        )
        self.steering_range = steering_range if steering_range else self.STEERING_RANGE
        self.speed_range = speed_range
        self.lateral = lateral
        self.longitudinal = longitudinal
        if not self.lateral and not self.longitudinal:
            raise ValueError(
                "Either longitudinal and/or lateral control must be enabled"
            )
        self.dynamical = dynamical
        self.clip = clip
        self.size = 2 if self.lateral and self.longitudinal else 1
        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return Vehicle if not self.dynamical else BicycleVehicle

    def get_action(self, action: np.ndarray):
        if self.clip:
            action = np.clip(action, -1, 1)
        if self.speed_range:
            (
                self.controlled_vehicle.MIN_SPEED,
                self.controlled_vehicle.MAX_SPEED,
            ) = self.speed_range
        if self.longitudinal and self.lateral:
            return {
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
            }
        elif self.longitudinal:
            return {
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": 0,
            }
        elif self.lateral:
            return {
                "acceleration": 0,
                "steering": utils.lmap(action[0], [-1, 1], self.steering_range),
            }

    def act(self, action: np.ndarray) -> None:
        self.controlled_vehicle.act(self.get_action(action))
        self.last_action = action


class DiscreteAction(ContinuousAction):
    def __init__(
        self,
        env: AbstractEnv,
        acceleration_range: tuple[float, float] | None = None,
        steering_range: tuple[float, float] | None = None,
        longitudinal: bool = True,
        lateral: bool = True,
        dynamical: bool = False,
        clip: bool = True,
        actions_per_axis: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            env,
            acceleration_range=acceleration_range,
            steering_range=steering_range,
            longitudinal=longitudinal,
            lateral=lateral,
            dynamical=dynamical,
            clip=clip,
        )
        self.actions_per_axis = actions_per_axis

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(self.actions_per_axis**self.size)

    def act(self, action: int) -> None:
        cont_space = super().space()
        axes = np.linspace(cont_space.low, cont_space.high, self.actions_per_axis).T
        all_actions = list(itertools.product(*axes))
        super().act(all_actions[action])


class DiscreteMetaAction(ActionType):
    """
    An discrete action space of meta-actions: lane changes, and cruise control set-point.
    """

    ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
    """A mapping of action indexes to labels."""

    ACTIONS_LONGI = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    """A mapping of longitudinal action indexes to labels."""

    ACTIONS_LAT = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}
    """A mapping of lateral action indexes to labels."""

    def __init__(
        self,
        env: AbstractEnv,
        longitudinal: bool = True,
        lateral: bool = True,
        target_speeds: Vector | None = None,
        **kwargs,
    ) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        :param target_speeds: the list of speeds the vehicle is able to track
        """
        super().__init__(env)
        self.longitudinal = longitudinal
        self.lateral = lateral
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else MDPVehicle.DEFAULT_TARGET_SPEEDS
        )
        self.actions = (
            self.ACTIONS_ALL
            if longitudinal and lateral
            else (
                self.ACTIONS_LONGI
                if longitudinal
                else self.ACTIONS_LAT if lateral else None
            )
        )
        if self.actions is None:
            raise ValueError(
                "At least longitudinal or lateral actions must be included"
            )
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(MDPVehicle, target_speeds=self.target_speeds)

    def act(self, action: int | np.ndarray) -> None:
        self.controlled_vehicle.act(self.actions[int(action)])

    def get_available_actions(self) -> list[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        actions = [self.actions_indexes["IDLE"]]
        network = self.controlled_vehicle.road.network
        for l_index in network.side_lanes(self.controlled_vehicle.lane_index):
            if (
                l_index[2] < self.controlled_vehicle.lane_index[2]
                and network.get_lane(l_index).is_reachable_from(
                    self.controlled_vehicle.position
                )
                and self.lateral
            ):
                actions.append(self.actions_indexes["LANE_LEFT"])
            if (
                l_index[2] > self.controlled_vehicle.lane_index[2]
                and network.get_lane(l_index).is_reachable_from(
                    self.controlled_vehicle.position
                )
                and self.lateral
            ):
                actions.append(self.actions_indexes["LANE_RIGHT"])
        if (
            self.controlled_vehicle.speed_index
            < self.controlled_vehicle.target_speeds.size - 1
            and self.longitudinal
        ):
            actions.append(self.actions_indexes["FASTER"])
        if self.controlled_vehicle.speed_index > 0 and self.longitudinal:
            actions.append(self.actions_indexes["SLOWER"])
        return actions

class DiscreteSteerMetaAction(ActionType):
    """
    Discrete meta action space for EgoVehicle:
      - Speed ladder: FASTER / SLOWER / IDLE
      - Steering bias (within lane): STEER_LEFT / STEER_RIGHT
      - Lane change primitives: LANE_LEFT / LANE_RIGHT

    IMPORTANT: indices 0..4 remain unchanged for backwards compatibility.
    """

    # Stable indices (do not reorder once you start training)
    ACTIONS = {
        0: "SLOWER",
        1: "IDLE",
        2: "FASTER",
        3: "STEER_LEFT",
        4: "STEER_RIGHT",
        # New actions appended (safe for older checkpoints if you don't load with shape mismatch)
        5: "LANE_LEFT",
        6: "LANE_RIGHT",
    }

    def __init__(
        self,
        env: AbstractEnv,
        target_speeds: Vector | None = None,
        # Optional: allow disabling subsets without changing indices
        enable_longitudinal: bool = True,
        enable_steer: bool = True,
        enable_lane_change: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(env)

        self.enable_longitudinal = bool(enable_longitudinal)
        self.enable_steer = bool(enable_steer)
        self.enable_lane_change = bool(enable_lane_change)

        self.target_speeds = (
            np.array(target_speeds, dtype=float)
            if target_speeds is not None
            else np.array(
                getattr(EgoVehicle, "DEFAULT_TARGET_SPEEDS", MDPVehicle.DEFAULT_TARGET_SPEEDS),
                dtype=float,
            )
        )

        self.actions = dict(self.ACTIONS)
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        """
        Use EgoVehicle so that:
          - speed_index/target_speeds are meaningful
          - STEER_LEFT/STEER_RIGHT and LANE_LEFT/LANE_RIGHT are implemented
        """
        return functools.partial(
            EgoVehicle,
            control_mode="discrete",
            target_speeds=self.target_speeds,
        )

    def act(self, action: int | np.ndarray) -> None:
        self.controlled_vehicle.act(self.actions[int(action)])

    def _lane_change_reachable(self, direction: int) -> bool:
        """
        direction: -1 for left lane (decrease lane id), +1 for right lane (increase lane id)
        Returns True if an adjacent lane exists and is reachable from current position.
        """
        v = self.controlled_vehicle
        if v is None or v.road is None or v.road.network is None:
            return False

        # Prefer target_lane_index if present, otherwise lane_index
        lane_index = getattr(v, "target_lane_index", None) or getattr(v, "lane_index", None)
        if lane_index is None:
            return False

        try:
            _from, _to, _id = lane_index
            lanes_list = v.road.network.graph[_from][_to]
            n_lanes = len(lanes_list)
            new_id = int(np.clip(_id + direction, 0, n_lanes - 1))
            if new_id == _id:
                return False
            new_index = (_from, _to, new_id)
            return v.road.network.get_lane(new_index).is_reachable_from(v.position)
        except Exception:
            # If your NGSIM road network doesn't follow this structure perfectly,
            # fail closed: don't expose lane change actions.
            return False

    def get_available_actions(self) -> list[int]:
        """
        Available actions:
          - IDLE always.
          - FASTER/SLOWER bounded by speed ladder.
          - STEER_LEFT/STEER_RIGHT always available if enabled.
          - LANE_LEFT/LANE_RIGHT only if enabled and adjacent lane is reachable.
        """
        avail = [self.actions_indexes["IDLE"]]

        # Longitudinal bounds
        if self.enable_longitudinal:
            speed_index = int(getattr(self.controlled_vehicle, "speed_index", 0))
            target_speeds = getattr(self.controlled_vehicle, "target_speeds", self.target_speeds)

            if speed_index < int(target_speeds.size - 1):
                avail.append(self.actions_indexes["FASTER"])
            if speed_index > 0:
                avail.append(self.actions_indexes["SLOWER"])

        # Steer within-lane setpoint actions
        if self.enable_steer:
            avail.append(self.actions_indexes["STEER_LEFT"])
            avail.append(self.actions_indexes["STEER_RIGHT"])

        # Lane changes
        if self.enable_lane_change:
            # Convention: LANE_LEFT = -1, LANE_RIGHT = +1
            if self._lane_change_reachable(direction=-1):
                avail.append(self.actions_indexes["LANE_LEFT"])
            if self._lane_change_reachable(direction=+1):
                avail.append(self.actions_indexes["LANE_RIGHT"])

        # Deduplicate while preserving order
        out = []
        seen = set()
        for a in avail:
            if a not in seen:
                out.append(a)
                seen.add(a)
        return out

class MultiAgentAction(ActionType):
    def __init__(self, env: AbstractEnv, action_config: dict, **kwargs) -> None:
        super().__init__(env)
        self.action_config = action_config
        self.agents_action_types = []
        for vehicle in self.env.controlled_vehicles:
            action_type = action_factory(self.env, self.action_config)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [action_type.space() for action_type in self.agents_action_types]
        )

    @property
    def vehicle_class(self) -> Callable:
        return action_factory(self.env, self.action_config).vehicle_class

    def act(self, action: Action) -> None:
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)

    def get_available_actions(self):
        return itertools.product(
            *[
                action_type.get_available_actions()
                for action_type in self.agents_action_types
            ]
        )


def action_factory(env: AbstractEnv, config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    if config["type"] == "DiscreteAction":
        return DiscreteAction(env, **config)
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    elif config["type"] == "DiscreteSteerMetaAction":  
        return DiscreteSteerMetaAction(env, **config)
    elif config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)
    else:
        raise ValueError("Unknown action type")
