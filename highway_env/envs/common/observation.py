from __future__ import annotations

from collections import OrderedDict
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.road.lane import AbstractLane
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


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


class GrayscaleObservation(ObservationType):
    """
    An observation class that collects directly what the simulator renders.

    Also stacks the collected frames as in the nature DQN.
    The observation shape is C x W x H.

    Specific keys are expected in the configuration dictionary passed.
    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84)
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion,
        }
    """

    def __init__(
        self,
        env: AbstractEnv,
        observation_shape: tuple[int, int],
        stack_size: int,
        weights: list[float],
        scaling: float | None = None,
        centering_position: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(env)
        self.observation_shape = observation_shape
        self.shape = (stack_size,) + self.observation_shape
        self.weights = weights
        self.obs = np.zeros(self.shape, dtype=np.uint8)

        # The viewer configuration can be different between this observation and env.render() (typically smaller)
        viewer_config = env.config.copy()
        viewer_config.update(
            {
                "offscreen_rendering": True,
                "screen_width": self.observation_shape[0],
                "screen_height": self.observation_shape[1],
                "scaling": scaling or viewer_config["scaling"],
                "centering_position": centering_position
                or viewer_config["centering_position"],
            }
        )
        from highway_env.envs.common.graphics import EnvViewer

        self.viewer = EnvViewer(env, config=viewer_config)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)

    def observe(self) -> np.ndarray:
        new_obs = self._render_to_grayscale()
        self.obs = np.roll(self.obs, -1, axis=0)
        self.obs[-1, :, :] = new_obs
        return self.obs

    def _render_to_grayscale(self) -> np.ndarray:
        self.viewer.observer_vehicle = self.observer_vehicle
        self.viewer.display()
        raw_rgb = self.viewer.get_image()  # H x W x C
        raw_rgb = np.moveaxis(raw_rgb, 0, 1)
        return np.dot(raw_rgb[..., :3], self.weights).clip(0, 255).astype(np.uint8)


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: AbstractEnv, horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(
                shape=self.observe().shape, low=0, high=1, dtype=np.float32
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(
                (3, 3, int(self.horizon * self.env.config["policy_frequency"]))
            )
        grid = compute_ttc_grid(
            self.env,
            vehicle=self.observer_vehicle,
            time_quantization=1 / self.env.config["policy_frequency"],
            horizon=self.horizon,
        )
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0 : lf + 1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0 : vf + 1, :, :]
        return clamped_grid.astype(np.float32)


class KinematicObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""

    FEATURES: list[str] = ["presence", "x", "y", "vx", "vy"]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] = None,
        vehicles_count: int = 5,
        features_range: dict[str, list[float]] = None,
        absolute: bool = False,
        order: str = "sorted",
        normalize: bool = True,
        clip: bool = True,
        see_behind: bool = False,
        observe_intentions: bool = False,
        include_obstacles: bool = True,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(
                self.observer_vehicle.lane_index
            )
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [
                    -AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                    AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                ],
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])
        # Add nearby traffic
        close_vehicles = self.env.road.close_objects_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
            sort=self.order == "sorted",
            vehicles_only=not self.include_obstacles,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            vehicles_df = pd.DataFrame.from_records(
                [
                    v.to_dict(origin, observe_intentions=self.observe_intentions)
                    for v in close_vehicles[-self.vehicles_count + 1 :]
                ]
            )
            df = pd.concat([df, vehicles_df], ignore_index=True)

        df = df[self.features]

        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)


class OccupancyGridObservation(ObservationType):
    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: list[str] = ["presence", "vx", "vy", "on_road"]
    GRID_SIZE: list[list[float]] = [[-5.5 * 5, 5.5 * 5], [-5.5 * 5, 5.5 * 5]]
    GRID_STEP: list[int] = [5, 5]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] | None = None,
        grid_size: tuple[tuple[float, float], tuple[float, float]] | None = None,
        grid_step: tuple[float, float] | None = None,
        features_range: dict[str, list[float]] = None,
        absolute: bool = False,
        align_to_vehicle_axes: bool = False,
        clip: bool = True,
        as_image: bool = False,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param grid_size: real world size of the grid [[min_x, max_x], [min_y, max_y]]
        :param grid_step: steps between two cells of the grid [step_x, step_y]
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: use absolute or relative coordinates
        :param align_to_vehicle_axes: if True, the grid axes are aligned with vehicle axes. Else, they are aligned
               with world axes.
        :param clip: clip the observation in [-1, 1]
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = (
            np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        )
        self.grid_step = (
            np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        )
        grid_shape = np.asarray(
            np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step),
            dtype=np.uint8,
        )
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image

    def space(self) -> spaces.Space:
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(
                shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32
            )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Initialize empty data
            self.grid.fill(np.nan)

            # Get nearby traffic data
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles]
            )
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # A vehicle feature
                    for _, vehicle in df[::-1].iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # Recover unnormalized coordinates for cell index
                        if "x" in self.features_range:
                            x = utils.lmap(
                                x,
                                [-1, 1],
                                [
                                    self.features_range["x"][0],
                                    self.features_range["x"][1],
                                ],
                            )
                        if "y" in self.features_range:
                            y = utils.lmap(
                                y,
                                [-1, 1],
                                [
                                    self.features_range["y"][0],
                                    self.features_range["y"][1],
                                ],
                            )
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        if (
                            0 <= cell[0] < self.grid.shape[-2]
                            and 0 <= cell[1] < self.grid.shape[-1]
                        ):
                            self.grid[layer, cell[0], cell[1]] = vehicle[feature]
                elif feature == "on_road":
                    self.fill_road_layer_by_lanes(layer)

            obs = self.grid

            if self.clip:
                obs = np.clip(obs, -1, 1)

            if self.as_image:
                obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            obs = np.nan_to_num(obs).astype(self.space().dtype)

            return obs

    def pos_to_index(self, position: Vector, relative: bool = False) -> tuple[int, int]:
        """
        Convert a world position to a grid cell index

        If align_to_vehicle_axes the cells are in the vehicle's frame, otherwise in the world frame.

        :param position: a world position
        :param relative: whether the position is already relative to the observer's position
        :return: the pair (i,j) of the cell index
        """
        if not relative:
            position -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(
                self.observer_vehicle.heading
            )
            position = np.array([[c, s], [-s, c]]) @ position
        return (
            int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),
            int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1])),
        )

    def index_to_pos(self, index: tuple[int, int]) -> np.ndarray:
        position = np.array(
            [
                (index[0] + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
                (index[1] + 0.5) * self.grid_step[1] + self.grid_size[1, 0],
            ]
        )

        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(
                -self.observer_vehicle.heading
            )
            position = np.array([[c, s], [-s, c]]) @ position

        position += self.observer_vehicle.position
        return position

    def fill_road_layer_by_lanes(
        self, layer_index: int, lane_perception_distance: float = 100
    ) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        Here, we iterate over lanes and regularly placed waypoints on these lanes to fill the corresponding cells.
        This approach is faster if the grid is large and the road network is small.

        :param layer_index: index of the layer in the grid
        :param lane_perception_distance: lanes are rendered +/- this distance from vehicle location
        """
        lane_waypoints_spacing = np.amin(self.grid_step)
        road = self.env.road

        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for lane in road.network.graph[_from][_to]:
                    origin, _ = lane.local_coordinates(self.observer_vehicle.position)
                    waypoints = np.arange(
                        origin - lane_perception_distance,
                        origin + lane_perception_distance,
                        lane_waypoints_spacing,
                    ).clip(0, lane.length)
                    for waypoint in waypoints:
                        cell = self.pos_to_index(lane.position(waypoint, 0))
                        if (
                            0 <= cell[0] < self.grid.shape[-2]
                            and 0 <= cell[1] < self.grid.shape[-1]
                        ):
                            self.grid[layer_index, cell[0], cell[1]] = 1

    def fill_road_layer_by_cell(self, layer_index) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        In this implementation, we iterate the grid cells and check whether the corresponding world position
        at the center of the cell is onroad/offroad. This approach is faster if the grid is small and the road network large.
        """
        road = self.env.road
        for i, j in product(range(self.grid.shape[-2]), range(self.grid.shape[-1])):
            for _from in road.network.graph.keys():
                for _to in road.network.graph[_from].keys():
                    for lane in road.network.graph[_from][_to]:
                        if lane.on_lane(self.index_to_pos((i, j))):
                            self.grid[layer_index, i, j] = 1


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: AbstractEnv, scales: list[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                dict(
                    desired_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["desired_goal"].shape,
                        dtype=np.float64,
                    ),
                    achieved_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["achieved_goal"].shape,
                        dtype=np.float64,
                    ),
                    observation=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["observation"].shape,
                        dtype=np.float64,
                    ),
                )
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> dict[str, np.ndarray]:
        if not self.observer_vehicle:
            return OrderedDict(
                [
                    ("observation", np.zeros((len(self.features),))),
                    ("achieved_goal", np.zeros((len(self.features),))),
                    ("desired_goal", np.zeros((len(self.features),))),
                ]
            )

        obs = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        )
        goal = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.goal.to_dict()])[
                self.features
            ]
        )
        obs = OrderedDict(
            [
                ("observation", obs / self.scales),
                ("achieved_goal", obs / self.scales),
                ("desired_goal", goal / self.scales),
            ]
        )
        return obs


class AttributesObservation(ObservationType):
    def __init__(self, env: AbstractEnv, attributes: list[str], **kwargs: dict) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                {
                    attribute: spaces.Box(
                        -np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float64
                    )
                    for attribute in self.attributes
                }
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> dict[str, np.ndarray]:
        return OrderedDict(
            [(attribute, getattr(self.env, attribute)) for attribute in self.attributes]
        )


class MultiAgentObservation(ObservationType):
    def __init__(self, env: AbstractEnv, observation_config: dict, **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [obs_type.space() for obs_type in self.agents_observation_types]
        )

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


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
        obstacle_entries = self.lidar_observation.collect_obstacle_entries()
        observations = []
        for vehicle in self.env.controlled_vehicles:
            self.lidar_observation.observer_vehicle = vehicle
            self.camera_observation.observer_vehicle = vehicle
            ego_state = self._build_ego_state(vehicle)
            observations.append(
                (
                    self.lidar_observation.observe(obstacle_entries=obstacle_entries),
                    self.camera_observation.observe(),
                    ego_state,
                )
            )
        return tuple(observations)


class TupleObservation(ObservationType):
    def __init__(
        self, env: AbstractEnv, observation_configs: list[dict], **kwargs
    ) -> None:
        super().__init__(env)
        self.observation_types = [
            observation_factory(self.env, obs_config)
            for obs_config in observation_configs
        ]

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


class ExitObservation(KinematicObservation):
    """Specific to exit_env, observe the distance to the next exit lane as part of a KinematicObservation."""

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        ego_dict = self.observer_vehicle.to_dict()
        exit_lane = self.env.road.network.get_lane(("1", "2", -1))
        ego_dict["x"] = exit_lane.local_coordinates(self.observer_vehicle.position)[0]
        df = pd.DataFrame.from_records([ego_dict])[self.features]

        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = pd.concat(
                [
                    df,
                    pd.DataFrame.from_records(
                        [
                            v.to_dict(
                                origin, observe_intentions=self.observe_intentions
                            )
                            for v in close_vehicles[-self.vehicles_count + 1 :]
                        ]
                    )[self.features],
                ],
                ignore_index=True,
            )
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)


class LidarObservation(ObservationType):
    """
    LiDAR observation with road-edge clamping.

    Key behavior:
      - Each LiDAR beam is truncated at the first point where the ray leaves the road.
      - The returned distance is the minimum of:
            (distance to nearest obstacle intersection) and (distance to road edge).
      - If no obstacle is hit, the beam returns the road edge distance (dense boundary cue).

    Output:
      grid[:, 0] = distance (meters)
      grid[:, 1] = relative speed along beam direction (m/s)
    """

    DISTANCE = 0
    SPEED = 1

    def __init__(
        self,
        env,
        cells: int = 16,
        maximum_range: float = 60.0,
        normalize: bool = True,
        edge_as_return: bool = True,
        coarse_step: float | None = None,
        refine_iters: int = 8,
        use_topology_fast_path: bool = True,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.cells = int(cells)
        self.maximum_range = float(maximum_range)
        self.normalize = bool(normalize)
        self.use_topology_fast_path = bool(use_topology_fast_path)

        # If True: empty beams return road-edge distance; else they return maximum_range.
        self.edge_as_return = bool(edge_as_return)

        # Ray-march parameters
        self.coarse_step = float(coarse_step) if coarse_step is not None else max(0.5, self.maximum_range / 60.0)
        self.refine_iters = int(refine_iters)

        self.angle = 2 * np.pi / self.cells
        self.grid = np.ones((self.cells, 2), dtype=np.float32) * self.maximum_range
        self.origin = None
        self._directions = np.stack(
            [self.index_to_direction(i) for i in range(self.cells)],
            axis=0,
        )

        # Cache lanes for faster on-road checks (safe for typical static road networks)
        self._lanes_cache = self._collect_lanes()
        self._edge_lane_cache = self._collect_edge_lanes()
        self._edge_bounds_cache = self._collect_edge_bounds()

    # ----------------- Gym space -----------------

    def space(self) -> spaces.Space:
        high = 1.0 if self.normalize else self.maximum_range
        return spaces.Box(
            shape=(self.cells, 2),
            low=-high,
            high=high,
            dtype=np.float32,
        )

    # ----------------- Main API -----------------

    def observe(
        self,
        obstacle_entries: list[tuple[object, np.ndarray, np.ndarray, float, np.ndarray]] | None = None,
    ) -> np.ndarray:
        obs = self.trace(
            self.observer_vehicle.position,
            self.observer_vehicle.velocity,
            obstacle_entries=obstacle_entries,
        ).copy()
        if self.normalize:
            obs /= self.maximum_range
        return obs

    # ----------------- Core LiDAR logic -----------------

    def collect_obstacle_entries(self) -> list[tuple[object, np.ndarray, np.ndarray, float, np.ndarray]]:
        road = getattr(self.env, "road", None)
        if road is None:
            return []

        entries: list[tuple[object, np.ndarray, np.ndarray, float, np.ndarray]] = []
        for obstacle in list(road.vehicles) + list(road.objects):
            if not getattr(obstacle, "solid", True):
                continue
            if hasattr(obstacle, "appear") and not getattr(obstacle, "appear", True):
                continue
            if hasattr(obstacle, "visible") and not getattr(obstacle, "visible", True):
                continue

            length = float(getattr(obstacle, "LENGTH", 0.0))
            width = float(getattr(obstacle, "WIDTH", 0.0))
            if length == 0.0 and width == 0.0:
                continue

            position = getattr(obstacle, "position", None)
            if position is None or not np.all(np.isfinite(position)):
                continue
            obstacle_pos = np.array(position, dtype=float)

            obs_vel = getattr(obstacle, "velocity", np.zeros(2, dtype=float))
            if obs_vel is None or not np.all(np.isfinite(obs_vel)):
                obs_vel = np.zeros(2, dtype=float)
            else:
                obs_vel = np.array(obs_vel, dtype=float)

            heading = float(getattr(obstacle, "heading", 0.0))
            corners = np.asarray(
                utils.rect_corners(obstacle_pos, length, width, heading),
                dtype=float,
            )
            entries.append((obstacle, obstacle_pos, obs_vel, width, corners))
        return entries

    def trace(
        self,
        origin: np.ndarray,
        origin_velocity: np.ndarray,
        obstacle_entries: list[tuple[object, np.ndarray, np.ndarray, float, np.ndarray]] | None = None,
    ) -> np.ndarray:
        self.origin = np.array(origin, dtype=float).copy()

        # Ensure velocity is finite
        if origin_velocity is None or not np.all(np.isfinite(origin_velocity)):
            origin_velocity = np.zeros(2, dtype=float)
        else:
            origin_velocity = np.array(origin_velocity, dtype=float)

        # Refresh lane cache if road object changed (rare, but safe)
        if self._lanes_cache is None or getattr(self.env, "road", None) is None:
            self._lanes_cache = self._collect_lanes()

        if obstacle_entries is None:
            obstacle_entries = self.collect_obstacle_entries()

        # Precompute per-ray road edge distance
        edge_dists = np.empty((self.cells,), dtype=np.float32)
        for i in range(self.cells):
            d = self._directions[i]
            edge = self._distance_to_road_edge(
                origin=self.origin,
                direction=d,
                max_range=self.maximum_range,
                coarse_step=self.coarse_step,
                refine_iters=self.refine_iters,
            )
            edge_dists[i] = np.float32(np.clip(edge, 0.0, self.maximum_range))

        # Initialize grid distances
        self.grid.fill(0.0)
        if self.edge_as_return:
            self.grid[:, self.DISTANCE] = edge_dists
        else:
            self.grid[:, self.DISTANCE] = self.maximum_range
        self.grid[:, self.SPEED] = 0.0

        # Iterate over road vehicles + static objects
        for obstacle, obstacle_pos, obs_vel, width, corners in obstacle_entries:
            if obstacle is self.observer_vehicle:
                continue
            center_vec = obstacle_pos - self.origin
            center_distance = float(np.linalg.norm(center_vec))
            if (not np.isfinite(center_distance)) or (center_distance > self.maximum_range):
                continue

            # Approximate center ray bin
            center_angle = self.position_to_angle(obstacle_pos, self.origin)
            center_index = self.angle_to_index(center_angle)

            # Quick cull: if obstacle center beyond the road edge for its bin plus its half width, likely irrelevant
            if center_distance > float(edge_dists[center_index]) + 0.5 * width:
                # Still might intersect another bin, but this removes many far obstacles cheaply
                pass  # keep conservative; do not continue
            angles = [self.position_to_angle(corner, self.origin) for corner in corners]
            angles = [a for a in angles if np.isfinite(a)]
            if len(angles) == 0:
                continue

            min_angle, max_angle = min(angles), max(angles)

            # Handle wrap-around across -pi / +pi
            if min_angle < -np.pi / 2 < np.pi / 2 < max_angle:
                min_angle, max_angle = max_angle, min_angle + 2 * np.pi

            start = self.angle_to_index(min_angle)
            end = self.angle_to_index(max_angle)

            if start <= end:
                indexes = np.arange(start, end + 1)
            else:
                indexes = np.hstack([np.arange(start, self.cells), np.arange(0, end + 1)])

            # Exact ray-rectangle distance per LiDAR cell, with road-edge clamping
            for index in indexes:
                max_t = float(edge_dists[index])
                if max_t <= 0.0:
                    continue

                direction = self._directions[int(index)]

                # IMPORTANT: clamp ray segment to road edge
                ray = [self.origin, self.origin + max_t * direction]
                dist = utils.distance_to_rect(ray, corners)

                if not np.isfinite(dist):
                    continue

                dist = float(np.clip(dist, 0.0, max_t))

                if dist <= float(self.grid[int(index), self.DISTANCE]):
                    rel_vel = float((obs_vel - origin_velocity).dot(direction))
                    self.grid[int(index), :] = [dist, rel_vel]

        # If we are NOT returning edge as return, still ensure beams do not exceed edge
        if not self.edge_as_return:
            self.grid[:, self.DISTANCE] = np.minimum(self.grid[:, self.DISTANCE], edge_dists)

        return self.grid

    # ----------------- Road boundary helpers -----------------

    def _collect_lanes(self):
        """
        Cache lane objects for on-road tests.
        """
        road = getattr(self.env, "road", None)
        if road is None or getattr(road, "network", None) is None:
            return None

        lanes = []
        try:
            for _from, tos in road.network.graph.items():
                for _to, lane_list in tos.items():
                    for lane in lane_list:
                        lanes.append(lane)
        except Exception:
            return None

        return lanes if lanes else None

    def _collect_edge_lanes(self) -> dict[tuple[str, str], list[AbstractLane]] | None:
        road = getattr(self.env, "road", None)
        if road is None or getattr(road, "network", None) is None:
            return None

        edge_lanes: dict[tuple[str, str], list[AbstractLane]] = {}
        try:
            for src, tos in road.network.graph.items():
                for dst, lane_list in tos.items():
                    edge_lanes[(src, dst)] = list(lane_list)
        except Exception:
            return None
        return edge_lanes if edge_lanes else None

    def _lane_bounds(self, lane: AbstractLane) -> tuple[float, float, float, float] | None:
        try:
            length = float(getattr(lane, "length", 0.0))
            if not np.isfinite(length) or length <= 0.0:
                sample_s = np.array([0.0], dtype=float)
            else:
                num = max(3, min(9, int(np.ceil(length / 40.0)) + 1))
                sample_s = np.linspace(0.0, length, num=num, dtype=float)

            points = []
            for longitudinal in sample_s:
                center = lane.position(float(longitudinal), 0.0)
                width = float(lane.width_at(float(longitudinal)))
                if center is None or not np.all(np.isfinite(center)) or not np.isfinite(width):
                    continue
                center = np.asarray(center, dtype=float)
                points.append(center)

                lateral = 0.5 * width
                points.append(np.asarray(lane.position(float(longitudinal), lateral), dtype=float))
                points.append(np.asarray(lane.position(float(longitudinal), -lateral), dtype=float))

            if not points:
                return None

            stacked = np.vstack(points)
            return (
                float(np.min(stacked[:, 0])),
                float(np.max(stacked[:, 0])),
                float(np.min(stacked[:, 1])),
                float(np.max(stacked[:, 1])),
            )
        except Exception:
            return None

    def _collect_edge_bounds(
        self,
    ) -> dict[tuple[str, str], tuple[float, float, float, float]] | None:
        if self._edge_lane_cache is None:
            return None

        edge_bounds: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        for edge, lane_list in self._edge_lane_cache.items():
            lane_bounds = [self._lane_bounds(lane) for lane in lane_list]
            lane_bounds = [bounds for bounds in lane_bounds if bounds is not None]
            if not lane_bounds:
                continue

            edge_bounds[edge] = (
                min(bounds[0] for bounds in lane_bounds),
                max(bounds[1] for bounds in lane_bounds),
                min(bounds[2] for bounds in lane_bounds),
                max(bounds[3] for bounds in lane_bounds),
            )
        return edge_bounds if edge_bounds else None

    def _candidate_lanes_for_point(self, p: np.ndarray) -> list[AbstractLane] | None:
        if not self.use_topology_fast_path:
            return None
        if self._edge_lane_cache is None or self._edge_bounds_cache is None:
            self._edge_lane_cache = self._collect_edge_lanes()
            self._edge_bounds_cache = self._collect_edge_bounds()
            if self._edge_lane_cache is None or self._edge_bounds_cache is None:
                return None

        x = float(p[0])
        y = float(p[1])
        observer_edge = None
        lane_index = getattr(self.observer_vehicle, "lane_index", None)
        if lane_index is not None:
            observer_edge = tuple(lane_index[:2])

        candidates: list[AbstractLane] = []
        seen: set[int] = set()
        bounds_margin = max(2.0, self.coarse_step + 1.0)

        def add_edge(edge: tuple[str, str]) -> None:
            for lane in self._edge_lane_cache.get(edge, []):
                key = id(lane)
                if key not in seen:
                    seen.add(key)
                    candidates.append(lane)

        if observer_edge is not None:
            add_edge(observer_edge)

        for edge, bounds in self._edge_bounds_cache.items():
            min_x, max_x, min_y, max_y = bounds
            if (
                min_x - bounds_margin <= x <= max_x + bounds_margin
                and min_y - bounds_margin <= y <= max_y + bounds_margin
            ):
                add_edge(edge)

        return candidates if candidates else None

    def _on_road_at(self, p: np.ndarray) -> bool:
        """
        Conservative point-on-road test: on-road if it lies on any lane surface.
        """
        if self._lanes_cache is None:
            self._lanes_cache = self._collect_lanes()
            if self._lanes_cache is None:
                # Fallback: assume on-road to avoid over-clamping in pathological cases
                return True

        candidate_lanes = self._candidate_lanes_for_point(p)
        lanes = candidate_lanes if candidate_lanes is not None else self._lanes_cache

        for lane in lanes:
            try:
                if lane.on_lane(p):
                    return True
            except Exception:
                continue
        if candidate_lanes is not None:
            for lane in self._lanes_cache:
                if lane in lanes:
                    continue
                try:
                    if lane.on_lane(p):
                        return True
                except Exception:
                    continue
        return False

    def _distance_to_road_edge(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_range: float,
        coarse_step: float,
        refine_iters: int,
    ) -> float:
        """
        Distance along ray until it first leaves the road surface.
        Returns in [0, max_range]. If road never ends within max_range, returns max_range.
        """
        d = direction / (np.linalg.norm(direction) + 1e-12)

        # If already off-road, edge is at 0
        if not self._on_road_at(origin):
            return 0.0

        t = 0.0
        last_on = 0.0

        # Coarse march
        while t < max_range:
            t = min(t + coarse_step, max_range)
            p = origin + t * d
            if not self._on_road_at(p):
                # Refine boundary between last_on (on) and t (off)
                lo, hi = last_on, t
                for _ in range(refine_iters):
                    mid = 0.5 * (lo + hi)
                    pm = origin + mid * d
                    if self._on_road_at(pm):
                        lo = mid
                    else:
                        hi = mid
                return lo
            last_on = t

        return max_range

    # ----------------- Helper functions -----------------

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        dx = float(position[0] - origin[0])
        dy = float(position[1] - origin[1])

        if not np.isfinite(dx) or not np.isfinite(dy):
            return 0.0

        ang = float(np.arctan2(dy, dx) + self.angle / 2.0)
        if not np.isfinite(ang):
            return 0.0
        return ang

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        if not np.isfinite(angle):
            return 0
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: int) -> np.ndarray:
        """
        Convert a LiDAR cell index into a unit direction vector in world coordinates.
        The beam is centered in the cell: angle = (index + 0.5) * cell_angle.
        """
        theta = (int(index) + 0.5) * self.angle
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)


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
        obs = self.trace_topology(vehicle.position, heading).copy()
        if self.normalize:
            obs[:, 1:] /= self.maximum_range
        return obs

    def trace_topology(self, origin: np.ndarray, heading: float) -> np.ndarray:
        self.origin = np.array(origin, dtype=float).copy()
        self.grid.fill(0.0)

        if self._lanes_cache is None or getattr(self.env, "road", None) is None:
            self._lanes_cache = self._collect_lanes()
        if self._boundary_points_cache is None:
            self._boundary_points_cache = self._collect_boundary_points()

        if self._boundary_points_cache is None or len(self._boundary_points_cache) == 0:
            return self.grid

        relative_points = self._boundary_points_cache - self.origin
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
            return self.grid

        ego_points = ego_points[valid]
        distances = distances[valid]
        angles = angles[valid]

        bin_indices = np.digitize(angles, self._bin_edges[1:-1], right=False)
        for index in range(self.cells):
            matches = np.where(bin_indices == index)[0]
            if matches.size == 0:
                continue
            nearest = matches[np.argmin(distances[matches])]
            self.grid[index, :] = [1.0, ego_points[nearest, 0], ego_points[nearest, 1]]

        return self.grid

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



def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif (
        config["type"] == "MultiAgentObservation"
        and isinstance(config.get("observation_config"), dict)
        and config["observation_config"].get("type") == "LidarCameraObservations"
    ):
        shared_cfg = config["observation_config"]
        return SharedMultiAgentLidarCameraObservations(
            env,
            lidar=shared_cfg.get("lidar"),
            camera=shared_cfg.get("camera"),
        )
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "SharedMultiAgentLidarCameraObservations":
        return SharedMultiAgentLidarCameraObservations(env, **config)
    elif config["type"] == "TupleObservation":
        return TupleObservation(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "LaneCameraObservation":
        return LaneCameraObservation(env, **config)
    elif config["type"] == "LidarCameraObservation":
        return LaneCameraObservation(env, **config)
    elif config["type"] == "LidarCameraObservations":
        return LidarCameraObservations(env, **config)
    elif config["type"] == "ExitObservation":
        return ExitObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
