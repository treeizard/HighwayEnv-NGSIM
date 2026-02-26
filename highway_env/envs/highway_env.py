from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    Straight multi-lane highway environment.

    Reward shaping (patched):
    - Collision penalty.
    - High forward-speed reward.
    - Lane-keeping reward (staying in the same lane across steps).
    - Angular-acceleration penalty (yaw-rate derivative) for smoother control.
    - Right-lane reward disabled (replaced by lane keeping).
    - Reward normalization disabled while iterating on custom reward terms.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {"type": "DiscreteMetaAction"},
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,

                # --- Reward weights ---
                "collision_reward": -1.0,
                "high_speed_reward": 0.4,

                # Patched: replace right-lane reward with lane-keeping reward
                "right_lane_reward": 0.0,
                "lane_keeping_reward": 0.1,

                # Patched: penalize angular acceleration (negative term)
                "angular_acc_penalty": 0.02,  # weight applied to -(ang_acc^2)

                "lane_change_reward": 0.0,
                "reward_speed_range": [20, 30],

                # Patched: disable normalization while redesigning reward
                "normalize_reward": False,

                "offroad_terminal": False,

                # --- Spawn feasibility params ---
                "spawn_speed_min": 25.0,
                "spawn_speed_max": 30.0,
                "spawn_max_decel": 10.0,       # [m/s^2] assumed available braking for spawn filter
                "spawn_safety_buffer": 3.0,    # [m] extra buffer for spawn feasibility
                "spawn_default_length": 5.0,   # [m] fallback length if instance length unavailable

                # Optional spawn knobs (already used in your code)
                "spawn_window": 120.0,
                "spawn_min_lane_gap": 12.0,
                "spawn_min_global_gap": 6.0,
                "spawn_max_tries": 200,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

        # For angular acceleration penalty (ego only)
        self._prev_heading: float | None = None
        self._yaw_rate: float = 0.0
        self._prev_yaw_rate: float = 0.0

        # Track lane for lane-keeping reward
        self._prev_lane_index = None

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """
        Spawn one controlled ego vehicle per controlled slot, then populate the scene with
        uncontrolled traffic sampled around each ego (both behind and ahead), with feasibility guards.
        """
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"],
            num_bins=self.config["controlled_vehicles"],
        )

        window = float(self.config.get("spawn_window", 120.0))
        min_lane_gap = float(self.config.get("spawn_min_lane_gap", 12.0))
        min_global_gap = float(self.config.get("spawn_min_global_gap", 6.0))
        max_tries = int(self.config.get("spawn_max_tries", 200))

        self.controlled_vehicles = []

        for n_others in other_per_controlled:
            ego_seed = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            ego = self.action_type.vehicle_class(
                self.road,
                ego_seed.position,
                ego_seed.heading,
                ego_seed.speed,
            )
            self.controlled_vehicles.append(ego)
            self.road.vehicles.append(ego)

            ego_x = float(ego.position[0])

            for _ in range(int(n_others)):
                ok = self._spawn_other_near_ego(
                    other_vehicles_type=other_vehicles_type,
                    ego_x=ego_x,
                    window=window,
                    min_lane_gap=min_lane_gap,
                    min_global_gap=min_global_gap,
                    tries=max_tries,
                )
                if not ok:
                    break

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0.0) * r for name, r in rewards.items())
        reward *= rewards["on_road_reward"]
        return float(reward)

    def _rewards(self, action: Action) -> dict[str, float]:
        # Forward speed (avoid rewarding lateral speed components)
        forward_speed = float(self.vehicle.speed * np.cos(self.vehicle.heading))
        scaled_speed = float(
            np.clip(utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1]), 0, 1)
        )

        # Angular acceleration penalty: -(d(yaw_rate)/dt)^2
        dt = 1.0 / float(self.config.get("simulation_frequency", 15))
        yaw_rate = float(getattr(self, "_yaw_rate", 0.0))
        prev_yaw_rate = float(getattr(self, "_prev_yaw_rate", 0.0))
        ang_acc = (yaw_rate - prev_yaw_rate) / dt
        ang_acc_penalty = -(ang_acc * ang_acc)
        print("ang_penal:", ang_acc_penalty)
        # Lane-keeping reward: 1 if same lane as previous step
        if self._prev_lane_index is None:
            same_lane = 1.0
        else:
            same_lane = float(self.vehicle.lane_index == self._prev_lane_index)

        return {
            "collision_reward": float(self.vehicle.crashed),
            "lane_keeping_reward": same_lane,
            "high_speed_reward": scaled_speed,
            "angular_acc_penalty": ang_acc_penalty,
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return bool(
            self.vehicle.crashed
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return bool(self.time >= self.config["duration"])

    # -----------------------------
    # Spawn helpers
    # -----------------------------
    def _lane_y(self, lane_index) -> float:
        """Return the lateral centerline y of a lane (straight road)."""
        lane = self.road.network.get_lane(lane_index)
        return float(lane.position(0.0, 0.0)[1])

    def _min_dist_to_vehicles(self, pos: np.ndarray) -> float:
        if not self.road.vehicles:
            return float("inf")
        d2 = [float(np.sum((v.position - pos) ** 2)) for v in self.road.vehicles]
        return float(np.sqrt(np.min(d2)))

    def _min_longitudinal_gap_in_same_lane(self, lane_index, x: float) -> float:
        """Minimum |dx| to vehicles currently in the same lane (approx by y proximity)."""
        y = self._lane_y(lane_index)
        gaps = []
        for v in self.road.vehicles:
            if abs(float(v.position[1]) - y) < 0.5:
                gaps.append(abs(float(v.position[0]) - x))
        return float(min(gaps)) if gaps else float("inf")

    def _spawn_other_near_ego(
        self,
        other_vehicles_type,
        ego_x: float,
        window: float = 120.0,
        min_lane_gap: float = 10.0,
        min_global_gap: float = 5.0,
        tries: int = 200,
    ) -> bool:
        """
        Spawn one other vehicle with x in [ego_x-window, ego_x+window] (behind or ahead),
        while rejecting dynamically infeasible rear-end setups w.r.t. the nearest front vehicle
        in the same lane.
        """
        lanes = int(self.config["lanes_count"])

        v_min = float(self.config.get("spawn_speed_min", 15.0))
        v_max = float(self.config.get("spawn_speed_max", 28.0))
        max_decel = float(self.config.get("spawn_max_decel", 6.0))  # positive magnitude
        safety_buffer = float(self.config.get("spawn_safety_buffer", 2.0))
        default_L = float(self.config.get("spawn_default_length", 5.0))

        for _ in range(int(tries)):
            lane_id = int(self.np_random.integers(0, lanes))
            lane_index = ("0", "1", lane_id)

            x = float(ego_x + self.np_random.uniform(-window, window))
            y = self._lane_y(lane_index)
            pos = np.array([x, y], dtype=float)

            # Geometric guards
            if self._min_longitudinal_gap_in_same_lane(lane_index, x) < float(min_lane_gap):
                continue
            if self._min_dist_to_vehicles(pos) < float(min_global_gap):
                continue

            # Sample speed
            v_new = float(self.np_random.uniform(v_min, v_max))

            # Find nearest front vehicle in the same lane (y tolerance)
            front = None
            best_dx = float("inf")
            for veh in self.road.vehicles:
                if abs(float(veh.position[1]) - y) >= 0.5:
                    continue
                dx = float(veh.position[0]) - x
                if 0.0 < dx < best_dx:
                    best_dx = dx
                    front = veh

            # Dynamic feasibility guard
            if front is not None:
                v_front = max(float(getattr(front, "speed", 0.0)), 0.0)

                # Approx bumper-to-bumper gap
                L_new = float(getattr(other_vehicles_type, "LENGTH", default_L))
                L_front = float(getattr(front, "length", getattr(front, "LENGTH", default_L)))
                gap = best_dx - 0.5 * (L_new + L_front) - safety_buffer
                if gap <= 0.0:
                    continue

                dv = v_new - v_front
                if dv > 0.0:
                    if (dv * dv) > 2.0 * max_decel * gap:
                        continue

            # Accept spawn
            v = other_vehicles_type(self.road, pos, heading=0.0, speed=0.0)
            v.speed = v_new
            if hasattr(v, "randomize_behavior"):
                v.randomize_behavior()
            self.road.vehicles.append(v)
            return True

        return False

    # -----------------------------
    # Step hook to maintain state for reward terms
    # -----------------------------
    def step(self, action: Action):
        # Pre-step bookkeeping for lane keeping + yaw terms
        if self.vehicle is not None:
            self._prev_lane_index = self.vehicle.lane_index
            if self._prev_heading is None:
                self._prev_heading = float(self.vehicle.heading)

        obs, reward, terminated, truncated, info = super().step(action)

        # Post-step compute yaw-rate
        dt = 1.0 / float(self.config.get("simulation_frequency", 15))
        if self.vehicle is not None and self._prev_heading is not None:
            heading = float(self.vehicle.heading)
            dpsi = float(utils.wrap_to_pi(heading - self._prev_heading))
            yaw_rate = dpsi / dt

            self._prev_yaw_rate = float(getattr(self, "_yaw_rate", yaw_rate))
            self._yaw_rate = float(yaw_rate)
            self._prev_heading = heading

        return obs, reward, terminated, truncated, info


class HighwayEnvFast(HighwayEnv):
    """
    Faster variant:
    - lower simulation frequency
    - fewer vehicles/lanes, shorter duration
    - disable collision checks for uncontrolled vehicles
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
