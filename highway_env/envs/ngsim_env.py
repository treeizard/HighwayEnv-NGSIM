# Modified by: Yide Tao (yide.tao@monash.edu)
# Reference: Huang et al. (2021), Leurent (2018)
from __future__ import annotations
import os
import numpy as np
from typing import List, Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road
from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle
from highway_env.ngsim_utils.gen_road import create_ngsim_101_road

from highway_env.ngsim_utils.trajectory_gen import (
    build_trajectory_from_chunk,
    process_raw_trajectory,
    first_valid_index,
)

Observation = np.ndarray


class NGSimEnv(AbstractEnv):
    """
    Fast NGSIM Driving Env using 10-second chunk episodes.
    Automatically randomizes ego vehicle ID if not provided.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "scene": "us-101",

            # Observation / action
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},

            # Frequencies
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "max_episode_steps": 600,

            # Ego override (if None → sample random)
            "ego_vehicle_ID": None,

            # Episode selection
            "episode_root": "highway_env/data/processed_10s",
            "replay_period": None,        # None = random episode each reset

            # Spawn safety
            "max_surrounding": 80,

            # Debug
            "show_trajectories": False,
            "log_overlaps": True,
            "seed": None,
        })
        return config

    # ---------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return 1.0 / float(self.config["simulation_frequency"])

    # ---------------------------------------------------------------------
    def _reset(self):
        self.steps = 0

        seed = self.config.get("seed", None)
        if seed is not None and hasattr(self, "seed"):
            self.seed(seed)

        self._load_trajectory()   # chunks + random ego selection
        self._create_road()
        self._create_vehicles()

    # ---------------------------------------------------------------------
    def _load_trajectory(self):
        scene = self.config["scene"]
        root = os.path.join(self.config["episode_root"], scene)

        # ---- dynamically scan episodes here instead of __init__ ----
        episodes = sorted(
            d for d in os.listdir(root)
            if d.startswith("t") and os.path.isdir(os.path.join(root, d))
        )
        if not episodes:
            raise RuntimeError(f"No 10-second episodes found in {root}")

        # select episode
        if self.config["replay_period"] is None:
            ep_name = self.np_random.choice(episodes)
        else:
            ep_name = episodes[int(self.config["replay_period"]) % len(episodes)]

        self._episode_dir = os.path.join(root, ep_name)
        print(f"[NGSimEnv] Using episode: {self._episode_dir}")

        # --- load vehicles inside this episode ---
        from highway_env.data.ngsim import ngsim_data
        ng = ngsim_data(scene)
        ng.load(self._episode_dir)
        vehicles = list(ng.veh_dict.keys())

        # --- random ego if not provided ---
        ego_id = self.config.get("ego_vehicle_ID", None)
        if ego_id is None:
            ego_id = int(self.np_random.choice(vehicles))
            print(f"[NGSimEnv] Random ego vehicle: {ego_id}")
        self._ego_id = int(ego_id)

        # --- build trajectories ---
        self.trajectory_set = build_trajectory_from_chunk(
            scene,
            vehicle_ID=self._ego_id,
            episode_dir=self._episode_dir,
        )

    # ---------------------------------------------------------------------
    def _create_road(self):
        net = create_ngsim_101_road()
        self.net = net
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    # ---------------------------------------------------------------------
    def _create_vehicles(self):
        ego_rec = self.trajectory_set["ego"]
        ego_traj = process_raw_trajectory(ego_rec["trajectory"])
        ego_len = ego_rec["length"]
        ego_wid = ego_rec["width"]

        ego_start_i = first_valid_index(ego_traj)
        if ego_start_i is None:
            raise RuntimeError("Ego trajectory contains no valid motion data.")

        # --- ego first state ---
        x0, y0, ego_speed, lane0 = ego_traj[ego_start_i]
        ego_xy = np.array([x0, y0])

        main_edge = ("s1", "s2")
        lane_index = int(lane0)
        ego_lane = self.net.get_lane((*main_edge, lane_index))
        ego_s, ego_r = ego_lane.local_coordinates(ego_xy)

        # snap to lane
        ego_xy = np.asarray(ego_lane.position(ego_s, ego_r), float)

        ego = self.action_type.vehicle_class(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=ego_lane.heading_at(ego_s),
        )
        self.road.vehicles.append(ego)
        self.vehicle = ego

        # --- surrounding vehicles ---
        for vid, meta in self.trajectory_set.items():
            if vid == "ego": continue
            traj = process_raw_trajectory(meta["trajectory"])
            if len(traj) < 2: continue

            v = NGSIMVehicle.create(
                road=self.road,
                vehicle_ID=vid,
                position=traj[1][:2],
                v_length=meta["length"] / 3.281,
                v_width=meta["width"] / 3.281,
                ngsim_traj=traj,
                speed=traj[1][2],
                color=(200, 0, 150),
            )
            self.road.vehicles.append(v)

    # ---------------------------------------------------------------------
    def _rewards(self, action: int) -> dict[str, float]:
        speed = float(getattr(self.vehicle, "speed", 0.0))
        scaled_speed = utils.lmap(
            speed,
            self.config.get("reward_speed_range", [20.0, 30.0]),
            [0.0, 1.0],
        )

        lane_tuple = getattr(self.vehicle, "lane_index", None)
        right_lane_reward = 0.0
        if lane_tuple is not None:
            try:
                n_on_edge = len(
                    self.road.network.graph[lane_tuple[0]][lane_tuple[1]]
                )
                right_lane_reward = (
                    lane_tuple[2] / max(1, n_on_edge - 1)
                ) if n_on_edge > 1 else 1.0
            except Exception:
                pass

        lane_change_pen = int(action in [0, 2])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": float(right_lane_reward),
            "high_speed_reward": float(scaled_speed),
            "lane_change_reward": float(lane_change_pen),
        }

    # ---------------------------------------------------------------------
    def _reward(self, action: int) -> float:
        terms = self._rewards(action)
        raw = sum(self.config.get(name, 0.0) * val for name, val in terms.items())
        worst = self.config.get("collision_reward", -1.0)
        best = self.config.get("high_speed_reward", 1.0)
        return utils.lmap(raw, [worst, best], [0.0, 1.0])

    # ---------------------------------------------------------------------
    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed)

    def _is_truncated(self) -> bool:
        max_steps = self.config.get("max_episode_steps", None)
        return max_steps is not None and self.steps >= max_steps

    # ---------------------------------------------------------------------
    def step(self, action: Action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.steps += 1

        if not truncated and self._is_truncated():
            truncated = True

        if self.config["log_overlaps"]:
            overlaps = []
            vehs = [v for v in self.road.vehicles if hasattr(v, "aabb")]
            for i in range(len(vehs)):
                for j in range(i + 1, len(vehs)):
                    if vehs[i].overlaps_aabb(vehs[j]):
                        overlaps.append(
                            (getattr(vehs[i], "vehicle_ID", -1),
                             getattr(vehs[j], "vehicle_ID", -1))
                        )
            if overlaps:
                info = dict(info) if info else {}
                info["overlaps"] = overlaps

        return obs, reward, terminated, truncated, info
