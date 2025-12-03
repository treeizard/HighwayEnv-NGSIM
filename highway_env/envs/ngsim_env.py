# Modified by: Yide Tao (yide.tao@monash.edu)
# Reference: Huang et al. (2021), Leurent (2018)
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Any

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road
from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle
from highway_env.ngsim_utils.ego_vehicle import ControlledVehicle
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
            "max_episode_steps": 300,

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

        # --- load expert actions ---
        self._load_expert_actions()
    
    def _load_expert_actions(self):
        """
        Load continuous expert actions for the chosen ego vehicle
        from processed_cont_actions.csv in the selected episode dir.

        Exposes:
            self._expert_actions_sim : np.ndarray [T, 2] as [steering, accel]
        """
        csv_path = os.path.join(self._episode_dir, "processed_cont_actions.csv")
        if not os.path.exists(csv_path):
            print(f"[NGSimEnv] WARNING: No processed_cont_actions.csv in {self._episode_dir}")
            self._expert_actions_sim = None
            return

        df = pd.read_csv(csv_path)

        # Keep only ego rows
        ego_id = self._ego_id
        df = df[df["veh_ID"] == ego_id].sort_values("time")
        if df.empty:
            print(f"[NGSimEnv] WARNING: No expert actions for ego {ego_id} in {csv_path}")
            self._expert_actions_sim = None
            return

        # Shape [T, 2]: [steering, accel] per raw / simulation step
        # column names from traj_cont_action: 'accel', 'steering'
        steering = df["steering"].to_numpy(dtype=float)
        accel = df["accel"].to_numpy(dtype=float)
        self._expert_actions_sim = np.stack([steering, accel], axis=-1)

        # Also store the times in case you want exact time alignment later
        self._expert_times = df["time"].to_numpy(dtype=float)

        print(
            f"[NGSimEnv] Loaded expert actions for ego {ego_id}: "
            f"T={len(self._expert_actions_sim)}"
        )

        # ---------------------------------------------------------------------
    def expert_action_at(self, policy_step: int | None = None):
        """
        Return expert continuous action [steering, accel] for a given policy step.

        Steps are counted from reset; assumes self._expert_actions_sim is at
        simulation frequency (simulation_frequency Hz).
        """
        if getattr(self, "_expert_actions_sim", None) is None:
            raise RuntimeError("Expert actions not loaded for this episode.")

        if policy_step is None:
            policy_step = self.steps  # current policy step (after last env.step)

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = int(sim_freq // pol_freq)

        sim_idx = policy_step * sim_per_policy

        # Clamp to last expert step, so we don't crash index at episode tail
        sim_idx = int(np.clip(sim_idx, 0, len(self._expert_actions_sim) - 1))

        steering, accel = self._expert_actions_sim[sim_idx]
        return np.array([steering, accel], dtype=float)

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

        # ------------- NEW: trajectory-based horizon -----------------
        # number of simulation steps from start of episode to end of traj
        n_sim_steps = len(ego_traj) - int(ego_start_i)

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = int(sim_freq // pol_freq)  # here: 15 // 5 = 3

        # ceil so we don't cut one frame early
        self._max_traj_policy_steps = int(
            np.ceil(n_sim_steps / float(sim_per_policy))
        )

        print(
            f"[NGSimEnv] Traj sim steps: {n_sim_steps}, "
            f"sim_per_policy: {sim_per_policy}, "
            f"max_traj_policy_steps: {self._max_traj_policy_steps}"
        )
        # -------------------------------------------------------------

        # --- ego first state ---
        x0, y0, ego_speed, lane0 = ego_traj[ego_start_i]
        ego_xy = np.array([x0, y0], dtype=float)

        # ----------------- choose the correct edge based on x -----------------
        # Must match the geometry in create_ngsim_101_road()
        length = 2150 / 3.281
        ends = [0.0,
                560 / 3.281,
                (698 + 578 + 150) / 3.281,
                length]

        x_m = float(x0)  # assume ego_traj already in meters
        if x_m < ends[1]:
            main_edge = ("s1", "s2")   # first 5-lane section
        elif x_m < ends[2]:
            main_edge = ("s2", "s3")   # 6-lane section (lane 5 lives here)
        else:
            main_edge = ("s3", "s4")   # last 5-lane section

        lane_index = int(lane0)

        # Optional: clamp lane_index so it doesn't blow up if out of range
        lanes_on_edge = self.net.graph[main_edge[0]][main_edge[1]]
        n_lanes = len(lanes_on_edge)
        if lane_index < 0 or lane_index >= n_lanes:
            print(
                f"[NGSimEnv] WARNING: lane_index {lane_index} out of range for "
                f"edge {main_edge} (n_lanes={n_lanes}); clamping."
            )
            lane_index = int(np.clip(lane_index, 0, n_lanes - 1))

        print("lane ids:", lane_index, ", position:", x0, ",", y0,
              ", edge:", main_edge)

        ego_lane = self.net.get_lane((*main_edge, lane_index))
        ego_s, ego_r = ego_lane.local_coordinates(ego_xy)

        # snap to lane centerline
        ego_xy = np.asarray(ego_lane.position(ego_s, ego_r), float)

        ego = ControlledVehicle(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=ego_lane.heading_at(ego_s),
            control_mode="continuous"
        )

        self.road.vehicles.append(ego)
        self.vehicle = ego

        # --- surrounding vehicles (unchanged) ---
        for vid, meta in self.trajectory_set.items():
            if vid == "ego":
                continue
            traj = process_raw_trajectory(meta["trajectory"])
            if len(traj) < 2:
                continue

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
    def _rewards(self, action: Any) -> dict[str, float]:
        # ----- Speed term -----
        speed = float(getattr(self.vehicle, "speed", 0.0))
        scaled_speed = utils.lmap(
            speed,
            self.config.get("reward_speed_range", [20.0, 30.0]),
            [0.0, 1.0],
        )

        # ----- Right-lane term -----
        lane_tuple = getattr(self.vehicle, "lane_index", None)
        right_lane_reward = 0.0
        if lane_tuple is not None:
            try:
                n_on_edge = len(self.road.network.graph[lane_tuple[0]][lane_tuple[1]])
                # Normalize lane id so rightmost lane gets 1.0
                right_lane_reward = (
                    lane_tuple[2] / max(1, n_on_edge - 1)
                ) if n_on_edge > 1 else 1.0
            except Exception:
                pass

        # ----- Lane-change / steering penalty -----
        lane_change_pen = 0.0

        # Case 1: DISCRETE action (int index)
        if isinstance(action, (int, np.integer)):
            # Indices that correspond to lane changes in your DiscreteMetaAction
            lane_change_indices = self.config.get("lane_change_action_indices", [0, 2])
            lane_change_pen = float(action in lane_change_indices)

        # Case 2: CONTINUOUS vector (e.g. Box action → [steering, accel])
        elif isinstance(action, (np.ndarray, list, tuple)):
            # Assume first element encodes steering (normalized or physical)
            steering_val = float(action[0]) if len(action) > 0 else 0.0
            steer_thresh = self.config.get("lane_change_steer_thresh", 0.3)
            lane_change_pen = float(abs(steering_val) > steer_thresh)

        # Case 3: CONTINUOUS dict (e.g. replay / ContinuousAction → {"steering": ..., "acceleration": ...})
        elif isinstance(action, dict):
            steering_val = float(action.get("steering", 0.0))
            steer_thresh = self.config.get("lane_change_steer_thresh", 0.3)
            lane_change_pen = float(abs(steering_val) > steer_thresh)

        # Fallback: anything else → no lane-change penalty

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": float(right_lane_reward),
            "high_speed_reward": float(scaled_speed),
            "lane_change_reward": float(lane_change_pen),
        }


    def _reward(self, action: Any) -> float:
        terms = self._rewards(action)
        raw = sum(self.config.get(name, 0.0) * val for name, val in terms.items())

        worst = self.config.get("collision_reward", -1.0)
        best = self.config.get("high_speed_reward", 1.0)

        # Map raw reward linearly into [0, 1] for scaling
        return utils.lmap(raw, [worst, best], [0.0, 1.0])

    # ---------------------------------------------------------------------
    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed)

    def _is_truncated(self) -> bool:
        # Config-based cap
        max_steps_cfg = self.config.get("max_episode_steps", None)
        # Trajectory-based cap (might not exist if something went wrong)
        max_steps_traj = getattr(self, "_max_traj_policy_steps", None)

        # Take the minimum of the defined caps
        candidates = [v for v in (max_steps_cfg, max_steps_traj) if v is not None]
        if not candidates:
            return False  # no truncation limit at all

        hard_cap = min(candidates)
        return self.steps >= hard_cap

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
    
    
