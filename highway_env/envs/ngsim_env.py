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

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road
from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle
from highway_env.ngsim_utils.ego_vehicle import ControlledVehicle
from highway_env.ngsim_utils.gen_road import create_ngsim_101_road, clamp_location_ngsim
from highway_env.ngsim_utils.trajectory_gen import (
    process_raw_trajectory,
    first_valid_index,
)

Observation = np.ndarray


class NGSimEnv(AbstractEnv):
    """
    Fast NGSIM Driving Env using 10-second chunk episodes.
    Automatically randomizes ego vehicle ID if not provided.
    Uses in-memory caching so that episodes, trajectories, and expert actions
    are only loaded once per (episode, ego_id) pair.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "scene": "us-101",

            # Observation / action
            "observation": {"type": "Kinematics"},
            "action": {"type": "ContinuousAction"},

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
            "seed": None,

            # Reward config (weights)
            "collision_reward": -1.0,
            "high_speed_reward": 1.0,
            "reward_speed_range": [20.0, 30.0],

            # Lane change heuristics
            "lane_change_action_indices": [0, 2],
            "lane_change_steer_thresh": 0.3,
        })
        return config

    # -------------------------------------------------------------------------
    # INIT / DT
    # -------------------------------------------------------------------------
    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        # Cache of available episodes for the current scene
        self._episodes: list[str] = []
        self._episode_root: str | None = None
        self._valid_ids_by_episode = {} 
        self._episode_cache: Dict[str, Dict[str, Any]] = {}
        self.ego_id = None

        # These are set each reset
        self._ego_id: int | None = None

        # Expert Actions for debugging
        self._expert_actions_sim: np.ndarray | None = None
        self._expert_times: np.ndarray | None = None
        self._max_traj_policy_steps: int | None = None

        # Load all Trajectories
        print(config)
        self._prebuilt_dir = os.path.join(config["episode_root"], config["scene"], "prebuilt")
        print(self._prebuilt_dir)
        veh_ids_path= os.path.join(self._prebuilt_dir, f"veh_ids_train.npy")
        traj_path = os.path.join(self._prebuilt_dir, f"trajectory_train.npy")
        self._valid_ids_by_episode = np.load(veh_ids_path, allow_pickle=True).item()
        self._traj_all_by_episode  = np.load(traj_path, allow_pickle=True).item()
        self._episodes = sorted(self._traj_all_by_episode.keys())
        '''
        self._traj_all_by_episode = build_all_trajectories_for_scene(
            scene= config["scene"],
            episodes_root=self._episode_root,
        )
        self._episodes = sorted(self._traj_all_by_episode.keys())

        # Valid Ego IDs
        for ep_name, veh_dict in self._traj_all_by_episode.items():
            valid_ids = []
            for vid, meta in veh_dict.items():
                traj = meta["trajectory"]
                if traj.shape[0] < 2:
                    continue
                nonzero = np.any(traj[:, :3] != 0.0, axis=1)
                if nonzero.sum() >= 2:
                    valid_ids.append(vid)
            self._valid_ids_by_episode[ep_name] = valid_ids

        # Let AbstractEnv do its normal setup (config, spaces, reset, etc.)
        '''
        super().__init__(config=config, render_mode=render_mode)

    @property
    def dt(self) -> float:
        return 1.0 / float(self.config["simulation_frequency"])

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------
    def _reset(self):
        self.steps = 0

        seed = self.config.get("seed", None)
        if seed is not None and hasattr(self, "seed"):
            self.seed(seed)

        self._load_trajectory()
        self._create_road()
        self._create_vehicles()
  

    # -------------------------------------------------------------------------
    # LOAD TRAJECTORY
    # -------------------------------------------------------------------------
    def _load_trajectory(self):
        """
        Choose an episode, select a valid ego_id, and attach:
          - self.trajectory_set
          - self._expert_actions_sim
          - self._expert_times
          -  self._ego_id

        Uses per-episode cache so disk I/O is minimized.
        """

        # 1. Randomly select episode
        self.episode_name = self.np_random.choice(self._episodes)
        
        # 2. Randomly select ego id
        self.ego_id = int(self.np_random.choice(self._valid_ids_by_episode[self.episode_name]))
        print("episode name:", self.episode_name, ", ego id:", self.ego_id )
        
        # 3. Define Trajectory set
        traj_all = self._traj_all_by_episode[self.episode_name]
        traj_set = {"ego": traj_all[self.ego_id]}
        for vid, meta in traj_all.items():
            if vid == self.ego_id:
                continue
            traj_set[vid] = meta
        self.trajectory_set = traj_set
            
    # -------------------------------------------------------------------------
    # EXPERT ACTION QUERY
    # -------------------------------------------------------------------------
    @staticmethod
    def _load_expert_actions_for(
        episode_dir: str,
        ego_id: int,
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Load continuous expert actions for a specific (episode_dir, ego_id).

        Returns:
            expert_actions_sim : np.ndarray [T, 2] or None
            expert_times       : np.ndarray [T]     or None
        """
        csv_path = os.path.join(episode_dir, "processed_cont_actions.csv")
        if not os.path.exists(csv_path):
            print(f"[NGSimEnv] WARNING: No processed_cont_actions.csv in {episode_dir}")
            return None, None

        df = pd.read_csv(csv_path)

        # Keep only ego rows
        df = df[df["veh_ID"] == ego_id].sort_values("time")
        if df.empty:
            print(f"[NGSimEnv] WARNING: No expert actions for ego {ego_id} in {csv_path}")
            return None, None

        # Shape [T, 2]: [steering, accel]
        steering = df["steering"].to_numpy(dtype=float)
        accel = df["accel"].to_numpy(dtype=float)
        expert_actions_sim = np.stack([steering, accel], axis=-1)
        expert_times = df["time"].to_numpy(dtype=float)
        """
        print(
            f"[NGSimEnv] Loaded expert actions for ego {ego_id}: "
            f"T={len(expert_actions_sim)}"
        )
        """
        return expert_actions_sim, expert_times

    def expert_action_at(self, policy_step: int | None = None) -> np.ndarray:
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

    # -------------------------------------------------------------------------
    # ROAD + VEHICLES
    # -------------------------------------------------------------------------
    def _create_road(self):
        net = create_ngsim_101_road()
        self.net = net
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self):
        # --- Ego trajectory record & convert to meters ---
        ego_rec = self.trajectory_set["ego"]
        ego_traj_full = process_raw_trajectory(ego_rec["trajectory"])  # [T, 4]: x, y, speed, lane
        ego_len = ego_rec["length"] / 3.281
        ego_wid = ego_rec["width"] / 3.281

        # Find the first valid index for the ego vehicle
        ego_start_i = first_valid_index(ego_rec["trajectory"])
        print("first valide index:",ego_start_i)
        
        if ego_start_i is None:
            raise RuntimeError("Ego trajectory contains no valid motion data.")

        # --------- TRUNCATE ALL TRAJECTORIES AT EGO SPAWN ---------
        # From now on, 'time step 0' in the simulation corresponds to ego_start_i
        ego_traj = ego_traj_full[ego_start_i:]
        if len(ego_traj) < 2:
            raise RuntimeError(
                f"Ego trajectory too short after truncation (len={len(ego_traj)})."
            )
        # Optionally store this for debugging or future use
        self._ego_start_index = int(ego_start_i)
        # -----------------------------------------------------------

        # --- Horizon in policy steps based on truncated trajectory length ---
        n_sim_steps = len(ego_traj)  # already truncated, so full length is usable

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = int(sim_freq // pol_freq)  # e.g., 15 // 5 = 3

        self._max_traj_policy_steps = int(
            np.ceil(n_sim_steps / float(sim_per_policy))
        )

        # --- Ego initial state (meters) from truncated traj[0] ---
        x0, y0, ego_speed, lane0 = ego_traj[0]
        ego_xy = np.array([x0, y0], dtype=float)

        # ----------------- choose the correct edge based on x -----------------
        ego_lane = clamp_location_ngsim(
            x_pos=x0,
            lane0=lane0,
            net=self.net,
            warning=False,
        )
        ego_s, ego_r = ego_lane.local_coordinates(ego_xy)

        # Snap to lane centerline
        ego_xy = np.asarray(ego_lane.position(ego_s, ego_r), float)

        # Create the ego Vehicle
        ego = ControlledVehicle(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=ego_lane.heading_at(ego_s),
            control_mode="continuous",
        )
        ego.set_ego_dimension(width=ego_wid, length=ego_len)

        self.road.vehicles.append(ego)
        self.vehicle = ego

        # --- Surrounding vehicles (also truncated to ego spawn time) ---
        for vid, meta in self.trajectory_set.items():
            if vid == "ego":
                continue

            traj_full = process_raw_trajectory(meta["trajectory"])
            if len(traj_full) <= self._ego_start_index:
                # This vehicle never appears after ego spawn
                continue

            # Truncate this vehicle's trajectory so time 0 aligns with ego spawn
            traj = traj_full[self._ego_start_index:]

            # Optional: drop leading all-zero rows (ghost segment)
            nonzero = np.any(traj[:, :3] != 0.0, axis=1)
            if not np.any(nonzero):
                # Vehicle is effectively absent after ego spawn
                continue
            first_idx = int(np.argmax(nonzero))
            traj = traj[first_idx:]

            if len(traj) < 2:
                continue

            v = NGSIMVehicle.create(
                road=self.road,
                vehicle_ID=vid,
                position=traj[0][:2],
                v_length=meta["length"] / 3.281,
                v_width=meta["width"] / 3.281,
                ngsim_traj=traj,
                speed=traj[0][2],
                color=(200, 0, 150),
            )
            self.road.vehicles.append(v)


    # -------------------------------------------------------------------------
    # REWARDS
    # -------------------------------------------------------------------------
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
            lane_change_indices = self.config.get("lane_change_action_indices", [0, 2])
            lane_change_pen = float(action in lane_change_indices)

        # Case 2: CONTINUOUS vector (Box → [steering, accel])
        elif isinstance(action, (np.ndarray, list, tuple)):
            steering_val = float(action[0]) if len(action) > 0 else 0.0
            steer_thresh = self.config.get("lane_change_steer_thresh", 0.3)
            lane_change_pen = float(abs(steering_val) > steer_thresh)

        # Case 3: CONTINUOUS dict ({"steering": ..., "acceleration": ...})
        elif isinstance(action, dict):
            steering_val = float(action.get("steering", 0.0))
            steer_thresh = self.config.get("lane_change_steer_thresh", 0.3)
            lane_change_pen = float(abs(steering_val) > steer_thresh)

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

        # Map raw reward linearly into [0, 1]
        return utils.lmap(raw, [worst, best], [0.0, 1.0])

    # -------------------------------------------------------------------------
    # TERMINATION / TRUNCATION
    # -------------------------------------------------------------------------
    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed)

    def _is_truncated(self) -> bool:
        # Config-based cap
        max_steps_cfg = self.config.get("max_episode_steps", None)
        # Trajectory-based cap (might not exist if something went wrong)
        max_steps_traj = getattr(self, "_max_traj_policy_steps", None)

        candidates = [v for v in (max_steps_cfg, max_steps_traj) if v is not None]
        if not candidates:
            return False  # no truncation limit at all

        hard_cap = min(candidates)
        return self.steps >= hard_cap

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action: Action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.steps += 1

        if not truncated and self._is_truncated():
            truncated = True
      
        return obs, reward, terminated, truncated, info
