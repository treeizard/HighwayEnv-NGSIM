# Modified by: Yide Tao (yide.tao@monash.edu)
# Reference: Huang et al. (2021), Leurent (2018)
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
            "log_overlaps": True,
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
        # ---- our custom state / caches (must exist before first reset) ----
        # Cache of available episodes for the current scene
        self._episodes: list[str] = []
        self._episode_root: str | None = None

        # Cache per episode:
        # ep_name -> {
        #   "valid_ids": list[int],
        #   "traj_by_ego": { ego_id: trajectory_set },
        #   "expert_by_ego": { ego_id: (expert_actions_sim, expert_times) }
        # }
        self._episode_cache: Dict[str, Dict[str, Any]] = {}

        # These are set each reset
        self._episode_dir: str | None = None
        self._ego_id: int | None = None

        self._expert_actions_sim: np.ndarray | None = None
        self._expert_times: np.ndarray | None = None
        self._max_traj_policy_steps: int | None = None

        # Load and Cache Trajectories
        self._load_trajectory() 
        # Let AbstractEnv do its normal setup (config, spaces, reset, etc.)
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
        
        self._create_road()
        self._create_vehicles()

    # -------------------------------------------------------------------------
    # EPISODE DISCOVERY / CACHE
    # -------------------------------------------------------------------------
    def _ensure_episode_list(self):
        """Scan the scene directory once and cache episode names."""
        if self._episodes:
            return

        scene = self.config["scene"]
        root = os.path.join(self.config["episode_root"], scene)

        episodes = sorted(
            d for d in os.listdir(root)
            if d.startswith("t") and os.path.isdir(os.path.join(root, d))
        )
        if not episodes:
            raise RuntimeError(f"No 10-second episodes found in {root}")

        self._episodes = episodes
        self._episode_root = root
        print(f"[NGSimEnv] Found {len(self._episodes)} episodes in {root}")

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

        print(
            f"[NGSimEnv] Loaded expert actions for ego {ego_id}: "
            f"T={len(expert_actions_sim)}"
        )
        return expert_actions_sim, expert_times

    # -------------------------------------------------------------------------
    # LOAD TRAJECTORY + EXPERT ACTIONS (CACHED)
    # -------------------------------------------------------------------------
    def _load_trajectory(self):
        """
        Choose an episode, select a valid ego_id, and attach:
          - self.trajectory_set
          - self._expert_actions_sim
          - self._expert_times
          - self._episode_dir, self._ego_id

        Uses per-episode cache so disk I/O is minimized.
        """
        self._ensure_episode_list()
        scene = self.config["scene"]

        # --- choose episode name ---
        if self.config["replay_period"] is None:
            ep_name = self.np_random.choice(self._episodes)
        else:
            ep_name = self._episodes[int(self.config["replay_period"]) % len(self._episodes)]

        self._episode_dir = os.path.join(self._episode_root, ep_name)
        print(f"[NGSimEnv] Using episode: {self._episode_dir}")

        # Get / create cache entry for this episode
        ep_cache = self._episode_cache.get(ep_name)
        if ep_cache is None:
            ep_cache = {
                "valid_ids": None,     # list[int]
                "traj_by_ego": {},     # ego_id -> trajectory_set
                "expert_by_ego": {},   # ego_id -> (expert_actions_sim, expert_times)
            }
            self._episode_cache[ep_name] = ep_cache

        # ------------------------------------------------------------------
        # 1) Compute valid ego IDs for this episode ONCE
        # ------------------------------------------------------------------
        if ep_cache["valid_ids"] is None:
            from highway_env.data.ngsim import ngsim_data, GLB_TIME_THRES

            ng = ngsim_data(scene)
            ng.load(self._episode_dir)

            def has_nonempty_trajectory(veh) -> bool:
                vr_list = veh.vr_list
                if len(vr_list) < 2:
                    return False
                # vr_list already time-sorted by ng.load()
                for prev, cur in zip(vr_list, vr_list[1:]):
                    if cur.unixtime - prev.unixtime <= GLB_TIME_THRES:
                        return True
                return False

            valid_vehicle_ids: list[int] = [
                vid for vid, veh in ng.veh_dict.items() if has_nonempty_trajectory(veh)
            ]

            if not valid_vehicle_ids:
                raise RuntimeError(
                    f"[NGSimEnv] No vehicles with non-empty trajectory in episode {self._episode_dir}"
                )

            ep_cache["valid_ids"] = valid_vehicle_ids
            print(
                f"[NGSimEnv] Cached {len(valid_vehicle_ids)} valid vehicles for episode {ep_name}"
            )
        else:
            valid_vehicle_ids = ep_cache["valid_ids"]

        # ------------------------------------------------------------------
        # 2) Choose ego_id (fixed or random among valid IDs)
        # ------------------------------------------------------------------
        ego_id_cfg = self.config.get("ego_vehicle_ID", None)
        if ego_id_cfg is not None:
            if ego_id_cfg not in valid_vehicle_ids:
                print(
                    f"[NGSimEnv] WARNING: requested ego_vehicle_ID={ego_id_cfg} "
                    f"has no valid trajectory in this episode; choosing random instead."
                )
                ego_id = int(self.np_random.choice(valid_vehicle_ids))
            else:
                ego_id = int(ego_id_cfg)
        else:
            ego_id = int(self.np_random.choice(valid_vehicle_ids))

        self._ego_id = ego_id
        print(
            f"[NGSimEnv] Selected ego vehicle: {self._ego_id} "
            f"(from {len(valid_vehicle_ids)} with non-empty trajectories)"
        )

        # ------------------------------------------------------------------
        # 3) Build / reuse trajectory_set for (episode, ego_id)
        # ------------------------------------------------------------------
        traj_by_ego: Dict[int, Dict[str, Any]] = ep_cache["traj_by_ego"]
        if ego_id not in traj_by_ego:
            print(
                f"[NGSimEnv] Building trajectory_set for episode {ep_name}, ego {ego_id}"
            )
            traj_set = build_trajectory_from_chunk(
                scene,
                vehicle_ID=ego_id,
                episode_dir=self._episode_dir,
            )
            traj_by_ego[ego_id] = traj_set
        else:
            traj_set = traj_by_ego[ego_id]
            print(
                f"[NGSimEnv] Reusing cached trajectory_set for episode {ep_name}, ego {ego_id}"
            )

        self.trajectory_set = traj_set

        # ------------------------------------------------------------------
        # 4) Build / reuse expert actions for (episode, ego_id)
        # ------------------------------------------------------------------
        expert_by_ego: Dict[int, Tuple[np.ndarray | None, np.ndarray | None]] = ep_cache["expert_by_ego"]
        if ego_id not in expert_by_ego:
            expert_actions_sim, expert_times = self._load_expert_actions_for(
                self._episode_dir, ego_id
            )
            expert_by_ego[ego_id] = (expert_actions_sim, expert_times)
        else:
            expert_actions_sim, expert_times = expert_by_ego[ego_id]
            print(
                f"[NGSimEnv] Reusing cached expert actions for episode {ep_name}, ego {ego_id}"
            )

        self._expert_actions_sim = expert_actions_sim
        self._expert_times = expert_times

    # -------------------------------------------------------------------------
    # EXPERT ACTION QUERY
    # -------------------------------------------------------------------------
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
        # --- Ego trajectory ---
        ego_rec = self.trajectory_set["ego"]
        ego_traj = process_raw_trajectory(ego_rec["trajectory"])
        ego_len = ego_rec["length"]
        ego_wid = ego_rec["width"]

        ego_start_i = first_valid_index(ego_traj)
        if ego_start_i is None:
            raise RuntimeError("Ego trajectory contains no valid motion data.")

        # --- Horizon in policy steps based on trajectory length ---
        n_sim_steps = len(ego_traj) - int(ego_start_i)

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = int(sim_freq // pol_freq)  # e.g., 15 // 5 = 3

        self._max_traj_policy_steps = int(
            np.ceil(n_sim_steps / float(sim_per_policy))
        )
        """
        print(
            f"[NGSimEnv] Traj sim steps: {n_sim_steps}, "
            f"sim_per_policy: {sim_per_policy}, "
            f"max_traj_policy_steps: {self._max_traj_policy_steps}"
        )
        """
        # --- Ego initial state ---
        x0, y0, ego_speed, lane0 = ego_traj[ego_start_i]
        ego_xy = np.array([x0, y0], dtype=float)

        # ----------------- choose the correct edge based on x -----------------
        # Must match the geometry in create_ngsim_101_road()
        length = 2150 / 3.281
        ends = [
            0.0,
            560 / 3.281,
            (698 + 578 + 150) / 3.281,
            length,
        ]

        x_m = float(x0)  # assume ego_traj already in meters
        if x_m < ends[1]:
            main_edge = ("s1", "s2")   # first 5-lane section
        elif x_m < ends[2]:
            main_edge = ("s2", "s3")   # 6-lane section (lane 5 lives here)
        else:
            main_edge = ("s3", "s4")   # last 5-lane section

        lane_index = int(lane0)

        lanes_on_edge = self.net.graph[main_edge[0]][main_edge[1]]
        n_lanes = len(lanes_on_edge)
        if lane_index < 0 or lane_index >= n_lanes:
            print(
                f"[NGSimEnv] WARNING: lane_index {lane_index} out of range for "
                f"edge {main_edge} (n_lanes={n_lanes}); clamping."
            )
            lane_index = int(np.clip(lane_index, 0, n_lanes - 1))

        print(
            "lane ids:", lane_index,
            ", position:", x0, ",", y0,
            ", edge:", main_edge
        )

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

        # --- Surrounding vehicles ---
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

        if self.config["log_overlaps"]:
            overlaps = []
            vehs = [v for v in self.road.vehicles if hasattr(v, "aabb")]
            for i in range(len(vehs)):
                for j in range(i + 1, len(vehs)):
                    if vehs[i].overlaps_aabb(vehs[j]):
                        overlaps.append(
                            (
                                getattr(vehs[i], "vehicle_ID", -1),
                                getattr(vehs[j], "vehicle_ID", -1),
                            )
                        )
            if overlaps:
                info = dict(info) if info else {}
                info["overlaps"] = overlaps

        return obs, reward, terminated, truncated, info
