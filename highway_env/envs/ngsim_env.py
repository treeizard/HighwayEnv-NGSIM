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
from highway_env.ngsim_utils.trajectory_to_action import traj_to_expert_actions, PurePursuitTracker
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
            "simulation_frequency": 10, # Must align !!!
            "policy_frequency": 10, # Must align !!!
            "max_episode_steps": 300,

            # Ego override (if None → sample random)
            "ego_vehicle_ID": None,

            # Episode selection
            "episode_root": "highway_env/data/processed_10s",
            "replay_period": None,        # None = random episode each reset

            # Spawn safety
            "max_surrounding": 80,

            # Debug
            "show_trajectories": True,
            "seed": None,

            # Reward config (weights)
            "collision_reward": -1.0,
            "high_speed_reward": 1.0,
            "reward_speed_range": [20.0, 30.0],

            # Lane change heuristics
            "lane_change_action_indices": [0, 2],
            "lane_change_steer_thresh": 0.3,

            # Expert Mode 
            "expert_test_mode": False,
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
        self._replay_xy_pol = []

        # Load all Trajectories
        self._prebuilt_dir = os.path.join(config["episode_root"], config["scene"], "prebuilt")
        print(self._prebuilt_dir)

        # Read in precomputed Trajectory data
        veh_ids_path= os.path.join(self._prebuilt_dir, f"veh_ids_train.npy")
        traj_path = os.path.join(self._prebuilt_dir, f"trajectory_train.npy")
        self._valid_ids_by_episode = np.load(veh_ids_path, allow_pickle=True).item()
        self._traj_all_by_episode  = np.load(traj_path, allow_pickle=True).item()
        self._episodes = sorted(self._traj_all_by_episode.keys())
        
        
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
        if self.config.get("expert_test_mode", False):
            self._replay_xy_pol = [self.vehicle.position.copy()]
  

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

        # Reset replay logger each episode (needed for compare script)
        if not hasattr(self, "_replay_xy_pol"):
            self._replay_xy_pol = []
        else:
            self._replay_xy_pol.clear()

        # expert mode:
        # expert mode:
        if self.config.get("expert_test_mode", False):
            # Define reference window: keep your original logic for start_idx if you want
            # If ego_traj_full is already truncated/valid, you can just use the full slice.
            # Here I keep your existing 'traj_to_expert_actions' start_idx selection ONLY to find a window.
            expert = traj_to_expert_actions(ego_traj_full, dt=self.dt, L_forward=ego_len)
            start_idx = int(expert["start_idx"])
            end_idx   = int(expert["end_idx"])
            if end_idx <= start_idx:
                raise RuntimeError(f"Invalid expert start/end idx: {start_idx}, {end_idx}")

            self._ego_start_index = start_idx

            sim_freq = float(self.config["simulation_frequency"])
            pol_freq = float(self.config["policy_frequency"])
            sim_per_policy = max(1, int(sim_freq // pol_freq))

            ref_slice = ego_traj_full[start_idx:end_idx + 1]
            self._expert_ref_xy_pol = ref_slice[:, :2][::sim_per_policy].copy()
            self._expert_ref_v_pol  = ref_slice[:, 2][::sim_per_policy].copy()

            # Initialize closed-loop tracker (IMPORTANT: use ego_len as L_forward)
            self._tracker = PurePursuitTracker(
                ref_xy=self._expert_ref_xy_pol,
                ref_v=self._expert_ref_v_pol,
                dt=1.0 / pol_freq,          # tracker runs at policy rate
                L_forward=ego_len,
                max_steer=np.pi / 4,
                # human-like defaults (tune later)
                Ld0=5.0,
                Ld_k=0.6,
                kp_v=0.8,
                steer_rate_limit=6.0,
                steer_lpf_tau=0.15,
                jerk_limit=10.0,
            )

            # Optional: store actions you generate (these are your "expert actions")
            self._expert_actions_policy = []
            self._tracker_dbg = []  # (nearest_idx, target_idx) per step


        # non-expert mode:
        else:
            ego_start_i = first_valid_index(ego_rec["trajectory"])
            if ego_start_i is None:
                raise RuntimeError("Ego trajectory contains no valid motion data.")
            self._ego_start_index = int(ego_start_i)

        # --------- TRUNCATE ALL TRAJECTORIES AT EGO START INDEX ---------
        ego_traj = ego_traj_full[self._ego_start_index :]
        if len(ego_traj) < 2:
            raise RuntimeError(f"Ego trajectory too short after truncation (len={len(ego_traj)}).")

        # --- Horizon in policy steps based on truncated trajectory length ---
        n_sim_steps = len(ego_traj)
        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = max(1, int(sim_freq // pol_freq))
        self._max_traj_policy_steps = int(np.ceil(n_sim_steps / float(sim_per_policy)))

        # --- Ego initial state (meters) from truncated traj[0] ---
        x0, y0, ego_speed, lane0 = ego_traj[0]
        ego_xy = np.array([x0, y0], dtype=float)
        '''
        ego_lane = clamp_location_ngsim(
            x_pos=x0,
            lane0=lane0,
            net=self.net,
            warning=False,
        )
        '''
        #ego_s, ego_r = ego_lane.local_coordinates(ego_xy)

        # Keep your original behaviour: snap to lane centerline
        #ego_xy = np.asarray(ego_lane.position(ego_s, ego_r), float)
        ego_xy = np.array([x0, y0], dtype=float)
        # Compute heading from ego_trajectory
        #print(ego_traj)
        dx0 = ego_traj[1, 0] - ego_traj[0, 0]
        dy0 = ego_traj[1, 1] - ego_traj[0, 1]

        disp = np.hypot(dx0, dy0)

        MIN_DISP = 0.1  # meters (reasonable for NGSIM @ 10Hz)

        if disp >= MIN_DISP:
            heading_raw = np.arctan2(dy0, dx0)
        else:
            heading_raw = 0.0  # safe only for first frame

        ego = ControlledVehicle(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=heading_raw,
            control_mode="continuous",
        )
        print(heading_raw)
        ego.set_ego_dimension(width=ego_wid, length=ego_len)
        # DEBUGGING:
        self.road.vehicles.append(ego)
        self.vehicle = ego

        # MINIMAL step-mismatch fix:
        # log t=0 position here, so replay and reference align without rep[:-1]
        if self.config.get("expert_test_mode", False):
            self._replay_xy_pol.append(self.vehicle.position.copy())

        # ---------------- MINIMAL CRASH FIX ----------------
        # honour max_surrounding. If 0, do not spawn any other vehicles.
        max_surr = int(self.config.get("max_surrounding", 0))
        if max_surr <= 0:
            return
        # ---------------------------------------------------

        # --- Surrounding vehicles (also truncated to ego spawn time) ---
        spawned = 0
        for vid, meta in self.trajectory_set.items():
            if vid == "ego":
                continue
            if spawned >= max_surr:
                break

            traj_full = process_raw_trajectory(meta["trajectory"])
            if len(traj_full) <= self._ego_start_index:
                continue

            traj = traj_full[self._ego_start_index :]

            # drop leading ghost segment
            nonzero = np.any(traj[:, :3] != 0.0, axis=1)
            if not np.any(nonzero):
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

            # keep collision off in expert debug to avoid the polygon NoneType crash
            if self.config.get("expert_test_mode", False):
                v.COLLISIONS_ENABLED = False

            self.road.vehicles.append(v)
            spawned += 1




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
    # EXPERT ACTION QUERY
    # -------------------------------------------------------------------------
    def _get_expert_action_for_step(self) -> np.ndarray:
        if not hasattr(self, "_expert_actions_policy") or self._expert_actions_policy is None:
            raise RuntimeError(
                "Expert test mode enabled but expert actions are not available."
            )

        # self.steps = number of policy steps taken so far
        idx = min(self.steps, self._expert_actions_policy.shape[0] - 1)

        return self._expert_actions_policy[idx]
    
    def expert_replay_metrics(self):
        if not hasattr(self, "_expert_ref_xy_pol"):
            raise RuntimeError("No expert reference saved. Run reset with expert_test_mode=True.")
        if not hasattr(self, "_replay_xy_pol"):
            raise RuntimeError("No replay recorded yet. Step the env first.")

        ref = np.asarray(self._expert_ref_xy_pol, dtype=float)
        rep = np.asarray(self._replay_xy_pol, dtype=float)

        T = min(len(ref), len(rep))
        ref = ref[:T]
        rep = rep[:T]

        err = np.linalg.norm(ref - rep, axis=1)
        ade = float(np.mean(err))
        fde = float(err[-1]) if T > 0 else np.nan

        return {"T": T, "ADE_m": ade, "FDE_m": fde, "err_per_step_m": err}

    
    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action: Action):
        # Expert mode debug
        if self.config.get("expert_test_mode", False):
            # closed-loop action from tracker based on current simulated state
            pos = self.vehicle.position
            hdg = float(self.vehicle.heading)
            spd = float(self.vehicle.speed)

            steer_cmd, accel_cmd, i_near, i_tgt = self._tracker.step(pos, hdg, spd)

            # IMPORTANT: match YOUR action convention.
            # You currently stack [accel, steering] normalized.
            MAX_ACCEL = 5.0
            MAX_STEER = np.pi / 4
            accel_norm = float(np.clip(accel_cmd / MAX_ACCEL, -1.0, 1.0))
            steer_norm = float(np.clip(steer_cmd / MAX_STEER, -1.0, 1.0))

            action = np.array([accel_norm, steer_norm], dtype=np.float32)

            # log the expert action actually applied
            self._expert_actions_policy.append(action.copy())
            self._tracker_dbg.append((i_near, i_tgt))


        obs, reward, terminated, truncated, info = super().step(action)
        #self.steps += 1
        #print("step:", self.steps)
        if self.config.get("expert_test_mode", False):            
            self._replay_xy_pol.append(self.vehicle.position.copy())

        if not truncated and self._is_truncated():
            truncated = True
      
        return obs, reward, terminated, truncated, info
