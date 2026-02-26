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
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle
from highway_env.ngsim_utils.gen_road import create_ngsim_101_road, clamp_location_ngsim
from highway_env.ngsim_utils.trajectory_to_action import traj_to_expert_actions, PurePursuitTracker
from highway_env.ngsim_utils.trajectory_gen import (
    process_raw_trajectory,
    first_valid_index,
)

Observation = np.ndarray

# Constants
f2m_conv = 3.281
MAX_ACCEL = 5.0
MAX_STEER = np.pi / 4

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
            "observation": {"type": "LidarObservation",
                "cells": 128,
                "maximum_range": 64,
                "normalize": True,
            },

            "action": {"type": "ContinuousAction"},

            # Frequencies
            "simulation_frequency": 10, # Must align with policy frequency
            "policy_frequency": 10, # Must align with policy frequency
            "max_episode_steps": 300,

            # Simulation override (if None -> sample random)
            "ego_vehicle_ID": None,
            "simulation_period": None,

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
            "lane_change_cooldown_steps": 10,
            "lateral_offset_step": 0.30,
            "lateral_offset_max": 1.50,
            "lane_change_commit_hyst_steps": 2, 

            # Expert Mode 
            "expert_test_mode": False,
            "expert_action_mode": "continuous",
            "target_speeds": list(np.arange(0.0, 35.0 + 1e-6, 2.0)),
            "expert_speed_deadband_mps": 0.5,
            "expert_steer_deadband_rad": 0.05,
            "expert_one_action_per_step": True,
            "expert_prefer_speed": False,
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
        #print(self._prebuilt_dir)

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
          - self.episode_name
          - self.ego_id

        Uses:
        - config["ego_vehicle_ID"]: optional explicit ego ID
        - config["simulation_period"]: optional episode/start override

        simulation_period can be:
        - None: random episode, random ego (current behaviour)
        - int:   ego start index only (handled later in _create_vehicles)
        - dict:  may contain:
            {
              "episode_name": <str>,        # required to fix episode
              "ego_start_index": <int>,     # optional start index override
            }
        """

        sim_period = self.config.get("simulation_period", None)
        explicit_ego_id = self.config.get("ego_vehicle_ID", None)

        # ---------------- Episode selection ----------------
        if isinstance(sim_period, dict) and "episode_name" in sim_period:
            # User explicitly chose which 10s chunk / episode to use
            episode_name = sim_period["episode_name"]
            if episode_name not in self._traj_all_by_episode:
                raise ValueError(
                    f"simulation_period.episode_name={episode_name!r} "
                    f"not in available episodes: {list(self._traj_all_by_episode.keys())[:10]}..."
                )
            self.episode_name = episode_name
        else:
            # Default: random episode as before
            self.episode_name = self.np_random.choice(self._episodes)

        # ---------------- Ego ID selection ----------------
        valid_ids = self._valid_ids_by_episode[self.episode_name]
        if explicit_ego_id is None:
            # Random ego within chosen episode
            self.ego_id = int(self.np_random.choice(valid_ids))
        else:
            # Use the requested ego_vehicle_ID
            if explicit_ego_id not in valid_ids:
                raise ValueError(
                    f"Requested ego_vehicle_ID={explicit_ego_id} "
                    f"not present in episode {self.episode_name}. "
                    f"Valid ids: {sorted(valid_ids)}"
                )
            self.ego_id = int(explicit_ego_id)

        #print("episode name:", self.episode_name, ", ego id:", self.ego_id)

        # ---------------- Build trajectory_set ----------------
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

        ego_len = ego_rec["length"] / f2m_conv 
        ego_wid = ego_rec["width"] / f2m_conv 

        # Reset replay logger each episode (needed for compare script)
        if not hasattr(self, "_replay_xy_pol"):
            self._replay_xy_pol = []
        else:
            self._replay_xy_pol.clear()

        # Check Expert Mode and Action Mode:
        expert_mode = bool(self.config.get("expert_test_mode", False))
        expert_action_mode = str(self.config.get("expert_action_mode", "continuous"))
        
        if expert_mode and expert_action_mode == "discrete":
            ego_control_mode = "discrete"
        else:
            ego_control_mode = "continuous" if (expert_mode and expert_action_mode == "continuous") else "continuous"

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

            # Initialize closed-loop tracker 
            self._tracker = PurePursuitTracker(
                ref_xy=self._expert_ref_xy_pol,
                ref_v=self._expert_ref_v_pol,
                dt=1.0 / pol_freq,          # tracker runs at policy rate
                L_forward=ego_len,
                max_steer=MAX_STEER,
                # Hard-coded Control Parameter
                Ld0= 5.0,
                Ld_k= 0.6,
                kp_v= 0.8,
                steer_rate_limit=6.0,
                steer_lpf_tau=0.15,
                jerk_limit=10.0,
            )

            #Store actions you generate (these are your "expert actions")
            self._expert_actions_policy = []
            self._tracker_dbg = []  


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
        # Compute heading from ego_trajectory
        dx0 = ego_traj[1, 0] - ego_traj[0, 0]
        dy0 = ego_traj[1, 1] - ego_traj[0, 1]

        disp = np.hypot(dx0, dy0)

        MIN_DISP = 0.1  # meters (reasonable for NGSIM @ 10Hz)

        if disp >= MIN_DISP:
            heading_raw = np.arctan2(dy0, dx0)
        else:
            heading_raw = 0.0  # safe only for first frame

        ego_lane = clamp_location_ngsim(x0, lane0, self.net, warning=False)
        target_lane_index = ego_lane.index  # LaneIndex tuple

        ego = EgoVehicle(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=heading_raw,
            control_mode=ego_control_mode,
            target_lane_index=target_lane_index,
            target_speeds=np.array(self.config.get("target_speeds", []), dtype=float) if self.config.get("target_speeds", None) is not None else None,
            lane_change_cooldown_steps=int(self.config.get("lane_change_cooldown_steps", 10)),
            lateral_offset_step=float(self.config.get("lateral_offset_step", EgoVehicle.DEFAULT_LATERAL_OFFSET_STEP)),
            lateral_offset_max=float(self.config.get("lateral_offset_max", EgoVehicle.DEFAULT_LATERAL_OFFSET_MAX)),
        )
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
                v_length=meta["length"] / f2m_conv,
                v_width=meta["width"] / f2m_conv,
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
    # REWARDS (minimal)
    # -------------------------------------------------------------------------
    def _rewards(self, action: Any) -> dict[str, float]:
        return {
            "collision_reward": float(self.vehicle.crashed),
        }

    def _reward(self, action: Any) -> float:
        return 0.0 if bool(self.vehicle.crashed) else 1.0

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
    def _get_expert_action_for_step(self):
        if not hasattr(self, "_expert_actions_policy") or self._expert_actions_policy is None:
            raise RuntimeError("Expert test mode enabled but expert actions are not available.")
        if len(self._expert_actions_policy) == 0:
            raise RuntimeError("No expert actions recorded yet.")
        idx = min(self.steps, len(self._expert_actions_policy) - 1)
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
        expert_action = None            # continuous expert action array
        expert_action_str = None        # discrete label, e.g. "FASTER"
        expert_action_idx = None        # discrete index passed to action_type

        expert_test = bool(self.config.get("expert_test_mode", False))
        if expert_test:
            expert_mode = str(self.config.get("expert_action_mode", "continuous")).lower()

            # --- tracker output ---
            pos = self.vehicle.position
            hdg = float(self.vehicle.heading)
            spd = float(self.vehicle.speed)
            steer_cmd, accel_cmd, i_near, i_tgt = self._tracker.step(pos, hdg, spd)

            if expert_mode == "continuous":
                accel_norm = float(np.clip(accel_cmd / MAX_ACCEL, -1.0, 1.0))
                steer_norm = float(np.clip(steer_cmd / MAX_STEER, -1.0, 1.0))
                expert_action = np.array([accel_norm, steer_norm], dtype=np.float32)

                # IMPORTANT: pass to super().step() and let ActionType apply it once.
                action = expert_action

            elif expert_mode == "discrete":
                # 1) compute label from tracker (your existing mapper)
                expert_action_str = self._discrete_expert_action_from_tracker(
                    steer_cmd, accel_cmd
                    # If you can, pass i_tgt and use it in your speed decision; optional:
                    # , i_tgt=i_tgt
                )

                # 2) convert label -> index expected by configured action_type
                #    This requires your env to be configured with an ActionType that supports these labels.
                if not hasattr(self, "action_type"):
                    raise RuntimeError("self.action_type is missing; cannot execute discrete actions.")

                if not hasattr(self.action_type, "actions_indexes"):
                    raise RuntimeError(
                        "Current action_type has no 'actions_indexes'. "
                        "Discrete expert replay requires a discrete/meta action type."
                    )

                if expert_action_str not in self.action_type.actions_indexes:
                    raise RuntimeError(
                        f"Expert produced action label {expert_action_str!r}, but action_type does not support it. "
                        f"Supported labels: {sorted(self.action_type.actions_indexes.keys())}"
                    )

                expert_action_idx = int(self.action_type.actions_indexes[expert_action_str])

                # IMPORTANT: pass the index to super().step(); DO NOT call self.vehicle.act() here.
                action = expert_action_idx

            else:
                raise ValueError(f"Unknown expert_action_mode={expert_mode!r}")

            # Optional debugging logs
            if hasattr(self, "_expert_actions_policy") and self._expert_actions_policy is not None:
                if expert_action is not None:
                    self._expert_actions_policy.append(expert_action.copy())
                elif expert_action_str is not None:
                    self._expert_actions_policy.append(expert_action_str)

            if hasattr(self, "_tracker_dbg") and self._tracker_dbg is not None:
                self._tracker_dbg.append((int(i_near), int(i_tgt)))

        # --- single application path: super().step applies action exactly once ---
        obs, reward, terminated, truncated, info = super().step(action)

        if info is None:
            info = {}

        # Log what you actually executed
        info["applied_action"] = action
        if expert_action is not None:
            info["expert_action_continuous"] = expert_action.copy()
        if expert_action_str is not None:
            info["expert_action_discrete"] = expert_action_str
            info["expert_action_discrete_idx"] = expert_action_idx

        if expert_test:
            self._replay_xy_pol.append(self.vehicle.position.copy())

        if not truncated and self._is_truncated():
            truncated = True

        return obs, reward, terminated, truncated, info

    
    def _nearest_speed_index(self, v: float) -> int:
        ts = np.asarray(self.config.get("target_speeds", []), dtype=float)
        if ts.size == 0:
            # fallback to EgoVehicle defaults if present
            ts = np.asarray(getattr(self.vehicle, "target_speeds", []), dtype=float)
        if ts.size == 0:
            raise RuntimeError("No target_speeds configured for discrete expert replay.")
        return int(np.argmin(np.abs(ts - float(v))))

    def _discrete_expert_action_from_tracker(self, steer_cmd: float, accel_cmd: float) -> str:
        """
        Map continuous (steer_cmd, accel_cmd) to one of:
        {"SLOWER","IDLE","FASTER","STEER_LEFT","STEER_RIGHT"}

        Uses reference speed at the tracker target index if available, otherwise uses accel_cmd sign.
        """
        v_dead = float(self.config.get("expert_speed_deadband_mps", 0.5))
        s_dead = float(self.config.get("expert_steer_deadband_rad", 0.05))
        one = bool(self.config.get("expert_one_action_per_step", True))
        prefer_speed = bool(self.config.get("expert_prefer_speed", False))

        # --- Steering desire ---
        steer_des = 0
        if steer_cmd > s_dead:
            steer_des = +1
        elif steer_cmd < -s_dead:
            steer_des = -1

        # --- Speed desire ---
        # Prefer using the expert reference speed at the next target index if tracker exposes it;
        # otherwise, use accel_cmd to infer whether to speed up or slow down.
        v_des = 0
        if hasattr(self, "_tracker") and hasattr(self, "_expert_ref_v_pol"):
            # use tracker target index if you logged it; else compare to nearest ref at current step
            # simplest robust version: compare current speed to nearest reference at current step index
            # (you can improve by using i_tgt returned by tracker.step)
            v_ref = float(self._expert_ref_v_pol[min(self.steps, len(self._expert_ref_v_pol) - 1)])
            if (v_ref - float(self.vehicle.speed)) > v_dead:
                v_des = +1
            elif (float(self.vehicle.speed) - v_ref) > v_dead:
                v_des = -1
        else:
            # fallback purely from accel sign
            if accel_cmd > 0.2:
                v_des = +1
            elif accel_cmd < -0.2:
                v_des = -1

        if not one:
            # If you ever decide to allow two actions per step, you need a different action space.
            # For now, keep one-action-per-step only.
            pass

        # Resolve conflict if both want to change
        if steer_des != 0 and v_des != 0:
            if prefer_speed:
                steer_des = 0
            else:
                v_des = 0

        if v_des > 0:
            return "FASTER"
        if v_des < 0:
            return "SLOWER"
        if steer_des < 0:
            return "STEER_RIGHT"
        if steer_des > 0:
            return "STEER_LEFT"
        return "IDLE"
