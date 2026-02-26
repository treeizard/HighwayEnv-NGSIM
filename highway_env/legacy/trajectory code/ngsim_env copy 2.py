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
from typing import Any, Dict

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road

from highway_env.ngsim_utils.helper_ngsim import load_ego_trajectory, get_ego_dimensions, setup_expert_tracker
from highway_env.ngsim_utils.obs_vehicle import spawn_surrounding_vehicles
from highway_env.ngsim_utils.ego_vehicle import create_ego_vehicle
from highway_env.ngsim_utils.gen_road import create_ngsim_101_road
from highway_env.ngsim_utils.trajectory_to_action import traj_to_expert_actions, map_discrete_expert_action, PurePursuitTracker
from highway_env.ngsim_utils.trajectory_gen import process_raw_trajectory, first_valid_index

'''
Debug
'''
from highway_env.ngsim_utils.gen_road import clamp_location_ngsim
from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle

# Constants
f2m_conv = 3.281
MAX_ACCEL = 5.0
MAX_STEER = np.pi / 4


class NGSimEnv(AbstractEnv):
    """
    Fast NGSIM Driving Env using 10-second chunk episodes.
    Expert replay supports:
      - continuous: normalized [accel, steer] from tracker
      - discrete: lane-change labels from reference lane-id + speed/steer labels from tracker

    Key for segmented road (your create_ngsim_101_road):
      - Lane count changes by x-section: ("s1","s2")=5, ("s2","s3")=6, ("s3","s4")=5
      - We clamp reference lane-id per step to the current section lane-count
      - Discrete lane-change is triggered on *commit* with hysteresis + cooldown
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
        config.update(
            {
                "scene": "us-101",
                # Observation / action
                "observation": {
                    "type": "LidarObservation",
                    "cells": 128,
                    "maximum_range": 64,
                    "normalize": True,
                },
                "action": {"type": "ContinuousAction"},
                # Frequencies
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "max_episode_steps": 300,
                # Simulation override
                "ego_vehicle_ID": None,
                "simulation_period": None,
                # Episode selection
                "episode_root": "highway_env/data/processed_10s",
                "replay_period": None,
                # Spawn safety
                "max_surrounding": 80,
                # Debug
                "show_trajectories": True,
                "seed": None,
                # Reward config
                "collision_reward": -1.0,
                "high_speed_reward": 1.0,
                "reward_speed_range": [20.0, 30.0],
                # Discrete lane-change + offset params (used by EgoVehicle + labeler)
                "lane_change_cooldown_steps": 10,
                "lateral_offset_step": 0.10,
                "lateral_offset_max": 1.50,
                "lane_change_commit_hyst_steps": 2,
                # Expert Mode
                "expert_test_mode": False,
                "expert_action_mode": "continuous",  # "continuous" or "discrete"
                "target_speeds": list(np.arange(0.0, 35.0 + 1e-6, 2.0)),
                "expert_speed_deadband_mps": 0.5,
                "expert_steer_deadband_rad": 0.05,
                "expert_one_action_per_step": True,
                "expert_prefer_speed": False,
            }
        )
        return config

    # -------------------------------------------------------------------------
    # INIT / DT
    # -------------------------------------------------------------------------
    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        # Normalize config early (before AbstractEnv builds action_type)
        if config is None:
            config = self.default_config()
        else:
            # Merge user config on top of defaults (in case caller passed partial config)
            merged = self.default_config()
            merged.update(config)
            config = merged

        # If expert discrete, you MUST use a discrete action type that supports:
        # "FASTER","SLOWER","IDLE","STEER_LEFT","STEER_RIGHT","LANE_LEFT","LANE_RIGHT"
        # We do not force it here (to avoid unexpected behavior), but we do validate at runtime.
        self._episodes: list[str] = []
        self._episode_root: str | None = None
        self._valid_ids_by_episode: dict[str, np.ndarray] = {}
        self._episode_cache: Dict[str, Dict[str, Any]] = {}
        self.ego_id: int | None = None

        self._ego_id: int | None = None

        # Expert debug
        self._expert_actions_sim: np.ndarray | None = None
        self._expert_times: np.ndarray | None = None
        self._max_traj_policy_steps: int | None = None
        self._replay_xy_pol: list[np.ndarray] = []

        # Prebuilt trajectories
        self._prebuilt_dir = os.path.join(config["episode_root"], config["scene"], "prebuilt")
        veh_ids_path = os.path.join(self._prebuilt_dir, "veh_ids_train.npy")
        traj_path = os.path.join(self._prebuilt_dir, "trajectory_train.npy")

        self._valid_ids_by_episode = np.load(veh_ids_path, allow_pickle=True).item()
        self._traj_all_by_episode = np.load(traj_path, allow_pickle=True).item()
        self._episodes = sorted(self._traj_all_by_episode.keys())

        super().__init__(config=config, render_mode=render_mode)

    @property
    def dt(self) -> float:
        return 1.0 / float(self.config["simulation_frequency"])

    # -------------------------------------------------------------------------
    # NGSIM SECTION HELPERS (consistent with create_ngsim_101_road)
    # -------------------------------------------------------------------------
    '''
    def _main_edge_from_x(self, x: float) -> tuple[str, str]:
        length = 2150 / 3.281
        ends = [0.0, 560 / 3.281, (698 + 578 + 150) / 3.281, length]
        x_m = float(x)
        if x_m < ends[1]:
            return ("s1", "s2")  # 5 lanes
        elif x_m < ends[2]:
            return ("s2", "s3")  # 6 lanes
        else:
            return ("s3", "s4")  # 5 lanes

    def _clamp_lane_id_for_x(self, x: float, lane_id: int) -> int:
        edge = self._main_edge_from_x(x)
        n_lanes = len(self.net.graph[edge[0]][edge[1]])
        return int(np.clip(int(lane_id), 0, n_lanes - 1))
    '''
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
        sim_period = self.config.get("simulation_period", None)
        explicit_ego_id = self.config.get("ego_vehicle_ID", None)

        # Episode selection
        if isinstance(sim_period, dict) and "episode_name" in sim_period:
            episode_name = sim_period["episode_name"]
            if episode_name not in self._traj_all_by_episode:
                raise ValueError(
                    f"simulation_period.episode_name={episode_name!r} not found. "
                    f"Available examples: {self._episodes[:10]}"
                )
            self.episode_name = episode_name
        else:
            self.episode_name = self.np_random.choice(self._episodes)

        # Ego selection
        valid_ids = self._valid_ids_by_episode[self.episode_name]
        if explicit_ego_id is None:
            self.ego_id = int(self.np_random.choice(valid_ids))
        else:
            if explicit_ego_id not in valid_ids:
                raise ValueError(
                    f"Requested ego_vehicle_ID={explicit_ego_id} not present in episode {self.episode_name}. "
                    f"Valid ids: {sorted(valid_ids)}"
                )
            self.ego_id = int(explicit_ego_id)

        # Build trajectory_set
        traj_all = self._traj_all_by_episode[self.episode_name]
        traj_set = {"ego": traj_all[self.ego_id]}
        for vid, meta in traj_all.items():
            if vid == self.ego_id:
                continue
            traj_set[vid] = meta
        self.trajectory_set = traj_set

        print("episode name:", self.episode_name, ", ego_id:", self.ego_id)

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
        
        # --------------- Initiate Ego Data (checked) ---------------------------
        ego_rec = self.trajectory_set["ego"]
        ego_traj_full = load_ego_trajectory(ego_rec)
        ego_len, ego_wid = get_ego_dimensions(ego_rec, f2m_conv)

        # Expert Testing Mode Set Up
        if not hasattr(self, "_replay_xy_pol"):
            self._replay_xy_pol = []
        else:
            self._replay_xy_pol.clear()
        
        expert_mode = bool(self.config.get("expert_test_mode", False))
        
        # ---------------  Expert mode ---------------------------
        if expert_mode:
            ref_xy_pol, ref_v_pol, lane_pol, start_idx = setup_expert_tracker(self.net, ego_traj_full, ego_len, self.config)
            
            # Store the expert reference data
            self._expert_ref_xy_pol = ref_xy_pol
            self._expert_ref_v_pol = ref_v_pol
            self._expert_ref_lane_pol = lane_pol
            self._ego_start_index = start_idx

            # Initialize the tracker with the processed reference trajectory data
            self._tracker = PurePursuitTracker(
                ref_xy=self._expert_ref_xy_pol,
                ref_v=self._expert_ref_v_pol,
                dt=1.0 / self.config["policy_frequency"],
                L_forward=ego_len,
                max_steer=MAX_STEER,
                Ld0=5.0,
                Ld_k=0.6,
                kp_v=0.8,
                steer_rate_limit=6.0,
                steer_lpf_tau=0.15,
                jerk_limit=10.0,
            )

            # Store actions generated by the expert tracker
            self._expert_actions_policy = []
            self._tracker_dbg = []  # Debugging tracker data

        # Proceed with non-expert mode if applicable
        else:
            ego_start_i = first_valid_index(ego_rec["trajectory"])
            if ego_start_i is None:
                raise RuntimeError("Ego trajectory contains no valid motion data.")
            self._ego_start_index = int(ego_start_i)
        

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
        expert_action_mode = str(self.config.get("expert_action_mode", "continuous"))
        if expert_mode and expert_action_mode == "discrete":
            ego_control_mode = "discrete"
        else:
            ego_control_mode = "continuous" if (expert_mode and expert_action_mode == "continuous") else "continuous"
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
        
        # --- Important: Essential for logging and replaying do not remove --- 
        if self.config.get("expert_test_mode", False):
            self._replay_xy_pol.append(self.vehicle.position.copy())
        
        # Surrounding vehicles
        max_surr = int(self.config.get("max_surrounding", 0))
        if max_surr > 0:
            spawn_surrounding_vehicles(self.trajectory_set, self._ego_start_index, max_surr, self.road)
        
        if expert_mode:
            self._replay_xy_pol.append(self.vehicle.position.copy())
    
    # -------------------------------------------------------------------------
    # REWARDS
    # -------------------------------------------------------------------------
    def _rewards(self, action: Any) -> dict[str, float]:
        return {"collision_reward": float(self.vehicle.crashed)}

    def _reward(self, action: Any) -> float:
        return 0.0 if bool(self.vehicle.crashed) else 1.0

    # -------------------------------------------------------------------------
    # TERMINATION / TRUNCATION
    # -------------------------------------------------------------------------
    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed)

    def _is_truncated(self) -> bool:
        max_steps_cfg = self.config.get("max_episode_steps", None)
        max_steps_traj = getattr(self, "_max_traj_policy_steps", None)

        candidates = [v for v in (max_steps_cfg, max_steps_traj) if v is not None]
        if not candidates:
            return False
        hard_cap = min(candidates)
        return self.steps >= hard_cap

    # -------------------------------------------------------------------------
    # EXPERT REPLAY METRICS
    # -------------------------------------------------------------------------
    def _discrete_expert_action_from_tracker(self, steer_cmd: float, accel_cmd: float) -> str:
        """
        Map (steer_cmd, accel_cmd) to one of:
        {"SLOWER", "IDLE", "FASTER", "STEER_LEFT", "STEER_RIGHT"}
        Lane-change is handled separately from reference lane-id.
        """
        # Retrieve config values for deadbands and preference
        v_dead = float(self.config.get("expert_speed_deadband_mps", 0.5))
        s_dead = 4 
        #float(self.config.get("expert_steer_deadband_rad", 5))
        prefer_speed = bool(self.config.get("expert_prefer_speed", False))
        steps = self.steps

        # Assuming `_expert_ref_v_pol` and `vehicle_speed` are available as instance attributes
        expert_ref_v_pol = self._expert_ref_v_pol
        vehicle_speed = self.vehicle.speed
        #print("steer_cmd:",steer_cmd)
        return map_discrete_expert_action(steer_cmd, accel_cmd, 
                                          expert_ref_v_pol, vehicle_speed, 
                                          steps,
                                          v_dead, s_dead, prefer_speed)

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
    # DISCRETE LANE-CHANGE LABELER (commit + hysteresis + cooldown)
    # -------------------------------------------------------------------------
    def _lane_change_label_at_step(self, t: int) -> str | None:
        """
        Return 'LANE_LEFT' / 'LANE_RIGHT' at commit moment, else None.

        Commit rule:
          - lane id changes between t-1 and t
          - lane_now persists for H steps (hysteresis)
          - cooldown prevents repeated toggling
        """
        if not hasattr(self, "_expert_ref_lane_pol"):
            return None
        if t <= 0 or t >= len(self._expert_ref_lane_pol):
            return None

        if getattr(self, "_lc_label_cooldown", 0) > 0:
            self._lc_label_cooldown -= 1
            return None

        H = int(self.config.get("lane_change_commit_hyst_steps", 2))
        H = max(0, H)

        lane_prev = int(self._expert_ref_lane_pol[t - 1])
        lane_now = int(self._expert_ref_lane_pol[t])

        if lane_now == lane_prev:
            return None

        end = min(t + H, len(self._expert_ref_lane_pol) - 1)
        if end > t:
            if not np.all(self._expert_ref_lane_pol[t : end + 1] == lane_now):
                return None

        # Convention (as used in your EgoVehicle): left = lane_id-1, right = lane_id+1.
        # Therefore delta>0 -> lane_id increased -> "LANE_RIGHT" under that convention.
        delta = lane_now - lane_prev
        label = "LANE_RIGHT" if delta > 0 else "LANE_LEFT"

        self._lc_label_cooldown = int(self.config.get("lane_change_cooldown_steps", 10))
        return label

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action: Action):
        expert_action = None
        expert_action_str = None
        expert_action_idx = None

        expert_test = bool(self.config.get("expert_test_mode", False))
        if expert_test:
            expert_mode = str(self.config.get("expert_action_mode", "continuous")).lower()

            pos = self.vehicle.position
            hdg = float(self.vehicle.heading)
            spd = float(self.vehicle.speed)
            steer_cmd, accel_cmd, i_near, i_tgt = self._tracker.step(pos, hdg, spd)

            if expert_mode == "continuous":
                accel_norm = float(np.clip(accel_cmd / MAX_ACCEL, -1.0, 1.0))
                steer_norm = float(np.clip(steer_cmd / MAX_STEER, -1.0, 1.0))
                expert_action = np.array([accel_norm, steer_norm], dtype=np.float32)
                action = expert_action

            elif expert_mode == "discrete":
                # 0) lane-change label takes priority (commit-based)
                lc = self._lane_change_label_at_step(self.steps)
                if lc is not None:
                    expert_action_str = lc
                    print("lane_change:", lc)
                else:
                    expert_action_str = self._discrete_expert_action_from_tracker(steer_cmd, accel_cmd)
                
                #print("Expert Meta action:",expert_action_str)

                # 1) validate action_type supports discrete labels
                if not hasattr(self, "action_type"):
                    raise RuntimeError("self.action_type is missing; cannot execute discrete actions.")
                if not hasattr(self.action_type, "actions_indexes"):
                    raise RuntimeError(
                        "Current action_type has no 'actions_indexes'. "
                        "Set config['action']['type'] to your discrete meta action type "
                        "(e.g., DiscreteSteerMetaAction) for discrete expert replay."
                    )
                if expert_action_str not in self.action_type.actions_indexes:
                    raise RuntimeError(
                        f"Expert produced action label {expert_action_str!r}, but action_type does not support it. "
                        f"Supported labels: {sorted(self.action_type.actions_indexes.keys())}"
                    )

                expert_action_idx = int(self.action_type.actions_indexes[expert_action_str])
                action = expert_action_idx

            else:
                raise ValueError(f"Unknown expert_action_mode={expert_mode!r}")
            #print("I am in: ",self.vehicle.lane_index)
            #print("my target_lane is: ", self.vehicle.target_lane_index)
            # Debug logs
            if hasattr(self, "_expert_actions_policy") and self._expert_actions_policy is not None:
                if expert_action is not None:
                    self._expert_actions_policy.append(expert_action.copy())
                elif expert_action_str is not None:
                    self._expert_actions_policy.append(expert_action_str)

            if hasattr(self, "_tracker_dbg") and self._tracker_dbg is not None:
                self._tracker_dbg.append((int(i_near), int(i_tgt)))

        # Single-application path
        obs, reward, terminated, truncated, info = super().step(action)
        if info is None:
            info = {}

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

