# Modified by: Yide Tao (yide.tao@monash.edu)
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road

from highway_env.ngsim_utils.helper_ngsim import load_ego_trajectory, get_ego_dimensions, setup_expert_tracker, clamp_lane_id_for_x
from highway_env.ngsim_utils.obs_vehicle import spawn_surrounding_vehicles
#from highway_env.ngsim_utils.ego_vehicle import create_ego_vehicle
from highway_env.ngsim_utils.gen_road import create_ngsim_101_road, clamp_location_ngsim
from highway_env.ngsim_utils.trajectory_to_action import map_discrete_expert_action, PurePursuitTracker #, traj_to_expert_actions
from highway_env.ngsim_utils.trajectory_gen import first_valid_index #, process_raw_trajectory

#from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle

# Constants
f2m_conv = 3.281
MAX_ACCEL = 5.0
MAX_STEER = np.pi / 4


class NGSimEnv(AbstractEnv):
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
                "observation": {
                    "type": "LidarObservation",
                    "cells": 128,
                    "maximum_range": 64,
                    "normalize": True,
                },
                "action": {"type": "ContinuousAction"},
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "max_episode_steps": 300,
                "ego_vehicle_ID": None,
                "simulation_period": None,
                "episode_root": "highway_env/data/processed_10s",
                "max_surrounding": 80,
                "show_trajectories": True,
                "seed": None,
                "lane_change_cooldown_steps": 10,
                "lateral_offset_step": 0.10,
                "lateral_offset_max": 1.50,
                "lane_change_commit_hyst_steps": 2,
                "expert_test_mode": False,
                "action_mode": "discrete",
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
        if config is None:
            config = self.default_config()
        else:
            merged = self.default_config()
            merged.update(config)
            config = merged

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


        # Define action_mode
        self.control_mode = str(config.get("action_mode", "continuous")).lower()

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
        sim_period = self.config.get("simulation_period", None)
        explicit_ego_id = self.config.get("ego_vehicle_ID", None)

        if isinstance(sim_period, dict) and "episode_name" in sim_period:
            episode_name = sim_period["episode_name"]
            if episode_name not in self._traj_all_by_episode:
                raise ValueError(f"Episode {episode_name} not found.")
            self.episode_name = episode_name
        else:
            self.episode_name = self.np_random.choice(self._episodes)

        valid_ids = self._valid_ids_by_episode[self.episode_name]
        if explicit_ego_id is None:
            self.ego_id = int(self.np_random.choice(valid_ids))
        else:
            if explicit_ego_id not in valid_ids:
                raise ValueError(f"Ego ID {explicit_ego_id} not in {self.episode_name}")
            self.ego_id = int(explicit_ego_id)

        traj_all = self._traj_all_by_episode[self.episode_name]
        traj_set = {"ego": traj_all[self.ego_id]}
        for vid, meta in traj_all.items():
            if vid == self.ego_id:
                continue
            traj_set[vid] = meta
        self.trajectory_set = traj_set

        #print("episode name:", self.episode_name, ", ego_id:", self.ego_id)

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
        ego_rec = self.trajectory_set["ego"]
        ego_traj_full = load_ego_trajectory(ego_rec)
        ego_len, ego_wid = get_ego_dimensions(ego_rec, f2m_conv)

        if not hasattr(self, "_replay_xy_pol"):
            self._replay_xy_pol = []
        else:
            self._replay_xy_pol.clear()
        
        expert_mode = bool(self.config.get("expert_test_mode", False))
        
        if expert_mode:
            # Note: setup_expert_tracker returns raw lane_pol for runtime clamping
            ref_xy_pol, ref_v_pol, lane_pol, start_idx = setup_expert_tracker(self.net, ego_traj_full, ego_len, self.config)
            #print(lane_pol)
            self._expert_ref_xy_pol = ref_xy_pol
            self._expert_ref_v_pol = ref_v_pol
            self._expert_ref_lane_pol = lane_pol - 1 
            self._ego_start_index = start_idx

            # UPDATE: Pass ref_lanes to tracker so step() can return target_lane_id
            self._tracker = PurePursuitTracker(
                ref_xy=self._expert_ref_xy_pol,
                ref_v=self._expert_ref_v_pol,
                ref_lanes=self._expert_ref_lane_pol,  # <--- CRITICAL UPDATE
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
            self._expert_actions_policy = []
            self._tracker_dbg = []
        else:
            ego_start_i = first_valid_index(ego_rec["trajectory"])
            if ego_start_i is None:
                raise RuntimeError("Ego trajectory contains no valid motion data.")
            self._ego_start_index = int(ego_start_i)
        
        ego_traj = ego_traj_full[self._ego_start_index :]
        if len(ego_traj) < 2:
            raise RuntimeError(f"Ego trajectory too short.")

        n_sim_steps = len(ego_traj)
        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = max(1, int(sim_freq // pol_freq))
        self._max_traj_policy_steps = int(np.ceil(n_sim_steps / float(sim_per_policy)))

        x0, y0, ego_speed, lane0 = ego_traj[0]
        ego_xy = np.array([x0, y0], dtype=float)
        
        # Init Heading
        dx0 = ego_traj[1, 0] - ego_traj[0, 0]
        dy0 = ego_traj[1, 1] - ego_traj[0, 1]
        disp = np.hypot(dx0, dy0)
        if disp >= 0.1:
            heading_raw = np.arctan2(dy0, dx0)
        else:
            heading_raw = 0.0

        #ego_lane = clamp_location_ngsim(x0, lane0, self.net, warning=False)
        #target_lane_index = ego_lane.index
    
        ego = EgoVehicle(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=heading_raw,
            control_mode= self.control_mode,
            #target_lane_index=target_lane_index,
            target_speeds= np.array(self.config.get("target_speeds", []), dtype=float) if self.config.get("target_speeds", None) is not None else None,
            lane_change_cooldown_steps= int(self.config.get("lane_change_cooldown_steps", 10)),
            lateral_offset_step= float(self.config.get("lateral_offset_step", EgoVehicle.DEFAULT_LATERAL_OFFSET_STEP)),
            lateral_offset_max= float(self.config.get("lateral_offset_max", EgoVehicle.DEFAULT_LATERAL_OFFSET_MAX)),
        )
        ego.set_ego_dimension(width=ego_wid, length=ego_len)
        self.road.vehicles.append(ego)
        self.vehicle = ego
        
        if self.config.get("expert_test_mode", False):
            self._replay_xy_pol.append(self.vehicle.position.copy())
        
        max_surr = int(self.config.get("max_surrounding", 0))
        if max_surr > 0:
            spawn_surrounding_vehicles(self.trajectory_set, self._ego_start_index, max_surr, self.road)
        
        if expert_mode:
            self._replay_xy_pol.append(self.vehicle.position.copy())
    
    # -------------------------------------------------------------------------
    # REWARDS & TERMINATION
    # -------------------------------------------------------------------------
    def _rewards(self, action: Any) -> dict[str, float]:
        return {"collision_reward": float(self.vehicle.crashed)}

    def _reward(self, action: Any) -> float:
        return 0.0

    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed)

    def _is_truncated(self) -> bool:
        max_steps_cfg = self.config.get("max_episode_steps", None)
        max_steps_traj = getattr(self, "_max_traj_policy_steps", None)
        candidates = [v for v in (max_steps_cfg, max_steps_traj) if v is not None]
        return self.steps >= min(candidates) if candidates else False

    # -------------------------------------------------------------------------
    # EXPERT METRICS & HELPERS
    # -------------------------------------------------------------------------
    def _discrete_expert_action_from_tracker(self, steer_cmd: float, accel_cmd: float, lateral_error: float = 0.0) -> str:
        """
        Map (steer_cmd, accel_cmd, lateral_error) to one of:
        {"SLOWER", "IDLE", "FASTER", "STEER_LEFT", "STEER_RIGHT"}
        """
        v_dead = float(self.config.get("expert_speed_deadband_mps", 0.5))
        s_dead = 4 
        # Using a deadband for lateral error to decide when to steer bias
        lat_dead = 0.15 
        prefer_speed = bool(self.config.get("expert_prefer_speed", False))
        steps = self.steps

        expert_ref_v_pol = self._expert_ref_v_pol
        vehicle_speed = self.vehicle.speed
        
        # Updated to pass lateral_error (for steering bias logic)
        return map_discrete_expert_action(steer_cmd, accel_cmd, 
                                          expert_ref_v_pol, vehicle_speed, 
                                          steps,
                                          lateral_error=lateral_error, 
                                          v_dead=v_dead, 
                                          s_dead=s_dead,
                                          lat_dead=lat_dead, 
                                          prefer_speed=prefer_speed)

    def expert_replay_metrics(self):
        if not hasattr(self, "_expert_ref_xy_pol"):
            raise RuntimeError("No expert reference.")
        if not hasattr(self, "_replay_xy_pol"):
            raise RuntimeError("No replay recorded.")

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
        
        expert_action = None
        expert_action_str = None
        expert_action_idx = None

        expert_test = bool(self.config.get("expert_test_mode", False))
        
        if expert_test:
            

            # 1. Update Tracker & Get Target Data
            pos = self.vehicle.position
            hdg = float(self.vehicle.heading)
            spd = float(self.vehicle.speed)
            
            # UNPACK 5 VALUES: Control Cmds + Indices + TARGET LANE ID
            # (Requires your PurePursuitTracker to return these 5 values)
            steer_cmd, accel_cmd, i_near, i_tgt, expert_target_lane_id = self._tracker.step(pos, hdg, spd)
            #print('computed Expert Lane ID:',expert_target_lane_id)
            #print('computed Steering Amount:',steer_cmd)

            # 2. Calculate Lateral Error (Cross-Track) for "Hybrid Mimic" Drifting
            lateral_error = 0.0
            if hasattr(self, "_expert_ref_xy_pol") and self.steps < len(self._expert_ref_xy_pol):
                expert_pos = self._expert_ref_xy_pol[self.steps]
                dx = expert_pos[0] - pos[0]
                dy = expert_pos[1] - pos[1]
                # Rotate global difference into vehicle's local frame
                # Local Y (Left) = -sin(h)*dx + cos(h)*dy
                lateral_error = -np.sin(hdg) * dx + np.cos(hdg) * dy

            # -----------------------------------------------------------
            # MODE A: CONTINUOUS EXPERT
            # -----------------------------------------------------------
            if self.control_mode == "continuous":
                accel_norm = float(np.clip(accel_cmd / MAX_ACCEL, -1.0, 1.0))
                steer_norm = float(np.clip(steer_cmd / MAX_STEER, -1.0, 1.0))
                expert_action = np.array([accel_norm, steer_norm], dtype=np.float32)
                action = expert_action

            # -----------------------------------------------------------
            # MODE B: DISCRETE EXPERT
            # -----------------------------------------------------------
            elif self.control_mode == "discrete":
                
                # --- PHASE 1: Lane Change Logic (Priority) ---
                # Logic: "If the tracker's lookahead point is in a different lane, switch."
                
                # A. Identify Ego's "Mental" Lane ID
                ego_current_id = 0
                if self.vehicle.target_lane_index:
                    ego_current_id = int(self.vehicle.target_lane_index[2])

                # B. Compare with Expert's Target Lane
                if expert_target_lane_id != -1:        
                    valid_target_id = clamp_lane_id_for_x(self.net, pos[0], expert_target_lane_id)
                    
                    delta = valid_target_id - ego_current_id
                    
                    # C. Issue Command if Valid
                    if delta > 0:
                        # Check if the right lane actually exists physically
                        if self.vehicle._adjacent_lane_index(1) is not None:
                            expert_action_str = "LANE_RIGHT"
                            
                    elif delta < 0:
                        # Check if the left lane actually exists physically
                        if self.vehicle._adjacent_lane_index(-1) is not None:
                            expert_action_str = "LANE_LEFT"

                # --- PHASE 2: Steer/Speed Logic (Secondary) ---
                # If no lane change is commanded, determine lateral/longitudinal adjustments
                #print('Lateral Error:', lateral_error)
                if expert_action_str is None:
                    expert_action_str = self._discrete_expert_action_from_tracker(
                        steer_cmd, accel_cmd, lateral_error=lateral_error
                    )

                # --- PHASE 3: Map String to Index ---
                #print(self.action_type)
                if not hasattr(self, "action_type") or not hasattr(self.action_type, "actions_indexes"):
                     raise RuntimeError("Action type mismatch. Config must use DiscreteSteerMetaAction.")
                
                if expert_action_str not in self.action_type.actions_indexes:
                    print(f"Warning: Expert action {expert_action_str} invalid. Defaulting to IDLE.")
                    expert_action_str = "IDLE"

                expert_action_idx = int(self.action_type.actions_indexes[expert_action_str])
                action = expert_action_idx
                #print("Expert Action:", expert_action_str)
            else:
                raise ValueError(f"Unknown expert_action_mode={self.control_mode!r}")

            # -----------------------------------------------------------
            # LOGGING
            # -----------------------------------------------------------
            if hasattr(self, "_expert_actions_policy") and self._expert_actions_policy is not None:
                if expert_action is not None:
                    self._expert_actions_policy.append(expert_action.copy())
                elif expert_action_str is not None:
                    self._expert_actions_policy.append(expert_action_str)

            if hasattr(self, "_tracker_dbg") and self._tracker_dbg is not None:
                self._tracker_dbg.append((int(i_near), int(i_tgt)))

        # -----------------------------------------------------------
        # EXECUTE SIMULATION STEP
        # -----------------------------------------------------------
        #print("EgoVehicle.act received:", action, type(action), "control_mode=", self.vehicle.control_mode)
        #print("vehicle speed:", self.vehicle.target_speed,"; control_mode: ", self.vehicle.control_mode, "; vehicle action: ", self.vehicle.act)
        obs, reward, terminated, truncated, info = super().step(action)
        if info is None:
            info = {}
        # Populate Info Dict
        info["applied_action"] = action
        if expert_action is not None:
            info["expert_action_continuous"] = expert_action.copy()
        if expert_action_str is not None:
            info["expert_action_discrete"] = expert_action_str
            info["expert_action_discrete_idx"] = expert_action_idx

        # Record Trajectory for ADE/FDE metrics
        if expert_test:
            self._replay_xy_pol.append(self.vehicle.position.copy())

        if not truncated and self._is_truncated():
            truncated = True

        return obs, reward, terminated, truncated, info