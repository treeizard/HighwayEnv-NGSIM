# Modified by: Yide Tao (yide.tao@monash.edu)
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road

from highway_env.ngsim_utils.helper_ngsim import (
    clamp_lane_id_for_x,
    get_ego_dimensions,
    load_ego_trajectory,
    setup_expert_tracker,
)
from highway_env.ngsim_utils.obs_vehicle import spawn_surrounding_vehicles
from highway_env.ngsim_utils.gen_road import create_ngsim_101_road
from highway_env.ngsim_utils.trajectory_to_action import (
    PurePursuitTracker,
    map_discrete_expert_action,
)
from highway_env.ngsim_utils.trajectory_gen import first_valid_index
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle


# Constants
f2m_conv = 3.281
MAX_ACCEL = 5.0
MAX_STEER = np.pi / 4


def _deep_update(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base and return the merged dict.
    """
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


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
                # Will be normalized from action_mode inside __init__
                "action": {"type": "ContinuousAction"},
                "action_mode": "discrete",  # "continuous" or "discrete"
                "action_config": {
                    "lateral_offset_step": 0.10,
                    "lateral_offset_max": 1.50,
                    "lane_change_cooldown_steps": 10,
                    "lane_change_commit_hyst_steps": 2,
                    "target_speeds": list(np.arange(0.0, 35.0 + 1e-6, 2.0)),
                },
                "expert_v": {
                    "expert_speed_deadband_mps": 0.5,
                    "expert_steer_deadband_rad": 0.05,
                    "expert_one_action_per_step": True,
                    "expert_prefer_speed": False,
                },
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "max_episode_steps": 300,
                "ego_vehicle_ID": None,
                "simulation_period": None,
                "episode_root": "highway_env/data/processed_10s",
                "max_surrounding": 80,
                "show_trajectories": True,
                "seed": None,
                "expert_test_mode": False,
            }
        )
        return config

    # -------------------------------------------------------------------------
    # INIT / DT
    # -------------------------------------------------------------------------
    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        cfg = self.default_config() if config is None else _deep_update(self.default_config(), config)

        # Normalize control/action mode before AbstractEnv constructs action_type
        self.control_mode = str(cfg.get("action_mode", "continuous")).lower()
        if self.control_mode == "continuous":
            cfg["action"] = {"type": "ContinuousAction"}
        elif self.control_mode == "discrete":
            cfg["action"] = {"type": "DiscreteSteerMetaAction"}
        else:
            raise ValueError(f"Unknown action_mode={self.control_mode!r}")

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
        self._prebuilt_dir = os.path.join(cfg["episode_root"], cfg["scene"], "prebuilt")
        veh_ids_path = os.path.join(self._prebuilt_dir, "veh_ids_train.npy")
        traj_path = os.path.join(self._prebuilt_dir, "trajectory_train.npy")

        self._valid_ids_by_episode = np.load(veh_ids_path, allow_pickle=True).item()
        self._traj_all_by_episode = np.load(traj_path, allow_pickle=True).item()
        self._episodes = sorted(self._traj_all_by_episode.keys())

        super().__init__(config=cfg, render_mode=render_mode)

    @property
    def dt(self) -> float:
        return 1.0 / float(self.config["simulation_frequency"])

    @property
    def action_cfg(self) -> dict:
        return self.config.get("action_config", {})

    @property
    def expert_cfg(self) -> dict:
        return self.config.get("expert_v", {})

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
            ref_xy_pol, ref_v_pol, lane_pol, start_idx = setup_expert_tracker(
                self.net, ego_traj_full, ego_len, self.config
            )
            self._expert_ref_xy_pol = ref_xy_pol
            self._expert_ref_v_pol = ref_v_pol
            self._expert_ref_lane_pol = lane_pol - 1
            self._ego_start_index = start_idx

            self._tracker = PurePursuitTracker(
                ref_xy=self._expert_ref_xy_pol,
                ref_v=self._expert_ref_v_pol,
                ref_lanes=self._expert_ref_lane_pol,
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
            raise RuntimeError("Ego trajectory too short.")

        n_sim_steps = len(ego_traj)
        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = max(1, int(sim_freq // pol_freq))
        self._max_traj_policy_steps = int(np.ceil(n_sim_steps / float(sim_per_policy)))

        x0, y0, ego_speed, lane0 = ego_traj[0]
        ego_xy = np.array([x0, y0], dtype=float)

        # Initial heading from first displacement
        dx0 = ego_traj[1, 0] - ego_traj[0, 0]
        dy0 = ego_traj[1, 1] - ego_traj[0, 1]
        disp = np.hypot(dx0, dy0)
        heading_raw = float(np.arctan2(dy0, dx0)) if disp >= 0.1 else 0.0

        target_speeds_cfg = self.action_cfg.get("target_speeds", None)
        target_speeds = (
            np.array(target_speeds_cfg, dtype=float) if target_speeds_cfg is not None else None
        )

        ego = EgoVehicle(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=heading_raw,
            control_mode=self.control_mode,
            target_speeds=target_speeds,
            lane_change_cooldown_steps=int(
                self.action_cfg.get("lane_change_cooldown_steps", 10)
            ),
            lateral_offset_step=float(
                self.action_cfg.get(
                    "lateral_offset_step",
                    EgoVehicle.DEFAULT_LATERAL_OFFSET_STEP,
                )
            ),
            lateral_offset_max=float(
                self.action_cfg.get(
                    "lateral_offset_max",
                    EgoVehicle.DEFAULT_LATERAL_OFFSET_MAX,
                )
            ),
        )
        ego.set_ego_dimension(width=ego_wid, length=ego_len)
        self.road.vehicles.append(ego)
        self.vehicle = ego

        if self.config.get("expert_test_mode", False):
            self._replay_xy_pol.append(self.vehicle.position.copy())

        max_surr = int(self.config.get("max_surrounding", 0))
        if max_surr > 0:
            spawn_surrounding_vehicles(
                self.trajectory_set,
                self._ego_start_index,
                max_surr,
                self.road,
            )

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
    def _discrete_expert_action_from_tracker(
        self,
        steer_cmd: float,
        accel_cmd: float,
        lateral_error: float = 0.0,
    ) -> str:
        """
        Map (steer_cmd, accel_cmd, lateral_error) to one of:
        {"SLOWER", "IDLE", "FASTER", "STEER_LEFT", "STEER_RIGHT"}
        """
        v_dead = float(self.expert_cfg.get("expert_speed_deadband_mps", 0.5))
        s_dead = float(self.expert_cfg.get("expert_steer_deadband_rad", 0.05))
        lat_dead = 0.15
        prefer_speed = bool(self.expert_cfg.get("expert_prefer_speed", False))
        steps = self.steps

        expert_ref_v_pol = self._expert_ref_v_pol
        vehicle_speed = self.vehicle.speed

        return map_discrete_expert_action(
            steer_cmd,
            accel_cmd,
            expert_ref_v_pol,
            vehicle_speed,
            steps,
            lateral_error=lateral_error,
            v_dead=v_dead,
            s_dead=s_dead,
            lat_dead=lat_dead,
            prefer_speed=prefer_speed,
        )

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
            pos = self.vehicle.position
            hdg = float(self.vehicle.heading)
            spd = float(self.vehicle.speed)

            steer_cmd, accel_cmd, i_near, i_tgt, expert_target_lane_id = self._tracker.step(
                pos, hdg, spd
            )

            lateral_error = 0.0
            if hasattr(self, "_expert_ref_xy_pol") and self.steps < len(self._expert_ref_xy_pol):
                expert_pos = self._expert_ref_xy_pol[self.steps]
                dx = expert_pos[0] - pos[0]
                dy = expert_pos[1] - pos[1]
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
                ego_current_id = 0
                if getattr(self.vehicle, "target_lane_index", None):
                    ego_current_id = int(self.vehicle.target_lane_index[2])

                if expert_target_lane_id != -1:
                    valid_target_id = clamp_lane_id_for_x(self.net, pos[0], expert_target_lane_id)
                    delta = valid_target_id - ego_current_id

                    if delta > 0:
                        if self.vehicle._adjacent_lane_index(1) is not None:
                            expert_action_str = "LANE_RIGHT"
                    elif delta < 0:
                        if self.vehicle._adjacent_lane_index(-1) is not None:
                            expert_action_str = "LANE_LEFT"

                if expert_action_str is None:
                    expert_action_str = self._discrete_expert_action_from_tracker(
                        steer_cmd, accel_cmd, lateral_error=lateral_error
                    )

                if not hasattr(self, "action_type") or not hasattr(self.action_type, "actions_indexes"):
                    raise RuntimeError(
                        "Action type mismatch. Config must use DiscreteSteerMetaAction."
                    )

                if expert_action_str not in self.action_type.actions_indexes:
                    print(f"Warning: Expert action {expert_action_str} invalid. Defaulting to IDLE.")
                    expert_action_str = "IDLE"

                expert_action_idx = int(self.action_type.actions_indexes[expert_action_str])
                action = expert_action_idx

            else:
                raise ValueError(f"Unknown action_mode={self.control_mode!r}")

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