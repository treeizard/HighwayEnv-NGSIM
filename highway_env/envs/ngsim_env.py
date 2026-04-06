# Modified by: Yide Tao (yide.tao@monash.edu)
from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.ngsim_utils.ngsim_expert_mixin import NGSimExpertMixin
from highway_env.road.road import Road

from highway_env.ngsim_utils.helper_ngsim import (
    get_ego_dimensions,
    load_ego_trajectory,
    setup_expert_tracker,
    target_lane_index_from_lane_id,
)
from highway_env.ngsim_utils.obs_vehicle import spawn_surrounding_vehicles
from highway_env.ngsim_utils.gen_road import create_ngsim_101_road, create_japanese_road
from highway_env.ngsim_utils.constants import FEET_PER_METER, MAX_STEER
from highway_env.ngsim_utils.trajectory_to_action import (
    PurePursuitTracker,
)
from highway_env.ngsim_utils.trajectory_gen import first_valid_index
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle


logger = logging.getLogger(__name__)
ROAD_BUILDERS = {
    "us-101": create_ngsim_101_road,
    "japanese": create_japanese_road,
}


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


class NGSimEnv(NGSimExpertMixin, AbstractEnv):
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
                    "target_speeds": list(np.arange(0.0, 35.0 + 1e-6, 2.0)),
                },
                "expert_v": {
                    "planner_horizon": 2,
                    "planner_branching": 5,
                    "planner_position_weight": 3.0,
                    "planner_heading_weight": 0.5,
                    "planner_speed_weight": 0.2,
                    "planner_clearance_weight": 8.0,
                    "planner_collision_cost": 1e6,
                    "planner_action_change_weight": 0.05,
                },
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "max_episode_steps": 300,
                "ego_vehicle_ID": None,
                "simulation_period": None,
                "episode_root": "highway_env/data/processed_10s",
                "max_surrounding": "all",
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
        raw_config = config or {}
        cfg = self.default_config() if config is None else _deep_update(self.default_config(), config)

        # Normalize control/action mode before AbstractEnv constructs action_type
        self.control_mode = self._normalize_action_mode(cfg, raw_config)
        self.scene = str(cfg["scene"])

        self._episodes: list[str] = []
        self._valid_ids_by_episode: dict[str, np.ndarray] = {}
        self.ego_id: int | None = None

        # Expert debug
        self._max_traj_policy_steps: int | None = None
        self._replay_xy_pol: list[np.ndarray] = []

        self._load_prebuilt_data(cfg["episode_root"], self.scene)

        super().__init__(config=cfg, render_mode=render_mode)

    def _normalize_action_mode(self, cfg: dict, raw_config: dict | None = None) -> str:
        raw_config = raw_config or {}
        if "action_mode" in raw_config:
            control_mode = str(raw_config["action_mode"]).lower()
        else:
            action_type = str(raw_config.get("action", {}).get("type", cfg.get("action", {}).get("type", ""))).lower()
            if action_type == "continuousaction":
                control_mode = "continuous"
            elif action_type == "discretesteermetaaction":
                control_mode = "discrete"
            else:
                control_mode = str(cfg.get("action_mode", "continuous")).lower()
        action_types = {
            "continuous": "ContinuousAction",
            "discrete": "DiscreteSteerMetaAction",
        }
        if control_mode not in action_types:
            raise ValueError(f"Unknown action_mode={control_mode!r}")
        cfg["action"] = {"type": action_types[control_mode]}
        return control_mode

    def _load_prebuilt_data(self, episode_root: str, scene: str) -> None:
        prebuilt_dir = os.path.join(episode_root, scene, "prebuilt")
        veh_ids_path = os.path.join(prebuilt_dir, "veh_ids_train.npy")
        traj_path = os.path.join(prebuilt_dir, "trajectory_train.npy")

        self._prebuilt_dir = prebuilt_dir
        self._valid_ids_by_episode = np.load(veh_ids_path, allow_pickle=True).item()
        self._traj_all_by_episode = np.load(traj_path, allow_pickle=True).item()
        self._episodes = sorted(self._traj_all_by_episode.keys())

    @property
    def dt(self) -> float:
        return 1.0 / float(self.config["simulation_frequency"])

    @property
    def action_cfg(self) -> dict:
        return self.config.get("action_config", {})

    @property
    def expert_cfg(self) -> dict:
        return self.config.get("expert_v", {})

    @property
    def expert_test_mode(self) -> bool:
        return bool(self.config.get("expert_test_mode", False))

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

        if self.expert_test_mode:
            self._replay_xy_pol = [self.vehicle.position.copy()]

    def _prune_removed_vehicles(self) -> None:
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if not getattr(vehicle, "remove_from_road", False)
        ]

    def _simulate(self, action: Action | None = None) -> None:
        """Run simulation frames and prune replay vehicles that have despawned."""
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for frame in range(frames):
            if (
                action is not None
                and not self.config["manual_control"]
                and self.steps
                % int(
                    self.config["simulation_frequency"]
                    // self.config["policy_frequency"]
                )
                == 0
            ):
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self._prune_removed_vehicles()
            self.steps += 1

            if frame < frames - 1:
                self._automatic_rendering()

        self.enable_auto_render = False

    # -------------------------------------------------------------------------
    # LOAD TRAJECTORY
    # -------------------------------------------------------------------------
    def _load_trajectory(self):
        sim_period = self.config.get("simulation_period", None)
        explicit_ego_id = self.config.get("ego_vehicle_ID", None)

        self.episode_name = self._select_episode_name(sim_period)
        valid_ids = self._valid_ids_by_episode[self.episode_name]
        self.ego_id = self._select_ego_id(valid_ids, explicit_ego_id)
        self.trajectory_set = self._build_trajectory_set(self.episode_name, self.ego_id)

        logger.info("Loaded episode=%s ego_id=%s", self.episode_name, self.ego_id)

    def _select_episode_name(self, sim_period: Any) -> str:
        if isinstance(sim_period, dict) and "episode_name" in sim_period:
            episode_name = sim_period["episode_name"]
            if episode_name not in self._traj_all_by_episode:
                raise ValueError(f"Episode {episode_name} not found.")
            return str(episode_name)
        return str(self.np_random.choice(self._episodes))

    def _select_ego_id(self, valid_ids: np.ndarray, explicit_ego_id: int | None) -> int:
        if explicit_ego_id is None:
            return int(self.np_random.choice(valid_ids))
        if explicit_ego_id not in valid_ids:
            raise ValueError(f"Ego ID {explicit_ego_id} not in {self.episode_name}")
        return int(explicit_ego_id)

    def _build_trajectory_set(self, episode_name: str, ego_id: int) -> dict[Any, Any]:
        traj_all = self._traj_all_by_episode[episode_name]
        return {
            "ego": traj_all[ego_id],
            **{vid: meta for vid, meta in traj_all.items() if vid != ego_id},
        }

    
    # -------------------------------------------------------------------------
    # ROAD + VEHICLES + Test Mode
    # -------------------------------------------------------------------------
    def _create_road(self):
        builder = ROAD_BUILDERS.get(self.scene)
        if builder is None:
            raise ValueError(f"Unsupported scene={self.scene!r}")
        net = builder()
        self.net = net
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self):
        ego_rec = self.trajectory_set["ego"]
        ego_traj_full = load_ego_trajectory(ego_rec, self.scene)
        ego_len, ego_wid = get_ego_dimensions(ego_rec, FEET_PER_METER, self.scene)
        self._replay_xy_pol.clear()

        ego_traj = self._prepare_ego_trajectory(ego_rec, ego_traj_full, ego_len)
        ego = self._build_ego_vehicle(ego_traj, ego_len, ego_wid)
        self.road.vehicles.append(ego)
        self.vehicle = ego

        if self.expert_test_mode:
            self._replay_xy_pol.append(self.vehicle.position.copy())

        self._spawn_surrounding_vehicles()

        if self.expert_test_mode:
            self._replay_xy_pol.append(self.vehicle.position.copy())

    def _prepare_ego_trajectory(
        self,
        ego_rec: dict[str, Any],
        ego_traj_full: np.ndarray,
        ego_len: float,
    ) -> np.ndarray:
        if self.expert_test_mode:
            self._setup_expert_tracker(ego_traj_full, ego_len)
        else:
            ego_start_i = first_valid_index(ego_rec["trajectory"])
            if ego_start_i is None:
                raise RuntimeError("Ego trajectory contains no valid motion data.")
            self._ego_start_index = int(ego_start_i)

        ego_traj = ego_traj_full[self._ego_start_index :]
        if len(ego_traj) < 2:
            raise RuntimeError("Ego trajectory too short.")

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = max(1, int(sim_freq // pol_freq))
        self._max_traj_policy_steps = int(np.ceil(len(ego_traj) / float(sim_per_policy)))
        return ego_traj

    def _setup_expert_tracker(self, ego_traj_full: np.ndarray, ego_len: float) -> None:
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

    def _build_ego_vehicle(
        self,
        ego_traj: np.ndarray,
        ego_len: float,
        ego_wid: float,
    ) -> EgoVehicle:
        x0, y0, ego_speed, lane0 = ego_traj[0]
        ego_xy = np.array([x0, y0], dtype=float)
        heading_raw = self._estimate_initial_heading(ego_traj)
        target_speeds = self._target_speeds_for_trajectory(ego_traj)

        ego = EgoVehicle(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=heading_raw,
            control_mode=self.control_mode,
            target_speeds=target_speeds,
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

        mapped_lane_index = target_lane_index_from_lane_id(
            self.road.network, self.scene, x0, int(lane0)
        )
        if mapped_lane_index is not None:
            ego.target_lane_index = mapped_lane_index
            ego.lane_index = mapped_lane_index
            ego.lane = self.road.network.get_lane(mapped_lane_index)
            s0, r0 = ego.lane.local_coordinates(ego.position)
            if not ego.lane.on_lane(ego.position, s0, r0):
                lane_margin = max(0.1, 0.5 * ego_wid)
                r0 = float(
                    np.clip(
                        r0,
                        -ego.lane.width_at(s0) / 2.0 + lane_margin,
                        ego.lane.width_at(s0) / 2.0 - lane_margin,
                    )
                )
                ego.position = ego.lane.position(s0, r0)
            ego.heading = float(ego.lane.heading_at(s0))

        return ego

    def _target_speeds_for_trajectory(self, ego_traj: np.ndarray) -> np.ndarray | None:
        if self.control_mode != "discrete":
            target_speeds_cfg = self.action_cfg.get("target_speeds", None)
            return np.array(target_speeds_cfg, dtype=float) if target_speeds_cfg is not None else None

        target_speeds_cfg = self.action_cfg.get("target_speeds", None)
        if target_speeds_cfg is None:
            base = np.array(EgoVehicle.DEFAULT_TARGET_SPEEDS, dtype=float)
        else:
            base = np.array(target_speeds_cfg, dtype=float)

        if base.ndim != 1 or base.size < 2 or not np.all(np.isfinite(base)):
            return np.array(EgoVehicle.DEFAULT_TARGET_SPEEDS, dtype=float)

        diffs = np.diff(base)
        positive_diffs = diffs[diffs > 1e-6]
        step = float(np.median(positive_diffs)) if positive_diffs.size else 2.0
        step = max(0.5, step)

        valid_speeds = ego_traj[:, 2]
        valid_speeds = valid_speeds[np.isfinite(valid_speeds) & (valid_speeds >= 0.0)]
        if valid_speeds.size == 0:
            return base

        max_speed = max(float(base[-1]), float(np.max(valid_speeds)) + step)
        count = int(np.ceil(max_speed / step)) + 1
        adaptive = np.arange(count, dtype=float) * step
        return adaptive

    def _estimate_initial_heading(self, ego_traj: np.ndarray) -> float:
        dx0 = ego_traj[1, 0] - ego_traj[0, 0]
        dy0 = ego_traj[1, 1] - ego_traj[0, 1]
        disp = np.hypot(dx0, dy0)
        return float(np.arctan2(dy0, dx0)) if disp >= 0.1 else 0.0

    def _spawn_surrounding_vehicles(self) -> None:
        max_surr_raw = self.config.get("max_surrounding", 0)
        spawn_all = max_surr_raw == "all"
        max_surr = None if spawn_all else int(max_surr_raw)
        if not spawn_all and max_surr <= 0:
            return

        spawn_surrounding_vehicles(
            self.trajectory_set,
            self._ego_start_index,
            max_surr,
            self.road,
            scene=self.scene,
        )
    
    def visualize(
        self,
        steps: int | None = None,
        width: int = 1200,
        height: int = 600,
        scaling: float = 5.5,
        mode: str = "all"
    ):
        """
        Visualize the environment.

        Args:
            steps: maximum rollout steps to render. If None, run until terminated/truncated.
            width: render window width in pixels.
            height: render window height in pixels.
            scaling: zoom factor for rendering.
            mode:
                - "road": render only the road layout
                - "all": reset env, create vehicles, and rollout
        Returns:
            Last observation if a rollout is executed, else None.
        """
        # --- set rendering config ---
        self.config["screen_width"] = width
        self.config["screen_height"] = height
        self.config["scaling"] = scaling

        # Ensure render mode is compatible with display
        if self.render_mode is None:
            self.render_mode = "human"

        if mode == "road":
            # Build an empty road scene and render once
            self._create_road()

            # Make sure road-side state expected by renderer exists
            if not hasattr(self, "vehicle"):
                self.vehicle = None

            self.render()
            return None

        elif mode == "all":
            # --- reset env ---
            reset_out = self.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, info = reset_out
            else:
                obs = reset_out
                info = {}

            done = False
            step_count = 0

            while not done:
                # no-op action
                if self.control_mode == "continuous":
                    action = np.zeros(self.action_space.shape, dtype=np.float32)
                else:
                    # for discrete meta-action env, IDLE is the proper no-op if available
                    if hasattr(self, "action_type") and hasattr(self.action_type, "actions_indexes"):
                        action = self.action_type.actions_indexes.get("IDLE", 0)
                    else:
                        action = 0

                obs, reward, terminated, truncated, info = self.step(action)

                # render frame
                self.render()

                done = terminated or truncated
                step_count += 1

                if steps is not None and step_count >= steps:
                    break

            return obs

        else:
            raise ValueError(f"Unknown mode={mode!r}. Expected 'road' or 'all'.")
            


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
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action: Action):
        expert_action = None
        expert_action_str = None
        expert_action_idx = None

        if self.expert_test_mode:
            action, expert_action, expert_action_str, expert_action_idx = self._resolve_expert_action()

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

        if self.expert_test_mode:
            self._replay_xy_pol.append(self.vehicle.position.copy())

        return obs, reward, terminated, truncated, info
