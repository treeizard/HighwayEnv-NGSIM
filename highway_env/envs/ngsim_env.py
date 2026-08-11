"""Define the ngsim env driving environment."""

# Developed by: Yide Tao (yide.tao@monash.edu)
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

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.ngsim_utils.expert.ngsim_expert_mixin import NGSimExpertMixin
from highway_env.road.road import Road

from highway_env.ngsim_utils.core.config import (
    deep_update,
    interaction_metric_targets_from_idm,
    normalize_action_mode,
    resolve_idm_parameters,
)
from highway_env.ngsim_utils.vehicles.ego_factory import build_ego_vehicle
from highway_env.ngsim_utils.data.episode_selection import (
    build_trajectory_set,
    select_ego_ids,
    select_episode_name,
)
from highway_env.ngsim_utils.data.ego_trajectory import (
    get_ego_dimensions,
    load_ego_trajectory,
    setup_expert_tracker,
)
from highway_env.ngsim_utils.road.lane_mapping import (
    heading_from_trajectory_row,
    resolve_target_lane_index_from_row,
)
from highway_env.ngsim_utils.vehicles.replay import (
    road_entity_conflicts_at_pose,
    spawn_surrounding_vehicles,
)
from highway_env.ngsim_utils.road.gen_road import create_ngsim_101_road, create_japanese_road
from highway_env.ngsim_utils.core.constants import (
    ACCELERATION_RANGE,
    FEET_PER_METER,
    MAX_STEER,
)
from highway_env.ngsim_utils.data.prebuilt import load_prebuilt_data
from highway_env.ngsim_utils.expert.trajectory_to_action import (
    PurePursuitTracker,
)
from highway_env.ngsim_utils.data.trajectory_gen import (
    common_first_valid_index,
    longest_continuous_active_span_bounds,
    trajectory_row_is_active,
    trajectory_step_speed_mps,
)
from highway_env.ngsim_utils.vehicles.ego import EgoVehicle


logger = logging.getLogger(__name__)
ROAD_BUILDERS = {
    "us-101": create_ngsim_101_road,
    "japanese": create_japanese_road,
}


class NGSimEnv(NGSimExpertMixin, AbstractEnv):
    _PREBUILT_CACHE: dict[
        tuple[str, str, str, float],
        tuple[dict[str, np.ndarray], dict[str, dict[Any, Any]], list[str]],
    ] = {}
    _NETWORK_CACHE: dict[tuple[str, str], Any] = {}
    _PROCESSED_TRAJECTORY_CACHE: dict[tuple[str, str, str, int], np.ndarray] = {}
    _EXPERT_REFERENCE_CACHE: dict[
        tuple[str, str, str, int, int, float],
        tuple[np.ndarray, np.ndarray, np.ndarray, int],
    ] = {}

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
                "action": {
                    "type": "ContinuousAction",
                    "acceleration_range": list(ACCELERATION_RANGE),
                    "zero_centered_acceleration": True,
                },
                "action_mode": "discrete",  # "continuous", "discrete", or "teleport"
                "action_config": {
                    "lateral_offset_step": 0.10,
                    "lateral_offset_max": 1.50,
                    "target_speeds": list(np.arange(0.0, 35.0 + 1e-6, 2.0)),
                },
                # The expert mode planner variables
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
                # Simulation Parameters
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "max_episode_steps": 300,
                # Replay parameters
                "ego_vehicle_ID": None,
                "simulation_period": None,
                # Raw data selections
                "episode_root": "data/highway_env/processed_20s",
                "prebuilt_split": "train",
                # When omitted, a run-owned ROAD_GEOMETRY.json beside the
                # Japanese prebuilt arrays is discovered automatically.
                "japanese_road_geometry": None,
                # Quality of life/ debugging
                "control_all_vehicles": False,
                "max_surrounding": "all",
                "show_trajectories": True,
                # Exact simulator acceleration. Legacy modes remain the default
                # so existing evaluation and interpretability jobs are unchanged.
                "road_query_mode": "legacy",  # "legacy" or "spatial"
                "road_query_cell_size": 25.0,
                "collision_check_mode": "legacy",  # "legacy" or "broadphase"
                "collision_broadphase_cell_size": 12.0,
                "collision_broadphase_min_entities": 32,
                "record_replay_diagnostics": True,
                "sensor_road_edge_mode": "per_vehicle",  # "per_vehicle" or "batched"
                "seed": None,
                "expert_test_mode": False,
                "discrete_expert_policy": "planner",
                "percentage_controlled_vehicles": 0.1,
                "clip_controlled_vehicles_to_available": True,
                "terminate_when_all_controlled_crashed": True,
                "truncate_to_trajectory_length": False,  # allow for replay
                "scene_dataset_collection_mode": False,
                # A causal collector may supply actions directly without
                # constructing the privileged trajectory tracker. The flag is
                # valid only for scene collection and affects activation
                # bookkeeping, never the actor observation.
                "scene_collection_external_controller": False,
                "disable_controlled_vehicle_collisions": False,
                "crash_controlled_vehicles_offroad": True,
                "complete_controlled_vehicles_at_road_end": True,
                "road_end_completion_lateral_margin": 0.25,
                "disable_scene_collection_spawn_safety": False,
                # Corrected corpus runs fail closed on negative speeds and use
                # only previous->current motion for actor-visible heading.
                # Kept opt-in so historical US checkpoints remain reproducible.
                "source_preserving_trajectory_state": False,
                "allow_idm": True,
                "controlled_vehicle_min_occupancy": 0.8,
                "scene_collection_min_occupancy_steps": None,
                "idm_parameters": None,
                "enable_interaction_metrics": False,
                "interaction_ttc_target": 0.0,
                "interaction_ttc_margin": 0.75,
                "interaction_ttc_floor": 0.0,
                "interaction_gap_target": 0.0,
                "interaction_gap_floor": 0.0,
                "debug_idm_handover": False,
                "debug_idm_handover_ids": None,
            }
        )
        return config

    # -------------------------------------------------------------------------
    # Initialize the environment, load prebuilt data, and normalize config options
    # -------------------------------------------------------------------------
    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        raw_config = config or {}
        cfg = self.default_config() if config is None else deep_update(self.default_config(), config)

        # Normalize control/action mode before AbstractEnv constructs action_type
        self.control_mode = normalize_action_mode(cfg, raw_config)
        self.scene = str(cfg["scene"])
        self.idm_parameters = resolve_idm_parameters(self.scene, cfg)
        cfg["idm_parameters"] = deepcopy(self.idm_parameters)

        self._episodes: list[str] = []
        self._valid_ids_by_episode: dict[str, np.ndarray] = {}
        self.ego_id: int | None = None
        self.ego_ids: list[int] = []
        self._ego_start_indices: dict[int, int] = {}
        self._expert_state_by_vehicle_id: dict[int, dict[str, Any]] = {}

        # Expert debug
        self._max_traj_policy_steps: int | None = None
        self._replay_xy_pol: list[np.ndarray] = []
        self._frames_per_action: int = 1

        (
            self._prebuilt_dir,
            self._valid_ids_by_episode,
            self._traj_all_by_episode,
            self._episodes,
        ) = load_prebuilt_data(
            cfg["episode_root"],
            self.scene,
            str(cfg.get("prebuilt_split", "train")),
            min_occupancy=float(cfg.get("controlled_vehicle_min_occupancy", 0.8)),
            cache=self._PREBUILT_CACHE,
        )

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

    @property
    def expert_test_mode(self) -> bool:
        return bool(self.config.get("expert_test_mode", False))

    @property
    def scene_dataset_collection_mode(self) -> bool:
        return bool(self.config.get("scene_dataset_collection_mode", False))

    def _policy_controlled_vehicles(self) -> list[EgoVehicle]:
        """
        Return the vehicles that are directly controlled by the action interface.

        With a single-agent action space only ``self.vehicle`` receives the user or
        policy action, even if additional ego vehicles are spawned for interaction.
        """
        if self.config.get("action", {}).get("type") == "MultiAgentAction":
            return list(self.controlled_vehicles)
        return [self.vehicle] if self.vehicle is not None else []

    def _termination_vehicles(self) -> list[EgoVehicle]:
        """
        Vehicles that participate in the episode termination condition.

        For multi-vehicle training we want all spawned controlled vehicles to be
        considered together, even when the action interface is still single-agent.
        """
        if self.scene_dataset_collection_mode:
            return []
        if len(self.controlled_vehicles) > 1:
            vehicles = list(self.controlled_vehicles)
        else:
            vehicles = self._policy_controlled_vehicles()
        return [vehicle for vehicle in vehicles if vehicle is not None]

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------
    def _reset(self):
        self.steps = 0
        self._frames_per_action = max(
            1,
            int(self.config["simulation_frequency"] // self.config["policy_frequency"]),
        )

        seed = self.config.get("seed", None)
        if seed is not None and hasattr(self, "seed"):
            self.seed(seed)

        self._load_trajectory()
        self._create_road()
        self._create_vehicles()

        if self.expert_test_mode and self.vehicle is not None:
            expert_state = self._expert_state_by_vehicle_id.get(int(self.vehicle.vehicle_ID))
            if expert_state is not None:
                self._replay_xy_pol = list(expert_state["replay_xy"])

    def _prune_removed_vehicles(self) -> None:
        if not any(getattr(vehicle, "remove_from_road", False) for vehicle in self.road.vehicles):
            return
        self.road.vehicles = [
            vehicle for vehicle in self.road.vehicles if not getattr(vehicle, "remove_from_road", False)
        ]

    def _simulate(self, action: Action | None = None) -> None:
        """Run simulation frames and prune replay vehicles that have despawned."""
        frames = self._frames_per_action
        dt = 1 / self.config["simulation_frequency"]
        for frame in range(frames):
            if action is not None and not self.config["manual_control"] and self.steps % frames == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(dt)
            self._complete_road_end_controlled_vehicles()
            self._crash_offroad_controlled_vehicles()
            self._prune_removed_vehicles()
            self.steps += 1

            if frame < frames - 1:
                self._automatic_rendering()

        self.enable_auto_render = False

    def _complete_road_end_controlled_vehicles(self) -> None:
        if (
            not bool(self.config.get("complete_controlled_vehicles_at_road_end", True))
            or self.scene_dataset_collection_mode
        ):
            return
        for vehicle in list(getattr(self, "controlled_vehicles", ())):
            if vehicle is None or bool(getattr(vehicle, "crashed", False)):
                continue
            if bool(getattr(vehicle, "completed", False)):
                continue
            if self._vehicle_reached_terminal_road_end(vehicle):
                vehicle.completed = True
                vehicle.reached_road_end = True
                vehicle.remove_from_road = True
                vehicle.collidable = False
                vehicle.check_collisions = False

    def _vehicle_reached_terminal_road_end(self, vehicle: EgoVehicle) -> bool:
        lane_indexes = [
            getattr(vehicle, "target_lane_index", None),
            getattr(vehicle, "lane_index", None),
        ]
        seen: set[tuple[str, str, int]] = set()
        for lane_index in lane_indexes:
            if lane_index is None or lane_index in seen:
                continue
            seen.add(lane_index)
            if self._lane_has_downstream_drivable_road(lane_index):
                continue
            try:
                lane = self.road.network.get_lane(lane_index)
                longitudinal, lateral = lane.local_coordinates(vehicle.position)
                width = float(lane.width_at(longitudinal))
            except Exception:
                continue

            length = float(getattr(vehicle, "LENGTH", lane.VEHICLE_LENGTH))
            front_bumper_reached_end = longitudinal >= float(lane.length) - 0.5 * length
            lateral_margin = float(self.config.get("road_end_completion_lateral_margin", 0.25))
            laterally_on_lane = abs(float(lateral)) <= width / 2.0 + lateral_margin
            if front_bumper_reached_end and laterally_on_lane:
                return True
        return False

    def _lane_has_downstream_drivable_road(self, lane_index: tuple[str, str, int]) -> bool:
        try:
            downstream = self.road.network.graph.get(lane_index[1], {})
        except Exception:
            return False
        for lanes in downstream.values():
            for lane in lanes:
                if not bool(getattr(lane, "forbidden", False)):
                    return True
        return False

    def _crash_offroad_controlled_vehicles(self) -> None:
        if (
            not bool(self.config.get("crash_controlled_vehicles_offroad", True))
            or self.scene_dataset_collection_mode
            or self.expert_test_mode
        ):
            return
        for vehicle in list(getattr(self, "controlled_vehicles", ())):
            if vehicle is None or bool(getattr(vehicle, "crashed", False)):
                continue
            if bool(getattr(vehicle, "completed", False)):
                continue
            try:
                is_on_road = bool(getattr(vehicle, "on_road", True))
            except Exception:
                is_on_road = False
            if not is_on_road:
                vehicle.crashed = True

    # -------------------------------------------------------------------------
    # LOAD TRAJECTORY
    # -------------------------------------------------------------------------
    def _load_trajectory(self):
        sim_period = self.config.get("simulation_period", None)
        explicit_ego_id = self.config.get("ego_vehicle_ID", None)

        self.episode_name = select_episode_name(
            sim_period,
            self._traj_all_by_episode,
            self._episodes,
            self.np_random,
        )
        valid_ids = self._valid_ids_by_episode[self.episode_name]
        self.ego_ids = select_ego_ids(
            valid_ids,
            explicit_ego_id,
            percentage_controlled_vehicles=self.config["percentage_controlled_vehicles"],
            np_random=self.np_random,
            episode_name=self.episode_name,
            control_all_vehicles=bool(self.config.get("control_all_vehicles", False)),
            clip_to_available=bool(self.config.get("clip_controlled_vehicles_to_available", True)),
        )
        self.ego_id = self.ego_ids[0] if self.ego_ids else None
        self.trajectory_set = build_trajectory_set(
            self._traj_all_by_episode,
            self.episode_name,
            self.ego_ids,
        )

        logger.info("Loaded episode=%s ego_ids=%s", self.episode_name, self.ego_ids)

    # -------------------------------------------------------------------------
    # ROAD + VEHICLES + Test Mode
    # -------------------------------------------------------------------------
    def _create_road(self):
        builder = ROAD_BUILDERS.get(self.scene)
        if builder is None:
            raise ValueError(f"Unsupported scene={self.scene!r}")
        query_mode = str(self.config.get("road_query_mode", "legacy")).lower()
        if query_mode not in {"legacy", "spatial", "optimized"}:
            raise ValueError("road_query_mode must be one of: legacy, spatial, optimized")
        collision_mode = str(self.config.get("collision_check_mode", "legacy")).lower()
        if collision_mode not in {"legacy", "broadphase", "optimized"}:
            raise ValueError("collision_check_mode must be one of: legacy, broadphase, optimized")
        sensor_mode = str(self.config.get("sensor_road_edge_mode", "per_vehicle")).lower()
        if sensor_mode not in {"per_vehicle", "batched", "optimized"}:
            raise ValueError("sensor_road_edge_mode must be one of: per_vehicle, batched, optimized")
        geometry_path = None
        if self.scene == "japanese":
            configured = self.config.get("japanese_road_geometry")
            candidate = (
                Path(str(configured)).expanduser().resolve()
                if configured
                else Path(self._prebuilt_dir).resolve() / "ROAD_GEOMETRY.json"
            )
            if candidate.is_file():
                geometry_path = candidate
            elif configured:
                raise FileNotFoundError(candidate)
        cache_key = (
            self.scene,
            str(geometry_path) if geometry_path is not None else "legacy",
        )
        net = self._NETWORK_CACHE.get(cache_key)
        if net is None:
            net = (
                create_japanese_road(geometry_path)
                if self.scene == "japanese" and geometry_path is not None
                else builder()
            )
            self._NETWORK_CACHE[cache_key] = net
        self.net = net
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            use_query_fast_path=query_mode != "legacy",
            query_cell_size=float(self.config.get("road_query_cell_size", 25.0)),
            use_collision_broadphase=collision_mode != "legacy",
            collision_cell_size=float(self.config.get("collision_broadphase_cell_size", 12.0)),
            collision_broadphase_min_entities=int(self.config.get("collision_broadphase_min_entities", 32)),
            record_replay_diagnostics=bool(self.config.get("record_replay_diagnostics", True)),
        )
        self.road.debug_idm_handover = bool(self.config.get("debug_idm_handover", False))
        debug_ids = self.config.get("debug_idm_handover_ids")
        self.road.debug_idm_handover_ids = {int(vehicle_id) for vehicle_id in debug_ids} if debug_ids else None

    def _create_vehicles(self):
        # Build ego vehicles first.
        self.controlled_vehicles = []
        self._ego_start_indices = {}
        self._expert_state_by_vehicle_id = {}
        self._replay_xy_pol.clear()
        ego_records = self.trajectory_set["ego"]
        shared_start_index = self._resolve_shared_ego_start_index(ego_records)
        max_traj_steps = []

        for ego_index, ego_id in enumerate(self.ego_ids):
            ego_rec = ego_records[ego_id]
            ego_traj_full = self._load_processed_ego_trajectory(ego_id, ego_rec)
            ego_len, ego_wid = get_ego_dimensions(ego_rec, FEET_PER_METER, self.scene)
            ego_traj, ego_start_index, ego_policy_steps = self._prepare_ego_trajectory(
                ego_id=ego_id,
                ego_rec=ego_rec,
                ego_traj_full=ego_traj_full,
                ego_len=ego_len,
                shared_start_index=shared_start_index,
            )
            ego = build_ego_vehicle(
                road=self.road,
                scene=self.scene,
                ego_traj=ego_traj,
                ego_len=ego_len,
                ego_wid=ego_wid,
                control_mode=self.control_mode,
                action_cfg=self.action_cfg,
            )
            ego.vehicle_ID = ego_id
            if self.scene_dataset_collection_mode:
                self._configure_scene_collection_vehicle(
                    ego=ego,
                    ego_rec=ego_rec,
                    ego_traj_full=ego_traj_full,
                )
                active_occupancy = self._scene_collection_spawn_active_occupancy(ego)
                min_occupancy = float(self.config.get("controlled_vehicle_min_occupancy", 0.0))
                if active_occupancy < min_occupancy:
                    logger.warning(
                        "Skipping controlled vehicle %s because its scene-collection active occupancy %.3f is below the configured minimum %.3f.",
                        ego_id,
                        active_occupancy,
                        min_occupancy,
                    )
                    continue
            elif self._vehicle_has_spawn_conflict(ego):
                logger.warning(
                    "Skipping controlled vehicle %s because its spawn pose overlaps an existing road entity.",
                    ego_id,
                )
                continue
            self.road.vehicles.append(ego)
            self.controlled_vehicles.append(ego)
            self._ego_start_indices[int(ego_id)] = int(ego_start_index)
            if self.scene_dataset_collection_mode:
                max_traj_steps.append(int(len(ego_traj_full)))
            else:
                max_traj_steps.append(int(ego_policy_steps))

            if bool(self.config.get("disable_controlled_vehicle_collisions", False)):
                ego.check_collisions = False
                ego.collidable = False

            if self.expert_test_mode:
                expert_state = self._expert_state_by_vehicle_id[int(ego_id)]
                expert_state["replay_xy"].append(ego.position.copy())
                if ego_index == 0:
                    self._replay_xy_pol = list(expert_state["replay_xy"])

        if not self.controlled_vehicles:
            raise RuntimeError("Failed to create any controlled vehicles.")

        self._max_traj_policy_steps = min(max_traj_steps) if max_traj_steps else None
        if self.scene_dataset_collection_mode:
            self._sync_scene_collection_controlled_vehicles(step_index=0)

        # Build obstacle vehicles
        self._spawn_surrounding_vehicles()

    def _resolve_shared_ego_start_index(self, ego_records: dict[int, dict[str, Any]]) -> int:
        if self.expert_test_mode:
            return 0

        start_idx = common_first_valid_index([ego_records[ego_id]["trajectory"] for ego_id in self.ego_ids])
        if start_idx is None:
            raise RuntimeError("At least one controlled trajectory contains no valid motion data.")
        return int(start_idx)

    def _prepare_ego_trajectory(
        self,
        ego_id: int,
        ego_rec: dict[str, Any],
        ego_traj_full: np.ndarray,
        ego_len: float,
        shared_start_index: int,
    ) -> tuple[np.ndarray, int, int]:
        if self.expert_test_mode:
            self._setup_expert_tracker(ego_id, ego_traj_full, ego_len)
            if self.scene_dataset_collection_mode:
                ego_start_index = 0
            else:
                ego_start_index = int(self._ego_start_index)
        else:
            ego_start_index = int(shared_start_index)

        ego_traj = ego_traj_full[ego_start_index:]
        if len(ego_traj) < 2:
            raise RuntimeError(f"Ego trajectory too short for vehicle {ego_id}.")

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])
        sim_per_policy = max(1, int(sim_freq // pol_freq))
        max_traj_policy_steps = int(np.ceil(len(ego_traj) / float(sim_per_policy)))
        if ego_id == self.ego_id or self.ego_id is None:
            self._ego_start_index = ego_start_index
        return ego_traj, ego_start_index, max_traj_policy_steps

    def _configure_scene_collection_vehicle(
        self,
        *,
        ego: EgoVehicle,
        ego_rec: dict[str, Any],
        ego_traj_full: np.ndarray,
    ) -> None:
        vehicle_id = int(getattr(ego, "vehicle_ID"))
        if self.control_mode == "teleport" or bool(
            self.config.get("scene_collection_external_controller", False)
        ):
            start_idx, end_idx, span_len = longest_continuous_active_span_bounds(
                ego_traj_full,
            )
            if start_idx is None or end_idx is None or span_len <= 0:
                raise RuntimeError(
                    f"Controlled vehicle {vehicle_id} has no continuous active span for scene collection."
                )
            start_index = int(start_idx)
            end_index = int(end_idx)
        else:
            expert_state = self._expert_state_by_vehicle_id[vehicle_id]
            start_index = int(expert_state["start_idx"])
            end_index = int(start_index + len(expert_state["ref_xy"]) - 1)
        first_row = np.asarray(ego_traj_full[start_index], dtype=float)
        x0, y0, speed0, lane0 = first_row[:4]

        ego.scene_collection_full_traj = np.asarray(ego_traj_full, dtype=float)
        provider_mask = ego_rec.get("provider_observation_mask")
        if provider_mask is None:
            ego.scene_collection_provider_observation_mask = None
        else:
            provider_mask = np.asarray(provider_mask, dtype=np.int8)
            if provider_mask.shape != (len(ego_traj_full),):
                raise ValueError(
                    "provider_observation_mask must align one-to-one with the "
                    f"trajectory for vehicle {vehicle_id}: "
                    f"{provider_mask.shape} != {(len(ego_traj_full),)}."
                )
            if not np.all(np.isin(provider_mask, [-1, 0, 1])):
                raise ValueError(
                    "provider_observation_mask may contain only -1 (absent), "
                    "0 (provider-interpolated), or 1 (image-detected)."
                )
            ego.scene_collection_provider_observation_mask = provider_mask.copy()
        ego.scene_collection_start_index = start_index
        ego.scene_collection_end_index = end_index
        ego.scene_collection_is_active = False
        ego.scene_collection_current_provider_lane_reconciled = False
        ego.scene_collection_real_length = float(getattr(ego, "LENGTH", 0.0))
        ego.scene_collection_real_width = float(getattr(ego, "WIDTH", 0.0))
        ego.scene_collection_spawn_position = np.array([x0, y0], dtype=float)
        ego.scene_collection_spawn_speed = float(speed0)
        ego.scene_collection_spawn_lane_id = int(lane0)
        ego.scene_collection_padding_position = np.array([0.0, 0.0], dtype=float)
        self._deactivate_scene_collection_vehicle(ego)

    def _set_scene_collection_vehicle_from_row(
        self,
        ego: EgoVehicle,
        row: np.ndarray,
        *,
        previous_row: np.ndarray | None = None,
        next_row: np.ndarray | None = None,
        provider_observation_flag: int | None = None,
    ) -> None:
        x, y, speed, lane_id = np.asarray(row, dtype=float)[:4]
        speed = trajectory_step_speed_mps(
            row,
            next_row,
            sample_frequency_hz=float(self.config["simulation_frequency"]),
        )
        ego.position = np.array([x, y], dtype=float)
        ego.speed = float(speed)
        ego.target_speed = float(speed)
        ego.visible = True
        ego.scene_collection_is_active = True
        ego.LENGTH = float(getattr(ego, "scene_collection_real_length", ego.LENGTH))
        ego.WIDTH = float(getattr(ego, "scene_collection_real_width", ego.WIDTH))

        fallback_heading = float(getattr(ego, "heading", 0.0))
        mapped_lane_index, provider_lane_reconciled = resolve_target_lane_index_from_row(
            self.road.network,
            self.scene,
            np.asarray(row, dtype=float),
            vehicle_width_m=float(ego.WIDTH),
            provider_observation_flag=provider_observation_flag,
        )
        ego.scene_collection_current_provider_lane_reconciled = bool(provider_lane_reconciled)
        if mapped_lane_index is not None:
            ego.target_lane_index = mapped_lane_index
            ego.lane_index = mapped_lane_index
            ego.lane = self.road.network.get_lane(mapped_lane_index)
            s0, _r0 = ego.lane.local_coordinates(ego.position)
            fallback_heading = float(ego.lane.heading_at(s0))
        ego.heading = self._heading_for_spawn_row(
            np.asarray(row, dtype=float),
            previous_row=(None if previous_row is None else np.asarray(previous_row, dtype=float)),
            next_row=None if next_row is None else np.asarray(next_row, dtype=float),
            fallback_heading=fallback_heading,
            prefer_motion=self.control_mode == "teleport",
            lane_index_override=mapped_lane_index,
        )

    def _heading_for_spawn_row(
        self,
        row: np.ndarray,
        *,
        previous_row: np.ndarray | None = None,
        next_row: np.ndarray | None = None,
        fallback_heading: float = 0.0,
        prefer_motion: bool = False,
        lane_index_override: tuple[str, str, int] | None = None,
    ) -> float:
        return heading_from_trajectory_row(
            self.road.network,
            self.scene,
            row,
            previous_row=previous_row,
            next_row=next_row,
            fallback_heading=fallback_heading,
            prefer_motion=prefer_motion,
            lane_index_override=lane_index_override,
        )

    def _vehicle_has_spawn_conflict(self, vehicle: EgoVehicle) -> bool:
        return road_entity_conflicts_at_pose(
            self.road,
            vehicle.position,
            heading=float(vehicle.heading),
            length=float(vehicle.LENGTH),
            width=float(vehicle.WIDTH),
            ignore_entity=vehicle,
        )

    def _scene_collection_row_has_conflict(
        self,
        ego: EgoVehicle,
        row: np.ndarray,
        *,
        previous_row: np.ndarray | None = None,
        next_row: np.ndarray | None = None,
    ) -> bool:
        row_arr = np.asarray(row, dtype=float)
        heading = self._heading_for_spawn_row(
            row_arr,
            previous_row=previous_row,
            next_row=next_row,
            fallback_heading=float(getattr(ego, "heading", 0.0)),
        )
        return road_entity_conflicts_at_pose(
            self.road,
            row_arr[:2],
            heading=heading,
            length=float(getattr(ego, "scene_collection_real_length", ego.LENGTH)),
            width=float(getattr(ego, "scene_collection_real_width", ego.WIDTH)),
            ignore_entity=ego,
        )

    def _scene_collection_min_occupancy_horizon_steps(self, traj_len: int) -> int:
        configured_steps = self.config.get("scene_collection_min_occupancy_steps")
        if configured_steps is None:
            configured_steps = self.config.get("max_episode_steps")
        max_steps = int(configured_steps or 0)
        if max_steps <= 0:
            return int(traj_len)
        return min(int(max_steps), int(traj_len))

    def _scene_collection_spawn_active_occupancy(self, ego: EgoVehicle) -> float:
        traj = np.asarray(getattr(ego, "scene_collection_full_traj"))
        if traj.ndim != 2 or traj.shape[0] <= 0:
            return 0.0
        horizon_steps = self._scene_collection_min_occupancy_horizon_steps(len(traj))
        if horizon_steps <= 0:
            return 0.0

        start_index = int(getattr(ego, "scene_collection_start_index", 0))
        end_index = int(getattr(ego, "scene_collection_end_index", len(traj) - 1))
        window_start = max(0, min(start_index, horizon_steps))
        window_end = max(window_start, min(end_index + 1, horizon_steps, len(traj)))
        active_steps = sum(1 for row in traj[window_start:window_end] if trajectory_row_is_active(row))
        return float(active_steps) / float(horizon_steps)

    def _scene_collection_spawn_has_min_occupancy(self, ego: EgoVehicle) -> bool:
        min_occupancy = float(self.config.get("controlled_vehicle_min_occupancy", 0.0))
        if min_occupancy <= 0.0:
            return True
        return self._scene_collection_spawn_active_occupancy(ego) >= min_occupancy

    def _activate_scene_collection_vehicle(
        self,
        ego: EgoVehicle,
        step_index: int,
        *,
        force_replay: bool = False,
    ) -> None:
        if getattr(ego, "scene_collection_is_active", False) and not force_replay:
            return
        traj = np.asarray(getattr(ego, "scene_collection_full_traj"))
        row = np.asarray(traj[step_index], dtype=float)
        previous_row = (
            np.asarray(traj[step_index - 1], dtype=float)
            if step_index > 0 and trajectory_row_is_active(traj[step_index - 1])
            else None
        )
        next_row = np.asarray(traj[step_index + 1], dtype=float) if step_index + 1 < len(traj) else None
        if not bool(
            self.config.get("disable_scene_collection_spawn_safety", False)
        ) and self._scene_collection_row_has_conflict(
            ego,
            row,
            previous_row=previous_row,
            next_row=next_row,
        ):
            self._deactivate_scene_collection_vehicle(ego)
            return
        if not force_replay and not bool(
            self.config.get("scene_collection_external_controller", False)
        ):
            vehicle_id = int(getattr(ego, "vehicle_ID"))
            expert_state = self._expert_state_by_vehicle_id.get(vehicle_id)
            if not isinstance(expert_state, dict):
                raise RuntimeError(f"Scene-collection activation has no expert tracker for vehicle {vehicle_id}.")
            tracker = expert_state.get("tracker")
            if tracker is None or not callable(getattr(tracker, "reset", None)):
                raise RuntimeError(
                    f"Scene-collection activation has an invalid expert tracker for vehicle {vehicle_id}."
                )
            start_index = int(getattr(ego, "scene_collection_start_index", 0))
            tracker_offset = int(step_index) - start_index
            if tracker_offset < 0:
                raise RuntimeError(
                    "Scene-collection tracker activation precedes its source "
                    f"start: step={step_index}, start={start_index}."
                )
            tracker.reset(k0=tracker_offset)
            expert_state["activation_tracker_offset"] = tracker_offset
        self._set_scene_collection_vehicle_from_row(
            ego,
            row,
            previous_row=previous_row,
            next_row=next_row,
            provider_observation_flag=(
                None
                if getattr(
                    ego,
                    "scene_collection_provider_observation_mask",
                    None,
                )
                is None
                else int(ego.scene_collection_provider_observation_mask[step_index])
            ),
        )

    def _deactivate_scene_collection_vehicle(self, ego: EgoVehicle) -> None:
        ego.scene_collection_is_active = False
        ego.scene_collection_current_provider_lane_reconciled = False
        ego.visible = False
        ego.position = np.array(getattr(ego, "scene_collection_padding_position"), dtype=float)
        ego.speed = 0.0
        ego.target_speed = 0.0
        ego.LENGTH = 0.0
        ego.WIDTH = 0.0

    def _sync_scene_collection_controlled_vehicles(self, step_index: int) -> None:
        if not self.scene_dataset_collection_mode:
            return
        force_replay = self.control_mode == "teleport"
        for ego in self.controlled_vehicles:
            start_index = int(getattr(ego, "scene_collection_start_index", 0))
            end_index = int(getattr(ego, "scene_collection_end_index", -1))
            if start_index <= step_index <= end_index:
                if force_replay:
                    traj = np.asarray(getattr(ego, "scene_collection_full_traj"))
                    if not trajectory_row_is_active(traj[step_index]):
                        self._deactivate_scene_collection_vehicle(ego)
                        continue
                self._activate_scene_collection_vehicle(
                    ego,
                    step_index=step_index,
                    force_replay=force_replay,
                )
            else:
                self._deactivate_scene_collection_vehicle(ego)

    def _refresh_scene_collection_observation_after_sync(
        self,
        obs: Any,
        action: Action,
    ) -> tuple[Any, dict[str, Any]]:
        """Align the returned next observation with post-step activation state.

        In scene collection, a vehicle may become active at the policy step
        reached by ``super().step``.  The superclass observation was computed
        before that activation sync.  Returning it would pair the next
        transition's expert action with an inactive/stale observation (most
        visibly a stale ego heading for late-start Japanese trajectories).
        Re-observe only after the state sync and rebuild info against the same
        observation.  Teleport collection follows its separate in-step path.
        """

        if not self.scene_dataset_collection_mode or self.control_mode == "teleport":
            return obs, {}
        self._sync_scene_collection_controlled_vehicles(step_index=int(self.steps))
        refreshed = self.observation_type.observe()
        return refreshed, self._info(refreshed, action)

    def _processed_trajectory_cache_key(self, episode_name: str, vehicle_id: int) -> tuple[str, str, str, int]:
        return (self._prebuilt_dir, self.scene, episode_name, int(vehicle_id))

    def _load_processed_ego_trajectory(self, ego_id: int, ego_rec: dict[str, Any]) -> np.ndarray:
        cache_key = self._processed_trajectory_cache_key(self.episode_name, ego_id)
        cached = self._PROCESSED_TRAJECTORY_CACHE.get(cache_key)
        if cached is None:
            cached = load_ego_trajectory(ego_rec, self.scene)
            self._PROCESSED_TRAJECTORY_CACHE[cache_key] = cached
        return cached

    def _expert_reference_cache_key(self, ego_id: int, ego_len: float) -> tuple[str, str, str, int, int, float]:
        return (
            self._prebuilt_dir,
            self.scene,
            self.episode_name,
            int(ego_id),
            int(self.config["policy_frequency"]),
            round(float(ego_len), 4),
        )

    def _setup_expert_tracker(self, ego_id: int, ego_traj_full: np.ndarray, ego_len: float) -> None:
        cache_key = self._expert_reference_cache_key(ego_id, ego_len)
        cached = self._EXPERT_REFERENCE_CACHE.get(cache_key)
        if cached is None:
            cached = setup_expert_tracker(self.net, ego_traj_full, ego_len, self.config)
            self._EXPERT_REFERENCE_CACHE[cache_key] = cached
        ref_xy_pol, ref_v_pol, lane_pol, start_idx = cached
        ref_lane_pol = lane_pol - 1
        tracker = PurePursuitTracker(
            ref_xy=ref_xy_pol,
            ref_v=ref_v_pol,
            ref_lanes=ref_lane_pol,
            dt=1.0 / self.config["policy_frequency"],
            L_forward=ego_len,
            max_steer=MAX_STEER,
            Ld0=5.0,
            Ld_k=0.6,
            kp_v=float(self.config.get("expert_tracker_kp_v", 0.8)),
            steer_rate_limit=6.0,
            steer_lpf_tau=0.15,
            jerk_limit=10.0,
        )
        state = {
            "ref_xy": ref_xy_pol,
            "ref_v": ref_v_pol,
            "ref_lane": ref_lane_pol,
            "start_idx": int(start_idx),
            "tracker": tracker,
            "actions_policy": [],
            "tracker_dbg": [],
            "replay_xy": [],
        }
        self._expert_state_by_vehicle_id[int(ego_id)] = state
        self._ego_start_index = int(start_idx)

        if ego_id == self.ego_id or self.ego_id is None:
            self._expert_ref_xy_pol = state["ref_xy"]
            self._expert_ref_v_pol = state["ref_v"]
            self._expert_ref_lane_pol = state["ref_lane"]
            self._tracker = state["tracker"]
            self._expert_actions_policy = state["actions_policy"]
            self._tracker_dbg = state["tracker_dbg"]

    def _spawn_surrounding_vehicles(self) -> None:
        max_surr_raw = self.config.get("max_surrounding", 0)
        spawn_all = max_surr_raw == "all"
        max_surr = None if spawn_all else int(max_surr_raw)
        if not spawn_all and max_surr <= 0:
            return

        spawn_surrounding_vehicles(
            self.trajectory_set,
            self._ego_start_indices,
            max_surr,
            self.road,
            scene=self.scene,
            allow_idm=bool(self.config.get("allow_idm", True)),
        )

    def visualize(
        self, steps: int | None = None, width: int = 1200, height: int = 600, scaling: float = 5.5, mode: str = "all"
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
    # INFO / REWARDS / TERMINATION
    # -------------------------------------------------------------------------
    @staticmethod
    def _vehicle_length_for_metrics(vehicle: object | None) -> float:
        if vehicle is None:
            return 0.0
        for name in ("LENGTH", "length"):
            if hasattr(vehicle, name):
                try:
                    value = float(getattr(vehicle, name))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0.0:
                    return value
        return 0.0

    @staticmethod
    def _vehicle_id_for_metrics(vehicle: object | None) -> int:
        if vehicle is None:
            return -1
        try:
            return int(getattr(vehicle, "vehicle_ID", -1))
        except (TypeError, ValueError):
            return -1

    def _interaction_metric_targets(self, vehicle: EgoVehicle) -> tuple[float, float, float, float]:
        return interaction_metric_targets_from_idm(
            self.idm_parameters,
            self.config,
            speed=float(getattr(vehicle, "speed", 0.0)),
        )

    def _bumper_gap_and_ttc(
        self,
        vehicle: EgoVehicle,
        other: object | None,
        *,
        rear: bool,
    ) -> tuple[float, float]:
        if other is None:
            return float("inf"), float("inf")
        try:
            center_gap = float(abs(vehicle.lane_distance_to(other)))
        except Exception:
            return float("inf"), float("inf")
        if not np.isfinite(center_gap):
            return float("inf"), float("inf")
        bumper_gap = center_gap - 0.5 * (
            self._vehicle_length_for_metrics(vehicle) + self._vehicle_length_for_metrics(other)
        )
        ego_speed = max(float(getattr(vehicle, "speed", 0.0)), 0.0)
        other_speed = max(float(getattr(other, "speed", 0.0)), 0.0)
        closing_speed = (other_speed - ego_speed) if rear else (ego_speed - other_speed)
        if closing_speed > 1.0e-6:
            ttc = max(float(bumper_gap), 0.0) / float(closing_speed)
        else:
            ttc = float("inf")
        return float(bumper_gap), float(ttc)

    def controlled_vehicle_interaction_metrics(self) -> list[dict[str, float | int | bool | str]]:
        if self.road is None:
            return []
        metrics: list[dict[str, float | int | bool | str]] = []
        crash_flags = [bool(getattr(vehicle, "crashed", False)) for vehicle in self.controlled_vehicles]
        offroad_flags = [
            (not bool(getattr(vehicle, "completed", False))) and (not bool(getattr(vehicle, "on_road", True)))
            for vehicle in self.controlled_vehicles
        ]
        for idx, vehicle in enumerate(self.controlled_vehicles):
            try:
                front_vehicle, rear_vehicle = self.road.neighbour_vehicles(vehicle)
            except Exception:
                front_vehicle, rear_vehicle = None, None
            front_gap, front_ttc = self._bumper_gap_and_ttc(vehicle, front_vehicle, rear=False)
            rear_gap, rear_ttc = self._bumper_gap_and_ttc(vehicle, rear_vehicle, rear=True)
            finite_gaps = [gap for gap in (front_gap, rear_gap) if np.isfinite(gap)]
            finite_ttcs = [ttc for ttc in (front_ttc, rear_ttc) if np.isfinite(ttc)]
            min_gap = min(finite_gaps) if finite_gaps else float("inf")
            min_ttc = min(finite_ttcs) if finite_ttcs else float("inf")
            ttc_target, ttc_floor, gap_target, gap_floor = self._interaction_metric_targets(vehicle)
            metrics.append(
                {
                    "vehicle_id": self._vehicle_id_for_metrics(vehicle),
                    "front_vehicle_id": self._vehicle_id_for_metrics(front_vehicle),
                    "rear_vehicle_id": self._vehicle_id_for_metrics(rear_vehicle),
                    "front_gap": float(front_gap),
                    "rear_gap": float(rear_gap),
                    "front_ttc": float(front_ttc),
                    "rear_ttc": float(rear_ttc),
                    "min_gap": float(min_gap),
                    "min_ttc": float(min_ttc),
                    "ttc_target": float(ttc_target),
                    "ttc_floor": float(ttc_floor),
                    "gap_target": float(gap_target),
                    "gap_floor": float(gap_floor),
                    "speed": float(getattr(vehicle, "speed", 0.0)),
                    "lane_index": str(getattr(vehicle, "lane_index", "")),
                    "crashed": bool(crash_flags[idx]) if idx < len(crash_flags) else False,
                    "offroad": bool(offroad_flags[idx]) if idx < len(offroad_flags) else False,
                }
            )
        return metrics

    def _info(self, obs: Any, action: Action | None = None) -> dict[str, Any]:
        info = super()._info(obs, action)
        policy_vehicles = self._policy_controlled_vehicles()
        termination_vehicles = self._termination_vehicles()
        info["speed"] = [float(vehicle.speed) for vehicle in policy_vehicles]
        info["all_controlled_vehicle_speeds"] = [float(vehicle.speed) for vehicle in self.controlled_vehicles]
        info["crashed"] = all(vehicle.crashed for vehicle in termination_vehicles) if termination_vehicles else False
        info["alive_controlled_vehicle_ids"] = [
            getattr(vehicle, "vehicle_ID", None)
            for vehicle in self.controlled_vehicles
            if not vehicle.crashed
            and not bool(getattr(vehicle, "completed", False))
            and (not self.scene_dataset_collection_mode or bool(getattr(vehicle, "scene_collection_is_active", False)))
        ]
        info["support_vehicle_ids"] = [
            getattr(vehicle, "vehicle_ID", None)
            for vehicle in self.controlled_vehicles
            if vehicle not in policy_vehicles
        ]
        info["requested_controlled_vehicle_ids"] = list(self.ego_ids)
        info["controlled_vehicle_ids"] = [
            int(getattr(vehicle, "vehicle_ID", -1)) for vehicle in self.controlled_vehicles
        ]
        info["controlled_vehicle_crashes"] = [bool(vehicle.crashed) for vehicle in self.controlled_vehicles]
        info["controlled_vehicle_completed"] = [
            bool(getattr(vehicle, "completed", False)) for vehicle in self.controlled_vehicles
        ]
        info["controlled_vehicle_on_road"] = [
            bool(getattr(vehicle, "completed", False)) or bool(getattr(vehicle, "on_road", True))
            for vehicle in self.controlled_vehicles
        ]
        info["controlled_vehicle_offroad"] = [
            (not bool(getattr(vehicle, "completed", False))) and (not bool(getattr(vehicle, "on_road", True)))
            for vehicle in self.controlled_vehicles
        ]
        info["controlled_vehicle_collision_partners"] = [
            {
                "vehicle_id": int(getattr(vehicle, "vehicle_ID", -1)),
                "partner_vehicle_id": (
                    None
                    if getattr(
                        vehicle,
                        "first_collision_partner_vehicle_id",
                        None,
                    )
                    is None
                    else int(vehicle.first_collision_partner_vehicle_id)
                ),
                "partner_type": getattr(
                    vehicle,
                    "first_collision_partner_type",
                    None,
                ),
                "provenance": getattr(
                    vehicle,
                    "first_collision_partner_provenance",
                    None,
                ),
            }
            for vehicle in self.controlled_vehicles
        ]
        controlled_identities = {id(vehicle) for vehicle in self.controlled_vehicles}
        info["background_idm_handovers"] = [
            {
                "vehicle_id": int(getattr(vehicle, "vehicle_ID", -1)),
                "handover_step": int(vehicle.idm_handover_step),
                "reason": str(getattr(vehicle, "idm_handover_reason", None) or "unspecified"),
            }
            for vehicle in list(getattr(self.road, "vehicles", ()) or ())
            if id(vehicle) not in controlled_identities and getattr(vehicle, "idm_handover_step", None) is not None
        ]
        if bool(self.config.get("enable_interaction_metrics", False)):
            info["controlled_vehicle_interaction_metrics"] = self.controlled_vehicle_interaction_metrics()
        info["scene_dataset_collection_mode"] = self.scene_dataset_collection_mode
        return info

    # -------------------------------------------------------------------------
    # REWARDS & TERMINATION
    # -------------------------------------------------------------------------
    def _rewards(self, action: Any) -> dict[str, float]:
        termination_vehicles = self._termination_vehicles()
        crashes = [float(vehicle.crashed) for vehicle in termination_vehicles]
        return {
            "collision_reward": max(crashes) if crashes else 0.0,
            "all_controlled_crashed": float(all(vehicle.crashed for vehicle in termination_vehicles))
            if termination_vehicles
            else 0.0,
            "all_controlled_terminal": float(
                all(self._vehicle_is_terminal(vehicle) for vehicle in termination_vehicles)
            )
            if termination_vehicles
            else 0.0,
        }

    def _reward(self, action: Any) -> float:
        return 0.0

    def _is_terminated(self) -> bool:
        termination_vehicles = self._termination_vehicles()
        if not termination_vehicles:
            return False

        if self.config.get("terminate_when_all_controlled_crashed", True):
            return all(self._vehicle_is_terminal(vehicle) for vehicle in termination_vehicles)
        return any(self._vehicle_is_terminal(vehicle) for vehicle in termination_vehicles)

    @staticmethod
    def _vehicle_is_terminal(vehicle: EgoVehicle) -> bool:
        return bool(getattr(vehicle, "crashed", False)) or bool(getattr(vehicle, "completed", False))

    def _is_truncated(self) -> bool:
        max_steps_cfg = self.config.get("max_episode_steps", None)
        max_steps_traj = getattr(self, "_max_traj_policy_steps", None)
        if self.config.get("truncate_to_trajectory_length", False):
            candidates = [v for v in (max_steps_cfg, max_steps_traj) if v is not None]
            return self.steps >= min(candidates) if candidates else False
        return self.steps >= max_steps_cfg if max_steps_cfg is not None else False

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action: Action):
        expert_action = None
        expert_action_str = None
        expert_action_idx = None
        expert_actions: list[np.ndarray | None] = []
        expert_action_strs: list[str | None] = []
        expert_action_idxs: list[int | None] = []

        if self.scene_dataset_collection_mode:
            self._sync_scene_collection_controlled_vehicles(step_index=int(self.steps))

        if self.expert_test_mode:
            if self.config.get("action", {}).get("type") == "MultiAgentAction":
                resolved_actions = []
                for vehicle in self.controlled_vehicles:
                    a_i, a_cont_i, a_str_i, a_idx_i = self._resolve_expert_action(vehicle=vehicle)
                    resolved_actions.append(a_i)
                    expert_actions.append(a_cont_i.copy() if a_cont_i is not None else None)
                    expert_action_strs.append(a_str_i)
                    expert_action_idxs.append(a_idx_i)

                action = tuple(resolved_actions)
                if expert_actions:
                    expert_action = expert_actions[0]
                    expert_action_str = expert_action_strs[0]
                    expert_action_idx = expert_action_idxs[0]
            else:
                action, expert_action, expert_action_str, expert_action_idx = self._resolve_expert_action()
                expert_actions = [expert_action.copy() if expert_action is not None else None]
                expert_action_strs = [expert_action_str]
                expert_action_idxs = [expert_action_idx]

        # -----------------------------------------------------------
        # EXECUTE SIMULATION STEP
        # -----------------------------------------------------------
        if self.scene_dataset_collection_mode and self.control_mode == "teleport":
            if self.road is None or self.vehicle is None:
                raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

            self.time += 1 / self.config["policy_frequency"]
            self._simulate(action)
            self._sync_scene_collection_controlled_vehicles(step_index=int(self.steps))
            obs = self.observation_type.observe()
            reward = self._reward(action)
            terminated = self._is_terminated()
            truncated = self._is_truncated()
            info = self._info(obs, action)
            if self.render_mode == "human":
                self.render()
        else:
            obs, reward, terminated, truncated, info = super().step(action)

        if info is None:
            info = {}

        info["applied_action"] = action
        info["expert_controlled_vehicle_ids"] = [
            int(getattr(vehicle, "vehicle_ID", -1)) for vehicle in self.controlled_vehicles
        ]
        if isinstance(action, tuple):
            info["applied_actions"] = tuple(action)
        if expert_action is not None:
            info["expert_action_continuous"] = expert_action.copy()
        if expert_actions:
            info["expert_action_continuous_all"] = [a.copy() if a is not None else None for a in expert_actions]
        if expert_action_str is not None:
            info["expert_action_discrete"] = expert_action_str
            info["expert_action_discrete_idx"] = expert_action_idx
        if expert_action_strs:
            info["expert_action_discrete_all"] = list(expert_action_strs)
            info["expert_action_discrete_idx_all"] = list(expert_action_idxs)

        if self.expert_test_mode:
            for vehicle in self.controlled_vehicles:
                vehicle_id = int(getattr(vehicle, "vehicle_ID", -1))
                expert_state = self._expert_state_by_vehicle_id.get(vehicle_id)
                if expert_state is not None:
                    expert_state["replay_xy"].append(vehicle.position.copy())
            if self.vehicle is not None:
                expert_state = self._expert_state_by_vehicle_id.get(int(self.vehicle.vehicle_ID))
                if expert_state is not None:
                    self._replay_xy_pol = list(expert_state["replay_xy"])

        if self.scene_dataset_collection_mode and self.control_mode != "teleport":
            obs, refreshed_info = self._refresh_scene_collection_observation_after_sync(
                obs,
                action,
            )
            info.update(refreshed_info)

        return obs, reward, terminated, truncated, info
