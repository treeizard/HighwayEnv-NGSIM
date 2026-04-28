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
from typing import Any

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.ngsim_utils.expert.ngsim_expert_mixin import NGSimExpertMixin
from highway_env.road.road import Road

from highway_env.ngsim_utils.core.config import (
    deep_update,
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
    target_lane_index_from_lane_id,
)
from highway_env.ngsim_utils.vehicles.replay import (
    road_entity_conflicts_at_pose,
    road_entity_pose_polygon,
    spawn_surrounding_vehicles,
)
from highway_env.ngsim_utils.road.gen_road import create_ngsim_101_road, create_japanese_road
from highway_env.ngsim_utils.core.constants import (
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
    _NETWORK_CACHE: dict[str, Any] = {}
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
                "action": {"type": "ContinuousAction"},
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
                "episode_root": "highway_env/data/processed_20s",
                "prebuilt_split": "train",
                # Quality of life/ debugging
                "control_all_vehicles": False,
                "max_surrounding": "all",
                "show_trajectories": True,
                "seed": None,
                "expert_test_mode": False,
                "discrete_expert_policy": "planner",
                "percentage_controlled_vehicles": 0.1,
                "terminate_when_all_controlled_crashed": True,
                "truncate_to_trajectory_length": False, # allow for replay
                "scene_dataset_collection_mode": False,
                "disable_controlled_vehicle_collisions": False,
                "crash_controlled_vehicles_offroad": True,
                "disable_scene_collection_spawn_safety": False,
                "allow_idm": True,
                "controlled_vehicle_min_occupancy": 0.8,
                "idm_parameters": None,
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
            int(
                self.config["simulation_frequency"]
                // self.config["policy_frequency"]
            ),
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
        if not any(
            getattr(vehicle, "remove_from_road", False)
            for vehicle in self.road.vehicles
        ):
            return
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if not getattr(vehicle, "remove_from_road", False)
        ]

    def _simulate(self, action: Action | None = None) -> None:
        """Run simulation frames and prune replay vehicles that have despawned."""
        frames = self._frames_per_action
        dt = 1 / self.config["simulation_frequency"]
        for frame in range(frames):
            if (
                action is not None
                and not self.config["manual_control"]
                and self.steps % frames == 0
            ):
                self.action_type.act(action)

            self.road.act()
            self.road.step(dt)
            self._crash_offroad_controlled_vehicles()
            self._prune_removed_vehicles()
            self.steps += 1

            if frame < frames - 1:
                self._automatic_rendering()

        self.enable_auto_render = False

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
            percentage_controlled_vehicles=self.config[
                "percentage_controlled_vehicles"
            ],
            np_random=self.np_random,
            episode_name=self.episode_name,
            control_all_vehicles=bool(self.config.get("control_all_vehicles", False)),
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
        net = self._NETWORK_CACHE.get(self.scene)
        if net is None:
            net = builder()
            self._NETWORK_CACHE[self.scene] = net
        self.net = net
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road.debug_idm_handover = bool(self.config.get("debug_idm_handover", False))
        debug_ids = self.config.get("debug_idm_handover_ids")
        self.road.debug_idm_handover_ids = (
            {int(vehicle_id) for vehicle_id in debug_ids}
            if debug_ids
            else None
        )

    def _create_vehicles(self):
        # Build ego vehicles first. 
        self.controlled_vehicles = []
        self._ego_start_indices = {}
        self._expert_state_by_vehicle_id = {}
        self._replay_xy_pol.clear()
        ego_records = self.trajectory_set["ego"]
        shared_start_index = self._resolve_shared_ego_start_index(ego_records)
        max_traj_steps = []
        scene_collection_spawn_records: list[dict[str, Any]] = []

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
                    ego_traj_full=ego_traj_full,
                )
                if not self._scene_collection_spawn_has_min_occupancy(ego):
                    logger.warning(
                        "Skipping controlled vehicle %s because its scene-collection active occupancy is below the configured minimum.",
                        ego_id,
                    )
                    continue
                if self._scene_collection_spawn_has_conflict(
                    ego,
                    scene_collection_spawn_records,
                ):
                    logger.warning(
                        "Skipping controlled vehicle %s because its scene-collection spawn pose overlaps an existing controlled vehicle.",
                        ego_id,
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
                scene_collection_spawn_records.append(
                    self._scene_collection_spawn_record(ego)
                )
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

    def _resolve_shared_ego_start_index(
        self, ego_records: dict[int, dict[str, Any]]
    ) -> int:
        if self.expert_test_mode:
            return 0

        start_idx = common_first_valid_index(
            [ego_records[ego_id]["trajectory"] for ego_id in self.ego_ids]
        )
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
        ego_traj_full: np.ndarray,
    ) -> None:
        vehicle_id = int(getattr(ego, "vehicle_ID"))
        if self.control_mode == "teleport":
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
        ego.scene_collection_start_index = start_index
        ego.scene_collection_end_index = end_index
        ego.scene_collection_is_active = False
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
        next_row: np.ndarray | None = None,
    ) -> None:
        x, y, speed, lane_id = np.asarray(row, dtype=float)[:4]
        if next_row is not None:
            nx, ny = np.asarray(next_row, dtype=float)[:2]
            speed_from_delta = float(
                np.hypot(float(nx - x), float(ny - y))
                * float(self.config["simulation_frequency"])
            )
            if speed_from_delta > 1e-3:
                speed = speed_from_delta
        ego.position = np.array([x, y], dtype=float)
        ego.speed = float(max(speed, 0.0))
        ego.target_speed = float(max(speed, 0.0))
        ego.visible = True
        ego.scene_collection_is_active = True
        ego.LENGTH = float(getattr(ego, "scene_collection_real_length", ego.LENGTH))
        ego.WIDTH = float(getattr(ego, "scene_collection_real_width", ego.WIDTH))

        mapped_lane_index = target_lane_index_from_lane_id(
            self.road.network,
            self.scene,
            float(x),
            int(lane_id),
        )
        if mapped_lane_index is not None:
            ego.target_lane_index = mapped_lane_index
            ego.lane_index = mapped_lane_index
            ego.lane = self.road.network.get_lane(mapped_lane_index)
            s0, _r0 = ego.lane.local_coordinates(ego.position)
            ego.heading = float(ego.lane.heading_at(s0))

    def _heading_for_spawn_row(
        self,
        row: np.ndarray,
        *,
        next_row: np.ndarray | None = None,
        fallback_heading: float = 0.0,
    ) -> float:
        return heading_from_trajectory_row(
            self.road.network,
            self.scene,
            row,
            next_row=next_row,
            fallback_heading=fallback_heading,
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
        next_row: np.ndarray | None = None,
    ) -> bool:
        row_arr = np.asarray(row, dtype=float)
        heading = self._heading_for_spawn_row(
            row_arr,
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

    def _scene_collection_spawn_record(self, ego: EgoVehicle) -> dict[str, Any]:
        traj = np.asarray(getattr(ego, "scene_collection_full_traj"))
        start_index = int(getattr(ego, "scene_collection_start_index", 0))
        row = np.asarray(traj[start_index], dtype=float)
        next_row = (
            np.asarray(traj[start_index + 1], dtype=float)
            if start_index + 1 < len(traj)
            else None
        )
        length = float(getattr(ego, "scene_collection_real_length", ego.LENGTH))
        width = float(getattr(ego, "scene_collection_real_width", ego.WIDTH))
        heading = self._heading_for_spawn_row(
            row,
            next_row=next_row,
            fallback_heading=float(getattr(ego, "heading", 0.0)),
        )
        position = np.asarray(row[:2], dtype=float)
        return {
            "vehicle_id": int(getattr(ego, "vehicle_ID", -1)),
            "position": position,
            "diagonal": float(np.hypot(length, width)),
            "polygon": road_entity_pose_polygon(position, heading, length, width),
        }

    def _scene_collection_spawn_has_min_occupancy(self, ego: EgoVehicle) -> bool:
        min_occupancy = float(self.config.get("controlled_vehicle_min_occupancy", 0.0))
        if min_occupancy <= 0.0:
            return True

        max_steps = int(self.config.get("max_episode_steps") or 0)
        if max_steps <= 0:
            return True

        traj = np.asarray(getattr(ego, "scene_collection_full_traj"))
        start_index = int(getattr(ego, "scene_collection_start_index", 0))
        # The acceptance/debug counters are recorded after each simulation step,
        # so compare against the same post-step trajectory window.
        window_start = start_index + 1
        end_index = min(len(traj), window_start + max_steps)
        active_steps = sum(
            1 for row in traj[window_start:end_index] if trajectory_row_is_active(row)
        )
        return (active_steps / float(max_steps)) >= min_occupancy

    def _scene_collection_spawn_has_conflict(
        self,
        ego: EgoVehicle,
        accepted_spawn_records: list[dict[str, Any]],
    ) -> bool:
        if bool(self.config.get("disable_scene_collection_spawn_safety", False)):
            return False

        traj = np.asarray(getattr(ego, "scene_collection_full_traj"))
        start_index = int(getattr(ego, "scene_collection_start_index", 0))
        row = np.asarray(traj[start_index], dtype=float)
        next_row = (
            np.asarray(traj[start_index + 1], dtype=float)
            if start_index + 1 < len(traj)
            else None
        )
        if self._scene_collection_row_has_conflict(ego, row, next_row=next_row):
            return True

        candidate = self._scene_collection_spawn_record(ego)
        zero_velocity = np.zeros(2, dtype=float)
        for accepted in accepted_spawn_records:
            if (
                np.linalg.norm(candidate["position"] - accepted["position"])
                > 0.5 * (candidate["diagonal"] + accepted["diagonal"])
            ):
                continue
            intersecting, _will_intersect, _transition = utils.are_polygons_intersecting(
                candidate["polygon"],
                accepted["polygon"],
                zero_velocity,
                zero_velocity,
            )
            if intersecting:
                return True
        return False

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
        next_row = (
            np.asarray(traj[step_index + 1], dtype=float)
            if step_index + 1 < len(traj)
            else None
        )
        if (
            not bool(self.config.get("disable_scene_collection_spawn_safety", False))
            and self._scene_collection_row_has_conflict(ego, row, next_row=next_row)
        ):
            self._deactivate_scene_collection_vehicle(ego)
            return
        self._set_scene_collection_vehicle_from_row(ego, row, next_row=next_row)

    def _deactivate_scene_collection_vehicle(self, ego: EgoVehicle) -> None:
        ego.scene_collection_is_active = False
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

    def _processed_trajectory_cache_key(
        self, episode_name: str, vehicle_id: int
    ) -> tuple[str, str, str, int]:
        return (self._prebuilt_dir, self.scene, episode_name, int(vehicle_id))

    def _load_processed_ego_trajectory(
        self, ego_id: int, ego_rec: dict[str, Any]
    ) -> np.ndarray:
        cache_key = self._processed_trajectory_cache_key(self.episode_name, ego_id)
        cached = self._PROCESSED_TRAJECTORY_CACHE.get(cache_key)
        if cached is None:
            cached = load_ego_trajectory(ego_rec, self.scene)
            self._PROCESSED_TRAJECTORY_CACHE[cache_key] = cached
        return cached

    def _expert_reference_cache_key(
        self, ego_id: int, ego_len: float
    ) -> tuple[str, str, str, int, int, float]:
        return (
            self._prebuilt_dir,
            self.scene,
            self.episode_name,
            int(ego_id),
            int(self.config["policy_frequency"]),
            round(float(ego_len), 4),
        )

    def _setup_expert_tracker(
        self, ego_id: int, ego_traj_full: np.ndarray, ego_len: float
    ) -> None:
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
            kp_v=0.8,
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
    # INFO / REWARDS / TERMINATION
    # -------------------------------------------------------------------------
    def _info(self, obs: Any, action: Action | None = None) -> dict[str, Any]:
        info = super()._info(obs, action)
        policy_vehicles = self._policy_controlled_vehicles()
        termination_vehicles = self._termination_vehicles()
        info["speed"] = [float(vehicle.speed) for vehicle in policy_vehicles]
        info["all_controlled_vehicle_speeds"] = [
            float(vehicle.speed) for vehicle in self.controlled_vehicles
        ]
        info["crashed"] = all(
            vehicle.crashed for vehicle in termination_vehicles
        ) if termination_vehicles else False
        info["alive_controlled_vehicle_ids"] = [
            getattr(vehicle, "vehicle_ID", None)
            for vehicle in self.controlled_vehicles
            if not vehicle.crashed
            and (
                not self.scene_dataset_collection_mode
                or bool(getattr(vehicle, "scene_collection_is_active", False))
            )
        ]
        info["support_vehicle_ids"] = [
            getattr(vehicle, "vehicle_ID", None)
            for vehicle in self.controlled_vehicles
            if vehicle not in policy_vehicles
        ]
        info["controlled_vehicle_ids"] = list(self.ego_ids)
        info["controlled_vehicle_crashes"] = [
            bool(vehicle.crashed) for vehicle in self.controlled_vehicles
        ]
        info["controlled_vehicle_on_road"] = [
            bool(getattr(vehicle, "on_road", True)) for vehicle in self.controlled_vehicles
        ]
        info["controlled_vehicle_offroad"] = [
            not bool(getattr(vehicle, "on_road", True)) for vehicle in self.controlled_vehicles
        ]
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
            "all_controlled_crashed": float(
                all(vehicle.crashed for vehicle in termination_vehicles)
            ) if termination_vehicles else 0.0,
        }

    def _reward(self, action: Any) -> float:
        return 0.0

    def _is_terminated(self) -> bool:
        termination_vehicles = self._termination_vehicles()
        if not termination_vehicles:
            return False

        if self.config.get("terminate_when_all_controlled_crashed", True):
            return all(vehicle.crashed for vehicle in termination_vehicles)
        return any(vehicle.crashed for vehicle in termination_vehicles)

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
                    a_i, a_cont_i, a_str_i, a_idx_i = self._resolve_expert_action(
                        vehicle=vehicle
                    )
                    resolved_actions.append(a_i)
                    expert_actions.append(
                        a_cont_i.copy() if a_cont_i is not None else None
                    )
                    expert_action_strs.append(a_str_i)
                    expert_action_idxs.append(a_idx_i)

                action = tuple(resolved_actions)
                if expert_actions:
                    expert_action = expert_actions[0]
                    expert_action_str = expert_action_strs[0]
                    expert_action_idx = expert_action_idxs[0]
            else:
                action, expert_action, expert_action_str, expert_action_idx = (
                    self._resolve_expert_action()
                )
                expert_actions = [
                    expert_action.copy() if expert_action is not None else None
                ]
                expert_action_strs = [expert_action_str]
                expert_action_idxs = [expert_action_idx]

        # -----------------------------------------------------------
        # EXECUTE SIMULATION STEP
        # -----------------------------------------------------------
        if self.scene_dataset_collection_mode and self.control_mode == "teleport":
            if self.road is None or self.vehicle is None:
                raise NotImplementedError(
                    "The road and vehicle must be initialized in the environment implementation"
                )

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
            int(getattr(vehicle, "vehicle_ID", -1))
            for vehicle in self.controlled_vehicles
        ]
        if isinstance(action, tuple):
            info["applied_actions"] = tuple(action)
        if expert_action is not None:
            info["expert_action_continuous"] = expert_action.copy()
        if expert_actions:
            info["expert_action_continuous_all"] = [
                a.copy() if a is not None else None for a in expert_actions
            ]
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
                expert_state = self._expert_state_by_vehicle_id.get(
                    int(self.vehicle.vehicle_ID)
                )
                if expert_state is not None:
                    self._replay_xy_pol = list(expert_state["replay_xy"])

        if self.scene_dataset_collection_mode and self.control_mode != "teleport":
            self._sync_scene_collection_controlled_vehicles(step_index=int(self.steps))
            info.update(self._info(obs, action))

        return obs, reward, terminated, truncated, info
