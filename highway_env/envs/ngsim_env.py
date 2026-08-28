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

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.ngsim_utils.core.config import (
    deep_update,
    interaction_metric_targets_from_idm,
    normalize_action_mode,
    resolve_idm_parameters,
)
from highway_env.ngsim_utils.core.constants import (
    ACCELERATION_RANGE,
    FEET_PER_METER,
    MANIFEST_RUNTIME_ENVIRONMENT_IDS,
    MAX_STEER,
)
from highway_env.ngsim_utils.data.ego_trajectory import (
    get_ego_dimensions,
    load_ego_trajectory,
    setup_expert_tracker,
)
from highway_env.ngsim_utils.data.episode_selection import (
    build_trajectory_set,
    select_ego_ids,
    select_episode_name,
)
from highway_env.ngsim_utils.data.prebuilt import (
    MANIFEST_CONTROLLED_WINDOW_FRAMES,
    DatasetManifestV2,
    EpisodeStoreV2,
    load_prebuilt_data,
    refine_valid_ids_by_episode,
    sha256_file,
)
from highway_env.ngsim_utils.data.trajectory_gen import (
    common_first_valid_index,
    first_valid_index,
    longest_continuous_active_span_bounds,
    trajectory_row_is_active,
    trajectory_step_speed_mps,
)
from highway_env.ngsim_utils.expert.ngsim_expert_mixin import NGSimExpertMixin
from highway_env.ngsim_utils.expert.trajectory_to_action import (
    PurePursuitTracker,
)
from highway_env.ngsim_utils.road.gen_road import create_japanese_road, create_ngsim_101_road
from highway_env.ngsim_utils.road.lane_mapping import (
    heading_from_trajectory_row,
    resolve_target_lane_index_from_row,
    target_lane_index_from_position_and_lane_id,
)
from highway_env.ngsim_utils.road.manifest_road import (
    RoadGeometryV3,
    build_road_network,
)
from highway_env.ngsim_utils.vehicles.ego import EgoVehicle
from highway_env.ngsim_utils.vehicles.ego_factory import build_ego_vehicle
from highway_env.ngsim_utils.vehicles.replay import (
    NGSIMVehicle,
    road_entity_conflicts_at_pose,
    spawn_surrounding_vehicles,
)
from highway_env.road.road import Road

logger = logging.getLogger(__name__)
SAMPLE_ANCHOR_V2_MINIMUM_FRAMES = 52
ROAD_BUILDERS = {
    "us-101": create_ngsim_101_road,
    "japanese": create_japanese_road,
}

VEHICLE_LIFECYCLE_CONTRACT_ID = "ngsim_vehicle_lifecycle_summary_v1"
VEHICLE_LIFECYCLE_EVENT_TYPES = (
    "source_before_active",
    "source_after_end",
    "controlled_spawn_conflict_deactivation",
    "background_initial_spawn_conflict",
    "background_delayed_spawn_conflict",
    "trajectory_exhaustion",
    "terminal_lane_removal",
    "road_end_completion",
    "offroad_crash",
    "collision_crash",
    "pruning",
    "idm_handover",
)


_PERSISTED_HEADING_KEYS = (
    "heading_rad",
    "heading_valid_mask",
    "heading_derivation",
)


def _persisted_heading_arrays(
    record: dict[str, Any], trajectory_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    present = [key in record for key in _PERSISTED_HEADING_KEYS]
    if any(present) and not all(present):
        missing = [key for key, is_present in zip(_PERSISTED_HEADING_KEYS, present) if not is_present]
        raise ValueError(f"Partial persisted-heading record; missing fields: {missing}.")
    if not any(present):
        return None
    headings = np.asarray(record["heading_rad"], dtype=np.float64)
    valid = np.asarray(record["heading_valid_mask"], dtype=bool)
    derivations = np.asarray(record["heading_derivation"], dtype=np.int8)
    expected = (int(trajectory_length),)
    if headings.shape != expected or valid.shape != expected or derivations.shape != expected:
        raise ValueError(
            "Persisted heading arrays must align one-to-one with trajectory rows: "
            f"heading={headings.shape}, valid={valid.shape}, derivation={derivations.shape}, "
            f"expected={expected}."
        )
    return headings, valid, derivations


def _persisted_heading_at(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    index: int,
) -> tuple[float, int] | None:
    if arrays is None:
        return None
    headings, valid, derivations = arrays
    if not bool(valid[index]):
        return None
    return float(headings[index]), int(derivations[index])


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
                # Additive, source-preserving contracts. When episode_store is
                # omitted, the historical eager NPY loader is used unchanged.
                "site_manifest": None,
                "episode_store": None,
                "road_geometry": None,
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
                # Fixed-horizon evaluators can retain collision/completion
                # state while suppressing Gym's terminal signal until the
                # configured truncation boundary.  Kept opt-in so historical
                # callers preserve their termination semantics.
                "suppress_controlled_vehicle_termination_until_truncation": False,
                "truncate_to_trajectory_length": False,  # allow for replay
                "scene_dataset_collection_mode": False,
                # Parent collectors may opt into source-anchor eligibility
                # semantics.  None preserves longest-continuous-span replay.
                "runtime_eligibility_mode": None,
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
                # Opt-in transition evidence for data/runtime qualification.
                # Disabled by default so legacy replay cost and info payloads
                # remain unchanged.
                "record_vehicle_lifecycle": False,
                # Source-faithful collection may retain background actors even
                # when noisy source poses overlap.  Default replay continues to
                # suppress unsafe initial and delayed activations.
                "disable_background_replay_spawn_safety": False,
                # Dataset-observation replay may retain logged background rows
                # past simplified terminal-lane geometry.  Default evaluation
                # still removes actors at terminal lanes.
                "source_faithful_background_replay": False,
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
        self._site_manifest: DatasetManifestV2 | None = None
        self._episode_store: EpisodeStoreV2 | None = None
        self._road_geometry_manifest: RoadGeometryV3 | None = None
        self._current_episode_data = None
        self._episode_timestep_s: float | None = None
        self._vehicle_lifecycle_enabled = False
        self._vehicle_lifecycle_events: list[dict[str, Any]] = []
        self._vehicle_lifecycle_once_keys: set[tuple[object, str, str]] = set()

        # Expert debug
        self._max_traj_policy_steps: int | None = None
        self._replay_xy_pol: list[np.ndarray] = []
        self._frames_per_action: int = 1

        runtime_contract_keys = ("site_manifest", "episode_store", "road_geometry")
        configured_contract_keys = {
            key for key in runtime_contract_keys if cfg.get(key) is not None
        }
        if configured_contract_keys and len(configured_contract_keys) != len(
            runtime_contract_keys
        ):
            missing = sorted(set(runtime_contract_keys).difference(configured_contract_keys))
            raise ValueError(
                "site_manifest, episode_store, and road_geometry form one atomic "
                f"runtime contract; missing configured keys: {missing}."
            )

        if cfg.get("site_manifest") is not None:
            self._site_manifest = DatasetManifestV2.from_source(cfg["site_manifest"])
            if self._site_manifest.scene != self.scene:
                raise ValueError(
                    "DatasetManifestV2 scene does not match NGSimEnv scene: "
                    f"{self._site_manifest.scene!r} != {self.scene!r}."
                )

        if cfg.get("road_geometry") is not None:
            self._road_geometry_manifest = RoadGeometryV3.from_source(
                cfg["road_geometry"]
            )
            if self._site_manifest is not None:
                self._road_geometry_manifest.validate_dataset(self._site_manifest)
                expected_road_path = self._site_manifest.road_geometry_path()
                if (
                    expected_road_path is not None
                    and self._road_geometry_manifest.path is not None
                    and expected_road_path != self._road_geometry_manifest.path
                ):
                    raise ValueError(
                        "Configured road_geometry is not the artifact referenced by "
                        "DatasetManifestV2."
                    )
                declared_road_sha256 = self._site_manifest.road_geometry_sha256()
                if declared_road_sha256 is not None:
                    if self._road_geometry_manifest.path is None:
                        raise ValueError(
                            "Hashed road_geometry binding requires a configured file path."
                        )
                    actual_road_sha256 = sha256_file(
                        self._road_geometry_manifest.path
                    )
                    if actual_road_sha256 != declared_road_sha256:
                        raise ValueError(
                            "Configured road_geometry SHA-256 does not match "
                            "DatasetManifestV2: "
                            f"{actual_road_sha256} != {declared_road_sha256}."
                        )

        if cfg.get("episode_store") is not None:
            self._episode_store = EpisodeStoreV2(cfg["episode_store"])
            requested_split = str(cfg.get("prebuilt_split", "train"))
            if self._episode_store.split != requested_split:
                raise ValueError(
                    "EpisodeStoreV2 split does not match prebuilt_split: "
                    f"{self._episode_store.split!r} != {requested_split!r}."
                )
            if self._site_manifest is not None and (
                self._episode_store.site_id != self._site_manifest.site_id
            ):
                raise ValueError(
                    "DatasetManifestV2 and EpisodeStoreV2 site_id values do not match."
                )
            if self._site_manifest is not None:
                expected_store_path = self._site_manifest.episode_store_path(
                    requested_split
                )
                if (
                    expected_store_path is not None
                    and self._episode_store.manifest_path is not None
                    and expected_store_path != self._episode_store.manifest_path
                ):
                    raise ValueError(
                        "Configured episode_store is not the split artifact referenced "
                        "by DatasetManifestV2."
                    )
                declared_store_sha256 = self._site_manifest.episode_store_sha256(
                    requested_split
                )
                if declared_store_sha256 is not None:
                    if self._episode_store.manifest_path is None:
                        raise ValueError(
                            "Hashed episode_store binding requires a configured file path."
                        )
                    actual_store_sha256 = sha256_file(
                        self._episode_store.manifest_path
                    )
                    if actual_store_sha256 != declared_store_sha256:
                        raise ValueError(
                            "Configured episode_store SHA-256 does not match "
                            "DatasetManifestV2: "
                            f"{actual_store_sha256} != {declared_store_sha256}."
                        )
            if self._road_geometry_manifest is not None and (
                self._episode_store.site_id != self._road_geometry_manifest.site_id
            ):
                raise ValueError(
                    "EpisodeStoreV2 and RoadGeometryV3 site_id values do not match."
                )
            environment_ids = (
                self._site_manifest.environment_id,
                self._episode_store.environment_id,
                self._road_geometry_manifest.environment_id,
            )
            if len(set(environment_ids)) != 1:
                raise ValueError(
                    "DatasetManifestV2, EpisodeStoreV2, and RoadGeometryV3 "
                    f"environment_id values do not match: {environment_ids!r}."
                )
            bound_environment_id = environment_ids[0]
            if (
                self.scene in MANIFEST_RUNTIME_ENVIRONMENT_IDS
                or bound_environment_id in MANIFEST_RUNTIME_ENVIRONMENT_IDS
            ) and bound_environment_id != self.scene:
                raise ValueError(
                    "Five-environment manifest environment_id must equal NGSimEnv "
                    f"scene: {bound_environment_id!r} != {self.scene!r}."
                )
            if not self._episode_store.episode_ids:
                raise ValueError("EpisodeStoreV2 contains no episodes for reset.")
            self._prebuilt_dir = str(self._episode_store.root)
            self._data_binding_id = (
                f"episode_store_v2:{self._episode_store.canonical_sha256}"
            )
            self._traj_all_by_episode: dict[str, dict[Any, Any]] = {}
            self._episodes = self._episode_store.episode_ids
        else:
            (
                self._prebuilt_dir,
                self._valid_ids_by_episode,
                self._traj_all_by_episode,
                self._episodes,
            ) = load_prebuilt_data(
                cfg["episode_root"],
                self.scene,
                str(cfg.get("prebuilt_split", "train")),
                min_occupancy=float(
                    cfg.get("controlled_vehicle_min_occupancy", 0.8)
                ),
                cache=self._PREBUILT_CACHE,
            )
            self._data_binding_id = self._prebuilt_dir

        self._runtime_contract_config = {
            key: deepcopy(cfg.get(key))
            for key in ("site_manifest", "episode_store", "road_geometry")
        }

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

    def _reset_vehicle_lifecycle(self) -> None:
        self._vehicle_lifecycle_enabled = bool(
            self.config.get("record_vehicle_lifecycle", False)
        )
        self._vehicle_lifecycle_events = []
        self._vehicle_lifecycle_once_keys = set()

    def _vehicle_lifecycle_environment_id(self) -> str:
        manifest = getattr(self, "_site_manifest", None)
        return str(
            getattr(manifest, "environment_id", None)
            or getattr(self, "scene", "unknown")
        )

    def _vehicle_lifecycle_role(self, vehicle: object | None) -> str:
        if vehicle is not None and any(
            candidate is vehicle
            for candidate in getattr(self, "controlled_vehicles", ())
        ):
            return "controlled"
        return "background"

    def _record_vehicle_lifecycle_event(
        self,
        event_type: str,
        *,
        vehicle: object | None = None,
        vehicle_id: int | None = None,
        actor_role: str | None = None,
        source_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not bool(getattr(self, "_vehicle_lifecycle_enabled", False)):
            return
        if event_type not in VEHICLE_LIFECYCLE_EVENT_TYPES:
            raise ValueError(f"Unknown vehicle lifecycle event: {event_type!r}.")
        role = str(actor_role or self._vehicle_lifecycle_role(vehicle))
        if role not in {"controlled", "background"}:
            raise ValueError(f"Unknown vehicle lifecycle actor role: {role!r}.")
        resolved_vehicle_id = (
            getattr(vehicle, "vehicle_ID", None)
            if vehicle_id is None
            else vehicle_id
        )
        resolved_vehicle_id = (
            None if resolved_vehicle_id is None else int(resolved_vehicle_id)
        )
        identity_key: object = (
            ("vehicle_id", resolved_vehicle_id)
            if resolved_vehicle_id is not None
            else ("object_id", id(vehicle))
        )
        once_key = (identity_key, role, event_type)
        if once_key in self._vehicle_lifecycle_once_keys:
            return
        self._vehicle_lifecycle_once_keys.add(once_key)
        event: dict[str, Any] = {
            "sequence": len(self._vehicle_lifecycle_events),
            "event_type": event_type,
            "environment_id": self._vehicle_lifecycle_environment_id(),
            "episode_id": (
                None
                if getattr(self, "episode_name", None) is None
                else str(self.episode_name)
            ),
            "simulation_step": int(getattr(self, "steps", 0)),
            "actor_role": role,
            "vehicle_id": resolved_vehicle_id,
        }
        if source_index is not None:
            event["source_index"] = int(source_index)
        if details:
            event["details"] = deepcopy(details)
        self._vehicle_lifecycle_events.append(event)

    def vehicle_lifecycle_summary(
        self,
        *,
        include_events: bool = True,
    ) -> dict[str, Any]:
        """Return deterministic, JSON-safe lifecycle evidence for the episode.

        Enable collection with ``record_vehicle_lifecycle``.  Per-step ``info``
        carries this contract without the event list; receipt writers should
        call this method with ``include_events=True`` after the rollout.
        """
        event_counts = {event_type: 0 for event_type in VEHICLE_LIFECYCLE_EVENT_TYPES}
        role_counts = {
            role: {event_type: 0 for event_type in VEHICLE_LIFECYCLE_EVENT_TYPES}
            for role in ("controlled", "background")
        }
        events = list(getattr(self, "_vehicle_lifecycle_events", ()))
        for event in events:
            event_type = str(event["event_type"])
            role = str(event["actor_role"])
            event_counts[event_type] += 1
            role_counts[role][event_type] += 1
        return {
            "schema_version": 1,
            "contract_id": VEHICLE_LIFECYCLE_CONTRACT_ID,
            "event_cardinality": "once_per_vehicle_role_event_type",
            "enabled": bool(getattr(self, "_vehicle_lifecycle_enabled", False)),
            "environment_id": self._vehicle_lifecycle_environment_id(),
            "episode_id": (
                None
                if getattr(self, "episode_name", None) is None
                else str(self.episode_name)
            ),
            "simulation_steps": int(getattr(self, "steps", 0)),
            "data_binding_id": (
                None
                if getattr(self, "_data_binding_id", None) is None
                else str(self._data_binding_id)
            ),
            "runtime_policy": {
                "record_vehicle_lifecycle": bool(
                    self.config.get("record_vehicle_lifecycle", False)
                ),
                "disable_background_replay_spawn_safety": bool(
                    self.config.get(
                        "disable_background_replay_spawn_safety",
                        False,
                    )
                ),
                "source_faithful_background_replay": bool(
                    self.config.get("source_faithful_background_replay", False)
                ),
                "disable_scene_collection_spawn_safety": bool(
                    self.config.get("disable_scene_collection_spawn_safety", False)
                ),
                "scene_dataset_collection_mode": bool(
                    self.config.get("scene_dataset_collection_mode", False)
                ),
                "allow_idm": bool(self.config.get("allow_idm", True)),
                "complete_controlled_vehicles_at_road_end": bool(
                    self.config.get(
                        "complete_controlled_vehicles_at_road_end",
                        True,
                    )
                ),
                "crash_controlled_vehicles_offroad": bool(
                    self.config.get("crash_controlled_vehicles_offroad", True)
                ),
            },
            "event_count": len(events),
            "event_counts": event_counts,
            "actor_role_event_counts": role_counts,
            "events_included": bool(include_events),
            "events": (
                deepcopy(events) if include_events else []
            ),
        }

    def manifest_runtime_diagnostics(self) -> dict[str, Any] | None:
        """Return selected-episode contract counts without opening other shards.

        This is deliberately observational: it exposes producer masks and
        reference metadata exactly as loaded and does not create a new
        eligibility rule or apply a coverage threshold.
        """

        episode = self._current_episode_data
        store = self._episode_store
        if episode is None or store is None:
            return None
        active_count = int(np.count_nonzero(episode.active_mask))
        diagnostics: dict[str, Any] = {
            "environment_id": store.environment_id,
            "site_id": store.site_id,
            "episode_id": episode.entry.episode_id,
            "loaded_episode_ids": store.loaded_episode_ids,
            "state_position_reference": store.state_position_reference,
            "source_position_reference": store.source_position_reference,
            "heading_contract": store.heading_contract,
            "mask_contract": store.mask_contract,
            "active_state_count": active_count,
        }
        if episode.road_valid_mask is not None:
            road_valid_count = int(np.count_nonzero(episode.road_valid_mask))
            training_eligible_count = int(
                np.count_nonzero(episode.training_eligible_mask)
            )
            diagnostics.update(
                {
                    "road_valid_state_count": road_valid_count,
                    "road_valid_active_rate": (
                        float(road_valid_count) / active_count if active_count else None
                    ),
                    "training_eligible_state_count": training_eligible_count,
                    "training_eligible_active_rate": (
                        float(training_eligible_count) / active_count
                        if active_count
                        else None
                    ),
                }
            )
        if episode.controlled_vehicle_eligible_mask is not None:
            diagnostics["controlled_vehicle_eligible_count"] = int(
                np.count_nonzero(episode.controlled_vehicle_eligible_mask)
            )
        surface = getattr(getattr(self, "viewer", None), "sim_surface", None)
        if surface is not None and hasattr(surface, "pos2pix"):
            origin = surface.pos2pix(0.0, 0.0)
            x_unit = surface.pos2pix(1.0, 0.0)
            y_unit = surface.pos2pix(0.0, 1.0)
            x_pixels = abs(int(x_unit[0]) - int(origin[0]))
            y_pixels = abs(int(y_unit[1]) - int(origin[1]))
            diagnostics["render_equal_aspect"] = abs(x_pixels - y_pixels) <= 1
            diagnostics["render_pixels_per_meter_xy"] = (x_pixels, y_pixels)
        else:
            diagnostics["render_equal_aspect"] = None
        return diagnostics

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
        current_contract_config = {
            key: self.config.get(key)
            for key in ("site_manifest", "episode_store", "road_geometry")
        }
        if current_contract_config != self._runtime_contract_config:
            raise RuntimeError(
                "site_manifest, episode_store, and road_geometry are bound during "
                "NGSimEnv construction; create a new environment to change them."
        )
        self.steps = 0
        self._reset_vehicle_lifecycle()
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
        removed = [
            vehicle
            for vehicle in self.road.vehicles
            if getattr(vehicle, "remove_from_road", False)
        ]
        if not removed:
            return
        for vehicle in removed:
            self._record_vehicle_lifecycle_event(
                "pruning",
                vehicle=vehicle,
                details={
                    "removal_reason": getattr(
                        vehicle,
                        "lifecycle_removal_reason",
                        None,
                    )
                },
            )
        self.road.vehicles = [
            vehicle for vehicle in self.road.vehicles if not getattr(vehicle, "remove_from_road", False)
        ]

    def _record_new_collision_crashes(
        self,
        crashed_before_step: dict[int, bool],
    ) -> None:
        for vehicle in list(getattr(self.road, "vehicles", ())):
            if crashed_before_step.get(id(vehicle), False) or not bool(
                getattr(vehicle, "crashed", False)
            ):
                continue
            partner_vehicle_id = getattr(
                vehicle,
                "first_collision_partner_vehicle_id",
                None,
            )
            self._record_vehicle_lifecycle_event(
                "collision_crash",
                vehicle=vehicle,
                details={
                    "partner_vehicle_id": (
                        None
                        if partner_vehicle_id is None
                        else int(partner_vehicle_id)
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
                },
            )

    def _simulate(self, action: Action | None = None) -> None:
        """Run simulation frames and prune replay vehicles that have despawned."""
        frames = self._frames_per_action
        dt = 1 / self.config["simulation_frequency"]
        for frame in range(frames):
            if action is not None and not self.config["manual_control"] and self.steps % frames == 0:
                self.action_type.act(action)

            self.road.act()
            crashed_before_step = (
                {
                    id(vehicle): bool(getattr(vehicle, "crashed", False))
                    for vehicle in self.road.vehicles
                }
                if bool(getattr(self, "_vehicle_lifecycle_enabled", False))
                else None
            )
            self.road.step(dt)
            if crashed_before_step is not None:
                self._record_new_collision_crashes(crashed_before_step)
            self._complete_road_end_controlled_vehicles()
            self._crash_offroad_controlled_vehicles()
            self._prune_removed_vehicles()
            self.steps += 1
            self._sync_source_faithful_background_vehicles(step_index=self.steps)

            if frame < frames - 1:
                self._automatic_rendering()

        self.enable_auto_render = False

    def _sync_source_faithful_background_vehicles(self, step_index: int) -> None:
        """Make background replay state match the frame being observed."""
        if not bool(self.config.get("source_faithful_background_replay", False)):
            return
        controlled_identity = {id(vehicle) for vehicle in self.controlled_vehicles}
        for vehicle in self.road.vehicles:
            if id(vehicle) in controlled_identity or not isinstance(vehicle, NGSIMVehicle):
                continue
            # ``allow_idm`` defines a hybrid replay contract: an actor follows
            # exact source rows until interaction requires an IDM/MOBIL
            # takeover.  Once that irreversible handover (or a collision) has
            # happened, source synchronization must not teleport the actor
            # back onto its logged trajectory.
            if bool(getattr(vehicle, "allow_idm", False)) and (
                bool(getattr(vehicle, "overtaken", False))
                or bool(getattr(vehicle, "crashed", False))
            ):
                continue
            trajectory = getattr(vehicle, "ngsim_traj", None)
            if trajectory is None or step_index < 0 or step_index >= len(trajectory):
                continue
            vehicle.synchronize_source_replay(step_index)

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
                if bool(getattr(self, "_vehicle_lifecycle_enabled", False)):
                    vehicle.lifecycle_removal_reason = "road_end_completion"
                vehicle.collidable = False
                vehicle.check_collisions = False
                self._record_vehicle_lifecycle_event(
                    "road_end_completion",
                    vehicle=vehicle,
                    actor_role="controlled",
                    details={"lane_index": str(getattr(vehicle, "lane_index", None))},
                )

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
                record_lifecycle = getattr(
                    self, "_record_vehicle_lifecycle_event", None
                )
                if callable(record_lifecycle):
                    record_lifecycle(
                        "offroad_crash",
                        vehicle=vehicle,
                        actor_role="controlled",
                        details={
                            "lane_index": str(getattr(vehicle, "lane_index", None))
                        },
                    )

    # -------------------------------------------------------------------------
    # LOAD TRAJECTORY
    # -------------------------------------------------------------------------
    def _load_trajectory(self):
        sim_period = self.config.get("simulation_period", None)
        explicit_ego_id = self.config.get("ego_vehicle_ID", None)
        episode_store = getattr(self, "_episode_store", None)

        self.episode_name = select_episode_name(
            sim_period,
            (
                episode_store
                if episode_store is not None
                else self._traj_all_by_episode
            ),
            self._episodes,
            self.np_random,
        )
        if episode_store is not None:
            episode = episode_store.load_episode(self.episode_name)
            self._current_episode_data = episode
            if len(episode.timestamps_ms) < 2:
                raise ValueError(
                    "NGSimEnv requires at least two EpisodeStoreV2 timestamps."
                )
            timestamp_deltas_ms = np.diff(episode.timestamps_ms)
            if not np.all(timestamp_deltas_ms == timestamp_deltas_ms[0]):
                raise ValueError(
                    "NGSimEnv requires uniformly sampled EpisodeStoreV2 timestamps."
                )
            self._episode_timestep_s = float(timestamp_deltas_ms[0]) / 1000.0
            simulator_timestep_s = 1.0 / float(
                self.config["simulation_frequency"]
            )
            if not np.isclose(
                self._episode_timestep_s,
                simulator_timestep_s,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "EpisodeStoreV2 timestamp cadence does not match "
                    "simulation_frequency: "
                    f"{self._episode_timestep_s}s != {simulator_timestep_s}s."
                )
            episode_records = episode.as_legacy_trajectory_dict()
            if episode_store.environment_id in MANIFEST_RUNTIME_ENVIRONMENT_IDS:
                explicit_ids = (
                    []
                    if explicit_ego_id is None
                    else np.asarray(explicit_ego_id).reshape(-1).astype(np.int64).tolist()
                )
                sample_anchor_explicit_ids = bool(explicit_ids) and (
                    self.config.get("runtime_eligibility_mode")
                    == "sample_anchor_v2"
                )
                minimum_frames = (
                    SAMPLE_ANCHOR_V2_MINIMUM_FRAMES
                    if sample_anchor_explicit_ids
                    else MANIFEST_CONTROLLED_WINDOW_FRAMES
                )
                if len(episode.timestamps_ms) < minimum_frames:
                    replay_contract = (
                        "sample_anchor_v2"
                        if sample_anchor_explicit_ids
                        else "the legacy controlled window"
                    )
                    raise ValueError(
                        "Five-environment controlled replay requires at least "
                        f"{minimum_frames} frames for {replay_contract}; episode "
                        f"{self.episode_name!r} has {len(episode.timestamps_ms)}."
                    )
                window = slice(
                    0,
                    min(
                        MANIFEST_CONTROLLED_WINDOW_FRAMES,
                        len(episode.timestamps_ms),
                    ),
                )
                controlled_valid = (
                    episode.active_mask[window]
                    & episode.road_valid_mask[window]
                    & episode.heading_valid_mask[window]
                )
                valid_columns = np.all(controlled_valid, axis=0)
                valid_columns &= episode.controlled_vehicle_eligible_mask
                valid_ids = episode.vehicle_ids[valid_columns]
                eligible_explicit_ids = (
                    episode.vehicle_ids
                    if sample_anchor_explicit_ids
                    else valid_ids
                )
                invalid_explicit = sorted(
                    set(int(vehicle_id) for vehicle_id in explicit_ids).difference(
                        int(vehicle_id) for vehicle_id in eligible_explicit_ids
                    )
                )
                if invalid_explicit:
                    if sample_anchor_explicit_ids:
                        raise ValueError(
                            "sample_anchor_v2 explicit ego_vehicle_ID values must "
                            "be present in the selected EpisodeStore episode; "
                            f"missing IDs: {invalid_explicit}."
                        )
                    raise ValueError(
                        "Explicit ego_vehicle_ID values must be active, road-valid, "
                        "heading-valid, and uniquely core-anchor eligible for the "
                        "complete 200-frame controlled "
                        f"window; invalid IDs: {invalid_explicit}."
                    )
                if sample_anchor_explicit_ids:
                    accepted_ids = {
                        int(vehicle_id) for vehicle_id in valid_ids
                    }
                    accepted_ids.update(
                        int(vehicle_id) for vehicle_id in explicit_ids
                    )
                    valid_ids = episode.vehicle_ids[
                        np.asarray(
                            [
                                int(vehicle_id) in accepted_ids
                                for vehicle_id in episode.vehicle_ids
                            ],
                            dtype=bool,
                        )
                    ]
            else:
                valid_ids = refine_valid_ids_by_episode(
                    {self.episode_name: episode.vehicle_ids},
                    {self.episode_name: episode_records},
                    min_occupancy=float(
                        self.config.get("controlled_vehicle_min_occupancy", 0.8)
                    ),
                    data_dt=self._episode_timestep_s,
                )[self.episode_name]
            self._valid_ids_by_episode[self.episode_name] = valid_ids
        else:
            self._current_episode_data = None
            episode_records = self._traj_all_by_episode[self.episode_name]
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
            {self.episode_name: episode_records},
            self.episode_name,
            self.ego_ids,
        )

        logger.info("Loaded episode=%s ego_ids=%s", self.episode_name, self.ego_ids)

    # -------------------------------------------------------------------------
    # ROAD + VEHICLES + Test Mode
    # -------------------------------------------------------------------------
    def _create_road(self):
        builder = ROAD_BUILDERS.get(self.scene)
        road_geometry_manifest = getattr(self, "_road_geometry_manifest", None)
        if builder is None and road_geometry_manifest is None:
            raise ValueError(
                f"Unsupported scene={self.scene!r} without RoadGeometryV3."
            )
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
        if self.scene == "japanese" and road_geometry_manifest is None:
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
        geometry_identity = (
            f"road_geometry_v3:{road_geometry_manifest.canonical_sha256}"
            if road_geometry_manifest is not None
            else str(geometry_path) if geometry_path is not None else "legacy"
        )
        cache_key = (self.scene, geometry_identity)
        net = self._NETWORK_CACHE.get(cache_key)
        if net is None:
            if road_geometry_manifest is not None:
                net = build_road_network(road_geometry_manifest)
            elif self.scene == "japanese" and geometry_path is not None:
                net = create_japanese_road(geometry_path)
            else:
                net = builder()
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
        self.road.vehicle_lifecycle_event_recorder = (
            self._record_vehicle_lifecycle_event
            if bool(getattr(self, "_vehicle_lifecycle_enabled", False))
            else None
        )
        self.road.disable_background_replay_spawn_safety = bool(
            self.config.get("disable_background_replay_spawn_safety", False)
        )
        self.road.source_faithful_background_replay = bool(
            self.config.get("source_faithful_background_replay", False)
        )
        self.road.debug_idm_handover = bool(self.config.get("debug_idm_handover", False))
        debug_ids = self.config.get("debug_idm_handover_ids")
        self.road.debug_idm_handover_ids = {int(vehicle_id) for vehicle_id in debug_ids} if debug_ids else None

    def _manifest_lane_index_from_row(
        self,
        row: np.ndarray,
    ) -> tuple[str, str, int] | None:
        raw_lane_indexes = getattr(
            self.road.network, "manifest_raw_lane_indexes", None
        )
        if not isinstance(raw_lane_indexes, dict):
            return None
        values = np.asarray(row, dtype=float)
        if values.shape[0] < 4 or not np.all(np.isfinite(values[:4])):
            return None
        return target_lane_index_from_position_and_lane_id(
            self.road.network,
            self.scene,
            values[:2],
            int(values[3]),
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

        for ego_index, ego_id in enumerate(self.ego_ids):
            ego_rec = ego_records[ego_id]
            ego_traj_full = self._load_processed_ego_trajectory(ego_id, ego_rec)
            if getattr(self, "_episode_store", None) is not None:
                ego_len, ego_wid = float(ego_rec["length"]), float(ego_rec["width"])
            else:
                ego_len, ego_wid = get_ego_dimensions(
                    ego_rec, FEET_PER_METER, self.scene
                )
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
            manifest_lane_index = self._manifest_lane_index_from_row(ego_traj[0])
            if manifest_lane_index is not None:
                ego.target_lane_index = manifest_lane_index
                ego.lane_index = manifest_lane_index
                ego.lane = self.road.network.get_lane(manifest_lane_index)
                longitudinal, _ = ego.lane.local_coordinates(ego.position)
                ego.heading = float(ego.lane.heading_at(longitudinal))
            persisted_heading = _persisted_heading_at(
                _persisted_heading_arrays(ego_rec, len(ego_traj_full)),
                ego_start_index,
            )
            if persisted_heading is not None:
                ego.heading, ego.current_heading_derivation = persisted_heading
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
                self._record_vehicle_lifecycle_event(
                    "controlled_spawn_conflict_deactivation",
                    vehicle=ego,
                    actor_role="controlled",
                    source_index=ego_start_index,
                    details={"activation_stage": "initial"},
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
        if self.expert_test_mode or (
            self.scene_dataset_collection_mode
            and self.config.get("runtime_eligibility_mode") == "sample_anchor_v2"
        ):
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
            sample_anchor_replay = (
                self.control_mode == "teleport"
                and self.config.get("runtime_eligibility_mode")
                == "sample_anchor_v2"
            )
            if sample_anchor_replay:
                active_indices = [
                    index
                    for index, row in enumerate(ego_traj_full)
                    if trajectory_row_is_active(row)
                ]
                start_idx = active_indices[0] if active_indices else None
                end_idx = active_indices[-1] if active_indices else None
                span_len = len(active_indices)
            else:
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
        persisted_heading_arrays = _persisted_heading_arrays(
            ego_rec, len(ego_traj_full)
        )
        if persisted_heading_arrays is None:
            ego.scene_collection_heading_rad = None
            ego.scene_collection_heading_valid_mask = None
            ego.scene_collection_heading_derivation = None
        else:
            headings, valid, derivations = persisted_heading_arrays
            ego.scene_collection_heading_rad = headings.copy()
            ego.scene_collection_heading_valid_mask = valid.copy()
            ego.scene_collection_heading_derivation = derivations.copy()
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
        persisted_heading: tuple[float, int] | None = None,
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
        if persisted_heading is not None:
            ego.heading, ego.scene_collection_current_heading_derivation = persisted_heading
        else:
            ego.heading = self._heading_for_spawn_row(
                np.asarray(row, dtype=float),
                previous_row=(None if previous_row is None else np.asarray(previous_row, dtype=float)),
                next_row=None if next_row is None else np.asarray(next_row, dtype=float),
                fallback_heading=fallback_heading,
                prefer_motion=self.control_mode == "teleport",
                lane_index_override=mapped_lane_index,
            )
            ego.scene_collection_current_heading_derivation = None

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
        persisted_heading: tuple[float, int] | None = None,
    ) -> bool:
        row_arr = np.asarray(row, dtype=float)
        heading = (
            float(persisted_heading[0])
            if persisted_heading is not None
            else self._heading_for_spawn_row(
                row_arr,
                previous_row=previous_row,
                next_row=next_row,
                fallback_heading=float(getattr(ego, "heading", 0.0)),
            )
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
        heading_arrays = None
        if getattr(ego, "scene_collection_heading_rad", None) is not None:
            heading_arrays = (
                ego.scene_collection_heading_rad,
                ego.scene_collection_heading_valid_mask,
                ego.scene_collection_heading_derivation,
            )
        persisted_heading = _persisted_heading_at(heading_arrays, step_index)
        conflict_kwargs = {
            "previous_row": previous_row,
            "next_row": next_row,
        }
        # Preserve the legacy scene-collection call surface when no persisted
        # heading exists.  Several downstream integrations replace these
        # helpers with callables that implement the historical keyword set;
        # manifest-driven episodes still receive the persisted heading
        # explicitly whenever one is present.
        if persisted_heading is not None:
            conflict_kwargs["persisted_heading"] = persisted_heading
        if not bool(
            self.config.get("disable_scene_collection_spawn_safety", False)
        ) and self._scene_collection_row_has_conflict(
            ego,
            row,
            **conflict_kwargs,
        ):
            self._record_vehicle_lifecycle_event(
                "controlled_spawn_conflict_deactivation",
                vehicle=ego,
                actor_role="controlled",
                source_index=step_index,
                details={
                    "activation_stage": (
                        "initial"
                        if step_index
                        == int(getattr(ego, "scene_collection_start_index", 0))
                        else "delayed"
                    )
                },
            )
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
        activation_kwargs = {
            "previous_row": previous_row,
            "next_row": next_row,
            "provider_observation_flag": (
                None
                if getattr(
                    ego,
                    "scene_collection_provider_observation_mask",
                    None,
                )
                is None
                else int(ego.scene_collection_provider_observation_mask[step_index])
            ),
        }
        if persisted_heading is not None:
            activation_kwargs["persisted_heading"] = persisted_heading
        self._set_scene_collection_vehicle_from_row(
            ego,
            row,
            **activation_kwargs,
        )

    def _deactivate_scene_collection_vehicle(self, ego: EgoVehicle) -> None:
        ego.scene_collection_is_active = False
        ego.scene_collection_current_provider_lane_reconciled = False
        ego.scene_collection_current_heading_derivation = None
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
                self._record_vehicle_lifecycle_event(
                    (
                        "source_before_active"
                        if step_index < start_index
                        else "source_after_end"
                    ),
                    vehicle=ego,
                    actor_role="controlled",
                    source_index=step_index,
                    details={
                        "source_start_index": start_index,
                        "source_end_index": end_index,
                    },
                )
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
        data_binding_id = getattr(self, "_data_binding_id", self._prebuilt_dir)
        return (data_binding_id, self.scene, episode_name, int(vehicle_id))

    def _load_processed_ego_trajectory(self, ego_id: int, ego_rec: dict[str, Any]) -> np.ndarray:
        cache_key = self._processed_trajectory_cache_key(self.episode_name, ego_id)
        cached = self._PROCESSED_TRAJECTORY_CACHE.get(cache_key)
        if cached is None:
            if getattr(self, "_episode_store", None) is not None:
                cached = np.asarray(ego_rec["trajectory"], dtype=float)
            else:
                cached = load_ego_trajectory(ego_rec, self.scene)
            self._PROCESSED_TRAJECTORY_CACHE[cache_key] = cached
        return cached

    def _expert_reference_cache_key(self, ego_id: int, ego_len: float) -> tuple[str, str, str, int, int, float]:
        return (
            getattr(self, "_data_binding_id", self._prebuilt_dir),
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

        if getattr(self, "_episode_store", None) is not None:
            self._spawn_si_surrounding_vehicles(max_surr)
            return

        spawn_surrounding_vehicles(
            self.trajectory_set,
            self._ego_start_indices,
            max_surr,
            self.road,
            scene=self.scene,
            allow_idm=bool(self.config.get("allow_idm", True)),
        )

    def _spawn_si_surrounding_vehicles(self, max_surrounding: int | None) -> None:
        """Spawn EpisodeStoreV2 actors without applying legacy unit conversion."""

        shared_start_index = (
            max(int(index) for index in self._ego_start_indices.values())
            if self._ego_start_indices
            else 0
        )
        ego_anchor_positions: list[np.ndarray] = []
        for ego_meta in self.trajectory_set.get("ego", {}).values():
            ego_traj_full = np.asarray(ego_meta["trajectory"], dtype=float)
            if len(ego_traj_full) <= shared_start_index:
                continue
            ego_traj = ego_traj_full[shared_start_index:]
            ego_first_index = first_valid_index(ego_traj)
            if ego_first_index is not None:
                ego_anchor_positions.append(
                    np.asarray(ego_traj[ego_first_index, :2], dtype=float)
                )

        candidates: list[tuple[float, int, dict[str, Any], np.ndarray, int]] = []
        for vehicle_id, metadata in self.trajectory_set.items():
            if vehicle_id == "ego":
                continue
            trajectory_full = np.asarray(metadata["trajectory"], dtype=float)
            if len(trajectory_full) <= shared_start_index:
                continue
            trajectory = trajectory_full[shared_start_index:]
            first_index = first_valid_index(trajectory)
            if first_index is None:
                continue
            first_position = np.asarray(trajectory[first_index, :2], dtype=float)
            priority = (
                min(
                    float(np.linalg.norm(first_position - ego_position))
                    for ego_position in ego_anchor_positions
                )
                if ego_anchor_positions
                else float("inf")
            )
            candidates.append(
                (priority, int(vehicle_id), metadata, trajectory, first_index)
            )

        if max_surrounding is not None:
            candidates.sort(key=lambda item: (item[0], item[1]))
        spawned = 0
        for _priority, vehicle_id, metadata, trajectory, first_index in candidates:
            if max_surrounding is not None and spawned >= max_surrounding:
                break
            spawn_row = np.asarray(trajectory[first_index], dtype=float)
            lane_index = self._manifest_lane_index_from_row(spawn_row)
            if lane_index is None:
                lane_index = self.road.network.get_closest_lane_index(spawn_row[:2])
            lane = self.road.network.get_lane(lane_index)
            longitudinal, _ = lane.local_coordinates(spawn_row[:2])
            spawn_heading = float(lane.heading_at(longitudinal))
            persisted_heading_arrays = _persisted_heading_arrays(
                metadata, len(trajectory_full)
            )
            replay_headings = None
            replay_heading_valid = None
            replay_heading_derivations = None
            if persisted_heading_arrays is not None:
                replay_headings = persisted_heading_arrays[0][shared_start_index:]
                replay_heading_valid = persisted_heading_arrays[1][shared_start_index:]
                replay_heading_derivations = persisted_heading_arrays[2][shared_start_index:]
                initial_persisted_heading = _persisted_heading_at(
                    (replay_headings, replay_heading_valid, replay_heading_derivations),
                    first_index,
                )
                if initial_persisted_heading is not None:
                    spawn_heading = initial_persisted_heading[0]
            vehicle = NGSIMVehicle.create(
                road=self.road,
                vehicle_ID=vehicle_id,
                position=trajectory[0, :2],
                v_length=float(metadata["length"]),
                v_width=float(metadata["width"]),
                ngsim_traj=trajectory,
                scene=self.scene,
                heading=spawn_heading,
                speed=float(trajectory[0, 2]),
                color=(200, 0, 150),
                allow_idm=bool(self.config.get("allow_idm", True)),
                ngsim_heading_rad=replay_headings,
                ngsim_heading_valid_mask=replay_heading_valid,
                ngsim_heading_derivation=replay_heading_derivations,
            )
            vehicle.DATA_DT = float(self._episode_timestep_s)
            vehicle.lane_index = lane_index
            vehicle.lane = lane
            next_row = (
                np.asarray(trajectory[first_index + 1], dtype=float)
                if first_index + 1 < len(trajectory)
                else None
            )
            if not vehicle._spawn_row_is_clear(
                spawn_row, next_row=next_row, row_index=first_index
            ):
                self._record_vehicle_lifecycle_event(
                    "background_initial_spawn_conflict",
                    vehicle=vehicle,
                    actor_role="background",
                    source_index=first_index,
                )
                continue
            self.road.vehicles.append(vehicle)
            spawned += 1

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
        if bool(getattr(self, "_vehicle_lifecycle_enabled", False)):
            info["vehicle_lifecycle_summary"] = self.vehicle_lifecycle_summary(
                include_events=False
            )
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
        if bool(
            self.config.get(
                "suppress_controlled_vehicle_termination_until_truncation", False
            )
        ):
            return False
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
