"""Stable simulator and trajectory-data API for downstream research.

The HighwayEnv fork retains its upstream-compatible internal layout.  Parent
research code should import the small surface below instead of binding to
implementation modules under ``ngsim_utils`` or ``imitation``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.common.observations.factory import observation_factory
from highway_env.imitation.behavior_intent import (
    BEHAVIOR_INTENT_DIM,
    MANEUVER_FAMILY_DIM,
    BehaviorCommandSchedulerConfig,
    BehaviorFeasibility,
    BehaviorIntent,
    BehaviorLabelerConfig,
    BehaviorLabels,
    ManeuverFamilyIntent,
    ProspectiveBehaviorCommandScheduler,
    behavior_intent_contract,
    behavior_intent_one_hot,
    behavior_to_maneuver_family_ids,
    current_vehicle_behavior_feasibility,
    label_behavior_rows,
    maneuver_family_contract,
    maneuver_family_one_hot,
)
from highway_env.imitation.expert_dataset import (
    ENV_ID,
    build_env_config,
    default_observation_config,
    register_ngsim_env,
)
from highway_env.imitation.expert_dataset import (
    SCHEMA_VERSION as EXPERT_DATASET_SCHEMA_VERSION,
)
from highway_env.imitation.lateral_command import (
    LATERAL_COMMAND_CONTRACT_ID,
    LATERAL_COMMAND_DIM,
    LATERAL_COMMAND_SCHEMA_VERSION,
    LATERAL_COMMAND_TEACHER_ID,
    CommandInstanceV2,
    InteractionOverlay,
    LateralCommand,
    LateralCommandSchedulerConfig,
    LateralCommandTeacherConfig,
    LateralFeasibility,
    OvertakePhase,
    ProspectiveLateralCommandScheduler,
    command_state_sha256,
    current_vehicle_lateral_feasibility,
    lateral_command_contract,
    lateral_command_one_hot,
    lateral_command_teacher_action,
    lateral_command_teacher_contract,
)
from highway_env.ngsim_utils.core.constants import (
    ACCELERATION_LIMIT_MPS2,
    ACCELERATION_RANGE,
    MAX_ACCEL,
    MAX_STEER,
    MIN_ACCEL,
    denormalize_acceleration,
    normalize_acceleration,
)
from highway_env.ngsim_utils.data.prebuilt import (
    PrebuiltCache,
    load_prebuilt_data,
    refine_valid_ids_by_episode,
)
from highway_env.ngsim_utils.road.gen_road import (
    create_japanese_road,
    create_ngsim_101_road,
)
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.route_intent import (
    TopologyAwareRouteIntentProvider,
    route_intent_contract,
)
from highway_env.vehicle.kinematics import Vehicle

PUBLIC_RESEARCH_API_VERSION = 4
HIGHWAY_ENV_PACKAGE_FILE = str(Path(__file__).with_name("__init__.py").resolve())
RESEARCH_API_FILE = str(Path(__file__).resolve())
ROUTE_INTENT_SOURCE_FILE = str(Path(inspect.getfile(TopologyAwareRouteIntentProvider)).resolve())


__all__ = [
    "ACCELERATION_LIMIT_MPS2",
    "ACCELERATION_RANGE",
    "AbstractEnv",
    "Action",
    "BEHAVIOR_INTENT_DIM",
    "MANEUVER_FAMILY_DIM",
    "BehaviorCommandSchedulerConfig",
    "BehaviorFeasibility",
    "BehaviorIntent",
    "BehaviorLabelerConfig",
    "BehaviorLabels",
    "ManeuverFamilyIntent",
    "ENV_ID",
    "EXPERT_DATASET_SCHEMA_VERSION",
    "HIGHWAY_ENV_PACKAGE_FILE",
    "LineType",
    "LATERAL_COMMAND_CONTRACT_ID",
    "LATERAL_COMMAND_DIM",
    "LATERAL_COMMAND_SCHEMA_VERSION",
    "LATERAL_COMMAND_TEACHER_ID",
    "CommandInstanceV2",
    "InteractionOverlay",
    "LateralCommand",
    "LateralCommandSchedulerConfig",
    "LateralCommandTeacherConfig",
    "LateralFeasibility",
    "MAX_ACCEL",
    "MAX_STEER",
    "MIN_ACCEL",
    "PUBLIC_RESEARCH_API_VERSION",
    "PrebuiltCache",
    "ProspectiveBehaviorCommandScheduler",
    "ProspectiveLateralCommandScheduler",
    "RESEARCH_API_FILE",
    "ROUTE_INTENT_SOURCE_FILE",
    "Road",
    "RoadNetwork",
    "StraightLane",
    "TopologyAwareRouteIntentProvider",
    "Vehicle",
    "build_env_config",
    "behavior_intent_contract",
    "behavior_intent_one_hot",
    "behavior_to_maneuver_family_ids",
    "command_state_sha256",
    "create_japanese_road",
    "create_ngsim_101_road",
    "current_vehicle_behavior_feasibility",
    "current_vehicle_lateral_feasibility",
    "default_observation_config",
    "denormalize_acceleration",
    "load_prebuilt_data",
    "label_behavior_rows",
    "maneuver_family_contract",
    "maneuver_family_one_hot",
    "lateral_command_contract",
    "lateral_command_one_hot",
    "lateral_command_teacher_action",
    "lateral_command_teacher_contract",
    "normalize_acceleration",
    "observation_factory",
    "refine_valid_ids_by_episode",
    "register_ngsim_env",
    "route_intent_contract",
    "OvertakePhase",
]
