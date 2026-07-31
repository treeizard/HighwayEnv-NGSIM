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
from highway_env.imitation.expert_dataset import (
    ENV_ID,
    build_env_config,
    default_observation_config,
    register_ngsim_env,
)
from highway_env.imitation.expert_dataset import (
    SCHEMA_VERSION as EXPERT_DATASET_SCHEMA_VERSION,
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

PUBLIC_RESEARCH_API_VERSION = 2
HIGHWAY_ENV_PACKAGE_FILE = str(Path(__file__).with_name("__init__.py").resolve())
RESEARCH_API_FILE = str(Path(__file__).resolve())
ROUTE_INTENT_SOURCE_FILE = str(
    Path(inspect.getfile(TopologyAwareRouteIntentProvider)).resolve()
)


__all__ = [
    "ACCELERATION_LIMIT_MPS2",
    "ACCELERATION_RANGE",
    "AbstractEnv",
    "Action",
    "ENV_ID",
    "EXPERT_DATASET_SCHEMA_VERSION",
    "HIGHWAY_ENV_PACKAGE_FILE",
    "LineType",
    "MAX_ACCEL",
    "MAX_STEER",
    "MIN_ACCEL",
    "PUBLIC_RESEARCH_API_VERSION",
    "PrebuiltCache",
    "RESEARCH_API_FILE",
    "ROUTE_INTENT_SOURCE_FILE",
    "Road",
    "RoadNetwork",
    "StraightLane",
    "TopologyAwareRouteIntentProvider",
    "Vehicle",
    "build_env_config",
    "create_japanese_road",
    "create_ngsim_101_road",
    "default_observation_config",
    "denormalize_acceleration",
    "load_prebuilt_data",
    "normalize_acceleration",
    "observation_factory",
    "refine_valid_ids_by_episode",
    "register_ngsim_env",
    "route_intent_contract",
]
