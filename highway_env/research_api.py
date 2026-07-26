"""Stable simulator and trajectory-data API for downstream research.

The HighwayEnv fork retains its upstream-compatible internal layout.  Parent
research code should import the small surface below instead of binding to
implementation modules under ``ngsim_utils`` or ``imitation``.
"""

from __future__ import annotations

from highway_env.imitation.expert_dataset import (
    ENV_ID,
    SCHEMA_VERSION as EXPERT_DATASET_SCHEMA_VERSION,
    build_env_config,
    default_observation_config,
    register_ngsim_env,
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


PUBLIC_RESEARCH_API_VERSION = 1


__all__ = [
    "ACCELERATION_LIMIT_MPS2",
    "ACCELERATION_RANGE",
    "ENV_ID",
    "EXPERT_DATASET_SCHEMA_VERSION",
    "MAX_ACCEL",
    "MAX_STEER",
    "MIN_ACCEL",
    "PUBLIC_RESEARCH_API_VERSION",
    "PrebuiltCache",
    "build_env_config",
    "default_observation_config",
    "denormalize_acceleration",
    "load_prebuilt_data",
    "normalize_acceleration",
    "refine_valid_ids_by_episode",
    "register_ngsim_env",
]
