"""Stable research API for policy construction, observation, and checkpoints.

The implementation remains in :mod:`scripts_gail.ps_gail`.  This module is the
supported boundary for the parent interpretability project so that policy
training internals can evolve without duplicating action, sensor, or
architecture contracts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from scripts_gail.ps_gail.checkpoints import (
    POLICY_ARCHITECTURE_FIELDS,
    SHARED_INTERPRETABLE_TRANSFORMER_CONTRACT_ID,
    policy_architecture_contract,
    sha256_file,
    shared_interpretable_transformer_architecture,
)
from scripts_gail.ps_gail.contracts import (
    LANE_CAMERA_CELLS,
    LANE_CAMERA_FEATURE_DIM,
    LIDAR_FEATURE_DIM,
    NORMALIZED_ACTION_COLUMNS,
    PHYSICAL_ACTION_COLUMNS,
    POLICY_EGO_COLUMNS,
    RAW_EGO_COLUMNS,
    ContinuousActionContract,
    infer_continuous_action_contract,
    runtime_continuous_action_contract,
)
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.observations import (
    EGO_STATE_DIM,
    POLICY_STATE_DIM,
    flatten_agent_observations,
    flatten_lidar_lane_state_observation,
    flatten_observation_value,
    policy_observations_from_flat,
)

PUBLIC_POLICY_API_VERSION = 1


@dataclass(frozen=True)
class PolicyBundle:
    """Loaded frozen policy plus its complete replay provenance."""

    policy: nn.Module
    config: dict[str, Any]
    checkpoint: dict[str, Any]
    checkpoint_path: str
    checkpoint_sha256: str
    policy_obs_dim: int
    critic_obs_dim: int
    device: torch.device


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return checkpoint configuration as a plain mapping."""
    raw = checkpoint.get("config", {}) or {}
    if hasattr(raw, "__dict__"):
        raw = vars(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint config type: {type(raw)!r}.")
    return dict(raw)


def _cfg(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def infer_policy_obs_dim(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> int:
    """Infer the policy observation width without constructing the model."""
    policy_model = str(_cfg(config, "policy_model", "")).lower()
    if policy_model == "recurrent_transformer":
        dense_weight = state_dict.get("dense_observation_proj.0.weight")
        if dense_weight is not None:
            return int(dense_weight.shape[1])
        position = state_dict.get("current_position_embedding")
        if position is None:
            raise KeyError(
                "current_position_embedding missing from recurrent transformer state dict."
            )
        max_current_tokens = int(position.shape[1])
        # semantic tokens = policy + 3 ego scalars + lidar cells + 21 lane cells
        lidar_cells = max_current_tokens - 1 - 3 - LANE_CAMERA_CELLS
        if lidar_cells > 0:
            return int(
                lidar_cells * LIDAR_FEATURE_DIM
                + LANE_CAMERA_CELLS * LANE_CAMERA_FEATURE_DIM
                + POLICY_STATE_DIM
            )
        return max_current_tokens - 1
    scalar_weight = state_dict.get("input_proj.weight")
    if scalar_weight is not None and "pos_embedding" in state_dict:
        return int(state_dict["pos_embedding"].shape[1] - 1)
    raise ValueError("Could not infer policy observation dimension from checkpoint.")


def infer_critic_obs_dim(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    *,
    policy_obs_dim: int,
) -> int:
    """Infer the critic input width for exact actor-critic reconstruction."""
    if not bool(_cfg(config, "centralized_critic", False)):
        return int(policy_obs_dim)
    pooling = str(_cfg(config, "central_critic_pooling", "flat")).lower()
    if pooling in {"attention", "attn"}:
        query_weight = state_dict.get("critic_encoder.query_proj.0.weight")
        if query_weight is None:
            raise KeyError(
                "critic_encoder.query_proj.0.weight missing from centralized attention critic."
            )
        max_vehicles = int(_cfg(config, "central_critic_max_vehicles", 64))
        vehicle_feature_dim = 5
        return int(max_vehicles * vehicle_feature_dim + query_weight.shape[1])
    first_weight = state_dict.get("critic_encoder.0.weight")
    if first_weight is None:
        raise KeyError(
            "critic_encoder.0.weight missing from centralized flat critic."
        )
    return int(first_weight.shape[1])


def load_policy_bundle(
    checkpoint_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
) -> PolicyBundle:
    """Load a policy checkpoint using the same architecture factory as training."""
    resolved_path = str(Path(checkpoint_path).resolve())
    resolved_device = torch.device(device)
    checkpoint = torch.load(
        resolved_path,
        map_location=resolved_device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}.")
    if "policy_state_dict" not in checkpoint:
        raise KeyError(f"{resolved_path} is missing policy_state_dict.")
    config = checkpoint_config(checkpoint)
    state_dict = checkpoint["policy_state_dict"]
    policy_obs_dim = infer_policy_obs_dim(config, state_dict)
    critic_obs_dim = infer_critic_obs_dim(
        config,
        state_dict,
        policy_obs_dim=policy_obs_dim,
    )

    policy = make_actor_critic(
        str(_cfg(config, "policy_model", "recurrent_transformer")),
        policy_obs_dim,
        int(_cfg(config, "hidden_size", 256)),
        action_mode=str(_cfg(config, "action_mode", "continuous")),
        continuous_action_dim=int(_cfg(config, "continuous_action_dim", 2)),
        transformer_layers=int(_cfg(config, "transformer_layers", 2)),
        transformer_heads=int(_cfg(config, "transformer_heads", 4)),
        transformer_dropout=float(_cfg(config, "transformer_dropout", 0.1)),
        transformer_norm_first=bool(
            _cfg(config, "transformer_norm_first", False)
        ),
        transformer_observation_normalization=bool(
            _cfg(config, "transformer_observation_normalization", False)
        ),
        transformer_observation_tokenization=str(
            _cfg(config, "transformer_observation_tokenization", "semantic")
        ),
        transformer_temporal_module=bool(
            _cfg(config, "transformer_temporal_module", False)
        ),
        transformer_temporal_kernel_size=int(
            _cfg(config, "transformer_temporal_kernel_size", 5)
        ),
        transformer_temporal_layers=int(
            _cfg(config, "transformer_temporal_layers", 1)
        ),
        transformer_memory_tokens=int(
            _cfg(config, "transformer_memory_tokens", 8)
        ),
        transformer_memory_context_length=int(
            _cfg(config, "transformer_memory_context_length", 32)
        ),
        transformer_use_causal_attention=bool(
            _cfg(config, "transformer_use_causal_attention", True)
        ),
        centralized_critic=bool(_cfg(config, "centralized_critic", False)),
        critic_obs_dim=critic_obs_dim,
        central_critic_pooling=str(
            _cfg(config, "central_critic_pooling", "flat")
        ),
        central_critic_max_vehicles=int(
            _cfg(config, "central_critic_max_vehicles", 64)
        ),
        central_critic_attention_heads=int(
            _cfg(config, "central_critic_attention_heads", 4)
        ),
    ).to(resolved_device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return PolicyBundle(
        policy=policy,
        config=config,
        checkpoint=checkpoint,
        checkpoint_path=resolved_path,
        checkpoint_sha256=sha256_file(resolved_path),
        policy_obs_dim=policy_obs_dim,
        critic_obs_dim=critic_obs_dim,
        device=resolved_device,
    )


__all__ = [
    "ContinuousActionContract",
    "EGO_STATE_DIM",
    "LANE_CAMERA_CELLS",
    "LANE_CAMERA_FEATURE_DIM",
    "LIDAR_FEATURE_DIM",
    "NORMALIZED_ACTION_COLUMNS",
    "PHYSICAL_ACTION_COLUMNS",
    "POLICY_ARCHITECTURE_FIELDS",
    "POLICY_EGO_COLUMNS",
    "POLICY_STATE_DIM",
    "PUBLIC_POLICY_API_VERSION",
    "PolicyBundle",
    "RAW_EGO_COLUMNS",
    "SHARED_INTERPRETABLE_TRANSFORMER_CONTRACT_ID",
    "checkpoint_config",
    "flatten_agent_observations",
    "flatten_lidar_lane_state_observation",
    "flatten_observation_value",
    "infer_continuous_action_contract",
    "infer_critic_obs_dim",
    "infer_policy_obs_dim",
    "load_policy_bundle",
    "make_actor_critic",
    "policy_architecture_contract",
    "policy_observations_from_flat",
    "runtime_continuous_action_contract",
    "sha256_file",
    "shared_interpretable_transformer_architecture",
]
