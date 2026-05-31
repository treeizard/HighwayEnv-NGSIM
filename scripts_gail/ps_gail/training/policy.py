"""Policy action, critic-observation, and recurrent-memory helpers for PS-GAIL."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import OrderedDict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env
from highway_env.ngsim_utils.core.constants import MAX_ACCEL
from highway_env.ngsim_utils.data.prebuilt import load_prebuilt_data

from ..config import PSGAILConfig
from ..data import (
    SCENE_FEATURE_DIM_PER_VEHICLE,
    build_sequence_windows,
    discriminator_features,
    normalize_trajectory_frame,
    scene_snapshot_features,
    standardize_features,
    transform_sequence_features,
)
from ..models import NUM_DISCRETE_META_ACTIONS, make_actor_critic
from ..observations import flatten_agent_observations, policy_observations_from_flat

from .torch_utils import SquashedNormal
from .types import RolloutBatch

CENTRAL_CRITIC_CONTEXT_DIM = 4

def infer_policy_obs_dim(env: gym.Env) -> int:
    obs, _ = env.reset()
    return int(policy_observations_from_flat(flatten_agent_observations(obs)).shape[1])

def centralized_critic_enabled(cfg: PSGAILConfig) -> bool:
    return bool(getattr(cfg, "centralized_critic", False))

def central_critic_observation_dim(policy_obs_dim: int, cfg: PSGAILConfig) -> int:
    if not centralized_critic_enabled(cfg):
        return int(policy_obs_dim)
    base_dim = max(1, int(getattr(cfg, "central_critic_max_vehicles", 64))) * SCENE_FEATURE_DIM_PER_VEHICLE
    if bool(getattr(cfg, "central_critic_include_local_obs", False)):
        base_dim += int(policy_obs_dim)
    return int(base_dim + CENTRAL_CRITIC_CONTEXT_DIM)

def infer_critic_obs_dim(
    env: gym.Env,
    cfg: PSGAILConfig,
    *,
    policy_obs_dim: int | None = None,
) -> int:
    if policy_obs_dim is None:
        policy_obs_dim = infer_policy_obs_dim(env)
    return central_critic_observation_dim(int(policy_obs_dim), cfg)

def infer_continuous_action_dim(env: gym.Env) -> int:
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Tuple):
        if not action_space.spaces:
            raise ValueError("Cannot infer continuous action dim from an empty Tuple action space.")
        action_space = action_space.spaces[0]
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError(f"Continuous action mode expects a Box action space, got {action_space!r}.")
    return int(np.prod(action_space.shape))

def _is_continuous(cfg: PSGAILConfig) -> bool:
    return str(cfg.action_mode).lower() == "continuous"

def _masked_discrete_logits(logits: torch.Tensor, action_masks: torch.Tensor | None) -> torch.Tensor:
    if action_masks is None:
        return logits
    masks = action_masks.to(device=logits.device, dtype=torch.bool)
    if masks.shape != logits.shape:
        raise ValueError(f"Action mask shape {tuple(masks.shape)} does not match logits {tuple(logits.shape)}.")
    if torch.any(~masks.any(dim=-1)):
        raise ValueError("Each discrete action mask row must contain at least one valid action.")
    return logits.masked_fill(~masks, -1.0e9)

def policy_action_dim(policy: nn.Module) -> int:
    policy_head = getattr(policy, "policy_head", None)
    out_features = getattr(policy_head, "out_features", None)
    return int(out_features) if out_features is not None else int(NUM_DISCRETE_META_ACTIONS)

def discrete_action_masks_from_env(
    env: gym.Env,
    *,
    num_agents: int,
    num_actions: int,
    enabled: bool = True,
) -> np.ndarray:
    if not bool(enabled):
        return np.ones((int(num_agents), int(num_actions)), dtype=bool)
    num_agents = int(num_agents)
    num_actions = int(num_actions)
    masks = np.zeros((num_agents, num_actions), dtype=bool)
    action_type = getattr(env.unwrapped, "action_type", None)
    agent_action_types = list(getattr(action_type, "agents_action_types", ()) or ())
    for agent_idx in range(num_agents):
        source = agent_action_types[agent_idx] if agent_idx < len(agent_action_types) else action_type
        try:
            available = list(source.get_available_actions()) if source is not None else list(range(num_actions))
        except Exception:
            available = list(range(num_actions))
        for action in available:
            try:
                action_idx = int(action)
            except (TypeError, ValueError):
                continue
            if 0 <= action_idx < num_actions:
                masks[agent_idx, action_idx] = True
        if not masks[agent_idx].any():
            masks[agent_idx] = True
    return masks

def _finite_scalar(value: object, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    return scalar if np.isfinite(scalar) else float(default)

def _vehicle_scalar(vehicle: object | None, names: tuple[str, ...], default: float = 0.0) -> float:
    if vehicle is None:
        return float(default)
    for name in names:
        if hasattr(vehicle, name):
            return _finite_scalar(getattr(vehicle, name), default)
    return float(default)

def _vehicle_origin(vehicle: object | None) -> np.ndarray | None:
    if vehicle is None or getattr(vehicle, "position", None) is None:
        return None
    try:
        origin = np.asarray(getattr(vehicle, "position"), dtype=np.float32).reshape(2)
    except (TypeError, ValueError):
        return None
    return origin if np.all(np.isfinite(origin)) else None

def central_critic_observations(
    env: gym.Env,
    cfg: PSGAILConfig,
    policy_observations: np.ndarray,
) -> np.ndarray:
    policy_obs = np.asarray(policy_observations, dtype=np.float32)
    if policy_obs.ndim != 2:
        raise ValueError(f"Expected rank-2 policy observations, got {policy_obs.shape}.")
    if not centralized_critic_enabled(cfg):
        return policy_obs.astype(np.float32, copy=True)

    controlled_vehicles = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
    road = getattr(env.unwrapped, "road", None)
    road_vehicles = (
        list(getattr(road, "vehicles", ()) or ())
        if road is not None
        else controlled_vehicles
    )
    max_vehicles = max(1, int(getattr(cfg, "central_critic_max_vehicles", 64)))
    include_local_obs = bool(getattr(cfg, "central_critic_include_local_obs", False))
    rows: list[np.ndarray] = []
    for agent_idx in range(policy_obs.shape[0]):
        vehicle = controlled_vehicles[agent_idx] if agent_idx < len(controlled_vehicles) else None
        scene = scene_snapshot_features(
            road_vehicles,
            max_vehicles=max_vehicles,
            origin=_vehicle_origin(vehicle),
        )
        context = np.asarray(
            [
                _vehicle_scalar(vehicle, ("speed",)),
                _vehicle_scalar(vehicle, ("heading",)),
                _vehicle_scalar(vehicle, ("WIDTH", "width")),
                _vehicle_scalar(vehicle, ("LENGTH", "length")),
            ],
            dtype=np.float32,
        )
        parts = [scene, context]
        if include_local_obs:
            parts.append(policy_obs[agent_idx])
        rows.append(np.concatenate(parts, axis=0).astype(np.float32, copy=False))
    if not rows:
        return np.zeros(
            (0, central_critic_observation_dim(policy_obs.shape[1], cfg)),
            dtype=np.float32,
        )
    return np.stack(rows, axis=0).astype(np.float32, copy=False)

def policy_distribution_and_values(
    policy: nn.Module,
    obs_tensor: torch.Tensor,
    cfg: PSGAILConfig,
    action_masks: torch.Tensor | None = None,
    critic_obs_tensor: torch.Tensor | None = None,
) -> tuple[Categorical | SquashedNormal, torch.Tensor]:
    dist, values, _memory = policy_distribution_values_memory(
        policy,
        obs_tensor,
        cfg,
        action_masks,
        critic_obs_tensor=critic_obs_tensor,
        return_memory=False,
    )
    return dist, values

def _sample_policy_actions(
    dist: Categorical | SquashedNormal,
    cfg: PSGAILConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = dist.sample()
    return actions, dist.log_prob(actions)

def _actions_to_env_tuple(actions: torch.Tensor, cfg: PSGAILConfig) -> tuple[object, ...]:
    actions_np = actions.detach().cpu().numpy()
    if _is_continuous(cfg):
        actions_np = np.asarray(actions_np, dtype=np.float32).reshape(
            -1,
            int(cfg.continuous_action_dim),
        )
        actions_np = np.clip(actions_np, -1.0, 1.0)
        return tuple(action.copy() for action in actions_np)
    return tuple(int(action) for action in actions_np.tolist())

def recurrent_policy_enabled(policy: nn.Module) -> bool:
    return bool(getattr(policy, "supports_recurrent_memory", False))

def _memory_storage_dtype(cfg: PSGAILConfig) -> np.dtype:
    name = str(getattr(cfg, "transformer_memory_storage_dtype", "float16")).lower()
    if name in {"float32", "fp32"}:
        return np.dtype(np.float32)
    if name in {"float16", "fp16", "half"}:
        return np.dtype(np.float16)
    raise ValueError(
        "transformer_memory_storage_dtype must be 'float16' or 'float32', "
        f"got {getattr(cfg, 'transformer_memory_storage_dtype', None)!r}."
    )

def _empty_step_memory(policy: nn.Module, cfg: PSGAILConfig) -> np.ndarray:
    tokens = int(getattr(policy, "memory_tokens", max(1, int(getattr(cfg, "transformer_memory_tokens", 8)))))
    hidden = int(getattr(policy, "hidden_size", int(getattr(cfg, "hidden_size", 256))))
    return np.zeros((tokens, hidden), dtype=_memory_storage_dtype(cfg))

def _memory_cache_from_state(
    policy: nn.Module,
    cfg: PSGAILConfig,
    keys: list[tuple[int, int]],
    memory_state: dict[tuple[int, int], list[np.ndarray]],
    device: torch.device,
) -> torch.Tensor | None:
    if not recurrent_policy_enabled(policy):
        return None
    context = int(getattr(policy, "memory_context_length", int(getattr(cfg, "transformer_memory_context_length", 32))))
    empty = _empty_step_memory(policy, cfg)
    rows: list[np.ndarray] = []
    for key in keys:
        history = list(memory_state.get(key, []))[-context:]
        if len(history) < context:
            history = [empty] * (context - len(history)) + history
        rows.append(np.stack(history, axis=0).astype(np.float32, copy=False))
    if not rows:
        return None
    return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=device)

def _step_memory_to_numpy(step_memory: torch.Tensor, cfg: PSGAILConfig) -> np.ndarray:
    return step_memory.detach().to(device="cpu", dtype=torch.float32).numpy().astype(
        _memory_storage_dtype(cfg),
        copy=False,
    )

def _update_memory_state(
    policy: nn.Module,
    cfg: PSGAILConfig,
    keys: list[tuple[int, int]],
    memory_state: dict[tuple[int, int], list[np.ndarray]],
    step_memory: torch.Tensor | None,
) -> np.ndarray:
    if not recurrent_policy_enabled(policy) or step_memory is None or not keys:
        return np.zeros((len(keys), 0, 0), dtype=_memory_storage_dtype(cfg))
    step_np = _step_memory_to_numpy(step_memory, cfg)
    context = int(getattr(policy, "memory_context_length", int(getattr(cfg, "transformer_memory_context_length", 32))))
    for i, key in enumerate(keys):
        history = memory_state.setdefault(key, [])
        history.append(step_np[i].copy())
        if len(history) > context:
            del history[: len(history) - context]
    return step_np

def _recurrent_memory_cache_from_prior_steps(
    policy: nn.Module,
    cfg: PSGAILConfig,
    rollout: RolloutBatch,
    trajectory_indices: np.ndarray,
    chunk_start: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not recurrent_policy_enabled(policy):
        return None
    context = int(getattr(policy, "memory_context_length", int(getattr(cfg, "transformer_memory_context_length", 32))))
    prior = np.asarray(trajectory_indices[max(0, int(chunk_start) - context) : int(chunk_start)], dtype=np.int64)
    empty = _empty_step_memory(policy, cfg)
    if prior.size:
        history = [
            np.asarray(rollout.policy_step_memories[int(idx)], dtype=np.float32)
            for idx in prior
        ]
    else:
        history = []
    if len(history) < context:
        history = [empty.astype(np.float32, copy=False)] * (context - len(history)) + history
    memory = np.stack(history[-context:], axis=0).astype(np.float32, copy=False)
    return torch.as_tensor(memory[None, ...], dtype=torch.float32, device=device)

def _shift_recurrent_memory(memory: torch.Tensor, step_memory: torch.Tensor) -> torch.Tensor:
    return torch.cat([memory[:, 1:], step_memory.unsqueeze(1)], dim=1)

def recurrent_memory_stats(rollout: RolloutBatch) -> dict[str, float]:
    memories = getattr(rollout, "policy_step_memories", None)
    if memories is None or not isinstance(memories, np.ndarray) or memories.size == 0:
        return {"perf/recurrent_memory_mb": 0.0, "perf/recurrent_memory_tokens": 0.0}
    return {
        "perf/recurrent_memory_mb": float(memories.nbytes / (1024.0 * 1024.0)),
        "perf/recurrent_memory_tokens": float(memories.shape[0] * memories.shape[1]),
    }

def policy_distribution_values_memory(
    policy: nn.Module,
    obs_tensor: torch.Tensor,
    cfg: PSGAILConfig,
    action_masks: torch.Tensor | None = None,
    critic_obs_tensor: torch.Tensor | None = None,
    memory: torch.Tensor | None = None,
    *,
    return_memory: bool = False,
) -> tuple[Categorical | SquashedNormal, torch.Tensor, torch.Tensor | None]:
    # Core of PPO, Need to be aware of recurrent policy memory handling and continuous vs discrete action modes.
    if recurrent_policy_enabled(policy):
        policy_out, values, new_memory = policy(
            obs_tensor,
            critic_obs_tensor,
            memory=memory,
            return_memory=True,
        )
    else:
        policy_out, values = policy(obs_tensor, critic_obs_tensor)
        new_memory = None
    if _is_continuous(cfg):
        if policy.log_std is None:
            raise RuntimeError("Continuous action mode requires policy.log_std.")
        std = torch.exp(policy.log_std).expand_as(policy_out)
        return SquashedNormal(policy_out, std), values, new_memory if return_memory else None
    dist = Categorical(logits=_masked_discrete_logits(policy_out, action_masks))
    return dist, values, new_memory if return_memory else None

def _make_policy_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    cfg: PSGAILConfig,
    policy_obs_dim: int,
    critic_obs_dim: int,
    device: torch.device,
) -> nn.Module:
    policy = make_actor_critic(
        cfg.policy_model,
        int(policy_obs_dim),
        int(cfg.hidden_size),
        action_mode=str(cfg.action_mode),
        continuous_action_dim=int(cfg.continuous_action_dim),
        transformer_layers=int(cfg.transformer_layers),
        transformer_heads=int(cfg.transformer_heads),
        transformer_dropout=float(cfg.transformer_dropout),
        transformer_temporal_module=bool(getattr(cfg, "transformer_temporal_module", False)),
        transformer_temporal_kernel_size=int(getattr(cfg, "transformer_temporal_kernel_size", 5)),
        transformer_temporal_layers=int(getattr(cfg, "transformer_temporal_layers", 1)),
        transformer_memory_tokens=int(getattr(cfg, "transformer_memory_tokens", 8)),
        transformer_memory_context_length=int(getattr(cfg, "transformer_memory_context_length", 32)),
        transformer_use_causal_attention=bool(getattr(cfg, "transformer_use_causal_attention", True)),
        centralized_critic=centralized_critic_enabled(cfg),
        critic_obs_dim=int(critic_obs_dim),
        central_critic_pooling=str(getattr(cfg, "central_critic_pooling", "flat")),
        central_critic_max_vehicles=int(getattr(cfg, "central_critic_max_vehicles", 64)),
        central_critic_attention_heads=int(getattr(cfg, "central_critic_attention_heads", 4)),
    )
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy

__all__ = [
    'infer_policy_obs_dim',
    'centralized_critic_enabled',
    'central_critic_observation_dim',
    'infer_critic_obs_dim',
    'infer_continuous_action_dim',
    '_is_continuous',
    '_masked_discrete_logits',
    'policy_action_dim',
    'discrete_action_masks_from_env',
    '_finite_scalar',
    '_vehicle_scalar',
    '_vehicle_origin',
    'central_critic_observations',
    'policy_distribution_and_values',
    '_sample_policy_actions',
    '_actions_to_env_tuple',
    'recurrent_policy_enabled',
    '_memory_storage_dtype',
    '_empty_step_memory',
    '_memory_cache_from_state',
    '_step_memory_to_numpy',
    '_update_memory_state',
    '_recurrent_memory_cache_from_prior_steps',
    '_shift_recurrent_memory',
    'recurrent_memory_stats',
    'policy_distribution_values_memory',
    '_make_policy_from_state_dict'
]
