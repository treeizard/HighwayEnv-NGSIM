"""Matched-trajectory evaluation utilities for PS-GAIL policies."""

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

from .policy import (
    _actions_to_env_tuple,
    _is_continuous,
    _make_policy_from_state_dict,
    _masked_discrete_logits,
    central_critic_observation_dim,
    central_critic_observations,
    centralized_critic_enabled,
    discrete_action_masks_from_env,
    policy_action_dim,
    recurrent_policy_enabled,
)

_EVAL_POLICY_CACHE = {}
_EVAL_ENV_CACHE = {}
_EVAL_ENV_CACHE_HITS = 0
_EVAL_ENV_CACHE_MISSES = 0

@contextmanager
def evaluation_thread_context(cfg: PSGAILConfig):
    """Apply the configured native thread budget while evaluation is running."""
    requested_workers = max(1, int(getattr(cfg, "evaluation_num_workers", 1)))
    if requested_workers != 1:
        print(
            "[evaluation] evaluation_num_workers="
            f"{requested_workers} requested; serial evaluation path is active for this call.",
            flush=True,
        )
    threads = max(1, int(getattr(cfg, "evaluation_worker_threads", 2)))
    env_names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    old_env = {name: os.environ.get(name) for name in env_names}
    old_torch_threads = torch.get_num_threads()
    try:
        for name in env_names:
            os.environ[name] = str(threads)
        torch.set_num_threads(threads)
        yield
    finally:
        torch.set_num_threads(old_torch_threads)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

def _parse_evaluation_horizons(cfg: PSGAILConfig) -> list[int]:
    values: list[int] = []
    for raw in str(getattr(cfg, "evaluation_horizons_seconds", "1,5,10,20")).replace(";", ",").split(","):
        text = raw.strip()
        if not text:
            continue
        horizon = int(round(float(text)))
        if horizon > 0 and horizon not in values:
            values.append(horizon)
    return values or [1, 5, 10, 20]

def _evaluation_scenarios(
    cfg: PSGAILConfig,
    *,
    split: str,
    episodes: int,
) -> list[tuple[str, int]]:
    if int(episodes) <= 0:
        return []
    _prebuilt_dir, valid_ids_by_episode, _traj_all_by_episode, episode_names = load_prebuilt_data(
        cfg.episode_root,
        cfg.scene,
        str(split),
        min_occupancy=0.8,
        cache={},
    )
    scenarios: list[tuple[str, int]] = []
    for episode_name in sorted(str(name) for name in episode_names):
        for vehicle_id in sorted(int(value) for value in valid_ids_by_episode.get(episode_name, [])):
            scenarios.append((episode_name, vehicle_id))
    if not scenarios:
        raise RuntimeError(f"No evaluation scenarios found for split={split!r}.")
    rng = np.random.default_rng(int(cfg.seed) + (17_003 if str(split) == "val" else 31_337))
    order = rng.permutation(len(scenarios))
    return [scenarios[int(idx)] for idx in order[: min(int(episodes), len(scenarios))]]

def _evaluation_episode_names(
    cfg: PSGAILConfig,
    *,
    split: str,
    episodes: int,
) -> list[str]:
    if int(episodes) <= 0:
        return []
    _prebuilt_dir, _valid_ids_by_episode, _traj_all_by_episode, episode_names = load_prebuilt_data(
        cfg.episode_root,
        cfg.scene,
        str(split),
        min_occupancy=0.8,
        cache={},
    )
    names = sorted(str(name) for name in episode_names)
    if not names:
        raise RuntimeError(f"No evaluation episodes found for split={split!r}.")
    rng = np.random.default_rng(int(cfg.seed) + (19_019 if str(split) == "val" else 37_037))
    order = rng.permutation(len(names))
    return [names[int(idx)] for idx in order[: min(int(episodes), len(names))]]

def _make_matched_eval_env(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
    vehicle_id: int,
) -> gym.Env:
    from ..envs import observation_config

    register_ngsim_env()
    env_cfg = build_env_config(
        scene=cfg.scene,
        action_mode=str(cfg.action_mode),
        episode_root=cfg.episode_root,
        prebuilt_split=str(split),
        percentage_controlled_vehicles=1.0,
        control_all_vehicles=False,
        max_surrounding=cfg.max_surrounding,
        observation_config=observation_config(cfg),
        simulation_frequency=cfg.simulation_frequency,
        policy_frequency=cfg.policy_frequency,
        max_episode_steps=cfg.max_episode_steps,
        seed=None,
        simulation_period={"episode_name": str(episode_name)},
        ego_vehicle_id=[int(vehicle_id)],
        scene_dataset_collection_mode=False,
        allow_idm=cfg.allow_idm,
    )
    # The matched-trajectory evaluator needs the expert reference arrays that
    # NGSimEnv prepares during expert-test reset, but policy actions must still
    # control the vehicle during evaluation. The eval loop flips this back off
    # immediately after reset and before the first env.step(...).
    env_cfg["expert_test_mode"] = True
    env_cfg["truncate_to_trajectory_length"] = False
    env_cfg["complete_controlled_vehicles_at_road_end"] = False
    env_cfg["disable_controlled_vehicle_collisions"] = not bool(cfg.enable_collision)
    env_cfg["terminate_when_all_controlled_crashed"] = bool(cfg.terminate_when_all_controlled_crashed)
    env_cfg["allow_idm"] = bool(cfg.allow_idm)
    env_cfg["crash_controlled_vehicles_offroad"] = True
    return gym.make(ENV_ID, config=env_cfg)

def _make_matched_eval_all_vehicle_env(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
) -> gym.Env:
    from ..envs import observation_config

    register_ngsim_env()
    env_cfg = build_env_config(
        scene=cfg.scene,
        action_mode=str(cfg.action_mode),
        episode_root=cfg.episode_root,
        prebuilt_split=str(split),
        percentage_controlled_vehicles=1.0,
        control_all_vehicles=True,
        max_surrounding=cfg.max_surrounding,
        observation_config=observation_config(cfg),
        simulation_frequency=cfg.simulation_frequency,
        policy_frequency=cfg.policy_frequency,
        max_episode_steps=cfg.max_episode_steps,
        seed=None,
        simulation_period={"episode_name": str(episode_name)},
        scene_dataset_collection_mode=False,
        allow_idm=cfg.allow_idm,
    )
    env_cfg["expert_test_mode"] = True
    env_cfg["truncate_to_trajectory_length"] = False
    env_cfg["complete_controlled_vehicles_at_road_end"] = False
    env_cfg["disable_controlled_vehicle_collisions"] = not bool(cfg.enable_collision)
    env_cfg["terminate_when_all_controlled_crashed"] = bool(cfg.terminate_when_all_controlled_crashed)
    env_cfg["allow_idm"] = bool(cfg.allow_idm)
    env_cfg["crash_controlled_vehicles_offroad"] = True
    return gym.make(ENV_ID, config=env_cfg)

def _deterministic_policy_action_tuple(
    policy: nn.Module,
    env: gym.Env,
    obs: object,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    memory: torch.Tensor | None = None,
    return_memory: bool = False,
) -> tuple[object, ...] | tuple[tuple[object, ...], torch.Tensor | None]:
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    critic_obs_agents = central_critic_observations(env, cfg, obs_agents)
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
        critic_obs_tensor = torch.as_tensor(critic_obs_agents, dtype=torch.float32, device=device)
        if recurrent_policy_enabled(policy):
            policy_out, _values, new_memory = policy(
                obs_tensor,
                critic_obs_tensor,
                memory=memory,
                return_memory=True,
            )
        else:
            policy_out, _values = policy(obs_tensor, critic_obs_tensor)
            new_memory = None
        if _is_continuous(cfg):
            actions = _actions_to_env_tuple(torch.clamp(policy_out, -1.0, 1.0), cfg)
            return (actions, new_memory) if return_memory else actions
        masks = discrete_action_masks_from_env(
            env,
            num_agents=len(obs_agents),
            num_actions=policy_action_dim(policy),
            enabled=bool(getattr(cfg, "enable_action_masking", True)),
        )
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
        logits = _masked_discrete_logits(policy_out, mask_tensor)
        actions = _actions_to_env_tuple(torch.argmax(logits, dim=-1), cfg)
        return (actions, new_memory) if return_memory else actions

def _lane_offset_for_position(env: gym.Env, position: np.ndarray, vehicle: object | None = None) -> float:
    position = np.asarray(position, dtype=np.float32).reshape(-1)[:2]
    lane = getattr(vehicle, "lane", None) if vehicle is not None else None
    if lane is None:
        road = getattr(env.unwrapped, "road", None)
        network = getattr(road, "network", None)
        if network is not None:
            try:
                lane = network.get_lane(network.get_closest_lane_index(position))
            except Exception:
                lane = None
    if lane is None:
        return float("nan")
    try:
        _longitudinal, lateral = lane.local_coordinates(position)
        return float(lateral)
    except Exception:
        return float("nan")

def _first_controlled_vehicle(env: gym.Env) -> object | None:
    controlled = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
    return controlled[0] if controlled else None

def _physical_accel_from_action(action_tuple: tuple[object, ...], cfg: PSGAILConfig) -> float:
    if not _is_continuous(cfg) or not action_tuple:
        return float("nan")
    action = np.asarray(action_tuple[0], dtype=np.float32).reshape(-1)
    if action.size < 1:
        return float("nan")
    return float(np.clip(float(action[0]), -1.0, 1.0) * float(MAX_ACCEL))

def _physical_accels_from_actions(action_tuple: tuple[object, ...], cfg: PSGAILConfig) -> np.ndarray:
    if not _is_continuous(cfg) or not action_tuple:
        return np.zeros((0,), dtype=np.float32)
    accels = []
    for action in action_tuple:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < 1:
            continue
        accels.append(float(np.clip(float(action_arr[0]), -1.0, 1.0) * float(MAX_ACCEL)))
    return np.asarray(accels, dtype=np.float32)

def _matched_eval_metrics(
    *,
    prefix: str,
    attempted_episodes: int,
    evaluated_episodes: int,
    skipped_missing_expert: int,
    skipped_bad_reference: int,
    skipped_empty_rollout: int,
    total_steps: int,
    collision_steps: int,
    offroad_steps: int,
    hard_brake_steps: int,
    episode_lengths: list[int],
    squared: dict[int, dict[str, list[float]]],
    final_squared: dict[str, list[float]],
    horizons: list[int],
    terminated_episodes: int = 0,
    truncated_episodes: int = 0,
    crashed_vehicle_episodes: int | None = None,
    offroad_vehicle_episodes: int | None = None,
    vehicles: int | None = None,
    vehicle_episodes: int | None = None,
    controlled_vehicle_counts: list[int] | None = None,
    vehicle_ids: set[int] | None = None,
    include_raw: bool = False,
) -> dict[str, float]:
    collision_duration_rate = float(collision_steps / total_steps) if total_steps else float("nan")
    offroad_duration_rate = float(offroad_steps / total_steps) if total_steps else float("nan")
    vehicle_denominator = (
        int(vehicle_episodes)
        if vehicle_episodes is not None
        else int(evaluated_episodes)
    )
    vehicle_crash_rate = (
        float(crashed_vehicle_episodes / vehicle_denominator)
        if crashed_vehicle_episodes is not None and vehicle_denominator > 0
        else float("nan")
    )
    vehicle_offroad_rate = (
        float(offroad_vehicle_episodes / vehicle_denominator)
        if offroad_vehicle_episodes is not None and vehicle_denominator > 0
        else float("nan")
    )
    metrics: dict[str, float] = {
        f"{prefix}/attempted_episodes": float(attempted_episodes),
        f"{prefix}/episodes": float(evaluated_episodes),
        f"{prefix}/skipped_missing_expert": float(skipped_missing_expert),
        f"{prefix}/skipped_bad_reference": float(skipped_bad_reference),
        f"{prefix}/skipped_empty_rollout": float(skipped_empty_rollout),
        f"{prefix}/evaluated_steps": float(total_steps),
        f"{prefix}/collision_duration_rate": collision_duration_rate,
        f"{prefix}/collision_rate": vehicle_crash_rate if np.isfinite(vehicle_crash_rate) else collision_duration_rate,
        f"{prefix}/vehicle_crash_rate": vehicle_crash_rate,
        f"{prefix}/offroad_duration_rate": offroad_duration_rate,
        f"{prefix}/vehicle_offroad_rate": vehicle_offroad_rate,
        f"{prefix}/hard_brake_rate": float(hard_brake_steps / total_steps) if total_steps else float("nan"),
        f"{prefix}/mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        f"{prefix}/terminated_episodes": float(terminated_episodes),
        f"{prefix}/truncated_episodes": float(truncated_episodes),
    }
    if vehicles is not None:
        metrics[f"{prefix}/vehicles"] = float(vehicles)
    if vehicle_episodes is not None:
        metrics[f"{prefix}/vehicle_episodes"] = float(vehicle_episodes)
    if controlled_vehicle_counts is not None:
        metrics[f"{prefix}/mean_controlled_vehicles_per_episode"] = (
            float(np.mean(controlled_vehicle_counts)) if controlled_vehicle_counts else 0.0
        )
    for horizon in horizons:
        for name, values in squared[horizon].items():
            metric_name = f"{prefix}/rmse_{name}_{horizon}s"
            metrics[metric_name] = float(np.sqrt(np.mean(values))) if values else float("nan")
    for name, values in final_squared.items():
        metric_name = f"{prefix}/rmse_{name}_final"
        metrics[metric_name] = float(np.sqrt(np.mean(values))) if values else float("nan")
    if include_raw:
        metrics[f"__raw/{prefix}/collision_steps"] = float(collision_steps)
        metrics[f"__raw/{prefix}/offroad_steps"] = float(offroad_steps)
        metrics[f"__raw/{prefix}/hard_brake_steps"] = float(hard_brake_steps)
        metrics[f"__raw/{prefix}/crashed_vehicle_episodes"] = float(crashed_vehicle_episodes or 0)
        metrics[f"__raw/{prefix}/offroad_vehicle_episodes"] = float(offroad_vehicle_episodes or 0)
        metrics[f"__raw/{prefix}/episode_lengths"] = tuple(int(value) for value in episode_lengths)
        metrics[f"__raw/{prefix}/controlled_vehicle_counts"] = tuple(
            int(value) for value in (controlled_vehicle_counts or [])
        )
        metrics[f"__raw/{prefix}/vehicle_ids"] = tuple(
            sorted(int(value) for value in (vehicle_ids or set()))
        )
        for horizon in horizons:
            for name, values in squared[horizon].items():
                arr = np.asarray(values, dtype=np.float64)
                metrics[f"__raw/{prefix}/sse_{name}_{horizon}s"] = float(arr.sum()) if arr.size else 0.0
                metrics[f"__raw/{prefix}/count_{name}_{horizon}s"] = float(arr.size)
        for name, values in final_squared.items():
            arr = np.asarray(values, dtype=np.float64)
            metrics[f"__raw/{prefix}/sse_{name}_final"] = float(arr.sum()) if arr.size else 0.0
            metrics[f"__raw/{prefix}/count_{name}_final"] = float(arr.size)
    return metrics

def _strip_internal_matched_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if not key.startswith("__raw/") and isinstance(value, (int, float, np.floating))
    }

def _combine_matched_eval_metric_dicts(
    parts: list[dict[str, object]],
    *,
    prefix: str,
    horizons: list[int],
) -> dict[str, float]:
    if not parts:
        return {}
    sum_keys = (
        "attempted_episodes",
        "episodes",
        "skipped_missing_expert",
        "skipped_bad_reference",
        "skipped_empty_rollout",
        "evaluated_steps",
        "terminated_episodes",
        "truncated_episodes",
        "vehicle_episodes",
    )
    metrics: dict[str, float] = {}
    for name in sum_keys:
        metrics[f"{prefix}/{name}"] = float(sum(float(part.get(f"{prefix}/{name}", 0.0)) for part in parts))
    collision_steps = float(sum(float(part.get(f"__raw/{prefix}/collision_steps", 0.0)) for part in parts))
    offroad_steps = float(sum(float(part.get(f"__raw/{prefix}/offroad_steps", 0.0)) for part in parts))
    hard_brake_steps = float(sum(float(part.get(f"__raw/{prefix}/hard_brake_steps", 0.0)) for part in parts))
    crashed_vehicle_episodes = float(
        sum(float(part.get(f"__raw/{prefix}/crashed_vehicle_episodes", 0.0)) for part in parts)
    )
    offroad_vehicle_episodes = float(
        sum(float(part.get(f"__raw/{prefix}/offroad_vehicle_episodes", 0.0)) for part in parts)
    )
    total_steps = float(metrics.get(f"{prefix}/evaluated_steps", 0.0))
    vehicle_episodes = float(metrics.get(f"{prefix}/vehicle_episodes", metrics.get(f"{prefix}/episodes", 0.0)))
    metrics[f"{prefix}/collision_duration_rate"] = collision_steps / total_steps if total_steps else float("nan")
    metrics[f"{prefix}/offroad_duration_rate"] = offroad_steps / total_steps if total_steps else float("nan")
    metrics[f"{prefix}/hard_brake_rate"] = hard_brake_steps / total_steps if total_steps else float("nan")
    metrics[f"{prefix}/vehicle_crash_rate"] = (
        crashed_vehicle_episodes / vehicle_episodes if vehicle_episodes else float("nan")
    )
    metrics[f"{prefix}/vehicle_offroad_rate"] = (
        offroad_vehicle_episodes / vehicle_episodes if vehicle_episodes else float("nan")
    )
    metrics[f"{prefix}/collision_rate"] = (
        metrics[f"{prefix}/vehicle_crash_rate"]
        if np.isfinite(metrics[f"{prefix}/vehicle_crash_rate"])
        else metrics[f"{prefix}/collision_duration_rate"]
    )
    for raw_name, metric_name in (
        ("policy_load_seconds", "eval_policy_load_seconds"),
        ("policy_forward_seconds", "eval_policy_forward_seconds"),
        ("env_step_seconds", "eval_env_step_seconds"),
        ("env_reset_seconds", "eval_env_reset_seconds"),
        ("env_cache_hits", "eval_env_cache_hits"),
        ("env_cache_misses", "eval_env_cache_misses"),
    ):
        metrics[f"{prefix}/{metric_name}"] = float(
            sum(float(part.get(f"__raw/{prefix}/{raw_name}", 0.0)) for part in parts)
        )
    episode_lengths: list[int] = []
    controlled_counts: list[int] = []
    vehicle_ids: set[int] = set()
    for part in parts:
        episode_lengths.extend(int(value) for value in part.get(f"__raw/{prefix}/episode_lengths", ()))
        controlled_counts.extend(int(value) for value in part.get(f"__raw/{prefix}/controlled_vehicle_counts", ()))
        vehicle_ids.update(int(value) for value in part.get(f"__raw/{prefix}/vehicle_ids", ()))
    metrics[f"{prefix}/mean_episode_length"] = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    if controlled_counts:
        metrics[f"{prefix}/mean_controlled_vehicles_per_episode"] = float(np.mean(controlled_counts))
    metrics[f"{prefix}/vehicles"] = (
        float(len(vehicle_ids))
        if controlled_counts and vehicle_ids
        else metrics.get(f"{prefix}/episodes", 0.0)
    )
    names = ("x", "y", "position", "speed", "lane_offset")
    for horizon in horizons:
        for name in names:
            sse = float(sum(float(part.get(f"__raw/{prefix}/sse_{name}_{horizon}s", 0.0)) for part in parts))
            count = float(sum(float(part.get(f"__raw/{prefix}/count_{name}_{horizon}s", 0.0)) for part in parts))
            metrics[f"{prefix}/rmse_{name}_{horizon}s"] = float(np.sqrt(sse / count)) if count else float("nan")
    for name in names:
        sse = float(sum(float(part.get(f"__raw/{prefix}/sse_{name}_final", 0.0)) for part in parts))
        count = float(sum(float(part.get(f"__raw/{prefix}/count_{name}_final", 0.0)) for part in parts))
        metrics[f"{prefix}/rmse_{name}_final"] = float(np.sqrt(sse / count)) if count else float("nan")
    return metrics

def _chunk_evenly(items: list[object], chunks: int) -> list[list[object]]:
    chunks = max(1, int(chunks))
    if not items:
        return []
    active = min(chunks, len(items))
    return [
        items[start::active]
        for start in range(active)
        if items[start::active]
    ]

def _policy_input_dim(policy: nn.Module) -> int:
    obs_dim = getattr(policy, "obs_dim", None)
    if obs_dim is not None:
        return int(obs_dim)
    encoder = getattr(policy, "encoder", None)
    first = encoder[0] if isinstance(encoder, nn.Sequential) and len(encoder) else None
    in_features = getattr(first, "in_features", None)
    if in_features is None:
        raise RuntimeError("Cannot infer policy observation dimension for parallel evaluation.")
    return int(in_features)

def _cpu_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}

def _configure_evaluation_worker_threads(cfg: PSGAILConfig) -> None:
    threads = max(1, int(getattr(cfg, "evaluation_worker_threads", 2)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)

def _eval_policy_cache_key(
    cfg: PSGAILConfig,
    policy_obs_dim: int,
    critic_obs_dim: int,
) -> tuple[object, ...]:
    return (
        str(cfg.policy_model),
        int(policy_obs_dim),
        int(critic_obs_dim),
        int(cfg.hidden_size),
        str(cfg.action_mode),
        int(cfg.continuous_action_dim),
        int(cfg.transformer_layers),
        int(cfg.transformer_heads),
        float(cfg.transformer_dropout),
        bool(getattr(cfg, "transformer_temporal_module", False)),
        int(getattr(cfg, "transformer_temporal_kernel_size", 5)),
        int(getattr(cfg, "transformer_temporal_layers", 1)),
        int(getattr(cfg, "transformer_memory_tokens", 8)),
        int(getattr(cfg, "transformer_memory_context_length", 32)),
        bool(getattr(cfg, "transformer_use_causal_attention", True)),
        bool(centralized_critic_enabled(cfg)),
        str(getattr(cfg, "central_critic_pooling", "flat")),
        int(getattr(cfg, "central_critic_max_vehicles", 64)),
        int(getattr(cfg, "central_critic_attention_heads", 4)),
    )

def _cached_eval_policy(
    cfg: PSGAILConfig,
    policy_state_dict: dict[str, torch.Tensor],
    policy_obs_dim: int,
    critic_obs_dim: int,
) -> nn.Module:
    key = _eval_policy_cache_key(cfg, policy_obs_dim, critic_obs_dim)
    policy = _EVAL_POLICY_CACHE.get(key)
    if policy is None:
        policy = _make_policy_from_state_dict(
            policy_state_dict,
            cfg,
            int(policy_obs_dim),
            int(critic_obs_dim),
            torch.device("cpu"),
        )
        _EVAL_POLICY_CACHE[key] = policy
    else:
        policy.load_state_dict(policy_state_dict)
        policy.eval()
    return policy

def _matched_eval_env_cache_key(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
    vehicle_id: int | None,
    all_vehicle: bool,
) -> tuple[object, ...]:
    return (
        "matched_all" if bool(all_vehicle) else "matched_single",
        str(cfg.scene),
        os.path.abspath(str(cfg.episode_root)),
        str(split),
        str(episode_name),
        None if vehicle_id is None else int(vehicle_id),
        str(cfg.action_mode),
        str(cfg.max_surrounding),
        int(cfg.cells),
        float(cfg.maximum_range),
        int(cfg.simulation_frequency),
        int(cfg.policy_frequency),
        int(cfg.max_episode_steps),
        bool(cfg.enable_collision),
        bool(cfg.terminate_when_all_controlled_crashed),
        bool(cfg.allow_idm),
    )

def _get_matched_eval_env(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
    vehicle_id: int | None = None,
    all_vehicle: bool,
) -> tuple[gym.Env, bool]:
    global _EVAL_ENV_CACHE_HITS, _EVAL_ENV_CACHE_MISSES
    cache_enabled = bool(getattr(cfg, "evaluation_cache_envs", True))
    if not cache_enabled:
        env = (
            _make_matched_eval_all_vehicle_env(cfg, split=split, episode_name=episode_name)
            if all_vehicle
            else _make_matched_eval_env(cfg, split=split, episode_name=episode_name, vehicle_id=int(vehicle_id))
        )
        _EVAL_ENV_CACHE_MISSES += 1
        return env, False
    key = _matched_eval_env_cache_key(
        cfg,
        split=split,
        episode_name=episode_name,
        vehicle_id=vehicle_id,
        all_vehicle=all_vehicle,
    )
    env = _EVAL_ENV_CACHE.get(key)
    if env is not None:
        _EVAL_ENV_CACHE.move_to_end(key)
        _EVAL_ENV_CACHE_HITS += 1
        env.unwrapped.config["expert_test_mode"] = True
        return env, True
    env = (
        _make_matched_eval_all_vehicle_env(cfg, split=split, episode_name=episode_name)
        if all_vehicle
        else _make_matched_eval_env(cfg, split=split, episode_name=episode_name, vehicle_id=int(vehicle_id))
    )
    _EVAL_ENV_CACHE[key] = env
    _EVAL_ENV_CACHE_MISSES += 1
    max_cached = max(0, int(getattr(cfg, "evaluation_max_cached_envs_per_worker", 0)))
    while max_cached > 0 and len(_EVAL_ENV_CACHE) > max_cached:
        _old_key, old_env = _EVAL_ENV_CACHE.popitem(last=False)
        old_env.close()
    return env, True

def _matched_eval_worker(
    cfg: PSGAILConfig,
    policy_state_dict: dict[str, torch.Tensor],
    policy_obs_dim: int,
    critic_obs_dim: int,
    split: str,
    prefix: str,
    worker_id: int,
    scenarios: list[tuple[int, str, int]] | None,
    episode_names: list[tuple[int, str]] | None,
) -> dict[str, object]:
    _configure_evaluation_worker_threads(cfg)
    worker_seed = int(cfg.seed) + 700_000 + int(worker_id)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_cfg = replace(cfg, device="cpu", evaluation_num_workers=1)
    hit_start = _EVAL_ENV_CACHE_HITS
    miss_start = _EVAL_ENV_CACHE_MISSES
    started = time.perf_counter()
    policy = _cached_eval_policy(worker_cfg, policy_state_dict, int(policy_obs_dim), int(critic_obs_dim))
    policy_load_seconds = time.perf_counter() - started
    metrics = _evaluate_policy_matched_trajectories_impl(
        policy,
        worker_cfg,
        torch.device("cpu"),
        split=split,
        episodes=len(episode_names or scenarios or ()),
        prefix=prefix,
        scenarios=scenarios,
        episode_names=episode_names,
        include_raw=True,
    )
    metrics[f"__raw/{prefix}/policy_load_seconds"] = float(policy_load_seconds)
    metrics[f"__raw/{prefix}/env_cache_hits"] = float(_EVAL_ENV_CACHE_HITS - hit_start)
    metrics[f"__raw/{prefix}/env_cache_misses"] = float(_EVAL_ENV_CACHE_MISSES - miss_start)
    return metrics

def _evaluate_policy_matched_all_vehicle_episodes(
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    split: str,
    episodes: int,
    prefix: str,
    episode_names: list[str] | None = None,
    include_raw: bool = False,
) -> dict[str, float]:
    raw_episode_names = (
        list(episode_names)
        if episode_names is not None
        else _evaluation_episode_names(cfg, split=split, episodes=int(episodes))
    )
    if not raw_episode_names:
        return {}
    indexed_episode_names: list[tuple[int, str]] = []
    for local_idx, item in enumerate(raw_episode_names):
        if isinstance(item, tuple) and len(item) == 2:
            indexed_episode_names.append((int(item[0]), str(item[1])))
        else:
            indexed_episode_names.append((int(local_idx), str(item)))
    horizons = _parse_evaluation_horizons(cfg)
    max_steps = max(1, int(max(horizons) * int(cfg.policy_frequency)))
    max_steps = min(max_steps, max(1, int(cfg.max_episode_steps)))
    squared: dict[int, dict[str, list[float]]] = {
        horizon: {"x": [], "y": [], "position": [], "speed": [], "lane_offset": []}
        for horizon in horizons
    }
    final_squared: dict[str, list[float]] = {
        "x": [], "y": [], "position": [], "speed": [], "lane_offset": []
    }
    collision_steps = 0
    offroad_steps = 0
    hard_brake_steps = 0
    total_agent_steps = 0
    skipped_missing_expert = 0
    skipped_bad_reference = 0
    skipped_empty_rollout = 0
    episode_lengths: list[int] = []
    controlled_vehicle_counts: list[int] = []
    evaluated_episodes = 0
    vehicle_episodes = 0
    evaluated_vehicle_ids: set[int] = set()
    terminated_episodes = 0
    truncated_episodes = 0
    crashed_vehicle_episodes = 0
    offroad_vehicle_episodes = 0
    eval_policy_seconds = 0.0
    eval_step_seconds = 0.0
    eval_reset_seconds = 0.0
    was_training = policy.training
    policy.eval()
    try:
        for episode_idx, episode_name in indexed_episode_names:
            env, env_cached = _get_matched_eval_env(
                cfg,
                split=split,
                episode_name=str(episode_name),
                all_vehicle=True,
            )
            try:
                reset_started = time.perf_counter()
                obs, _info = env.reset(seed=int(cfg.seed) + 100_000 + episode_idx)
                eval_reset_seconds += time.perf_counter() - reset_started
                env.unwrapped.config["expert_test_mode"] = False
                controlled = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
                controlled_vehicle_counts.append(len(controlled))
                expert_states = getattr(env.unwrapped, "_expert_state_by_vehicle_id", {})
                vehicle_data: dict[int, dict[str, object]] = {}
                for vehicle in controlled:
                    vehicle_id = int(getattr(vehicle, "vehicle_ID", len(vehicle_data)))
                    expert_state = expert_states.get(vehicle_id)
                    if expert_state is None:
                        skipped_missing_expert += 1
                        continue
                    ref_xy = np.asarray(expert_state.get("ref_xy", []), dtype=np.float32)
                    ref_v = np.asarray(expert_state.get("ref_v", []), dtype=np.float32).reshape(-1)
                    if ref_xy.ndim != 2 or ref_xy.shape[1] < 2 or ref_v.size == 0:
                        skipped_bad_reference += 1
                        continue
                    vehicle_data[vehicle_id] = {
                        "vehicle": vehicle,
                        "ref_xy": ref_xy,
                        "ref_v": ref_v,
                        "pred_xy": [],
                        "pred_speed": [],
                        "pred_lane_offset": [],
                        "expert_lane_offset": [],
                    }
                if not vehicle_data:
                    skipped_empty_rollout += 1
                    continue
                eval_memory = (
                    policy.initial_memory(len(controlled), device=device, dtype=torch.float32)
                    if recurrent_policy_enabled(policy)
                    else None
                )
                length = 0
                last_terminated = False
                last_truncated = False
                episode_crashed_vehicle_ids: set[int] = set()
                episode_offroad_vehicle_ids: set[int] = set()
                for _step in range(max_steps):
                    live_controlled = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
                    if not live_controlled:
                        break
                    for vehicle in live_controlled:
                        vehicle_id = int(getattr(vehicle, "vehicle_ID", -1))
                        data = vehicle_data.get(vehicle_id)
                        if data is None:
                            continue
                        position = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=np.float32).reshape(-1)[:2]
                        pred_xy = data["pred_xy"]
                        pred_speed = data["pred_speed"]
                        pred_lane_offset = data["pred_lane_offset"]
                        expert_lane_offset = data["expert_lane_offset"]
                        ref_xy = data["ref_xy"]
                        assert isinstance(pred_xy, list)
                        assert isinstance(pred_speed, list)
                        assert isinstance(pred_lane_offset, list)
                        assert isinstance(expert_lane_offset, list)
                        assert isinstance(ref_xy, np.ndarray)
                        pred_xy.append(position.copy())
                        pred_speed.append(float(getattr(vehicle, "speed", 0.0)))
                        pred_lane_offset.append(_lane_offset_for_position(env, position, vehicle))
                        expert_idx = min(len(pred_xy) - 1, ref_xy.shape[0] - 1)
                        expert_lane_offset.append(_lane_offset_for_position(env, ref_xy[expert_idx, :2]))

                    if recurrent_policy_enabled(policy):
                        policy_started = time.perf_counter()
                        action_tuple, eval_memory = _deterministic_policy_action_tuple(
                            policy,
                            env,
                            obs,
                            cfg,
                            device,
                            memory=eval_memory,
                            return_memory=True,
                        )
                        eval_policy_seconds += time.perf_counter() - policy_started
                    else:
                        policy_started = time.perf_counter()
                        action_tuple = _deterministic_policy_action_tuple(policy, env, obs, cfg, device)
                        eval_policy_seconds += time.perf_counter() - policy_started
                    accels = _physical_accels_from_actions(action_tuple, cfg)
                    hard_brake_steps += int(np.sum(accels < float(cfg.hard_brake_accel_threshold)))
                    step_started = time.perf_counter()
                    obs, _reward, terminated, truncated, info = env.step(action_tuple)
                    eval_step_seconds += time.perf_counter() - step_started
                    crash_flags = list(info.get("controlled_vehicle_crashes", []) or [])
                    offroad_flags = list(info.get("controlled_vehicle_offroad", []) or [])
                    info_vehicle_ids = list(info.get("controlled_vehicle_ids", []) or [])
                    collision_steps += int(sum(bool(flag) for flag in crash_flags))
                    offroad_steps += int(sum(bool(flag) for flag in offroad_flags))
                    for flag_index, flag in enumerate(crash_flags):
                        if bool(flag) and flag_index < len(info_vehicle_ids):
                            episode_crashed_vehicle_ids.add(int(info_vehicle_ids[flag_index]))
                    for flag_index, flag in enumerate(offroad_flags):
                        if bool(flag) and flag_index < len(info_vehicle_ids):
                            episode_offroad_vehicle_ids.add(int(info_vehicle_ids[flag_index]))
                    total_agent_steps += int(max(len(crash_flags), len(offroad_flags), len(live_controlled)))
                    length += 1
                    last_terminated = bool(terminated)
                    last_truncated = bool(truncated)
                    if terminated or truncated:
                        break
                if length <= 0:
                    skipped_empty_rollout += 1
                    continue
                evaluated_episodes += 1
                terminated_episodes += int(last_terminated)
                truncated_episodes += int(last_truncated)
                crashed_vehicle_episodes += len(episode_crashed_vehicle_ids & set(vehicle_data.keys()))
                offroad_vehicle_episodes += len(episode_offroad_vehicle_ids & set(vehicle_data.keys()))
                episode_lengths.append(length)
                for vehicle_id, data in vehicle_data.items():
                    pred_xy = data["pred_xy"]
                    pred_speed = data["pred_speed"]
                    pred_lane_offset = data["pred_lane_offset"]
                    expert_lane_offset = data["expert_lane_offset"]
                    ref_xy = data["ref_xy"]
                    ref_v = data["ref_v"]
                    assert isinstance(pred_xy, list)
                    assert isinstance(pred_speed, list)
                    assert isinstance(pred_lane_offset, list)
                    assert isinstance(expert_lane_offset, list)
                    assert isinstance(ref_xy, np.ndarray)
                    assert isinstance(ref_v, np.ndarray)
                    if not pred_xy:
                        skipped_empty_rollout += 1
                        continue
                    vehicle_episodes += 1
                    evaluated_vehicle_ids.add(int(vehicle_id))
                    final_idx = min(len(pred_xy), ref_xy.shape[0], ref_v.size) - 1
                    if final_idx >= 0:
                        dx = float(pred_xy[final_idx][0] - ref_xy[final_idx, 0])
                        dy = float(pred_xy[final_idx][1] - ref_xy[final_idx, 1])
                        ds = float(pred_speed[final_idx] - ref_v[final_idx])
                        dlat = float(pred_lane_offset[final_idx] - expert_lane_offset[final_idx])
                        final_squared["x"].append(dx * dx)
                        final_squared["y"].append(dy * dy)
                        final_squared["position"].append(dx * dx + dy * dy)
                        final_squared["speed"].append(ds * ds)
                        if np.isfinite(dlat):
                            final_squared["lane_offset"].append(dlat * dlat)
                    for horizon in horizons:
                        idx = int(horizon * int(cfg.policy_frequency)) - 1
                        if idx < 0 or idx >= len(pred_xy) or idx >= ref_xy.shape[0] or idx >= ref_v.size:
                            continue
                        dx = float(pred_xy[idx][0] - ref_xy[idx, 0])
                        dy = float(pred_xy[idx][1] - ref_xy[idx, 1])
                        ds = float(pred_speed[idx] - ref_v[idx])
                        dlat = float(pred_lane_offset[idx] - expert_lane_offset[idx])
                        squared[horizon]["x"].append(dx * dx)
                        squared[horizon]["y"].append(dy * dy)
                        squared[horizon]["position"].append(dx * dx + dy * dy)
                        squared[horizon]["speed"].append(ds * ds)
                        if np.isfinite(dlat):
                            squared[horizon]["lane_offset"].append(dlat * dlat)
            finally:
                if not env_cached:
                    env.close()
    finally:
        if was_training:
            policy.train()
    metrics = _matched_eval_metrics(
        prefix=prefix,
        attempted_episodes=len(indexed_episode_names),
        evaluated_episodes=evaluated_episodes,
        skipped_missing_expert=skipped_missing_expert,
        skipped_bad_reference=skipped_bad_reference,
        skipped_empty_rollout=skipped_empty_rollout,
        total_steps=total_agent_steps,
        collision_steps=collision_steps,
        offroad_steps=offroad_steps,
        hard_brake_steps=hard_brake_steps,
        episode_lengths=episode_lengths,
        squared=squared,
        final_squared=final_squared,
        horizons=horizons,
        terminated_episodes=terminated_episodes,
        truncated_episodes=truncated_episodes,
        crashed_vehicle_episodes=crashed_vehicle_episodes,
        offroad_vehicle_episodes=offroad_vehicle_episodes,
        vehicles=len(evaluated_vehicle_ids),
        vehicle_episodes=vehicle_episodes,
        controlled_vehicle_counts=controlled_vehicle_counts,
        vehicle_ids=evaluated_vehicle_ids,
        include_raw=include_raw,
    )
    if include_raw:
        metrics[f"__raw/{prefix}/policy_forward_seconds"] = float(eval_policy_seconds)
        metrics[f"__raw/{prefix}/env_step_seconds"] = float(eval_step_seconds)
        metrics[f"__raw/{prefix}/env_reset_seconds"] = float(eval_reset_seconds)
    return metrics

def evaluate_policy_matched_trajectories(
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    split: str,
    episodes: int,
    prefix: str,
    evaluation_executor: ProcessPoolExecutor | None = None,
) -> dict[str, float]:
    requested_workers = max(1, int(getattr(cfg, "evaluation_num_workers", 1)))
    all_vehicle_mode = bool(getattr(cfg, f"{prefix}_control_all_vehicles", False))
    if all_vehicle_mode:
        selected_episode_names = _evaluation_episode_names(cfg, split=split, episodes=int(episodes))
        indexed_episode_names = [(idx, name) for idx, name in enumerate(selected_episode_names)]
        indexed_scenarios = None
        parallel_items: list[object] = indexed_episode_names
    else:
        selected_scenarios = _evaluation_scenarios(cfg, split=split, episodes=int(episodes))
        indexed_scenarios = [(idx, name, vehicle_id) for idx, (name, vehicle_id) in enumerate(selected_scenarios)]
        indexed_episode_names = None
        parallel_items = indexed_scenarios
    active_workers = min(requested_workers, len(parallel_items))
    if active_workers <= 1 and evaluation_executor is None:
        with evaluation_thread_context(cfg):
            return _strip_internal_matched_metrics(_evaluate_policy_matched_trajectories_impl(
                policy,
                cfg,
                device,
                split=split,
                episodes=episodes,
                prefix=prefix,
                scenarios=indexed_scenarios,
                episode_names=indexed_episode_names,
                include_raw=True,
            ))

    policy_obs_dim = _policy_input_dim(policy)
    critic_obs_dim = int(getattr(policy, "critic_obs_dim", central_critic_observation_dim(policy_obs_dim, cfg)))
    policy_state_dict = _cpu_state_dict(policy)
    horizons = _parse_evaluation_horizons(cfg)
    chunks = _chunk_evenly(parallel_items, active_workers)
    print(
        f"[{prefix}] parallel evaluation workers={active_workers} "
        f"worker_threads={max(1, int(getattr(cfg, 'evaluation_worker_threads', 2)))} "
        f"items={len(parallel_items)}",
        flush=True,
    )
    executor_context = (
        nullcontext(evaluation_executor)
        if evaluation_executor is not None
        else ProcessPoolExecutor(max_workers=active_workers, mp_context=mp.get_context("spawn"))
    )
    with executor_context as executor:
        futures = []
        for worker_id, chunk in enumerate(chunks):
            worker_scenarios = None if all_vehicle_mode else list(chunk)
            worker_episode_names = list(chunk) if all_vehicle_mode else None
            futures.append(
                executor.submit(
                    _matched_eval_worker,
                    cfg,
                    policy_state_dict,
                    policy_obs_dim,
                    critic_obs_dim,
                    split,
                    prefix,
                    worker_id,
                    worker_scenarios,
                    worker_episode_names,
                )
            )
        parts = [future.result() for future in futures]
    metrics = _combine_matched_eval_metric_dicts(parts, prefix=prefix, horizons=horizons)
    print(
        f"[{prefix}] eval_timing "
        f"policy_load={metrics.get(f'{prefix}/eval_policy_load_seconds', 0.0):.3f}s "
        f"reset={metrics.get(f'{prefix}/eval_env_reset_seconds', 0.0):.3f}s "
        f"policy={metrics.get(f'{prefix}/eval_policy_forward_seconds', 0.0):.3f}s "
        f"env_step={metrics.get(f'{prefix}/eval_env_step_seconds', 0.0):.3f}s "
        f"cache_hits={metrics.get(f'{prefix}/eval_env_cache_hits', 0.0):.0f} "
        f"cache_misses={metrics.get(f'{prefix}/eval_env_cache_misses', 0.0):.0f}",
        flush=True,
    )
    return metrics

def _evaluate_policy_matched_trajectories_impl(
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    split: str,
    episodes: int,
    prefix: str,
    scenarios: list[tuple[str, int]] | None = None,
    episode_names: list[str] | None = None,
    include_raw: bool = False,
) -> dict[str, float]:
    if bool(getattr(cfg, f"{prefix}_control_all_vehicles", False)):
        return _evaluate_policy_matched_all_vehicle_episodes(
            policy,
            cfg,
            device,
            split=split,
            episodes=int(episodes),
            prefix=prefix,
            episode_names=episode_names,
            include_raw=include_raw,
        )
    raw_scenarios = (
        list(scenarios)
        if scenarios is not None
        else _evaluation_scenarios(cfg, split=split, episodes=int(episodes))
    )
    if not raw_scenarios:
        return {}
    indexed_scenarios: list[tuple[int, str, int]] = []
    for local_idx, item in enumerate(raw_scenarios):
        if isinstance(item, tuple) and len(item) == 3:
            indexed_scenarios.append((int(item[0]), str(item[1]), int(item[2])))
        else:
            episode_name, vehicle_id = item
            indexed_scenarios.append((int(local_idx), str(episode_name), int(vehicle_id)))
    horizons = _parse_evaluation_horizons(cfg)
    max_steps = max(1, int(max(horizons) * int(cfg.policy_frequency)))
    max_steps = min(max_steps, max(1, int(cfg.max_episode_steps)))
    squared: dict[int, dict[str, list[float]]] = {
        horizon: {"x": [], "y": [], "position": [], "speed": [], "lane_offset": []}
        for horizon in horizons
    }
    final_squared: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "position": [],
        "speed": [],
        "lane_offset": [],
    }
    collision_steps = 0
    offroad_steps = 0
    hard_brake_steps = 0
    total_steps = 0
    skipped_missing_expert = 0
    skipped_bad_reference = 0
    skipped_empty_rollout = 0
    episode_lengths: list[int] = []
    evaluated_episodes = 0
    evaluated_vehicle_ids: set[int] = set()
    terminated_episodes = 0
    truncated_episodes = 0
    crashed_vehicle_episodes = 0
    offroad_vehicle_episodes = 0
    eval_policy_seconds = 0.0
    eval_step_seconds = 0.0
    eval_reset_seconds = 0.0
    was_training = policy.training
    policy.eval()
    try:
        for scenario_idx, episode_name, vehicle_id in indexed_scenarios:
            env, env_cached = _get_matched_eval_env(
                cfg,
                split=split,
                episode_name=episode_name,
                vehicle_id=int(vehicle_id),
                all_vehicle=False,
            )
            try:
                reset_started = time.perf_counter()
                obs, _info = env.reset(seed=int(cfg.seed) + 100_000 + scenario_idx)
                eval_reset_seconds += time.perf_counter() - reset_started
                env.unwrapped.config["expert_test_mode"] = False
                expert_state = getattr(env.unwrapped, "_expert_state_by_vehicle_id", {}).get(int(vehicle_id))
                if expert_state is None:
                    skipped_missing_expert += 1
                    continue
                ref_xy = np.asarray(expert_state.get("ref_xy", []), dtype=np.float32)
                ref_v = np.asarray(expert_state.get("ref_v", []), dtype=np.float32).reshape(-1)
                if ref_xy.ndim != 2 or ref_xy.shape[1] < 2 or ref_v.size == 0:
                    skipped_bad_reference += 1
                    continue
                pred_xy: list[np.ndarray] = []
                pred_speed: list[float] = []
                pred_lane_offset: list[float] = []
                expert_lane_offset: list[float] = []
                length = 0
                episode_crashed = False
                episode_offroad = False
                eval_memory = (
                    policy.initial_memory(1, device=device, dtype=torch.float32)
                    if recurrent_policy_enabled(policy)
                    else None
                )
                for _step in range(max_steps):
                    vehicle = _first_controlled_vehicle(env)
                    if vehicle is None:
                        break
                    position = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=np.float32).reshape(-1)[:2]
                    pred_xy.append(position.copy())
                    pred_speed.append(float(getattr(vehicle, "speed", 0.0)))
                    pred_lane_offset.append(_lane_offset_for_position(env, position, vehicle))
                    expert_idx = min(length, ref_xy.shape[0] - 1)
                    expert_lane_offset.append(_lane_offset_for_position(env, ref_xy[expert_idx, :2]))
                    if recurrent_policy_enabled(policy):
                        policy_started = time.perf_counter()
                        action_tuple, eval_memory = _deterministic_policy_action_tuple(
                            policy,
                            env,
                            obs,
                            cfg,
                            device,
                            memory=eval_memory,
                            return_memory=True,
                        )
                        eval_policy_seconds += time.perf_counter() - policy_started
                    else:
                        policy_started = time.perf_counter()
                        action_tuple = _deterministic_policy_action_tuple(policy, env, obs, cfg, device)
                        eval_policy_seconds += time.perf_counter() - policy_started
                    accel = _physical_accel_from_action(action_tuple, cfg)
                    if np.isfinite(accel) and accel < float(cfg.hard_brake_accel_threshold):
                        hard_brake_steps += 1
                    step_started = time.perf_counter()
                    obs, _reward, terminated, truncated, info = env.step(action_tuple)
                    eval_step_seconds += time.perf_counter() - step_started
                    crash_flags = list(info.get("controlled_vehicle_crashes", []) or [])
                    offroad_flags = list(info.get("controlled_vehicle_offroad", []) or [])
                    episode_crashed = episode_crashed or any(bool(flag) for flag in crash_flags)
                    episode_offroad = episode_offroad or any(bool(flag) for flag in offroad_flags)
                    collision_steps += int(any(bool(flag) for flag in crash_flags))
                    offroad_steps += int(any(bool(flag) for flag in offroad_flags))
                    total_steps += 1
                    length += 1
                    if terminated or truncated:
                        break
                if length <= 0:
                    skipped_empty_rollout += 1
                    continue
                evaluated_episodes += 1
                evaluated_vehicle_ids.add(int(vehicle_id))
                terminated_episodes += int(bool(terminated))
                truncated_episodes += int(bool(truncated))
                crashed_vehicle_episodes += int(episode_crashed)
                offroad_vehicle_episodes += int(episode_offroad)
                episode_lengths.append(length)
                final_idx = min(len(pred_xy), ref_xy.shape[0], ref_v.size) - 1
                if final_idx >= 0:
                    dx = float(pred_xy[final_idx][0] - ref_xy[final_idx, 0])
                    dy = float(pred_xy[final_idx][1] - ref_xy[final_idx, 1])
                    ds = float(pred_speed[final_idx] - ref_v[final_idx])
                    dlat = float(pred_lane_offset[final_idx] - expert_lane_offset[final_idx])
                    final_squared["x"].append(dx * dx)
                    final_squared["y"].append(dy * dy)
                    final_squared["position"].append(dx * dx + dy * dy)
                    final_squared["speed"].append(ds * ds)
                    if np.isfinite(dlat):
                        final_squared["lane_offset"].append(dlat * dlat)
                for horizon in horizons:
                    idx = int(horizon * int(cfg.policy_frequency)) - 1
                    if idx < 0 or idx >= len(pred_xy) or idx >= ref_xy.shape[0] or idx >= ref_v.size:
                        continue
                    dx = float(pred_xy[idx][0] - ref_xy[idx, 0])
                    dy = float(pred_xy[idx][1] - ref_xy[idx, 1])
                    ds = float(pred_speed[idx] - ref_v[idx])
                    dlat = float(pred_lane_offset[idx] - expert_lane_offset[idx])
                    squared[horizon]["x"].append(dx * dx)
                    squared[horizon]["y"].append(dy * dy)
                    squared[horizon]["position"].append(dx * dx + dy * dy)
                    squared[horizon]["speed"].append(ds * ds)
                    if np.isfinite(dlat):
                        squared[horizon]["lane_offset"].append(dlat * dlat)
            finally:
                if not env_cached:
                    env.close()
    finally:
        if was_training:
            policy.train()

    metrics = _matched_eval_metrics(
        prefix=prefix,
        attempted_episodes=len(indexed_scenarios),
        evaluated_episodes=evaluated_episodes,
        skipped_missing_expert=skipped_missing_expert,
        skipped_bad_reference=skipped_bad_reference,
        skipped_empty_rollout=skipped_empty_rollout,
        total_steps=total_steps,
        collision_steps=collision_steps,
        offroad_steps=offroad_steps,
        hard_brake_steps=hard_brake_steps,
        episode_lengths=episode_lengths,
        squared=squared,
        final_squared=final_squared,
        horizons=horizons,
        terminated_episodes=terminated_episodes,
        truncated_episodes=truncated_episodes,
        crashed_vehicle_episodes=crashed_vehicle_episodes,
        offroad_vehicle_episodes=offroad_vehicle_episodes,
        vehicles=evaluated_episodes,
        vehicle_episodes=evaluated_episodes,
        vehicle_ids=evaluated_vehicle_ids,
        include_raw=include_raw,
    )
    if include_raw:
        metrics[f"__raw/{prefix}/policy_forward_seconds"] = float(eval_policy_seconds)
        metrics[f"__raw/{prefix}/env_step_seconds"] = float(eval_step_seconds)
        metrics[f"__raw/{prefix}/env_reset_seconds"] = float(eval_reset_seconds)
    return metrics

__all__ = [
    'evaluation_thread_context',
    '_parse_evaluation_horizons',
    '_evaluation_scenarios',
    '_evaluation_episode_names',
    '_make_matched_eval_env',
    '_make_matched_eval_all_vehicle_env',
    '_deterministic_policy_action_tuple',
    '_lane_offset_for_position',
    '_first_controlled_vehicle',
    '_physical_accel_from_action',
    '_physical_accels_from_actions',
    '_matched_eval_metrics',
    '_strip_internal_matched_metrics',
    '_combine_matched_eval_metric_dicts',
    '_chunk_evenly',
    '_policy_input_dim',
    '_cpu_state_dict',
    '_configure_evaluation_worker_threads',
    '_eval_policy_cache_key',
    '_cached_eval_policy',
    '_matched_eval_env_cache_key',
    '_get_matched_eval_env',
    '_matched_eval_worker',
    '_evaluate_policy_matched_all_vehicle_episodes',
    'evaluate_policy_matched_trajectories',
    '_evaluate_policy_matched_trajectories_impl'
]
