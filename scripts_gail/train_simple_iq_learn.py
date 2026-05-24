#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for lean local envs.
    tqdm = None

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scripts_gail.ps_gail.config import PSGAILConfig, should_save_checkpoint_video
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.envs import make_training_env
from scripts_gail.ps_gail.monitoring import WandbMonitor
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.observations import flatten_agent_observations, policy_observations_from_flat
from scripts_gail.ps_gail.schedule import config_for_round
from scripts_gail.ps_gail.trainer import (
    evaluation_thread_context,
    evaluate_policy_matched_trajectories,
    infer_continuous_action_dim,
    infer_policy_obs_dim,
    make_evaluation_executor,
    resolve_device,
    set_optimizer_lr,
)
from scripts_gail.train_simple_airl import save_checkpoint_video


@dataclass(frozen=True)
class ExpertReplayTensors:
    policy_observations: torch.Tensor
    actions_continuous_env: torch.Tensor
    next_policy_observations: torch.Tensor
    dones: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.policy_observations.shape[0])


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(obs_dim) + int(action_dim), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


@dataclass
class ConvergenceTracker:
    best_score: float = float("-inf")
    best_update: int = 0
    eval_count: int = 0
    stale_evals: int = 0


def _require_finite_metrics(metrics: dict[str, float], *, update_idx: int) -> None:
    bad = [
        key
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.floating)) and not np.isfinite(float(value))
    ]
    if bad:
        raise RuntimeError(f"Non-finite IQ-Learn metric(s) at update {update_idx}: {', '.join(sorted(bad))}")


def convergence_score(
    update_stats: dict[str, float],
    eval_stats: dict[str, float],
    *,
    crash_penalty: float,
    bc_score_weight: float,
) -> float:
    return (
        float(eval_stats.get("eval/mean_length", 0.0))
        + float(eval_stats.get("eval/mean_reward", 0.0))
        - float(crash_penalty) * float(eval_stats.get("eval/crashes", 0.0))
        - float(bc_score_weight) * float(update_stats.get("bc_loss", 0.0))
    )


def convergence_reached(
    update_stats: dict[str, float],
    eval_stats: dict[str, float],
    *,
    target_eval_mean_length: float,
    target_bc_loss: float,
    max_eval_crashes: float,
) -> bool:
    if float(target_eval_mean_length) > 0.0 and float(eval_stats.get("eval/mean_length", 0.0)) < float(target_eval_mean_length):
        return False
    if float(target_bc_loss) > 0.0 and float(update_stats.get("bc_loss", float("inf"))) > float(target_bc_loss):
        return False
    if float(max_eval_crashes) >= 0.0 and float(eval_stats.get("eval/crashes", 0.0)) > float(max_eval_crashes):
        return False
    return float(target_eval_mean_length) > 0.0 or float(target_bc_loss) > 0.0


def _resolve_replay_device(requested: str, train_device: torch.device) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return train_device if train_device.type == "cuda" else torch.device("cpu")
    if requested == "cuda":
        if train_device.type != "cuda":
            raise ValueError("--replay-device cuda requires training on a CUDA device.")
        return train_device
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported replay device {requested!r}; expected auto, cuda, or cpu.")


def _as_replay_tensor(
    array: np.ndarray,
    *,
    dtype: torch.dtype,
    replay_device: torch.device,
    pin_cpu_memory: bool,
) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=dtype)
    if replay_device.type == "cpu":
        if bool(pin_cpu_memory) and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        return tensor
    return tensor.to(replay_device, non_blocking=True)


def make_expert_replay_tensors(
    expert,
    *,
    replay_device: torch.device,
    pin_cpu_memory: bool,
) -> ExpertReplayTensors:
    return ExpertReplayTensors(
        policy_observations=_as_replay_tensor(
            expert.policy_observations,
            dtype=torch.float32,
            replay_device=replay_device,
            pin_cpu_memory=pin_cpu_memory,
        ),
        actions_continuous_env=_as_replay_tensor(
            expert.actions_continuous_env,
            dtype=torch.float32,
            replay_device=replay_device,
            pin_cpu_memory=pin_cpu_memory,
        ),
        next_policy_observations=_as_replay_tensor(
            expert.next_policy_observations,
            dtype=torch.float32,
            replay_device=replay_device,
            pin_cpu_memory=pin_cpu_memory,
        ),
        dones=_as_replay_tensor(
            expert.dones.astype(np.float32, copy=False),
            dtype=torch.float32,
            replay_device=replay_device,
            pin_cpu_memory=pin_cpu_memory,
        ),
    )


def _batch_from_replay(
    replay: ExpertReplayTensors,
    *,
    batch_size: int,
    train_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    replay_device = replay.policy_observations.device
    idx = torch.randint(
        replay.num_samples,
        (max(1, int(batch_size)),),
        device=replay_device,
    )
    obs_t = replay.policy_observations.index_select(0, idx)
    actions_t = replay.actions_continuous_env.index_select(0, idx)
    next_obs_t = replay.next_policy_observations.index_select(0, idx)
    dones_t = replay.dones.index_select(0, idx)
    if obs_t.device != train_device:
        obs_t = obs_t.to(train_device, non_blocking=True)
        actions_t = actions_t.to(train_device, non_blocking=True)
        next_obs_t = next_obs_t.to(train_device, non_blocking=True)
        dones_t = dones_t.to(train_device, non_blocking=True)
    return obs_t, actions_t, next_obs_t, dones_t


def _replay_size_mb(replay: ExpertReplayTensors) -> float:
    tensors = (
        replay.policy_observations,
        replay.actions_continuous_env,
        replay.next_policy_observations,
        replay.dones,
    )
    total_bytes = sum(int(t.numel()) * int(t.element_size()) for t in tensors)
    return float(total_bytes / (1024.0 * 1024.0))


def _cuda_peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def _policy_input_dim(policy: nn.Module) -> int:
    obs_dim = getattr(policy, "obs_dim", None)
    if obs_dim is not None:
        return int(obs_dim)
    encoder = getattr(policy, "encoder", None)
    first = encoder[0] if isinstance(encoder, nn.Sequential) and len(encoder) else None
    in_features = getattr(first, "in_features", None)
    if in_features is None:
        raise RuntimeError("Cannot infer IQ-Learn policy observation dimension for parallel evaluation.")
    return int(in_features)


def _cpu_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}


def _chunk_episode_indices(episodes: int, workers: int) -> list[list[int]]:
    indices = list(range(max(1, int(episodes))))
    active = min(max(1, int(workers)), len(indices))
    return [indices[start::active] for start in range(active) if indices[start::active]]


def _metric_or_nan(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, float("nan")))


def _print_matched_eval(prefix: str, step_label: str, metrics: dict[str, float]) -> None:
    if not metrics:
        return
    horizon_parts = []
    for horizon in (1, 5, 10, 20):
        value = _metric_or_nan(metrics, f"{prefix}/rmse_position_{horizon}s")
        horizon_parts.append(f"rmse_pos_{horizon}s={value:.4f}")
    print(
        f"[{prefix} {step_label}] "
        f"episodes={metrics.get(f'{prefix}/episodes', 0):.0f}/"
        f"{metrics.get(f'{prefix}/attempted_episodes', metrics.get(f'{prefix}/episodes', 0)):.0f} "
        f"mean_len={metrics.get(f'{prefix}/mean_episode_length', 0.0):.1f} "
        f"terminated={metrics.get(f'{prefix}/terminated_episodes', 0):.0f} "
        f"truncated={metrics.get(f'{prefix}/truncated_episodes', 0):.0f} "
        f"{' '.join(horizon_parts)} "
        f"rmse_pos_final={_metric_or_nan(metrics, f'{prefix}/rmse_position_final'):.4f} "
        f"collision={metrics.get(f'{prefix}/vehicle_crash_rate', metrics.get(f'{prefix}/collision_rate', 0.0)):.4f} "
        f"offroad={metrics.get(f'{prefix}/vehicle_offroad_rate', metrics.get(f'{prefix}/offroad_duration_rate', 0.0)):.4f} "
        f"collision_dur={metrics.get(f'{prefix}/collision_duration_rate', metrics.get(f'{prefix}/collision_rate', 0.0)):.4f} "
        f"offroad_dur={metrics.get(f'{prefix}/offroad_duration_rate', 0.0):.4f} "
        f"hard_brake={metrics.get(f'{prefix}/hard_brake_rate', 0.0):.4f} "
        f"skipped_missing_expert={metrics.get(f'{prefix}/skipped_missing_expert', 0):.0f}"
    )


def _set_matmul_precision(precision: str) -> None:
    precision = str(precision).lower()
    if precision in {"", "default"}:
        return
    if precision not in {"highest", "high", "medium"}:
        raise ValueError("--torch-matmul-precision must be one of default, highest, high, or medium.")
    setter = getattr(torch, "set_float32_matmul_precision", None)
    if setter is not None:
        setter(precision)


def _sample_policy(
    policy: nn.Module,
    obs: torch.Tensor,
    *,
    log_std_min: float,
    log_std_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, _values = policy(obs)
    log_std = getattr(policy, "log_std", None)
    if log_std is None:
        raise RuntimeError("IQ-Learn test trainer expects a continuous policy.")
    log_std = torch.clamp(log_std, float(log_std_min), float(log_std_max))
    std = torch.exp(log_std).expand_as(mean)
    dist = Independent(Normal(mean, std), 1)
    raw_action = dist.rsample()
    action = torch.clamp(raw_action, -1.0, 1.0)
    return action, dist.log_prob(raw_action), mean


def _make_iq_eval_policy(
    state_dict: dict[str, torch.Tensor],
    cfg: PSGAILConfig,
    policy_obs_dim: int,
) -> nn.Module:
    policy = make_actor_critic(
        cfg.policy_model,
        int(policy_obs_dim),
        int(cfg.hidden_size),
        action_mode="continuous",
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
    )
    policy.load_state_dict(state_dict)
    policy.to(torch.device("cpu"))
    policy.eval()
    return policy


_IQ_EVAL_POLICY_CACHE: dict[tuple[object, ...], nn.Module] = {}
_IQ_EVAL_ENV_CACHE: OrderedDict[tuple[object, ...], object] = OrderedDict()
_IQ_EVAL_ENV_CACHE_HITS = 0
_IQ_EVAL_ENV_CACHE_MISSES = 0


def _iq_eval_policy_cache_key(cfg: PSGAILConfig, policy_obs_dim: int) -> tuple[object, ...]:
    return (
        str(cfg.policy_model),
        int(policy_obs_dim),
        int(cfg.hidden_size),
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
    )


def _cached_iq_eval_policy(
    state_dict: dict[str, torch.Tensor],
    cfg: PSGAILConfig,
    policy_obs_dim: int,
) -> nn.Module:
    key = _iq_eval_policy_cache_key(cfg, int(policy_obs_dim))
    policy = _IQ_EVAL_POLICY_CACHE.get(key)
    if policy is None:
        policy = _make_iq_eval_policy(state_dict, cfg, int(policy_obs_dim))
        _IQ_EVAL_POLICY_CACHE[key] = policy
    else:
        policy.load_state_dict(state_dict)
        policy.to(torch.device("cpu"))
        policy.eval()
    return policy


def _iq_eval_env_cache_key(cfg: PSGAILConfig) -> tuple[object, ...]:
    return (
        str(getattr(cfg, "env_id", "")),
        str(getattr(cfg, "scene", "")),
        os.path.abspath(str(getattr(cfg, "episode_root", ""))),
        str(getattr(cfg, "prebuilt_split", "")),
        str(getattr(cfg, "trajectory_frame", "")),
        str(cfg.action_mode),
        int(cfg.continuous_action_dim),
        bool(cfg.control_all_vehicles),
        float(cfg.percentage_controlled_vehicles),
        str(cfg.max_surrounding),
        int(getattr(cfg, "cells", 0)),
        float(getattr(cfg, "maximum_range", 0.0)),
        int(cfg.max_episode_steps),
        bool(cfg.enable_collision),
        bool(cfg.terminate_when_all_controlled_crashed),
        bool(cfg.allow_idm),
        int(cfg.policy_frequency),
        int(cfg.simulation_frequency),
    )


def _get_iq_eval_env(cfg: PSGAILConfig):
    global _IQ_EVAL_ENV_CACHE_HITS, _IQ_EVAL_ENV_CACHE_MISSES
    if not bool(getattr(cfg, "evaluation_cache_envs", True)):
        _IQ_EVAL_ENV_CACHE_MISSES += 1
        return make_training_env(cfg), False
    key = _iq_eval_env_cache_key(cfg)
    env = _IQ_EVAL_ENV_CACHE.get(key)
    if env is not None:
        _IQ_EVAL_ENV_CACHE.move_to_end(key)
        _IQ_EVAL_ENV_CACHE_HITS += 1
        return env, True
    env = make_training_env(cfg)
    _IQ_EVAL_ENV_CACHE[key] = env
    _IQ_EVAL_ENV_CACHE_MISSES += 1
    max_cached = max(0, int(getattr(cfg, "evaluation_max_cached_envs_per_worker", 0)))
    while max_cached > 0 and len(_IQ_EVAL_ENV_CACHE) > max_cached:
        _old_key, old_env = _IQ_EVAL_ENV_CACHE.popitem(last=False)
        old_env.close()
    return env, True


def _evaluate_policy_worker(
    cfg: PSGAILConfig,
    policy_state_dict: dict[str, torch.Tensor],
    policy_obs_dim: int,
    episode_indices: list[int],
) -> dict[str, object]:
    global _IQ_EVAL_ENV_CACHE_HITS, _IQ_EVAL_ENV_CACHE_MISSES
    threads = max(1, int(getattr(cfg, "evaluation_worker_threads", 2)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    np.random.seed(int(cfg.seed) + 900_000 + min(episode_indices or [0]))
    torch.manual_seed(int(cfg.seed) + 900_000 + min(episode_indices or [0]))
    worker_cfg = replace(cfg, device="cpu", evaluation_num_workers=1)
    hit_start = _IQ_EVAL_ENV_CACHE_HITS
    miss_start = _IQ_EVAL_ENV_CACHE_MISSES
    policy_load_started = time.perf_counter()
    policy = _cached_iq_eval_policy(policy_state_dict, worker_cfg, int(policy_obs_dim))
    policy_load_seconds = time.perf_counter() - policy_load_started
    env, env_cached = _get_iq_eval_env(worker_cfg)
    lengths: list[int] = []
    rewards: list[float] = []
    crashes = 0
    reset_seconds = 0.0
    policy_forward_seconds = 0.0
    env_step_seconds = 0.0
    try:
        for ep_idx in episode_indices:
            reset_started = time.perf_counter()
            obs, _info = env.reset(seed=int(worker_cfg.seed) + 100_000 + int(ep_idx))
            reset_seconds += time.perf_counter() - reset_started
            obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
            eval_memory = (
                policy.initial_memory(len(obs_agents), device=torch.device("cpu"), dtype=torch.float32)
                if bool(getattr(policy, "supports_recurrent_memory", False))
                else None
            )
            total_reward = 0.0
            length = 0
            for _ in range(max(1, int(worker_cfg.max_episode_steps))):
                obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32)
                    policy_started = time.perf_counter()
                    if bool(getattr(policy, "supports_recurrent_memory", False)):
                        mean, _values, eval_memory = policy(
                            obs_tensor,
                            memory=eval_memory,
                            return_memory=True,
                        )
                    else:
                        mean, _values = policy(obs_tensor)
                    policy_forward_seconds += time.perf_counter() - policy_started
                    action_arr = torch.clamp(mean, -1.0, 1.0).detach().numpy().astype(np.float32)
                action = tuple(row.copy() for row in action_arr.reshape(-1, int(worker_cfg.continuous_action_dim)))
                step_started = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                env_step_seconds += time.perf_counter() - step_started
                total_reward += float(np.asarray(reward, dtype=np.float32).mean())
                length += 1
                if bool(info.get("crashed", False)) if isinstance(info, dict) else False:
                    crashes += 1
                if terminated or truncated:
                    break
            lengths.append(length)
            rewards.append(total_reward)
    finally:
        if not env_cached:
            env.close()
    return {
        "lengths": tuple(lengths),
        "rewards": tuple(rewards),
        "crashes": float(crashes),
        "policy_load_seconds": float(policy_load_seconds),
        "policy_forward_seconds": float(policy_forward_seconds),
        "env_step_seconds": float(env_step_seconds),
        "env_reset_seconds": float(reset_seconds),
        "env_cache_hits": float(_IQ_EVAL_ENV_CACHE_HITS - hit_start),
        "env_cache_misses": float(_IQ_EVAL_ENV_CACHE_MISSES - miss_start),
    }



def _clip_symmetric(values: torch.Tensor, limit: float) -> torch.Tensor:
    limit = float(limit)
    if limit <= 0.0:
        return values
    return torch.clamp(values, -limit, limit)


def update_iq_learn(
    policy: nn.Module,
    q_net: SoftQNetwork,
    target_q_net: SoftQNetwork,
    policy_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    replay: ExpertReplayTensors,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    update_idx: int,
    batch_size: int,
    gamma: float,
    alpha: float,
    tau: float,
    bc_coef: float,
    bc_warmup_coef: float,
    bc_warmup_updates: int,
    policy_bc_only_updates: int,
    q_l2_coef: float,
    q_policy_reg_coef: float,
    conservative_q_coef: float,
    target_value_clip: float,
    policy_q_clip: float,
    log_std_min: float,
    log_std_max: float,
    collect_stats: bool,
) -> dict[str, float]:
    obs_t, actions_t, next_obs_t, dones_t = _batch_from_replay(
        replay,
        batch_size=batch_size,
        train_device=device,
    )

    with torch.no_grad():
        next_action, next_log_prob, _next_mean = _sample_policy(
            policy,
            next_obs_t,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        target_v = target_q_net(next_obs_t, next_action) - float(alpha) * next_log_prob
        unclipped_target_v = target_v
        target_v = _clip_symmetric(target_v, target_value_clip)
        target = (1.0 - dones_t) * float(gamma) * target_v
        policy_reg_action, _policy_reg_log_prob, _policy_reg_mean = _sample_policy(
            policy,
            obs_t,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

    expert_q = q_net(obs_t, actions_t)
    policy_reg_q = q_net(obs_t, policy_reg_action)
    random_actions = torch.empty_like(actions_t).uniform_(-1.0, 1.0)
    random_q = q_net(obs_t, random_actions)
    # A compact IQ-Learn-style objective: expert transitions should satisfy a
    # positive Bellman gap, while explicit penalties keep Q scale bounded.
    bellman_gap = expert_q - target
    inverse_q_loss = F.softplus(-bellman_gap).mean()
    q_l2_loss = 0.5 * (torch.square(expert_q).mean() + torch.square(policy_reg_q).mean())
    policy_q_reg_loss = torch.square(policy_reg_q).mean()
    conservative_q_loss = 0.5 * (
        F.softplus(policy_reg_q - expert_q.detach()).mean()
        + F.softplus(random_q - expert_q.detach()).mean()
    )
    q_loss = inverse_q_loss
    if float(q_l2_coef) > 0.0:
        q_loss = q_loss + float(q_l2_coef) * q_l2_loss
    if float(q_policy_reg_coef) > 0.0:
        q_loss = q_loss + float(q_policy_reg_coef) * policy_q_reg_loss
    if float(conservative_q_coef) > 0.0:
        q_loss = q_loss + float(conservative_q_coef) * conservative_q_loss
    q_optimizer.zero_grad()
    q_loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), float(cfg.max_grad_norm))
    q_optimizer.step()

    policy_action, log_prob, mean_action = _sample_policy(
        policy,
        obs_t,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
    )
    policy_q = q_net(obs_t, policy_action)
    policy_q_for_loss = _clip_symmetric(policy_q, policy_q_clip)
    bc_loss = F.mse_loss(mean_action, actions_t)
    bc_weight = float(bc_warmup_coef) if int(update_idx) <= int(bc_warmup_updates) else float(bc_coef)
    policy_rl_loss = (float(alpha) * log_prob - policy_q_for_loss).mean()
    if int(update_idx) <= int(policy_bc_only_updates):
        policy_loss = bc_weight * bc_loss
    else:
        policy_loss = policy_rl_loss + bc_weight * bc_loss
    policy_optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm))
    policy_optimizer.step()

    with torch.no_grad():
        for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
            target_param.data.mul_(1.0 - float(tau)).add_(param.data, alpha=float(tau))

    if not bool(collect_stats):
        return {}

    with torch.no_grad():
        log_std_param = getattr(policy, "log_std", None)
        log_std_clamped = torch.clamp(log_std_param, float(log_std_min), float(log_std_max))

    return {
        "q_loss": float(q_loss.detach().cpu().item()),
        "inverse_q_loss": float(inverse_q_loss.detach().cpu().item()),
        "q_l2_loss": float(q_l2_loss.detach().cpu().item()),
        "policy_q_reg_loss": float(policy_q_reg_loss.detach().cpu().item()),
        "conservative_q_loss": float(conservative_q_loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "policy_rl_loss": float(policy_rl_loss.detach().cpu().item()),
        "bc_loss": float(bc_loss.detach().cpu().item()),
        "bc_coef": float(bc_weight),
        "expert_q": float(expert_q.detach().mean().cpu().item()),
        "expert_q_abs": float(expert_q.detach().abs().mean().cpu().item()),
        "target_v": float(target.detach().mean().cpu().item()),
        "target_v_unclipped": float(unclipped_target_v.detach().mean().cpu().item()),
        "target_v_abs": float(target.detach().abs().mean().cpu().item()),
        "policy_q": float(policy_q.detach().mean().cpu().item()),
        "policy_q_for_loss": float(policy_q_for_loss.detach().mean().cpu().item()),
        "policy_q_abs": float(policy_q.detach().abs().mean().cpu().item()),
        "random_q": float(random_q.detach().mean().cpu().item()),
        "entropy": float((-log_prob).detach().mean().cpu().item()),
        "log_std_mean": float(log_std_param.detach().mean().cpu().item()),
        "log_std_clamped_mean": float(log_std_clamped.detach().mean().cpu().item()),
        "action_abs": float(policy_action.detach().abs().mean().cpu().item()),
        "action_max_abs": float(policy_action.detach().abs().max().cpu().item()),
        "cuda_peak_mb": _cuda_peak_mb(device),
    }


def evaluate_policy(
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    episodes: int,
    evaluation_executor: ProcessPoolExecutor | None = None,
) -> dict[str, float]:
    requested_workers = max(1, int(getattr(cfg, "evaluation_num_workers", 1)))
    episode_chunks = _chunk_episode_indices(int(episodes), requested_workers)
    if len(episode_chunks) > 1 or evaluation_executor is not None:
        policy_obs_dim = _policy_input_dim(policy)
        policy_state_dict = _cpu_state_dict(policy)
        print(
            f"[eval] parallel evaluation workers={len(episode_chunks)} "
            f"worker_threads={max(1, int(getattr(cfg, 'evaluation_worker_threads', 2)))} "
            f"episodes={max(1, int(episodes))}",
            flush=True,
        )
        executor_context = (
            nullcontext(evaluation_executor)
            if evaluation_executor is not None
            else ProcessPoolExecutor(max_workers=len(episode_chunks), mp_context=mp.get_context("spawn"))
        )
        with executor_context as executor:
            futures = [
                executor.submit(_evaluate_policy_worker, cfg, policy_state_dict, policy_obs_dim, chunk)
                for chunk in episode_chunks
            ]
            parts = [future.result() for future in futures]
        lengths = [int(value) for part in parts for value in part.get("lengths", ())]
        rewards = [float(value) for part in parts for value in part.get("rewards", ())]
        crashes = float(sum(float(part.get("crashes", 0.0)) for part in parts))
        policy_load_seconds = float(sum(float(part.get("policy_load_seconds", 0.0)) for part in parts))
        policy_forward_seconds = float(sum(float(part.get("policy_forward_seconds", 0.0)) for part in parts))
        env_step_seconds = float(sum(float(part.get("env_step_seconds", 0.0)) for part in parts))
        env_reset_seconds = float(sum(float(part.get("env_reset_seconds", 0.0)) for part in parts))
        env_cache_hits = float(sum(float(part.get("env_cache_hits", 0.0)) for part in parts))
        env_cache_misses = float(sum(float(part.get("env_cache_misses", 0.0)) for part in parts))
        print(
            f"[eval] eval_timing "
            f"policy_load={policy_load_seconds:.3f}s "
            f"reset={env_reset_seconds:.3f}s "
            f"policy={policy_forward_seconds:.3f}s "
            f"env_step={env_step_seconds:.3f}s "
            f"cache_hits={env_cache_hits:.0f} "
            f"cache_misses={env_cache_misses:.0f}",
            flush=True,
        )
        return {
            "eval/episodes": float(len(lengths)),
            "eval/mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "eval/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "eval/crashes": float(crashes),
            "eval/policy_load_seconds": policy_load_seconds,
            "eval/policy_forward_seconds": policy_forward_seconds,
            "eval/env_step_seconds": env_step_seconds,
            "eval/env_reset_seconds": env_reset_seconds,
            "eval/env_cache_hits": env_cache_hits,
            "eval/env_cache_misses": env_cache_misses,
        }
    env = None
    lengths: list[int] = []
    rewards: list[float] = []
    crashes = 0
    was_training = policy.training
    policy.eval()
    try:
        with evaluation_thread_context(cfg):
            env = make_training_env(cfg)
            for ep_idx in range(max(1, int(episodes))):
                obs, _info = env.reset(seed=int(cfg.seed) + 100_000 + ep_idx)
                obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
                eval_memory = (
                    policy.initial_memory(len(obs_agents), device=device, dtype=torch.float32)
                    if bool(getattr(policy, "supports_recurrent_memory", False))
                    else None
                )
                total_reward = 0.0
                length = 0
                for _ in range(max(1, int(cfg.max_episode_steps))):
                    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
                        if bool(getattr(policy, "supports_recurrent_memory", False)):
                            mean, _values, eval_memory = policy(
                                obs_tensor,
                                memory=eval_memory,
                                return_memory=True,
                            )
                        else:
                            mean, _values = policy(obs_tensor)
                        action_arr = torch.clamp(mean, -1.0, 1.0).detach().cpu().numpy().astype(np.float32)
                    action = tuple(row.copy() for row in action_arr.reshape(-1, int(cfg.continuous_action_dim)))
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(np.asarray(reward, dtype=np.float32).mean())
                    length += 1
                    if bool(info.get("crashed", False)) if isinstance(info, dict) else False:
                        crashes += 1
                    if terminated or truncated:
                        break
                lengths.append(length)
                rewards.append(total_reward)
    finally:
        if was_training:
            policy.train()
        if env is not None:
            env.close()
    return {
        "eval/episodes": float(len(lengths)),
        "eval/mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "eval/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "eval/crashes": float(crashes),
    }


def parse_args() -> tuple[PSGAILConfig, argparse.Namespace]:
    defaults = PSGAILConfig(action_mode="continuous", run_name="simple_iq_learn")
    parser = argparse.ArgumentParser(description="Lightweight continuous IQ-Learn test trainer for unified NGSIM expert data.")
    for field in fields(PSGAILConfig):
        value = getattr(defaults, field.name)
        arg = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    parser.add_argument("--total-updates", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--iq-alpha", type=float, default=0.05)
    parser.add_argument("--target-tau", type=float, default=0.002)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--bc-coef", type=float, default=0.5)
    parser.add_argument("--bc-warmup-coef", type=float, default=2.0)
    parser.add_argument("--bc-warmup-updates", type=int, default=5_000)
    parser.add_argument("--policy-bc-only-updates", type=int, default=1_000)
    parser.add_argument("--q-l2-coef", type=float, default=1e-3)
    parser.add_argument("--q-policy-reg-coef", type=float, default=1e-3)
    parser.add_argument("--conservative-q-coef", type=float, default=0.05)
    parser.add_argument("--target-value-clip", type=float, default=20.0)
    parser.add_argument("--policy-q-clip", type=float, default=20.0)
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=0.5)
    parser.add_argument("--replay-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--pin-cpu-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abort-on-nonfinite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abort-on-stalled-convergence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--convergence-patience-evals", type=int, default=8)
    parser.add_argument("--min-evals-before-stall", type=int, default=4)
    parser.add_argument("--convergence-min-delta", type=float, default=1e-3)
    parser.add_argument("--convergence-crash-penalty", type=float, default=25.0)
    parser.add_argument("--bc-score-weight", type=float, default=1.0)
    parser.add_argument("--target-eval-mean-length", type=float, default=0.0)
    parser.add_argument("--target-bc-loss", type=float, default=0.0)
    parser.add_argument("--max-eval-crashes", type=float, default=0.0)
    parser.add_argument("--max-q-abs", type=float, default=100.0)
    parser.add_argument(
        "--torch-matmul-precision",
        choices=("default", "highest", "high", "medium"),
        default="high",
    )
    args = parser.parse_args()
    values = vars(args).copy()
    extra_keys = {
        "total_updates",
        "eval_every",
        "replay_size",
        "iq_alpha",
        "target_tau",
        "eval_episodes",
        "bc_coef",
        "bc_warmup_coef",
        "bc_warmup_updates",
        "policy_bc_only_updates",
        "q_l2_coef",
        "q_policy_reg_coef",
        "conservative_q_coef",
        "target_value_clip",
        "policy_q_clip",
        "log_std_min",
        "log_std_max",
        "replay_device",
        "pin_cpu_replay",
        "save_best_checkpoint",
        "abort_on_nonfinite",
        "abort_on_stalled_convergence",
        "convergence_patience_evals",
        "min_evals_before_stall",
        "convergence_min_delta",
        "convergence_crash_penalty",
        "bc_score_weight",
        "target_eval_mean_length",
        "target_bc_loss",
        "max_eval_crashes",
        "max_q_abs",
        "torch_matmul_precision",
    }
    cfg_values = {key: value for key, value in values.items() if key not in extra_keys}
    extras = argparse.Namespace(**{key: values[key] for key in extra_keys})
    return PSGAILConfig(**cfg_values), extras


def main() -> None:
    cfg, extras = parse_args()
    if str(cfg.action_mode).lower() != "continuous":
        raise ValueError("This IQ-Learn test trainer currently supports --action-mode continuous only.")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    _set_matmul_precision(extras.torch_matmul_precision)
    if device.type == "cuda":
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    run_dir = os.path.abspath(os.path.join("logs", "iq_learn", cfg.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    monitor = WandbMonitor(cfg, run_dir)
    monitor.start()

    env = None
    evaluation_executor = None
    try:
        expert = load_expert_transition_data(
            cfg.expert_data,
            max_samples=min(int(cfg.max_expert_samples), int(extras.replay_size)),
            seed=cfg.seed,
            trajectory_frame=cfg.trajectory_frame,
        )
        env_cfg = config_for_round(cfg, 1)
        env = make_training_env(env_cfg)
        cfg.continuous_action_dim = infer_continuous_action_dim(env)
        env_cfg.continuous_action_dim = cfg.continuous_action_dim
        policy_obs_dim = infer_policy_obs_dim(env)
        if policy_obs_dim != expert.policy_observations.shape[1]:
            raise RuntimeError(f"Expert/env observation mismatch: {expert.policy_observations.shape[1]} != {policy_obs_dim}.")
        replay_device = _resolve_replay_device(extras.replay_device, device)
        replay = make_expert_replay_tensors(
            expert,
            replay_device=replay_device,
            pin_cpu_memory=bool(extras.pin_cpu_replay),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        policy = make_actor_critic(
            cfg.policy_model,
            policy_obs_dim,
            cfg.hidden_size,
            action_mode="continuous",
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
        ).to(device)
        q_net = SoftQNetwork(policy_obs_dim, int(cfg.continuous_action_dim), cfg.hidden_size).to(device)
        target_q_net = SoftQNetwork(policy_obs_dim, int(cfg.continuous_action_dim), cfg.hidden_size).to(device)
        target_q_net.load_state_dict(q_net.state_dict())
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        q_optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.disc_learning_rate)
        monitor.watch(policy, q_net)
        evaluation_executor = make_evaluation_executor(cfg)

        print(f"Loaded expert folder: {os.path.abspath(cfg.expert_data)}")
        print(f"expert_obs={expert.policy_observations.shape} expert_actions={expert.actions_continuous_env.shape}")
        print(f"policy_obs_dim={policy_obs_dim} action_dim={cfg.continuous_action_dim} device={device}")
        print(
            f"replay_device={replay.policy_observations.device} "
            f"replay_mb={_replay_size_mb(replay):.1f} "
            f"batch_size={cfg.batch_size} matmul_precision={extras.torch_matmul_precision}"
        )
        print(
            "policy="
            f"{cfg.policy_model} hidden={cfg.hidden_size} "
            f"layers={cfg.transformer_layers} heads={cfg.transformer_heads} "
            f"dropout={cfg.transformer_dropout} temporal={cfg.transformer_temporal_module} "
            f"kernel={cfg.transformer_temporal_kernel_size} temporal_layers={cfg.transformer_temporal_layers}"
        )
        print(
            "evaluation="
            f"workers={cfg.evaluation_num_workers} "
            f"worker_threads={cfg.evaluation_worker_threads} "
            f"validation_all_vehicles={cfg.validation_control_all_vehicles}"
        )

        latest_stats: dict[str, float] = {}
        convergence = ConvergenceTracker()
        best_path = os.path.join(run_dir, "best.pt")
        completed_update = 0
        update_iter = range(1, int(extras.total_updates) + 1)
        progress = (
            tqdm(update_iter, total=int(extras.total_updates), dynamic_ncols=True, desc="IQ-Learn")
            if tqdm is not None
            else update_iter
        )
        for update_idx in progress:
            completed_update = int(update_idx)
            round_cfg = config_for_round(cfg, update_idx)
            round_cfg.continuous_action_dim = cfg.continuous_action_dim
            set_optimizer_lr(policy_optimizer, float(round_cfg.learning_rate))
            set_optimizer_lr(q_optimizer, float(round_cfg.disc_learning_rate))
            validation_due = (
                int(getattr(cfg, "validation_every", 0)) > 0
                and update_idx % int(cfg.validation_every) == 0
            )
            should_log = update_idx == 1 or update_idx % max(1, int(extras.eval_every)) == 0 or validation_due
            update_stats = update_iq_learn(
                policy,
                q_net,
                target_q_net,
                policy_optimizer,
                q_optimizer,
                replay,
                round_cfg,
                device,
                update_idx=update_idx,
                batch_size=int(cfg.batch_size),
                gamma=float(round_cfg.gamma),
                alpha=float(extras.iq_alpha),
                tau=float(extras.target_tau),
                bc_coef=float(extras.bc_coef),
                bc_warmup_coef=float(extras.bc_warmup_coef),
                bc_warmup_updates=int(extras.bc_warmup_updates),
                policy_bc_only_updates=int(extras.policy_bc_only_updates),
                q_l2_coef=float(extras.q_l2_coef),
                q_policy_reg_coef=float(extras.q_policy_reg_coef),
                conservative_q_coef=float(extras.conservative_q_coef),
                target_value_clip=float(extras.target_value_clip),
                policy_q_clip=float(extras.policy_q_clip),
                log_std_min=float(extras.log_std_min),
                log_std_max=float(extras.log_std_max),
                collect_stats=should_log,
            )
            if should_log:
                latest_stats = update_stats
                if bool(extras.abort_on_nonfinite):
                    _require_finite_metrics(latest_stats, update_idx=update_idx)
                max_q_abs = float(extras.max_q_abs)
                q_abs = max(
                    abs(float(latest_stats.get("expert_q", 0.0))),
                    abs(float(latest_stats.get("policy_q", 0.0))),
                    abs(float(latest_stats.get("target_v", 0.0))),
                    abs(float(latest_stats.get("target_v_unclipped", 0.0))),
                )
                if max_q_abs > 0.0 and q_abs > max_q_abs:
                    raise RuntimeError(
                        f"Q scale diverged at update {update_idx}: abs_q={q_abs:.4f} > max_q_abs={max_q_abs:.4f}. "
                        "Lower --learning-rate/--disc-learning-rate or increase Q regularization."
                    )
                metrics = {"update": update_idx, **{f"iq/{k}": v for k, v in latest_stats.items()}}
                eval_stats = evaluate_policy(
                    policy,
                    round_cfg,
                    device,
                    episodes=int(extras.eval_episodes),
                    evaluation_executor=evaluation_executor,
                )
                metrics.update(eval_stats)
                metrics["train/gamma"] = float(round_cfg.gamma)
                metrics["train/policy_learning_rate"] = float(round_cfg.learning_rate)
                metrics["train/q_learning_rate"] = float(round_cfg.disc_learning_rate)
                if validation_due:
                    val_cfg = config_for_round(cfg, update_idx)
                    val_cfg.continuous_action_dim = cfg.continuous_action_dim
                    val_metrics = evaluate_policy_matched_trajectories(
                        policy,
                        val_cfg,
                        device,
                        split=str(getattr(cfg, "validation_prebuilt_split", "val")),
                        episodes=int(getattr(cfg, "validation_episodes", 0)),
                        prefix="validation",
                        evaluation_executor=evaluation_executor,
                    )
                    if val_metrics:
                        metrics.update(val_metrics)
                        monitor.log(val_metrics, step=update_idx)
                        _print_matched_eval("validation", f"{update_idx:06d}", val_metrics)
                score = convergence_score(
                    latest_stats,
                    eval_stats,
                    crash_penalty=float(extras.convergence_crash_penalty),
                    bc_score_weight=float(extras.bc_score_weight),
                )
                convergence.eval_count += 1
                improved = score > convergence.best_score + float(extras.convergence_min_delta)
                if improved:
                    convergence.best_score = float(score)
                    convergence.best_update = int(update_idx)
                    convergence.stale_evals = 0
                    if bool(extras.save_best_checkpoint):
                        torch.save(
                            {
                                "update": update_idx,
                                "policy_state_dict": policy.state_dict(),
                                "q_state_dict": q_net.state_dict(),
                                "target_q_state_dict": target_q_net.state_dict(),
                                "expert_metadata": expert.metadata,
                                "config": vars(cfg),
                                "iq_config": vars(extras),
                                "latest_stats": latest_stats,
                                "eval_stats": eval_stats,
                                "convergence_score": float(score),
                            },
                            best_path,
                        )
                        monitor.save(best_path)
                else:
                    convergence.stale_evals += 1
                metrics.update(
                    {
                        "convergence/score": float(score),
                        "convergence/best_score": float(convergence.best_score),
                        "convergence/best_update": float(convergence.best_update),
                        "convergence/stale_evals": float(convergence.stale_evals),
                    }
                )
                monitor.log(metrics, step=update_idx)
                if tqdm is not None:
                    progress.set_postfix(
                        q=f"{latest_stats['q_loss']:.3f}",
                        bc=f"{latest_stats['bc_loss']:.4f}",
                        eval_len=f"{eval_stats['eval/mean_length']:.1f}",
                        score=f"{score:.2f}",
                    )
                print(
                    f"[update {update_idx:06d}] q_loss={latest_stats['q_loss']:.4f} "
                    f"policy_loss={latest_stats['policy_loss']:.4f} "
                    f"bc_loss={latest_stats['bc_loss']:.4f} "
                    f"expert_q={latest_stats['expert_q']:.4f} "
                    f"policy_q={latest_stats['policy_q']:.4f} "
                    f"bc_coef={latest_stats['bc_coef']:.3f} "
                    f"cuda_peak_mb={latest_stats['cuda_peak_mb']:.1f} "
                    f"eval_reward={eval_stats['eval/mean_reward']:.4f} "
                    f"eval_len={eval_stats['eval/mean_length']:.1f} "
                    f"score={score:.4f} best={convergence.best_score:.4f}@{convergence.best_update}"
                )
                if convergence_reached(
                    latest_stats,
                    eval_stats,
                    target_eval_mean_length=float(extras.target_eval_mean_length),
                    target_bc_loss=float(extras.target_bc_loss),
                    max_eval_crashes=float(extras.max_eval_crashes),
                ):
                    print(
                        "Convergence target reached: "
                        f"eval_len={eval_stats['eval/mean_length']:.1f}, "
                        f"bc_loss={latest_stats['bc_loss']:.6f}, "
                        f"crashes={eval_stats['eval/crashes']:.0f}."
                    )
                    break
                if (
                    bool(extras.abort_on_stalled_convergence)
                    and convergence.eval_count >= int(extras.min_evals_before_stall)
                    and convergence.stale_evals >= int(extras.convergence_patience_evals)
                ):
                    raise RuntimeError(
                        "Convergence appears stalled: "
                        f"best_score={convergence.best_score:.4f} at update {convergence.best_update}, "
                        f"no improvement for {convergence.stale_evals} evals."
                    )
            if cfg.checkpoint_every > 0 and update_idx % int(cfg.checkpoint_every) == 0:
                checkpoint_path = os.path.join(ckpt_dir, f"update_{update_idx:06d}.pt")
                torch.save(
                    {
                        "update": update_idx,
                        "policy_state_dict": policy.state_dict(),
                        "q_state_dict": q_net.state_dict(),
                        "target_q_state_dict": target_q_net.state_dict(),
                        "expert_metadata": expert.metadata,
                        "config": vars(cfg),
                        "iq_config": vars(extras),
                    },
                    checkpoint_path,
                )
                monitor.save(checkpoint_path)
            if should_save_checkpoint_video(cfg, update_idx):
                video_path = save_checkpoint_video(policy, round_cfg, run_dir=run_dir, round_idx=update_idx, device=device)
                if video_path is not None:
                    monitor.log_video("checkpoint/policy_video", video_path, step=update_idx, fps=int(round_cfg.policy_frequency))

        final_path = os.path.join(run_dir, "final.pt")
        torch.save(
            {
                "update": int(completed_update),
                "policy_state_dict": policy.state_dict(),
                "q_state_dict": q_net.state_dict(),
                "target_q_state_dict": target_q_net.state_dict(),
                "expert_metadata": expert.metadata,
                "config": vars(cfg),
                "iq_config": vars(extras),
                "latest_stats": latest_stats,
                "best_update": int(convergence.best_update),
                "best_score": float(convergence.best_score),
            },
            final_path,
        )
        monitor.save(final_path)
        final_cfg = config_for_round(cfg, int(completed_update))
        final_cfg.continuous_action_dim = cfg.continuous_action_dim
        final_video_path = save_checkpoint_video(policy, final_cfg, run_dir=run_dir, round_idx=int(completed_update), device=device)
        if final_video_path is not None:
            monitor.log_video("checkpoint/final_policy_video", final_video_path, step=int(completed_update), fps=int(final_cfg.policy_frequency))
        if int(getattr(cfg, "test_episodes", 0)) > 0:
            test_metrics = evaluate_policy_matched_trajectories(
                policy,
                final_cfg,
                device,
                split=str(getattr(cfg, "test_prebuilt_split", "test")),
                episodes=int(getattr(cfg, "test_episodes", 0)),
                prefix="test",
                evaluation_executor=evaluation_executor,
            )
            if test_metrics:
                monitor.log(test_metrics, step=int(completed_update))
                _print_matched_eval("test", "final", test_metrics)
    finally:
        if evaluation_executor is not None:
            evaluation_executor.shutdown(wait=True, cancel_futures=True)
        if env is not None:
            env.close()
        monitor.finish()
    print(f"Saved final checkpoint under: {run_dir}")


if __name__ == "__main__":
    main()
