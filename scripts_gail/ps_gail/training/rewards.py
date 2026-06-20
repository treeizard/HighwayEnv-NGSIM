"""Reward shaping and return computation helpers for adversarial imitation training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PSGAILConfig
from ..data import standardize_features

from .policy import _is_continuous
from .torch_utils import _as_device_tensor

def compute_returns_and_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    trajectory_ids: np.ndarray,
    cfg: PSGAILConfig,
) -> tuple[np.ndarray, np.ndarray]:
    returns = np.zeros_like(rewards, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    for trajectory_id in np.unique(trajectory_ids):
        indices = np.where(trajectory_ids == trajectory_id)[0]
        next_advantage = 0.0
        next_value = 0.0
        for idx in reversed(indices):
            nonterminal = 0.0 if dones[idx] else 1.0
            delta = rewards[idx] + cfg.gamma * next_value * nonterminal - values[idx]
            next_advantage = delta + cfg.gamma * cfg.gae_lambda * nonterminal * next_advantage
            advantages[idx] = next_advantage
            returns[idx] = advantages[idx] + values[idx]
            next_value = values[idx]
    if advantages.size > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages

def discriminator_reward(
    discriminator: nn.Module,
    generator_features: np.ndarray,
    device: torch.device,
    *,
    feature_normalizer: tuple[np.ndarray, np.ndarray] | None = None,
    feature_clip: float = 0.0,
    loss_type: str = "bce",
) -> np.ndarray:
    features = generator_features
    if feature_normalizer is not None:
        mean, std = feature_normalizer
        features = standardize_features(features, mean, std, clip=feature_clip)
    was_training = discriminator.training
    discriminator.eval()
    try:
        with torch.no_grad():
            logits = discriminator(_as_device_tensor(features, dtype=torch.float32, device=device))
            rewards = logits if str(loss_type).lower() == "wgan_gp" else F.softplus(logits)
    finally:
        if was_training:
            discriminator.train()
    return rewards.cpu().numpy().astype(np.float32)

def discriminator_input_mode(cfg: PSGAILConfig) -> str:
    mode = str(getattr(cfg, "discriminator_input", "auto")).lower()
    if mode == "auto":
        return "action" if _is_continuous(cfg) else "trajectory"
    if mode not in {"trajectory", "action"}:
        raise ValueError(
            f"Unsupported discriminator_input={mode!r}. Expected 'auto', 'trajectory', or 'action'."
        )
    if mode == "action" and not _is_continuous(cfg):
        raise ValueError("discriminator_input='action' currently requires continuous action mode.")
    return mode

def _transition_array(values: np.ndarray, n: int, *, dtype: np.dtype, fill: float = 0.0) -> np.ndarray:
    n = max(0, int(n))
    arr = np.asarray(values, dtype=dtype)
    if arr.shape == (n,):
        return arr.astype(dtype, copy=False)
    return np.full(n, fill, dtype=dtype)

def _metric_float(metric: dict[str, object], key: str, default: float) -> float:
    try:
        value = float(metric.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)

def player_challenge_pressure_from_metric(
    metric: dict[str, object] | None,
    cfg: PSGAILConfig,
) -> tuple[float, float, float]:
    if metric is None:
        return 0.0, 0.0, 0.0
    ttc_target = _metric_float(metric, "ttc_target", 0.0)
    ttc_floor = _metric_float(metric, "ttc_floor", 0.0)
    gap_target = _metric_float(metric, "gap_target", 0.0)
    gap_floor = _metric_float(metric, "gap_floor", 0.0)
    min_ttc = _metric_float(metric, "min_ttc", float("inf"))
    min_gap = _metric_float(metric, "min_gap", float("inf"))
    if bool(metric.get("crashed", False)) or bool(metric.get("offroad", False)):
        return 0.0, float(ttc_target), float(gap_target)
    if min_ttc < ttc_floor or min_gap < gap_floor:
        return 0.0, float(ttc_target), float(gap_target)

    if np.isfinite(min_ttc) and ttc_target > ttc_floor:
        ttc_pressure = np.clip((ttc_target - min_ttc) / (ttc_target - ttc_floor), 0.0, 1.0)
    else:
        ttc_pressure = 0.0
    if np.isfinite(min_gap) and gap_target > gap_floor:
        gap_pressure = np.clip((gap_target - min_gap) / (gap_target - gap_floor), 0.0, 1.0)
    else:
        gap_pressure = 0.0

    ttc_weight = max(0.0, float(getattr(cfg, "challenge_ttc_weight", 0.6)))
    gap_weight = max(0.0, float(getattr(cfg, "challenge_gap_weight", 0.4)))
    weight_sum = ttc_weight + gap_weight
    if weight_sum <= 0.0:
        return 0.0, float(ttc_target), float(gap_target)
    pressure = (ttc_weight * float(ttc_pressure) + gap_weight * float(gap_pressure)) / weight_sum
    return float(np.clip(pressure, 0.0, 1.0)), float(ttc_target), float(gap_target)

def player_challenge_payoff(
    pressure: float,
    *,
    crash_rate_ema: float,
    offroad_rate_ema: float,
    cfg: PSGAILConfig,
) -> float:
    if pressure <= 0.0:
        return 0.0
    risk = (
        1.0
        + max(0.0, float(getattr(cfg, "challenge_crash_weight", 4.0))) * max(0.0, float(crash_rate_ema))
        + max(0.0, float(getattr(cfg, "challenge_offroad_weight", 2.0))) * max(0.0, float(offroad_rate_ema))
    )
    return float(max(0.0, pressure) / max(1.0e-6, risk))

def player_challenge_bonus(
    payoffs: np.ndarray,
    primary_rewards: np.ndarray,
    cfg: PSGAILConfig,
) -> np.ndarray:
    primary = np.asarray(primary_rewards, dtype=np.float32)
    if not bool(getattr(cfg, "enable_player_challenge_reward", False)) or primary.size == 0:
        return np.zeros_like(primary, dtype=np.float32)
    payoff_arr = _transition_array(payoffs, len(primary), dtype=np.float32, fill=0.0)
    payoff_arr = np.where(np.isfinite(payoff_arr), np.maximum(payoff_arr, 0.0), 0.0)

    quantile = float(getattr(cfg, "challenge_expert_like_quantile", 0.25))
    if 0.0 < quantile < 1.0 and primary.size > 1:
        threshold = float(np.quantile(primary, quantile))
        payoff_arr = np.where(primary >= threshold, payoff_arr, 0.0)

    bonus = float(getattr(cfg, "challenge_reward_coef", 0.2)) * payoff_arr
    absolute_clip = float(getattr(cfg, "challenge_reward_clip", 0.25))
    if absolute_clip > 0.0:
        bonus = np.clip(bonus, 0.0, absolute_clip)

    fraction = max(0.0, float(getattr(cfg, "challenge_max_primary_reward_fraction", 0.10)))
    if fraction <= 0.0:
        return np.zeros_like(primary, dtype=np.float32)
    mean_abs_primary = float(np.mean(np.abs(primary))) if primary.size else 0.0
    if mean_abs_primary <= 1.0e-8:
        return np.zeros_like(primary, dtype=np.float32)
    cap_scale = np.maximum(np.abs(primary), mean_abs_primary)
    bonus = np.minimum(bonus, fraction * cap_scale)
    return bonus.astype(np.float32, copy=False)

def combine_primary_env_challenge_rewards(
    primary_rewards: np.ndarray,
    env_penalties: np.ndarray,
    cfg: PSGAILConfig,
    *,
    challenge_payoffs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    primary = np.asarray(primary_rewards, dtype=np.float32)
    penalties = np.asarray(env_penalties, dtype=np.float32)
    if primary.shape != penalties.shape:
        raise ValueError(f"Reward/penalty shape mismatch: {primary.shape} != {penalties.shape}")
    bonus = player_challenge_bonus(
        np.zeros_like(primary, dtype=np.float32) if challenge_payoffs is None else challenge_payoffs,
        primary,
        cfg,
    )
    rewards = primary + penalties + bonus
    if float(cfg.final_reward_clip) > 0:
        clip = float(cfg.final_reward_clip)
        rewards = np.clip(rewards, -clip, clip)
    return rewards.astype(np.float32, copy=False), bonus.astype(np.float32, copy=False)

def sequence_rewards_to_transition_rewards(
    sequence_rewards: np.ndarray,
    *,
    num_transitions: int,
    sequence_last_indices: np.ndarray,
    sequence_transition_indices: np.ndarray | None = None,
    assignment: str = "last",
) -> np.ndarray:
    sequence_rewards = np.asarray(sequence_rewards, dtype=np.float32).reshape(-1)
    num_transitions = max(0, int(num_transitions))
    per_transition = np.zeros(num_transitions, dtype=np.float32)
    if sequence_rewards.size == 0 or num_transitions == 0:
        return per_transition

    mode = str(assignment).lower()
    if mode in {"last", "last_step", "terminal"}:
        last_indices = np.asarray(sequence_last_indices, dtype=np.int64).reshape(-1)
        if len(last_indices) != len(sequence_rewards):
            raise ValueError(
                "Sequence reward/last-index count mismatch: "
                f"{len(sequence_rewards)} != {len(last_indices)}."
            )
        counts = np.zeros(num_transitions, dtype=np.float32)
        for reward, last_idx in zip(sequence_rewards, last_indices):
            last_idx = int(last_idx)
            if 0 <= last_idx < num_transitions:
                per_transition[last_idx] += float(reward)
                counts[last_idx] += 1.0
        mask = counts > 0
        per_transition[mask] /= counts[mask]
        return per_transition

    if sequence_transition_indices is None:
        raise ValueError(
            f"sequence_reward_assignment={mode!r} requires sequence_transition_indices."
        )
    transition_indices = np.asarray(sequence_transition_indices, dtype=np.int64)
    if transition_indices.ndim != 2 or transition_indices.shape[0] != len(sequence_rewards):
        raise ValueError(
            "Sequence reward/window-index shape mismatch: "
            f"rewards={sequence_rewards.shape} window_indices={transition_indices.shape}."
        )

    if mode in {"mean", "average", "dense_mean"}:
        counts = np.zeros(num_transitions, dtype=np.float32)
        for reward, window_indices in zip(sequence_rewards, transition_indices):
            valid = window_indices[(0 <= window_indices) & (window_indices < num_transitions)]
            if valid.size:
                per_transition[valid] += float(reward)
                counts[valid] += 1.0
        mask = counts > 0
        per_transition[mask] /= counts[mask]
        return per_transition

    if mode in {"sum", "dense_sum"}:
        for reward, window_indices in zip(sequence_rewards, transition_indices):
            valid = window_indices[(0 <= window_indices) & (window_indices < num_transitions)]
            if valid.size:
                per_transition[valid] += float(reward)
        return per_transition

    raise ValueError(
        f"Unsupported sequence_reward_assignment={mode!r}. Expected 'last', 'mean', or 'sum'."
    )

def sequence_window_counts_to_transition_counts(
    *,
    num_transitions: int,
    sequence_last_indices: np.ndarray,
    sequence_transition_indices: np.ndarray | None = None,
    assignment: str = "last",
) -> np.ndarray:
    num_transitions = max(0, int(num_transitions))
    counts = np.zeros(num_transitions, dtype=np.float32)
    if num_transitions == 0:
        return counts
    mode = str(assignment).lower()
    if mode in {"last", "last_step", "terminal"}:
        for last_idx in np.asarray(sequence_last_indices, dtype=np.int64).reshape(-1):
            last_idx = int(last_idx)
            if 0 <= last_idx < num_transitions:
                counts[last_idx] += 1.0
        return counts
    if sequence_transition_indices is None:
        return counts
    if mode in {"mean", "average", "dense_mean", "sum", "dense_sum"}:
        for window_indices in np.asarray(sequence_transition_indices, dtype=np.int64):
            valid = window_indices[(0 <= window_indices) & (window_indices < num_transitions)]
            if valid.size:
                counts[valid] += 1.0
        return counts
    return counts.astype(np.float32, copy=False)

def action_conditioned_features(policy_observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    policy_observations = np.asarray(policy_observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if policy_observations.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"Action-conditioned features require rank-2 obs/actions, got {policy_observations.shape} and {actions.shape}."
        )
    if len(policy_observations) != len(actions):
        raise ValueError(f"Observation/action count mismatch: {len(policy_observations)} != {len(actions)}.")
    return np.concatenate([policy_observations, actions], axis=1).astype(np.float32, copy=False)

def should_normalize_gail_reward(cfg: PSGAILConfig) -> bool:
    if not bool(getattr(cfg, "normalize_gail_reward", False)):
        return False
    if str(getattr(cfg, "discriminator_loss", "bce")).lower() == "wgan_gp":
        return bool(getattr(cfg, "allow_wgan_reward_normalization", False))
    return True

def should_apply_gail_reward_clip(cfg: PSGAILConfig) -> bool:
    if float(getattr(cfg, "gail_reward_clip", 0.0)) <= 0.0:
        return False
    return str(getattr(cfg, "discriminator_loss", "bce")).lower() != "wgan_gp"

def safe_normalize_adversarial_rewards(rewards: np.ndarray, cfg: PSGAILConfig) -> np.ndarray:
    shaped = np.asarray(rewards, dtype=np.float32)
    if shaped.size <= 1:
        return shaped.astype(np.float32, copy=True)
    centered = shaped - float(shaped.mean())
    std = float(shaped.std())
    min_std = max(1.0e-8, float(getattr(cfg, "wgan_reward_norm_min_std", 1.0e-3)))
    normalized = centered / max(std, min_std)
    clip = float(getattr(cfg, "wgan_reward_norm_clip", 0.0))
    if clip > 0.0:
        normalized = np.clip(normalized, -clip, clip)
    return normalized.astype(np.float32, copy=False)

def shape_adversarial_rewards(raw_gail_rewards: np.ndarray, cfg: PSGAILConfig) -> np.ndarray:
    raw = np.asarray(raw_gail_rewards, dtype=np.float32)
    shaped_gail = raw.astype(np.float32, copy=True)
    if str(getattr(cfg, "discriminator_loss", "bce")).lower() == "wgan_gp":
        if bool(getattr(cfg, "wgan_reward_center", False)) and shaped_gail.size > 1:
            shaped_gail = shaped_gail - shaped_gail.mean()
        reward_scale = float(getattr(cfg, "wgan_reward_scale", 1.0))
        if reward_scale != 1.0:
            shaped_gail = shaped_gail * reward_scale
        wgan_clip = float(getattr(cfg, "wgan_reward_clip", 0.0))
        if wgan_clip > 0:
            shaped_gail = np.clip(shaped_gail, -wgan_clip, wgan_clip)
    if should_normalize_gail_reward(cfg) and shaped_gail.size > 1:
        shaped_gail = safe_normalize_adversarial_rewards(shaped_gail, cfg)
    if should_apply_gail_reward_clip(cfg):
        clip = float(cfg.gail_reward_clip)
        shaped_gail = np.clip(shaped_gail, -clip, clip)
    return shaped_gail.astype(np.float32, copy=False)

def shape_rollout_rewards(
    raw_gail_rewards: np.ndarray,
    env_penalties: np.ndarray,
    cfg: PSGAILConfig,
) -> tuple[np.ndarray, np.ndarray]:
    shaped_gail = shape_adversarial_rewards(raw_gail_rewards, cfg)
    rewards, _challenge_bonus = combine_primary_env_challenge_rewards(
        shaped_gail,
        env_penalties,
        cfg,
        challenge_payoffs=None,
    )
    return rewards.astype(np.float32), shaped_gail.astype(np.float32)

__all__ = [
    'compute_returns_and_advantages',
    'discriminator_reward',
    'discriminator_input_mode',
    '_transition_array',
    '_metric_float',
    'player_challenge_pressure_from_metric',
    'player_challenge_payoff',
    'player_challenge_bonus',
    'combine_primary_env_challenge_rewards',
    'sequence_rewards_to_transition_rewards',
    'sequence_window_counts_to_transition_counts',
    'action_conditioned_features',
    'should_normalize_gail_reward',
    'should_apply_gail_reward_clip',
    'safe_normalize_adversarial_rewards',
    'shape_adversarial_rewards',
    'shape_rollout_rewards'
]
