"""PPO policy update helpers for PS-GAIL policies."""

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
    _is_continuous,
    _recurrent_memory_cache_from_prior_steps,
    _shift_recurrent_memory,
    policy_distribution_and_values,
    policy_distribution_values_memory,
    recurrent_memory_stats,
    recurrent_policy_enabled,
)
from .types import RolloutBatch

def _recurrent_rollout_chunks(
    rollout: RolloutBatch,
    *,
    sequence_length: int,
) -> list[tuple[np.ndarray, int, int]]:
    chunks: list[tuple[np.ndarray, int, int]] = []
    seq_len = max(1, int(sequence_length))
    for trajectory_id in np.unique(rollout.trajectory_ids):
        indices = np.nonzero(rollout.trajectory_ids == int(trajectory_id))[0].astype(np.int64)
        if indices.size == 0:
            continue
        for start in range(0, int(indices.size), seq_len):
            end = min(int(indices.size), start + seq_len)
            if end > start:
                chunks.append((indices, start, end))
    return chunks

def _update_recurrent_policy(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    expert_policy_observations: np.ndarray | None = None,
    expert_actions: np.ndarray | None = None,
) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    cpu_device = torch.device("cpu")
    action_tensor = (
        torch.as_tensor(rollout.actions, dtype=torch.float32, device=cpu_device)
        if _is_continuous(cfg)
        else torch.as_tensor(rollout.actions, dtype=torch.long, device=cpu_device)
    )
    obs_tensor = torch.as_tensor(rollout.policy_observations, dtype=torch.float32, device=cpu_device)
    critic_obs_tensor = torch.as_tensor(rollout.critic_observations, dtype=torch.float32, device=cpu_device)
    action_mask_tensor = (
        torch.as_tensor(rollout.action_masks, dtype=torch.bool, device=cpu_device)
        if not _is_continuous(cfg)
        and bool(getattr(cfg, "enable_action_masking", True))
        and rollout.action_masks.size
        else None
    )
    old_log_probs_tensor = torch.as_tensor(rollout.old_log_probs, dtype=torch.float32, device=cpu_device)
    returns_tensor = torch.as_tensor(rollout.returns, dtype=torch.float32, device=cpu_device)
    advantages_tensor = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=cpu_device)
    log_std = getattr(policy, "log_std", None)
    initial_log_std_mean = float(log_std.detach().mean().cpu().item()) if log_std is not None else float("nan")
    initial_action_std_mean = float(torch.exp(log_std.detach()).mean().cpu().item()) if log_std is not None else float("nan")
    bc_coef = max(0.0, float(getattr(cfg, "policy_bc_regularization_coef", 0.0)))
    if bc_coef > 0.0:
        if not _is_continuous(cfg):
            raise ValueError("policy_bc_regularization_coef currently requires continuous action mode.")
        if expert_policy_observations is None or expert_actions is None:
            raise ValueError("policy_bc_regularization_coef requires expert policy observations and actions.")
        expert_obs_tensor = torch.as_tensor(
            np.asarray(expert_policy_observations, dtype=np.float32),
            dtype=torch.float32,
            device=cpu_device,
        )
        expert_action_tensor = torch.as_tensor(
            np.clip(np.asarray(expert_actions, dtype=np.float32), -1.0, 1.0),
            dtype=torch.float32,
            device=cpu_device,
        )
    else:
        expert_obs_tensor = None
        expert_action_tensor = None

    chunks = _recurrent_rollout_chunks(
        rollout,
        sequence_length=int(getattr(cfg, "transformer_recurrent_sequence_length", 32)),
    )
    if not chunks:
        raise RuntimeError("Recurrent PPO received no trajectory chunks.")

    seqs_per_batch = max(1, int(getattr(cfg, "transformer_recurrent_sequences_per_batch", 32)))
    micro_sequences = max(
        1,
        min(seqs_per_batch, int(getattr(cfg, "transformer_recurrent_micro_batch_sequences", 8))),
    )
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + int(rollout.num_agent_steps))
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    ratio_means: list[float] = []
    ratio_stds: list[float] = []
    bc_losses: list[float] = []
    post_update_approx_kl = float("nan")
    post_update_ratio_mean = float("nan")
    post_update_ratio_std = float("nan")

    def cpu_to_device(tensor: torch.Tensor) -> torch.Tensor:
        if device.type == "cuda":
            return tensor.pin_memory().to(device=device, non_blocking=True)
        return tensor.to(device=device)

    def make_micro_indices(micro_chunks: list[tuple[np.ndarray, int, int]]) -> tuple[np.ndarray, np.ndarray]:
        max_len = max(end - start for _indices, start, end in micro_chunks)
        indices = np.full((len(micro_chunks), max_len), -1, dtype=np.int64)
        valid = np.zeros((len(micro_chunks), max_len), dtype=bool)
        for row, (trajectory_indices, start, end) in enumerate(micro_chunks):
            values = trajectory_indices[start:end]
            indices[row, : len(values)] = values
            valid[row, : len(values)] = True
        return indices, valid

    def make_initial_memory(micro_chunks: list[tuple[np.ndarray, int, int]]) -> torch.Tensor:
        memories = [
            _recurrent_memory_cache_from_prior_steps(
                policy,
                cfg,
                rollout,
                trajectory_indices,
                start,
                device,
            )
            for trajectory_indices, start, _end in micro_chunks
        ]
        return torch.cat([memory for memory in memories if memory is not None], dim=0)

    def recurrent_micro_forward(
        micro_chunks: list[tuple[np.ndarray, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index_matrix, valid_matrix = make_micro_indices(micro_chunks)
        batch_count, max_len = index_matrix.shape
        memory = make_initial_memory(micro_chunks)
        policy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        approx_kl_terms: list[torch.Tensor] = []
        ratios: list[torch.Tensor] = []
        valid_count = 0
        for step in range(max_len):
            step_indices = index_matrix[:, step]
            active_np = valid_matrix[:, step]
            active = torch.as_tensor(active_np, dtype=torch.bool, device=device)
            if not bool(active.any()):
                continue
            safe_indices = np.where(active_np, step_indices, step_indices[active_np][0])
            safe_index_tensor = torch.as_tensor(safe_indices, dtype=torch.long, device=cpu_device)
            obs = cpu_to_device(obs_tensor[safe_index_tensor])
            critic_obs = cpu_to_device(critic_obs_tensor[safe_index_tensor])
            actions = cpu_to_device(action_tensor[safe_index_tensor])
            old_log_probs = cpu_to_device(old_log_probs_tensor[safe_index_tensor])
            returns = cpu_to_device(returns_tensor[safe_index_tensor])
            advantages = cpu_to_device(advantages_tensor[safe_index_tensor])
            masks = (
                cpu_to_device(action_mask_tensor[safe_index_tensor])
                if action_mask_tensor is not None
                else None
            )
            dist, values, step_memory = policy_distribution_values_memory(
                policy,
                obs,
                cfg,
                masks,
                critic_obs_tensor=critic_obs,
                memory=memory,
                return_memory=True,
            )
            if step_memory is None:
                raise RuntimeError("Recurrent policy did not return step memory.")
            log_probs = dist.log_prob(actions)
            active_log_probs = log_probs[active]
            active_old_log_probs = old_log_probs[active]
            active_advantages = advantages[active]
            active_returns = returns[active]
            active_values = values[active]
            log_ratio = active_log_probs - active_old_log_probs
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
            policy_terms.append(-torch.min(ratio * active_advantages, clipped_ratio * active_advantages))
            value_terms.append(torch.square(active_values - active_returns))
            entropy_terms.append(dist.entropy()[active])
            approx_kl_terms.append((ratio - 1.0) - log_ratio)
            ratios.append(ratio)
            valid_count += int(active_np.sum())
            shifted = _shift_recurrent_memory(memory, step_memory)
            memory = torch.where(active[:, None, None, None], shifted, memory)
        if valid_count <= 0:
            raise RuntimeError("Recurrent PPO micro-batch had no valid transitions.")
        policy_values = torch.cat(policy_terms, dim=0)
        value_values = torch.cat(value_terms, dim=0)
        entropy_values = torch.cat(entropy_terms, dim=0)
        approx_kl_values = torch.cat(approx_kl_terms, dim=0)
        ratio_values = torch.cat(ratios, dim=0)
        return (
            policy_values.mean(),
            value_values.mean(),
            entropy_values.mean(),
            approx_kl_values.mean(),
            ratio_values,
            valid_count,
        )

    try:
        for _epoch in range(int(cfg.ppo_epochs)):
            order = rng.permutation(len(chunks))
            shuffled = [chunks[int(index)] for index in order]
            for start in range(0, len(shuffled), seqs_per_batch):
                batch_chunks = shuffled[start : start + seqs_per_batch]
                batch_transition_count = sum(end - chunk_start for _indices, chunk_start, end in batch_chunks)
                optimizer.zero_grad(set_to_none=True)
                weighted_policy_loss = 0.0
                weighted_value_loss = 0.0
                weighted_entropy = 0.0
                weighted_approx_kl = 0.0
                weighted_clip_fraction = 0.0
                weighted_ratio_mean = 0.0
                weighted_ratio_std = 0.0
                for micro_start in range(0, len(batch_chunks), micro_sequences):
                    micro_chunks = batch_chunks[micro_start : micro_start + micro_sequences]
                    (
                        policy_loss,
                        value_loss,
                        entropy,
                        approx_kl,
                        ratio,
                        valid_count,
                    ) = recurrent_micro_forward(micro_chunks)
                    micro_weight = float(valid_count) / float(max(1, batch_transition_count))
                    # Calculate for Action Critic Loss (With Entropy)
                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                    bc_loss = None
                    if bc_coef > 0.0 and expert_obs_tensor is not None and expert_action_tensor is not None:
                        expert_idx = torch.randint(
                            0,
                            int(expert_obs_tensor.shape[0]),
                            (valid_count,),
                            device=cpu_device,
                        )
                        expert_obs = cpu_to_device(expert_obs_tensor[expert_idx])
                        expert_actions_for_bc = cpu_to_device(expert_action_tensor[expert_idx])
                        expert_memory = policy.initial_memory(
                            int(expert_obs.shape[0]),
                            device=device,
                            dtype=expert_obs.dtype,
                        )
                        pred_expert_actions, _values, _memory = policy(
                            expert_obs,
                            memory=expert_memory,
                            return_memory=True,
                        )
                        bc_loss = F.mse_loss(pred_expert_actions, expert_actions_for_bc)
                        loss = loss + bc_coef * bc_loss
                    (loss * micro_weight).backward()
                    with torch.no_grad():
                        clip_fraction = (
                            (torch.abs(ratio - 1.0) > float(cfg.clip_range)).float().mean()
                        )
                        weighted_policy_loss += micro_weight * float(policy_loss.detach().cpu().item())
                        weighted_value_loss += micro_weight * float(value_loss.detach().cpu().item())
                        weighted_entropy += micro_weight * float(entropy.detach().cpu().item())
                        weighted_approx_kl += micro_weight * float(approx_kl.detach().cpu().item())
                        weighted_clip_fraction += micro_weight * float(clip_fraction.detach().cpu().item())
                        weighted_ratio_mean += micro_weight * float(ratio.mean().detach().cpu().item())
                        weighted_ratio_std += micro_weight * float(
                            ratio.std(unbiased=False).detach().cpu().item()
                        )
                        if bc_loss is not None:
                            bc_losses.append(float(bc_loss.detach().cpu().item()))
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
                policy_losses.append(weighted_policy_loss)
                value_losses.append(weighted_value_loss)
                entropies.append(weighted_entropy)
                approx_kls.append(weighted_approx_kl)
                clip_fractions.append(weighted_clip_fraction)
                ratio_means.append(weighted_ratio_mean)
                ratio_stds.append(weighted_ratio_std)

        diagnostic_chunks = chunks[: min(len(chunks), max(1, seqs_per_batch))]
        if diagnostic_chunks:
            diagnostic_transition_count = sum(end - start for _indices, start, end in diagnostic_chunks)
            weighted_post_kl = 0.0
            weighted_post_ratio_mean = 0.0
            weighted_post_ratio_std = 0.0
            with torch.no_grad():
                for micro_start in range(0, len(diagnostic_chunks), micro_sequences):
                    micro_chunks = diagnostic_chunks[micro_start : micro_start + micro_sequences]
                    _pl, _vl, _ent, approx_kl, ratio, valid_count = recurrent_micro_forward(micro_chunks)
                    micro_weight = float(valid_count) / float(max(1, diagnostic_transition_count))
                    weighted_post_kl += micro_weight * float(approx_kl.detach().cpu().item())
                    weighted_post_ratio_mean += micro_weight * float(ratio.mean().detach().cpu().item())
                    weighted_post_ratio_std += micro_weight * float(
                        ratio.std(unbiased=False).detach().cpu().item()
                    )
            post_update_approx_kl = weighted_post_kl
            post_update_ratio_mean = weighted_post_ratio_mean
            post_update_ratio_std = weighted_post_ratio_std
    finally:
        if was_training:
            policy.train()
    final_log_std_mean = float(log_std.detach().mean().cpu().item()) if log_std is not None else float("nan")
    final_action_std_mean = float(torch.exp(log_std.detach()).mean().cpu().item()) if log_std is not None else float("nan")

    stats = {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
        "post_update_approx_kl": float(post_update_approx_kl),
        "clip_fraction": float(np.mean(clip_fractions)),
        "ratio_mean": float(np.mean(ratio_means)),
        "ratio_std": float(np.mean(ratio_stds)),
        "post_update_ratio_mean": float(post_update_ratio_mean),
        "post_update_ratio_std": float(post_update_ratio_std),
        "advantage_mean": float(np.mean(rollout.advantages)) if rollout.advantages.size else 0.0,
        "advantage_std": float(np.std(rollout.advantages)) if rollout.advantages.size else 0.0,
        "bc_regularization_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
        "bc_regularization_coef": float(bc_coef),
        "ppo_micro_batch_size": float(micro_sequences),
        "log_std_mean": final_log_std_mean,
        "action_std_param_mean": final_action_std_mean,
        "log_std_delta": final_log_std_mean - initial_log_std_mean,
        "action_std_param_delta": final_action_std_mean - initial_action_std_mean,
        "transformer_recurrent_chunks": float(len(chunks)),
        "transformer_recurrent_sequence_length": float(
            int(getattr(cfg, "transformer_recurrent_sequence_length", 32))
        ),
    }
    stats.update(recurrent_memory_stats(rollout))
    if device.type == "cuda":
        stats["perf/cuda_max_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return stats

def update_policy(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    expert_policy_observations: np.ndarray | None = None,
    expert_actions: np.ndarray | None = None,
) -> dict[str, float]:
    if recurrent_policy_enabled(policy):
        return _update_recurrent_policy(
            policy,
            optimizer,
            rollout,
            cfg,
            device,
            expert_policy_observations=expert_policy_observations,
            expert_actions=expert_actions,
        )
    was_training = policy.training
    # Rollout log-probabilities are collected with policy.eval(). Keeping PPO
    # forwards in eval mode prevents dropout from corrupting the likelihood
    # ratio, while gradients still flow through all parameters.
    policy.eval()
    cpu_device = torch.device("cpu")
    action_tensor = (
        torch.as_tensor(rollout.actions, dtype=torch.float32, device=cpu_device)
        if _is_continuous(cfg)
        else torch.as_tensor(rollout.actions, dtype=torch.long, device=cpu_device)
    )
    obs_tensor = torch.as_tensor(rollout.policy_observations, dtype=torch.float32, device=cpu_device)
    critic_obs_tensor = torch.as_tensor(rollout.critic_observations, dtype=torch.float32, device=cpu_device)
    action_mask_tensor = (
        torch.as_tensor(rollout.action_masks, dtype=torch.bool, device=cpu_device)
        if not _is_continuous(cfg)
        and bool(getattr(cfg, "enable_action_masking", True))
        and rollout.action_masks.size
        else None
    )
    old_log_probs_tensor = torch.as_tensor(rollout.old_log_probs, dtype=torch.float32, device=cpu_device)
    returns_tensor = torch.as_tensor(rollout.returns, dtype=torch.float32, device=cpu_device)
    advantages_tensor = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=cpu_device)
    bc_coef = max(0.0, float(getattr(cfg, "policy_bc_regularization_coef", 0.0))) # Danger (BC Not used, please check if possible)
    log_std = getattr(policy, "log_std", None)
    initial_log_std_mean = float(log_std.detach().mean().cpu().item()) if log_std is not None else float("nan")
    initial_action_std_mean = float(torch.exp(log_std.detach()).mean().cpu().item()) if log_std is not None else float("nan")
    if bc_coef > 0.0:
        if not _is_continuous(cfg):
            raise ValueError("policy_bc_regularization_coef currently requires continuous action mode.")
        if expert_policy_observations is None or expert_actions is None:
            raise ValueError("policy_bc_regularization_coef requires expert policy observations and actions.")
        expert_obs_tensor = torch.as_tensor(
            np.asarray(expert_policy_observations, dtype=np.float32),
            dtype=torch.float32,
            device=cpu_device,
        )
        expert_action_tensor = torch.as_tensor(
            np.clip(np.asarray(expert_actions, dtype=np.float32), -1.0, 1.0),
            dtype=torch.float32,
            device=cpu_device,
        )
        if len(expert_obs_tensor) != len(expert_action_tensor):
            raise ValueError(
                "Expert BC regularization observation/action count mismatch: "
                f"{len(expert_obs_tensor)} != {len(expert_action_tensor)}."
            )
        if len(expert_obs_tensor) == 0:
            raise ValueError("Expert BC regularization received no expert samples.")
    else:
        expert_obs_tensor = None
        expert_action_tensor = None
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    ratio_means: list[float] = []
    ratio_stds: list[float] = []
    bc_losses: list[float] = []
    post_update_approx_kl = float("nan")
    post_update_ratio_mean = float("nan")
    post_update_ratio_std = float("nan")
    batch_size = max(1, int(cfg.batch_size))
    num_samples = int(obs_tensor.shape[0])
    # Micro Batch and Mini Batch
    configured_micro_batch_size = int(getattr(cfg, "ppo_micro_batch_size", 0))
    if configured_micro_batch_size > 0:
        micro_batch_size = max(1, min(batch_size, configured_micro_batch_size))
    elif str(getattr(cfg, "policy_model", "")).lower() in {"transformer", "recurrent_transformer"}:
        micro_batch_size = min(batch_size, 128)
    else:
        micro_batch_size = batch_size

    def device_batch(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch = tensor[indices]
        if device.type == "cuda":
            batch = batch.pin_memory().to(device=device, non_blocking=True)
        else:
            batch = batch.to(device=device)
        return batch

    try:
        for _ in range(int(cfg.ppo_epochs)):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_idx = permutation[start : start + batch_size]
                batch_count = int(batch_idx.numel())
                optimizer.zero_grad(set_to_none=True)
                weighted_policy_loss = 0.0
                weighted_value_loss = 0.0
                weighted_entropy = 0.0
                weighted_approx_kl = 0.0
                weighted_clip_fraction = 0.0
                weighted_ratio_mean = 0.0
                weighted_ratio_std = 0.0
                for micro_start in range(0, batch_count, micro_batch_size):
                    micro_idx = batch_idx[micro_start : micro_start + micro_batch_size]
                    micro_count = int(micro_idx.numel())
                    micro_weight = float(micro_count) / float(batch_count)
                    obs = device_batch(obs_tensor, micro_idx)
                    actions = device_batch(action_tensor, micro_idx)
                    old_log_probs = device_batch(old_log_probs_tensor, micro_idx)
                    returns = device_batch(returns_tensor, micro_idx)
                    advantages = device_batch(advantages_tensor, micro_idx)
                    critic_obs = device_batch(critic_obs_tensor, micro_idx)
                    masks = device_batch(action_mask_tensor, micro_idx) if action_mask_tensor is not None else None
                    dist, values = policy_distribution_and_values(
                        policy,
                        obs,
                        cfg,
                        masks,
                        critic_obs_tensor=critic_obs,
                    )
                    log_probs = dist.log_prob(actions)
                    log_ratio = log_probs - old_log_probs
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    policy_loss = -torch.min(
                        ratio * advantages,
                        clipped_ratio * advantages,
                    ).mean()
                    value_loss = F.mse_loss(values, returns)
                    entropy = dist.entropy().mean()
                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                    bc_loss = None
                    if bc_coef > 0.0 and expert_obs_tensor is not None and expert_action_tensor is not None:
                        expert_idx = torch.randint(
                            0,
                            int(expert_obs_tensor.shape[0]),
                            (micro_count,),
                            device=cpu_device,
                        )
                        expert_obs = device_batch(expert_obs_tensor, expert_idx)
                        expert_actions_for_bc = device_batch(expert_action_tensor, expert_idx)
                        pred_expert_actions, _expert_values = policy(expert_obs)
                        bc_loss = F.mse_loss(pred_expert_actions, expert_actions_for_bc)
                        loss = loss + bc_coef * bc_loss
                    (loss * micro_weight).backward()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fraction = (
                            (torch.abs(ratio - 1.0) > float(cfg.clip_range)).float().mean()
                        )
                        weighted_policy_loss += micro_weight * float(policy_loss.detach().cpu().item())
                        weighted_value_loss += micro_weight * float(value_loss.detach().cpu().item())
                        weighted_entropy += micro_weight * float(entropy.detach().cpu().item())
                        weighted_approx_kl += micro_weight * float(approx_kl.detach().cpu().item())
                        weighted_clip_fraction += micro_weight * float(clip_fraction.detach().cpu().item())
                        weighted_ratio_mean += micro_weight * float(ratio.mean().detach().cpu().item())
                        weighted_ratio_std += micro_weight * float(
                            ratio.std(unbiased=False).detach().cpu().item()
                        )
                        if bc_loss is not None:
                            bc_losses.append(float(bc_loss.detach().cpu().item()))
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
                policy_losses.append(weighted_policy_loss)
                value_losses.append(weighted_value_loss)
                entropies.append(weighted_entropy)
                approx_kls.append(weighted_approx_kl)
                clip_fractions.append(weighted_clip_fraction)
                ratio_means.append(weighted_ratio_mean)
                ratio_stds.append(weighted_ratio_std)

        diagnostic_count = min(num_samples, max(batch_size, 4096))
        if diagnostic_count > 0:
            diagnostic_idx = torch.randperm(num_samples)[:diagnostic_count]
            weighted_post_kl = 0.0
            weighted_post_ratio_mean = 0.0
            weighted_post_ratio_std = 0.0
            total_count = int(diagnostic_idx.numel())
            with torch.no_grad():
                for micro_start in range(0, total_count, micro_batch_size):
                    micro_idx = diagnostic_idx[micro_start : micro_start + micro_batch_size]
                    micro_count = int(micro_idx.numel())
                    micro_weight = float(micro_count) / float(total_count)
                    obs = device_batch(obs_tensor, micro_idx)
                    actions = device_batch(action_tensor, micro_idx)
                    old_log_probs = device_batch(old_log_probs_tensor, micro_idx)
                    critic_obs = device_batch(critic_obs_tensor, micro_idx)
                    masks = device_batch(action_mask_tensor, micro_idx) if action_mask_tensor is not None else None
                    dist, _values = policy_distribution_and_values(
                        policy,
                        obs,
                        cfg,
                        masks,
                        critic_obs_tensor=critic_obs,
                    )
                    log_ratio = dist.log_prob(actions) - old_log_probs
                    ratio = torch.exp(log_ratio)
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    weighted_post_kl += micro_weight * float(approx_kl.detach().cpu().item())
                    weighted_post_ratio_mean += micro_weight * float(ratio.mean().detach().cpu().item())
                    weighted_post_ratio_std += micro_weight * float(
                        ratio.std(unbiased=False).detach().cpu().item()
                    )
            post_update_approx_kl = weighted_post_kl
            post_update_ratio_mean = weighted_post_ratio_mean
            post_update_ratio_std = weighted_post_ratio_std
    finally:
        if was_training:
            policy.train()
    final_log_std_mean = float(log_std.detach().mean().cpu().item()) if log_std is not None else float("nan")
    final_action_std_mean = float(torch.exp(log_std.detach()).mean().cpu().item()) if log_std is not None else float("nan")

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
        "post_update_approx_kl": float(post_update_approx_kl),
        "clip_fraction": float(np.mean(clip_fractions)),
        "ratio_mean": float(np.mean(ratio_means)),
        "ratio_std": float(np.mean(ratio_stds)),
        "post_update_ratio_mean": float(post_update_ratio_mean),
        "post_update_ratio_std": float(post_update_ratio_std),
        "advantage_mean": float(np.mean(rollout.advantages)) if rollout.advantages.size else 0.0,
        "advantage_std": float(np.std(rollout.advantages)) if rollout.advantages.size else 0.0,
        "bc_regularization_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
        "bc_regularization_coef": float(bc_coef),
        "ppo_micro_batch_size": float(micro_batch_size),
        "log_std_mean": final_log_std_mean,
        "action_std_param_mean": final_action_std_mean,
        "log_std_delta": final_log_std_mean - initial_log_std_mean,
        "action_std_param_delta": final_action_std_mean - initial_action_std_mean,
    }

__all__ = [
    '_recurrent_rollout_chunks',
    '_update_recurrent_policy',
    'update_policy'
]
