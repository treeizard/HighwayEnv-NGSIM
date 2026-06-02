#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scripts_gail.ps_gail.config import PSGAILConfig, should_save_checkpoint_video
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.envs import make_training_env
from scripts_gail.ps_gail.monitoring import WandbMonitor
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.models import make_relu_mlp
from scripts_gail.ps_gail.observations import flatten_agent_observations, policy_observations_from_flat
from scripts_gail.ps_gail.schedule import config_for_round
from scripts_gail.ps_gail.trainer import (
    RolloutBatch,
    collect_round_rollouts,
    combine_primary_env_challenge_rewards,
    compute_returns_and_advantages,
    evaluate_policy_matched_trajectories,
    infer_continuous_action_dim,
    infer_critic_obs_dim,
    infer_policy_obs_dim,
    make_evaluation_executor,
    make_rollout_executor,
    policy_distribution_and_values,
    policy_distribution_memory,
    recurrent_policy_enabled,
    resolve_device,
    safe_normalize_adversarial_rewards,
    set_optimizer_lr,
    should_apply_gail_reward_clip,
    should_normalize_gail_reward,
    subsample_rollout_for_training,
    update_policy,
)
from scripts_gail.ps_gail.validation import (
    best_checkpoint_payload,
    matched_validation_summary,
    scored_validation_metrics,
)
from scripts_gail.train_simple_ps_gail import (
    append_policy_archive,
    behavior_clone_pretrain,
    evaluate_policy_survival,
    policy_archive_snapshot,
    training_risk_warnings,
)


class AIRLReward(nn.Module):
    """Potential-based AIRL reward model for continuous NGSIM actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int | None = None,
        *,
        hidden_sizes: str | int | tuple[int, ...] | list[int] | None = None,
        dropout: float = 0.0,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        critic_hidden_sizes = hidden_sizes if hidden_sizes is not None else hidden_size
        self.reward = make_relu_mlp(
            int(obs_dim) + int(action_dim),
            critic_hidden_sizes,
            1,
            dropout=dropout,
            spectral_norm=spectral_norm,
        )
        self.potential = make_relu_mlp(
            int(obs_dim),
            critic_hidden_sizes,
            1,
            dropout=dropout,
            spectral_norm=spectral_norm,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.reward(torch.cat([obs, actions], dim=-1)).squeeze(-1)

    def shaped_logits(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        *,
        gamma: float,
    ) -> torch.Tensor:
        reward = self.forward(obs, actions)
        current_potential = self.potential(obs).squeeze(-1)
        next_potential = self.potential(next_obs).squeeze(-1)
        return reward + float(gamma) * (1.0 - dones.float()) * next_potential - current_potential


@dataclass(frozen=True)
class AIRLReplayEntry:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    trajectory_ids: np.ndarray
    timesteps: np.ndarray


def _unique_trajectory_values(trajectory_ids: np.ndarray) -> list[object]:
    return list(dict.fromkeys(np.asarray(trajectory_ids).reshape(-1).tolist()))


def _trajectory_timesteps(trajectory_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(trajectory_ids).reshape(-1)
    timesteps = np.zeros(ids.shape[0], dtype=np.int64)
    for trajectory_id in _unique_trajectory_values(ids):
        indices = np.flatnonzero(ids == trajectory_id).astype(np.int64)
        timesteps[indices] = np.arange(indices.shape[0], dtype=np.int64)
    return timesteps


def _prefixed_trajectory_ids(trajectory_ids: np.ndarray, *, source: str) -> np.ndarray:
    ids = np.asarray(trajectory_ids).reshape(-1)
    return np.asarray([f"{source}:{item}" for item in ids.tolist()], dtype=object)


def _subsample_complete_trajectories(
    trajectory_ids: np.ndarray,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ids = np.asarray(trajectory_ids).reshape(-1)
    if max_samples <= 0 or ids.shape[0] <= max_samples:
        return np.arange(ids.shape[0], dtype=np.int64)
    trajectory_values = _unique_trajectory_values(ids)
    rng.shuffle(trajectory_values)
    selected: list[np.ndarray] = []
    selected_count = 0
    for trajectory_id in trajectory_values:
        indices = np.flatnonzero(ids == trajectory_id).astype(np.int64)
        if selected and selected_count + indices.shape[0] > max_samples:
            continue
        selected.append(indices)
        selected_count += int(indices.shape[0])
        if selected_count >= max_samples:
            break
    if not selected:
        first_indices = np.flatnonzero(ids == trajectory_values[0]).astype(np.int64)
        selected.append(first_indices)
    return np.sort(np.concatenate(selected, axis=0).astype(np.int64, copy=False))


def append_airl_replay(
    replay: list[AIRLReplayEntry],
    rollout: RolloutBatch,
    cfg: PSGAILConfig,
    *,
    round_idx: int | None = None,
) -> None:
    replay_rounds = max(0, int(getattr(cfg, "discriminator_replay_rounds", 0)))
    if replay_rounds <= 0:
        replay.clear()
        return
    if int(rollout.num_agent_steps) <= 0:
        return
    source = f"round:{int(round_idx)}" if round_idx is not None else f"rollout:{id(rollout)}"
    replay.append(
        AIRLReplayEntry(
            observations=np.asarray(rollout.policy_observations, dtype=np.float32).copy(),
            actions=np.asarray(rollout.actions, dtype=np.float32).copy(),
            next_observations=np.asarray(rollout.next_policy_observations, dtype=np.float32).copy(),
            dones=np.asarray(rollout.dones, dtype=bool).copy(),
            trajectory_ids=_prefixed_trajectory_ids(rollout.trajectory_ids, source=source),
            timesteps=_trajectory_timesteps(rollout.trajectory_ids),
        )
    )
    del replay[:-replay_rounds]


def concat_airl_replay(
    rollout: RolloutBatch,
    replay: list[AIRLReplayEntry],
    cfg: PSGAILConfig,
    *,
    seed: int,
    preserve_recurrent_context: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    current_trajectory_ids = _prefixed_trajectory_ids(rollout.trajectory_ids, source="current")
    current_timesteps = _trajectory_timesteps(rollout.trajectory_ids)
    current = (
        np.asarray(rollout.policy_observations, dtype=np.float32),
        np.asarray(rollout.actions, dtype=np.float32),
        np.asarray(rollout.next_policy_observations, dtype=np.float32),
        np.asarray(rollout.dones, dtype=bool),
        current_trajectory_ids,
        current_timesteps,
    )
    current_log_probs = np.asarray(rollout.old_log_probs, dtype=np.float32)
    replay_rounds = max(0, int(getattr(cfg, "discriminator_replay_rounds", 0)))
    if replay_rounds <= 0 or not replay:
        return (*current[:4], current_log_probs, current_trajectory_ids, current_timesteps)
    entries = [entry for entry in replay[-replay_rounds:] if len(entry.observations) > 0]
    if not entries:
        return (*current[:4], current_log_probs, current_trajectory_ids, current_timesteps)
    obs = np.concatenate([*(entry.observations for entry in entries), current[0]], axis=0).astype(
        np.float32,
        copy=False,
    )
    actions = np.concatenate([*(entry.actions for entry in entries), current[1]], axis=0).astype(
        np.float32,
        copy=False,
    )
    next_obs = np.concatenate([*(entry.next_observations for entry in entries), current[2]], axis=0).astype(
        np.float32,
        copy=False,
    )
    dones = np.concatenate([*(entry.dones for entry in entries), current[3]], axis=0).astype(bool, copy=False)
    trajectory_ids = np.concatenate([*(entry.trajectory_ids for entry in entries), current[4]], axis=0).astype(
        object,
        copy=False,
    )
    timesteps = np.concatenate([*(entry.timesteps for entry in entries), current[5]], axis=0).astype(
        np.int64,
        copy=False,
    )
    max_samples = max(0, int(getattr(cfg, "discriminator_replay_max_samples", 0)))
    if max_samples > 0 and len(obs) > max_samples:
        rng = np.random.default_rng(int(seed))
        if preserve_recurrent_context:
            # Recurrent AIRL needs preceding observations to rebuild transformer
            # memory under the current policy, so replay sampling keeps whole
            # trajectory segments instead of isolated transitions.
            indices = _subsample_complete_trajectories(
                trajectory_ids,
                max_samples=max_samples,
                rng=rng,
            )
        else:
            indices = rng.choice(len(obs), size=max_samples, replace=False)
        obs = obs[indices]
        actions = actions[indices]
        next_obs = next_obs[indices]
        dones = dones[indices]
        trajectory_ids = trajectory_ids[indices]
        timesteps = timesteps[indices]
    return obs, actions, next_obs, dones, None, trajectory_ids, timesteps


def _as_tensor(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def env_signature(cfg: PSGAILConfig) -> tuple[object, ...]:
    return (
        str(cfg.action_mode),
        bool(cfg.control_all_vehicles),
        float(cfg.percentage_controlled_vehicles),
        str(cfg.max_surrounding),
        bool(cfg.enable_collision),
        bool(cfg.terminate_when_all_controlled_crashed),
        bool(cfg.allow_idm),
        int(cfg.max_episode_steps),
        str(cfg.prebuilt_split),
    )


def airl_checkpoint_payload(
    *,
    round_idx: int,
    policy: torch.nn.Module,
    reward_model: torch.nn.Module,
    expert_metadata: dict[str, object],
    cfg: PSGAILConfig,
    round_cfg: PSGAILConfig,
) -> dict[str, object]:
    return {
        "round": int(round_idx),
        "policy_state_dict": policy.state_dict(),
        "reward_state_dict": reward_model.state_dict(),
        "expert_metadata": expert_metadata,
        "config": vars(cfg),
        "round_config": vars(round_cfg),
    }


def load_airl_resume_checkpoint(
    *,
    resume_checkpoint: str,
    policy: torch.nn.Module,
    reward_model: torch.nn.Module,
    device: torch.device,
    allow_missing_reward_state: bool = False,
) -> dict[str, object]:
    if not os.path.isfile(resume_checkpoint):
        raise FileNotFoundError(f"resume_checkpoint does not exist: {resume_checkpoint}")
    try:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
    if "policy_state_dict" not in checkpoint:
        raise RuntimeError(f"Checkpoint is missing policy_state_dict: {resume_checkpoint}")
    if "reward_state_dict" not in checkpoint and not bool(allow_missing_reward_state):
        raise RuntimeError(
            f"Checkpoint {resume_checkpoint!r} is missing reward_state_dict. "
            "This is not a valid AIRL resume checkpoint; pass "
            "--allow-airl-resume-without-reward only for an intentional ablation."
        )
    try:
        policy.load_state_dict(checkpoint["policy_state_dict"])
        if "reward_state_dict" in checkpoint:
            reward_model.load_state_dict(checkpoint["reward_state_dict"])
        else:
            warnings.warn(
                f"Checkpoint {resume_checkpoint!r} has no reward_state_dict; "
                "AIRL reward model will start from initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load AIRL resume checkpoint. Check that policy/reward "
            "architecture settings match the checkpoint."
        ) from exc
    return checkpoint


def _policy_log_probs(
    policy: nn.Module,
    cfg: PSGAILConfig,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    dist, _values = policy_distribution_and_values(policy, obs, cfg, None)
    return dist.log_prob(actions)


def _recurrent_context_windows(
    trajectory_ids: np.ndarray,
    timesteps: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(trajectory_ids)
    steps = np.asarray(timesteps, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if ids.shape[0] != steps.shape[0]:
        raise ValueError(f"trajectory_ids/timesteps length mismatch: {ids.shape[0]} != {steps.shape[0]}.")
    if targets.size == 0:
        return np.zeros((0, 1), dtype=np.int64)

    orders: dict[object, np.ndarray] = {}
    positions: dict[int, int] = {}
    for trajectory_id in _unique_trajectory_values(ids):
        trajectory_indices = np.flatnonzero(ids == trajectory_id).astype(np.int64)
        ordered = trajectory_indices[
            np.lexsort((trajectory_indices, steps[trajectory_indices]))
        ].astype(np.int64, copy=False)
        orders[trajectory_id] = ordered
        for pos, source_index in enumerate(ordered):
            positions[int(source_index)] = int(pos)

    sequences: list[np.ndarray] = []
    for row, target_index in enumerate(targets):
        trajectory_id = ids[int(target_index)]
        ordered = orders[trajectory_id]
        pos = positions[int(target_index)]
        sequence = [int(target_index)]
        cursor = pos
        # Step-memory is itself produced by a recurrent forward pass, so exact
        # AIRL log_pi reconstruction must warm up from the start of the
        # contiguous expert segment, not only from the last memory-context
        # observations. Gaps reset memory because skipped frames cannot be
        # replayed through the policy.
        while cursor > 0:
            current_index = int(ordered[cursor])
            previous_index = int(ordered[cursor - 1])
            if int(steps[current_index]) - int(steps[previous_index]) != 1:
                break
            sequence.append(previous_index)
            cursor -= 1
        sequence.reverse()
        sequences.append(np.asarray(sequence, dtype=np.int64))

    window_width = max(int(len(sequence)) for sequence in sequences)
    windows = np.full((targets.size, window_width), -1, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        windows[row, -len(sequence) :] = sequence
    return windows


def _recurrent_policy_log_probs_for_indices(
    policy: nn.Module,
    cfg: PSGAILConfig,
    observations: np.ndarray,
    actions: np.ndarray,
    trajectory_ids: np.ndarray,
    timesteps: np.ndarray,
    target_indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    target_indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if target_indices.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if not recurrent_policy_enabled(policy):
        obs = _as_tensor(np.asarray(observations, dtype=np.float32)[target_indices], device=device)
        act = _as_tensor(np.asarray(actions, dtype=np.float32)[target_indices], device=device)
        with torch.no_grad():
            return _policy_log_probs(policy, cfg, obs, act).detach().cpu().numpy().astype(np.float32)

    obs_np = np.asarray(observations, dtype=np.float32)
    action_np = np.asarray(actions, dtype=np.float32)
    windows = _recurrent_context_windows(
        trajectory_ids,
        timesteps,
        target_indices,
    )
    result = np.empty((target_indices.size,), dtype=np.float32)
    micro_batch = max(1, min(max(1, int(batch_size)), int(target_indices.size)))

    with torch.no_grad():
        for start in range(0, target_indices.size, micro_batch):
            stop = min(target_indices.size, start + micro_batch)
            batch_windows = windows[start:stop]
            batch_count = int(batch_windows.shape[0])
            memory = policy.initial_memory(batch_count, device=device, dtype=torch.float32)
            batch_log_probs = torch.zeros(batch_count, dtype=torch.float32, device=device)
            for column in range(batch_windows.shape[1]):
                active_np = batch_windows[:, column] >= 0
                if not np.any(active_np):
                    continue
                active_rows_np = np.flatnonzero(active_np).astype(np.int64)
                source_indices = batch_windows[active_rows_np, column].astype(np.int64)
                active_rows = torch.as_tensor(active_rows_np, dtype=torch.long, device=device)
                step_memory = memory[active_rows]
                dist, new_step_memory = policy_distribution_memory(
                    policy,
                    _as_tensor(obs_np[source_indices], device=device),
                    cfg,
                    None,
                    memory=step_memory,
                    return_memory=True,
                )
                if new_step_memory is None:
                    raise RuntimeError("Recurrent AIRL log-prob reconstruction did not return memory.")
                if column == batch_windows.shape[1] - 1:
                    batch_log_probs[active_rows] = dist.log_prob(
                        _as_tensor(action_np[source_indices], device=device)
                    )
                updated_memory = torch.cat([step_memory[:, 1:], new_step_memory.unsqueeze(1)], dim=1)
                memory = memory.clone()
                memory[active_rows] = updated_memory
            result[start:stop] = batch_log_probs.detach().cpu().numpy().astype(np.float32, copy=False)
    return result


def _airl_wgan_gradient_penalty(
    reward_model: AIRLReward,
    expert_obs: torch.Tensor,
    expert_actions: torch.Tensor,
    expert_next_obs: torch.Tensor,
    expert_dones: torch.Tensor,
    gen_obs: torch.Tensor,
    gen_actions: torch.Tensor,
    gen_next_obs: torch.Tensor,
    gen_dones: torch.Tensor,
    *,
    gamma: float,
    gp_lambda: float,
) -> torch.Tensor:
    alpha = torch.rand((expert_obs.shape[0], 1), dtype=expert_obs.dtype, device=expert_obs.device)
    obs = (alpha * expert_obs + (1.0 - alpha) * gen_obs).requires_grad_(True)
    actions = (alpha * expert_actions + (1.0 - alpha) * gen_actions).requires_grad_(True)
    next_obs = (alpha * expert_next_obs + (1.0 - alpha) * gen_next_obs).requires_grad_(True)
    dones = (alpha.squeeze(-1) * expert_dones + (1.0 - alpha.squeeze(-1)) * gen_dones).requires_grad_(True)
    # Keep the policy correction in the WGAN critic loss, but do not double-backward
    # through the policy network for GP; transformer policies make that graph huge.
    scores = reward_model.shaped_logits(obs, actions, next_obs, dones, gamma=gamma)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=(obs, actions, next_obs, dones),
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    flat_gradients = torch.cat([gradient.reshape(gradient.shape[0], -1) for gradient in gradients], dim=1)
    gradient_norm = flat_gradients.norm(2, dim=1)
    return float(gp_lambda) * torch.square(gradient_norm - 1.0).mean()


def update_reward_model(
    reward_model: AIRLReward,
    optimizer: torch.optim.Optimizer,
    policy: nn.Module,
    expert_obs: np.ndarray,
    expert_actions: np.ndarray,
    expert_next_obs: np.ndarray,
    expert_dones: np.ndarray,
    generator_obs: np.ndarray,
    generator_actions: np.ndarray,
    generator_next_obs: np.ndarray,
    generator_dones: np.ndarray,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    reward_batch_size: int,
    log_prob_batch_size: int | None = None,
    expert_trajectory_ids: np.ndarray | None = None,
    expert_timesteps: np.ndarray | None = None,
    generator_log_probs: np.ndarray | None = None,
    generator_trajectory_ids: np.ndarray | None = None,
    generator_timesteps: np.ndarray | None = None,
) -> dict[str, float]:
    n = int(len(generator_obs))
    expert_idx = np.random.choice(len(expert_obs), size=n, replace=len(expert_obs) < n)
    recurrent_policy = recurrent_policy_enabled(policy)
    expert_obs_t = _as_tensor(expert_obs[expert_idx], device=device)
    expert_actions_t = _as_tensor(expert_actions[expert_idx], device=device)
    expert_next_obs_t = _as_tensor(expert_next_obs[expert_idx], device=device)
    expert_dones_t = _as_tensor(expert_dones[expert_idx].astype(np.float32), device=device)
    gen_obs_t = _as_tensor(generator_obs, device=device)
    gen_actions_t = _as_tensor(generator_actions, device=device)
    gen_next_obs_t = _as_tensor(generator_next_obs, device=device)
    gen_dones_t = _as_tensor(generator_dones.astype(np.float32), device=device)

    policy.eval()
    reward_model.train()
    if recurrent_policy:
        log_prob_micro_batch_size = max(
            1,
            int(log_prob_batch_size if log_prob_batch_size is not None else reward_batch_size),
        )
        if expert_trajectory_ids is None or expert_timesteps is None:
            raise ValueError(
                "Recurrent AIRL requires expert_trajectory_ids and expert_timesteps "
                "to reconstruct expert policy memory for AIRL log_pi."
            )
        if generator_log_probs is None:
            if generator_trajectory_ids is None or generator_timesteps is None:
                raise ValueError(
                    "Recurrent AIRL requires generator_log_probs or generator_trajectory_ids/"
                    "generator_timesteps. Replayed samples must rebuild current-policy "
                    "transformer memory instead of using blank-memory log_pi."
                )
            # Replay samples came from older policies, so their stored old_log_probs
            # are stale. Recompute current-policy log_pi by replaying each stored
            # trajectory context through the recurrent policy.
            generator_log_probs = _recurrent_policy_log_probs_for_indices(
                policy,
                cfg,
                generator_obs,
                generator_actions,
                generator_trajectory_ids,
                generator_timesteps,
                np.arange(n, dtype=np.int64),
                device,
                batch_size=log_prob_micro_batch_size,
            )
        expert_log_probs = _recurrent_policy_log_probs_for_indices(
            policy,
            cfg,
            expert_obs,
            expert_actions,
            expert_trajectory_ids,
            expert_timesteps,
            expert_idx,
            device,
            batch_size=log_prob_micro_batch_size,
        )
    else:
        expert_log_probs = None
    gen_log_probs_t = (
        _as_tensor(np.asarray(generator_log_probs, dtype=np.float32), device=device)
        if generator_log_probs is not None
        else None
    )
    if gen_log_probs_t is not None and int(gen_log_probs_t.shape[0]) != n:
        raise ValueError(
            "generator_log_probs length must match generator samples: "
            f"{int(gen_log_probs_t.shape[0])} != {n}."
        )
    expert_log_probs_t = (
        _as_tensor(np.asarray(expert_log_probs, dtype=np.float32), device=device)
        if expert_log_probs is not None
        else None
    )
    losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []
    wgan_losses: list[torch.Tensor] = []
    gradient_penalties: list[torch.Tensor] = []
    critic_gaps: list[torch.Tensor] = []
    expert_accs: list[torch.Tensor] = []
    gen_accs: list[torch.Tensor] = []
    expert_rewards: list[torch.Tensor] = []
    gen_rewards: list[torch.Tensor] = []
    batch_size = max(1, int(reward_batch_size))
    loss_type = str(getattr(cfg, "discriminator_loss", "airl_bce")).lower()
    if loss_type not in {"airl", "airl_bce", "bce", "wgan_gp"}:
        raise ValueError(f"Unsupported AIRL discriminator_loss={loss_type!r}.")

    policy_params = list(policy.parameters())
    policy_requires_grad = [param.requires_grad for param in policy_params]
    for param in policy_params:
        param.requires_grad_(False)

    try:
        for _ in range(max(1, int(cfg.disc_updates_per_round))):
            permutation = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = permutation[start : start + batch_size]
                eo = expert_obs_t[idx]
                ea = expert_actions_t[idx]
                eno = expert_next_obs_t[idx]
                ed = expert_dones_t[idx]
                go = gen_obs_t[idx]
                ga = gen_actions_t[idx]
                gno = gen_next_obs_t[idx]
                gd = gen_dones_t[idx]
                expert_shaped = reward_model.shaped_logits(eo, ea, eno, ed, gamma=float(cfg.gamma))
                gen_shaped = reward_model.shaped_logits(go, ga, gno, gd, gamma=float(cfg.gamma))
                expert_reward = reward_model(eo, ea)
                gen_reward = reward_model(go, ga)
                with torch.no_grad():
                    expert_log_pi = (
                        expert_log_probs_t[idx]
                        if expert_log_probs_t is not None
                        else _policy_log_probs(policy, cfg, eo, ea)
                    )
                    # For recurrent policies this must be the rollout-time
                    # likelihood, which already includes transformer memory.
                    # Recomputing through _policy_log_probs would silently use
                    # blank memory and corrupt AIRL's f - log pi discriminator.
                    gen_log_pi = (
                        gen_log_probs_t[idx]
                        if gen_log_probs_t is not None
                        else _policy_log_probs(policy, cfg, go, ga)
                    )
                expert_logits = expert_shaped - expert_log_pi
                gen_logits = gen_shaped - gen_log_pi
                if loss_type == "wgan_gp":
                    wgan_loss = gen_logits.mean() - expert_logits.mean()
                    gradient_penalty = _airl_wgan_gradient_penalty(
                        reward_model,
                        eo,
                        ea,
                        eno,
                        ed,
                        go,
                        ga,
                        gno,
                        gd,
                        gamma=float(cfg.gamma),
                        gp_lambda=float(getattr(cfg, "wgan_gp_lambda", 2.0)),
                    )
                    loss = wgan_loss + gradient_penalty
                else:
                    logits = torch.cat([expert_logits, gen_logits], dim=0)
                    labels = torch.cat(
                        [
                            torch.full_like(expert_logits, float(cfg.disc_expert_label)),
                            torch.full_like(gen_logits, float(cfg.disc_generator_label)),
                        ],
                        dim=0,
                    )
                    bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
                    loss = bce_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(reward_model.parameters(), float(cfg.max_grad_norm))
                optimizer.step()
                with torch.no_grad():
                    if loss_type == "wgan_gp":
                        centered_threshold = 0.5 * (expert_logits.mean() + gen_logits.mean())
                        expert_accs.append((expert_logits > centered_threshold).float().mean())
                        gen_accs.append((gen_logits < centered_threshold).float().mean())
                        wgan_losses.append(wgan_loss.detach())
                        gradient_penalties.append(gradient_penalty.detach())
                        critic_gaps.append((expert_logits.mean() - gen_logits.mean()).detach())
                    else:
                        expert_accs.append((expert_logits > 0.0).float().mean())
                        gen_accs.append((gen_logits < 0.0).float().mean())
                        bce_losses.append(bce_loss.detach())
                        critic_gaps.append((expert_logits.mean() - gen_logits.mean()).detach())
                    expert_rewards.append(expert_reward.mean())
                    gen_rewards.append(gen_reward.mean())
                losses.append(loss.detach())
    finally:
        for param, requires_grad in zip(policy_params, policy_requires_grad):
            param.requires_grad_(requires_grad)

    def mean(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().detach().cpu().item()) if values else float("nan")

    return {
        "reward_loss": mean(losses),
        "bce_loss": mean(bce_losses),
        "wgan_loss": mean(wgan_losses),
        "gradient_penalty": mean(gradient_penalties),
        "critic_gap": mean(critic_gaps),
        "expert_acc": mean(expert_accs),
        "gen_acc": mean(gen_accs),
        "expert_reward": mean(expert_rewards),
        "gen_reward": mean(gen_rewards),
    }


def refresh_airl_rewards(
    rollout: RolloutBatch,
    reward_model: AIRLReward,
    cfg: PSGAILConfig,
    device: torch.device,
) -> RolloutBatch:
    reward_model.eval()
    with torch.no_grad():
        raw = reward_model(
            _as_tensor(rollout.policy_observations, device=device),
            _as_tensor(rollout.actions.astype(np.float32, copy=False), device=device),
        ).detach().cpu().numpy().astype(np.float32)
        shaped_logits = reward_model.shaped_logits(
            _as_tensor(rollout.policy_observations, device=device),
            _as_tensor(rollout.actions.astype(np.float32, copy=False), device=device),
            _as_tensor(rollout.next_policy_observations, device=device),
            _as_tensor(rollout.dones.astype(np.float32), device=device),
            gamma=float(cfg.gamma),
        ).detach().cpu().numpy().astype(np.float32)
    corrected_logits = (shaped_logits - rollout.old_log_probs).astype(np.float32, copy=True)
    reward_mode = str(getattr(cfg, "airl_policy_reward_mode", "shaped")).lower()
    if reward_mode not in {"discriminator", "shaped", "reward"}:
        raise ValueError(
            "airl_policy_reward_mode must be one of 'discriminator', 'shaped', or 'reward', "
            f"got {reward_mode!r}."
        )
    if str(getattr(cfg, "discriminator_loss", "airl_bce")).lower() == "wgan_gp":
        shaped = (
            corrected_logits
            if reward_mode == "discriminator"
            else shaped_logits
            if reward_mode == "shaped"
            else raw
        )
        if bool(getattr(cfg, "wgan_reward_center", False)) and shaped.size > 1:
            shaped = shaped - shaped.mean()
        reward_scale = float(getattr(cfg, "wgan_reward_scale", 1.0))
        if reward_scale != 1.0:
            shaped = shaped * reward_scale
        wgan_clip = float(getattr(cfg, "wgan_reward_clip", 0.0))
        if wgan_clip > 0.0:
            shaped = np.clip(shaped, -wgan_clip, wgan_clip)
    else:
        # Canonical AIRL trains the policy on log D - log(1-D), which simplifies to
        # f(s,a,s') - log pi(a|s) for D = exp(f)/(exp(f) + pi(a|s)).
        shaped = (
            corrected_logits
            if reward_mode == "discriminator"
            else shaped_logits
            if reward_mode == "shaped"
            else raw
        )
    if should_normalize_gail_reward(cfg) and shaped.size > 1:
        shaped = safe_normalize_adversarial_rewards(shaped, cfg)
    if should_apply_gail_reward_clip(cfg):
        shaped = np.clip(shaped, -float(cfg.gail_reward_clip), float(cfg.gail_reward_clip))
    rewards, challenge_bonuses = combine_primary_env_challenge_rewards(
        shaped,
        rollout.env_penalties,
        cfg,
        challenge_payoffs=rollout.challenge_payoffs,
    )
    returns, advantages = compute_returns_and_advantages(
        rewards.astype(np.float32),
        rollout.old_values,
        rollout.dones,
        rollout.trajectory_ids,
        cfg,
    )
    return replace(
        rollout,
        rewards=rewards.astype(np.float32),
        gail_rewards_raw=(
            corrected_logits
            if str(getattr(cfg, "discriminator_loss", "airl_bce")).lower() == "wgan_gp"
            else raw
        ),
        gail_rewards_normalized=shaped.astype(np.float32),
        challenge_bonuses=challenge_bonuses,
        returns=returns,
        advantages=advantages,
        mean_raw_gail_reward=(
            float(corrected_logits.mean())
            if str(getattr(cfg, "discriminator_loss", "airl_bce")).lower() == "wgan_gp" and corrected_logits.size
            else float(raw.mean()) if raw.size else 0.0
        ),
        mean_normalized_gail_reward=float(shaped.mean()) if shaped.size else 0.0,
    )


def _policy_action_tuple(
    policy: nn.Module,
    obs,
    *,
    device: torch.device,
    cfg: PSGAILConfig,
    memory: torch.Tensor | None = None,
    return_memory: bool = False,
) -> tuple[object, ...] | tuple[tuple[object, ...], torch.Tensor | None]:
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
        if recurrent_policy_enabled(policy) and memory is not None and int(memory.shape[0]) != len(obs_agents):
            memory = policy.initial_memory(len(obs_agents), device=device, dtype=torch.float32)
        if recurrent_policy_enabled(policy):
            mean, _values, new_memory = policy(obs_tensor, memory=memory, return_memory=True)
        else:
            mean, _values = policy(obs_tensor)
            new_memory = None
        if getattr(policy, "log_std", None) is None:
            raise RuntimeError("AIRL checkpoint video expects a continuous policy.")
        actions = torch.clamp(mean, -1.0, 1.0).detach().cpu().numpy().astype(np.float32)
    action_tuple = tuple(action.copy() for action in actions.reshape(-1, int(cfg.continuous_action_dim)))
    return (action_tuple, new_memory) if return_memory else action_tuple


def save_checkpoint_video(policy: nn.Module, cfg: PSGAILConfig, *, run_dir: str, round_idx: int, device: torch.device) -> str | None:
    if not bool(cfg.save_checkpoint_video):
        return None
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError:
        warnings.warn("imageio is not installed; skipping checkpoint video export.", stacklevel=2)
        return None
    video_dir = os.path.join(run_dir, str(cfg.checkpoint_video_dir))
    os.makedirs(video_dir, exist_ok=True)
    path = os.path.join(video_dir, f"round_{int(round_idx):04d}.mp4")
    env = make_training_env(cfg, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    was_training = policy.training
    policy.eval()
    try:
        base = env.unwrapped
        base.config["offscreen_rendering"] = True
        base.config["screen_width"] = int(cfg.checkpoint_video_width)
        base.config["screen_height"] = int(cfg.checkpoint_video_height)
        base.config["scaling"] = float(cfg.checkpoint_video_scaling)
        obs, _info = env.reset(seed=int(cfg.seed) + int(round_idx) * 1009)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        video_memory = (
            policy.initial_memory(
                len(policy_observations_from_flat(flatten_agent_observations(obs))),
                device=device,
                dtype=torch.float32,
            )
            if recurrent_policy_enabled(policy)
            else None
        )
        for _ in range(max(1, int(cfg.checkpoint_video_steps))):
            if recurrent_policy_enabled(policy):
                action, video_memory = _policy_action_tuple(
                    policy,
                    obs,
                    device=device,
                    cfg=cfg,
                    memory=video_memory,
                    return_memory=True,
                )
            else:
                action = _policy_action_tuple(policy, obs, device=device, cfg=cfg)
            obs, _reward, terminated, truncated, _info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            if terminated or truncated:
                break
    finally:
        env.close()
        if was_training:
            policy.train()
    if not frames:
        return None
    with imageio.get_writer(path, fps=max(1, int(cfg.policy_frequency))) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return path


def parse_args() -> tuple[PSGAILConfig, int, int]:
    defaults = PSGAILConfig(
        action_mode="continuous",
        run_name="simple_airl",
        policy_model="transformer",
        batch_size=4096,
        disc_batch_size=4096,
        discriminator_loss="wgan_gp",
        disc_learning_rate=4e-4,
        disc_updates_per_round=2,
        discriminator_replay_rounds=3,
        discriminator_replay_max_samples=120_000,
        normalize_gail_reward=True,
        allow_wgan_reward_normalization=True,
        transformer_temporal_module=True,
        disc_expert_label=0.8,
        disc_generator_label=0.2,
        validation_episodes=4,
    )
    parser = argparse.ArgumentParser(description="Lightweight continuous AIRL test trainer for unified NGSIM expert data.")
    for field in fields(PSGAILConfig):
        value = getattr(defaults, field.name)
        arg = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    parser.add_argument("--reward-batch-size", type=int, default=int(defaults.disc_batch_size))
    parser.add_argument("--airl-log-prob-batch-size", type=int, default=512)
    args = parser.parse_args()
    values = vars(args)
    reward_batch_size = int(values.pop("reward_batch_size"))
    airl_log_prob_batch_size = int(values.pop("airl_log_prob_batch_size"))
    return PSGAILConfig(**values), reward_batch_size, airl_log_prob_batch_size


def main() -> None:
    cfg, reward_batch_size, airl_log_prob_batch_size = parse_args()
    if str(cfg.action_mode).lower() != "continuous":
        raise ValueError("This AIRL test trainer currently supports --action-mode continuous only.")
    airl_objective = str(cfg.discriminator_loss).lower()
    if airl_objective not in {"airl", "airl_bce", "bce", "wgan_gp"}:
        raise ValueError(
            "train_simple_airl.py supports canonical BCE AIRL and WGAN-GP AIRL-style training. "
            f"Received discriminator_loss={cfg.discriminator_loss!r}."
        )
    if bool(getattr(cfg, "enable_scene_discriminator", False)):
        raise ValueError(
            "AIRL does not implement the auxiliary scene discriminator. "
            "Disable --enable-scene-discriminator for AIRL runs."
        )
    if bool(getattr(cfg, "enable_sequence_discriminator", False)):
        raise ValueError(
            "AIRL does not implement the auxiliary sequence discriminator. "
            "Disable --enable-sequence-discriminator for AIRL runs."
        )
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = os.path.abspath(os.path.join("logs", "airl", cfg.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    monitor = WandbMonitor(cfg, run_dir)
    monitor.start()

    env = None
    rollout_executor = None
    evaluation_executor = None
    try:
        expert = load_expert_transition_data(
            cfg.expert_data,
            max_samples=cfg.max_expert_samples,
            seed=cfg.seed,
            trajectory_frame=cfg.trajectory_frame,
        )
        env_cfg = config_for_round(cfg, 1)
        env = make_training_env(env_cfg)
        cfg.continuous_action_dim = infer_continuous_action_dim(env)
        env_cfg.continuous_action_dim = cfg.continuous_action_dim
        policy_obs_dim = infer_policy_obs_dim(env)
        critic_obs_dim = infer_critic_obs_dim(env, cfg, policy_obs_dim=policy_obs_dim)
        if policy_obs_dim != expert.policy_observations.shape[1]:
            raise RuntimeError(f"Expert/env observation mismatch: {expert.policy_observations.shape[1]} != {policy_obs_dim}.")
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
            centralized_critic=bool(cfg.centralized_critic),
            critic_obs_dim=critic_obs_dim,
            central_critic_pooling=str(cfg.central_critic_pooling),
            central_critic_max_vehicles=int(cfg.central_critic_max_vehicles),
            central_critic_attention_heads=int(cfg.central_critic_attention_heads),
        ).to(device)
        reward_model = AIRLReward(
            policy_obs_dim,
            int(cfg.continuous_action_dim),
            hidden_sizes=cfg.discriminator_hidden_sizes,
            dropout=float(cfg.discriminator_dropout),
            spectral_norm=bool(cfg.discriminator_spectral_norm),
        ).to(device)
        resume_checkpoint = str(getattr(cfg, "resume_checkpoint", "") or "").strip()
        if resume_checkpoint:
            checkpoint = load_airl_resume_checkpoint(
                resume_checkpoint=resume_checkpoint,
                policy=policy,
                reward_model=reward_model,
                device=device,
                allow_missing_reward_state=bool(cfg.allow_airl_resume_without_reward),
            )
            print(
                "resumed_checkpoint="
                f"{os.path.abspath(resume_checkpoint)} "
                f"round={checkpoint.get('round', 'unknown')}"
            )
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=cfg.disc_learning_rate)
        monitor.watch(policy, reward_model)
        rollout_executor = make_rollout_executor(cfg)
        evaluation_executor = make_evaluation_executor(cfg)
        current_env_signature = env_signature(env_cfg)

        print(f"Loaded expert folder: {os.path.abspath(cfg.expert_data)}")
        print(f"expert_obs={expert.policy_observations.shape} expert_actions={expert.actions_continuous_env.shape}")
        print(
            f"policy_obs_dim={policy_obs_dim} critic_obs_dim={critic_obs_dim} "
            f"centralized_critic={cfg.centralized_critic} "
            f"central_critic_max_vehicles={cfg.central_critic_max_vehicles} "
            f"central_critic_include_local_obs={cfg.central_critic_include_local_obs} "
            f"central_critic_pooling={cfg.central_critic_pooling} "
            f"central_critic_attention_heads={cfg.central_critic_attention_heads} "
            f"action_dim={cfg.continuous_action_dim} device={device} "
            f"policy_model={cfg.policy_model} transformer_layers={cfg.transformer_layers} "
            f"transformer_heads={cfg.transformer_heads} transformer_dropout={cfg.transformer_dropout} "
            f"airl_objective={cfg.discriminator_loss} wgan_gp_lambda={cfg.wgan_gp_lambda} "
            f"airl_policy_reward_mode={cfg.airl_policy_reward_mode} "
            f"reward_hidden={cfg.discriminator_hidden_sizes} "
            f"reward_dropout={cfg.discriminator_dropout} "
            f"reward_spectral_norm={cfg.discriminator_spectral_norm} "
            f"wgan_reward_center={cfg.wgan_reward_center} wgan_reward_clip={cfg.wgan_reward_clip} "
            f"wgan_reward_scale={cfg.wgan_reward_scale} "
            f"rollout_workers={cfg.num_rollout_workers} "
            f"rollout_worker_threads={cfg.rollout_worker_threads} "
            f"evaluation_workers={cfg.evaluation_num_workers} "
            f"evaluation_worker_threads={cfg.evaluation_worker_threads}"
        )
        for message in training_risk_warnings(cfg):
            print(f"training warning: {message}", flush=True)
        if int(cfg.bc_pretrain_epochs) > 0:
            print(
                "bc_pretrain="
                f"epochs={cfg.bc_pretrain_epochs} "
                f"batch={cfg.bc_pretrain_batch_size} "
                f"micro_batch={cfg.bc_pretrain_micro_batch_size} "
                f"lr={cfg.bc_pretrain_learning_rate} "
                f"val_fraction={cfg.bc_pretrain_validation_fraction} "
                f"eval_episodes={cfg.bc_pretrain_eval_episodes}"
            )
        if int(cfg.warmup_rounds) > 0 or int(cfg.vehicle_increase_warmup_rounds) > 0:
            print(
                "warmup="
                f"rounds={cfg.warmup_rounds} "
                f"vehicle_increase_rounds={cfg.vehicle_increase_warmup_rounds} "
                f"policy_lr={cfg.warmup_learning_rate or cfg.learning_rate}->{cfg.learning_rate} "
                f"reward_lr={cfg.warmup_disc_learning_rate or cfg.disc_learning_rate}->{cfg.disc_learning_rate} "
                f"entropy={cfg.warmup_entropy_coef if cfg.warmup_entropy_coef >= 0 else cfg.entropy_coef}->{cfg.entropy_coef} "
                f"clip={cfg.warmup_clip_range or cfg.clip_range}->{cfg.clip_range}"
            )
        if float(cfg.policy_bc_regularization_coef) > 0.0:
            print(
                "policy_bc_regularization="
                f"coef={cfg.policy_bc_regularization_coef} "
                f"final={cfg.policy_bc_regularization_final_coef} "
                f"decay_rounds={cfg.policy_bc_regularization_decay_rounds}"
            )
        if cfg.controlled_vehicle_curriculum:
            print(
                "controlled_vehicle_curriculum="
                f"initial={cfg.initial_controlled_vehicles:.4f} "
                f"final={cfg.final_controlled_vehicles:.4f} "
                f"rounds={cfg.controlled_vehicle_curriculum_rounds} "
                f"increment_rounds={cfg.controlled_vehicle_increment_rounds} "
                f"schedule={cfg.controlled_vehicle_schedule or 'linear'}"
            )
        if int(cfg.initial_rollout_target_agent_steps) > 0 or int(cfg.final_rollout_target_agent_steps) > 0:
            print(
                "rollout_target_agent_steps_curriculum="
                f"initial={cfg.initial_rollout_target_agent_steps} "
                f"final={cfg.final_rollout_target_agent_steps} "
                f"rounds={cfg.rollout_target_agent_steps_curriculum_rounds} "
                f"schedule={cfg.rollout_target_agent_steps_schedule or 'linear'}"
            )
        if float(cfg.initial_gamma) > 0.0 or float(cfg.final_gamma) > 0.0:
            print(
                "gamma_curriculum="
                f"initial={cfg.initial_gamma or cfg.gamma:.4f} "
                f"final={cfg.final_gamma or cfg.gamma:.4f} "
                f"rounds={cfg.gamma_curriculum_rounds} "
                f"schedule={cfg.gamma_schedule or 'linear'}"
            )
        if cfg.max_episode_steps_schedule:
            print(f"max_episode_steps_schedule={cfg.max_episode_steps_schedule}")

        if int(cfg.bc_pretrain_epochs) > 0:
            bc_stats = behavior_clone_pretrain(policy, expert, cfg, device)
            bc_eval_stats = evaluate_policy_survival(
                policy,
                env_cfg,
                device,
                episodes=int(cfg.bc_pretrain_eval_episodes),
                seed_offset=50_000,
            )
            print(
                "[bc final] "
                f"train_mse={bc_stats['bc/train_mse']:.6f} "
                f"train_mae={bc_stats['bc/train_mae']:.6f} "
                f"val_mse={bc_stats['bc/val_mse']:.6f} "
                f"val_mae={bc_stats['bc/val_mae']:.6f}"
            )
            if bc_eval_stats:
                print(
                    "[bc env_eval] "
                    f"episodes={int(bc_eval_stats['bc_eval/episodes'])} "
                    f"ep_len={bc_eval_stats['bc_eval/mean_episode_length']:.1f}"
                    f"[{int(bc_eval_stats['bc_eval/min_episode_length'])},"
                    f"{int(bc_eval_stats['bc_eval/max_episode_length'])}] "
                    f"crash_eps={int(bc_eval_stats['bc_eval/crash_episodes'])} "
                    f"offroad_eps={int(bc_eval_stats['bc_eval/offroad_episodes'])} "
                    f"veh={bc_eval_stats['bc_eval/mean_controlled_vehicles']:.1f}/"
                    f"{bc_eval_stats['bc_eval/mean_road_vehicles']:.1f}"
                )
                min_mean_len = float(cfg.bc_pretrain_min_mean_episode_length)
                if (
                    min_mean_len > 0.0
                    and bc_eval_stats["bc_eval/mean_episode_length"] < min_mean_len
                ):
                    message = (
                        "BC env validation did not reach the requested mean episode length: "
                        f"{bc_eval_stats['bc_eval/mean_episode_length']:.1f} < {min_mean_len:.1f}."
                    )
                    if bool(cfg.bc_pretrain_abort_on_failed_eval):
                        raise RuntimeError(message)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
            monitor.log({**bc_stats, **bc_eval_stats}, step=0)
            bc_path = os.path.join(ckpt_dir, "bc_pretrained.pt")
            torch.save(
                {
                    "round": 0,
                    "policy_state_dict": policy.state_dict(),
                    "reward_state_dict": reward_model.state_dict(),
                    "expert_metadata": expert.metadata,
                    "config": vars(cfg),
                    "round_config": vars(env_cfg),
                    "bc_stats": bc_stats,
                    "bc_eval_stats": bc_eval_stats,
                },
                bc_path,
            )
            monitor.save(bc_path)

        last_checkpoint_video_path = None
        last_checkpoint_video_round = 0
        airl_replay: list[AIRLReplayEntry] = []
        psro_policy_archive: list[dict[str, torch.Tensor]] = []
        best_validation_score = float("-inf")
        best_validation_round = 0
        best_path = os.path.join(run_dir, "best.pt")
        last_validation_stress_round = 0
        if bool(getattr(cfg, "psro_lite", False)):
            append_policy_archive(psro_policy_archive, policy, cfg)
            print(
                "psro_lite="
                f"enabled archive_every={cfg.psro_archive_every} "
                f"archive_size={cfg.psro_archive_size} "
                f"after_jump_rounds={cfg.psro_mixture_after_jump_rounds} "
                f"current_fraction={cfg.psro_current_policy_fraction:.3f}"
            )
        previous_controlled_vehicles = None
        last_vehicle_jump_round = None
        for round_idx in range(1, int(cfg.total_rounds) + 1):
            round_cfg = config_for_round(cfg, round_idx)
            round_cfg.continuous_action_dim = cfg.continuous_action_dim
            if previous_controlled_vehicles is not None and (
                float(round_cfg.percentage_controlled_vehicles)
                > float(previous_controlled_vehicles) + 1.0e-9
            ):
                last_vehicle_jump_round = int(round_idx)
            previous_controlled_vehicles = float(round_cfg.percentage_controlled_vehicles)
            psro_window = max(0, int(getattr(round_cfg, "psro_mixture_after_jump_rounds", 0)))
            psro_archive_for_round = (
                psro_policy_archive
                if (
                    bool(getattr(round_cfg, "psro_lite", False))
                    and psro_policy_archive
                    and last_vehicle_jump_round is not None
                    and int(round_idx) - int(last_vehicle_jump_round) < psro_window
                )
                else None
            )
            set_optimizer_lr(policy_optimizer, float(round_cfg.learning_rate))
            set_optimizer_lr(reward_optimizer, float(round_cfg.disc_learning_rate))
            if env_signature(round_cfg) != current_env_signature:
                env.close()
                env = make_training_env(round_cfg)
                current_env_signature = env_signature(round_cfg)
            collected_rollout = collect_round_rollouts(
                env,
                policy,
                round_cfg,
                device,
                policy_obs_dim,
                critic_obs_dim,
                round_idx=round_idx,
                rollout_executor=rollout_executor,
                archive_policy_state_dicts=psro_archive_for_round,
            )
            rollout = subsample_rollout_for_training(
                collected_rollout,
                round_cfg,
                seed=int(round_cfg.seed) + int(round_idx) * 104729,
            )
            rollout_was_subsampled = int(rollout.num_agent_steps) != int(collected_rollout.num_agent_steps)
            (
                reward_generator_obs,
                reward_generator_actions,
                reward_generator_next_obs,
                reward_generator_dones,
                reward_generator_log_probs,
                reward_generator_trajectory_ids,
                reward_generator_timesteps,
            ) = concat_airl_replay(
                rollout,
                airl_replay,
                round_cfg,
                seed=int(round_cfg.seed) + int(round_idx) * 65537,
                preserve_recurrent_context=recurrent_policy_enabled(policy),
            )
            reward_stats = update_reward_model(
                reward_model,
                reward_optimizer,
                policy,
                expert.policy_observations,
                expert.actions_continuous_env,
                expert.next_policy_observations,
                expert.dones,
                reward_generator_obs,
                reward_generator_actions,
                reward_generator_next_obs,
                reward_generator_dones,
                round_cfg,
                device,
                reward_batch_size=reward_batch_size,
                log_prob_batch_size=airl_log_prob_batch_size,
                expert_trajectory_ids=expert.trajectory_ids,
                expert_timesteps=expert.timesteps,
                generator_log_probs=reward_generator_log_probs,
                generator_trajectory_ids=reward_generator_trajectory_ids,
                generator_timesteps=reward_generator_timesteps,
            )
            rollout = refresh_airl_rewards(rollout, reward_model, round_cfg, device)
            policy_stats = update_policy(
                policy,
                policy_optimizer,
                rollout,
                round_cfg,
                device,
                expert_policy_observations=expert.policy_observations,
                expert_actions=expert.actions_continuous_env,
            )
            append_airl_replay(airl_replay, rollout, round_cfg, round_idx=round_idx)
            print(
                f"[round {round_idx:04d}] env_steps={collected_rollout.num_env_steps} "
                f"agent_steps={collected_rollout.num_agent_steps} "
                f"train_steps={rollout.num_agent_steps} episodes={collected_rollout.num_episodes} "
                f"ep_len={collected_rollout.mean_episode_length:.1f} "
                f"[{collected_rollout.min_episode_length}-{collected_rollout.max_episode_length}] "
                f"term/trunc={collected_rollout.num_terminated}/{collected_rollout.num_truncated} "
                f"crash/offroad={collected_rollout.num_crash_events}/{collected_rollout.num_offroad_events} "
                f"ctrl_frac={round_cfg.percentage_controlled_vehicles:.4f} "
                f"veh={collected_rollout.mean_controlled_vehicles:.1f}/{collected_rollout.mean_road_vehicles:.1f} "
                + (
                    f"psro_current={collected_rollout.psro_current_fraction:.3f} "
                    f"archive={len(psro_archive_for_round or [])} "
                    if collected_rollout.psro_active
                    else ""
                )
                + (
                f"reward_loss={reward_stats['reward_loss']:.4f} "
                )
                + (
                    f"gap={reward_stats['critic_gap']:.4f} "
                    f"wgan_loss={reward_stats['wgan_loss']:.4f} "
                    f"gp={reward_stats['gradient_penalty']:.4f} "
                    if str(round_cfg.discriminator_loss).lower() == "wgan_gp"
                    else ""
                )
                + (
                f"expert_acc={reward_stats['expert_acc']:.3f} gen_acc={reward_stats['gen_acc']:.3f} "
                f"policy_loss={policy_stats['policy_loss']:.4f} value_loss={policy_stats['value_loss']:.4f} "
                + (
                    f"bc_reg={policy_stats['bc_regularization_loss']:.6f} "
                    f"bc_coef={policy_stats['bc_regularization_coef']:.4f} "
                    if float(policy_stats["bc_regularization_coef"]) > 0.0
                    else ""
                )
                + (
                f"lr={round_cfg.learning_rate:.2e}/{round_cfg.disc_learning_rate:.2e} "
                f"kl={policy_stats['approx_kl']:.5f} entropy={policy_stats['entropy']:.4f} "
                f"log_std={policy_stats['log_std_mean']:.3f} "
                f"post_kl={policy_stats['post_update_approx_kl']:.5f} "
                f"clip={policy_stats['clip_fraction']:.3f} "
                f"airl_reward={reward_stats['expert_reward']:.3f}/{reward_stats['gen_reward']:.3f} "
                f"reward={float(rollout.rewards.mean()):.4f}"
                )
                )
            )
            selected_action_valid = float("nan")
            mean_available_actions = float("nan")
            if (
                str(round_cfg.action_mode).lower() != "continuous"
                and rollout.action_masks.size
                and rollout.actions.ndim == 1
            ):
                action_indices = rollout.actions.astype(np.int64, copy=False)
                row_indices = np.arange(len(action_indices), dtype=np.int64)
                in_range = (action_indices >= 0) & (action_indices < rollout.action_masks.shape[1])
                selected_valid = np.zeros(len(action_indices), dtype=bool)
                selected_valid[in_range] = rollout.action_masks[row_indices[in_range], action_indices[in_range]]
                selected_action_valid = float(selected_valid.mean()) if selected_valid.size else float("nan")
                mean_available_actions = float(rollout.action_masks.sum(axis=1).mean())
            metrics = {
                "round": round_idx,
                "rollout/env_steps": collected_rollout.num_env_steps,
                "rollout/agent_steps": collected_rollout.num_agent_steps,
                "rollout/training_agent_steps": rollout.num_agent_steps,
                "rollout/training_subsampled": int(rollout_was_subsampled),
                "rollout/episodes": collected_rollout.num_episodes,
                "rollout/terminated": collected_rollout.num_terminated,
                "rollout/truncated": collected_rollout.num_truncated,
                "rollout/crash_episodes": collected_rollout.num_crash_events,
                "rollout/offroad_episodes": collected_rollout.num_offroad_events,
                "rollout/crash_events": collected_rollout.num_crash_events,
                "rollout/offroad_events": collected_rollout.num_offroad_events,
                "rollout/crash_agent_fraction": collected_rollout.crash_agent_fraction,
                "rollout/offroad_agent_fraction": collected_rollout.offroad_agent_fraction,
                "rollout/mean_episode_length": collected_rollout.mean_episode_length,
                "rollout/min_episode_length": collected_rollout.min_episode_length,
                "rollout/max_episode_length": collected_rollout.max_episode_length,
                "rollout/unique_episode_names": collected_rollout.unique_episode_names,
                "rollout/controlled_vehicle_fraction": float(round_cfg.percentage_controlled_vehicles),
                "rollout/mean_controlled_vehicles": collected_rollout.mean_controlled_vehicles,
                "rollout/mean_road_vehicles": collected_rollout.mean_road_vehicles,
                "psro/active": int(bool(collected_rollout.psro_active)),
                "psro/archive_size": int(len(psro_policy_archive)),
                "psro/archive_used": int(len(psro_archive_for_round or [])),
                "psro/current_decisions": int(collected_rollout.psro_current_decisions),
                "psro/archive_decisions": int(collected_rollout.psro_archive_decisions),
                "psro/current_fraction": float(collected_rollout.psro_current_fraction),
                "psro/last_vehicle_jump_round": int(last_vehicle_jump_round or 0),
                "rollout/scene_samples": int(len(rollout.scene_features)),
                "rollout/sequence_samples": int(len(rollout.sequence_features)),
                "rollout/mean_reward": float(rollout.rewards.mean()),
                "rollout/mean_gail_reward": float(rollout.rewards.mean()),
                "rollout/mean_airl_reward": float(rollout.rewards.mean()),
                "rollout/mean_raw_gail_reward": rollout.mean_raw_gail_reward,
                "rollout/mean_raw_airl_reward": rollout.mean_raw_gail_reward,
                "rollout/mean_normalized_gail_reward": rollout.mean_normalized_gail_reward,
                "rollout/mean_normalized_airl_reward": rollout.mean_normalized_gail_reward,
                "rollout/mean_env_penalty": rollout.mean_env_penalty,
                "rollout/reward_std": float(rollout.rewards.std()),
                "rollout/raw_gail_reward_std": float(rollout.gail_rewards_raw.std()),
                "rollout/raw_airl_reward_std": float(rollout.gail_rewards_raw.std()),
                "rollout/normalized_gail_reward_std": float(rollout.gail_rewards_normalized.std()),
                "rollout/normalized_airl_reward_std": float(rollout.gail_rewards_normalized.std()),
                "rollout/action_mean": float(rollout.actions.mean()),
                "rollout/action_std": float(rollout.actions.std()),
                "rollout/selected_action_valid_fraction": selected_action_valid,
                "rollout/mean_available_actions": mean_available_actions,
                "airl/reward_loss": reward_stats["reward_loss"],
                "airl/bce_loss": reward_stats["bce_loss"],
                "airl/wgan_loss": reward_stats["wgan_loss"],
                "airl/gradient_penalty": reward_stats["gradient_penalty"],
                "airl/critic_gap": reward_stats["critic_gap"],
                "airl/expert_acc": reward_stats["expert_acc"],
                "airl/gen_acc": reward_stats["gen_acc"],
                "airl/expert_reward": reward_stats["expert_reward"],
                "airl/gen_reward": reward_stats["gen_reward"],
                "discriminator/loss": reward_stats["reward_loss"],
                "discriminator/bce_loss": reward_stats["bce_loss"],
                "discriminator/wgan_loss": reward_stats["wgan_loss"],
                "discriminator/gradient_penalty": reward_stats["gradient_penalty"],
                "discriminator/critic_gap": reward_stats["critic_gap"],
                "discriminator/expert_acc": reward_stats["expert_acc"],
                "discriminator/gen_acc": reward_stats["gen_acc"],
                "discriminator/expert_reward": reward_stats["expert_reward"],
                "discriminator/gen_reward": reward_stats["gen_reward"],
                "discriminator/replay_rounds": int(len(airl_replay)),
                "discriminator/current_generator_samples": int(rollout.num_agent_steps),
                "discriminator/train_generator_samples": int(len(reward_generator_obs)),
                "train/discriminator_loss_type_wgan_gp": int(
                    str(round_cfg.discriminator_loss).lower() == "wgan_gp"
                ),
                "train/wgan_gp_lambda": float(round_cfg.wgan_gp_lambda),
                "train/reward_spectral_norm": int(bool(round_cfg.discriminator_spectral_norm)),
                "train/discriminator_spectral_norm": int(bool(round_cfg.discriminator_spectral_norm)),
                "train/policy_obs_dim": policy_obs_dim,
                "train/critic_obs_dim": critic_obs_dim,
                "train/centralized_critic": int(bool(round_cfg.centralized_critic)),
                "train/central_critic_pooling_attention": int(
                    str(round_cfg.central_critic_pooling).lower() in {"attention", "attn"}
                ),
                "train/central_critic_attention_heads": int(round_cfg.central_critic_attention_heads),
                "train/wgan_reward_center": int(bool(round_cfg.wgan_reward_center)),
                "train/wgan_reward_clip": float(round_cfg.wgan_reward_clip),
                "train/wgan_reward_scale": float(round_cfg.wgan_reward_scale),
                "train/wgan_reward_norm_min_std": float(getattr(round_cfg, "wgan_reward_norm_min_std", 1.0e-3)),
                "train/wgan_reward_norm_clip": float(getattr(round_cfg, "wgan_reward_norm_clip", 0.0)),
                "train/airl_policy_reward_mode_discriminator": int(
                    str(getattr(round_cfg, "airl_policy_reward_mode", "shaped")).lower()
                    == "discriminator"
                ),
                "train/airl_policy_reward_mode_shaped": int(
                    str(getattr(round_cfg, "airl_policy_reward_mode", "shaped")).lower()
                    == "shaped"
                ),
                "train/airl_policy_reward_mode_reward": int(
                    str(getattr(round_cfg, "airl_policy_reward_mode", "shaped")).lower()
                    == "reward"
                ),
                "train/normalize_gail_reward_requested": int(bool(round_cfg.normalize_gail_reward)),
                "train/allow_wgan_reward_normalization": int(
                    bool(getattr(round_cfg, "allow_wgan_reward_normalization", False))
                ),
                "train/normalize_gail_reward_effective": int(
                    should_normalize_gail_reward(round_cfg)
                ),
                "policy/loss": policy_stats["policy_loss"],
                "policy/value_loss": policy_stats["value_loss"],
                "policy/bc_regularization_loss": policy_stats["bc_regularization_loss"],
                "policy/bc_regularization_coef": policy_stats["bc_regularization_coef"],
                "policy/entropy": policy_stats["entropy"],
                "policy/approx_kl": policy_stats["approx_kl"],
                "policy/post_update_approx_kl": policy_stats["post_update_approx_kl"],
                "policy/clip_fraction": policy_stats["clip_fraction"],
                "policy/ratio_mean": policy_stats["ratio_mean"],
                "policy/ratio_std": policy_stats["ratio_std"],
                "policy/post_update_ratio_mean": policy_stats["post_update_ratio_mean"],
                "policy/post_update_ratio_std": policy_stats["post_update_ratio_std"],
                "policy/advantage_mean": policy_stats["advantage_mean"],
                "policy/advantage_std": policy_stats["advantage_std"],
                "policy/ppo_micro_batch_size": policy_stats["ppo_micro_batch_size"],
                "policy/log_std_mean": policy_stats["log_std_mean"],
                "policy/action_std_param_mean": policy_stats["action_std_param_mean"],
                "policy/log_std_delta": policy_stats["log_std_delta"],
                "policy/action_std_param_delta": policy_stats["action_std_param_delta"],
                "train/policy_learning_rate": float(round_cfg.learning_rate),
                "train/reward_learning_rate": float(round_cfg.disc_learning_rate),
                "train/disc_learning_rate": float(round_cfg.disc_learning_rate),
                "train/entropy_coef": float(round_cfg.entropy_coef),
                "train/clip_range": float(round_cfg.clip_range),
                "train/reward_updates_per_round": int(round_cfg.disc_updates_per_round),
                "train/disc_updates_per_round": int(round_cfg.disc_updates_per_round),
                "train/expert_samples": int(expert.policy_observations.shape[0]),
                "train/reward_batch_size": int(reward_batch_size),
                "train/airl_log_prob_batch_size": int(airl_log_prob_batch_size),
                "train/rollout_workers": int(round_cfg.num_rollout_workers),
                "train/rollout_worker_threads": int(round_cfg.rollout_worker_threads),
                "train/evaluation_workers": int(round_cfg.evaluation_num_workers),
                "train/evaluation_worker_threads": int(round_cfg.evaluation_worker_threads),
                "train/discriminator_replay_rounds": int(getattr(round_cfg, "discriminator_replay_rounds", 0)),
                "train/discriminator_replay_max_samples": int(
                    getattr(round_cfg, "discriminator_replay_max_samples", 0)
                ),
                "train/transformer_temporal_module": int(
                    bool(getattr(round_cfg, "transformer_temporal_module", False))
                ),
            }
            for stat_key, stat_value in policy_stats.items():
                if stat_key.startswith("perf/"):
                    metrics[stat_key] = float(stat_value)
                elif stat_key.startswith("transformer_"):
                    metrics[f"policy/{stat_key}"] = float(stat_value)
            monitor.log(metrics, step=round_idx)
            if int(getattr(cfg, "validation_every", 0)) > 0 and round_idx % int(cfg.validation_every) == 0:
                val_metrics = evaluate_policy_matched_trajectories(
                    policy,
                    round_cfg,
                    device,
                    split=str(getattr(cfg, "validation_prebuilt_split", "val")),
                    episodes=int(getattr(cfg, "validation_episodes", 0)),
                    prefix="validation",
                    evaluation_executor=evaluation_executor,
                )
                if val_metrics:
                    val_metrics, val_cost, val_score = scored_validation_metrics(
                        val_metrics,
                        cfg,
                        prefix="validation",
                    )
                    monitor.log(val_metrics, step=round_idx)
                    improved = (
                        bool(getattr(cfg, "save_best_checkpoint", True))
                        and np.isfinite(val_score)
                        and val_score > best_validation_score + float(getattr(cfg, "validation_min_delta", 0.0))
                    )
                    if improved:
                        best_validation_score = float(val_score)
                        best_validation_round = int(round_idx)
                        torch.save(
                            best_checkpoint_payload(
                                airl_checkpoint_payload(
                                    round_idx=round_idx,
                                    policy=policy,
                                    reward_model=reward_model,
                                    expert_metadata=expert.metadata,
                                    cfg=cfg,
                                    round_cfg=round_cfg,
                                ),
                                round_idx=round_idx,
                                validation_metrics=val_metrics,
                                validation_score=val_score,
                                validation_cost=val_cost,
                            ),
                            best_path,
                        )
                        monitor.save(best_path)
                    print(
                        matched_validation_summary("validation", f"{round_idx:04d}", val_metrics)
                        + f" best={best_validation_score:.4f}@{best_validation_round}"
                    )
            stress_every = int(getattr(cfg, "validation_stress_every", 0))
            if (
                stress_every > 0
                and int(getattr(cfg, "validation_stress_episodes", 0)) > 0
                and round_idx % stress_every == 0
            ):
                stress_metrics = evaluate_policy_matched_trajectories(
                    policy,
                    round_cfg,
                    device,
                    split=str(getattr(cfg, "validation_prebuilt_split", "val")),
                    episodes=int(getattr(cfg, "validation_stress_episodes", 0)),
                    prefix="validation_stress",
                    evaluation_executor=evaluation_executor,
                )
                if stress_metrics:
                    stress_metrics, _stress_cost, _stress_score = scored_validation_metrics(
                        stress_metrics,
                        cfg,
                        prefix="validation_stress",
                    )
                    monitor.log(stress_metrics, step=round_idx)
                    print(matched_validation_summary("validation_stress", f"{round_idx:04d}", stress_metrics))
                    last_validation_stress_round = int(round_idx)
            if cfg.checkpoint_every > 0 and round_idx % int(cfg.checkpoint_every) == 0:
                checkpoint_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}.pt")
                torch.save(
                    airl_checkpoint_payload(
                        round_idx=round_idx,
                        policy=policy,
                        reward_model=reward_model,
                        expert_metadata=expert.metadata,
                        cfg=cfg,
                        round_cfg=round_cfg,
                    ),
                    checkpoint_path,
                )
                monitor.save(checkpoint_path)
            if should_save_checkpoint_video(round_cfg, round_idx):
                video_path = save_checkpoint_video(policy, round_cfg, run_dir=run_dir, round_idx=round_idx, device=device)
                if video_path is not None:
                    last_checkpoint_video_path = video_path
                    last_checkpoint_video_round = round_idx
                    monitor.log_video("checkpoint/policy_video", video_path, step=round_idx, fps=int(round_cfg.policy_frequency))
            if (
                bool(getattr(cfg, "psro_lite", False))
                and int(getattr(cfg, "psro_archive_every", 0)) > 0
                and round_idx % int(cfg.psro_archive_every) == 0
            ):
                append_policy_archive(psro_policy_archive, policy, cfg)

        final_round = int(cfg.total_rounds)
        final_round_cfg = config_for_round(cfg, final_round)
        final_path = os.path.join(run_dir, "final.pt")
        torch.save(
            airl_checkpoint_payload(
                round_idx=final_round,
                policy=policy,
                reward_model=reward_model,
                expert_metadata=expert.metadata,
                cfg=cfg,
                round_cfg=final_round_cfg,
            ),
            final_path,
        )
        monitor.save(final_path)
        final_video_path = (
            last_checkpoint_video_path
            if last_checkpoint_video_round == final_round
            else None
        )
        if final_video_path is None:
            final_video_path = save_checkpoint_video(
                policy,
                final_round_cfg,
                run_dir=run_dir,
                round_idx=final_round,
                device=device,
            )
        if final_video_path is not None:
            monitor.log_video(
                "checkpoint/final_policy_video",
                final_video_path,
                step=final_round,
                fps=int(cfg.policy_frequency),
            )
        if (
            int(getattr(cfg, "validation_stress_episodes", 0)) > 0
            and last_validation_stress_round != final_round
        ):
            stress_metrics = evaluate_policy_matched_trajectories(
                policy,
                final_round_cfg,
                device,
                split=str(getattr(cfg, "validation_prebuilt_split", "val")),
                episodes=int(getattr(cfg, "validation_stress_episodes", 0)),
                prefix="validation_stress",
                evaluation_executor=evaluation_executor,
            )
            if stress_metrics:
                stress_metrics, _stress_cost, _stress_score = scored_validation_metrics(
                    stress_metrics,
                    cfg,
                    prefix="validation_stress",
                )
                monitor.log(stress_metrics, step=final_round)
                print(matched_validation_summary("validation_stress", "final", stress_metrics))
        if int(getattr(cfg, "test_episodes", 0)) > 0:
            test_metrics = evaluate_policy_matched_trajectories(
                policy,
                final_round_cfg,
                device,
                split=str(getattr(cfg, "test_prebuilt_split", "test")),
                episodes=int(getattr(cfg, "test_episodes", 0)),
                prefix="test",
                evaluation_executor=evaluation_executor,
            )
            if test_metrics:
                monitor.log(test_metrics, step=final_round)
                print(
                    f"[test final] "
                    f"episodes={test_metrics.get('test/episodes', 0):.0f} "
                    f"rmse_pos_20s={test_metrics.get('test/rmse_position_20s', float('nan')):.4f} "
                    f"collision={test_metrics.get('test/vehicle_crash_rate', test_metrics.get('test/collision_rate', 0.0)):.4f} "
                    f"offroad={test_metrics.get('test/vehicle_offroad_rate', test_metrics.get('test/offroad_duration_rate', 0.0)):.4f} "
                    f"hard_brake={test_metrics.get('test/hard_brake_rate', 0.0):.4f}"
                )
    finally:
        if rollout_executor is not None:
            rollout_executor.shutdown(wait=True, cancel_futures=True)
        if evaluation_executor is not None:
            evaluation_executor.shutdown(wait=True, cancel_futures=True)
        if env is not None:
            env.close()
        monitor.finish()
    print(f"Saved final checkpoint under: {run_dir}")


if __name__ == "__main__":
    main()
