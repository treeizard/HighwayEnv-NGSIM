"""Rollout collection, merging, and sampling utilities for PS-GAIL training."""

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
    _memory_cache_from_state,
    _memory_storage_dtype,
    _sample_policy_actions,
    _update_memory_state,
    central_critic_observation_dim,
    central_critic_observations,
    centralized_critic_enabled,
    discrete_action_masks_from_env,
    policy_action_dim,
    policy_distribution_values_memory,
    recurrent_policy_enabled,
)
from .rewards import (
    _transition_array,
    action_conditioned_features,
    combine_primary_env_challenge_rewards,
    compute_returns_and_advantages,
    discriminator_input_mode,
    discriminator_reward,
    player_challenge_payoff,
    player_challenge_pressure_from_metric,
    sequence_rewards_to_transition_rewards,
    shape_adversarial_rewards,
)
from .types import AgentTransition, RolloutBatch

class _RolloutPerfProfiler:
    enabled = os.environ.get("HIGHWAY_ENV_OBS_PROFILE", "").lower() in {"1", "true", "yes", "on"}
    report_every = max(1, int(os.environ.get("HIGHWAY_ENV_OBS_PROFILE_EVERY", "1000")))
    counts: defaultdict[str, int] = defaultdict(int)
    totals: defaultdict[str, float] = defaultdict(float)
    events = 0

    @classmethod
    def record(cls, name: str, elapsed: float) -> None:
        if not cls.enabled:
            return
        cls.counts[name] += 1
        cls.totals[name] += float(elapsed)
        if name == "policy_inference":
            cls.events += 1
            if cls.events % cls.report_every == 0:
                cls.report()

    @classmethod
    def report(cls) -> None:
        if not cls.enabled:
            return
        parts = []
        for name in sorted(cls.totals):
            count = max(1, int(cls.counts[name]))
            total_ms = 1000.0 * float(cls.totals[name])
            parts.append(f"{name}={total_ms / count:.3f}ms avg ({total_ms:.1f}ms/{count})")
        print("[rollout_profile] " + " ".join(parts), flush=True)

def _actions_to_rollout_array(transitions: list[AgentTransition], cfg: PSGAILConfig) -> np.ndarray:
    if _is_continuous(cfg):
        return np.stack(
            [np.asarray(tr.action, dtype=np.float32) for tr in transitions],
            axis=0,
        ).astype(np.float32)
    return np.asarray([tr.action for tr in transitions], dtype=np.int64)

def _assign_psro_sources(
    num_agents: int,
    *,
    num_archive_policies: int,
    current_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    num_agents = int(num_agents)
    if num_agents <= 0 or int(num_archive_policies) <= 0:
        return np.zeros(max(0, num_agents), dtype=np.int16)
    current_fraction = min(1.0, max(0.0, float(current_fraction)))
    current_count = int(round(current_fraction * num_agents))
    current_count = min(num_agents, max(1, current_count))
    sources = np.zeros(num_agents, dtype=np.int16)
    if current_count >= num_agents:
        return sources
    archived_indices = rng.choice(num_agents, size=num_agents - current_count, replace=False)
    sources[archived_indices] = rng.integers(
        1,
        int(num_archive_policies) + 1,
        size=len(archived_indices),
        dtype=np.int16,
    )
    return sources

def _mixed_policy_actions(
    current_policy: nn.Module,
    archive_policies: list[nn.Module],
    source_ids: np.ndarray,
    obs_tensor: torch.Tensor,
    critic_obs_tensor: torch.Tensor,
    action_mask_tensor: torch.Tensor | None,
    cfg: PSGAILConfig,
    *,
    agent_keys: list[tuple[int, int]] | None = None,
    current_memory: torch.Tensor | None = None,
    archive_memory_states: list[dict[tuple[int, int], list[np.ndarray]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    current_dist, current_values, current_step_memory = policy_distribution_values_memory(
        current_policy,
        obs_tensor,
        cfg,
        action_mask_tensor,
        critic_obs_tensor=critic_obs_tensor,
        memory=current_memory,
        return_memory=True,
    )
    actions, log_probs = _sample_policy_actions(current_dist, cfg)
    if not archive_policies:
        return actions, log_probs, current_values, current_step_memory

    source_tensor = torch.as_tensor(source_ids, dtype=torch.long, device=obs_tensor.device)
    archive_memory_states = archive_memory_states if archive_memory_states is not None else []
    agent_keys = list(agent_keys or [])
    for archive_idx, archive_policy in enumerate(archive_policies, start=1):
        mask = source_tensor == int(archive_idx)
        if not bool(mask.any()):
            continue
        archive_masks = action_mask_tensor[mask] if action_mask_tensor is not None else None
        archive_memory = None
        selected_keys: list[tuple[int, int]] = []
        if agent_keys and recurrent_policy_enabled(archive_policy):
            selected_indices = torch.nonzero(mask, as_tuple=False).reshape(-1).detach().cpu().numpy()
            selected_keys = [agent_keys[int(index)] for index in selected_indices.tolist()]
            while len(archive_memory_states) < archive_idx:
                archive_memory_states.append({})
            archive_memory = _memory_cache_from_state(
                archive_policy,
                cfg,
                selected_keys,
                archive_memory_states[archive_idx - 1],
                obs_tensor.device,
            )
        archive_dist, _archive_values, archive_step_memory = policy_distribution_values_memory(
            archive_policy,
            obs_tensor[mask],
            cfg,
            archive_masks,
            critic_obs_tensor=critic_obs_tensor[mask],
            memory=archive_memory,
            return_memory=True,
        )
        archive_actions, _archive_log_probs = _sample_policy_actions(archive_dist, cfg)
        actions = actions.clone()
        actions[mask] = archive_actions
        if selected_keys and archive_step_memory is not None:
            _update_memory_state(
                archive_policy,
                cfg,
                selected_keys,
                archive_memory_states[archive_idx - 1],
                archive_step_memory,
            )
    return actions, log_probs, current_values, current_step_memory

def _target_rollout_episodes(cfg: PSGAILConfig) -> int:
    target = max(1, int(cfg.rollout_min_episodes))
    if bool(getattr(cfg, "rollout_full_episodes", True)):
        max_episode_steps = int(cfg.max_episode_steps)
        if max_episode_steps > 0:
            target = max(target, int(np.ceil(max(1, int(cfg.rollout_steps)) / max_episode_steps)))
    return target

def _estimate_controlled_vehicles_for_target(cfg: PSGAILConfig) -> int:
    configured = float(getattr(cfg, "percentage_controlled_vehicles", 1.0))
    if configured >= 1.0:
        return max(1, int(round(configured)))
    return 1

def _target_aware_rollout_episodes(
    cfg: PSGAILConfig,
    *,
    remaining_agent_steps: int,
) -> int:
    if (
        not bool(getattr(cfg, "rollout_full_episodes", True))
        or not bool(getattr(cfg, "rollout_target_aware_episodes", True))
        or int(remaining_agent_steps) <= 0
    ):
        return _target_rollout_episodes(cfg)
    max_episode_steps = max(1, int(cfg.max_episode_steps))
    estimated_controlled = _estimate_controlled_vehicles_for_target(cfg)
    estimated_agent_steps_per_episode = max(1, max_episode_steps * estimated_controlled)
    safety_factor = max(1.0, float(getattr(cfg, "rollout_target_episode_safety_factor", 1.0)))
    target_episodes = int(
        np.ceil(
            safety_factor
            * int(remaining_agent_steps)
            / float(estimated_agent_steps_per_episode)
        )
    )
    diversity_floor = max(1, int(getattr(cfg, "rollout_target_min_episodes", 1)))
    return max(diversity_floor, target_episodes)

def refresh_rollout_rewards(
    rollout: RolloutBatch,
    discriminator: nn.Module | None,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    scene_discriminator: nn.Module | None = None,
    sequence_discriminator: nn.Module | None = None,
    discriminator_normalizer: tuple[np.ndarray, np.ndarray] | None = None,
    scene_discriminator_normalizer: tuple[np.ndarray, np.ndarray] | None = None,
    sequence_discriminator_normalizer: tuple[np.ndarray, np.ndarray] | None = None,
) -> RolloutBatch:
    feature_clip = float(getattr(cfg, "discriminator_feature_clip", 0.0))
    if discriminator is None:
        combined_raw_gail_rewards = np.zeros(int(rollout.num_agent_steps), dtype=np.float32)
    else:
        raw_gail_rewards = discriminator_reward(
            discriminator,
            rollout.generator_features,
            device,
            feature_normalizer=discriminator_normalizer,
            feature_clip=feature_clip,
            loss_type=str(getattr(cfg, "discriminator_loss", "bce")),
        )
        combined_raw_gail_rewards = raw_gail_rewards.astype(np.float32, copy=True)
    if scene_discriminator is not None and rollout.scene_features.size:
        scene_rewards = discriminator_reward(
            scene_discriminator,
            rollout.scene_features,
            device,
            feature_normalizer=scene_discriminator_normalizer,
            feature_clip=feature_clip,
            loss_type=str(getattr(cfg, "discriminator_loss", "bce")),
        )
        scene_rewards = scene_rewards * float(cfg.scene_reward_coef)
        valid = (
            (rollout.transition_scene_indices >= 0)
            & (rollout.transition_scene_indices < len(scene_rewards))
        )
        combined_raw_gail_rewards[valid] += scene_rewards[rollout.transition_scene_indices[valid]]
    if sequence_discriminator is not None and rollout.sequence_features.size:
        sequence_rewards = discriminator_reward(
            sequence_discriminator,
            rollout.sequence_features,
            device,
            feature_normalizer=sequence_discriminator_normalizer,
            feature_clip=feature_clip,
            loss_type=str(getattr(cfg, "discriminator_loss", "bce")),
        )
        sequence_rewards = sequence_rewards * float(cfg.sequence_reward_coef)
        combined_raw_gail_rewards += sequence_rewards_to_transition_rewards(
            sequence_rewards,
            num_transitions=len(combined_raw_gail_rewards),
            sequence_last_indices=rollout.sequence_last_indices,
            sequence_transition_indices=rollout.sequence_transition_indices,
            assignment=str(getattr(cfg, "sequence_reward_assignment", "last")),
        )
    normalized_gail_rewards = shape_adversarial_rewards(combined_raw_gail_rewards, cfg)
    rewards, challenge_bonuses = combine_primary_env_challenge_rewards(
        normalized_gail_rewards,
        rollout.env_penalties,
        cfg,
        challenge_payoffs=rollout.challenge_payoffs,
    )
    returns, advantages = compute_returns_and_advantages(
        rewards,
        rollout.old_values,
        rollout.dones,
        rollout.trajectory_ids,
        cfg,
    )
    return RolloutBatch(
        policy_observations=rollout.policy_observations,
        next_policy_observations=rollout.next_policy_observations,
        critic_observations=rollout.critic_observations,
        next_critic_observations=rollout.next_critic_observations,
        actions=rollout.actions,
        action_masks=rollout.action_masks,
        old_log_probs=rollout.old_log_probs,
        old_values=rollout.old_values,
        trajectory_ids=rollout.trajectory_ids,
        dones=rollout.dones,
        rewards=rewards,
        gail_rewards_raw=combined_raw_gail_rewards,
        gail_rewards_normalized=normalized_gail_rewards,
        env_penalties=rollout.env_penalties,
        returns=returns,
        advantages=advantages,
        generator_features=rollout.generator_features,
        scene_features=rollout.scene_features,
        transition_scene_indices=rollout.transition_scene_indices,
        sequence_features=rollout.sequence_features,
        sequence_last_indices=rollout.sequence_last_indices,
        sequence_transition_indices=rollout.sequence_transition_indices,
        num_env_steps=rollout.num_env_steps,
        num_agent_steps=rollout.num_agent_steps,
        vehicle_ids=rollout.vehicle_ids,
        policy_step_memories=rollout.policy_step_memories,
        challenge_pressures=rollout.challenge_pressures,
        challenge_payoffs=rollout.challenge_payoffs,
        challenge_bonuses=challenge_bonuses,
        challenge_crash_rate_ema=rollout.challenge_crash_rate_ema,
        challenge_offroad_rate_ema=rollout.challenge_offroad_rate_ema,
        challenge_ttc_targets=rollout.challenge_ttc_targets,
        challenge_gap_targets=rollout.challenge_gap_targets,
        num_episodes=rollout.num_episodes,
        num_terminated=rollout.num_terminated,
        num_truncated=rollout.num_truncated,
        num_crash_events=rollout.num_crash_events,
        num_offroad_events=rollout.num_offroad_events,
        crash_agent_fraction=rollout.crash_agent_fraction,
        offroad_agent_fraction=rollout.offroad_agent_fraction,
        mean_env_penalty=float(rollout.env_penalties.mean()) if rollout.env_penalties.size else 0.0,
        mean_raw_gail_reward=float(combined_raw_gail_rewards.mean()) if combined_raw_gail_rewards.size else 0.0,
        mean_normalized_gail_reward=float(normalized_gail_rewards.mean()) if normalized_gail_rewards.size else 0.0,
        mean_episode_length=rollout.mean_episode_length,
        min_episode_length=rollout.min_episode_length,
        max_episode_length=rollout.max_episode_length,
        unique_episode_names=rollout.unique_episode_names,
        episode_names=rollout.episode_names,
        mean_controlled_vehicles=rollout.mean_controlled_vehicles,
        mean_road_vehicles=rollout.mean_road_vehicles,
    )

def collect_rollout(
    env: gym.Env,
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    seed: int | None = None,
    archive_policy_state_dicts: list[dict[str, torch.Tensor]] | None = None,
    policy_obs_dim: int | None = None,
    critic_obs_dim: int | None = None,
) -> RolloutBatch:
    policy.eval()
    rollout_seed = int(cfg.seed if seed is None else seed)
    rng = np.random.default_rng(rollout_seed + 7919)
    obs, _ = env.reset(seed=rollout_seed)
    started = time.perf_counter()
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    _RolloutPerfProfiler.record("flatten_observation", time.perf_counter() - started)
    if policy_obs_dim is None:
        policy_obs_dim = int(obs_agents.shape[1])
    if critic_obs_dim is None:
        critic_obs_dim = central_critic_observation_dim(int(policy_obs_dim), cfg)
    psro_enabled = bool(getattr(cfg, "psro_lite", False))
    archive_policy_state_dicts = list(archive_policy_state_dicts or [])
    archive_policies = (
        [
            _make_policy_from_state_dict(
                state_dict,
                cfg,
                int(policy_obs_dim),
                int(critic_obs_dim),
                device,
            )
            for state_dict in archive_policy_state_dicts
        ]
        if psro_enabled and archive_policy_state_dicts
        else []
    )
    psro_current_fraction_target = min(
        1.0,
        max(0.0, float(getattr(cfg, "psro_current_policy_fraction", 1.0))),
    )
    transitions: list[AgentTransition] = []
    scene_features: list[np.ndarray] = []
    collect_scene_features = bool(getattr(cfg, "enable_scene_discriminator", False))
    key_to_trajectory_id: dict[tuple[int, int], int] = {}
    key_to_policy_source: dict[tuple[int, int], int] = {}
    current_memory_state: dict[tuple[int, int], list[np.ndarray]] = {}
    archive_memory_states: list[dict[tuple[int, int], list[np.ndarray]]] = [
        {} for _ in archive_policies
    ]
    episode_counter = 0
    env_steps = 0
    episode_steps = 0
    episode_lengths: list[int] = []
    episode_names = {str(getattr(env.unwrapped, "episode_name", ""))}
    controlled_counts: list[int] = []
    road_vehicle_counts: list[int] = []
    terminated_count = 0
    truncated_count = 0
    crash_event_count = 0
    offroad_event_count = 0
    episode_had_crash = False
    episode_had_offroad = False
    psro_current_decisions = 0
    psro_archive_decisions = 0
    challenge_enabled = bool(getattr(cfg, "enable_player_challenge_reward", False))
    challenge_risk_state: dict[tuple[int, int], tuple[float, float]] = {}
    challenge_beta = min(0.999, max(0.0, float(getattr(cfg, "challenge_risk_ema_beta", 0.95))))

    collect_full_episodes = bool(getattr(cfg, "rollout_full_episodes", True))
    forced_reset_cap = 0 if collect_full_episodes else int(cfg.rollout_max_episode_steps)
    target_episodes = _target_rollout_episodes(cfg)

    while (
        len(episode_lengths) < target_episodes
        if collect_full_episodes
        else env_steps < int(cfg.rollout_steps) or len(episode_lengths) < target_episodes
    ):
        from ..envs import controlled_vehicle_snapshot

        vehicle_ids, trajectory_states = controlled_vehicle_snapshot(env)
        controlled_counts.append(len(vehicle_ids))
        road = getattr(env.unwrapped, "road", None)
        road_vehicles = list(getattr(road, "vehicles", ())) if road is not None else []
        road_vehicle_counts.append(len(road_vehicles))
        scene_index = -1
        if collect_scene_features:
            scene_index = len(scene_features)
            controlled_positions = [
                np.asarray(vehicle.position, dtype=np.float32)
                for vehicle in getattr(env.unwrapped, "controlled_vehicles", ())
                if getattr(vehicle, "position", None) is not None
            ]
            scene_origin = (
                np.mean(np.stack(controlled_positions, axis=0), axis=0)
                if controlled_positions
                else None
            )
            scene_features.append(
                scene_snapshot_features(
                    road_vehicles,
                    max_vehicles=int(cfg.scene_max_vehicles),
                    origin=scene_origin,
                )
            )
        if len(vehicle_ids) != len(obs_agents):
            raise RuntimeError(
                f"Observation/vehicle mismatch: obs_agents={len(obs_agents)} vehicles={len(vehicle_ids)}"
            )
        keys = [(episode_counter, vehicle_id) for vehicle_id in vehicle_ids]
        missing_source_indices = [idx for idx, key in enumerate(keys) if key not in key_to_policy_source]
        if missing_source_indices:
            source_ids = _assign_psro_sources(
                len(missing_source_indices),
                num_archive_policies=len(archive_policies),
                current_fraction=psro_current_fraction_target,
                rng=rng,
            )
            for local_idx, source_id in zip(missing_source_indices, source_ids):
                key_to_policy_source[keys[local_idx]] = int(source_id)
        policy_sources = np.asarray(
            [key_to_policy_source.get(key, 0) for key in keys],
            dtype=np.int16,
        )
        psro_current_decisions += int(np.sum(policy_sources == 0))
        psro_archive_decisions += int(np.sum(policy_sources > 0))
        for key, source_id in zip(keys, policy_sources):
            if int(source_id) == 0 and key not in key_to_trajectory_id:
                key_to_trajectory_id[key] = len(key_to_trajectory_id)

        critic_obs_agents = central_critic_observations(env, cfg, obs_agents)
        started = time.perf_counter()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
            critic_obs_tensor = torch.as_tensor(critic_obs_agents, dtype=torch.float32, device=device)
            action_masks = (
                discrete_action_masks_from_env(
                    env,
                    num_agents=len(obs_agents),
                    num_actions=policy_action_dim(policy),
                    enabled=bool(getattr(cfg, "enable_action_masking", True)),
                )
                if not _is_continuous(cfg)
                else np.ones((len(obs_agents), 0), dtype=bool)
            )
            action_mask_tensor = (
                torch.as_tensor(action_masks, dtype=torch.bool, device=device)
                if not _is_continuous(cfg)
                else None
            )
            current_memory = _memory_cache_from_state(
                policy,
                cfg,
                keys,
                current_memory_state,
                device,
            )
            actions, log_probs, values, current_step_memory = _mixed_policy_actions(
                policy,
                archive_policies,
                policy_sources,
                obs_tensor,
                critic_obs_tensor,
                action_mask_tensor,
                cfg,
                agent_keys=keys,
                current_memory=current_memory,
                archive_memory_states=archive_memory_states,
            )
            current_step_memory_np = _update_memory_state(
                policy,
                cfg,
                keys,
                current_memory_state,
                current_step_memory,
            )
        _RolloutPerfProfiler.record("policy_inference", time.perf_counter() - started)

        action_tuple = _actions_to_env_tuple(actions, cfg)
        next_obs, _env_reward, terminated, truncated, info = env.step(action_tuple)
        started = time.perf_counter()
        next_obs_agents = policy_observations_from_flat(flatten_agent_observations(next_obs))
        _RolloutPerfProfiler.record("flatten_observation", time.perf_counter() - started)
        if len(next_obs_agents) != len(obs_agents):
            next_obs_agents = obs_agents.copy()
            next_critic_obs_agents = critic_obs_agents.copy()
        else:
            next_critic_obs_agents = central_critic_observations(env, cfg, next_obs_agents)
            if next_critic_obs_agents.shape != critic_obs_agents.shape:
                next_critic_obs_agents = critic_obs_agents.copy()
        episode_steps += 1
        force_rollout_reset = forced_reset_cap > 0 and episode_steps >= forced_reset_cap
        done = bool(terminated or truncated or force_rollout_reset)
        crash_flags = info.get("controlled_vehicle_crashes", [])
        offroad_flags = info.get("controlled_vehicle_offroad", [])
        interaction_metrics = (
            list(info.get("controlled_vehicle_interaction_metrics", []) or [])
            if challenge_enabled
            else []
        )
        episode_had_crash = bool(episode_had_crash or any(bool(flag) for flag in crash_flags))
        episode_had_offroad = bool(episode_had_offroad or any(bool(flag) for flag in offroad_flags))

        for i, key in enumerate(keys):
            if int(policy_sources[i]) != 0:
                continue
            vehicle_id = int(vehicle_ids[i]) if i < len(vehicle_ids) else -1
            crashed = bool(crash_flags[i]) if i < len(crash_flags) else False
            offroad = bool(offroad_flags[i]) if i < len(offroad_flags) else False
            env_penalty = 0.0
            if crashed:
                env_penalty -= float(cfg.collision_penalty)
            if offroad:
                env_penalty -= float(cfg.offroad_penalty)
            prev_crash_ema, prev_offroad_ema = challenge_risk_state.get(key, (0.0, 0.0))
            crash_ema = challenge_beta * prev_crash_ema + (1.0 - challenge_beta) * float(crashed)
            offroad_ema = challenge_beta * prev_offroad_ema + (1.0 - challenge_beta) * float(offroad)
            challenge_risk_state[key] = (float(crash_ema), float(offroad_ema))
            if challenge_enabled:
                metric = interaction_metrics[i] if i < len(interaction_metrics) else None
                pressure, ttc_target, gap_target = player_challenge_pressure_from_metric(metric, cfg)
                payoff = player_challenge_payoff(
                    pressure,
                    crash_rate_ema=float(crash_ema),
                    offroad_rate_ema=float(offroad_ema),
                    cfg=cfg,
                )
            else:
                pressure = payoff = ttc_target = gap_target = 0.0
            transitions.append(
                AgentTransition(
                    vehicle_id=vehicle_id,
                    policy_observation=obs_agents[i].copy(),
                    next_policy_observation=next_obs_agents[i].copy(),
                    critic_observation=critic_obs_agents[i].copy(),
                    next_critic_observation=next_critic_obs_agents[i].copy(),
                    action=(
                        np.asarray(action_tuple[i], dtype=np.float32).copy()
                        if _is_continuous(cfg)
                        else int(action_tuple[i])
                    ),
                    action_mask=action_masks[i].copy(),
                    log_prob=float(log_probs[i].cpu().item()),
                    value=float(values[i].cpu().item()),
                    trajectory_id=int(key_to_trajectory_id[key]),
                    trajectory_state=trajectory_states[i].copy(),
                    scene_index=int(scene_index),
                    env_penalty=float(env_penalty),
                    crashed=crashed,
                    offroad=offroad,
                    challenge_pressure=float(pressure),
                    challenge_payoff=float(payoff),
                    challenge_crash_rate_ema=float(crash_ema),
                    challenge_offroad_rate_ema=float(offroad_ema),
                    challenge_ttc_target=float(ttc_target),
                    challenge_gap_target=float(gap_target),
                    done=done,
                    policy_step_memory=(
                        current_step_memory_np[i].copy()
                        if current_step_memory_np.size
                        else np.zeros((0, 0), dtype=_memory_storage_dtype(cfg))
                    ),
                )
            )

        env_steps += 1
        if done:
            episode_lengths.append(int(episode_steps))
            terminated_count += int(bool(terminated))
            truncated_count += int(bool(truncated or force_rollout_reset))
            crash_event_count += int(episode_had_crash)
            offroad_event_count += int(episode_had_offroad)
            old_episode_counter = int(episode_counter)
            current_memory_state = {
                key: value for key, value in current_memory_state.items() if key[0] != old_episode_counter
            }
            for archive_state in archive_memory_states:
                stale_keys = [key for key in archive_state if key[0] == old_episode_counter]
                for key in stale_keys:
                    del archive_state[key]
            episode_counter += 1
            episode_steps = 0
            episode_had_crash = False
            episode_had_offroad = False
            obs, _ = env.reset()
            episode_names.add(str(getattr(env.unwrapped, "episode_name", "")))
        else:
            obs = next_obs
        obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))

    if episode_steps > 0 and not collect_full_episodes:
        episode_lengths.append(int(episode_steps))
        crash_event_count += int(episode_had_crash)
        offroad_event_count += int(episode_had_offroad)
    if not transitions:
        raise RuntimeError("Rollout produced no trainable current-policy transitions.")

    policy_obs = np.stack([tr.policy_observation for tr in transitions], axis=0).astype(np.float32)
    next_policy_obs = np.stack([tr.next_policy_observation for tr in transitions], axis=0).astype(np.float32)
    critic_obs = np.stack([tr.critic_observation for tr in transitions], axis=0).astype(np.float32)
    next_critic_obs = np.stack([tr.next_critic_observation for tr in transitions], axis=0).astype(np.float32)
    trajectory_states = np.stack([tr.trajectory_state for tr in transitions], axis=0).astype(np.float32)
    actions = _actions_to_rollout_array(transitions, cfg)
    action_masks = np.stack([tr.action_mask for tr in transitions], axis=0).astype(bool)
    old_log_probs = np.asarray([tr.log_prob for tr in transitions], dtype=np.float32)
    old_values = np.asarray([tr.value for tr in transitions], dtype=np.float32)
    policy_step_memories = np.stack([tr.policy_step_memory for tr in transitions], axis=0).astype(
        _memory_storage_dtype(cfg),
        copy=False,
    )
    dones = np.asarray([tr.done for tr in transitions], dtype=bool)
    env_penalties = np.asarray([tr.env_penalty for tr in transitions], dtype=np.float32)
    transition_scene_indices = np.asarray([tr.scene_index for tr in transitions], dtype=np.int64)
    crashed = np.asarray([tr.crashed for tr in transitions], dtype=bool)
    offroad = np.asarray([tr.offroad for tr in transitions], dtype=bool)
    rollout_vehicle_ids = np.asarray([tr.vehicle_id for tr in transitions], dtype=np.int64)
    challenge_pressures = np.asarray([tr.challenge_pressure for tr in transitions], dtype=np.float32)
    challenge_payoffs = np.asarray([tr.challenge_payoff for tr in transitions], dtype=np.float32)
    challenge_crash_rate_ema = np.asarray(
        [tr.challenge_crash_rate_ema for tr in transitions],
        dtype=np.float32,
    )
    challenge_offroad_rate_ema = np.asarray(
        [tr.challenge_offroad_rate_ema for tr in transitions],
        dtype=np.float32,
    )
    challenge_ttc_targets = np.asarray([tr.challenge_ttc_target for tr in transitions], dtype=np.float32)
    challenge_gap_targets = np.asarray([tr.challenge_gap_target for tr in transitions], dtype=np.float32)
    trajectory_ids = np.asarray([tr.trajectory_id for tr in transitions], dtype=np.int32)
    trajectory_states = normalize_trajectory_frame(
        trajectory_states,
        trajectory_ids,
        frame=cfg.trajectory_frame,
    )
    if discriminator_input_mode(cfg) == "action":
        gen_features = action_conditioned_features(policy_obs, actions)
    else:
        gen_features = discriminator_features(policy_obs, trajectory_states)
    sequence_features, sequence_last_indices, sequence_transition_indices = build_sequence_windows(
        gen_features,
        trajectory_ids,
        sequence_length=int(cfg.sequence_length),
        stride=int(cfg.sequence_stride),
        return_window_indices=True,
    )
    sequence_features = transform_sequence_features(
        sequence_features,
        mode=str(getattr(cfg, "sequence_feature_mode", "raw")),
    )
    rewards = np.zeros(len(transitions), dtype=np.float32)
    returns, advantages = compute_returns_and_advantages(rewards, old_values, dones, trajectory_ids, cfg)
    return RolloutBatch(
        policy_observations=policy_obs,
        next_policy_observations=next_policy_obs,
        critic_observations=critic_obs,
        next_critic_observations=next_critic_obs,
        actions=actions,
        action_masks=action_masks,
        old_log_probs=old_log_probs,
        old_values=old_values,
        trajectory_ids=trajectory_ids,
        dones=dones,
        rewards=rewards,
        gail_rewards_raw=np.zeros(len(transitions), dtype=np.float32),
        gail_rewards_normalized=np.zeros(len(transitions), dtype=np.float32),
        env_penalties=env_penalties,
        returns=returns,
        advantages=advantages,
        generator_features=gen_features,
        scene_features=np.stack(scene_features, axis=0).astype(np.float32, copy=False)
        if scene_features
        else np.zeros(
            (0, int(cfg.scene_max_vehicles) * int(cfg.scene_feature_dim_per_vehicle)),
            dtype=np.float32,
        ),
        transition_scene_indices=transition_scene_indices,
        sequence_features=sequence_features,
        sequence_last_indices=sequence_last_indices,
        sequence_transition_indices=sequence_transition_indices,
        num_env_steps=env_steps,
        num_agent_steps=len(transitions),
        vehicle_ids=rollout_vehicle_ids,
        policy_step_memories=policy_step_memories,
        challenge_pressures=challenge_pressures,
        challenge_payoffs=challenge_payoffs,
        challenge_bonuses=np.zeros(len(transitions), dtype=np.float32),
        challenge_crash_rate_ema=challenge_crash_rate_ema,
        challenge_offroad_rate_ema=challenge_offroad_rate_ema,
        challenge_ttc_targets=challenge_ttc_targets,
        challenge_gap_targets=challenge_gap_targets,
        num_episodes=len(episode_lengths),
        num_terminated=terminated_count,
        num_truncated=truncated_count,
        num_crash_events=crash_event_count,
        num_offroad_events=offroad_event_count,
        crash_agent_fraction=float(crashed.mean()) if crashed.size else 0.0,
        offroad_agent_fraction=float(offroad.mean()) if offroad.size else 0.0,
        mean_env_penalty=float(env_penalties.mean()) if env_penalties.size else 0.0,
        mean_episode_length=float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        min_episode_length=int(np.min(episode_lengths)) if episode_lengths else 0,
        max_episode_length=int(np.max(episode_lengths)) if episode_lengths else 0,
        unique_episode_names=len({name for name in episode_names if name}),
        episode_names=tuple(sorted(name for name in episode_names if name)),
        mean_controlled_vehicles=float(np.mean(controlled_counts)) if controlled_counts else 0.0,
        mean_road_vehicles=float(np.mean(road_vehicle_counts)) if road_vehicle_counts else 0.0,
        psro_active=bool(archive_policies and psro_archive_decisions > 0),
        psro_current_decisions=int(psro_current_decisions),
        psro_archive_decisions=int(psro_archive_decisions),
        psro_current_fraction=(
            float(psro_current_decisions)
            / float(max(1, psro_current_decisions + psro_archive_decisions))
        ),
    )

def merge_rollout_batches(batches: list[RolloutBatch], cfg: PSGAILConfig) -> RolloutBatch:
    if not batches:
        raise ValueError("Cannot merge an empty rollout batch list.")

    trajectory_ids: list[np.ndarray] = []
    trajectory_offset = 0
    transition_offset = 0
    scene_offset = 0
    transition_scene_indices: list[np.ndarray] = []
    sequence_last_indices: list[np.ndarray] = []
    sequence_transition_indices: list[np.ndarray] = []
    for batch in batches:
        ids = batch.trajectory_ids.astype(np.int32, copy=True)
        if ids.size:
            ids += trajectory_offset
            trajectory_offset = int(ids.max()) + 1
        trajectory_ids.append(ids)
        scene_ids = batch.transition_scene_indices.astype(np.int64, copy=True)
        if scene_ids.size:
            valid = scene_ids >= 0
            scene_ids[valid] += scene_offset
        transition_scene_indices.append(scene_ids)
        last_ids = batch.sequence_last_indices.astype(np.int64, copy=True)
        if last_ids.size:
            last_ids += transition_offset
        sequence_last_indices.append(last_ids)
        window_ids = batch.sequence_transition_indices.astype(np.int64, copy=True)
        if window_ids.size:
            valid = window_ids >= 0
            window_ids[valid] += transition_offset
        sequence_transition_indices.append(window_ids)
        transition_offset += int(batch.num_agent_steps)
        scene_offset += int(len(batch.scene_features))

    rewards = np.concatenate([batch.rewards for batch in batches], axis=0).astype(np.float32)
    gail_rewards_raw = np.concatenate([batch.gail_rewards_raw for batch in batches], axis=0).astype(
        np.float32
    )
    gail_rewards_normalized = np.concatenate(
        [batch.gail_rewards_normalized for batch in batches], axis=0
    ).astype(np.float32)
    env_penalties = np.concatenate([batch.env_penalties for batch in batches], axis=0).astype(
        np.float32
    )
    old_values = np.concatenate([batch.old_values for batch in batches], axis=0).astype(np.float32)
    dones = np.concatenate([batch.dones for batch in batches], axis=0).astype(bool)
    merged_trajectory_ids = np.concatenate(trajectory_ids, axis=0).astype(np.int32)
    returns, advantages = compute_returns_and_advantages(
        rewards,
        old_values,
        dones,
        merged_trajectory_ids,
        cfg,
    )
    return RolloutBatch(
        policy_observations=np.concatenate([batch.policy_observations for batch in batches], axis=0).astype(
            np.float32
        ),
        next_policy_observations=np.concatenate(
            [batch.next_policy_observations for batch in batches],
            axis=0,
        ).astype(np.float32),
        critic_observations=np.concatenate([batch.critic_observations for batch in batches], axis=0).astype(
            np.float32
        ),
        next_critic_observations=np.concatenate(
            [batch.next_critic_observations for batch in batches],
            axis=0,
        ).astype(np.float32),
        actions=np.concatenate([batch.actions for batch in batches], axis=0).astype(
            np.float32 if _is_continuous(cfg) else np.int64
        ),
        action_masks=np.concatenate([batch.action_masks for batch in batches], axis=0).astype(bool),
        old_log_probs=np.concatenate([batch.old_log_probs for batch in batches], axis=0).astype(np.float32),
        old_values=old_values,
        trajectory_ids=merged_trajectory_ids,
        dones=dones,
        rewards=rewards,
        gail_rewards_raw=gail_rewards_raw,
        gail_rewards_normalized=gail_rewards_normalized,
        env_penalties=env_penalties,
        returns=returns,
        advantages=advantages,
        generator_features=np.concatenate([batch.generator_features for batch in batches], axis=0).astype(
            np.float32
        ),
        scene_features=np.concatenate([batch.scene_features for batch in batches], axis=0).astype(np.float32),
        transition_scene_indices=np.concatenate(transition_scene_indices, axis=0).astype(np.int64),
        sequence_features=np.concatenate([batch.sequence_features for batch in batches], axis=0).astype(np.float32),
        sequence_last_indices=np.concatenate(sequence_last_indices, axis=0).astype(np.int64),
        sequence_transition_indices=np.concatenate(sequence_transition_indices, axis=0).astype(np.int64),
        num_env_steps=sum(batch.num_env_steps for batch in batches),
        num_agent_steps=sum(batch.num_agent_steps for batch in batches),
        vehicle_ids=np.concatenate(
            [_transition_array(batch.vehicle_ids, batch.num_agent_steps, dtype=np.int64, fill=-1) for batch in batches],
            axis=0,
        ).astype(np.int64),
        policy_step_memories=np.concatenate(
            [batch.policy_step_memories for batch in batches],
            axis=0,
        ).astype(_memory_storage_dtype(cfg), copy=False),
        challenge_pressures=np.concatenate(
            [
                _transition_array(batch.challenge_pressures, batch.num_agent_steps, dtype=np.float32)
                for batch in batches
            ],
            axis=0,
        ).astype(np.float32),
        challenge_payoffs=np.concatenate(
            [
                _transition_array(batch.challenge_payoffs, batch.num_agent_steps, dtype=np.float32)
                for batch in batches
            ],
            axis=0,
        ).astype(np.float32),
        challenge_bonuses=np.concatenate(
            [
                _transition_array(batch.challenge_bonuses, batch.num_agent_steps, dtype=np.float32)
                for batch in batches
            ],
            axis=0,
        ).astype(np.float32),
        challenge_crash_rate_ema=np.concatenate(
            [
                _transition_array(batch.challenge_crash_rate_ema, batch.num_agent_steps, dtype=np.float32)
                for batch in batches
            ],
            axis=0,
        ).astype(np.float32),
        challenge_offroad_rate_ema=np.concatenate(
            [
                _transition_array(batch.challenge_offroad_rate_ema, batch.num_agent_steps, dtype=np.float32)
                for batch in batches
            ],
            axis=0,
        ).astype(np.float32),
        challenge_ttc_targets=np.concatenate(
            [
                _transition_array(batch.challenge_ttc_targets, batch.num_agent_steps, dtype=np.float32)
                for batch in batches
            ],
            axis=0,
        ).astype(np.float32),
        challenge_gap_targets=np.concatenate(
            [
                _transition_array(batch.challenge_gap_targets, batch.num_agent_steps, dtype=np.float32)
                for batch in batches
            ],
            axis=0,
        ).astype(np.float32),
        num_episodes=sum(batch.num_episodes for batch in batches),
        num_terminated=sum(batch.num_terminated for batch in batches),
        num_truncated=sum(batch.num_truncated for batch in batches),
        num_crash_events=sum(batch.num_crash_events for batch in batches),
        num_offroad_events=sum(batch.num_offroad_events for batch in batches),
        crash_agent_fraction=float(
            np.average(
                [batch.crash_agent_fraction for batch in batches],
                weights=[max(1, batch.num_agent_steps) for batch in batches],
            )
        ),
        offroad_agent_fraction=float(
            np.average(
                [batch.offroad_agent_fraction for batch in batches],
                weights=[max(1, batch.num_agent_steps) for batch in batches],
            )
        ),
        mean_env_penalty=float(env_penalties.mean()) if env_penalties.size else 0.0,
        mean_raw_gail_reward=float(gail_rewards_raw.mean()) if gail_rewards_raw.size else 0.0,
        mean_normalized_gail_reward=float(gail_rewards_normalized.mean())
        if gail_rewards_normalized.size
        else 0.0,
        mean_episode_length=float(
            np.average(
                [batch.mean_episode_length for batch in batches],
                weights=[max(1, batch.num_episodes) for batch in batches],
            )
        ),
        min_episode_length=min(
            [batch.min_episode_length for batch in batches if batch.min_episode_length > 0],
            default=0,
        ),
        max_episode_length=max(batch.max_episode_length for batch in batches),
        unique_episode_names=len(
            {
                name
                for batch in batches
                for name in batch.episode_names
                if name
            }
        ),
        episode_names=tuple(
            sorted(
                {
                    name
                    for batch in batches
                    for name in batch.episode_names
                    if name
                }
            )
        ),
        mean_controlled_vehicles=float(np.mean([batch.mean_controlled_vehicles for batch in batches])),
        mean_road_vehicles=float(np.mean([batch.mean_road_vehicles for batch in batches])),
        psro_active=any(bool(batch.psro_active) for batch in batches),
        psro_current_decisions=sum(int(batch.psro_current_decisions) for batch in batches),
        psro_archive_decisions=sum(int(batch.psro_archive_decisions) for batch in batches),
        psro_current_fraction=(
            float(sum(int(batch.psro_current_decisions) for batch in batches))
            / float(
                max(
                    1,
                    sum(
                        int(batch.psro_current_decisions) + int(batch.psro_archive_decisions)
                        for batch in batches
                    ),
                )
            )
        ),
    )

def rollout_training_agent_step_cap(cfg: PSGAILConfig) -> int:
    if not bool(getattr(cfg, "rollout_training_subsample", True)):
        return 0
    configured = int(getattr(cfg, "rollout_training_agent_steps", 0))
    if configured < 0:
        return 0
    if configured > 0:
        return configured
    return max(0, int(getattr(cfg, "rollout_target_agent_steps", 0)))

def _trajectory_sample_indices(
    trajectory_ids: np.ndarray,
    *,
    target_agent_steps: int,
    seed: int,
) -> np.ndarray:
    ids = np.asarray(trajectory_ids, dtype=np.int64)
    if ids.size <= max(0, int(target_agent_steps)):
        return np.arange(ids.size, dtype=np.int64)
    unique_ids, counts = np.unique(ids, return_counts=True)
    if unique_ids.size == 0:
        return np.arange(ids.size, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    selected_steps = 0
    for unique_index in rng.permutation(unique_ids.size):
        selected_id = int(unique_ids[int(unique_index)])
        selected.append(selected_id)
        selected_steps += int(counts[int(unique_index)])
        if selected_steps >= int(target_agent_steps):
            break
    if not selected:
        selected.append(int(unique_ids[0]))
    mask = np.isin(ids, np.asarray(selected, dtype=ids.dtype))
    return np.nonzero(mask)[0].astype(np.int64, copy=False)

def _remap_trajectory_ids(trajectory_ids: np.ndarray) -> np.ndarray:
    old_ids = np.asarray(trajectory_ids, dtype=np.int64)
    unique_ids = np.unique(old_ids)
    mapping = {int(old_id): new_id for new_id, old_id in enumerate(unique_ids)}
    return np.asarray([mapping[int(old_id)] for old_id in old_ids], dtype=np.int32)

def _subset_scene_features(
    rollout: RolloutBatch,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transition_scene_indices = rollout.transition_scene_indices[selected_indices].astype(np.int64, copy=True)
    scene_features = rollout.scene_features
    if not scene_features.size or not transition_scene_indices.size:
        return scene_features[:0].astype(np.float32, copy=False), np.full(
            transition_scene_indices.shape,
            -1,
            dtype=np.int64,
        )

    valid = (transition_scene_indices >= 0) & (transition_scene_indices < len(scene_features))
    used_scene_ids = np.unique(transition_scene_indices[valid])
    if used_scene_ids.size == 0:
        return scene_features[:0].astype(np.float32, copy=False), np.full(
            transition_scene_indices.shape,
            -1,
            dtype=np.int64,
        )

    scene_id_map = np.full(len(scene_features), -1, dtype=np.int64)
    scene_id_map[used_scene_ids] = np.arange(used_scene_ids.size, dtype=np.int64)
    remapped = np.full(transition_scene_indices.shape, -1, dtype=np.int64)
    remapped[valid] = scene_id_map[transition_scene_indices[valid]]
    return scene_features[used_scene_ids].astype(np.float32, copy=False), remapped

def _subset_sequence_features(
    rollout: RolloutBatch,
    transition_id_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence_transition_indices = rollout.sequence_transition_indices
    if not rollout.sequence_features.size or not sequence_transition_indices.size:
        return (
            rollout.sequence_features[:0].astype(np.float32, copy=False),
            rollout.sequence_last_indices[:0].astype(np.int64, copy=False),
            sequence_transition_indices[:0].astype(np.int64, copy=False),
        )

    clipped_windows = np.maximum(sequence_transition_indices, 0)
    mapped_windows = transition_id_map[clipped_windows]
    valid_entries = sequence_transition_indices >= 0
    valid_windows = np.all(~valid_entries | (mapped_windows >= 0), axis=1)
    if rollout.sequence_last_indices.size:
        last_indices = rollout.sequence_last_indices.astype(np.int64, copy=False)
        valid_last = last_indices >= 0
        last_selected = np.ones(last_indices.shape, dtype=bool)
        last_selected[valid_last] = transition_id_map[last_indices[valid_last]] >= 0
        valid_windows &= last_selected
    if not np.any(valid_windows):
        return (
            rollout.sequence_features[:0].astype(np.float32, copy=False),
            rollout.sequence_last_indices[:0].astype(np.int64, copy=False),
            sequence_transition_indices[:0].astype(np.int64, copy=False),
        )

    remapped_windows = sequence_transition_indices[valid_windows].astype(np.int64, copy=True)
    valid_remapped_entries = remapped_windows >= 0
    remapped_windows[valid_remapped_entries] = transition_id_map[
        remapped_windows[valid_remapped_entries]
    ]
    remapped_last_indices = rollout.sequence_last_indices[valid_windows].astype(np.int64, copy=True)
    valid_last = remapped_last_indices >= 0
    remapped_last_indices[valid_last] = transition_id_map[remapped_last_indices[valid_last]]
    return (
        rollout.sequence_features[valid_windows].astype(np.float32, copy=False),
        remapped_last_indices,
        remapped_windows,
    )

def subsample_rollout_for_training(
    rollout: RolloutBatch,
    cfg: PSGAILConfig,
    *,
    seed: int,
) -> RolloutBatch:
    target_agent_steps = rollout_training_agent_step_cap(cfg)
    if target_agent_steps <= 0 or int(rollout.num_agent_steps) <= int(target_agent_steps):
        return rollout

    selected_indices = _trajectory_sample_indices(
        rollout.trajectory_ids,
        target_agent_steps=int(target_agent_steps),
        seed=int(seed),
    )
    if selected_indices.size == 0 or selected_indices.size >= int(rollout.num_agent_steps):
        return rollout

    transition_id_map = np.full(int(rollout.num_agent_steps), -1, dtype=np.int64)
    transition_id_map[selected_indices] = np.arange(selected_indices.size, dtype=np.int64)
    scene_features, transition_scene_indices = _subset_scene_features(rollout, selected_indices)
    sequence_features, sequence_last_indices, sequence_transition_indices = _subset_sequence_features(
        rollout,
        transition_id_map,
    )

    trajectory_ids = _remap_trajectory_ids(rollout.trajectory_ids[selected_indices])
    rewards = rollout.rewards[selected_indices].astype(np.float32, copy=False)
    old_values = rollout.old_values[selected_indices].astype(np.float32, copy=False)
    dones = rollout.dones[selected_indices].astype(bool, copy=False)
    returns, advantages = compute_returns_and_advantages(rewards, old_values, dones, trajectory_ids, cfg)
    env_penalties = rollout.env_penalties[selected_indices].astype(np.float32, copy=False)
    gail_rewards_raw = rollout.gail_rewards_raw[selected_indices].astype(np.float32, copy=False)
    gail_rewards_normalized = rollout.gail_rewards_normalized[selected_indices].astype(
        np.float32,
        copy=False,
    )
    return RolloutBatch(
        policy_observations=rollout.policy_observations[selected_indices].astype(np.float32, copy=False),
        next_policy_observations=rollout.next_policy_observations[selected_indices].astype(
            np.float32,
            copy=False,
        ),
        critic_observations=rollout.critic_observations[selected_indices].astype(np.float32, copy=False),
        next_critic_observations=rollout.next_critic_observations[selected_indices].astype(
            np.float32,
            copy=False,
        ),
        actions=rollout.actions[selected_indices].astype(rollout.actions.dtype, copy=False),
        action_masks=rollout.action_masks[selected_indices].astype(bool, copy=False),
        old_log_probs=rollout.old_log_probs[selected_indices].astype(np.float32, copy=False),
        old_values=old_values,
        trajectory_ids=trajectory_ids,
        dones=dones,
        rewards=rewards,
        gail_rewards_raw=gail_rewards_raw,
        gail_rewards_normalized=gail_rewards_normalized,
        env_penalties=env_penalties,
        returns=returns,
        advantages=advantages,
        generator_features=rollout.generator_features[selected_indices].astype(np.float32, copy=False),
        scene_features=scene_features,
        transition_scene_indices=transition_scene_indices,
        sequence_features=sequence_features,
        sequence_last_indices=sequence_last_indices,
        sequence_transition_indices=sequence_transition_indices,
        num_env_steps=rollout.num_env_steps,
        num_agent_steps=int(selected_indices.size),
        vehicle_ids=_transition_array(
            rollout.vehicle_ids,
            rollout.num_agent_steps,
            dtype=np.int64,
            fill=-1,
        )[selected_indices].astype(np.int64, copy=False),
        policy_step_memories=rollout.policy_step_memories[selected_indices].astype(
            _memory_storage_dtype(cfg),
            copy=False,
        ),
        challenge_pressures=_transition_array(
            rollout.challenge_pressures,
            rollout.num_agent_steps,
            dtype=np.float32,
        )[selected_indices].astype(np.float32, copy=False),
        challenge_payoffs=_transition_array(
            rollout.challenge_payoffs,
            rollout.num_agent_steps,
            dtype=np.float32,
        )[selected_indices].astype(np.float32, copy=False),
        challenge_bonuses=_transition_array(
            rollout.challenge_bonuses,
            rollout.num_agent_steps,
            dtype=np.float32,
        )[selected_indices].astype(np.float32, copy=False),
        challenge_crash_rate_ema=_transition_array(
            rollout.challenge_crash_rate_ema,
            rollout.num_agent_steps,
            dtype=np.float32,
        )[selected_indices].astype(np.float32, copy=False),
        challenge_offroad_rate_ema=_transition_array(
            rollout.challenge_offroad_rate_ema,
            rollout.num_agent_steps,
            dtype=np.float32,
        )[selected_indices].astype(np.float32, copy=False),
        challenge_ttc_targets=_transition_array(
            rollout.challenge_ttc_targets,
            rollout.num_agent_steps,
            dtype=np.float32,
        )[selected_indices].astype(np.float32, copy=False),
        challenge_gap_targets=_transition_array(
            rollout.challenge_gap_targets,
            rollout.num_agent_steps,
            dtype=np.float32,
        )[selected_indices].astype(np.float32, copy=False),
        num_episodes=rollout.num_episodes,
        num_terminated=rollout.num_terminated,
        num_truncated=rollout.num_truncated,
        num_crash_events=rollout.num_crash_events,
        num_offroad_events=rollout.num_offroad_events,
        crash_agent_fraction=rollout.crash_agent_fraction,
        offroad_agent_fraction=rollout.offroad_agent_fraction,
        mean_env_penalty=float(env_penalties.mean()) if env_penalties.size else 0.0,
        mean_raw_gail_reward=float(gail_rewards_raw.mean()) if gail_rewards_raw.size else 0.0,
        mean_normalized_gail_reward=float(gail_rewards_normalized.mean())
        if gail_rewards_normalized.size
        else 0.0,
        mean_episode_length=rollout.mean_episode_length,
        min_episode_length=rollout.min_episode_length,
        max_episode_length=rollout.max_episode_length,
        unique_episode_names=rollout.unique_episode_names,
        episode_names=rollout.episode_names,
        mean_controlled_vehicles=rollout.mean_controlled_vehicles,
        mean_road_vehicles=rollout.mean_road_vehicles,
        psro_active=rollout.psro_active,
        psro_current_decisions=rollout.psro_current_decisions,
        psro_archive_decisions=rollout.psro_archive_decisions,
        psro_current_fraction=rollout.psro_current_fraction,
    )

def collect_round_rollouts(
    env: gym.Env,
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    policy_obs_dim: int,
    critic_obs_dim: int | None = None,
    *,
    round_idx: int,
    rollout_executor,
    archive_policy_state_dicts: list[dict[str, torch.Tensor]] | None = None,
) -> RolloutBatch:
    target_agent_steps = max(0, int(getattr(cfg, "rollout_target_agent_steps", 0)))
    worker_stride = max(1, int(cfg.num_rollout_workers))
    batches: list[RolloutBatch] = []
    attempt = 0
    while True:
        seed_offset = (int(round_idx) - 1) * worker_stride + attempt * 1_000_003
        collected_agent_steps = sum(int(item.num_agent_steps) for item in batches)
        remaining_agent_steps = (
            max(0, target_agent_steps - collected_agent_steps)
            if target_agent_steps > 0
            else 0
        )
        rollout_cfg = (
            replace(
                cfg,
                rollout_min_episodes=_target_aware_rollout_episodes(
                    cfg,
                    remaining_agent_steps=remaining_agent_steps,
                ),
            )
            if target_agent_steps > 0
            else cfg
        )
        batch = collect_rollouts(
            env,
            policy,
            rollout_cfg,
            device,
            policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            seed_offset=seed_offset,
            executor=rollout_executor,
            archive_policy_state_dicts=archive_policy_state_dicts,
        )
        batches.append(batch)
        agent_steps = sum(int(item.num_agent_steps) for item in batches)
        if target_agent_steps <= 0 or agent_steps >= target_agent_steps:
            break
        attempt += 1
        print(
            f"[round {round_idx:04d}] accumulating rollout agent_steps="
            f"{agent_steps}/{target_agent_steps}"
        )
    return batches[0] if len(batches) == 1 else merge_rollout_batches(batches, cfg)

def _rollout_worker(
    cfg: PSGAILConfig,
    policy_state_dict: dict[str, torch.Tensor],
    archive_policy_state_dicts: list[dict[str, torch.Tensor]] | None,
    policy_obs_dim: int,
    critic_obs_dim: int,
    worker_id: int,
    rollout_steps: int,
    rollout_min_episodes: int,
) -> RolloutBatch:
    threads = max(1, int(cfg.rollout_worker_threads))
    torch.set_num_threads(threads)
    worker_seed = int(cfg.seed) + int(worker_id)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    worker_cfg = replace(
        cfg,
        rollout_steps=int(rollout_steps),
        rollout_min_episodes=int(rollout_min_episodes),
        seed=worker_seed,
        device="cpu",
    )
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
    policy.load_state_dict(policy_state_dict)
    policy.to(torch.device("cpu"))

    from ..envs import make_training_env

    env = make_training_env(worker_cfg)
    try:
        return collect_rollout(
            env,
            policy,
            worker_cfg,
            torch.device("cpu"),
            seed=int(worker_cfg.seed),
            archive_policy_state_dicts=archive_policy_state_dicts,
            policy_obs_dim=int(policy_obs_dim),
            critic_obs_dim=int(critic_obs_dim),
        )
    finally:
        env.close()

def collect_rollouts(
    env: gym.Env,
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    policy_obs_dim: int,
    critic_obs_dim: int | None = None,
    seed_offset: int = 0,
    executor: ProcessPoolExecutor | None = None,
    archive_policy_state_dicts: list[dict[str, torch.Tensor]] | None = None,
) -> RolloutBatch:
    num_workers = max(1, int(cfg.num_rollout_workers))
    total_steps = max(1, int(cfg.rollout_steps))
    rollout_seed = int(cfg.seed) + int(seed_offset)
    if num_workers == 1:
        return collect_rollout(
            env,
            policy,
            cfg,
            device,
            seed=rollout_seed,
            archive_policy_state_dicts=archive_policy_state_dicts,
            policy_obs_dim=int(policy_obs_dim),
            critic_obs_dim=(
                int(critic_obs_dim)
                if critic_obs_dim is not None
                else central_critic_observation_dim(int(policy_obs_dim), cfg)
            ),
        )
    critic_obs_dim = int(
        critic_obs_dim
        if critic_obs_dim is not None
        else central_critic_observation_dim(int(policy_obs_dim), cfg)
    )

    if bool(getattr(cfg, "rollout_full_episodes", True)):
        total_episodes = _target_rollout_episodes(cfg)
        active_workers = min(num_workers, total_episodes)
        worker_episodes = [
            total_episodes // active_workers
            + (1 if worker_id < total_episodes % active_workers else 0)
            for worker_id in range(active_workers)
        ]
        max_episode_steps = max(1, int(cfg.max_episode_steps))
        worker_steps = [episodes * max_episode_steps for episodes in worker_episodes]
    else:
        active_workers = num_workers
        worker_steps = [
            total_steps // active_workers
            + (1 if worker_id < total_steps % active_workers else 0)
            for worker_id in range(active_workers)
        ]
        min_episodes = max(1, int(cfg.rollout_min_episodes))
        worker_episodes = [
            min_episodes // active_workers
            + (1 if worker_id < min_episodes % active_workers else 0)
            for worker_id in range(active_workers)
        ]
        worker_episodes = [max(1, episodes) for episodes in worker_episodes]

    cpu_state_dict = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    executor_context = (
        nullcontext(executor)
        if executor is not None
        else ProcessPoolExecutor(max_workers=active_workers, mp_context=mp.get_context("spawn"))
    )
    with executor_context as pool:
        futures = [
            pool.submit(
                _rollout_worker,
                cfg,
                cpu_state_dict,
                archive_policy_state_dicts,
                int(policy_obs_dim),
                int(critic_obs_dim),
                int(seed_offset) + worker_id,
                int(worker_steps[worker_id]),
                int(worker_episodes[worker_id]),
            )
            for worker_id in range(active_workers)
        ]
        batches = [future.result() for future in futures]
    return merge_rollout_batches(batches, cfg)

def make_rollout_executor(cfg: PSGAILConfig) -> ProcessPoolExecutor | None:
    num_workers = max(1, int(cfg.num_rollout_workers))
    if num_workers == 1:
        return None
    return ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn"))

def make_evaluation_executor(cfg: PSGAILConfig) -> ProcessPoolExecutor | None:
    requested_workers = int(getattr(cfg, "evaluation_num_workers", 1))
    if requested_workers <= 0:
        return None
    num_workers = max(1, requested_workers)
    return ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn"))

__all__ = [
    '_RolloutPerfProfiler',
    '_actions_to_rollout_array',
    '_assign_psro_sources',
    '_mixed_policy_actions',
    '_target_rollout_episodes',
    '_estimate_controlled_vehicles_for_target',
    '_target_aware_rollout_episodes',
    'refresh_rollout_rewards',
    'collect_rollout',
    'merge_rollout_batches',
    'rollout_training_agent_step_cap',
    '_trajectory_sample_indices',
    '_remap_trajectory_ids',
    '_subset_scene_features',
    '_subset_sequence_features',
    'subsample_rollout_for_training',
    'collect_round_rollouts',
    '_rollout_worker',
    'collect_rollouts',
    'make_rollout_executor',
    'make_evaluation_executor'
]
