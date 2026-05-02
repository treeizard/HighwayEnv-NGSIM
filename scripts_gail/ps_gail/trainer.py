from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

from .config import PSGAILConfig
from .data import (
    build_sequence_windows,
    discriminator_features,
    normalize_trajectory_frame,
    scene_snapshot_features,
    standardize_features,
    transform_sequence_features,
)
from .envs import controlled_vehicle_snapshot, make_training_env
from .models import NUM_DISCRETE_META_ACTIONS, make_actor_critic
from .observations import flatten_agent_observations, policy_observations_from_flat


@dataclass
class AgentTransition:
    policy_observation: np.ndarray
    action: object
    action_mask: np.ndarray
    log_prob: float
    value: float
    trajectory_id: int
    trajectory_state: np.ndarray
    scene_index: int
    env_penalty: float
    crashed: bool
    offroad: bool
    done: bool


@dataclass
class RolloutBatch:
    policy_observations: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    trajectory_ids: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    gail_rewards_raw: np.ndarray
    gail_rewards_normalized: np.ndarray
    env_penalties: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    generator_features: np.ndarray
    scene_features: np.ndarray
    transition_scene_indices: np.ndarray
    sequence_features: np.ndarray
    sequence_last_indices: np.ndarray
    num_env_steps: int
    num_agent_steps: int
    num_episodes: int = 0
    num_terminated: int = 0
    num_truncated: int = 0
    num_crash_events: int = 0
    num_offroad_events: int = 0
    crash_agent_fraction: float = 0.0
    offroad_agent_fraction: float = 0.0
    mean_env_penalty: float = 0.0
    mean_raw_gail_reward: float = 0.0
    mean_normalized_gail_reward: float = 0.0
    mean_episode_length: float = 0.0
    min_episode_length: int = 0
    max_episode_length: int = 0
    unique_episode_names: int = 0
    episode_names: tuple[str, ...] = ()
    mean_controlled_vehicles: float = 0.0
    mean_road_vehicles: float = 0.0


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def infer_policy_obs_dim(env: gym.Env) -> int:
    obs, _ = env.reset()
    return int(policy_observations_from_flat(flatten_agent_observations(obs)).shape[1])


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


def _as_device_tensor(array: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=dtype)
    if device.type == "cuda":
        tensor = tensor.pin_memory().to(device=device, non_blocking=True)
    else:
        tensor = tensor.to(device=device)
    return tensor


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


def policy_distribution_and_values(
    policy: nn.Module,
    obs_tensor: torch.Tensor,
    cfg: PSGAILConfig,
    action_masks: torch.Tensor | None = None,
) -> tuple[Categorical | Independent, torch.Tensor]:
    policy_out, values = policy(obs_tensor)
    if _is_continuous(cfg):
        if policy.log_std is None:
            raise RuntimeError("Continuous action mode requires policy.log_std.")
        std = torch.exp(policy.log_std).expand_as(policy_out)
        return Independent(Normal(policy_out, std), 1), values
    return Categorical(logits=_masked_discrete_logits(policy_out, action_masks)), values


def _sample_policy_actions(
    dist: Categorical | Independent,
    cfg: PSGAILConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = dist.sample()
    if _is_continuous(cfg):
        actions = torch.clamp(actions, -1.0, 1.0)
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


def _actions_to_rollout_array(transitions: list[AgentTransition], cfg: PSGAILConfig) -> np.ndarray:
    if _is_continuous(cfg):
        return np.stack(
            [np.asarray(tr.action, dtype=np.float32) for tr in transitions],
            axis=0,
        ).astype(np.float32)
    return np.asarray([tr.action for tr in transitions], dtype=np.int64)


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
    wgan_reward_clip: float = 0.0,
) -> np.ndarray:
    features = generator_features
    if feature_normalizer is not None:
        mean, std = feature_normalizer
        features = standardize_features(features, mean, std, clip=feature_clip)
    with torch.no_grad():
        logits = discriminator(_as_device_tensor(features, dtype=torch.float32, device=device))
        if str(loss_type).lower() == "wgan_gp":
            rewards = logits
            if float(wgan_reward_clip) > 0:
                rewards = torch.clamp(rewards, -float(wgan_reward_clip), float(wgan_reward_clip))
        else:
            rewards = F.softplus(logits)
    return rewards.cpu().numpy().astype(np.float32)


def shape_rollout_rewards(
    raw_gail_rewards: np.ndarray,
    env_penalties: np.ndarray,
    cfg: PSGAILConfig,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_gail_rewards, dtype=np.float32)
    penalties = np.asarray(env_penalties, dtype=np.float32)
    if raw.shape != penalties.shape:
        raise ValueError(f"Reward/penalty shape mismatch: {raw.shape} != {penalties.shape}")

    shaped_gail = raw.astype(np.float32, copy=True)
    if bool(cfg.normalize_gail_reward) and shaped_gail.size > 1:
        shaped_gail = (shaped_gail - shaped_gail.mean()) / (shaped_gail.std() + 1e-8)
    if float(cfg.gail_reward_clip) > 0:
        clip = float(cfg.gail_reward_clip)
        shaped_gail = np.clip(shaped_gail, -clip, clip)

    rewards = shaped_gail + penalties
    if float(cfg.final_reward_clip) > 0:
        clip = float(cfg.final_reward_clip)
        rewards = np.clip(rewards, -clip, clip)
    return rewards.astype(np.float32), shaped_gail.astype(np.float32)


def _target_rollout_episodes(cfg: PSGAILConfig) -> int:
    target = max(1, int(cfg.rollout_min_episodes))
    if bool(getattr(cfg, "rollout_full_episodes", True)):
        max_episode_steps = int(cfg.max_episode_steps)
        if max_episode_steps > 0:
            target = max(target, int(np.ceil(max(1, int(cfg.rollout_steps)) / max_episode_steps)))
    return target


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
            wgan_reward_clip=float(getattr(cfg, "wgan_reward_clip", 0.0)),
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
            wgan_reward_clip=float(getattr(cfg, "wgan_reward_clip", 0.0)),
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
            wgan_reward_clip=float(getattr(cfg, "wgan_reward_clip", 0.0)),
        )
        sequence_rewards = sequence_rewards * float(cfg.sequence_reward_coef)
        per_transition = np.zeros_like(combined_raw_gail_rewards, dtype=np.float32)
        counts = np.zeros_like(combined_raw_gail_rewards, dtype=np.float32)
        for reward, last_idx in zip(sequence_rewards, rollout.sequence_last_indices):
            last_idx = int(last_idx)
            if 0 <= last_idx < len(per_transition):
                per_transition[last_idx] += float(reward)
                counts[last_idx] += 1.0
        mask = counts > 0
        per_transition[mask] /= counts[mask]
        combined_raw_gail_rewards += per_transition
    rewards, normalized_gail_rewards = shape_rollout_rewards(
        combined_raw_gail_rewards,
        rollout.env_penalties,
        cfg,
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
        num_env_steps=rollout.num_env_steps,
        num_agent_steps=rollout.num_agent_steps,
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
) -> RolloutBatch:
    policy.eval()
    obs, _ = env.reset(seed=int(cfg.seed if seed is None else seed))
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    transitions: list[AgentTransition] = []
    scene_features: list[np.ndarray] = []
    collect_scene_features = bool(getattr(cfg, "enable_scene_discriminator", False))
    key_to_trajectory_id: dict[tuple[int, int], int] = {}
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

    collect_full_episodes = bool(getattr(cfg, "rollout_full_episodes", True))
    forced_reset_cap = 0 if collect_full_episodes else int(cfg.rollout_max_episode_steps)
    target_episodes = _target_rollout_episodes(cfg)

    while (
        len(episode_lengths) < target_episodes
        if collect_full_episodes
        else env_steps < int(cfg.rollout_steps) or len(episode_lengths) < target_episodes
    ):
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
        for key in keys:
            if key not in key_to_trajectory_id:
                key_to_trajectory_id[key] = len(key_to_trajectory_id)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
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
            dist, values = policy_distribution_and_values(policy, obs_tensor, cfg, action_mask_tensor)
            actions, log_probs = _sample_policy_actions(dist, cfg)

        action_tuple = _actions_to_env_tuple(actions, cfg)
        next_obs, _env_reward, terminated, truncated, info = env.step(action_tuple)
        episode_steps += 1
        force_rollout_reset = forced_reset_cap > 0 and episode_steps >= forced_reset_cap
        done = bool(terminated or truncated or force_rollout_reset)
        crash_flags = info.get("controlled_vehicle_crashes", [])
        offroad_flags = info.get("controlled_vehicle_offroad", [])
        episode_had_crash = bool(episode_had_crash or any(bool(flag) for flag in crash_flags))
        episode_had_offroad = bool(episode_had_offroad or any(bool(flag) for flag in offroad_flags))

        for i, key in enumerate(keys):
            crashed = bool(crash_flags[i]) if i < len(crash_flags) else False
            offroad = bool(offroad_flags[i]) if i < len(offroad_flags) else False
            env_penalty = 0.0
            if crashed:
                env_penalty -= float(cfg.collision_penalty)
            if offroad:
                env_penalty -= float(cfg.offroad_penalty)
            transitions.append(
                AgentTransition(
                    policy_observation=obs_agents[i].copy(),
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
                    done=done,
                )
            )

        env_steps += 1
        if done:
            episode_lengths.append(int(episode_steps))
            terminated_count += int(bool(terminated))
            truncated_count += int(bool(truncated or force_rollout_reset))
            crash_event_count += int(episode_had_crash)
            offroad_event_count += int(episode_had_offroad)
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

    policy_obs = np.stack([tr.policy_observation for tr in transitions], axis=0).astype(np.float32)
    trajectory_states = np.stack([tr.trajectory_state for tr in transitions], axis=0).astype(np.float32)
    actions = _actions_to_rollout_array(transitions, cfg)
    action_masks = np.stack([tr.action_mask for tr in transitions], axis=0).astype(bool)
    old_log_probs = np.asarray([tr.log_prob for tr in transitions], dtype=np.float32)
    old_values = np.asarray([tr.value for tr in transitions], dtype=np.float32)
    dones = np.asarray([tr.done for tr in transitions], dtype=bool)
    env_penalties = np.asarray([tr.env_penalty for tr in transitions], dtype=np.float32)
    transition_scene_indices = np.asarray([tr.scene_index for tr in transitions], dtype=np.int64)
    crashed = np.asarray([tr.crashed for tr in transitions], dtype=bool)
    offroad = np.asarray([tr.offroad for tr in transitions], dtype=bool)
    trajectory_ids = np.asarray([tr.trajectory_id for tr in transitions], dtype=np.int32)
    trajectory_states = normalize_trajectory_frame(
        trajectory_states,
        trajectory_ids,
        frame=cfg.trajectory_frame,
    )
    gen_features = discriminator_features(policy_obs, trajectory_states)
    sequence_features, sequence_last_indices = build_sequence_windows(
        gen_features,
        trajectory_ids,
        sequence_length=int(cfg.sequence_length),
        stride=int(cfg.sequence_stride),
    )
    sequence_features = transform_sequence_features(
        sequence_features,
        mode=str(getattr(cfg, "sequence_feature_mode", "raw")),
    )
    rewards = np.zeros(len(transitions), dtype=np.float32)
    returns, advantages = compute_returns_and_advantages(rewards, old_values, dones, trajectory_ids, cfg)
    return RolloutBatch(
        policy_observations=policy_obs,
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
        num_env_steps=env_steps,
        num_agent_steps=len(transitions),
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
        num_env_steps=sum(batch.num_env_steps for batch in batches),
        num_agent_steps=sum(batch.num_agent_steps for batch in batches),
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
    )


def _rollout_worker(
    cfg: PSGAILConfig,
    policy_state_dict: dict[str, torch.Tensor],
    policy_obs_dim: int,
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
    )
    policy.load_state_dict(policy_state_dict)
    policy.to(torch.device("cpu"))

    env = make_training_env(worker_cfg)
    try:
        return collect_rollout(
            env,
            policy,
            worker_cfg,
            torch.device("cpu"),
            seed=int(worker_cfg.seed),
        )
    finally:
        env.close()


def collect_rollouts(
    env: gym.Env,
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    policy_obs_dim: int,
    seed_offset: int = 0,
    executor: ProcessPoolExecutor | None = None,
) -> RolloutBatch:
    num_workers = max(1, int(cfg.num_rollout_workers))
    total_steps = max(1, int(cfg.rollout_steps))
    rollout_seed = int(cfg.seed) + int(seed_offset)
    if num_workers == 1:
        return collect_rollout(env, policy, cfg, device, seed=rollout_seed)

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
                int(policy_obs_dim),
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


def _wgan_gradient_penalty(
    discriminator: nn.Module,
    expert: torch.Tensor,
    generator: torch.Tensor,
    *,
    gp_lambda: float,
) -> torch.Tensor:
    alpha_shape = (expert.shape[0],) + (1,) * (expert.ndim - 1)
    alpha = torch.rand(alpha_shape, dtype=expert.dtype, device=expert.device)
    interpolated = (alpha * expert + (1.0 - alpha) * generator).requires_grad_(True)
    scores = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.shape[0], -1)
    gradient_norm = gradients.norm(2, dim=1)
    return float(gp_lambda) * torch.square(gradient_norm - 1.0).mean()


def update_discriminator(
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    expert_features: np.ndarray,
    generator_features: np.ndarray,
    cfg: PSGAILConfig,
    device: torch.device,
) -> dict[str, float]:
    expert_idx = np.random.choice(
        len(expert_features),
        size=len(generator_features),
        replace=len(expert_features) < len(generator_features),
    )
    expert = expert_features[expert_idx]
    expert_tensor = _as_device_tensor(expert.astype(np.float32, copy=False), dtype=torch.float32, device=device)
    generator_tensor = _as_device_tensor(
        generator_features.astype(np.float32, copy=False),
        dtype=torch.float32,
        device=device,
    )
    discriminator.train()
    losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []
    cgail_penalties: list[torch.Tensor] = []
    wgan_losses: list[torch.Tensor] = []
    gradient_penalties: list[torch.Tensor] = []
    prob_means: list[torch.Tensor] = []
    prob_stds: list[torch.Tensor] = []
    expert_prob_means: list[torch.Tensor] = []
    gen_prob_means: list[torch.Tensor] = []
    expert_score_means: list[torch.Tensor] = []
    gen_score_means: list[torch.Tensor] = []
    critic_gaps: list[torch.Tensor] = []
    expert_accs: list[torch.Tensor] = []
    gen_accs: list[torch.Tensor] = []
    expert_centered_accs: list[torch.Tensor] = []
    gen_centered_accs: list[torch.Tensor] = []
    cgail_k = max(0.0, float(getattr(cfg, "cgail_k", 0.0)))
    loss_type = str(getattr(cfg, "discriminator_loss", "bce")).lower()
    batch_size = max(1, int(cfg.disc_batch_size))
    num_pairs = int(generator_tensor.shape[0])
    if loss_type not in {"bce", "wgan_gp"}:
        raise ValueError(f"Unsupported discriminator_loss={loss_type!r}. Expected 'bce' or 'wgan_gp'.")

    if loss_type == "wgan_gp":
        gp_lambda = float(getattr(cfg, "wgan_gp_lambda", 2.0))
        for _ in range(int(cfg.disc_updates_per_round)):
            permutation = torch.randperm(num_pairs, device=device)
            for start in range(0, num_pairs, batch_size):
                batch_idx = permutation[start : start + batch_size]
                batch_expert = expert_tensor[batch_idx]
                batch_generator = generator_tensor[batch_idx]
                expert_scores = discriminator(batch_expert)
                gen_scores = discriminator(batch_generator)
                wgan_loss = gen_scores.mean() - expert_scores.mean()
                gradient_penalty = _wgan_gradient_penalty(
                    discriminator,
                    batch_expert,
                    batch_generator,
                    gp_lambda=gp_lambda,
                )
                loss = wgan_loss + gradient_penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    scores = torch.cat([expert_scores, gen_scores], dim=0)
                    probs = torch.sigmoid(scores)
                    centered_threshold = 0.5 * (expert_scores.mean() + gen_scores.mean())
                    prob_means.append(probs.mean().detach())
                    prob_stds.append(probs.std(unbiased=False).detach())
                    expert_prob_means.append(torch.sigmoid(expert_scores).mean().detach())
                    gen_prob_means.append(torch.sigmoid(gen_scores).mean().detach())
                    expert_score_means.append(expert_scores.mean().detach())
                    gen_score_means.append(gen_scores.mean().detach())
                    critic_gaps.append((expert_scores.mean() - gen_scores.mean()).detach())
                    expert_accs.append((expert_scores > 0.0).float().mean().detach())
                    gen_accs.append((gen_scores < 0.0).float().mean().detach())
                    expert_centered_accs.append(
                        (expert_scores > centered_threshold).float().mean().detach()
                    )
                    gen_centered_accs.append(
                        (gen_scores < centered_threshold).float().mean().detach()
                    )
                losses.append(loss.detach())
                wgan_losses.append(wgan_loss.detach())
                gradient_penalties.append(gradient_penalty.detach())
    else:
        x = torch.cat([expert_tensor, generator_tensor], dim=0)
        expert_label = float(cfg.disc_expert_label)
        generator_label = float(cfg.disc_generator_label)
        y = torch.cat(
            [
                torch.full((len(expert_tensor),), expert_label, dtype=torch.float32, device=device),
                torch.full((len(generator_tensor),), generator_label, dtype=torch.float32, device=device),
            ],
            dim=0,
        )
        num_samples = int(x.shape[0])
        for _ in range(int(cfg.disc_updates_per_round)):
            permutation = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, batch_size):
                batch_idx = permutation[start : start + batch_size]
                batch_x = x[batch_idx]
                batch_y = y[batch_idx]
                logits = discriminator(batch_x)
                bce_loss = F.binary_cross_entropy_with_logits(logits, batch_y)
                probs = torch.sigmoid(logits)
                cgail_penalty = 0.5 * cgail_k * torch.square(probs - 0.5).mean()
                loss = bce_loss + cgail_penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred = probs >= 0.5
                    expert_mask = batch_y > 0.5
                    gen_mask = batch_y < 0.5
                    prob_means.append(probs.mean().detach())
                    prob_stds.append(probs.std(unbiased=False).detach())
                    if expert_mask.any():
                        expert_accs.append((pred[expert_mask] == 1).float().mean().detach())
                        expert_centered_accs.append((pred[expert_mask] == 1).float().mean().detach())
                        expert_prob_means.append(probs[expert_mask].mean().detach())
                        expert_score_means.append(logits[expert_mask].mean().detach())
                    if gen_mask.any():
                        gen_accs.append((pred[gen_mask] == 0).float().mean().detach())
                        gen_centered_accs.append((pred[gen_mask] == 0).float().mean().detach())
                        gen_prob_means.append(probs[gen_mask].mean().detach())
                        gen_score_means.append(logits[gen_mask].mean().detach())
                    if expert_mask.any() and gen_mask.any():
                        critic_gaps.append(
                            (logits[expert_mask].mean() - logits[gen_mask].mean()).detach()
                        )
                losses.append(loss.detach())
                bce_losses.append(bce_loss.detach())
                cgail_penalties.append(cgail_penalty.detach())

    def mean_or_nan(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().cpu().item()) if values else float("nan")

    return {
        "disc_loss": mean_or_nan(losses),
        "disc_bce_loss": mean_or_nan(bce_losses),
        "cgail_penalty": mean_or_nan(cgail_penalties),
        "wgan_loss": mean_or_nan(wgan_losses),
        "gradient_penalty": mean_or_nan(gradient_penalties),
        "expert_score_mean": mean_or_nan(expert_score_means),
        "gen_score_mean": mean_or_nan(gen_score_means),
        "critic_gap": mean_or_nan(critic_gaps),
        "disc_prob_mean": mean_or_nan(prob_means),
        "disc_prob_std": mean_or_nan(prob_stds),
        "expert_prob_mean": mean_or_nan(expert_prob_means),
        "gen_prob_mean": mean_or_nan(gen_prob_means),
        "expert_acc": mean_or_nan(expert_accs),
        "gen_acc": mean_or_nan(gen_accs),
        "expert_centered_acc": mean_or_nan(expert_centered_accs),
        "gen_centered_acc": mean_or_nan(gen_centered_accs),
    }


def update_policy(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    cfg: PSGAILConfig,
    device: torch.device,
) -> dict[str, float]:
    was_training = policy.training
    # Rollout log-probabilities are collected with policy.eval(). Keeping PPO
    # forwards in eval mode prevents dropout from corrupting the likelihood
    # ratio, while gradients still flow through all parameters.
    policy.eval()
    action_tensor = (
        _as_device_tensor(rollout.actions, dtype=torch.float32, device=device)
        if _is_continuous(cfg)
        else _as_device_tensor(rollout.actions, dtype=torch.long, device=device)
    )
    obs_tensor = _as_device_tensor(rollout.policy_observations, dtype=torch.float32, device=device)
    action_mask_tensor = (
        _as_device_tensor(rollout.action_masks, dtype=torch.bool, device=device)
        if not _is_continuous(cfg)
        and bool(getattr(cfg, "enable_action_masking", True))
        and rollout.action_masks.size
        else None
    )
    old_log_probs_tensor = _as_device_tensor(rollout.old_log_probs, dtype=torch.float32, device=device)
    returns_tensor = _as_device_tensor(rollout.returns, dtype=torch.float32, device=device)
    advantages_tensor = _as_device_tensor(rollout.advantages, dtype=torch.float32, device=device)
    policy_losses: list[torch.Tensor] = []
    value_losses: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    approx_kls: list[torch.Tensor] = []
    clip_fractions: list[torch.Tensor] = []
    ratio_means: list[torch.Tensor] = []
    ratio_stds: list[torch.Tensor] = []
    batch_size = max(1, int(cfg.batch_size))
    num_samples = int(obs_tensor.shape[0])
    try:
        for _ in range(int(cfg.ppo_epochs)):
            permutation = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, batch_size):
                batch_idx = permutation[start : start + batch_size]
                obs = obs_tensor[batch_idx]
                actions = action_tensor[batch_idx]
                old_log_probs = old_log_probs_tensor[batch_idx]
                returns = returns_tensor[batch_idx]
                advantages = advantages_tensor[batch_idx]
                masks = action_mask_tensor[batch_idx] if action_mask_tensor is not None else None
                dist, values = policy_distribution_and_values(policy, obs, cfg, masks)
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
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > float(cfg.clip_range)).float().mean()
                    )
                    approx_kls.append(approx_kl.detach())
                    clip_fractions.append(clip_fraction.detach())
                    ratio_means.append(ratio.mean().detach())
                    ratio_stds.append(ratio.std(unbiased=False).detach())
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropies.append(entropy.detach())
    finally:
        if was_training:
            policy.train()

    return {
        "policy_loss": float(torch.stack(policy_losses).mean().cpu().item()),
        "value_loss": float(torch.stack(value_losses).mean().cpu().item()),
        "entropy": float(torch.stack(entropies).mean().cpu().item()),
        "approx_kl": float(torch.stack(approx_kls).mean().cpu().item()),
        "clip_fraction": float(torch.stack(clip_fractions).mean().cpu().item()),
        "ratio_mean": float(torch.stack(ratio_means).mean().cpu().item()),
        "ratio_std": float(torch.stack(ratio_stds).mean().cpu().item()),
    }
