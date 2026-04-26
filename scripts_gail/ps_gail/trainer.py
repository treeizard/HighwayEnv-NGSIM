from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from .config import PSGAILConfig
from .data import discriminator_features
from .envs import controlled_vehicle_snapshot, make_training_env
from .models import SharedActorCritic, TrajectoryDiscriminator
from .observations import flatten_agent_observations, policy_observations_from_flat


@dataclass
class AgentTransition:
    policy_observation: np.ndarray
    action: int
    log_prob: float
    value: float
    trajectory_id: int
    trajectory_state: np.ndarray
    done: bool


@dataclass
class RolloutBatch:
    policy_observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    trajectory_ids: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    generator_features: np.ndarray
    num_env_steps: int
    num_agent_steps: int


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def infer_policy_obs_dim(env: gym.Env) -> int:
    obs, _ = env.reset()
    return int(policy_observations_from_flat(flatten_agent_observations(obs)).shape[1])


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
    discriminator: TrajectoryDiscriminator,
    generator_features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        logits = discriminator(torch.as_tensor(generator_features, dtype=torch.float32, device=device))
        rewards = F.softplus(logits)
    return rewards.cpu().numpy().astype(np.float32)


def refresh_rollout_rewards(
    rollout: RolloutBatch,
    discriminator: TrajectoryDiscriminator,
    cfg: PSGAILConfig,
    device: torch.device,
) -> RolloutBatch:
    rewards = discriminator_reward(discriminator, rollout.generator_features, device)
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
        old_log_probs=rollout.old_log_probs,
        old_values=rollout.old_values,
        trajectory_ids=rollout.trajectory_ids,
        dones=rollout.dones,
        rewards=rewards,
        returns=returns,
        advantages=advantages,
        generator_features=rollout.generator_features,
        num_env_steps=rollout.num_env_steps,
        num_agent_steps=rollout.num_agent_steps,
    )


def collect_rollout(
    env: gym.Env,
    policy: SharedActorCritic,
    cfg: PSGAILConfig,
    device: torch.device,
    seed: int | None = None,
) -> RolloutBatch:
    policy.eval()
    obs, _ = env.reset(seed=int(cfg.seed if seed is None else seed))
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    transitions: list[AgentTransition] = []
    key_to_trajectory_id: dict[tuple[int, int], int] = {}
    episode_counter = 0
    env_steps = 0

    while env_steps < int(cfg.rollout_steps):
        vehicle_ids, trajectory_states = controlled_vehicle_snapshot(env)
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
            logits, values = policy(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        action_tuple = tuple(int(action) for action in actions.cpu().numpy().tolist())
        next_obs, _env_reward, terminated, truncated, _info = env.step(action_tuple)
        done = bool(terminated or truncated)

        for i, key in enumerate(keys):
            transitions.append(
                AgentTransition(
                    policy_observation=obs_agents[i].copy(),
                    action=int(action_tuple[i]),
                    log_prob=float(log_probs[i].cpu().item()),
                    value=float(values[i].cpu().item()),
                    trajectory_id=int(key_to_trajectory_id[key]),
                    trajectory_state=trajectory_states[i].copy(),
                    done=done,
                )
            )

        env_steps += 1
        if done:
            episode_counter += 1
            obs, _ = env.reset()
        else:
            obs = next_obs
        obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))

    policy_obs = np.stack([tr.policy_observation for tr in transitions], axis=0).astype(np.float32)
    trajectory_states = np.stack([tr.trajectory_state for tr in transitions], axis=0).astype(np.float32)
    gen_features = discriminator_features(policy_obs, trajectory_states)
    actions = np.asarray([tr.action for tr in transitions], dtype=np.int64)
    old_log_probs = np.asarray([tr.log_prob for tr in transitions], dtype=np.float32)
    old_values = np.asarray([tr.value for tr in transitions], dtype=np.float32)
    dones = np.asarray([tr.done for tr in transitions], dtype=bool)
    trajectory_ids = np.asarray([tr.trajectory_id for tr in transitions], dtype=np.int32)
    rewards = np.zeros(len(transitions), dtype=np.float32)
    returns, advantages = compute_returns_and_advantages(rewards, old_values, dones, trajectory_ids, cfg)
    return RolloutBatch(
        policy_observations=policy_obs,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        trajectory_ids=trajectory_ids,
        dones=dones,
        rewards=rewards,
        returns=returns,
        advantages=advantages,
        generator_features=gen_features,
        num_env_steps=env_steps,
        num_agent_steps=len(transitions),
    )


def merge_rollout_batches(batches: list[RolloutBatch], cfg: PSGAILConfig) -> RolloutBatch:
    if not batches:
        raise ValueError("Cannot merge an empty rollout batch list.")

    trajectory_ids: list[np.ndarray] = []
    trajectory_offset = 0
    for batch in batches:
        ids = batch.trajectory_ids.astype(np.int32, copy=True)
        if ids.size:
            ids += trajectory_offset
            trajectory_offset = int(ids.max()) + 1
        trajectory_ids.append(ids)

    rewards = np.concatenate([batch.rewards for batch in batches], axis=0).astype(np.float32)
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
        actions=np.concatenate([batch.actions for batch in batches], axis=0).astype(np.int64),
        old_log_probs=np.concatenate([batch.old_log_probs for batch in batches], axis=0).astype(np.float32),
        old_values=old_values,
        trajectory_ids=merged_trajectory_ids,
        dones=dones,
        rewards=rewards,
        returns=returns,
        advantages=advantages,
        generator_features=np.concatenate([batch.generator_features for batch in batches], axis=0).astype(
            np.float32
        ),
        num_env_steps=sum(batch.num_env_steps for batch in batches),
        num_agent_steps=sum(batch.num_agent_steps for batch in batches),
    )


def _rollout_worker(
    cfg: PSGAILConfig,
    policy_state_dict: dict[str, torch.Tensor],
    policy_obs_dim: int,
    worker_id: int,
    rollout_steps: int,
) -> RolloutBatch:
    threads = max(1, int(cfg.rollout_worker_threads))
    torch.set_num_threads(threads)
    np.random.seed(int(cfg.seed) + int(worker_id))

    worker_cfg = replace(
        cfg,
        rollout_steps=int(rollout_steps),
        seed=int(cfg.seed) + int(worker_id),
        device="cpu",
    )
    policy = SharedActorCritic(int(policy_obs_dim), int(cfg.hidden_size))
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
    policy: SharedActorCritic,
    cfg: PSGAILConfig,
    device: torch.device,
    policy_obs_dim: int,
) -> RolloutBatch:
    num_workers = max(1, int(cfg.num_rollout_workers))
    total_steps = max(1, int(cfg.rollout_steps))
    if num_workers == 1:
        return collect_rollout(env, policy, cfg, device)

    num_workers = min(num_workers, total_steps)
    base_steps = total_steps // num_workers
    extra_steps = total_steps % num_workers
    steps_by_worker = [base_steps + (1 if worker_id < extra_steps else 0) for worker_id in range(num_workers)]

    cpu_state_dict = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
        futures = [
            executor.submit(
                _rollout_worker,
                cfg,
                cpu_state_dict,
                int(policy_obs_dim),
                worker_id,
                steps,
            )
            for worker_id, steps in enumerate(steps_by_worker)
        ]
        batches = [future.result() for future in futures]
    return merge_rollout_batches(batches, cfg)


def update_discriminator(
    discriminator: TrajectoryDiscriminator,
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
    x = np.concatenate([expert, generator_features], axis=0).astype(np.float32)
    y = np.concatenate(
        [np.ones(len(expert), dtype=np.float32), np.zeros(len(generator_features), dtype=np.float32)],
        axis=0,
    )
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x), torch.as_tensor(y)),
        batch_size=int(cfg.disc_batch_size),
        shuffle=True,
    )
    discriminator.train()
    losses: list[float] = []
    expert_accs: list[float] = []
    gen_accs: list[float] = []
    for _ in range(int(cfg.disc_updates_per_round)):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)
            logits = discriminator(batch_x)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = torch.sigmoid(logits) >= 0.5
                expert_mask = batch_y >= 0.5
                gen_mask = ~expert_mask
                if expert_mask.any():
                    expert_accs.append(float((pred[expert_mask] == 1).float().mean().cpu().item()))
                if gen_mask.any():
                    gen_accs.append(float((pred[gen_mask] == 0).float().mean().cpu().item()))
            losses.append(float(loss.detach().cpu().item()))
    return {
        "disc_loss": float(np.mean(losses)),
        "expert_acc": float(np.mean(expert_accs)) if expert_accs else float("nan"),
        "gen_acc": float(np.mean(gen_accs)) if gen_accs else float("nan"),
    }


def update_policy(
    policy: SharedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    cfg: PSGAILConfig,
    device: torch.device,
) -> dict[str, float]:
    policy.train()
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(rollout.policy_observations, dtype=torch.float32),
            torch.as_tensor(rollout.actions, dtype=torch.long),
            torch.as_tensor(rollout.old_log_probs, dtype=torch.float32),
            torch.as_tensor(rollout.returns, dtype=torch.float32),
            torch.as_tensor(rollout.advantages, dtype=torch.float32),
        ),
        batch_size=int(cfg.batch_size),
        shuffle=True,
    )
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    for _ in range(int(cfg.ppo_epochs)):
        for obs, actions, old_log_probs, returns, advantages in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            old_log_probs = old_log_probs.to(device)
            returns = returns.to(device)
            advantages = advantages.to(device)
            logits, values = policy(obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * advantages,
            ).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            entropies.append(float(entropy.detach().cpu().item()))
    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }
