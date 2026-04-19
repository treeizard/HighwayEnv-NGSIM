#!/usr/bin/env python3
from __future__ import annotations

"""Parameter-sharing raw-trajectory GAIL for NGSIM discrete meta-actions.

This trainer keeps the policy interface discrete, but removes inferred discrete
expert actions from the discriminator target. Expert discriminator samples are
loaded from a dataset built by ``build_ps_traj_expert_discrete.py``. Generator
samples are produced by applying one shared categorical policy independently to
every controlled vehicle in a scene replay.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import (  # noqa: E402
    ENV_ID,
    build_env_config,
    register_ngsim_env,
)
from build_ps_traj_expert_discrete import (  # noqa: E402
    discriminator_features as _discriminator_features,
    flatten_agent_observations,
    load_ps_traj_expert_dataset,
)


NUM_DISCRETE_ACTIONS = 5


@dataclass
class AgentTransition:
    observation: np.ndarray
    action: int
    log_prob: float
    value: float
    trajectory_id: int
    trajectory_state: np.ndarray
    done: bool


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    trajectory_ids: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    generator_trajectory_states: np.ndarray
    generator_features: np.ndarray
    mean_episode_reward: float
    num_env_steps: int
    num_agent_steps: int


class SharedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, num_actions: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        return self.policy_head(encoded), self.value_head(encoded).squeeze(-1)


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a parameter-sharing discrete policy with raw-trajectory GAIL. "
            "Expert states come from raw NGSIM trajectories; generator states come "
            "from rolling the shared policy on all vehicles in the simulator."
        )
    )
    parser.add_argument("--scene", type=str, default="us-101")
    parser.add_argument("--episode-root", type=str, default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", choices=["train", "val", "test"], default="train")
    parser.add_argument(
        "--expert-data",
        type=str,
        default="expert_data/ngsim_ps_traj_expert_discrete.npz",
        help="Path built by scripts_gail/build_ps_traj_expert_discrete.py.",
    )
    parser.add_argument("--run-name", type=str, default="ps_traj_gail_discrete")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-rounds", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        default=True,
        help="Apply the shared policy to every viable controlled vehicle in each scene.",
    )
    parser.add_argument(
        "--no-control-all-vehicles",
        dest="control_all_vehicles",
        action="store_false",
        help="Debug/ablation mode: control only --controlled-vehicles vehicles.",
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=4,
        help="Number of controlled vehicles when --no-control-all-vehicles is used.",
    )
    parser.add_argument(
        "--max-controlled-vehicles",
        type=int,
        default=0,
        help=(
            "Optional cap for --control-all-vehicles scenes. "
            "0 keeps every start-aligned vehicle."
        ),
    )
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--maximum-range", type=float, default=64.0)
    parser.add_argument("--simulation-frequency", type=int, default=10)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--disc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--disc-batch-size", type=int, default=1024)
    parser.add_argument("--disc-updates-per-round", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_scene_env(args: argparse.Namespace) -> gym.Env:
    observation_config = {
        "type": "LidarCameraObservations",
        "lidar": {
            "cells": int(args.cells),
            "maximum_range": float(args.maximum_range),
            "normalize": True,
        },
        "camera": {
            "cells": 21,
            "maximum_range": float(args.maximum_range),
            "field_of_view": np.pi / 2,
            "normalize": True,
        },
    }
    cfg = build_env_config(
        scene=args.scene,
        action_mode="discrete",
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        controlled_vehicles=max(1, int(args.controlled_vehicles)),
        control_all_vehicles=bool(args.control_all_vehicles),
        max_surrounding=args.max_surrounding,
        observation_config=observation_config,
        simulation_frequency=args.simulation_frequency,
        policy_frequency=args.policy_frequency,
        max_episode_steps=args.max_episode_steps,
        seed=None,
        scene_dataset_collection_mode=False,
    )
    cfg["expert_test_mode"] = False
    cfg["disable_controlled_vehicle_collisions"] = True
    cfg["terminate_when_all_controlled_crashed"] = False
    if int(args.max_controlled_vehicles) > 0:
        cfg["max_controlled_vehicles"] = int(args.max_controlled_vehicles)
    return gym.make(ENV_ID, config=cfg)


def controlled_vehicle_snapshot(env: gym.Env) -> tuple[list[int], np.ndarray, np.ndarray]:
    base = env.unwrapped
    vehicles = list(getattr(base, "controlled_vehicles", []))
    ids = [int(getattr(vehicle, "vehicle_ID", idx)) for idx, vehicle in enumerate(vehicles)]
    xy = np.asarray([vehicle.position for vehicle in vehicles], dtype=np.float32)
    speed = np.asarray([float(vehicle.speed) for vehicle in vehicles], dtype=np.float32)
    return ids, xy, speed


def raw_trajectory_states_from_snapshot(xy: np.ndarray, speed: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    speed = np.asarray(speed, dtype=np.float32).reshape(-1, 1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy [N, 2], got {xy.shape}.")
    if len(xy) != len(speed):
        raise ValueError(f"xy/speed count mismatch: {len(xy)} != {len(speed)}.")
    return np.concatenate([xy, speed], axis=1).astype(np.float32, copy=False)




def discriminator_reward(
    discriminator: TrajectoryDiscriminator,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        logits = discriminator(torch.as_tensor(features, dtype=torch.float32, device=device))
        reward = F.softplus(logits)
    return reward.cpu().numpy().astype(np.float32)


def compute_returns_and_advantages(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    trajectory_ids: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    returns = np.zeros_like(rewards, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    for trajectory_id in np.unique(trajectory_ids):
        indices = np.where(trajectory_ids == trajectory_id)[0]
        next_advantage = 0.0
        next_value = 0.0
        for idx in reversed(indices):
            nonterminal = 0.0 if dones[idx] else 1.0
            delta = rewards[idx] + args.gamma * next_value * nonterminal - values[idx]
            next_advantage = delta + args.gamma * args.gae_lambda * nonterminal * next_advantage
            advantages[idx] = next_advantage
            returns[idx] = advantages[idx] + values[idx]
            next_value = values[idx]

    if advantages.size > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def refresh_rollout_rewards(
    *,
    rollout: RolloutBatch,
    discriminator: TrajectoryDiscriminator,
    args: argparse.Namespace,
    device: torch.device,
) -> RolloutBatch:
    rewards = np.zeros(len(rollout.observations), dtype=np.float32)
    if len(rollout.generator_features):
        rewards_for_valid = discriminator_reward(discriminator, rollout.generator_features, device)
        rewards[:] = rewards_for_valid
    returns, advantages = compute_returns_and_advantages(
        rewards=rewards,
        values=rollout.old_values,
        dones=rollout.dones,
        trajectory_ids=rollout.trajectory_ids,
        args=args,
    )
    return RolloutBatch(
        observations=rollout.observations,
        actions=rollout.actions,
        old_log_probs=rollout.old_log_probs,
        old_values=rollout.old_values,
        trajectory_ids=rollout.trajectory_ids,
        dones=rollout.dones,
        rewards=rewards,
        returns=returns,
        advantages=advantages,
        generator_trajectory_states=rollout.generator_trajectory_states,
        generator_features=rollout.generator_features,
        mean_episode_reward=float(rewards.mean()) if rewards.size else 0.0,
        num_env_steps=rollout.num_env_steps,
        num_agent_steps=rollout.num_agent_steps,
    )


def collect_rollout(
    *,
    env: gym.Env,
    policy: SharedActorCritic,
    args: argparse.Namespace,
    device: torch.device,
) -> RolloutBatch:
    policy.eval()

    obs, _ = env.reset()
    obs_agents = flatten_agent_observations(obs)
    transitions: list[AgentTransition] = []
    key_to_trajectory_id: dict[tuple[int, int], int] = {}
    episode_counter = 0
    env_steps = 0

    while env_steps < args.rollout_steps:
        ids, xy, speed = controlled_vehicle_snapshot(env)
        if len(ids) != len(obs_agents):
            raise RuntimeError(
                f"Observation/vehicle mismatch: obs_agents={len(obs_agents)} vehicles={len(ids)}"
            )

        keys = [(episode_counter, int(vehicle_id)) for vehicle_id in ids]
        trajectory_states = raw_trajectory_states_from_snapshot(xy, speed)
        for key in keys:
            if key not in key_to_trajectory_id:
                key_to_trajectory_id[key] = len(key_to_trajectory_id)

        obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, values = policy(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        action_tuple = tuple(int(a) for a in actions.cpu().numpy().tolist())
        next_obs, _env_reward, terminated, truncated, _info = env.step(action_tuple)
        done = bool(terminated or truncated)

        for i, key in enumerate(keys):
            transitions.append(
                AgentTransition(
                    observation=obs_agents[i].copy(),
                    action=int(action_tuple[i]),
                    log_prob=float(log_probs[i].cpu().item()),
                    value=float(values[i].cpu().item()),
                    trajectory_id=int(key_to_trajectory_id[key]),
                    trajectory_state=trajectory_states[i].copy(),
                    done=done,
                )
            )

        env_steps += 1
        obs_agents = flatten_agent_observations(next_obs)

        if done:
            episode_counter += 1
            obs, _ = env.reset()
            obs_agents = flatten_agent_observations(obs)

    rewards = np.zeros(len(transitions), dtype=np.float32)

    observations = np.stack([tr.observation for tr in transitions], axis=0).astype(np.float32)
    generator_states_arr = np.stack(
        [tr.trajectory_state for tr in transitions],
        axis=0,
    ).astype(np.float32, copy=False)
    gen_features_arr = _discriminator_features(observations, generator_states_arr)
    actions = np.asarray([tr.action for tr in transitions], dtype=np.int64)
    old_log_probs = np.asarray([tr.log_prob for tr in transitions], dtype=np.float32)
    old_values = np.asarray([tr.value for tr in transitions], dtype=np.float32)
    dones = np.asarray([tr.done for tr in transitions], dtype=bool)
    trajectory_ids = np.asarray(
        [tr.trajectory_id for tr in transitions],
        dtype=np.int32,
    )
    returns, advantages = compute_returns_and_advantages(
        rewards=rewards,
        values=old_values,
        dones=dones,
        trajectory_ids=trajectory_ids,
        args=args,
    )

    return RolloutBatch(
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        trajectory_ids=trajectory_ids,
        dones=dones,
        rewards=rewards,
        returns=returns,
        advantages=advantages,
        generator_trajectory_states=generator_states_arr,
        generator_features=gen_features_arr,
        mean_episode_reward=float(rewards.mean()) if rewards.size else 0.0,
        num_env_steps=env_steps,
        num_agent_steps=len(transitions),
    )


def update_discriminator(
    *,
    discriminator: TrajectoryDiscriminator,
    optimizer: torch.optim.Optimizer,
    expert_features: np.ndarray,
    generator_features: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if len(generator_features) == 0:
        return {"disc_loss": float("nan"), "expert_acc": float("nan"), "gen_acc": float("nan")}

    expert_count = len(generator_features)
    expert_idx = np.random.choice(
        len(expert_features),
        size=expert_count,
        replace=len(expert_features) < expert_count,
    )
    expert = expert_features[expert_idx]
    generated = generator_features
    x = np.concatenate([expert, generated], axis=0).astype(np.float32)
    y = np.concatenate(
        [np.ones(len(expert), dtype=np.float32), np.zeros(len(generated), dtype=np.float32)],
        axis=0,
    )
    dataset = TensorDataset(torch.as_tensor(x), torch.as_tensor(y))
    loader = DataLoader(dataset, batch_size=args.disc_batch_size, shuffle=True)

    discriminator.train()
    losses: list[float] = []
    expert_accs: list[float] = []
    gen_accs: list[float] = []
    for _ in range(args.disc_updates_per_round):
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
    *,
    policy: SharedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    policy.train()
    dataset = TensorDataset(
        torch.as_tensor(rollout.observations, dtype=torch.float32),
        torch.as_tensor(rollout.actions, dtype=torch.long),
        torch.as_tensor(rollout.old_log_probs, dtype=torch.float32),
        torch.as_tensor(rollout.returns, dtype=torch.float32),
        torch.as_tensor(rollout.advantages, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    for _ in range(args.ppo_epochs):
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
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            entropies.append(float(entropy.detach().cpu().item()))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }


def infer_obs_dim(env: gym.Env) -> int:
    obs, _ = env.reset()
    obs_agents = flatten_agent_observations(obs)
    return int(obs_agents.shape[1])


def main() -> None:
    args = parse_args()
    register_ngsim_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)

    run_dir = os.path.abspath(os.path.join("logs", "ps_traj_gail_discrete", args.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    expert_data_path = os.path.abspath(args.expert_data)
    expert_features, expert_metadata, _ = load_ps_traj_expert_dataset(expert_data_path)
    expert_obs_dim = int(expert_metadata.get("observation_dim", 0))
    expert_traj_state_dim = int(expert_metadata.get("trajectory_state_dim", 0))

    env = make_scene_env(args)
    try:
        obs_dim = infer_obs_dim(env)
        if obs_dim != expert_obs_dim:
            raise RuntimeError(
                f"Expert obs_dim={expert_obs_dim} does not match generator obs_dim={obs_dim}."
            )
        trajectory_state_dim = int(expert_features.shape[1] - obs_dim)
        if expert_traj_state_dim and trajectory_state_dim != expert_traj_state_dim:
            raise RuntimeError(
                f"Expert trajectory_state_dim={expert_traj_state_dim} does not match "
                f"feature-derived dim={trajectory_state_dim}."
            )
        if trajectory_state_dim != 3:
            raise RuntimeError(
                f"This trainer expects per-step raw trajectory states [x, y, speed], "
                f"got dim={trajectory_state_dim}."
            )
        feature_dim = obs_dim + trajectory_state_dim
        policy = SharedActorCritic(
            obs_dim=obs_dim,
            hidden_size=args.hidden_size,
            num_actions=NUM_DISCRETE_ACTIONS,
        ).to(device)
        discriminator = TrajectoryDiscriminator(
            input_dim=feature_dim,
            hidden_size=args.hidden_size,
        ).to(device)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_learning_rate)

        print(f"Loaded expert sensor/trajectory features: {expert_data_path}")
        print(f"expert_features={expert_features.shape}")
        print(
            f"obs_dim={obs_dim} trajectory_state_dim={trajectory_state_dim} "
            f"disc_feature_dim={feature_dim} device={device}"
        )
        print(f"Training parameter-sharing trajectory GAIL for {args.total_rounds} rounds")

        for round_idx in range(1, args.total_rounds + 1):
            rollout = collect_rollout(
                env=env,
                policy=policy,
                args=args,
                device=device,
            )
            disc_stats = update_discriminator(
                discriminator=discriminator,
                optimizer=disc_optimizer,
                expert_features=expert_features,
                generator_features=rollout.generator_features,
                args=args,
                device=device,
            )
            rollout = refresh_rollout_rewards(
                rollout=rollout,
                discriminator=discriminator,
                args=args,
                device=device,
            )
            policy_stats = update_policy(
                policy=policy,
                optimizer=policy_optimizer,
                rollout=rollout,
                args=args,
                device=device,
            )

            print(
                f"[round {round_idx:04d}] "
                f"env_steps={rollout.num_env_steps} agent_steps={rollout.num_agent_steps} "
                f"gen_states={len(rollout.generator_trajectory_states)} "
                f"disc_loss={disc_stats['disc_loss']:.4f} "
                f"expert_acc={disc_stats['expert_acc']:.3f} gen_acc={disc_stats['gen_acc']:.3f} "
                f"policy_loss={policy_stats['policy_loss']:.4f} "
                f"value_loss={policy_stats['value_loss']:.4f} "
                f"entropy={policy_stats['entropy']:.4f} "
                f"mean_reward={rollout.mean_episode_reward:.4f}"
            )

            if args.checkpoint_every > 0 and round_idx % args.checkpoint_every == 0:
                torch.save(
                    {
                        "round": round_idx,
                        "policy_state_dict": policy.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "args": vars(args),
                    },
                    os.path.join(ckpt_dir, f"traj_gail_round_{round_idx:04d}.pt"),
                )

        torch.save(
            {
                "round": args.total_rounds,
                "policy_state_dict": policy.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "args": vars(args),
            },
            os.path.join(run_dir, "traj_gail_final.pt"),
        )
    finally:
        env.close()

    print(f"Saved final trajectory-GAIL checkpoint under: {run_dir}")


if __name__ == "__main__":
    main()
