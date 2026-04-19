#!/usr/bin/env python3
from __future__ import annotations

"""Burn-InfoGAIL-style discrete NGSIM demo.

This script is a project-aligned demo of Burn-InfoGAIL ideas on top of the
existing NGSIM replay + `imitation` stack used in this repository.

What it adds compared with the plain PS-GAIL baseline:

- learns a latent driving-style code from a burn-in prefix of expert data
- augments expert transitions with the inferred latent code
- augments generator observations with the same latent code when the replay
  episode matches a known expert trajectory

This version pushes the Burn-InfoGAIL-style auxiliary objectives directly into
the discriminator update rather than only using them during a separate latent
pretraining stage.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box
from gymnasium.wrappers import FlattenObservation
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import logger as imitation_logger
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from torch.utils.data import DataLoader, Dataset


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import load_expert_dataset, register_ngsim_env


NUM_DISCRETE_ACTIONS = 5


@dataclass(frozen=True)
class TrajectoryKey:
    episode_name: str
    agent_id: int


@dataclass
class BurnInTrajectory:
    key: TrajectoryKey
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    timesteps: np.ndarray
    scenario_id: str
    episode_id: int


class BurnInDataset(Dataset):
    def __init__(self, trajectories: list[BurnInTrajectory], burn_in_steps: int) -> None:
        self.samples: list[dict[str, torch.Tensor | TrajectoryKey]] = []
        self.burn_in_steps = int(max(1, burn_in_steps))

        for traj in trajectories:
            T = int(len(traj.actions))
            if T < 2:
                continue

            prefix_len = min(self.burn_in_steps, T - 1)
            burn_obs = traj.observations[:prefix_len]
            burn_actions = traj.actions[:prefix_len]
            future_obs = traj.observations[prefix_len:]
            future_actions = traj.actions[prefix_len:]
            if len(future_actions) == 0:
                continue

            self.samples.append(
                {
                    "key": traj.key,
                    "burn_obs": torch.as_tensor(burn_obs, dtype=torch.float32),
                    "burn_actions": torch.as_tensor(burn_actions, dtype=torch.long),
                    "future_obs": torch.as_tensor(future_obs, dtype=torch.float32),
                    "future_actions": torch.as_tensor(future_actions, dtype=torch.long),
                }
            )

        if not self.samples:
            raise RuntimeError(
                "No burn-in samples could be created. "
                "Try reducing --burn-in-steps or using a larger expert dataset."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | TrajectoryKey]:
        return self.samples[index]


class BurnInPosterior(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        num_actions: int,
        num_codes: int,
        latent_dim: int,
        hidden_size: int,
        action_embed_dim: int,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_size)
        self.rnn = nn.GRU(
            input_size=hidden_size + action_embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.logits_head = nn.Linear(hidden_size, num_codes)
        self.codebook = nn.Embedding(num_codes, latent_dim)

    def forward(
        self,
        burn_obs: torch.Tensor,
        burn_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_feat = torch.tanh(self.obs_proj(burn_obs))
        act_feat = self.action_embedding(burn_actions)
        seq = torch.cat([obs_feat, act_feat], dim=-1)
        _, hidden = self.rnn(seq)
        hidden_last = hidden[-1]
        logits = self.logits_head(hidden_last)
        probs = torch.softmax(logits, dim=-1)
        latent = probs @ self.codebook.weight
        return logits, probs, latent


class ActionDecoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden_size: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, observations: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observations, latent], dim=-1)
        return self.net(x)


class BurnInfoRewardNet(BasicRewardNet):
    def __init__(
        self,
        *,
        observation_space,
        action_space,
        obs_dim: int,
        latent_dim: int,
        hidden_size: int,
        num_codes: int,
        num_actions: int,
        codebook_weight: np.ndarray,
        decoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            **kwargs,
        )
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.num_codes = int(num_codes)
        self.num_actions = int(num_actions)
        self.decoder = ActionDecoder(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            hidden_size=hidden_size,
            num_actions=self.num_actions,
        )
        if decoder_state_dict is not None:
            self.decoder.load_state_dict(decoder_state_dict)

        transition_dim = (2 * self.obs_dim) + self.latent_dim + self.num_actions + 1
        self.transition_encoder = nn.Sequential(
            nn.Linear(transition_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.latent_head = nn.Linear(hidden_size, self.num_codes)
        self.register_buffer(
            "codebook",
            torch.as_tensor(codebook_weight, dtype=torch.float32),
        )

    def _split_augmented_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_obs = state[..., : self.obs_dim]
        latent = state[..., self.obs_dim : self.obs_dim + self.latent_dim]
        return raw_obs, latent

    def _action_indices(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim > 1:
            action = action.reshape(-1)
        return action.long()

    def transition_code_logits(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        raw_obs, latent = self._split_augmented_state(state)
        next_raw_obs, _ = self._split_augmented_state(next_state)
        action_idx = self._action_indices(action)
        action_one_hot = F.one_hot(action_idx, num_classes=self.num_actions).float()
        done_feat = done.float().reshape(-1, 1)
        features = torch.cat(
            [raw_obs, next_raw_obs, latent, action_one_hot, done_feat],
            dim=-1,
        )
        encoded = self.transition_encoder(features)
        return self.latent_head(encoded)

    def latent_targets_from_state(self, state: torch.Tensor) -> torch.Tensor:
        _, latent = self._split_augmented_state(state)
        distances = torch.cdist(latent, self.codebook)
        return torch.argmin(distances, dim=-1)

    def action_reconstruction_logits(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_obs, latent = self._split_augmented_state(state)
        return self.decoder(raw_obs, latent), latent


class BurnInfoGAIL(GAIL):
    def __init__(
        self,
        *,
        info_loss_coef: float,
        recon_loss_coef: float,
        prior_kl_coef: float,
        latent_entropy_coef: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.info_loss_coef = float(info_loss_coef)
        self.recon_loss_coef = float(recon_loss_coef)
        self.prior_kl_coef = float(prior_kl_coef)
        self.latent_entropy_coef = float(latent_entropy_coef)
        if not isinstance(self._reward_net, BurnInfoRewardNet):
            raise TypeError("BurnInfoGAIL requires a BurnInfoRewardNet.")

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        with self.logger.accumulate_means("disc"):
            self._disc_opt.zero_grad()

            last_stats: dict[str, float] = {}
            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )
            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                adv_loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    batch["labels_expert_is_one"].float(),
                )

                code_logits = self._reward_net.transition_code_logits(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                )
                latent_targets = self._reward_net.latent_targets_from_state(batch["state"])
                info_loss = F.cross_entropy(code_logits, latent_targets)

                action_logits, _ = self._reward_net.action_reconstruction_logits(batch["state"])
                action_targets = batch["action"].reshape(-1).long()
                recon_loss = F.cross_entropy(action_logits, action_targets)

                probs = torch.softmax(code_logits, dim=-1)
                avg_probs = probs.mean(dim=0)
                uniform = torch.full_like(avg_probs, 1.0 / float(avg_probs.numel()))
                prior_kl_loss = torch.sum(
                    avg_probs * (torch.log(avg_probs + 1e-8) - torch.log(uniform + 1e-8))
                )
                entropy_bonus = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

                total_loss = (
                    adv_loss
                    + (self.info_loss_coef * info_loss)
                    + (self.recon_loss_coef * recon_loss)
                    + (self.prior_kl_coef * prior_kl_loss)
                    - (self.latent_entropy_coef * entropy_bonus)
                )

                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                loss = total_loss * (self.demo_minibatch_size / self.demo_batch_size)
                loss.backward()

                with torch.no_grad():
                    expert_mask = batch["labels_expert_is_one"].bool()
                    gen_mask = ~expert_mask
                    expert_code_acc = (
                        (code_logits[expert_mask].argmax(dim=-1) == latent_targets[expert_mask])
                        .float()
                        .mean()
                        .item()
                    )
                    gen_code_acc = (
                        (code_logits[gen_mask].argmax(dim=-1) == latent_targets[gen_mask])
                        .float()
                        .mean()
                        .item()
                    )
                    disc_acc = (
                        ((disc_logits > 0) == batch["labels_expert_is_one"].bool())
                        .float()
                        .mean()
                        .item()
                    )
                    last_stats = {
                        "disc_loss": float(loss.item()),
                        "disc_adv_loss": float(adv_loss.item()),
                        "disc_info_loss": float(info_loss.item()),
                        "disc_recon_loss": float(recon_loss.item()),
                        "disc_prior_kl_loss": float(prior_kl_loss.item()),
                        "disc_entropy_bonus": float(entropy_bonus.item()),
                        "disc_acc": disc_acc,
                        "disc_code_acc_expert": expert_code_acc,
                        "disc_code_acc_gen": gen_code_acc,
                    }

            self._disc_opt.step()
            self._disc_step += 1
            self.logger.record("global_step", self._global_step)
            for key, value in last_stats.items():
                self.logger.record(key, value)
            self.logger.dump(self._disc_step)
        return last_stats


class LatentObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        latent_lookup: dict[TrajectoryKey, np.ndarray],
        latent_dim: int,
        fallback_latent: np.ndarray,
    ) -> None:
        super().__init__(env)
        self.latent_lookup = latent_lookup
        self.latent_dim = int(latent_dim)
        self.fallback_latent = np.asarray(fallback_latent, dtype=np.float32).reshape(self.latent_dim)
        base_space = self.observation_space
        if not isinstance(base_space, Box):
            raise TypeError("LatentObservationWrapper expects a Box observation space after flattening.")
        low = np.full((self.latent_dim,), -np.inf, dtype=np.float32)
        high = np.full((self.latent_dim,), np.inf, dtype=np.float32)
        self.observation_space = Box(
            low=np.concatenate([base_space.low.astype(np.float32), low], axis=0),
            high=np.concatenate([base_space.high.astype(np.float32), high], axis=0),
            dtype=np.float32,
        )
        self.current_latent = self.fallback_latent.copy()

    def _resolve_key(self) -> TrajectoryKey | None:
        base = self.unwrapped
        episode_name = getattr(base, "episode_name", None)
        ego_id = getattr(base, "ego_id", None)
        if episode_name is None or ego_id is None:
            return None
        return TrajectoryKey(str(episode_name), int(ego_id))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        key = self._resolve_key()
        self.current_latent = self.latent_lookup.get(key, self.fallback_latent).astype(np.float32, copy=True)
        if key is not None:
            info = dict(info)
            info["burn_info_key"] = {"episode_name": key.episode_name, "agent_id": key.agent_id}
        return self.observation(obs), info

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        return np.concatenate([obs, self.current_latent], axis=0).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Burn-InfoGAIL-style discrete NGSIM demo by learning a latent "
            "style code from burn-in prefixes and concatenating that code to policy inputs."
        )
    )
    parser.add_argument(
        "--expert-data",
        type=str,
        default="expert_data/ngsim_single_train_episode_discrete.npz",
        help="Path to a saved discrete expert dataset (.npz).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="burn_info_gail_discrete",
        help="Run name under logs/burn_info_gail_discrete/.",
    )
    parser.add_argument("--scene", type=str, default="us-101")
    parser.add_argument("--episode-root", type=str, default="highway_env/data/processed_20s")
    parser.add_argument(
        "--prebuilt-split",
        choices=["train", "val", "test"],
        default="train",
        help="Replay split used for generator rollouts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--simulation-frequency", type=int, default=10)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--max-surrounding", default=32)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--maximum-range", type=float, default=64.0)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--demo-batch-size", type=int, default=512)
    parser.add_argument("--disc-updates-per-round", type=int, default=4)
    parser.add_argument("--gen-replay-buffer-capacity", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument(
        "--dummy-vec",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv for easier debugging.",
    )
    parser.add_argument(
        "--skip-inactive-agents",
        action="store_true",
        default=True,
        help=(
            "When loading a scene dataset, drop padded transitions whose alive_mask is false."
        ),
    )
    parser.add_argument(
        "--keep-inactive-agents",
        dest="skip_inactive_agents",
        action="store_false",
        help="Keep inactive padded scene slots in the expert transition set.",
    )
    parser.add_argument(
        "--burn-in-steps",
        type=int,
        default=8,
        help="Number of expert prefix steps used to infer the latent code.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Dimension of the continuous latent style embedding appended to observations.",
    )
    parser.add_argument(
        "--num-latent-codes",
        type=int,
        default=6,
        help="Number of discrete latent codes in the posterior codebook.",
    )
    parser.add_argument(
        "--posterior-hidden-size",
        type=int,
        default=128,
        help="Hidden size for the burn-in posterior GRU and action decoder.",
    )
    parser.add_argument(
        "--posterior-action-embed-dim",
        type=int,
        default=16,
        help="Embedding size used for discrete actions inside the burn-in posterior.",
    )
    parser.add_argument(
        "--posterior-epochs",
        type=int,
        default=20,
        help="Number of epochs for burn-in posterior pretraining.",
    )
    parser.add_argument(
        "--posterior-batch-size",
        type=int,
        default=64,
        help="Batch size for burn-in posterior pretraining.",
    )
    parser.add_argument(
        "--posterior-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for burn-in posterior pretraining.",
    )
    parser.add_argument(
        "--latent-balance-coef",
        type=float,
        default=0.05,
        help="Weight encouraging broad use of the latent codebook.",
    )
    parser.add_argument(
        "--latent-entropy-coef",
        type=float,
        default=0.001,
        help="Weight discouraging overly high-entropy posteriors per sample.",
    )
    parser.add_argument(
        "--adv-info-loss-coef",
        type=float,
        default=0.5,
        help="Weight of the latent code prediction term inside the discriminator loss.",
    )
    parser.add_argument(
        "--adv-recon-loss-coef",
        type=float,
        default=0.5,
        help="Weight of the action reconstruction term inside the discriminator loss.",
    )
    parser.add_argument(
        "--adv-prior-kl-coef",
        type=float,
        default=0.05,
        help="Weight of the latent prior matching KL term inside the discriminator loss.",
    )
    parser.add_argument(
        "--adv-latent-entropy-coef",
        type=float,
        default=0.001,
        help="Weight of the latent entropy bonus inside the discriminator loss.",
    )
    return parser.parse_args()


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(len(obs), -1)


def extract_trajectories_from_dataset(
    path: str,
    *,
    skip_inactive_agents: bool = True,
) -> tuple[list[BurnInTrajectory], dict[str, Any], dict[str, Any]]:
    dataset = load_expert_dataset(path)
    metadata = dataset["metadata"]
    if str(metadata.get("action_mode", "")).lower() != "discrete":
        raise ValueError("Burn-InfoGAIL demo expects a discrete-action expert dataset.")

    trajectories: list[BurnInTrajectory] = []
    dataset_mode = str(metadata.get("dataset_mode", "per_vehicle"))
    summary: dict[str, Any] = {"dataset_mode": dataset_mode}

    if dataset_mode == "per_vehicle":
        for episode_idx in range(len(dataset["episode_id"])):
            observations = _flatten_obs(np.asarray(dataset["observations"][episode_idx], dtype=np.float32))
            next_observations = _flatten_obs(
                np.asarray(dataset["next_observations"][episode_idx], dtype=np.float32)
            )
            actions = np.asarray(dataset["actions"][episode_idx], dtype=np.int64).reshape(-1)
            dones = np.asarray(dataset["dones"][episode_idx], dtype=bool).reshape(-1)
            timesteps = np.asarray(dataset["timesteps"][episode_idx], dtype=np.int32).reshape(-1)
            ego_id = int(dataset["ego_id"][episode_idx])
            episode_name = str(dataset["episode_name"][episode_idx])
            trajectories.append(
                BurnInTrajectory(
                    key=TrajectoryKey(episode_name=episode_name, agent_id=ego_id),
                    observations=observations,
                    actions=actions,
                    next_observations=next_observations,
                    dones=dones,
                    timesteps=timesteps,
                    scenario_id=str(dataset["scenario_id"][episode_idx]),
                    episode_id=int(dataset["episode_id"][episode_idx]),
                )
            )
    elif dataset_mode == "scene":
        alive_true = 0
        alive_total = 0
        for episode_idx in range(len(dataset["episode_id"])):
            observations = np.asarray(dataset["observations"][episode_idx], dtype=np.float32)
            next_observations = np.asarray(dataset["next_observations"][episode_idx], dtype=np.float32)
            actions = np.asarray(dataset["actions"][episode_idx], dtype=np.int64)
            dones = np.asarray(dataset["dones"][episode_idx], dtype=bool).reshape(-1)
            timesteps = np.asarray(dataset["timesteps"][episode_idx], dtype=np.int32).reshape(-1)
            agent_ids = np.asarray(dataset["agent_ids"][episode_idx], dtype=np.int32).reshape(-1)
            alive_mask = np.asarray(dataset["alive_mask"][episode_idx], dtype=bool)
            alive_true += int(alive_mask.sum())
            alive_total += int(alive_mask.size)

            T, N = observations.shape[:2]
            for agent_idx in range(N):
                agent_alive = alive_mask[:, agent_idx]
                if skip_inactive_agents:
                    keep_idx = np.flatnonzero(agent_alive)
                else:
                    keep_idx = np.arange(T, dtype=np.int64)
                if keep_idx.size == 0:
                    continue

                trajectories.append(
                    BurnInTrajectory(
                        key=TrajectoryKey(
                            episode_name=str(dataset["episode_name"][episode_idx]),
                            agent_id=int(agent_ids[agent_idx]),
                        ),
                        observations=_flatten_obs(observations[keep_idx, agent_idx]),
                        actions=np.asarray(actions[keep_idx, agent_idx], dtype=np.int64).reshape(-1),
                        next_observations=_flatten_obs(next_observations[keep_idx, agent_idx]),
                        dones=np.asarray(dones[keep_idx], dtype=bool).reshape(-1),
                        timesteps=np.asarray(timesteps[keep_idx], dtype=np.int32).reshape(-1),
                        scenario_id=str(dataset["scenario_id"][episode_idx]),
                        episode_id=int(dataset["episode_id"][episode_idx]),
                    )
                )

        summary["alive_true_count"] = alive_true
        summary["alive_total_count"] = alive_total
        summary["alive_ratio"] = float(alive_true) / float(alive_total) if alive_total else 0.0
        summary["skip_inactive_agents"] = bool(skip_inactive_agents)
    else:
        raise ValueError(f"Unsupported dataset_mode={dataset_mode!r}")

    if not trajectories:
        raise RuntimeError("No trajectories could be extracted from the expert dataset.")

    summary["trajectory_count"] = len(trajectories)
    summary["observation_dim"] = int(trajectories[0].observations.shape[-1])
    summary["transition_count"] = int(sum(len(traj.actions) for traj in trajectories))
    return trajectories, metadata, summary


def _collate_burn_in(batch: list[dict[str, torch.Tensor | TrajectoryKey]]) -> dict[str, Any]:
    burn_obs = torch.stack([item["burn_obs"] for item in batch], dim=0)
    burn_actions = torch.stack([item["burn_actions"] for item in batch], dim=0)
    future_obs = nn.utils.rnn.pad_sequence(
        [item["future_obs"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    future_actions = nn.utils.rnn.pad_sequence(
        [item["future_actions"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    future_mask = future_actions.ne(-100)
    keys = [item["key"] for item in batch]
    return {
        "keys": keys,
        "burn_obs": burn_obs,
        "burn_actions": burn_actions,
        "future_obs": future_obs,
        "future_actions": future_actions,
        "future_mask": future_mask,
    }


def train_burn_in_posterior(
    trajectories: list[BurnInTrajectory],
    *,
    obs_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[BurnInPosterior, ActionDecoder, dict[TrajectoryKey, np.ndarray], np.ndarray]:
    dataset = BurnInDataset(trajectories, burn_in_steps=args.burn_in_steps)
    loader = DataLoader(
        dataset,
        batch_size=args.posterior_batch_size,
        shuffle=True,
        collate_fn=_collate_burn_in,
    )

    posterior = BurnInPosterior(
        obs_dim=obs_dim,
        num_actions=NUM_DISCRETE_ACTIONS,
        num_codes=args.num_latent_codes,
        latent_dim=args.latent_dim,
        hidden_size=args.posterior_hidden_size,
        action_embed_dim=args.posterior_action_embed_dim,
    ).to(device)
    decoder = ActionDecoder(
        obs_dim=obs_dim,
        latent_dim=args.latent_dim,
        hidden_size=args.posterior_hidden_size,
        num_actions=NUM_DISCRETE_ACTIONS,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(posterior.parameters()) + list(decoder.parameters()),
        lr=args.posterior_learning_rate,
    )

    for epoch in range(args.posterior_epochs):
        posterior.train()
        decoder.train()
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in loader:
            burn_obs = batch["burn_obs"].to(device)
            burn_actions = batch["burn_actions"].to(device)
            future_obs = batch["future_obs"].to(device)
            future_actions = batch["future_actions"].to(device)
            future_mask = batch["future_mask"].to(device)

            logits, probs, latent = posterior(burn_obs, burn_actions)
            latent_seq = latent.unsqueeze(1).expand(-1, future_obs.shape[1], -1)
            action_logits = decoder(future_obs, latent_seq)
            ce = F.cross_entropy(
                action_logits.reshape(-1, NUM_DISCRETE_ACTIONS),
                future_actions.reshape(-1),
                ignore_index=-100,
            )

            usage = probs.mean(dim=0)
            balance_loss = torch.sum(usage * torch.log(usage + 1e-8))
            posterior_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            valid_ratio = future_mask.float().mean().item()

            loss = (
                ce
                + float(args.latent_balance_coef) * balance_loss
                - float(args.latent_entropy_coef) * posterior_entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_steps += 1

        if epoch_steps:
            print(
                f"[burn-in] epoch={epoch + 1}/{args.posterior_epochs} "
                f"loss={epoch_loss / epoch_steps:.4f} valid_ratio={valid_ratio:.3f}"
            )

    posterior.eval()
    decoder.eval()

    latent_lookup: dict[TrajectoryKey, np.ndarray] = {}
    with torch.no_grad():
        for traj in trajectories:
            prefix_len = min(int(args.burn_in_steps), len(traj.actions))
            burn_obs = torch.as_tensor(traj.observations[:prefix_len], dtype=torch.float32, device=device).unsqueeze(0)
            burn_actions = torch.as_tensor(traj.actions[:prefix_len], dtype=torch.long, device=device).unsqueeze(0)
            _, _, latent = posterior(burn_obs, burn_actions)
            latent_lookup[traj.key] = latent.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    fallback_latent = posterior.codebook.weight.detach().mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
    return posterior, decoder, latent_lookup, fallback_latent


def build_augmented_transitions(
    trajectories: list[BurnInTrajectory],
    *,
    latent_lookup: dict[TrajectoryKey, np.ndarray],
) -> Transitions:
    obs_parts: list[np.ndarray] = []
    act_parts: list[np.ndarray] = []
    next_obs_parts: list[np.ndarray] = []
    done_parts: list[np.ndarray] = []
    info_parts: list[dict[str, Any]] = []

    for traj in trajectories:
        latent = latent_lookup[traj.key].reshape(1, -1)
        obs_aug = np.concatenate(
            [traj.observations, np.repeat(latent, len(traj.observations), axis=0)],
            axis=-1,
        ).astype(np.float32, copy=False)
        next_obs_aug = np.concatenate(
            [traj.next_observations, np.repeat(latent, len(traj.next_observations), axis=0)],
            axis=-1,
        ).astype(np.float32, copy=False)

        obs_parts.append(obs_aug)
        act_parts.append(np.asarray(traj.actions, dtype=np.int64))
        next_obs_parts.append(next_obs_aug)
        done_parts.append(np.asarray(traj.dones, dtype=bool))

        for step_idx in range(len(traj.actions)):
            info_parts.append(
                {
                    "episode_id": int(traj.episode_id),
                    "timestep": int(traj.timesteps[step_idx]),
                    "episode_name": traj.key.episode_name,
                    "scenario_id": traj.scenario_id,
                    "agent_id": int(traj.key.agent_id),
                }
            )

    obs = np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)
    acts = np.concatenate(act_parts, axis=0).astype(np.int64, copy=False)
    next_obs = np.concatenate(next_obs_parts, axis=0).astype(np.float32, copy=False)
    dones = np.concatenate(done_parts, axis=0).astype(bool, copy=False)
    infos = np.asarray(info_parts, dtype=object)
    return Transitions(obs=obs, acts=acts, infos=infos, next_obs=next_obs, dones=dones)


def make_env_factory(
    *,
    scene: str,
    episode_root: str,
    prebuilt_split: str,
    max_surrounding: str | int,
    cells: int,
    maximum_range: float,
    simulation_frequency: int,
    policy_frequency: int,
    max_episode_steps: int,
    seed: int,
    latent_lookup: dict[TrajectoryKey, np.ndarray],
    latent_dim: int,
    fallback_latent: np.ndarray,
):
    def _factory(rank: int):
        def _init():
            config = {
                "scene": str(scene),
                "episode_root": str(episode_root),
                "prebuilt_split": str(prebuilt_split),
                "observation": {
                    "type": "LidarCameraObservations",
                    "lidar": {
                        "cells": int(cells),
                        "maximum_range": float(maximum_range),
                        "normalize": True,
                    },
                    "camera": {
                        "cells": 21,
                        "maximum_range": float(maximum_range),
                        "field_of_view": np.pi / 2,
                        "normalize": True,
                    },
                },
                "action": {"type": "DiscreteSteerMetaAction"},
                "action_mode": "discrete",
                "simulation_frequency": int(simulation_frequency),
                "policy_frequency": int(policy_frequency),
                "max_episode_steps": int(max_episode_steps),
                "expert_test_mode": False,
                "truncate_to_trajectory_length": True,
                "max_surrounding": max_surrounding,
                "offscreen_rendering": True,
                "seed": int(seed + rank),
            }
            env = gym.make("NGSim-US101-v0", config=config)
            env = FlattenObservation(env)
            env = LatentObservationWrapper(
                env,
                latent_lookup=latent_lookup,
                latent_dim=latent_dim,
                fallback_latent=fallback_latent,
            )
            env.reset(seed=seed + rank)
            return env

        return _init

    return _factory


def build_vec_env(args: argparse.Namespace, training: bool, latent_lookup, fallback_latent):
    factory = make_env_factory(
        scene=args.scene,
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        max_surrounding=args.max_surrounding,
        cells=args.cells,
        maximum_range=args.maximum_range,
        simulation_frequency=args.simulation_frequency,
        policy_frequency=args.policy_frequency,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed + (0 if training else 10_000),
        latent_lookup=latent_lookup,
        latent_dim=args.latent_dim,
        fallback_latent=fallback_latent,
    )
    if args.dummy_vec or args.n_envs <= 1:
        env = DummyVecEnv([factory(0)])
    else:
        env = SubprocVecEnv([factory(rank) for rank in range(args.n_envs)])
    return VecMonitor(env)


def evaluate_generator(learner: PPO, eval_env, n_eval_episodes: int) -> tuple[float, float]:
    mean_reward, std_reward = evaluate_policy(
        model=learner,
        env=eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        return_episode_rewards=False,
    )
    return float(mean_reward), float(std_reward)


def main() -> None:
    args = parse_args()
    register_ngsim_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert_data_path = os.path.abspath(args.expert_data)
    run_dir = os.path.abspath(os.path.join("logs", "burn_info_gail_discrete", args.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    trajectories, metadata, data_summary = extract_trajectories_from_dataset(
        expert_data_path,
        skip_inactive_agents=args.skip_inactive_agents,
    )
    obs_dim = int(trajectories[0].observations.shape[-1])

    print(f"Loaded expert dataset: {expert_data_path}")
    print(
        f"trajectory_count={data_summary['trajectory_count']} "
        f"transition_count={data_summary['transition_count']} "
        f"dataset_mode={data_summary['dataset_mode']} "
        f"observation_dim={obs_dim}"
    )
    if "alive_ratio" in data_summary:
        print(
            f"alive_mask true/total={data_summary['alive_true_count']}/"
            f"{data_summary['alive_total_count']} "
            f"(ratio={data_summary['alive_ratio']:.3f}) "
            f"skip_inactive_agents={data_summary['skip_inactive_agents']}"
        )

    print(
        f"Pretraining burn-in posterior with burn_in_steps={args.burn_in_steps} "
        f"latent_dim={args.latent_dim} num_codes={args.num_latent_codes}"
    )
    posterior, decoder, latent_lookup, fallback_latent = train_burn_in_posterior(
        trajectories,
        obs_dim=obs_dim,
        args=args,
        device=device,
    )

    demos = build_augmented_transitions(trajectories, latent_lookup=latent_lookup)
    effective_demo_batch_size = min(int(args.demo_batch_size), int(len(demos.obs)))
    if effective_demo_batch_size < int(args.demo_batch_size):
        print(
            f"Reducing demo_batch_size from {args.demo_batch_size} to "
            f"{effective_demo_batch_size} because the dataset is small."
        )

    print(
        f"Built burn-in conditioned demos: transitions={len(demos.obs)} "
        f"augmented_observation_dim={demos.obs.shape[-1]}"
    )

    train_env = build_vec_env(args, training=True, latent_lookup=latent_lookup, fallback_latent=fallback_latent)
    eval_env = build_vec_env(args, training=False, latent_lookup=latent_lookup, fallback_latent=fallback_latent)

    learner = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=run_dir,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        policy_kwargs={"net_arch": [256, 256]},
    )

    reward_net = BurnInfoRewardNet(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        obs_dim=obs_dim,
        latent_dim=args.latent_dim,
        hidden_size=args.posterior_hidden_size,
        num_codes=args.num_latent_codes,
        num_actions=NUM_DISCRETE_ACTIONS,
        codebook_weight=posterior.codebook.weight.detach().cpu().numpy(),
        decoder_state_dict=decoder.state_dict(),
        normalize_input_layer=RunningNorm,
    )

    gail_logger = imitation_logger.configure(
        folder=os.path.join(run_dir, "gail_metrics"),
        format_strs=["stdout", "tensorboard"],
    )

    trainer = BurnInfoGAIL(
        demonstrations=demos,
        demo_batch_size=effective_demo_batch_size,
        gen_replay_buffer_capacity=args.gen_replay_buffer_capacity,
        n_disc_updates_per_round=args.disc_updates_per_round,
        venv=train_env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        custom_logger=gail_logger,
        info_loss_coef=args.adv_info_loss_coef,
        recon_loss_coef=args.adv_recon_loss_coef,
        prior_kl_coef=args.adv_prior_kl_coef,
        latent_entropy_coef=args.adv_latent_entropy_coef,
    )

    torch.save(
        {
            "posterior_state_dict": posterior.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "reward_net_state_dict": reward_net.state_dict(),
            "latent_lookup": {
                (key.episode_name, key.agent_id): value
                for key, value in latent_lookup.items()
            },
            "fallback_latent": fallback_latent,
            "metadata": metadata,
            "args": vars(args),
        },
        os.path.join(run_dir, "burn_in_latent_model.pt"),
    )

    print(
        "Training Burn-InfoGAIL demo "
        f"for {args.total_timesteps} timesteps "
        f"(info={args.adv_info_loss_coef}, recon={args.adv_recon_loss_coef}, "
        f"prior_kl={args.adv_prior_kl_coef}, entropy={args.adv_latent_entropy_coef})"
    )
    print(f"TensorBoard: tensorboard --logdir {run_dir}")

    best_eval_reward = -np.inf

    def _on_round(round_idx: int) -> None:
        sampled_steps = int((round_idx + 1) * trainer.gen_train_timesteps)
        if sampled_steps % max(1, args.checkpoint_every) == 0:
            learner.save(os.path.join(ckpt_dir, f"ppo_generator_step_{sampled_steps}"))

        if sampled_steps % max(1, args.eval_every) == 0:
            mean_reward, std_reward = evaluate_generator(
                learner=learner,
                eval_env=eval_env,
                n_eval_episodes=args.eval_episodes,
            )
            nonlocal_best = _on_round.best_eval_reward
            print(
                f"[eval] sampled_steps={sampled_steps} "
                f"mean_reward={mean_reward:.3f} std_reward={std_reward:.3f}"
            )
            if mean_reward > nonlocal_best:
                _on_round.best_eval_reward = mean_reward
                learner.save(os.path.join(eval_dir, "best_model"))

    _on_round.best_eval_reward = best_eval_reward

    try:
        trainer.train(total_timesteps=args.total_timesteps, callback=_on_round)
    finally:
        learner.save(os.path.join(run_dir, "ppo_generator_final"))
        train_env.close()
        eval_env.close()

    print(f"Saved final generator policy under: {run_dir}")


if __name__ == "__main__":
    main()
