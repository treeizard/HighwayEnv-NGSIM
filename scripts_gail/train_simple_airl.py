#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import fields
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.envs import make_training_env
from scripts_gail.ps_gail.monitoring import WandbMonitor
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.observations import flatten_agent_observations, policy_observations_from_flat
from scripts_gail.ps_gail.trainer import (
    RolloutBatch,
    collect_rollouts,
    compute_returns_and_advantages,
    infer_continuous_action_dim,
    infer_policy_obs_dim,
    make_rollout_executor,
    policy_distribution_and_values,
    resolve_device,
    update_policy,
)


class AIRLReward(nn.Module):
    """Potential-based AIRL reward model for continuous NGSIM actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.reward = nn.Sequential(
            nn.Linear(int(obs_dim) + int(action_dim), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), 1),
        )
        self.potential = nn.Sequential(
            nn.Linear(int(obs_dim), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), 1),
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


def _as_tensor(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _clamp_fraction(value: float) -> float:
    return float(min(1.0, max(1e-6, float(value))))


def config_for_round(cfg: PSGAILConfig, round_idx: int) -> PSGAILConfig:
    if not bool(cfg.controlled_vehicle_curriculum):
        return cfg
    rounds = max(1, int(cfg.controlled_vehicle_curriculum_rounds))
    progress = 1.0 if rounds == 1 else min(1.0, max(0.0, (int(round_idx) - 1) / float(rounds - 1)))
    initial = _clamp_fraction(cfg.initial_controlled_vehicle_fraction)
    final = _clamp_fraction(cfg.final_controlled_vehicle_fraction)
    return replace(
        cfg,
        control_all_vehicles=False,
        percentage_controlled_vehicles=_clamp_fraction(initial + (final - initial) * progress),
    )


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


def _policy_log_probs(
    policy: nn.Module,
    cfg: PSGAILConfig,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    dist, _values = policy_distribution_and_values(policy, obs, cfg, None)
    return dist.log_prob(actions)


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
) -> dict[str, float]:
    n = int(len(generator_obs))
    expert_idx = np.random.choice(len(expert_obs), size=n, replace=len(expert_obs) < n)
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
    losses: list[torch.Tensor] = []
    expert_accs: list[torch.Tensor] = []
    gen_accs: list[torch.Tensor] = []
    expert_rewards: list[torch.Tensor] = []
    gen_rewards: list[torch.Tensor] = []
    batch_size = max(1, int(reward_batch_size))

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
            with torch.no_grad():
                expert_log_pi = _policy_log_probs(policy, cfg, eo, ea)
                gen_log_pi = _policy_log_probs(policy, cfg, go, ga)
            expert_shaped = reward_model.shaped_logits(eo, ea, eno, ed, gamma=float(cfg.gamma))
            gen_shaped = reward_model.shaped_logits(go, ga, gno, gd, gamma=float(cfg.gamma))
            expert_reward = reward_model(eo, ea)
            gen_reward = reward_model(go, ga)
            expert_logits = expert_shaped - expert_log_pi
            gen_logits = gen_shaped - gen_log_pi
            logits = torch.cat([expert_logits, gen_logits], dim=0)
            labels = torch.cat([torch.ones_like(expert_logits), torch.zeros_like(gen_logits)], dim=0)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(reward_model.parameters(), float(cfg.max_grad_norm))
            optimizer.step()
            with torch.no_grad():
                expert_accs.append((expert_logits > 0.0).float().mean())
                gen_accs.append((gen_logits < 0.0).float().mean())
                expert_rewards.append(expert_reward.mean())
                gen_rewards.append(gen_reward.mean())
            losses.append(loss.detach())

    def mean(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().detach().cpu().item()) if values else float("nan")

    return {
        "reward_loss": mean(losses),
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
    shaped = shaped_logits.astype(np.float32, copy=True)
    if bool(cfg.normalize_gail_reward) and shaped.size > 1:
        shaped = (shaped - shaped.mean()) / (shaped.std() + 1e-8)
    if float(cfg.gail_reward_clip) > 0.0:
        shaped = np.clip(shaped, -float(cfg.gail_reward_clip), float(cfg.gail_reward_clip))
    rewards = shaped + rollout.env_penalties
    if float(cfg.final_reward_clip) > 0.0:
        rewards = np.clip(rewards, -float(cfg.final_reward_clip), float(cfg.final_reward_clip))
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
        gail_rewards_raw=raw,
        gail_rewards_normalized=shaped.astype(np.float32),
        returns=returns,
        advantages=advantages,
        mean_raw_gail_reward=float(raw.mean()) if raw.size else 0.0,
        mean_normalized_gail_reward=float(shaped.mean()) if shaped.size else 0.0,
    )


def _policy_action_tuple(policy: nn.Module, obs, *, device: torch.device, cfg: PSGAILConfig) -> tuple[object, ...]:
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
        mean, _values = policy(obs_tensor)
        if getattr(policy, "log_std", None) is None:
            raise RuntimeError("AIRL checkpoint video expects a continuous policy.")
        actions = torch.clamp(mean, -1.0, 1.0).detach().cpu().numpy().astype(np.float32)
    return tuple(action.copy() for action in actions.reshape(-1, int(cfg.continuous_action_dim)))


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
        for _ in range(max(1, int(cfg.checkpoint_video_steps))):
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


def parse_args() -> tuple[PSGAILConfig, int]:
    defaults = PSGAILConfig(action_mode="continuous", run_name="simple_airl")
    parser = argparse.ArgumentParser(description="Lightweight continuous AIRL test trainer for unified NGSIM expert data.")
    for field in fields(PSGAILConfig):
        value = getattr(defaults, field.name)
        arg = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    parser.add_argument("--reward-batch-size", type=int, default=1024)
    args = parser.parse_args()
    values = vars(args)
    reward_batch_size = int(values.pop("reward_batch_size"))
    return PSGAILConfig(**values), reward_batch_size


def main() -> None:
    cfg, reward_batch_size = parse_args()
    if str(cfg.action_mode).lower() != "continuous":
        raise ValueError("This AIRL test trainer currently supports --action-mode continuous only.")
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
        ).to(device)
        reward_model = AIRLReward(policy_obs_dim, int(cfg.continuous_action_dim), cfg.hidden_size).to(device)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=cfg.disc_learning_rate)
        monitor.watch(policy, reward_model)
        rollout_executor = make_rollout_executor(cfg)
        current_env_signature = env_signature(env_cfg)

        print(f"Loaded expert folder: {os.path.abspath(cfg.expert_data)}")
        print(f"expert_obs={expert.policy_observations.shape} expert_actions={expert.actions_continuous_env.shape}")
        print(f"policy_obs_dim={policy_obs_dim} action_dim={cfg.continuous_action_dim} device={device}")

        for round_idx in range(1, int(cfg.total_rounds) + 1):
            round_cfg = config_for_round(cfg, round_idx)
            round_cfg.continuous_action_dim = cfg.continuous_action_dim
            if env_signature(round_cfg) != current_env_signature:
                env.close()
                env = make_training_env(round_cfg)
                current_env_signature = env_signature(round_cfg)
            rollout = collect_rollouts(
                env,
                policy,
                round_cfg,
                device,
                policy_obs_dim,
                seed_offset=(round_idx - 1) * max(1, int(cfg.num_rollout_workers)),
                executor=rollout_executor,
            )
            reward_stats = update_reward_model(
                reward_model,
                reward_optimizer,
                policy,
                expert.policy_observations,
                expert.actions_continuous_env,
                expert.next_policy_observations,
                expert.dones,
                rollout.policy_observations,
                rollout.actions.astype(np.float32, copy=False),
                rollout.next_policy_observations,
                rollout.dones,
                round_cfg,
                device,
                reward_batch_size=reward_batch_size,
            )
            rollout = refresh_airl_rewards(rollout, reward_model, round_cfg, device)
            policy_stats = update_policy(policy, policy_optimizer, rollout, round_cfg, device)
            print(
                f"[round {round_idx:04d}] env_steps={rollout.num_env_steps} "
                f"agent_steps={rollout.num_agent_steps} episodes={rollout.num_episodes} "
                f"reward_loss={reward_stats['reward_loss']:.4f} "
                f"expert_acc={reward_stats['expert_acc']:.3f} gen_acc={reward_stats['gen_acc']:.3f} "
                f"policy_loss={policy_stats['policy_loss']:.4f} value_loss={policy_stats['value_loss']:.4f} "
                f"reward={float(rollout.rewards.mean()):.4f}"
            )
            metrics = {
                "round": round_idx,
                "rollout/env_steps": rollout.num_env_steps,
                "rollout/agent_steps": rollout.num_agent_steps,
                "rollout/episodes": rollout.num_episodes,
                "rollout/mean_reward": float(rollout.rewards.mean()),
                "airl/reward_loss": reward_stats["reward_loss"],
                "airl/expert_acc": reward_stats["expert_acc"],
                "airl/gen_acc": reward_stats["gen_acc"],
                "airl/expert_reward": reward_stats["expert_reward"],
                "airl/gen_reward": reward_stats["gen_reward"],
                "policy/loss": policy_stats["policy_loss"],
                "policy/value_loss": policy_stats["value_loss"],
                "policy/entropy": policy_stats["entropy"],
                "policy/approx_kl": policy_stats["approx_kl"],
            }
            monitor.log(metrics, step=round_idx)
            if cfg.checkpoint_every > 0 and round_idx % int(cfg.checkpoint_every) == 0:
                checkpoint_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}.pt")
                torch.save(
                    {
                        "round": round_idx,
                        "policy_state_dict": policy.state_dict(),
                        "reward_state_dict": reward_model.state_dict(),
                        "expert_metadata": expert.metadata,
                        "config": vars(cfg),
                    },
                    checkpoint_path,
                )
                monitor.save(checkpoint_path)
                video_path = save_checkpoint_video(policy, round_cfg, run_dir=run_dir, round_idx=round_idx, device=device)
                if video_path is not None:
                    monitor.log_video("checkpoint/policy_video", video_path, step=round_idx, fps=int(round_cfg.policy_frequency))

        final_path = os.path.join(run_dir, "final.pt")
        torch.save(
            {
                "round": int(cfg.total_rounds),
                "policy_state_dict": policy.state_dict(),
                "reward_state_dict": reward_model.state_dict(),
                "expert_metadata": expert.metadata,
                "config": vars(cfg),
            },
            final_path,
        )
        monitor.save(final_path)
        final_video_path = save_checkpoint_video(policy, config_for_round(cfg, int(cfg.total_rounds)), run_dir=run_dir, round_idx=int(cfg.total_rounds), device=device)
        if final_video_path is not None:
            monitor.log_video("checkpoint/final_policy_video", final_video_path, step=int(cfg.total_rounds), fps=int(cfg.policy_frequency))
    finally:
        if rollout_executor is not None:
            rollout_executor.shutdown(wait=True, cancel_futures=True)
        if env is not None:
            env.close()
        monitor.finish()
    print(f"Saved final checkpoint under: {run_dir}")


if __name__ == "__main__":
    main()
