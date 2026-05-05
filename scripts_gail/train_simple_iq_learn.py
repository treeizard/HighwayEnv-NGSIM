#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields

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
from scripts_gail.ps_gail.trainer import infer_continuous_action_dim, infer_policy_obs_dim, resolve_device
from scripts_gail.train_simple_airl import config_for_round, save_checkpoint_video


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


def _as_tensor(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _sample_policy(policy: nn.Module, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, _values = policy(obs)
    log_std = getattr(policy, "log_std", None)
    if log_std is None:
        raise RuntimeError("IQ-Learn test trainer expects a continuous policy.")
    std = torch.exp(log_std).expand_as(mean)
    dist = Independent(Normal(mean, std), 1)
    action = torch.clamp(dist.rsample(), -1.0, 1.0)
    return action, dist.log_prob(action), mean


def update_iq_learn(
    policy: nn.Module,
    q_net: SoftQNetwork,
    target_q_net: SoftQNetwork,
    policy_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    batch_size: int,
    gamma: float,
    alpha: float,
    tau: float,
) -> dict[str, float]:
    n = int(len(obs))
    idx = np.random.randint(0, n, size=max(1, int(batch_size)))
    obs_t = _as_tensor(obs[idx], device=device)
    actions_t = _as_tensor(actions[idx], device=device)
    next_obs_t = _as_tensor(next_obs[idx], device=device)
    dones_t = _as_tensor(dones[idx].astype(np.float32), device=device)

    with torch.no_grad():
        next_action, next_log_prob, _next_mean = _sample_policy(policy, next_obs_t)
        target_v = target_q_net(next_obs_t, next_action) - float(alpha) * next_log_prob
        target = (1.0 - dones_t) * float(gamma) * target_v

    expert_q = q_net(obs_t, actions_t)
    # A compact IQ-Learn-style objective: expert transitions should satisfy a
    # positive Bellman gap, while the soft value target regularizes Q scale.
    bellman_gap = expert_q - target
    q_loss = F.softplus(-bellman_gap).mean() + 0.5 * torch.square(expert_q).mean() * 1e-4
    q_optimizer.zero_grad()
    q_loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), float(cfg.max_grad_norm))
    q_optimizer.step()

    policy_action, log_prob, mean_action = _sample_policy(policy, obs_t)
    policy_q = q_net(obs_t, policy_action)
    bc_loss = F.mse_loss(mean_action, actions_t)
    policy_loss = (float(alpha) * log_prob - policy_q).mean() + 0.05 * bc_loss
    policy_optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm))
    policy_optimizer.step()

    with torch.no_grad():
        for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
            target_param.data.mul_(1.0 - float(tau)).add_(param.data, alpha=float(tau))

    return {
        "q_loss": float(q_loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "bc_loss": float(bc_loss.detach().cpu().item()),
        "expert_q": float(expert_q.detach().mean().cpu().item()),
        "target_v": float(target.detach().mean().cpu().item()),
        "policy_q": float(policy_q.detach().mean().cpu().item()),
        "entropy": float((-log_prob).detach().mean().cpu().item()),
    }


def evaluate_policy(policy: nn.Module, cfg: PSGAILConfig, device: torch.device, *, episodes: int) -> dict[str, float]:
    env = make_training_env(cfg)
    lengths: list[int] = []
    rewards: list[float] = []
    crashes = 0
    try:
        for ep_idx in range(max(1, int(episodes))):
            obs, _info = env.reset(seed=int(cfg.seed) + 100_000 + ep_idx)
            total_reward = 0.0
            length = 0
            for _ in range(max(1, int(cfg.max_episode_steps))):
                obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
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
    parser.add_argument("--iq-alpha", type=float, default=0.1)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--eval-episodes", type=int, default=2)
    args = parser.parse_args()
    values = vars(args).copy()
    extra_keys = {"total_updates", "eval_every", "replay_size", "iq_alpha", "target_tau", "eval_episodes"}
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
    run_dir = os.path.abspath(os.path.join("logs", "iq_learn", cfg.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    monitor = WandbMonitor(cfg, run_dir)
    monitor.start()

    env = None
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
        q_net = SoftQNetwork(policy_obs_dim, int(cfg.continuous_action_dim), cfg.hidden_size).to(device)
        target_q_net = SoftQNetwork(policy_obs_dim, int(cfg.continuous_action_dim), cfg.hidden_size).to(device)
        target_q_net.load_state_dict(q_net.state_dict())
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        q_optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.disc_learning_rate)
        monitor.watch(policy, q_net)

        print(f"Loaded expert folder: {os.path.abspath(cfg.expert_data)}")
        print(f"expert_obs={expert.policy_observations.shape} expert_actions={expert.actions_continuous_env.shape}")
        print(f"policy_obs_dim={policy_obs_dim} action_dim={cfg.continuous_action_dim} device={device}")

        latest_stats: dict[str, float] = {}
        for update_idx in range(1, int(extras.total_updates) + 1):
            latest_stats = update_iq_learn(
                policy,
                q_net,
                target_q_net,
                policy_optimizer,
                q_optimizer,
                expert.policy_observations,
                expert.actions_continuous_env,
                expert.next_policy_observations,
                expert.dones,
                cfg,
                device,
                batch_size=int(cfg.batch_size),
                gamma=float(cfg.gamma),
                alpha=float(extras.iq_alpha),
                tau=float(extras.target_tau),
            )
            if update_idx == 1 or update_idx % max(1, int(extras.eval_every)) == 0:
                metrics = {"update": update_idx, **{f"iq/{k}": v for k, v in latest_stats.items()}}
                eval_stats = evaluate_policy(policy, cfg, device, episodes=int(extras.eval_episodes))
                metrics.update(eval_stats)
                monitor.log(metrics, step=update_idx)
                print(
                    f"[update {update_idx:06d}] q_loss={latest_stats['q_loss']:.4f} "
                    f"policy_loss={latest_stats['policy_loss']:.4f} "
                    f"bc_loss={latest_stats['bc_loss']:.4f} "
                    f"eval_reward={eval_stats['eval/mean_reward']:.4f}"
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
                video_path = save_checkpoint_video(policy, cfg, run_dir=run_dir, round_idx=update_idx, device=device)
                if video_path is not None:
                    monitor.log_video("checkpoint/policy_video", video_path, step=update_idx, fps=int(cfg.policy_frequency))

        final_path = os.path.join(run_dir, "final.pt")
        torch.save(
            {
                "update": int(extras.total_updates),
                "policy_state_dict": policy.state_dict(),
                "q_state_dict": q_net.state_dict(),
                "target_q_state_dict": target_q_net.state_dict(),
                "expert_metadata": expert.metadata,
                "config": vars(cfg),
                "iq_config": vars(extras),
                "latest_stats": latest_stats,
            },
            final_path,
        )
        monitor.save(final_path)
        final_video_path = save_checkpoint_video(policy, cfg, run_dir=run_dir, round_idx=int(extras.total_updates), device=device)
        if final_video_path is not None:
            monitor.log_video("checkpoint/final_policy_video", final_video_path, step=int(extras.total_updates), fps=int(cfg.policy_frequency))
    finally:
        if env is not None:
            env.close()
        monitor.finish()
    print(f"Saved final checkpoint under: {run_dir}")


if __name__ == "__main__":
    main()
