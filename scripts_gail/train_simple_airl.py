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
    compute_returns_and_advantages,
    infer_continuous_action_dim,
    infer_critic_obs_dim,
    infer_policy_obs_dim,
    make_rollout_executor,
    policy_distribution_and_values,
    resolve_device,
    set_optimizer_lr,
    subsample_rollout_for_training,
    update_policy,
)
from scripts_gail.train_simple_ps_gail import behavior_clone_pretrain, evaluate_policy_survival


class AIRLReward(nn.Module):
    """Potential-based AIRL reward model for continuous NGSIM actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int | None = None,
        *,
        hidden_sizes: str | int | tuple[int, ...] | list[int] | None = None,
        dropout: float = 0.2,
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


def _policy_log_probs(
    policy: nn.Module,
    cfg: PSGAILConfig,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    dist, _values = policy_distribution_and_values(policy, obs, cfg, None)
    return dist.log_prob(actions)


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
            if loss_type == "wgan_gp":
                wgan_loss = gen_shaped.mean() - expert_shaped.mean()
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
                with torch.no_grad():
                    expert_log_pi = _policy_log_probs(policy, cfg, eo, ea)
                    gen_log_pi = _policy_log_probs(policy, cfg, go, ga)
                expert_logits = expert_shaped - expert_log_pi
                gen_logits = gen_shaped - gen_log_pi
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
                    centered_threshold = 0.5 * (expert_shaped.mean() + gen_shaped.mean())
                    expert_accs.append((expert_shaped > centered_threshold).float().mean())
                    gen_accs.append((gen_shaped < centered_threshold).float().mean())
                    wgan_losses.append(wgan_loss.detach())
                    gradient_penalties.append(gradient_penalty.detach())
                    critic_gaps.append((expert_shaped.mean() - gen_shaped.mean()).detach())
                else:
                    expert_accs.append((expert_logits > 0.0).float().mean())
                    gen_accs.append((gen_logits < 0.0).float().mean())
                    bce_losses.append(bce_loss.detach())
                    critic_gaps.append((expert_logits.mean() - gen_logits.mean()).detach())
                expert_rewards.append(expert_reward.mean())
                gen_rewards.append(gen_reward.mean())
            losses.append(loss.detach())

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
    if str(getattr(cfg, "discriminator_loss", "airl_bce")).lower() == "wgan_gp":
        shaped = shaped_logits.astype(np.float32, copy=True)
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
        shaped = (shaped_logits - rollout.old_log_probs).astype(np.float32, copy=True)
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
    defaults = PSGAILConfig(
        action_mode="continuous",
        run_name="simple_airl",
        policy_model="transformer",
        batch_size=4096,
        disc_batch_size=4096,
        discriminator_loss="wgan_gp",
        disc_learning_rate=5e-5,
        disc_updates_per_round=2,
        disc_expert_label=0.8,
        disc_generator_label=0.2,
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
    args = parser.parse_args()
    values = vars(args)
    reward_batch_size = int(values.pop("reward_batch_size"))
    return PSGAILConfig(**values), reward_batch_size


def main() -> None:
    cfg, reward_batch_size = parse_args()
    if str(cfg.action_mode).lower() != "continuous":
        raise ValueError("This AIRL test trainer currently supports --action-mode continuous only.")
    airl_objective = str(cfg.discriminator_loss).lower()
    if airl_objective not in {"airl", "airl_bce", "bce", "wgan_gp"}:
        raise ValueError(
            "train_simple_airl.py supports canonical BCE AIRL and WGAN-GP AIRL-style training. "
            f"Received discriminator_loss={cfg.discriminator_loss!r}."
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
            centralized_critic=bool(cfg.centralized_critic),
            critic_obs_dim=critic_obs_dim,
        ).to(device)
        reward_model = AIRLReward(
            policy_obs_dim,
            int(cfg.continuous_action_dim),
            hidden_sizes=cfg.discriminator_hidden_sizes,
            dropout=float(cfg.discriminator_dropout),
            spectral_norm=bool(cfg.discriminator_spectral_norm),
        ).to(device)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=cfg.disc_learning_rate)
        monitor.watch(policy, reward_model)
        rollout_executor = make_rollout_executor(cfg)
        current_env_signature = env_signature(env_cfg)

        print(f"Loaded expert folder: {os.path.abspath(cfg.expert_data)}")
        print(f"expert_obs={expert.policy_observations.shape} expert_actions={expert.actions_continuous_env.shape}")
        print(
            f"policy_obs_dim={policy_obs_dim} critic_obs_dim={critic_obs_dim} "
            f"centralized_critic={cfg.centralized_critic} "
            f"central_critic_max_vehicles={cfg.central_critic_max_vehicles} "
            f"central_critic_include_local_obs={cfg.central_critic_include_local_obs} "
            f"action_dim={cfg.continuous_action_dim} device={device} "
            f"policy_model={cfg.policy_model} transformer_layers={cfg.transformer_layers} "
            f"transformer_heads={cfg.transformer_heads} transformer_dropout={cfg.transformer_dropout} "
            f"airl_objective={cfg.discriminator_loss} wgan_gp_lambda={cfg.wgan_gp_lambda} "
            f"reward_hidden={cfg.discriminator_hidden_sizes} "
            f"reward_dropout={cfg.discriminator_dropout} "
            f"reward_spectral_norm={cfg.discriminator_spectral_norm} "
            f"wgan_reward_center={cfg.wgan_reward_center} wgan_reward_clip={cfg.wgan_reward_clip} "
            f"wgan_reward_scale={cfg.wgan_reward_scale}"
        )
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
        for round_idx in range(1, int(cfg.total_rounds) + 1):
            round_cfg = config_for_round(cfg, round_idx)
            round_cfg.continuous_action_dim = cfg.continuous_action_dim
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
            )
            rollout = subsample_rollout_for_training(
                collected_rollout,
                round_cfg,
                seed=int(round_cfg.seed) + int(round_idx) * 104729,
            )
            rollout_was_subsampled = int(rollout.num_agent_steps) != int(collected_rollout.num_agent_steps)
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
            policy_stats = update_policy(
                policy,
                policy_optimizer,
                rollout,
                round_cfg,
                device,
                expert_policy_observations=expert.policy_observations,
                expert_actions=expert.actions_continuous_env,
            )
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
                f"reward_loss={reward_stats['reward_loss']:.4f} "
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
                "train/discriminator_loss_type_wgan_gp": int(
                    str(round_cfg.discriminator_loss).lower() == "wgan_gp"
                ),
                "train/wgan_gp_lambda": float(round_cfg.wgan_gp_lambda),
                "train/reward_spectral_norm": int(bool(round_cfg.discriminator_spectral_norm)),
                "train/discriminator_spectral_norm": int(bool(round_cfg.discriminator_spectral_norm)),
                "train/policy_obs_dim": policy_obs_dim,
                "train/critic_obs_dim": critic_obs_dim,
                "train/centralized_critic": int(bool(round_cfg.centralized_critic)),
                "train/wgan_reward_center": int(bool(round_cfg.wgan_reward_center)),
                "train/wgan_reward_clip": float(round_cfg.wgan_reward_clip),
                "train/wgan_reward_scale": float(round_cfg.wgan_reward_scale),
                "policy/loss": policy_stats["policy_loss"],
                "policy/value_loss": policy_stats["value_loss"],
                "policy/bc_regularization_loss": policy_stats["bc_regularization_loss"],
                "policy/bc_regularization_coef": policy_stats["bc_regularization_coef"],
                "policy/entropy": policy_stats["entropy"],
                "policy/approx_kl": policy_stats["approx_kl"],
                "policy/clip_fraction": policy_stats["clip_fraction"],
                "policy/ratio_mean": policy_stats["ratio_mean"],
                "policy/ratio_std": policy_stats["ratio_std"],
                "policy/ppo_micro_batch_size": policy_stats["ppo_micro_batch_size"],
                "train/policy_learning_rate": float(round_cfg.learning_rate),
                "train/reward_learning_rate": float(round_cfg.disc_learning_rate),
                "train/disc_learning_rate": float(round_cfg.disc_learning_rate),
                "train/entropy_coef": float(round_cfg.entropy_coef),
                "train/clip_range": float(round_cfg.clip_range),
                "train/reward_updates_per_round": int(round_cfg.disc_updates_per_round),
                "train/disc_updates_per_round": int(round_cfg.disc_updates_per_round),
                "train/expert_samples": int(expert.policy_observations.shape[0]),
                "train/reward_batch_size": int(reward_batch_size),
                "train/rollout_workers": int(round_cfg.num_rollout_workers),
                "train/rollout_worker_threads": int(round_cfg.rollout_worker_threads),
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
                        "round_config": vars(round_cfg),
                    },
                    checkpoint_path,
                )
                monitor.save(checkpoint_path)
            if should_save_checkpoint_video(round_cfg, round_idx):
                video_path = save_checkpoint_video(policy, round_cfg, run_dir=run_dir, round_idx=round_idx, device=device)
                if video_path is not None:
                    last_checkpoint_video_path = video_path
                    last_checkpoint_video_round = round_idx
                    monitor.log_video("checkpoint/policy_video", video_path, step=round_idx, fps=int(round_cfg.policy_frequency))

        final_round = int(cfg.total_rounds)
        final_round_cfg = config_for_round(cfg, final_round)
        final_path = os.path.join(run_dir, "final.pt")
        torch.save(
            {
                "round": final_round,
                "policy_state_dict": policy.state_dict(),
                "reward_state_dict": reward_model.state_dict(),
                "expert_metadata": expert.metadata,
                "config": vars(cfg),
                "round_config": vars(final_round_cfg),
            },
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
    finally:
        if rollout_executor is not None:
            rollout_executor.shutdown(wait=True, cancel_futures=True)
        if env is not None:
            env.close()
        monitor.finish()
    print(f"Saved final checkpoint under: {run_dir}")


if __name__ == "__main__":
    main()
