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
from torch.distributions import Categorical, Independent, Normal

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.data import (
    load_expert_policy_and_disc_data,
    load_expert_scene_data,
    load_expert_sequence_data,
)
from scripts_gail.ps_gail.envs import make_training_env
from scripts_gail.ps_gail.monitoring import WandbMonitor
from scripts_gail.ps_gail.models import (
    SceneDiscriminator,
    SequenceTrajectoryDiscriminator,
    SharedActorCritic,
    TrajectoryDiscriminator,
    make_actor_critic,
)
from scripts_gail.ps_gail.observations import flatten_agent_observations, policy_observations_from_flat
from scripts_gail.ps_gail.trainer import (
    collect_rollouts,
    infer_continuous_action_dim,
    infer_policy_obs_dim,
    make_rollout_executor,
    refresh_rollout_rewards,
    resolve_device,
    update_discriminator,
    update_policy,
)


def _clamp_fraction(value: float) -> float:
    return float(min(1.0, max(1e-6, float(value))))


def config_for_round(cfg: PSGAILConfig, round_idx: int) -> PSGAILConfig:
    if not bool(cfg.controlled_vehicle_curriculum):
        return cfg

    curriculum_rounds = max(1, int(cfg.controlled_vehicle_curriculum_rounds))
    progress = 1.0 if curriculum_rounds == 1 else (
        min(1.0, max(0.0, (int(round_idx) - 1) / float(curriculum_rounds - 1)))
    )
    initial = _clamp_fraction(cfg.initial_controlled_vehicle_fraction)
    final = _clamp_fraction(cfg.final_controlled_vehicle_fraction)
    fraction = initial + (final - initial) * progress
    return replace(
        cfg,
        control_all_vehicles=False,
        percentage_controlled_vehicles=_clamp_fraction(fraction),
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


def _policy_action_tuple(
    policy: SharedActorCritic,
    obs,
    *,
    device: torch.device,
    deterministic: bool,
) -> tuple[object, ...]:
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
        policy_out, _values = policy(obs_tensor)
        if str(policy.action_mode).lower() == "continuous":
            if deterministic:
                actions = policy_out
            else:
                if policy.log_std is None:
                    raise RuntimeError("Continuous action mode requires policy.log_std.")
                std = torch.exp(policy.log_std).expand_as(policy_out)
                actions = Independent(Normal(policy_out, std), 1).sample()
            actions_np = torch.clamp(actions, -1.0, 1.0).detach().cpu().numpy().astype(np.float32)
            return tuple(action.copy() for action in actions_np)
        if deterministic:
            actions = torch.argmax(policy_out, dim=-1)
        else:
            actions = Categorical(logits=policy_out).sample()
    return tuple(int(action) for action in actions.detach().cpu().numpy().tolist())


def save_checkpoint_video(
    policy: SharedActorCritic,
    cfg: PSGAILConfig,
    *,
    run_dir: str,
    round_idx: int,
    device: torch.device,
) -> str | None:
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

    video_env = make_training_env(cfg, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    was_training = policy.training
    policy.eval()
    try:
        base = video_env.unwrapped
        base.config["offscreen_rendering"] = True
        base.config["screen_width"] = int(cfg.checkpoint_video_width)
        base.config["screen_height"] = int(cfg.checkpoint_video_height)
        base.config["scaling"] = float(cfg.checkpoint_video_scaling)
        obs, _info = video_env.reset(seed=int(cfg.seed) + int(round_idx) * 1009)
        frame = video_env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        for _step in range(max(1, int(cfg.checkpoint_video_steps))):
            action = _policy_action_tuple(
                policy,
                obs,
                device=device,
                deterministic=bool(cfg.checkpoint_video_deterministic),
            )
            obs, _reward, terminated, truncated, _info = video_env.step(action)
            frame = video_env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            if terminated or truncated:
                break
    finally:
        video_env.close()
        if was_training:
            policy.train()

    if not frames:
        return None
    with imageio.get_writer(path, fps=max(1, int(cfg.policy_frequency))) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return path


def parse_args() -> PSGAILConfig:
    defaults = PSGAILConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Demonstration PS-GAIL trainer for NGSIM discrete or continuous actions. "
            "Policy input is lidar + lane + [length, velocity, heading]. "
            "The discriminator compares policy observation + trajectory [x, y, v], not DMA."
        )
    )
    for field in fields(PSGAILConfig):
        value = getattr(defaults, field.name)
        arg = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    args = parser.parse_args()
    return PSGAILConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    if bool(cfg.enable_sequence_discriminator) and bool(cfg.enable_scene_discriminator):
        raise ValueError(
            "Sequence-discriminator training now uses a single sequential discriminator. "
            "Disable --enable-scene-discriminator for sequential GAIL."
        )
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = os.path.abspath(os.path.join("logs", "simple_ps_gail", cfg.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    monitor = WandbMonitor(cfg, run_dir)
    monitor.start()

    env = None
    rollout_executor = None
    try:
        expert_policy_obs, expert_features, expert_metadata = load_expert_policy_and_disc_data(
            cfg.expert_data,
            max_samples=cfg.max_expert_samples,
            seed=cfg.seed,
            trajectory_frame=cfg.trajectory_frame,
        )
        sequence_only_discriminator = bool(cfg.enable_sequence_discriminator)
        expert_scene_features = None
        expert_scene_metadata = {}
        if bool(cfg.enable_scene_discriminator):
            expert_scene_features, expert_scene_metadata = load_expert_scene_data(
                cfg.expert_data,
                max_samples=cfg.max_expert_samples,
                seed=cfg.seed,
                scene_max_vehicles=cfg.scene_max_vehicles,
            )
        expert_sequence_features = None
        expert_sequence_metadata = {}
        if bool(cfg.enable_sequence_discriminator):
            expert_sequence_features, expert_sequence_metadata = load_expert_sequence_data(
                cfg.expert_data,
                max_samples=cfg.max_expert_samples,
                seed=cfg.seed,
                trajectory_frame=cfg.trajectory_frame,
                sequence_length=cfg.sequence_length,
                sequence_stride=cfg.sequence_stride,
            )
        env_cfg = config_for_round(cfg, 1)
        env = make_training_env(env_cfg)
        if str(env_cfg.action_mode).lower() == "continuous":
            continuous_action_dim = infer_continuous_action_dim(env)
            cfg.continuous_action_dim = continuous_action_dim
            env_cfg.continuous_action_dim = continuous_action_dim
        current_env_signature = env_signature(env_cfg)
        policy_obs_dim = infer_policy_obs_dim(env)
        if policy_obs_dim != expert_metadata["policy_observation_dim"]:
            raise RuntimeError(
                "Expert/generator policy observation dimensions differ: "
                f"{expert_metadata['policy_observation_dim']} != {policy_obs_dim}."
            )
        feature_dim = int(expert_features.shape[1])
        policy = make_actor_critic(
            cfg.policy_model,
            policy_obs_dim,
            cfg.hidden_size,
            action_mode=str(cfg.action_mode),
            continuous_action_dim=int(cfg.continuous_action_dim),
            transformer_layers=int(cfg.transformer_layers),
            transformer_heads=int(cfg.transformer_heads),
            transformer_dropout=float(cfg.transformer_dropout),
        ).to(device)
        if sequence_only_discriminator:
            if expert_sequence_features is None:
                raise RuntimeError("Sequence discriminator was enabled but no expert sequence features were loaded.")
            discriminator = SequenceTrajectoryDiscriminator(
                int(expert_sequence_features.shape[-1]),
                cfg.hidden_size,
            ).to(device)
            discriminator_expert_features = expert_sequence_features
            discriminator_name = "sequence"
        else:
            discriminator = TrajectoryDiscriminator(feature_dim, cfg.hidden_size).to(device)
            discriminator_expert_features = expert_features
            discriminator_name = "trajectory"
        scene_discriminator = (
            SceneDiscriminator(int(expert_scene_features.shape[1]), cfg.hidden_size).to(device)
            if expert_scene_features is not None
            else None
        )
        sequence_discriminator = discriminator if sequence_only_discriminator else None
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.disc_learning_rate)
        scene_disc_optimizer = (
            torch.optim.Adam(scene_discriminator.parameters(), lr=cfg.disc_learning_rate)
            if scene_discriminator is not None
            else None
        )
        monitor.watch(policy, discriminator)

        print(f"Loaded expert folder: {os.path.abspath(cfg.expert_data)}")
        print(
            f"expert_policy_obs={expert_policy_obs.shape} expert_disc_features={expert_features.shape}"
        )
        print(
            "expert_sampling="
            f"{expert_metadata.get('sampling', 'unknown')} "
            f"trajectory_frame={expert_metadata.get('trajectory_frame', 'unknown')} "
            f"loaded_files={expert_metadata.get('num_files_loaded', expert_metadata.get('num_files_seen'))} "
            f"loaded_samples={expert_metadata.get('num_samples')} "
            f"source_samples={expert_metadata.get('num_source_samples', 'unknown')}"
        )
        print(
            f"collision_enabled={cfg.enable_collision} allow_idm={cfg.allow_idm} "
            f"action_mode={cfg.action_mode} "
            f"policy_model={cfg.policy_model} "
            f"policy_obs_dim={policy_obs_dim} disc_feature_dim={feature_dim} device={device} "
            f"discriminator={discriminator_name} "
            f"cgail_k={cfg.cgail_k} "
            f"rollout_workers={cfg.num_rollout_workers} "
            f"rollout_worker_threads={cfg.rollout_worker_threads}"
        )
        if expert_scene_features is not None:
            print(
                "scene_discriminator="
                f"samples={expert_scene_metadata.get('num_scene_samples')} "
                f"feature_dim={expert_scene_metadata.get('scene_feature_dim')} "
                f"max_vehicles={cfg.scene_max_vehicles}"
            )
        if expert_sequence_features is not None:
            print(
                "sequence_discriminator="
                f"samples={expert_sequence_metadata.get('num_sequence_samples')} "
                f"length={expert_sequence_metadata.get('sequence_length')} "
                f"feature_dim={expert_sequence_metadata.get('sequence_feature_dim')} "
                f"trajectory_frame={cfg.trajectory_frame}"
            )
        if cfg.controlled_vehicle_curriculum:
            print(
                "controlled_vehicle_curriculum="
                f"initial={cfg.initial_controlled_vehicle_fraction:.4f} "
                f"final={cfg.final_controlled_vehicle_fraction:.4f} "
                f"rounds={cfg.controlled_vehicle_curriculum_rounds}"
            )
        rollout_executor = make_rollout_executor(cfg)

        for round_idx in range(1, int(cfg.total_rounds) + 1):
            round_cfg = config_for_round(cfg, round_idx)
            round_env_signature = env_signature(round_cfg)
            if round_env_signature != current_env_signature:
                if env is not None:
                    env.close()
                env = make_training_env(round_cfg)
                current_env_signature = round_env_signature

            rollout = collect_rollouts(
                env,
                policy,
                round_cfg,
                device,
                policy_obs_dim,
                seed_offset=(round_idx - 1) * max(1, int(cfg.num_rollout_workers)),
                executor=rollout_executor,
            )
            if sequence_only_discriminator and not rollout.sequence_features.size:
                raise RuntimeError(
                    "Sequence discriminator was enabled but rollout produced no sequence windows. "
                    "Lower --sequence-length or collect longer rollout episodes."
                )
            disc_stats = update_discriminator(
                discriminator,
                disc_optimizer,
                discriminator_expert_features,
                rollout.sequence_features if sequence_only_discriminator else rollout.generator_features,
                round_cfg,
                device,
            )
            scene_disc_stats = {}
            if scene_discriminator is not None:
                if scene_disc_optimizer is None or expert_scene_features is None:
                    raise RuntimeError("Scene discriminator was enabled without optimizer/expert features.")
                if not rollout.scene_features.size:
                    raise RuntimeError("Scene discriminator was enabled but rollout produced no scene features.")
                scene_disc_stats = update_discriminator(
                    scene_discriminator,
                    scene_disc_optimizer,
                    expert_scene_features,
                    rollout.scene_features,
                    round_cfg,
                    device,
                )
            sequence_disc_stats = disc_stats if sequence_only_discriminator else {}
            rollout = refresh_rollout_rewards(
                rollout,
                None if sequence_only_discriminator else discriminator,
                round_cfg,
                device,
                scene_discriminator=scene_discriminator,
                sequence_discriminator=sequence_discriminator,
            )
            policy_stats = update_policy(policy, policy_optimizer, rollout, round_cfg, device)

            print(
                f"[round {round_idx:04d}] "
                f"env_steps={rollout.num_env_steps} agent_steps={rollout.num_agent_steps} "
                f"episodes={rollout.num_episodes} term={rollout.num_terminated} "
                f"trunc={rollout.num_truncated} crash_eps={rollout.num_crash_events} "
                f"offroad_eps={rollout.num_offroad_events} "
                f"ep_len={rollout.mean_episode_length:.1f}"
                f"[{rollout.min_episode_length},{rollout.max_episode_length}] "
                f"uniq_eps={rollout.unique_episode_names} "
                f"ctrl_frac={round_cfg.percentage_controlled_vehicles:.4f} "
                f"veh={rollout.mean_controlled_vehicles:.1f}/{rollout.mean_road_vehicles:.1f} "
                f"disc_loss={disc_stats['disc_loss']:.4f} "
                + (
                    f"cgail_penalty={disc_stats['cgail_penalty']:.6f} "
                    if float(round_cfg.cgail_k) > 0.0
                    else ""
                )
                + (
                    f"scene_disc_loss={scene_disc_stats['disc_loss']:.4f} "
                    if scene_disc_stats
                    else ""
                )
                + (
                    f"seq_disc_loss={sequence_disc_stats['disc_loss']:.4f} "
                    if sequence_disc_stats
                    else ""
                )
                + (
                    f"scene_n={len(rollout.scene_features)} "
                    if scene_discriminator is not None
                    else ""
                )
                + (
                    f"seq_n={len(rollout.sequence_features)} "
                    if sequence_discriminator is not None
                    else ""
                )
                + (
                f"expert_acc={disc_stats['expert_acc']:.3f} gen_acc={disc_stats['gen_acc']:.3f} "
                f"policy_loss={policy_stats['policy_loss']:.4f} "
                f"value_loss={policy_stats['value_loss']:.4f} "
                f"entropy={policy_stats['entropy']:.4f} "
                f"raw_gail={rollout.mean_raw_gail_reward:.4f} "
                f"norm_gail={rollout.mean_normalized_gail_reward:.4f} "
                f"env_penalty={rollout.mean_env_penalty:.4f} "
                f"reward={float(rollout.rewards.mean()):.4f}"
                )
            )
            metrics = {
                "round": round_idx,
                "rollout/env_steps": rollout.num_env_steps,
                "rollout/agent_steps": rollout.num_agent_steps,
                "rollout/episodes": rollout.num_episodes,
                "rollout/terminated": rollout.num_terminated,
                "rollout/truncated": rollout.num_truncated,
                "rollout/crash_episodes": rollout.num_crash_events,
                "rollout/offroad_episodes": rollout.num_offroad_events,
                "rollout/crash_agent_fraction": rollout.crash_agent_fraction,
                "rollout/offroad_agent_fraction": rollout.offroad_agent_fraction,
                "rollout/mean_episode_length": rollout.mean_episode_length,
                "rollout/min_episode_length": rollout.min_episode_length,
                "rollout/max_episode_length": rollout.max_episode_length,
                "rollout/unique_episode_names": rollout.unique_episode_names,
                "rollout/controlled_vehicle_fraction": float(round_cfg.percentage_controlled_vehicles),
                "rollout/mean_controlled_vehicles": rollout.mean_controlled_vehicles,
                "rollout/mean_road_vehicles": rollout.mean_road_vehicles,
                "rollout/scene_samples": int(len(rollout.scene_features)),
                "rollout/sequence_samples": int(len(rollout.sequence_features)),
                "rollout/mean_gail_reward": float(rollout.rewards.mean()),
                "rollout/mean_raw_gail_reward": rollout.mean_raw_gail_reward,
                "rollout/mean_normalized_gail_reward": rollout.mean_normalized_gail_reward,
                "rollout/mean_env_penalty": rollout.mean_env_penalty,
                "rollout/reward_std": float(rollout.rewards.std()),
                "rollout/raw_gail_reward_std": float(rollout.gail_rewards_raw.std()),
                "rollout/normalized_gail_reward_std": float(rollout.gail_rewards_normalized.std()),
                "rollout/action_mean": float(rollout.actions.mean()),
                "rollout/action_std": float(rollout.actions.std()),
                "discriminator/loss": disc_stats["disc_loss"],
                "discriminator/bce_loss": disc_stats["disc_bce_loss"],
                "discriminator/cgail_penalty": disc_stats["cgail_penalty"],
                "discriminator/prob_mean": disc_stats["disc_prob_mean"],
                "discriminator/prob_std": disc_stats["disc_prob_std"],
                "discriminator/expert_prob_mean": disc_stats["expert_prob_mean"],
                "discriminator/gen_prob_mean": disc_stats["gen_prob_mean"],
                "discriminator/expert_acc": disc_stats["expert_acc"],
                "discriminator/gen_acc": disc_stats["gen_acc"],
                "policy/loss": policy_stats["policy_loss"],
                "policy/value_loss": policy_stats["value_loss"],
                "policy/entropy": policy_stats["entropy"],
                "train/policy_obs_dim": policy_obs_dim,
                "train/disc_feature_dim": feature_dim,
                "train/expert_samples": int(expert_features.shape[0]),
                "train/rollout_workers": int(cfg.num_rollout_workers),
                "train/rollout_worker_threads": int(cfg.rollout_worker_threads),
            }
            if scene_disc_stats:
                metrics.update(
                    {
                        "scene_discriminator/loss": scene_disc_stats["disc_loss"],
                        "scene_discriminator/bce_loss": scene_disc_stats["disc_bce_loss"],
                        "scene_discriminator/cgail_penalty": scene_disc_stats["cgail_penalty"],
                        "scene_discriminator/prob_mean": scene_disc_stats["disc_prob_mean"],
                        "scene_discriminator/prob_std": scene_disc_stats["disc_prob_std"],
                        "scene_discriminator/expert_prob_mean": scene_disc_stats["expert_prob_mean"],
                        "scene_discriminator/gen_prob_mean": scene_disc_stats["gen_prob_mean"],
                        "scene_discriminator/expert_acc": scene_disc_stats["expert_acc"],
                        "scene_discriminator/gen_acc": scene_disc_stats["gen_acc"],
                    }
                )
            if sequence_disc_stats:
                metrics.update(
                    {
                        "sequence_discriminator/loss": sequence_disc_stats["disc_loss"],
                        "sequence_discriminator/bce_loss": sequence_disc_stats["disc_bce_loss"],
                        "sequence_discriminator/cgail_penalty": sequence_disc_stats["cgail_penalty"],
                        "sequence_discriminator/prob_mean": sequence_disc_stats["disc_prob_mean"],
                        "sequence_discriminator/prob_std": sequence_disc_stats["disc_prob_std"],
                        "sequence_discriminator/expert_prob_mean": sequence_disc_stats["expert_prob_mean"],
                        "sequence_discriminator/gen_prob_mean": sequence_disc_stats["gen_prob_mean"],
                        "sequence_discriminator/expert_acc": sequence_disc_stats["expert_acc"],
                        "sequence_discriminator/gen_acc": sequence_disc_stats["gen_acc"],
                    }
                )
            monitor.log(
                metrics,
                step=round_idx,
            )

            if cfg.checkpoint_every > 0 and round_idx % int(cfg.checkpoint_every) == 0:
                checkpoint_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}.pt")
                torch.save(
                    {
                        "round": round_idx,
                        "policy_state_dict": policy.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "scene_discriminator_state_dict": scene_discriminator.state_dict()
                        if scene_discriminator is not None
                        else None,
                        "sequence_discriminator_state_dict": discriminator.state_dict()
                        if sequence_only_discriminator
                        else None,
                        "discriminator_type": discriminator_name,
                        "config": vars(cfg),
                        "round_config": vars(round_cfg),
                    },
                    checkpoint_path,
                )
                monitor.save(checkpoint_path)
                video_path = save_checkpoint_video(
                    policy,
                    round_cfg,
                    run_dir=run_dir,
                    round_idx=round_idx,
                    device=device,
                )
                if video_path is not None:
                    monitor.log_video(
                        "checkpoint/policy_video",
                        video_path,
                        step=round_idx,
                        fps=int(round_cfg.policy_frequency),
                    )

        final_path = os.path.join(run_dir, "final.pt")
        torch.save(
            {
                "round": int(cfg.total_rounds),
                "policy_state_dict": policy.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "scene_discriminator_state_dict": scene_discriminator.state_dict()
                if scene_discriminator is not None
                else None,
                "sequence_discriminator_state_dict": discriminator.state_dict()
                if sequence_only_discriminator
                else None,
                "discriminator_type": discriminator_name,
                "config": vars(cfg),
                "round_config": vars(config_for_round(cfg, int(cfg.total_rounds))),
            },
            final_path,
        )
        monitor.save(final_path)
        final_video_path = save_checkpoint_video(
            policy,
            config_for_round(cfg, int(cfg.total_rounds)),
            run_dir=run_dir,
            round_idx=int(cfg.total_rounds),
            device=device,
        )
        if final_video_path is not None:
            monitor.log_video(
                "checkpoint/final_policy_video",
                final_video_path,
                step=int(cfg.total_rounds),
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
