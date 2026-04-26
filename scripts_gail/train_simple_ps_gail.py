#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields

import numpy as np
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.data import load_expert_policy_and_disc_data
from scripts_gail.ps_gail.envs import make_training_env
from scripts_gail.ps_gail.monitoring import WandbMonitor
from scripts_gail.ps_gail.models import SharedActorCritic, TrajectoryDiscriminator
from scripts_gail.ps_gail.trainer import (
    collect_rollout,
    infer_policy_obs_dim,
    refresh_rollout_rewards,
    resolve_device,
    update_discriminator,
    update_policy,
)


def parse_args() -> PSGAILConfig:
    defaults = PSGAILConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Demonstration PS-GAIL trainer for NGSIM discrete meta actions. "
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
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = os.path.abspath(os.path.join("logs", "simple_ps_gail", cfg.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    monitor = WandbMonitor(cfg, run_dir)
    monitor.start()

    env = None
    try:
        expert_policy_obs, expert_features, expert_metadata = load_expert_policy_and_disc_data(
            cfg.expert_data,
            max_samples=cfg.max_expert_samples,
            seed=cfg.seed,
        )
        env = make_training_env(cfg)
        policy_obs_dim = infer_policy_obs_dim(env)
        if policy_obs_dim != expert_metadata["policy_observation_dim"]:
            raise RuntimeError(
                "Expert/generator policy observation dimensions differ: "
                f"{expert_metadata['policy_observation_dim']} != {policy_obs_dim}."
            )
        feature_dim = int(expert_features.shape[1])
        policy = SharedActorCritic(policy_obs_dim, cfg.hidden_size).to(device)
        discriminator = TrajectoryDiscriminator(feature_dim, cfg.hidden_size).to(device)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.disc_learning_rate)
        monitor.watch(policy, discriminator)

        print(f"Loaded expert folder: {os.path.abspath(cfg.expert_data)}")
        print(
            f"expert_policy_obs={expert_policy_obs.shape} expert_disc_features={expert_features.shape}"
        )
        print(
            f"collision_enabled={cfg.enable_collision} allow_idm={cfg.allow_idm} "
            f"policy_obs_dim={policy_obs_dim} disc_feature_dim={feature_dim} device={device}"
        )

        for round_idx in range(1, int(cfg.total_rounds) + 1):
            rollout = collect_rollout(env, policy, cfg, device)
            disc_stats = update_discriminator(
                discriminator,
                disc_optimizer,
                expert_features,
                rollout.generator_features,
                cfg,
                device,
            )
            rollout = refresh_rollout_rewards(rollout, discriminator, cfg, device)
            policy_stats = update_policy(policy, policy_optimizer, rollout, cfg, device)

            print(
                f"[round {round_idx:04d}] "
                f"env_steps={rollout.num_env_steps} agent_steps={rollout.num_agent_steps} "
                f"disc_loss={disc_stats['disc_loss']:.4f} "
                f"expert_acc={disc_stats['expert_acc']:.3f} gen_acc={disc_stats['gen_acc']:.3f} "
                f"policy_loss={policy_stats['policy_loss']:.4f} "
                f"value_loss={policy_stats['value_loss']:.4f} "
                f"entropy={policy_stats['entropy']:.4f} "
                f"mean_gail_reward={float(rollout.rewards.mean()):.4f}"
            )
            monitor.log(
                {
                    "round": round_idx,
                    "rollout/env_steps": rollout.num_env_steps,
                    "rollout/agent_steps": rollout.num_agent_steps,
                    "rollout/mean_gail_reward": float(rollout.rewards.mean()),
                    "rollout/reward_std": float(rollout.rewards.std()),
                    "rollout/action_mean": float(rollout.actions.mean()),
                    "rollout/action_std": float(rollout.actions.std()),
                    "discriminator/loss": disc_stats["disc_loss"],
                    "discriminator/expert_acc": disc_stats["expert_acc"],
                    "discriminator/gen_acc": disc_stats["gen_acc"],
                    "policy/loss": policy_stats["policy_loss"],
                    "policy/value_loss": policy_stats["value_loss"],
                    "policy/entropy": policy_stats["entropy"],
                    "train/policy_obs_dim": policy_obs_dim,
                    "train/disc_feature_dim": feature_dim,
                    "train/expert_samples": int(expert_features.shape[0]),
                },
                step=round_idx,
            )

            if cfg.checkpoint_every > 0 and round_idx % int(cfg.checkpoint_every) == 0:
                checkpoint_path = os.path.join(ckpt_dir, f"round_{round_idx:04d}.pt")
                torch.save(
                    {
                        "round": round_idx,
                        "policy_state_dict": policy.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "config": vars(cfg),
                    },
                    checkpoint_path,
                )
                monitor.save(checkpoint_path)

        final_path = os.path.join(run_dir, "final.pt")
        torch.save(
            {
                "round": int(cfg.total_rounds),
                "policy_state_dict": policy.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "config": vars(cfg),
            },
            final_path,
        )
        monitor.save(final_path)
    finally:
        if env is not None:
            env.close()
        monitor.finish()

    print(f"Saved final checkpoint under: {run_dir}")


if __name__ == "__main__":
    main()
