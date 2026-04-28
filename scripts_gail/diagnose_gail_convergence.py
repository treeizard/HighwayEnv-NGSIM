#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scripts_gail.ps_gail.config import PSGAILConfig  # noqa: E402
from scripts_gail.ps_gail.data import (  # noqa: E402
    load_expert_policy_and_disc_data,
    load_expert_scene_data,
    load_expert_sequence_data,
)
from scripts_gail.ps_gail.models import (  # noqa: E402
    SceneDiscriminator,
    SequenceTrajectoryDiscriminator,
    SharedActorCritic,
    TrajectoryDiscriminator,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short PS-GAIL convergence diagnostics before a long training job."
    )
    parser.add_argument("--expert-data", default="expert_data/ngsim_ps_traj_expert_discrete_54902119")
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-expert-samples", type=int, default=8192)
    parser.add_argument("--rollout-episodes", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--percentage-controlled-vehicles", type=float, default=0.2)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--trajectory-frame", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--disc-updates", type=int, default=3)
    parser.add_argument("--disc-batch-size", type=int, default=512)
    parser.add_argument("--enable-scene-discriminator", action="store_true")
    parser.add_argument("--enable-sequence-discriminator", action="store_true")
    parser.add_argument("--scene-max-vehicles", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=1)
    return parser.parse_args()


def percentile_summary(name: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {f"{name}_count": 0.0}
    return {
        f"{name}_count": float(arr.size),
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_std": float(np.std(arr)),
        f"{name}_min": float(np.min(arr)),
        f"{name}_p05": float(np.percentile(arr, 5)),
        f"{name}_p50": float(np.percentile(arr, 50)),
        f"{name}_p95": float(np.percentile(arr, 95)),
        f"{name}_max": float(np.max(arr)),
    }


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


def feature_scale_report(name: str, expert: np.ndarray, generator: np.ndarray) -> None:
    expert = np.asarray(expert, dtype=np.float32)
    generator = np.asarray(generator, dtype=np.float32)
    if expert.ndim == 3:
        expert = expert.reshape(-1, expert.shape[-1])
    if generator.ndim == 3:
        generator = generator.reshape(-1, generator.shape[-1])
    exp_std = expert.std(axis=0)
    gen_std = generator.std(axis=0)
    mean_gap = np.abs(expert.mean(axis=0) - generator.mean(axis=0))
    pooled_std = np.maximum(1e-6, 0.5 * (exp_std + gen_std))
    standardized_gap = mean_gap / pooled_std
    print_summary(
        f"{name} Feature Scale",
        {
            "expert_shape": tuple(expert.shape),
            "generator_shape": tuple(generator.shape),
            "expert_abs_max": float(np.max(np.abs(expert))) if expert.size else 0.0,
            "generator_abs_max": float(np.max(np.abs(generator))) if generator.size else 0.0,
            "max_standardized_mean_gap": float(np.max(standardized_gap)) if standardized_gap.size else 0.0,
            "median_standardized_mean_gap": float(np.median(standardized_gap)) if standardized_gap.size else 0.0,
            "near_constant_expert_dims": int(np.sum(exp_std < 1e-6)),
            "near_constant_generator_dims": int(np.sum(gen_std < 1e-6)),
        },
    )


def discriminator_probe(
    name: str,
    model: torch.nn.Module,
    expert: np.ndarray,
    generator: np.ndarray,
    cfg: PSGAILConfig,
    device: torch.device,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.disc_learning_rate))
    stats = update_discriminator_probe(model, optimizer, expert, generator, cfg, device)
    with torch.no_grad():
        expert_logits = model(torch.as_tensor(expert[: min(len(expert), 4096)], dtype=torch.float32, device=device))
        gen_logits = model(torch.as_tensor(generator[: min(len(generator), 4096)], dtype=torch.float32, device=device))
        expert_prob = torch.sigmoid(expert_logits).detach().cpu().numpy()
        gen_prob = torch.sigmoid(gen_logits).detach().cpu().numpy()
    report = {
        **stats,
        "expert_prob_mean": float(expert_prob.mean()) if expert_prob.size else float("nan"),
        "generator_prob_mean": float(gen_prob.mean()) if gen_prob.size else float("nan"),
        "prob_gap": float(expert_prob.mean() - gen_prob.mean()) if expert_prob.size and gen_prob.size else float("nan"),
    }
    print_summary(f"{name} Discriminator Probe", report)
    return report


def update_discriminator_probe(
    discriminator: torch.nn.Module,
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
        [
            np.full(len(expert), float(cfg.disc_expert_label), dtype=np.float32),
            np.full(len(generator_features), float(cfg.disc_generator_label), dtype=np.float32),
        ],
        axis=0,
    )
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(x), torch.as_tensor(y))
    loader = torch.utils.data.DataLoader(
        dataset,
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
                expert_mask = batch_y > 0.5
                gen_mask = batch_y < 0.5
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


def main() -> None:
    args = parse_args()
    from scripts_gail.ps_gail.envs import make_training_env
    from scripts_gail.ps_gail.trainer import collect_rollouts, infer_policy_obs_dim, refresh_rollout_rewards

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)
    cfg = PSGAILConfig(
        expert_data=args.expert_data,
        scene=args.scene,
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        seed=int(args.seed),
        max_expert_samples=int(args.max_expert_samples),
        trajectory_frame=str(args.trajectory_frame),
        max_surrounding=args.max_surrounding,
        control_all_vehicles=False,
        percentage_controlled_vehicles=float(args.percentage_controlled_vehicles),
        rollout_steps=int(args.rollout_steps),
        rollout_min_episodes=int(args.rollout_episodes),
        rollout_full_episodes=True,
        max_episode_steps=int(args.max_episode_steps),
        hidden_size=int(args.hidden_size),
        disc_updates_per_round=int(args.disc_updates),
        disc_batch_size=int(args.disc_batch_size),
        enable_scene_discriminator=bool(args.enable_scene_discriminator),
        enable_sequence_discriminator=bool(args.enable_sequence_discriminator),
        scene_max_vehicles=int(args.scene_max_vehicles),
        sequence_length=int(args.sequence_length),
        sequence_stride=int(args.sequence_stride),
        save_checkpoint_video=False,
        device=args.device,
    )

    expert_policy_obs, expert_features, expert_metadata = load_expert_policy_and_disc_data(
        cfg.expert_data,
        max_samples=cfg.max_expert_samples,
        seed=cfg.seed,
        trajectory_frame=cfg.trajectory_frame,
    )
    env = make_training_env(cfg)
    try:
        policy_obs_dim = infer_policy_obs_dim(env)
        policy = SharedActorCritic(
            policy_obs_dim,
            cfg.hidden_size,
            action_mode=cfg.action_mode,
            continuous_action_dim=cfg.continuous_action_dim,
        ).to(device)
        rollout = collect_rollouts(
            env,
            policy,
            cfg,
            device,
            policy_obs_dim,
            seed_offset=0,
        )
    finally:
        env.close()

    print_summary(
        "Expert Loader",
        {
            "expert_policy_obs_shape": tuple(expert_policy_obs.shape),
            "expert_features_shape": tuple(expert_features.shape),
            "num_files_loaded": expert_metadata.get("num_files_loaded"),
            "num_samples": expert_metadata.get("num_samples"),
            "trajectory_frame": expert_metadata.get("trajectory_frame"),
        },
    )
    print_summary(
        "Rollout",
        {
            "env_steps": rollout.num_env_steps,
            "agent_steps": rollout.num_agent_steps,
            "episodes": rollout.num_episodes,
            "mean_episode_length": rollout.mean_episode_length,
            "unique_episode_names": rollout.unique_episode_names,
            "mean_controlled_vehicles": rollout.mean_controlled_vehicles,
            "mean_road_vehicles": rollout.mean_road_vehicles,
            "crash_agent_fraction": rollout.crash_agent_fraction,
            "offroad_agent_fraction": rollout.offroad_agent_fraction,
            "sequence_windows": int(len(rollout.sequence_features)),
            "scene_snapshots": int(len(rollout.scene_features)),
        },
    )
    print_summary(
        "Action Histogram",
        {f"action_{int(action)}": int(count) for action, count in zip(*np.unique(rollout.actions, return_counts=True))},
    )
    feature_scale_report("State", expert_features, rollout.generator_features)
    base_disc = TrajectoryDiscriminator(int(expert_features.shape[1]), cfg.hidden_size).to(device)
    base_stats = discriminator_probe("State", base_disc, expert_features, rollout.generator_features, cfg, device)
    rollout = refresh_rollout_rewards(rollout, base_disc, cfg, device)
    print_summary(
        "Reward After Probe",
        {
            **percentile_summary("raw_gail", rollout.gail_rewards_raw),
            **percentile_summary("normalized_gail", rollout.gail_rewards_normalized),
            **percentile_summary("final_reward", rollout.rewards),
            **percentile_summary("advantages", rollout.advantages),
        },
    )

    if bool(args.enable_scene_discriminator):
        expert_scene, scene_meta = load_expert_scene_data(
            cfg.expert_data,
            max_samples=cfg.max_expert_samples,
            seed=cfg.seed,
            scene_max_vehicles=cfg.scene_max_vehicles,
        )
        print_summary("Scene Expert Loader", scene_meta)
        feature_scale_report("Scene", expert_scene, rollout.scene_features)
        scene_disc = SceneDiscriminator(int(expert_scene.shape[1]), cfg.hidden_size).to(device)
        discriminator_probe("Scene", scene_disc, expert_scene, rollout.scene_features, cfg, device)

    if bool(args.enable_sequence_discriminator):
        expert_seq, seq_meta = load_expert_sequence_data(
            cfg.expert_data,
            max_samples=cfg.max_expert_samples,
            seed=cfg.seed,
            trajectory_frame=cfg.trajectory_frame,
            sequence_length=cfg.sequence_length,
            sequence_stride=cfg.sequence_stride,
        )
        print_summary("Sequence Expert Loader", seq_meta)
        feature_scale_report("Sequence", expert_seq, rollout.sequence_features)
        seq_disc = SequenceTrajectoryDiscriminator(int(expert_seq.shape[-1]), cfg.hidden_size).to(device)
        discriminator_probe("Sequence", seq_disc, expert_seq, rollout.sequence_features, cfg, device)

    warnings = []
    if base_stats.get("expert_acc", 0.0) > 0.95 and base_stats.get("gen_acc", 0.0) > 0.95:
        warnings.append("State discriminator separates expert/generator almost immediately; check feature scaling or env mismatch.")
    if rollout.mean_episode_length < 0.5 * int(cfg.max_episode_steps):
        warnings.append("Rollouts are ending early; crashes/truncation may dominate policy gradients.")
    if rollout.gail_rewards_raw.std() < 1e-4:
        warnings.append("Raw GAIL reward has almost no variance after probe; policy will receive weak imitation signal.")
    if len(np.unique(rollout.actions)) <= 2:
        warnings.append("Random initial policy sampled very few action ids; check logits/action mapping if this persists after training.")
    print_summary("Warnings", {f"warning_{idx+1}": warning for idx, warning in enumerate(warnings)} or {"status": "no obvious red flags in short probe"})


if __name__ == "__main__":
    main()
