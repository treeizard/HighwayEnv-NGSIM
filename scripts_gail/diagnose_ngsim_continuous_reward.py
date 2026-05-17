#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scripts_gail.ps_gail.config import PSGAILConfig  # noqa: E402
from scripts_gail.ps_gail.data import load_expert_transition_data, standardize_features  # noqa: E402
from scripts_gail.ps_gail.envs import make_training_env  # noqa: E402
from scripts_gail.ps_gail.models import TrajectoryDiscriminator, make_actor_critic  # noqa: E402
from scripts_gail.ps_gail.trainer import (  # noqa: E402
    action_conditioned_features,
    collect_rollouts,
    discriminator_reward,
    infer_continuous_action_dim,
    infer_critic_obs_dim,
    infer_policy_obs_dim,
    refresh_rollout_rewards,
    shape_rollout_rewards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether action-conditioned WGAN rewards in NGSimEnv favor negative "
            "continuous actions, and explain WGAN expert_acc/centered_acc behavior."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="logs/simple_ps_gail/ps_gail_unified_continuous_test_55205758/final.pt",
    )
    parser.add_argument("--expert-data", default="")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollout-episodes", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--percentage-controlled-vehicles", type=float, default=-1.0)
    parser.add_argument("--max-expert-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-action-episodes", type=int, default=2)
    return parser.parse_args()


def localize_path(path: str, *, fallback: str) -> str:
    if not path:
        return fallback
    if os.path.exists(path):
        return path
    marker = "HighwayEnv-NGSIM"
    normalized = path.replace("\\", "/")
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1].lstrip("/")
        candidate = os.path.join(PARENT_DIR, suffix)
        if os.path.exists(candidate):
            return candidate
    return fallback


def cfg_from_checkpoint(checkpoint: dict[str, Any], args: argparse.Namespace) -> PSGAILConfig:
    defaults = PSGAILConfig()
    raw = checkpoint.get("config", {}) or {}
    valid = {field.name for field in fields(PSGAILConfig)}
    kwargs = {name: raw[name] for name in valid if name in raw}
    cfg = replace(defaults, **kwargs)
    cfg.action_mode = "continuous"
    cfg.discriminator_input = "action"
    cfg.discriminator_loss = str(getattr(cfg, "discriminator_loss", "wgan_gp") or "wgan_gp")
    cfg.expert_data = localize_path(
        args.expert_data or str(getattr(cfg, "expert_data", "")),
        fallback="expert_data/ngsim_ps_unified_expert_continuous_55145982",
    )
    cfg.episode_root = localize_path(
        str(args.episode_root or getattr(cfg, "episode_root", "")),
        fallback="highway_env/data/processed_20s",
    )
    cfg.device = str(args.device)
    cfg.seed = int(args.seed)
    cfg.max_expert_samples = int(args.max_expert_samples)
    cfg.rollout_steps = int(args.rollout_steps)
    cfg.rollout_min_episodes = int(args.rollout_episodes)
    cfg.rollout_full_episodes = True
    cfg.max_episode_steps = int(args.max_episode_steps)
    cfg.num_rollout_workers = 1
    cfg.save_checkpoint_video = False
    cfg.enable_scene_discriminator = False
    cfg.enable_sequence_discriminator = False
    if float(args.percentage_controlled_vehicles) >= 0.0:
        cfg.percentage_controlled_vehicles = float(args.percentage_controlled_vehicles)
    return cfg


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    keys = list(rows[0].keys())
    widths = {
        key: max(len(key), *(len(format_value(row.get(key))) for row in rows))
        for key in keys
    }
    print("  ".join(key.ljust(widths[key]) for key in keys))
    print("  ".join("-" * widths[key] for key in keys))
    for row in rows:
        print("  ".join(format_value(row.get(key)).ljust(widths[key]) for key in keys))


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def summarize_scores(name: str, expert_scores: np.ndarray, gen_scores: np.ndarray) -> None:
    expert_scores = np.asarray(expert_scores, dtype=np.float32).reshape(-1)
    gen_scores = np.asarray(gen_scores, dtype=np.float32).reshape(-1)
    threshold = 0.5 * (float(expert_scores.mean()) + float(gen_scores.mean()))
    print_table(
        name,
        [
            {
                "metric": "expert_score_mean",
                "value": float(expert_scores.mean()),
            },
            {"metric": "gen_score_mean", "value": float(gen_scores.mean())},
            {"metric": "critic_gap", "value": float(expert_scores.mean() - gen_scores.mean())},
            {"metric": "expert_positive_frac_zero_threshold", "value": float((expert_scores > 0).mean())},
            {"metric": "gen_negative_frac_zero_threshold", "value": float((gen_scores < 0).mean())},
            {"metric": "centered_threshold", "value": threshold},
            {"metric": "expert_acc_centered_threshold", "value": float((expert_scores > threshold).mean())},
            {"metric": "gen_acc_centered_threshold", "value": float((gen_scores < threshold).mean())},
        ],
    )


def fixed_action_reward_probe(
    discriminator: torch.nn.Module,
    observations: np.ndarray,
    cfg: PSGAILConfig,
    device: torch.device,
    normalizer: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    candidates = {
        "hard_brake": np.asarray([-1.0, 0.0], dtype=np.float32),
        "mild_brake": np.asarray([-0.3, 0.0], dtype=np.float32),
        "idle": np.asarray([0.0, 0.0], dtype=np.float32),
        "mild_accel": np.asarray([0.3, 0.0], dtype=np.float32),
        "hard_accel": np.asarray([1.0, 0.0], dtype=np.float32),
        "steer_left": np.asarray([0.0, 0.3], dtype=np.float32),
        "steer_right": np.asarray([0.0, -0.3], dtype=np.float32),
    }
    per_action_raw: dict[str, np.ndarray] = {}
    rows = []
    pooled_raw = []
    pooled_labels = []
    for label, action in candidates.items():
        actions = np.repeat(action.reshape(1, -1), len(observations), axis=0)
        features = action_conditioned_features(observations, actions)
        raw = discriminator_reward(
            discriminator,
            features,
            device,
            feature_normalizer=normalizer,
            feature_clip=float(cfg.discriminator_feature_clip),
            loss_type=str(cfg.discriminator_loss),
        )
        per_action_raw[label] = raw
        pooled_raw.append(raw)
        pooled_labels.extend([label] * len(raw))
    pooled_raw_arr = np.concatenate(pooled_raw).astype(np.float32)
    _unused, pooled_norm = shape_rollout_rewards(
        pooled_raw_arr,
        np.zeros_like(pooled_raw_arr, dtype=np.float32),
        replace(
            cfg,
            normalize_gail_reward=True,
            allow_wgan_reward_normalization=True,
            final_reward_clip=0.0,
            gail_reward_clip=0.0,
        ),
    )
    cursor = 0
    for label, raw in per_action_raw.items():
        norm = pooled_norm[cursor : cursor + len(raw)]
        cursor += len(raw)
        rows.append(
            {
                "action": label,
                "accel": float(candidates[label][0]),
                "steer": float(candidates[label][1]),
                "raw_mean": float(raw.mean()),
                "raw_p50": float(np.percentile(raw, 50)),
                "pooled_norm_mean": float(norm.mean()),
            }
        )
    rows = sorted(rows, key=lambda row: float(row["raw_mean"]), reverse=True)
    print_table("Fixed Action Discriminator Reward on NGSimEnv States", rows)


def policy_action_reward_probe(rollout, cfg: PSGAILConfig) -> None:
    rewards_norm_on, gail_norm_on = shape_rollout_rewards(
        rollout.gail_rewards_raw,
        rollout.env_penalties,
        replace(cfg, normalize_gail_reward=True, allow_wgan_reward_normalization=True),
    )
    rewards_norm_off, gail_norm_off = shape_rollout_rewards(
        rollout.gail_rewards_raw,
        rollout.env_penalties,
        replace(cfg, normalize_gail_reward=False),
    )
    actions = np.asarray(rollout.actions, dtype=np.float32)
    accel = actions[:, 0]
    steer = actions[:, 1]
    raw = np.asarray(rollout.gail_rewards_raw, dtype=np.float32)
    top = raw >= np.percentile(raw, 80)
    bottom = raw <= np.percentile(raw, 20)
    rows = [
        {"metric": "accel_mean", "value": float(accel.mean())},
        {"metric": "accel_negative_fraction", "value": float((accel < 0).mean())},
        {"metric": "steer_mean", "value": float(steer.mean())},
        {"metric": "raw_reward_mean", "value": float(raw.mean())},
        {"metric": "raw_reward_std", "value": float(raw.std())},
        {"metric": "norm_on_gail_mean", "value": float(gail_norm_on.mean())},
        {"metric": "norm_off_gail_mean", "value": float(gail_norm_off.mean())},
        {"metric": "norm_on_final_reward_mean", "value": float(rewards_norm_on.mean())},
        {"metric": "norm_off_final_reward_mean", "value": float(rewards_norm_off.mean())},
        {"metric": "top20_raw_accel_mean", "value": float(accel[top].mean())},
        {"metric": "top20_raw_accel_negative_fraction", "value": float((accel[top] < 0).mean())},
        {"metric": "bottom20_raw_accel_mean", "value": float(accel[bottom].mean())},
        {"metric": "bottom20_raw_accel_negative_fraction", "value": float((accel[bottom] < 0).mean())},
    ]
    if len(raw) > 2 and float(raw.std()) > 1e-8 and float(accel.std()) > 1e-8:
        rows.append({"metric": "corr_accel_raw_reward", "value": float(np.corrcoef(accel, raw)[0, 1])})
    if len(raw) > 2 and float(raw.std()) > 1e-8 and float(steer.std()) > 1e-8:
        rows.append({"metric": "corr_steer_raw_reward", "value": float(np.corrcoef(steer, raw)[0, 1])})
    print_table("Policy Rollout Action/Reward Probe", rows)


def deterministic_policy_probe(
    policy: torch.nn.Module,
    observations: np.ndarray,
    device: torch.device,
) -> None:
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        actions, _values = policy(obs)
    actions_np = actions.detach().cpu().numpy().astype(np.float32)
    rows = []
    for idx, name in enumerate(("acceleration", "steering")[: actions_np.shape[1]]):
        values = actions_np[:, idx]
        rows.extend(
            [
                {"metric": f"{name}_mean", "value": float(values.mean())},
                {"metric": f"{name}_std", "value": float(values.std())},
                {"metric": f"{name}_negative_fraction", "value": float((values < 0.0).mean())},
                {"metric": f"{name}_p05", "value": float(np.percentile(values, 5))},
                {"metric": f"{name}_p95", "value": float(np.percentile(values, 95))},
            ]
        )
    print_table("Deterministic Policy Output on NGSimEnv States", rows)


def fixed_action_env_probe(cfg: PSGAILConfig, action: np.ndarray, episodes: int) -> dict[str, Any]:
    env = make_training_env(cfg)
    lengths = []
    crashes = 0
    offroads = 0
    try:
        for ep in range(max(1, int(episodes))):
            obs, _info = env.reset(seed=int(cfg.seed) + ep)
            done = False
            steps = 0
            while not done and steps < int(cfg.max_episode_steps):
                num_agents = len(env.unwrapped.controlled_vehicles)
                action_tuple = tuple(action.astype(np.float32).copy() for _ in range(num_agents))
                obs, _reward, terminated, truncated, info = env.step(action_tuple)
                done = bool(terminated or truncated)
                steps += 1
                crashes += int(any(bool(flag) for flag in info.get("controlled_vehicle_crashes", [])))
                offroads += int(any(bool(flag) for flag in info.get("controlled_vehicle_offroad", [])))
            lengths.append(steps)
    finally:
        env.close()
    return {
        "mean_len": float(np.mean(lengths)) if lengths else 0.0,
        "min_len": int(np.min(lengths)) if lengths else 0,
        "crash_steps": int(crashes),
        "offroad_steps": int(offroads),
    }


def infer_discriminator_hidden_sizes(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    layer_shapes: list[tuple[int, int]] = []
    for key, value in state_dict.items():
        if not key.startswith("net.") or not key.endswith(".weight"):
            continue
        if value.ndim != 2:
            continue
        layer_shapes.append((int(key.split(".")[1]), int(value.shape[0])))
    if not layer_shapes:
        return None
    ordered = [out_dim for _layer_idx, out_dim in sorted(layer_shapes)]
    hidden = tuple(out_dim for out_dim in ordered if out_dim != 1)
    return hidden or None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(checkpoint, args)
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))

    env = make_training_env(cfg)
    try:
        cfg.continuous_action_dim = infer_continuous_action_dim(env)
        policy_obs_dim = infer_policy_obs_dim(env)
        critic_obs_dim = infer_critic_obs_dim(env, cfg, policy_obs_dim=policy_obs_dim)
    finally:
        env.close()

    transitions = load_expert_transition_data(
        cfg.expert_data,
        max_samples=int(cfg.max_expert_samples),
        seed=int(cfg.seed),
        trajectory_frame=str(cfg.trajectory_frame),
    )
    expert_features = action_conditioned_features(
        transitions.policy_observations,
        transitions.actions_continuous_env,
    )
    disc_state = checkpoint["discriminator_state_dict"]
    inferred_hidden_sizes = infer_discriminator_hidden_sizes(disc_state)
    discriminator = TrajectoryDiscriminator(
        int(expert_features.shape[1]),
        hidden_sizes=inferred_hidden_sizes or cfg.discriminator_hidden_sizes,
        dropout=float(cfg.discriminator_dropout),
        spectral_norm=bool(cfg.discriminator_spectral_norm),
    ).to(device)
    discriminator.load_state_dict(disc_state)
    discriminator.eval()
    normalizer = checkpoint.get("discriminator_feature_normalizer")

    policy = make_actor_critic(
        cfg.policy_model,
        policy_obs_dim,
        int(cfg.hidden_size),
        action_mode="continuous",
        continuous_action_dim=int(cfg.continuous_action_dim),
        transformer_layers=int(cfg.transformer_layers),
        transformer_heads=int(cfg.transformer_heads),
        transformer_dropout=float(cfg.transformer_dropout),
        centralized_critic=bool(cfg.centralized_critic),
        critic_obs_dim=critic_obs_dim,
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    env = make_training_env(cfg)
    try:
        rollout = collect_rollouts(
            env,
            policy,
            cfg,
            device,
            policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
        )
    finally:
        env.close()
    rollout = refresh_rollout_rewards(
        rollout,
        discriminator,
        cfg,
        device,
        discriminator_normalizer=normalizer,
    )

    expert_sample = expert_features[: min(len(expert_features), len(rollout.generator_features))]
    gen_sample = rollout.generator_features[: len(expert_sample)]
    if normalizer is not None:
        mean, std = normalizer
        expert_sample = standardize_features(
            expert_sample,
            mean,
            std,
            clip=float(cfg.discriminator_feature_clip),
        )
        gen_sample = standardize_features(
            gen_sample,
            mean,
            std,
            clip=float(cfg.discriminator_feature_clip),
        )
    with torch.no_grad():
        expert_scores = discriminator(torch.as_tensor(expert_sample, dtype=torch.float32, device=device)).cpu().numpy()
        gen_scores = discriminator(torch.as_tensor(gen_sample, dtype=torch.float32, device=device)).cpu().numpy()

    print(f"checkpoint: {Path(args.checkpoint)}")
    print(f"expert_data: {cfg.expert_data}")
    print(f"episode_root: {cfg.episode_root}")
    print(f"discriminator_loss: {cfg.discriminator_loss}")
    print(f"normalize_gail_reward_config: {cfg.normalize_gail_reward}")
    summarize_scores("WGAN Score Threshold Probe", expert_scores, gen_scores)
    fixed_action_reward_probe(discriminator, rollout.policy_observations, cfg, device, normalizer)
    deterministic_policy_probe(policy, rollout.policy_observations, device)
    policy_action_reward_probe(rollout, cfg)

    env_rows = []
    for label, action in {
        "hard_brake": np.asarray([-1.0, 0.0], dtype=np.float32),
        "idle": np.asarray([0.0, 0.0], dtype=np.float32),
        "hard_accel": np.asarray([1.0, 0.0], dtype=np.float32),
    }.items():
        row = {"action": label}
        row.update(fixed_action_env_probe(cfg, action, int(args.fixed_action_episodes)))
        env_rows.append(row)
    print_table("Fixed Action NGSimEnv Outcome Probe", env_rows)


if __name__ == "__main__":
    main()
