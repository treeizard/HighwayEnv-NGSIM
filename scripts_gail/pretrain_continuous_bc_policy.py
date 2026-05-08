#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.envs import observation_config
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.observations import flatten_agent_observations, policy_observations_from_flat
from scripts_gail.ps_gail.trainer import infer_continuous_action_dim, infer_policy_obs_dim, resolve_device
from scripts_gail.train_simple_ps_gail import behavior_clone_pretrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the continuous PS-GAIL policy by behavior cloning, save a reusable "
            "checkpoint, and render one selected NGSIM replay vehicle with the pretrained policy."
        )
    )
    parser.add_argument("--expert-data", default="expert_data/ngsim_ps_unified_expert_continuous_55145982")
    parser.add_argument("--checkpoint-out", default="logs/bc_pretrain_continuous/bc_policy.pt")
    parser.add_argument("--checkpoint-in", default="")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--max-expert-samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", default="train")
    parser.add_argument("--trajectory-frame", default="relative", choices=["relative", "absolute"])
    parser.add_argument("--episode-name", default="")
    parser.add_argument("--vehicle-id", type=int, default=-1)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--allow-idm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminate-when-all-controlled-crashed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--simulation-frequency", type=int, default=10)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--maximum-range", type=float, default=64.0)

    parser.add_argument("--policy-model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)

    parser.add_argument("--video-out", default="logs/bc_pretrain_continuous/sample_replay.mp4")
    parser.add_argument("--video-steps", type=int, default=200)
    parser.add_argument("--screen-width", type=int, default=1200)
    parser.add_argument("--screen-height", type=int, default=608)
    parser.add_argument("--scaling", type=float, default=5.5)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> PSGAILConfig:
    return PSGAILConfig(
        expert_data=str(args.expert_data),
        scene=str(args.scene),
        action_mode="continuous",
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.prebuilt_split),
        seed=int(args.seed),
        max_expert_samples=int(args.max_expert_samples),
        trajectory_frame=str(args.trajectory_frame),
        max_surrounding=args.max_surrounding,
        control_all_vehicles=False,
        percentage_controlled_vehicles=1.0,
        enable_collision=bool(args.enable_collision),
        terminate_when_all_controlled_crashed=bool(args.terminate_when_all_controlled_crashed),
        allow_idm=bool(args.allow_idm),
        cells=int(args.cells),
        maximum_range=float(args.maximum_range),
        simulation_frequency=int(args.simulation_frequency),
        policy_frequency=int(args.policy_frequency),
        max_episode_steps=int(args.max_episode_steps),
        policy_model=str(args.policy_model),
        hidden_size=int(args.hidden_size),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_dropout=float(args.transformer_dropout),
        bc_pretrain_epochs=int(args.epochs),
        bc_pretrain_learning_rate=float(args.learning_rate),
        bc_pretrain_batch_size=int(args.batch_size),
        bc_pretrain_weight_decay=float(args.weight_decay),
        bc_pretrain_validation_fraction=float(args.validation_fraction),
        device=str(args.device),
    )


def default_scenario_from_expert_folder(path: str) -> tuple[str | None, int | None]:
    files = []
    if os.path.isdir(path):
        files = sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".npz"))
    elif os.path.exists(path):
        files = [path]
    for file_path in files:
        with np.load(file_path, allow_pickle=True) as data:
            metadata: dict[str, Any] = {}
            if "metadata_json" in data.files:
                metadata = json.loads(str(data["metadata_json"].item()))
            episode_name = (
                metadata.get("collection_episode_name")
                or metadata.get("episode_name")
                or metadata.get("fixed_episode_name")
            )
            vehicle_ids = np.asarray(data["vehicle_ids"], dtype=np.int64) if "vehicle_ids" in data.files else np.asarray([])
            if episode_name is not None and vehicle_ids.size:
                return str(episode_name), int(np.unique(vehicle_ids)[0])
    return None, None


def make_selected_replay_env(
    cfg: PSGAILConfig,
    *,
    episode_name: str,
    vehicle_id: int,
    render_mode: str | None,
) -> gym.Env:
    register_ngsim_env()
    env_cfg = build_env_config(
        scene=cfg.scene,
        action_mode="continuous",
        episode_root=cfg.episode_root,
        prebuilt_split=cfg.prebuilt_split,
        percentage_controlled_vehicles=1.0,
        control_all_vehicles=False,
        max_surrounding=cfg.max_surrounding,
        observation_config=observation_config(cfg),
        simulation_frequency=cfg.simulation_frequency,
        policy_frequency=cfg.policy_frequency,
        max_episode_steps=cfg.max_episode_steps,
        seed=None,
        simulation_period={"episode_name": str(episode_name)},
        ego_vehicle_id=[int(vehicle_id)],
        scene_dataset_collection_mode=False,
        allow_idm=cfg.allow_idm,
    )
    env_cfg["expert_test_mode"] = False
    env_cfg["disable_controlled_vehicle_collisions"] = not bool(cfg.enable_collision)
    env_cfg["terminate_when_all_controlled_crashed"] = bool(cfg.terminate_when_all_controlled_crashed)
    env_cfg["allow_idm"] = bool(cfg.allow_idm)
    env_cfg["crash_controlled_vehicles_offroad"] = True
    env_cfg["offscreen_rendering"] = render_mode == "rgb_array"
    env_cfg["screen_width"] = int(getattr(cfg, "checkpoint_video_width", 1200))
    env_cfg["screen_height"] = int(getattr(cfg, "checkpoint_video_height", 608))
    env_cfg["scaling"] = float(getattr(cfg, "checkpoint_video_scaling", 5.5))
    return gym.make(ENV_ID, render_mode=render_mode, config=env_cfg)


def build_policy_for_env(cfg: PSGAILConfig, env: gym.Env, device: torch.device) -> tuple[torch.nn.Module, int, int]:
    obs_dim = infer_policy_obs_dim(env)
    action_dim = infer_continuous_action_dim(env)
    policy = make_actor_critic(
        cfg.policy_model,
        obs_dim,
        cfg.hidden_size,
        action_mode="continuous",
        continuous_action_dim=action_dim,
        transformer_layers=int(cfg.transformer_layers),
        transformer_heads=int(cfg.transformer_heads),
        transformer_dropout=float(cfg.transformer_dropout),
    ).to(device)
    return policy, obs_dim, action_dim


def load_policy_checkpoint(policy: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    policy.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def policy_action_tuple(
    policy: torch.nn.Module,
    obs: Any,
    *,
    device: torch.device,
    deterministic: bool,
) -> tuple[np.ndarray, ...]:
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
        mean_actions, _values = policy(obs_tensor)
        if deterministic:
            actions = mean_actions
        else:
            log_std = getattr(policy, "log_std", None)
            if log_std is None:
                raise RuntimeError("Continuous policy checkpoint does not expose log_std.")
            std = torch.exp(log_std).expand_as(mean_actions)
            actions = torch.normal(mean_actions, std)
    actions_np = torch.clamp(actions, -1.0, 1.0).detach().cpu().numpy().astype(np.float32)
    return tuple(action.copy() for action in actions_np)


def render_selected_replay(
    policy: torch.nn.Module,
    cfg: PSGAILConfig,
    *,
    episode_name: str,
    vehicle_id: int,
    device: torch.device,
    video_path: str,
    steps: int,
    deterministic: bool,
    screen_width: int,
    screen_height: int,
    scaling: float,
) -> dict[str, Any]:
    video_cfg = replace(
        cfg,
        checkpoint_video_width=int(screen_width),
        checkpoint_video_height=int(screen_height),
        checkpoint_video_scaling=float(scaling),
    )
    env = make_selected_replay_env(
        video_cfg,
        episode_name=episode_name,
        vehicle_id=int(vehicle_id),
        render_mode="rgb_array",
    )
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    frames: list[np.ndarray] = []
    lengths = 0
    crash = False
    offroad = False
    policy.eval()
    try:
        obs, _info = env.reset(seed=int(cfg.seed))
        base = env.unwrapped
        controlled_ids = [int(getattr(vehicle, "vehicle_ID", -1)) for vehicle in base.controlled_vehicles]
        first_frame = env.render()
        if first_frame is not None:
            frames.append(np.asarray(first_frame, dtype=np.uint8))
        for _step in range(max(1, int(steps))):
            action = policy_action_tuple(
                policy,
                obs,
                device=device,
                deterministic=bool(deterministic),
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            crash = bool(crash or any(bool(flag) for flag in info.get("controlled_vehicle_crashes", [])))
            offroad = bool(offroad or any(bool(flag) for flag in info.get("controlled_vehicle_offroad", [])))
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))
            lengths += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    if not frames:
        raise RuntimeError("Renderer produced no frames.")
    with imageio.get_writer(video_path, fps=max(1, int(cfg.policy_frequency))) as writer:
        for frame in frames:
            writer.append_data(frame)
    return {
        "video_path": os.path.abspath(video_path),
        "episode_name": episode_name,
        "requested_vehicle_id": int(vehicle_id),
        "controlled_vehicle_ids": controlled_ids,
        "steps": int(lengths),
        "crash": bool(crash),
        "offroad": bool(offroad),
        "frames": int(len(frames)),
    }


def save_checkpoint(
    path: str,
    policy: torch.nn.Module,
    cfg: PSGAILConfig,
    *,
    obs_dim: int,
    action_dim: int,
    stats: dict[str, float],
    scenario: tuple[str | None, int | None],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": vars(cfg),
            "policy_architecture": {
                "policy_model": cfg.policy_model,
                "obs_dim": int(obs_dim),
                "hidden_size": int(cfg.hidden_size),
                "action_mode": "continuous",
                "continuous_action_dim": int(action_dim),
                "transformer_layers": int(cfg.transformer_layers),
                "transformer_heads": int(cfg.transformer_heads),
                "transformer_dropout": float(cfg.transformer_dropout),
            },
            "bc_stats": stats,
            "default_video_scenario": {
                "episode_name": scenario[0],
                "vehicle_id": scenario[1],
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = cfg_from_args(args)
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    device = resolve_device(str(args.device))

    default_episode, default_vehicle = default_scenario_from_expert_folder(str(args.expert_data))
    episode_name = str(args.episode_name or default_episode or "")
    vehicle_id = int(args.vehicle_id if int(args.vehicle_id) >= 0 else (default_vehicle if default_vehicle is not None else -1))
    if not episode_name or vehicle_id < 0:
        raise RuntimeError(
            "Could not infer a replay scenario. Pass --episode-name and --vehicle-id explicitly."
        )

    arch_env = make_selected_replay_env(
        cfg,
        episode_name=episode_name,
        vehicle_id=vehicle_id,
        render_mode=None,
    )
    try:
        policy, obs_dim, action_dim = build_policy_for_env(cfg, arch_env, device)
    finally:
        arch_env.close()
    cfg.continuous_action_dim = int(action_dim)

    checkpoint_in = str(args.checkpoint_in or "")
    checkpoint_out = os.path.abspath(str(args.checkpoint_out))
    existing_checkpoint = checkpoint_in or (checkpoint_out if os.path.exists(checkpoint_out) else "")
    should_train = bool(args.force_train or (not args.skip_train and not existing_checkpoint))
    if existing_checkpoint and not should_train:
        load_policy_checkpoint(policy, existing_checkpoint, device)
        print(f"Loaded BC checkpoint: {os.path.abspath(existing_checkpoint)}")
        bc_stats: dict[str, float] = {}
    else:
        transitions = load_expert_transition_data(
            cfg.expert_data,
            max_samples=int(cfg.max_expert_samples),
            seed=int(cfg.seed),
            trajectory_frame=str(cfg.trajectory_frame),
        )
        if int(transitions.policy_observations.shape[1]) != int(obs_dim):
            raise RuntimeError(
                "Expert/env policy observation dimensions differ: "
                f"{transitions.policy_observations.shape[1]} != {obs_dim}."
            )
        if int(transitions.actions_continuous_env.shape[1]) != int(action_dim):
            raise RuntimeError(
                "Expert/env continuous action dimensions differ: "
                f"{transitions.actions_continuous_env.shape[1]} != {action_dim}."
            )
        print(
            "Policy architecture: "
            f"model={cfg.policy_model} obs_dim={obs_dim} hidden={cfg.hidden_size} "
            f"action_dim={action_dim} device={device}"
        )
        print(
            "BC data: "
            f"samples={len(transitions.policy_observations)} "
            f"actions={transitions.actions_continuous_env.shape}"
        )
        bc_stats = behavior_clone_pretrain(policy, transitions, cfg, device)
        save_checkpoint(
            checkpoint_out,
            policy,
            cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
            stats=bc_stats,
            scenario=(episode_name, vehicle_id),
        )
        print(f"Saved BC checkpoint: {checkpoint_out}")

    video_stats = render_selected_replay(
        policy,
        cfg,
        episode_name=episode_name,
        vehicle_id=vehicle_id,
        device=device,
        video_path=str(args.video_out),
        steps=int(args.video_steps),
        deterministic=bool(args.deterministic),
        screen_width=int(args.screen_width),
        screen_height=int(args.screen_height),
        scaling=float(args.scaling),
    )
    print(
        "Replay video: "
        f"path={video_stats['video_path']} "
        f"episode={video_stats['episode_name']} "
        f"vehicle={video_stats['requested_vehicle_id']} "
        f"controlled={video_stats['controlled_vehicle_ids']} "
        f"steps={video_stats['steps']} "
        f"crash={video_stats['crash']} offroad={video_stats['offroad']}"
    )
    if bc_stats:
        print(
            "BC final: "
            f"train_mse={bc_stats['bc/train_mse']:.6f} "
            f"train_mae={bc_stats['bc/train_mae']:.6f} "
            f"val_mse={bc_stats['bc/val_mse']:.6f} "
            f"val_mae={bc_stats['bc/val_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
