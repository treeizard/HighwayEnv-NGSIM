#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import RecordVideo

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle

ENV_ID = "NGSim-US101-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def hybrid_observation_config() -> dict[str, Any]:
    return {
        "type": "LidarCameraObservations",
        "lidar": {
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "camera": {
            "cells": 21,
            "maximum_range": 64,
            "field_of_view": np.pi / 2,
            "normalize": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize discrete expert replay in NGSimEnv, including multi-expert cases."
    )
    parser.add_argument("--scene", type=str, default="us-101", help="Scene name.")
    parser.add_argument(
        "--episode-root",
        type=str,
        default="highway_env/data/processed_20s",
        help="Dataset root containing <scene>/prebuilt/*.npy. The bundled default is a 20-second dataset.",
    )
    parser.add_argument(
        "--prebuilt-split",
        choices=["train", "val"],
        default="train",
        help="Which prebuilt split to sample replay scenarios from.",
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=1,
        help="Number of expert-controlled vehicles to replay.",
    )
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        help="Replay all valid vehicles in the selected traffic segment as controlled experts.",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="Optional fixed episode name.",
    )
    parser.add_argument(
        "--ego-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit ego ids. Length must match --controlled-vehicles.",
    )
    parser.add_argument(
        "--max-surrounding",
        default="all",
        help="Surrounding vehicles to spawn. Use 'all' for full replay context.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Maximum rollout steps per episode.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="rgb_array",
        help="Use 'human' for on-screen rendering or 'rgb_array' for off-screen video export.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record rollout video when moviepy is available.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos_discrete_test",
        help="Directory used when --record-video is enabled.",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1600,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=600,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=3.0,
        help="Render zoom.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base reset seed.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to print rollout diagnostics.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    observation_cfg: dict[str, Any]
    action_cfg: dict[str, Any]

    if args.controlled_vehicles > 1:
        observation_cfg = {
            "type": "MultiAgentObservation",
            "observation_config": hybrid_observation_config(),
        }
        action_cfg = {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteSteerMetaAction"},
        }
    else:
        observation_cfg = hybrid_observation_config()
        action_cfg = {"type": "DiscreteSteerMetaAction"}

    if args.control_all_vehicles:
        observation_cfg = {
            "type": "MultiAgentObservation",
            "observation_config": hybrid_observation_config(),
        }
        action_cfg = {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteSteerMetaAction"},
        }

    simulation_period = (
        {"episode_name": args.episode_name}
        if args.episode_name is not None
        else None
    )

    return {
        "scene": args.scene,
        "observation": observation_cfg,
        "action": action_cfg,
        "show_trajectories": False,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": int(args.screen_width),
        "screen_height": int(args.screen_height),
        "scaling": float(args.scaling),
        "offscreen_rendering": args.render_mode == "rgb_array",
        "episode_root": args.episode_root,
        "prebuilt_split": args.prebuilt_split,
        "simulation_period": simulation_period,
        "ego_vehicle_ID": args.ego_ids,
        "controlled_vehicles": int(args.controlled_vehicles),
        "control_all_vehicles": bool(args.control_all_vehicles),
        "max_surrounding": args.max_surrounding,
        "expert_test_mode": True,
        "discrete_expert_policy": "tracker_map",
        "action_mode": "discrete",
        "expert_prefer_speed": False,
        "disable_controlled_vehicle_collisions": True,
    }


def dummy_action(env: gym.Env) -> Any:
    if env.unwrapped.config.get("action", {}).get("type") == "MultiAgentAction":
        return tuple(0 for _ in env.unwrapped.controlled_vehicles)
    return 0


def colorize_controlled_vehicles(env: gym.Env) -> None:
    shared_color = (255, 255, 0)
    for vehicle in env.unwrapped.controlled_vehicles:
        vehicle.color = shared_color


def update_controlled_vehicle_colors(env: gym.Env) -> None:
    crashed_color = (120, 120, 120)
    for vehicle in env.unwrapped.controlled_vehicles:
        if bool(getattr(vehicle, "crashed", False)):
            vehicle.color = crashed_color


def print_episode_header(env: gym.Env, episode_idx: int) -> None:
    base = env.unwrapped
    print(f"\n=== Episode {episode_idx} ===")
    print(f"episode_name={getattr(base, 'episode_name', None)}")
    print(f"ego_ids={getattr(base, 'ego_ids', None)}")
    print(f"controlled={len(base.controlled_vehicles)} total_road_vehicles={len(base.road.vehicles)}")


def print_surrounding_replay_diagnostics(env: gym.Env) -> None:
    base = env.unwrapped
    obs_vehicles = [
        vehicle
        for vehicle in base.road.vehicles
        if isinstance(vehicle, NGSIMVehicle)
    ]
    if not obs_vehicles:
        print("surrounding replay vehicles: 0")
        return

    lengths = np.array([len(vehicle.ngsim_traj) for vehicle in obs_vehicles], dtype=int)
    print(
        "surrounding replay lengths "
        f"count={len(obs_vehicles)} "
        f"min={int(lengths.min())} "
        f"median={int(np.median(lengths))} "
        f"max={int(lengths.max())} steps"
    )
    print(
        "surrounding replay lengths seconds "
        f"min={lengths.min() / 10.0:.1f} "
        f"median={np.median(lengths) / 10.0:.1f} "
        f"max={lengths.max() / 10.0:.1f}"
    )
    for threshold in (10, 20, 30, 40, 50):
        print(
            f"surrounding replay vehicles with >={threshold} steps: "
            f"{int(np.sum(lengths >= threshold))}"
        )


def print_controlled_replay_diagnostics(env: gym.Env) -> None:
    base = env.unwrapped
    ego_ids = getattr(base, "ego_ids", [])
    if not ego_ids:
        return

    ref_lengths = {
        int(ego_id): len(base._expert_state_by_vehicle_id[int(ego_id)]["ref_xy"])
        for ego_id in ego_ids
    }
    min_ref_length = min(ref_lengths.values())
    print(f"expert ref lengths by ego_id: {ref_lengths}")
    print(f"shortest controlled expert reference: {min_ref_length} steps")

    max_episode_steps = int(base.config.get("max_episode_steps", 0))
    if max_episode_steps and min_ref_length < max_episode_steps:
        print(
            "warning: loaded expert references are shorter than max_episode_steps "
            f"({min_ref_length} < {max_episode_steps}). "
            "If you expected ~300 steps, check --episode-root."
        )


def print_episode_metrics(env: gym.Env) -> None:
    base = env.unwrapped
    ego_ids = getattr(base, "ego_ids", [])
    for ego_id in ego_ids:
        try:
            metrics = base.expert_replay_metrics(ego_id=ego_id)
        except Exception:
            continue
        print(
            f"metrics ego_id={ego_id} "
            f"T={metrics['T']} ADE_m={metrics['ADE_m']:.4f} FDE_m={metrics['FDE_m']:.4f}"
        )


def run_one_episode(
    env: gym.Env,
    episode_idx: int,
    seed: int,
    max_steps: int,
    log_interval: int,
    render_mode: str,
) -> None:
    obs, info = env.reset(seed=seed)
    del obs, info

    colorize_controlled_vehicles(env)
    print_episode_header(env, episode_idx)
    print(f"episode_root={env.unwrapped.config.get('episode_root')}")
    print_controlled_replay_diagnostics(env)
    print_surrounding_replay_diagnostics(env)

    action_counts: dict[int, dict[str, int]] = {}

    for step in range(max_steps):
        _, _, terminated, truncated, info = env.step(dummy_action(env))
        update_controlled_vehicle_colors(env)

        action_labels = info.get("expert_action_discrete_all")
        vehicle_ids = info.get("expert_controlled_vehicle_ids", [])
        if action_labels is None and "expert_action_discrete" in info:
            action_labels = [info["expert_action_discrete"]]
        for vehicle_id, label in zip(vehicle_ids, action_labels or []):
            vehicle_id = int(vehicle_id)
            label_counts = action_counts.setdefault(vehicle_id, {})
            label_str = str(label)
            label_counts[label_str] = label_counts.get(label_str, 0) + 1

        if render_mode == "human":
            env.render()
            time.sleep(1.0 / max(1.0, env.unwrapped.config["policy_frequency"]))

        if step % max(1, log_interval) == 0:
            print(
                f"step={step:03d} "
                f"alive={info.get('alive_controlled_vehicle_ids', [])} "
                f"speeds={np.round(np.asarray(info.get('all_controlled_vehicle_speeds', []), dtype=float), 2).tolist()} "
                f"expert_actions={info.get('expert_action_discrete_all', info.get('expert_action_discrete', None))}"
            )

        if terminated or truncated:
            print(
                f"episode ended at step={step} "
                f"terminated={terminated} truncated={truncated}"
            )
            break

    print_episode_metrics(env)
    for vehicle_id in sorted(action_counts):
        print(f"action counts ego_id={vehicle_id}: {action_counts[vehicle_id]}")


def maybe_wrap_record_video(env: gym.Env, args: argparse.Namespace) -> tuple[gym.Env, bool]:
    if not args.record_video:
        return env, False

    out_dir = os.path.abspath(args.video_dir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        import moviepy  # noqa: F401

        wrapped = RecordVideo(
            env,
            video_folder=out_dir,
            episode_trigger=lambda ep_idx: True,
            name_prefix="ngsim_expert_discrete",
        )
        return wrapped, True
    except ModuleNotFoundError:
        warnings.warn(
            "moviepy is not installed; continuing without RecordVideo.",
            stacklevel=2,
        )
        return env, False


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    env = gym.make(ENV_ID, render_mode=args.render_mode, config=cfg)
    env, record_video = maybe_wrap_record_video(env, args)

    try:
        for ep in range(args.episodes):
            run_one_episode(
                env=env,
                episode_idx=ep,
                seed=args.seed + ep,
                max_steps=int(args.max_steps),
                log_interval=int(args.log_interval),
                render_mode=args.render_mode,
            )
    finally:
        env.close()

    if record_video:
        print(f"\nSaved videos to: {os.path.abspath(args.video_dir)}")
    elif args.record_video:
        print("\nVideo recording skipped because moviepy is not installed.")
    print(
        f"Configured render resolution: {args.screen_width}x{args.screen_height} "
        f"controlled_vehicles={args.controlled_vehicles}"
    )


if __name__ == "__main__":
    main()
