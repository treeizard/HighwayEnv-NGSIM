#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

ENV_ID = "NGSim-US101-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize multiple controlled vehicles in NGSimEnv."
    )
    parser.add_argument("--scene", type=str, default="us-101", help="Scene name.")
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=3,
        help="Number of controlled ego vehicles to spawn.",
    )
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        help="Control all valid vehicles in the selected traffic segment.",
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
        help="Optional explicit ego vehicle IDs. Must match --controlled-vehicles in length.",
    )
    parser.add_argument(
        "--max-surrounding",
        default='all',
        help="Surrounding vehicles to spawn. Use 'all' for every replay vehicle.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Maximum rollout steps requested by the script loop.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=250,
        help="Environment horizon used for truncation.",
    )
    parser.add_argument(
        "--terminate-when-all-controlled-crashed",
        action="store_true",
        default=True,
        help="End the episode only when all controlled vehicles are crashed.",
    )
    parser.add_argument(
        "--terminate-when-any-controlled-crashed",
        action="store_true",
        help="Override and end the episode when any controlled vehicle crashes.",
    )
    parser.add_argument(
        "--truncate-to-trajectory-length",
        action="store_true",
        help="Also truncate at the shortest ego trajectory length.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="How often to print rollout diagnostics.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Action interface used for the primary controlled vehicle.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array"],
        default="human",
        help="Use 'human' for on-screen rendering or 'rgb_array' for off-screen stepping.",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1400,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=700,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=5.5,
        help="Render zoom.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reset seed.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    action_type = (
        "ContinuousAction"
        if args.action_mode == "continuous"
        else "DiscreteSteerMetaAction"
    )
    simulation_period = (
        {"episode_name": args.episode_name}
        if args.episode_name is not None
        else None
    )
    return {
        "scene": args.scene,
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "action": {"type": action_type},
        "action_mode": args.action_mode,
        "controlled_vehicles": int(args.controlled_vehicles),
        "control_all_vehicles": bool(args.control_all_vehicles),
        "ego_vehicle_ID": args.ego_ids,
        "simulation_period": simulation_period,
        "show_trajectories": True,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "max_episode_steps": int(args.max_episode_steps),
        "terminate_when_all_controlled_crashed": (
            not args.terminate_when_any_controlled_crashed
        ),
        "truncate_to_trajectory_length": bool(args.truncate_to_trajectory_length),
        "screen_width": int(args.screen_width),
        "screen_height": int(args.screen_height),
        "scaling": float(args.scaling),
        "offscreen_rendering": args.render_mode == "rgb_array",
        "episode_root": "highway_env/data/processed_20s",
        "prebuilt_split": "train",
        "max_surrounding": 0 if args.control_all_vehicles else args.max_surrounding,
        "expert_test_mode": False,
    }


def idle_action(env: gym.Env) -> int | np.ndarray:
    if hasattr(env.action_space, "n"):
        action_type = getattr(env.unwrapped, "action_type", None)
        if action_type is not None and hasattr(action_type, "actions_indexes"):
            return int(action_type.actions_indexes.get("IDLE", 0))
        return 0
    return np.zeros(env.action_space.shape, dtype=np.float32)


def colorize_controlled_vehicles(env: gym.Env) -> None:
    palette = [
        (230, 60, 60),
        (40, 170, 255),
        (255, 170, 40),
        (60, 210, 120),
        (210, 90, 230),
        (255, 255, 70),
    ]
    for idx, vehicle in enumerate(env.unwrapped.controlled_vehicles):
        vehicle.color = palette[idx % len(palette)]


def update_controlled_vehicle_colors(env: gym.Env) -> None:
    crashed_color = (120, 120, 120)
    for vehicle in env.unwrapped.controlled_vehicles:
        if bool(getattr(vehicle, "crashed", False)):
            vehicle.color = crashed_color


def print_episode_summary(env: gym.Env) -> None:
    base = env.unwrapped
    print(f"episode_name={getattr(base, 'episode_name', None)}")
    print(f"ego_ids={getattr(base, 'ego_ids', None)}")
    print(f"controlled={len(base.controlled_vehicles)} total_road_vehicles={len(base.road.vehicles)}")
    print(f"ego_start_indices={getattr(base, '_ego_start_indices', None)}")
    print(
        "termination_config="
        f"all_crashed={base.config.get('terminate_when_all_controlled_crashed', True)} "
        f"truncate_to_traj={base.config.get('truncate_to_trajectory_length', False)} "
        f"max_episode_steps={base.config.get('max_episode_steps', None)}"
    )

    for idx, vehicle in enumerate(base.controlled_vehicles):
        print(
            f"  ego[{idx}] id={getattr(vehicle, 'vehicle_ID', None)} "
            f"pos=({vehicle.position[0]:.2f}, {vehicle.position[1]:.2f}) "
            f"speed={vehicle.speed:.2f}"
        )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    env = gym.make(ENV_ID, render_mode=args.render_mode, config=cfg)

    try:
        obs, info = env.reset(seed=args.seed)
        del obs, info
        colorize_controlled_vehicles(env)
        print_episode_summary(env)
        print("Rendering multi-ego rollout. Close the window or press Ctrl+C to stop.")

        for step in range(args.steps):
            action = idle_action(env)
            _, _, terminated, truncated, info = env.step(action)
            update_controlled_vehicle_colors(env)

            if args.render_mode == "human":
                env.render()
                time.sleep(1.0 / max(1.0, env.unwrapped.config["policy_frequency"]))

            if step % max(1, args.log_interval) == 0:
                crash_flags = info.get("controlled_vehicle_crashes", None)
                all_speeds = info.get("all_controlled_vehicle_speeds", info.get("speed", []))
                alive_ids = info.get("alive_controlled_vehicle_ids", [])
                print(
                    f"step={step:03d} "
                    f"alive={len(alive_ids)}/{len(env.unwrapped.controlled_vehicles)} "
                    f"policy_done={info.get('crashed', None)} "
                    f"speeds={np.round(np.asarray(all_speeds, dtype=float), 2).tolist()} "
                    f"crashes={crash_flags}"
                )

            if terminated or truncated:
                print(
                    f"episode ended at step={step} "
                    f"terminated={terminated} truncated={truncated}"
                )
                print(
                    f"alive_controlled_vehicle_ids={info.get('alive_controlled_vehicle_ids', None)}"
                )
                print(
                    f"controlled_vehicle_crashes={info.get('controlled_vehicle_crashes', None)}"
                )
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
