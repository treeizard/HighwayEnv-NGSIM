#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
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


def build_base_config(scene: str, action_mode: str) -> dict[str, Any]:
    action_type = "ContinuousAction" if action_mode == "continuous" else "DiscreteSteerMetaAction"
    return {
        "scene": scene,
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "action": {"type": action_type},
        "action_mode": action_mode,
        "show_trajectories": False,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "offscreen_rendering": True,
        "episode_root": "highway_env/data/processed_20s",
        "prebuilt_split": "train",
        "ego_vehicle_ID": None,
        "simulation_period": None,
        "max_surrounding": 0,
        "expert_test_mode": True,
    }


def build_unique_scenarios(env: gym.Env) -> list[tuple[str, int]]:
    base = env.unwrapped
    scenarios: list[tuple[str, int]] = []

    for ep_name in base._episodes:
        for ego_id in base._valid_ids_by_episode.get(ep_name, []):
            scenarios.append((str(ep_name), int(ego_id)))

    if not scenarios:
        raise RuntimeError("No scenarios available in the selected scene.")

    return scenarios


def run_case(base_cfg: dict[str, Any], episode_name: str, ego_id: int, seed: int, max_steps: int | None) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["simulation_period"] = {"episode_name": episode_name}
    cfg["ego_vehicle_ID"] = int(ego_id)

    env = gym.make(ENV_ID, config=cfg)
    try:
        obs, info = env.reset(seed=seed)
        del obs, info

        steps = 0
        done = False
        while not done:
            if max_steps is not None and steps >= max_steps:
                break

            if base_cfg["action_mode"] == "continuous":
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action = 0

            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            done = bool(terminated or truncated)

        metrics = env.unwrapped.expert_replay_metrics()
        return {
            "episode_name": episode_name,
            "ego_id": int(ego_id),
            "steps": int(steps),
            "T": int(metrics["T"]),
            "ADE_m": float(metrics["ADE_m"]),
            "FDE_m": float(metrics["FDE_m"]),
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple expert-replay trajectory cases without surrounding vehicles."
    )
    parser.add_argument("--scene", type=str, default="us-101", help="Scene name.")
    parser.add_argument(
        "--action-mode",
        type=str,
        choices=["discrete", "continuous"],
        default="discrete",
        help="Expert replay control mode.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=5,
        help="Number of unique scenarios to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max rollout steps per case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for scenario shuffling and env reset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = build_base_config(scene=args.scene, action_mode=args.action_mode)

    probe_env = gym.make(ENV_ID, config=base_cfg)
    try:
        scenarios = build_unique_scenarios(probe_env)
    finally:
        probe_env.close()

    rng = np.random.default_rng(args.seed)
    rng.shuffle(scenarios)
    selected = scenarios[: args.cases]

    if len(selected) < args.cases:
        raise ValueError(
            f"Requested {args.cases} cases but only found {len(selected)} scenarios."
        )

    results: list[dict[str, Any]] = []
    for idx, (episode_name, ego_id) in enumerate(selected, start=1):
        result = run_case(
            base_cfg=base_cfg,
            episode_name=episode_name,
            ego_id=ego_id,
            seed=args.seed + idx - 1,
            max_steps=args.max_steps,
        )
        results.append(result)
        print(
            f"[{idx}/{len(selected)}] "
            f"episode={result['episode_name']} ego_id={result['ego_id']} "
            f"steps={result['steps']} T={result['T']} "
            f"ADE={result['ADE_m']:.4f}m FDE={result['FDE_m']:.4f}m"
        )

    ade = np.array([row["ADE_m"] for row in results], dtype=float)
    fde = np.array([row["FDE_m"] for row in results], dtype=float)
    steps = np.array([row["steps"] for row in results], dtype=float)

    print("\nSummary")
    print(f"scene={args.scene} action_mode={args.action_mode} cases={len(results)}")
    print(f"avg_steps={np.mean(steps):.2f}")
    print(f"avg_ADE_m={np.mean(ade):.6f}")
    print(f"avg_FDE_m={np.mean(fde):.6f}")
    print(f"max_ADE_m={np.max(ade):.6f}")
    print(f"max_FDE_m={np.max(fde):.6f}")


if __name__ == "__main__":
    main()
