#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.envs import observation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark exact all-vehicle NGSIM LidarCamera env stepping.",
    )
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--output", default="")
    parser.add_argument("--obs-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compare-determinism", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compare-legacy-lidar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compare-on-road-many", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-info", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def make_env(args: argparse.Namespace):
    import gymnasium as gym

    cfg = PSGAILConfig(
        action_mode="continuous",
        scene=str(args.scene),
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.split),
        max_surrounding=str(args.max_surrounding),
        control_all_vehicles=True,
        percentage_controlled_vehicles=1.0,
        max_episode_steps=int(args.steps),
    )
    register_ngsim_env()
    env_cfg = build_env_config(
        scene=cfg.scene,
        action_mode="continuous",
        episode_root=cfg.episode_root,
        prebuilt_split=cfg.prebuilt_split,
        percentage_controlled_vehicles=1.0,
        control_all_vehicles=True,
        max_surrounding=cfg.max_surrounding,
        observation_config=observation_config(cfg),
        simulation_frequency=cfg.simulation_frequency,
        policy_frequency=cfg.policy_frequency,
        max_episode_steps=int(args.steps),
        seed=None,
        scene_dataset_collection_mode=False,
        allow_idm=cfg.allow_idm,
    )
    env_cfg["expert_test_mode"] = False
    env_cfg["disable_controlled_vehicle_collisions"] = not bool(cfg.enable_collision)
    env_cfg["terminate_when_all_controlled_crashed"] = bool(cfg.terminate_when_all_controlled_crashed)
    env_cfg["allow_idm"] = bool(cfg.allow_idm)
    env_cfg["crash_controlled_vehicles_offroad"] = True
    return gym.make(ENV_ID, config=env_cfg)


def summarize_info(info: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    keys = (
        "controlled_vehicle_crashes",
        "controlled_vehicle_completed",
        "controlled_vehicle_on_road",
        "controlled_vehicle_offroad",
        "alive_controlled_vehicle_ids",
    )
    if full:
        return {key: info.get(key) for key in keys if key in info}

    summary: dict[str, Any] = {}
    for key in keys:
        if key not in info:
            continue
        value = info.get(key)
        if isinstance(value, (list, tuple)):
            numeric = [item for item in value if isinstance(item, (bool, int, float, np.number))]
            summary[f"{key}_count"] = len(value)
            if len(numeric) == len(value):
                summary[f"{key}_sum"] = float(np.sum(numeric)) if value else 0.0
        else:
            summary[key] = value
    return summary


def observations_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(observations_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def observation_diff_summary(left: Any, right: Any) -> dict[str, Any]:
    summary = {
        "arrays_compared": 0,
        "array_mismatches": 0,
        "max_abs_diff": 0.0,
        "structure_mismatch": False,
    }

    def visit(a: Any, b: Any) -> None:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or a.shape != b.shape:
                summary["structure_mismatch"] = True
                summary["array_mismatches"] += 1
                return
            summary["arrays_compared"] += 1
            if not np.array_equal(a, b):
                summary["array_mismatches"] += 1
                if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                    diff = np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                    summary["max_abs_diff"] = max(float(summary["max_abs_diff"]), float(diff))
            return
        if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
            if not isinstance(a, type(b)) or len(a) != len(b):
                summary["structure_mismatch"] = True
                return
            for item_a, item_b in zip(a, b):
                visit(item_a, item_b)
            return
        if a != b:
            summary["structure_mismatch"] = True

    visit(left, right)
    summary["observations_equal"] = (
        not bool(summary["structure_mismatch"]) and int(summary["array_mismatches"]) == 0
    )
    return summary


def run_zero_action_rollout(args: argparse.Namespace, *, full_info: bool = True) -> dict[str, Any]:
    env = make_env(args)
    obs, _info = env.reset(seed=int(args.seed))
    controlled = list(getattr(env.unwrapped, "controlled_vehicles", []) or [])
    action = tuple(np.zeros((2,), dtype=np.float32) for _ in controlled)
    observations = [obs]
    rewards = []
    terminated_flags = []
    truncated_flags = []
    infos = []
    try:
        for _step in range(max(0, int(args.steps))):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            infos.append(summarize_info(info, full=full_info))
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "observations": observations,
        "rewards": rewards,
        "terminated": terminated_flags,
        "truncated": truncated_flags,
        "infos": infos,
    }


def compare_legacy_lidar(args: argparse.Namespace) -> dict[str, Any]:
    from highway_env import utils
    from highway_env.envs.common import observation as observation_module

    env = make_env(args)
    original = observation_module.LidarObservation._distance_to_rect_precomputed

    def legacy_distance_to_rect(*, origin, direction, max_t, entry):
        q = origin + float(max_t) * direction
        return utils.distance_to_rect((origin, q), entry.corners)

    try:
        obs_optimized, _info = env.reset(seed=int(args.seed))
        obs_type = getattr(env.unwrapped, "observation_type", None)
        observation_module.LidarObservation._distance_to_rect_precomputed = staticmethod(legacy_distance_to_rect)
        obs_legacy = obs_type.observe()
        reset_compare = observation_diff_summary(obs_optimized, obs_legacy)

        controlled = list(getattr(env.unwrapped, "controlled_vehicles", []) or [])
        action = tuple(np.zeros((2,), dtype=np.float32) for _ in controlled)
        observation_module.LidarObservation._distance_to_rect_precomputed = staticmethod(original)
        obs_optimized_step, reward_opt, terminated_opt, truncated_opt, info_opt = env.step(action)

        observation_module.LidarObservation._distance_to_rect_precomputed = staticmethod(legacy_distance_to_rect)
        obs_legacy_step = obs_type.observe()
        step_compare = observation_diff_summary(obs_optimized_step, obs_legacy_step)

        return {
            "reset": reset_compare,
            "after_step": step_compare,
            "reward_type": type(reward_opt).__name__,
            "terminated": bool(terminated_opt),
            "truncated": bool(truncated_opt),
            "info": summarize_info(info_opt, full=False),
        }
    finally:
        observation_module.LidarObservation._distance_to_rect_precomputed = staticmethod(original)
        env.close()


def compare_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    first = run_zero_action_rollout(args, full_info=True)
    second = run_zero_action_rollout(args, full_info=True)
    return {
        "steps_first": len(first["rewards"]),
        "steps_second": len(second["rewards"]),
        "observations_equal": observations_equal(first["observations"], second["observations"]),
        "rewards_equal": first["rewards"] == second["rewards"],
        "terminated_equal": first["terminated"] == second["terminated"],
        "truncated_equal": first["truncated"] == second["truncated"],
        "infos_equal": first["infos"] == second["infos"],
    }


def main() -> None:
    args = parse_args()
    if args.compare_on_road_many:
        os.environ["HIGHWAY_ENV_ON_ROAD_MANY_COMPARE"] = "1"
    if args.obs_profile:
        os.environ["HIGHWAY_ENV_OBS_PROFILE"] = "1"
        os.environ.setdefault("HIGHWAY_ENV_OBS_PROFILE_EVERY", "1")
        from highway_env.envs.common.observation import _ObservationProfiler

        _ObservationProfiler.enabled = True
        _ObservationProfiler.report_every = max(1, int(os.environ["HIGHWAY_ENV_OBS_PROFILE_EVERY"]))
        _ObservationProfiler.reset()

    if args.compare_determinism:
        result = compare_rollouts(args)
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
        return

    if args.compare_legacy_lidar:
        result = compare_legacy_lidar(args)
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
        return

    started = time.perf_counter()
    env = make_env(args)
    make_seconds = time.perf_counter() - started
    reset_started = time.perf_counter()
    obs, info = env.reset(seed=int(args.seed))
    reset_seconds = time.perf_counter() - reset_started

    controlled = list(getattr(env.unwrapped, "controlled_vehicles", []) or [])
    road = getattr(env.unwrapped, "road", None)
    action = tuple(np.zeros((2,), dtype=np.float32) for _ in controlled)

    step_seconds: list[float] = []
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    for _step in range(max(0, int(args.steps))):
        step_started = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_seconds.append(time.perf_counter() - step_started)
        final_info = summarize_info(info, full=bool(args.full_info))
        if terminated or truncated:
            break

    result = {
        "episode": getattr(env.unwrapped, "episode_name", None),
        "controlled_vehicles": len(controlled),
        "road_vehicles": len(getattr(road, "vehicles", []) or []),
        "observation_type": type(getattr(env.unwrapped, "observation_type", None)).__name__,
        "make_seconds": make_seconds,
        "reset_seconds": reset_seconds,
        "steps_completed": len(step_seconds),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "step_total_seconds": float(np.sum(step_seconds)) if step_seconds else 0.0,
        "step_mean_seconds": float(np.mean(step_seconds)) if step_seconds else 0.0,
        "step_p50_seconds": float(np.percentile(step_seconds, 50)) if step_seconds else 0.0,
        "step_p95_seconds": float(np.percentile(step_seconds, 95)) if step_seconds else 0.0,
        "final_info": final_info,
    }
    env.close()

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
