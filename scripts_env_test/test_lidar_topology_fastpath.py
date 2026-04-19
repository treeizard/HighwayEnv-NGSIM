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


from highway_env.envs.common.observation import LidarObservation  # noqa: E402


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
        description=(
            "Compare the topology-aware LiDAR road-edge fast path against the "
            "legacy all-lanes fallback for both us-101 and japanese scenes."
        )
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["us-101", "japanese"],
        help="Scenes to test.",
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=3,
        help="Requested number of controlled vehicles.",
    )
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        default=True,
        help="Use all valid controlled vehicles and cap them with --max-controlled-vehicles.",
    )
    parser.add_argument(
        "--max-controlled-vehicles",
        type=int,
        default=3,
        help="Cap when --control-all-vehicles is enabled.",
    )
    parser.add_argument(
        "--max-surrounding",
        default=0,
        help="Set to 0 to isolate road-edge behaviour from replay traffic.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of rollout steps to compare.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        help="How many repeated observe() calls to time per vehicle/step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base reset seed.",
    )
    return parser.parse_args()


def build_config(scene: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scene": scene,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": hybrid_observation_config(),
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteSteerMetaAction"},
        },
        "action_mode": "discrete",
        "controlled_vehicles": int(args.controlled_vehicles),
        "control_all_vehicles": bool(args.control_all_vehicles),
        "max_controlled_vehicles": int(args.max_controlled_vehicles),
        "show_trajectories": False,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "max_episode_steps": max(10, int(args.steps) + 2),
        "episode_root": "highway_env/data/processed_20s",
        "prebuilt_split": "train",
        "max_surrounding": args.max_surrounding,
        "expert_test_mode": False,
        "offscreen_rendering": True,
    }


def idle_action(env: gym.Env) -> tuple[int, ...]:
    return tuple(1 for _ in env.unwrapped.controlled_vehicles)


def benchmark_observer(
    observer: LidarObservation,
    vehicle: Any,
    repeats: int,
) -> tuple[np.ndarray, float]:
    observer.observer_vehicle = vehicle
    total = 0.0
    out = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = observer.observe()
        total += time.perf_counter() - t0
    assert out is not None
    return out, total / max(1, repeats)


def main() -> None:
    args = parse_args()

    overall_fast = 0.0
    overall_slow = 0.0
    comparisons = 0

    for scene_idx, scene in enumerate(args.scenes):
        cfg = build_config(scene, args)
        env = gym.make(ENV_ID, config=cfg)
        try:
            obs, info = env.reset(seed=int(args.seed) + scene_idx)
            del obs, info
            base = env.unwrapped

            fast_observer = LidarObservation(
                base,
                cells=128,
                maximum_range=64.0,
                normalize=True,
                use_topology_fast_path=True,
            )
            slow_observer = LidarObservation(
                base,
                cells=128,
                maximum_range=64.0,
                normalize=True,
                use_topology_fast_path=False,
            )

            print(f"\n=== scene={scene} controlled={len(base.controlled_vehicles)} ===")

            for step in range(int(args.steps)):
                vehicles = list(base.controlled_vehicles)
                if not vehicles:
                    raise RuntimeError(f"No controlled vehicles available for scene={scene}.")

                for vehicle in vehicles:
                    fast_obs, fast_dt = benchmark_observer(
                        fast_observer,
                        vehicle,
                        repeats=int(args.benchmark_repeats),
                    )
                    slow_obs, slow_dt = benchmark_observer(
                        slow_observer,
                        vehicle,
                        repeats=int(args.benchmark_repeats),
                    )
                    max_abs = float(np.max(np.abs(fast_obs - slow_obs)))
                    if not np.allclose(fast_obs, slow_obs, atol=1e-5, rtol=1e-5):
                        raise AssertionError(
                            "Topology fast path changed LiDAR output: "
                            f"scene={scene} step={step} vehicle_id={getattr(vehicle, 'vehicle_ID', None)} "
                            f"max_abs_diff={max_abs:.8f}"
                        )

                    overall_fast += fast_dt
                    overall_slow += slow_dt
                    comparisons += 1

                print(
                    f"step={step} vehicles={len(vehicles)} "
                    f"avg_fast_ms={(overall_fast / comparisons) * 1e3:.3f} "
                    f"avg_slow_ms={(overall_slow / comparisons) * 1e3:.3f}"
                )

                obs, _reward, terminated, truncated, _info = env.step(idle_action(env))
                del obs
                if terminated or truncated:
                    break

        finally:
            env.close()

    if comparisons == 0:
        raise RuntimeError("No LiDAR comparisons were executed.")

    speedup = overall_slow / overall_fast if overall_fast > 0.0 else float("inf")
    print("\nSummary")
    print(f"comparisons={comparisons}")
    print(f"avg_fast_ms={(overall_fast / comparisons) * 1e3:.3f}")
    print(f"avg_slow_ms={(overall_slow / comparisons) * 1e3:.3f}")
    print(f"speedup_x={speedup:.3f}")


if __name__ == "__main__":
    main()
