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


def hybrid_observation_config() -> dict[str, Any]:
    return {
        "type": "LidarCameraObservations",
        "lidar": {
            "cells": 64,
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
            "Compare road-query fast path against the legacy scan for "
            "close_objects_to() and neighbour_vehicles() across scenes."
        )
    )
    parser.add_argument("--scenes", nargs="+", default=["us-101", "japanese"])
    parser.add_argument("--percentage-controlled-vehicles", type=float, default=0.1)
    parser.add_argument("--control-all-vehicles", action="store_true", default=False)
    parser.add_argument("--max-surrounding", default=20)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
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
        "percentage_controlled_vehicles": float(args.percentage_controlled_vehicles),
        "control_all_vehicles": bool(args.control_all_vehicles),
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


def object_token(obj: Any) -> tuple[str, int]:
    vehicle_id = getattr(obj, "vehicle_ID", None)
    if vehicle_id is not None:
        return (obj.__class__.__name__, int(vehicle_id))
    return (obj.__class__.__name__, id(obj))


def neighbour_tokens(pair: tuple[Any | None, Any | None]) -> tuple[Any, Any]:
    front, rear = pair
    return (
        None if front is None else object_token(front),
        None if rear is None else object_token(rear),
    )


def benchmark_close_objects(
    road: Any,
    vehicle: Any,
    repeats: int,
    *,
    use_fast: bool,
) -> tuple[list[Any], float]:
    total = 0.0
    out = None
    for _ in range(max(1, repeats)):
        road.use_query_fast_path = use_fast
        t0 = time.perf_counter()
        out = road.close_objects_to(
            vehicle,
            distance=200.0,
            count=12,
            see_behind=True,
            sort=True,
            vehicles_only=False,
        )
        total += time.perf_counter() - t0
    return out or [], total / max(1, repeats)


def benchmark_neighbours(
    road: Any,
    vehicle: Any,
    repeats: int,
    *,
    use_fast: bool,
) -> tuple[tuple[Any | None, Any | None], float]:
    total = 0.0
    out = None
    for _ in range(max(1, repeats)):
        road.use_query_fast_path = use_fast
        t0 = time.perf_counter()
        out = road.neighbour_vehicles(vehicle)
        total += time.perf_counter() - t0
    assert out is not None
    return out, total / max(1, repeats)


def main() -> None:
    args = parse_args()

    total_close_fast = 0.0
    total_close_slow = 0.0
    total_neigh_fast = 0.0
    total_neigh_slow = 0.0
    close_checks = 0
    neigh_checks = 0

    for scene_idx, scene in enumerate(args.scenes):
        env = gym.make(ENV_ID, config=build_config(scene, args))
        try:
            obs, info = env.reset(seed=int(args.seed) + scene_idx)
            del obs, info
            base = env.unwrapped
            road = base.road

            print(f"\n=== scene={scene} controlled={len(base.controlled_vehicles)} road_vehicles={len(road.vehicles)} ===")

            for step in range(int(args.steps)):
                vehicles = list(base.controlled_vehicles)
                if not vehicles:
                    raise RuntimeError(f"No controlled vehicles available for scene={scene}")

                for vehicle in vehicles:
                    close_fast, close_fast_dt = benchmark_close_objects(
                        road, vehicle, int(args.benchmark_repeats), use_fast=True
                    )
                    close_slow, close_slow_dt = benchmark_close_objects(
                        road, vehicle, int(args.benchmark_repeats), use_fast=False
                    )
                    if [object_token(obj) for obj in close_fast] != [object_token(obj) for obj in close_slow]:
                        raise AssertionError(
                            "close_objects_to mismatch: "
                            f"scene={scene} step={step} vehicle_id={getattr(vehicle, 'vehicle_ID', None)}"
                        )
                    total_close_fast += close_fast_dt
                    total_close_slow += close_slow_dt
                    close_checks += 1

                    neigh_fast, neigh_fast_dt = benchmark_neighbours(
                        road, vehicle, int(args.benchmark_repeats), use_fast=True
                    )
                    neigh_slow, neigh_slow_dt = benchmark_neighbours(
                        road, vehicle, int(args.benchmark_repeats), use_fast=False
                    )
                    if neighbour_tokens(neigh_fast) != neighbour_tokens(neigh_slow):
                        raise AssertionError(
                            "neighbour_vehicles mismatch: "
                            f"scene={scene} step={step} vehicle_id={getattr(vehicle, 'vehicle_ID', None)} "
                            f"fast={neighbour_tokens(neigh_fast)} slow={neighbour_tokens(neigh_slow)}"
                        )
                    total_neigh_fast += neigh_fast_dt
                    total_neigh_slow += neigh_slow_dt
                    neigh_checks += 1

                print(
                    f"step={step} "
                    f"close_fast_ms={(total_close_fast / max(1, close_checks)) * 1e3:.3f} "
                    f"close_slow_ms={(total_close_slow / max(1, close_checks)) * 1e3:.3f} "
                    f"neigh_fast_ms={(total_neigh_fast / max(1, neigh_checks)) * 1e3:.3f} "
                    f"neigh_slow_ms={(total_neigh_slow / max(1, neigh_checks)) * 1e3:.3f}"
                )

                obs, _reward, terminated, truncated, _info = env.step(idle_action(env))
                del obs
                if terminated or truncated:
                    break
        finally:
            env.close()

    if close_checks == 0 or neigh_checks == 0:
        raise RuntimeError("No fast-path comparisons were executed.")

    print("\nSummary")
    print(f"close_checks={close_checks}")
    print(f"avg_close_fast_ms={(total_close_fast / close_checks) * 1e3:.3f}")
    print(f"avg_close_slow_ms={(total_close_slow / close_checks) * 1e3:.3f}")
    print(f"close_speedup_x={total_close_slow / total_close_fast if total_close_fast > 0 else float('inf'):.3f}")
    print(f"neigh_checks={neigh_checks}")
    print(f"avg_neigh_fast_ms={(total_neigh_fast / neigh_checks) * 1e3:.3f}")
    print(f"avg_neigh_slow_ms={(total_neigh_slow / neigh_checks) * 1e3:.3f}")
    print(f"neigh_speedup_x={total_neigh_slow / total_neigh_fast if total_neigh_fast > 0 else float('inf'):.3f}")


if __name__ == "__main__":
    main()
