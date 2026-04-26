#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env  # noqa: E402
from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track NGSIM replay vehicles that hand over to IDM/MOBIL and report "
            "handover, crash, and front-gap diagnostics for a chosen episode."
        )
    )
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episode-name", required=True)
    parser.add_argument("--ego-vehicle-id", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--controlled-vehicles", type=int, default=4)
    parser.add_argument("--max-controlled-vehicles", type=int, default=0)
    parser.add_argument("--control-all-vehicles", action="store_true", default=True)
    parser.add_argument("--no-control-all-vehicles", dest="control_all_vehicles", action="store_false")
    parser.add_argument("--allow-idm", action="store_true", default=False)
    parser.add_argument("--no-allow-idm", dest="allow_idm", action="store_false")
    parser.add_argument("--controlled-min-occupancy", type=float, default=0.8)
    parser.add_argument(
        "--track-vehicle-id",
        type=int,
        action="append",
        default=[],
        help="Vehicle id to trace in detail. Repeat for multiple ids.",
    )
    parser.add_argument("--trace-from-step", type=int, default=0)
    parser.add_argument("--trace-to-step", type=int, default=0)
    parser.add_argument(
        "--suspicious-gap-ratio",
        type=float,
        default=1.5,
        help="Report overtaken vehicles braking with gap/desired-gap above this ratio.",
    )
    return parser.parse_args()


def observation_config() -> dict[str, Any]:
    return {
        "type": "Kinematics",
        "vehicles_count": 1,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
        "clip": False,
        "absolute": True,
        "include_obstacles": True,
    }


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = build_env_config(
        scene=str(args.scene),
        action_mode="continuous",
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.prebuilt_split),
        controlled_vehicles=max(1, int(args.controlled_vehicles)),
        control_all_vehicles=bool(args.control_all_vehicles),
        max_surrounding=args.max_surrounding,
        observation_config=observation_config(),
        simulation_frequency=10,
        policy_frequency=10,
        max_episode_steps=int(args.max_steps),
        seed=None,
        simulation_period={"episode_name": str(args.episode_name)},
        ego_vehicle_id=int(args.ego_vehicle_id) if args.ego_vehicle_id is not None else None,
        scene_dataset_collection_mode=True,
        allow_idm=bool(args.allow_idm),
    )
    cfg["expert_test_mode"] = True
    cfg["disable_controlled_vehicle_collisions"] = True
    cfg["terminate_when_all_controlled_crashed"] = False
    cfg["controlled_vehicle_min_occupancy"] = float(args.controlled_min_occupancy)
    if int(args.max_controlled_vehicles) > 0:
        cfg["max_controlled_vehicles"] = int(args.max_controlled_vehicles)
    return cfg


def idle_action(env: gym.Env) -> Any:
    if hasattr(env.action_space, "spaces"):
        return tuple(np.zeros(space.shape, dtype=np.float32) for space in env.action_space.spaces)
    return np.zeros(env.action_space.shape, dtype=np.float32)


def trace_vehicle(base_env, vehicle: NGSIMVehicle, step: int) -> None:
    front, rear = base_env.road.neighbour_vehicles(vehicle, vehicle.lane_index)
    gap = None if front is None else float(vehicle.lane_distance_to(front))
    desired_gap = None if front is None else float(vehicle.desired_gap(vehicle, front))
    print(
        {
            "step": int(step),
            "id": int(vehicle.vehicle_ID),
            "pos": np.round(vehicle.position, 3).tolist(),
            "speed": round(float(vehicle.speed), 3),
            "action_acc": round(float(vehicle.action.get("acceleration", 0.0)), 3),
            "lane": vehicle.lane_index,
            "target_lane": vehicle.target_lane_index,
            "overtaken": bool(vehicle.overtaken),
            "crashed": bool(vehicle.crashed),
            "front_id": None if front is None else int(getattr(front, "vehicle_ID", -1)),
            "front_type": None if front is None else type(front).__name__,
            "front_speed": None if front is None else round(float(getattr(front, "speed", 0.0)), 3),
            "gap": None if gap is None else round(gap, 3),
            "desired_gap": None if desired_gap is None else round(desired_gap, 3),
        }
    )


def main() -> None:
    args = parse_args()
    register_ngsim_env()
    env = gym.make(ENV_ID, config=build_config(args))

    try:
        obs, info = env.reset(seed=int(args.seed))
        del obs, info
        base = env.unwrapped
        tracked_ids = {int(v) for v in args.track_vehicle_id}
        handover_steps: dict[int, int] = {}
        crash_steps: dict[int, int] = {}
        suspicious = defaultdict(int)

        print(
            f"episode={base.episode_name} controlled={len(base.controlled_vehicles)} "
            f"road_vehicles={len(base.road.vehicles)} tracked={sorted(tracked_ids)}"
        )

        for _ in range(int(args.max_steps)):
            obs, reward, terminated, truncated, info = env.step(idle_action(env))
            del obs, reward, info
            step = int(base.steps)

            for vehicle in list(base.road.vehicles):
                if not isinstance(vehicle, NGSIMVehicle):
                    continue
                vid = int(vehicle.vehicle_ID)
                if vehicle.overtaken and vid not in handover_steps:
                    handover_steps[vid] = step
                    if vid in tracked_ids:
                        print(f"handover id={vid} step={step}")
                if vehicle.crashed and vid not in crash_steps:
                    crash_steps[vid] = step
                    if vid in tracked_ids:
                        print(f"crash id={vid} step={step}")

                front, _rear = base.road.neighbour_vehicles(vehicle, vehicle.lane_index)
                gap = None if front is None else float(vehicle.lane_distance_to(front))
                desired_gap = None if front is None else float(vehicle.desired_gap(vehicle, front))
                acc = float(vehicle.action.get("acceleration", 0.0))
                if (
                    vehicle.overtaken
                    and acc < -0.05
                    and front is not None
                    and desired_gap is not None
                    and desired_gap > 1e-6
                    and gap / desired_gap > float(args.suspicious_gap_ratio)
                ):
                    suspicious[vid] += 1

                should_trace = vid in tracked_ids and (
                    vehicle.overtaken
                    or vehicle.crashed
                    or (
                        int(args.trace_to_step) > 0
                        and int(args.trace_from_step) <= step <= int(args.trace_to_step)
                    )
                )
                if should_trace:
                    trace_vehicle(base, vehicle, step)

            if terminated or truncated:
                break

        print("handover_summary", dict(sorted(handover_steps.items())[:20]))
        print("crash_summary", dict(sorted(crash_steps.items())[:20]))
        suspicious_rows = sorted(
            ((count, vid) for vid, count in suspicious.items()),
            reverse=True,
        )
        print("suspicious_gap_braking", suspicious_rows[:20])
    finally:
        env.close()


if __name__ == "__main__":
    main()
