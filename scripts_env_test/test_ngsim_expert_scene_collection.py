#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm.auto import tqdm


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight expert-control regression check for scene collection. "
            "Uses cheap Kinematics observations so we can validate controlled-vehicle "
            "activity windows and hidden-vehicle ghosting without lidar/camera cost."
        )
    )
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--controlled-vehicles", type=int, default=4)
    parser.add_argument("--max-controlled-vehicles", type=int, default=0)
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        default=True,
        help="Check every viable controlled vehicle in the chosen scene.",
    )
    parser.add_argument(
        "--no-control-all-vehicles",
        dest="control_all_vehicles",
        action="store_false",
    )
    parser.add_argument(
        "--expert-control-mode",
        choices=["teleport", "continuous"],
        default="continuous",
    )
    parser.add_argument(
        "--allow-idm",
        action="store_true",
        default=False,
        help="Allow surrounding replay vehicles to hand over to IDM.",
    )
    parser.add_argument(
        "--no-allow-idm",
        dest="allow_idm",
        action="store_false",
    )
    parser.add_argument(
        "--controlled-min-occupancy",
        type=float,
        default=0.8,
        help="Minimum required active ratio for each controlled vehicle.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record the regression rollout to disk.",
    )
    parser.add_argument(
        "--video-dir",
        default="expert_data/videos",
        help="Output directory for recorded videos.",
    )
    parser.add_argument(
        "--video-prefix",
        default="ngsim_expert_scene_collection",
        help="Filename prefix for recorded videos.",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1200,
        help="Render width used for video/offscreen rendering.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=600,
        help="Render height used for video/offscreen rendering.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=5.5,
        help="Scene zoom factor. Larger values zoom in more.",
    )
    return parser.parse_args()


def light_observation_config() -> dict[str, Any]:
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
        action_mode=str(args.expert_control_mode),
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.prebuilt_split),
        controlled_vehicles=max(1, int(args.controlled_vehicles)),
        control_all_vehicles=bool(args.control_all_vehicles),
        max_surrounding=args.max_surrounding,
        observation_config=light_observation_config(),
        simulation_frequency=10,
        policy_frequency=10,
        max_episode_steps=int(args.max_steps),
        seed=None,
        scene_dataset_collection_mode=True,
        allow_idm=bool(args.allow_idm),
    )
    cfg["expert_test_mode"] = str(args.expert_control_mode) != "teleport"
    cfg["disable_controlled_vehicle_collisions"] = True
    cfg["terminate_when_all_controlled_crashed"] = False
    cfg["controlled_vehicle_min_occupancy"] = float(args.controlled_min_occupancy)
    cfg["screen_width"] = int(args.screen_width)
    cfg["screen_height"] = int(args.screen_height)
    cfg["scaling"] = float(args.scaling)
    cfg["offscreen_rendering"] = bool(args.save_video)
    if int(args.max_controlled_vehicles) > 0:
        cfg["max_controlled_vehicles"] = int(args.max_controlled_vehicles)
    return cfg


def make_env(args: argparse.Namespace) -> gym.Env:
    cfg = build_config(args)
    render_mode = "rgb_array" if bool(args.save_video) else None
    env = gym.make(ENV_ID, render_mode=render_mode, config=cfg)
    if bool(args.save_video):
        os.makedirs(str(args.video_dir), exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(args.video_dir),
            name_prefix=str(args.video_prefix),
            episode_trigger=lambda episode_id: True,
            disable_logger=True,
        )
    return env


def idle_action(env: gym.Env, num_agents: int) -> Any:
    if hasattr(env.action_space, "spaces"):
        return tuple(1 for _ in range(num_agents))
    return 1


def hidden_vehicle_issues(base_env) -> list[str]:
    issues: list[str] = []
    for vehicle in list(base_env.road.vehicles):
        is_scene_controlled_hidden = hasattr(vehicle, "scene_collection_is_active") and not bool(
            getattr(vehicle, "scene_collection_is_active", True)
        )
        is_replay_hidden = hasattr(vehicle, "appear") and not bool(getattr(vehicle, "appear", True))
        if not (is_scene_controlled_hidden or is_replay_hidden):
            continue

        position = np.asarray(getattr(vehicle, "position", np.zeros(2, dtype=float)), dtype=float)
        length = float(getattr(vehicle, "LENGTH", 0.0))
        width = float(getattr(vehicle, "WIDTH", 0.0))
        speed = float(getattr(vehicle, "speed", 0.0))
        if (
            not np.allclose(position[:2], 0.0, atol=1e-6)
            or abs(length) > 1e-6
            or abs(width) > 1e-6
            or abs(speed) > 1e-6
        ):
            issues.append(
                f"hidden vehicle id={getattr(vehicle, 'vehicle_ID', None)} "
                f"pos={position[:2].tolist()} L={length:.4f} W={width:.4f} speed={speed:.4f}"
            )
    return issues


def update_path_errors(base_env, path_errors: dict[int, list[float]]) -> None:
    step_index = int(getattr(base_env, "steps", 0))
    for vehicle in base_env.controlled_vehicles:
        vehicle_id = int(getattr(vehicle, "vehicle_ID", -1))
        if vehicle_id not in path_errors:
            continue
        if not bool(getattr(vehicle, "scene_collection_is_active", False)):
            continue

        traj = np.asarray(getattr(vehicle, "scene_collection_full_traj", ()), dtype=float)
        if traj.ndim != 2 or step_index >= len(traj):
            continue

        ref_row = np.asarray(traj[step_index], dtype=float)
        ref_xy = ref_row[:2]
        sim_xy = np.asarray(getattr(vehicle, "position", np.zeros(2, dtype=float)), dtype=float)[:2]
        err = float(np.linalg.norm(sim_xy - ref_xy))
        path_errors[vehicle_id].append(err)


def path_metric_summary(path_errors: dict[int, list[float]]) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    per_vehicle: dict[int, dict[str, float]] = {}
    ades: list[float] = []
    fdes: list[float] = []

    for vehicle_id, errors in path_errors.items():
        if not errors:
            per_vehicle[vehicle_id] = {"T": 0.0, "ADE_m": np.nan, "FDE_m": np.nan}
            continue
        err_arr = np.asarray(errors, dtype=float)
        ade = float(np.mean(err_arr))
        fde = float(err_arr[-1])
        per_vehicle[vehicle_id] = {
            "T": float(len(err_arr)),
            "ADE_m": ade,
            "FDE_m": fde,
        }
        ades.append(ade)
        fdes.append(fde)

    overall = {
        "avg_ADE_m": float(np.mean(ades)) if ades else np.nan,
        "avg_FDE_m": float(np.mean(fdes)) if fdes else np.nan,
        "max_ADE_m": float(np.max(ades)) if ades else np.nan,
        "max_FDE_m": float(np.max(fdes)) if fdes else np.nan,
    }
    return per_vehicle, overall


def run_episode(env: gym.Env, episode_idx: int, args: argparse.Namespace) -> None:
    obs, info = env.reset(seed=int(args.seed) + int(episode_idx))
    del obs, info

    base = env.unwrapped
    controlled_ids = [int(v.vehicle_ID) for v in base.controlled_vehicles]
    active_counts = {vehicle_id: 0 for vehicle_id in controlled_ids}
    path_errors = {vehicle_id: [] for vehicle_id in controlled_ids}
    total_steps = 0
    hidden_issues: list[str] = []

    done = False
    progress = tqdm(
        total=int(args.max_steps),
        desc=f"episode {episode_idx + 1}",
        leave=False,
    )
    try:
        while not done and total_steps < int(args.max_steps):
            action = idle_action(env, len(base.controlled_vehicles))
            _obs, _reward, terminated, truncated, info = env.step(action)
            del _obs, _reward
            total_steps += 1
            progress.update(1)

            alive_ids = {int(v) for v in info.get("alive_controlled_vehicle_ids", []) if v is not None}
            progress.set_postfix(
                alive=len(alive_ids),
                controlled=len(controlled_ids),
                refresh=False,
            )
            for vehicle_id in alive_ids:
                if vehicle_id in active_counts:
                    active_counts[vehicle_id] += 1

            update_path_errors(base, path_errors)
            hidden_issues.extend(hidden_vehicle_issues(base))
            done = bool(terminated or truncated)
    finally:
        progress.close()

    print(
        f"\nepisode={base.episode_name} controlled={len(controlled_ids)} "
        f"steps={total_steps} ids={controlled_ids}"
    )
    threshold = float(args.controlled_min_occupancy)
    per_vehicle_path_metrics, overall_path_metrics = path_metric_summary(path_errors)
    for vehicle_id in controlled_ids:
        ratio = (float(active_counts[vehicle_id]) / float(max(total_steps, 1)))
        status = "OK" if ratio >= threshold else "FAIL"
        metrics = per_vehicle_path_metrics[vehicle_id]
        ade_str = f"{metrics['ADE_m']:.4f}" if np.isfinite(metrics["ADE_m"]) else "nan"
        fde_str = f"{metrics['FDE_m']:.4f}" if np.isfinite(metrics["FDE_m"]) else "nan"
        print(
            f"  controlled_id={vehicle_id} active_steps={active_counts[vehicle_id]} "
            f"ratio={ratio:.3f} threshold={threshold:.3f} status={status} "
            f"ADE_m={ade_str} FDE_m={fde_str}"
        )

    avg_ade_str = (
        f"{overall_path_metrics['avg_ADE_m']:.4f}"
        if np.isfinite(overall_path_metrics["avg_ADE_m"])
        else "nan"
    )
    avg_fde_str = (
        f"{overall_path_metrics['avg_FDE_m']:.4f}"
        if np.isfinite(overall_path_metrics["avg_FDE_m"])
        else "nan"
    )
    max_ade_str = (
        f"{overall_path_metrics['max_ADE_m']:.4f}"
        if np.isfinite(overall_path_metrics["max_ADE_m"])
        else "nan"
    )
    max_fde_str = (
        f"{overall_path_metrics['max_FDE_m']:.4f}"
        if np.isfinite(overall_path_metrics["max_FDE_m"])
        else "nan"
    )
    print(
        f"  path_summary avg_ADE_m={avg_ade_str} avg_FDE_m={avg_fde_str} "
        f"max_ADE_m={max_ade_str} max_FDE_m={max_fde_str}"
    )

    deduped_hidden_issues = list(dict.fromkeys(hidden_issues))
    if deduped_hidden_issues:
        print("  hidden-state issues:")
        for line in deduped_hidden_issues[:20]:
            print(f"    {line}")
        if len(deduped_hidden_issues) > 20:
            print(f"    ... {len(deduped_hidden_issues) - 20} more")
    else:
        print("  hidden-state issues: none")


def main() -> None:
    args = parse_args()
    register_ngsim_env()
    env = make_env(args)
    try:
        for episode_idx in range(int(args.episodes)):
            run_episode(env, episode_idx, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
