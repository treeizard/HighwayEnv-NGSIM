#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env  # noqa: E402
from highway_env.ngsim_utils.data.trajectory_gen import (  # noqa: E402
    longest_continuous_active_span_bounds,
    trajectory_has_min_continuous_occupancy,
    trajectory_row_is_active,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose NGSIM scene replay by comparing raw trajectory occupancy against "
            "replayed controlled/obstacle vehicles, and by plotting vehicle size "
            "distributions behind controlled-vehicle eligibility."
        )
    )
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episode-name", default="t1118847759700")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--percentage-controlled-vehicles", type=float, default=0.1)
    parser.add_argument("--controlled-min-occupancy", type=float, default=0.8)
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        default=False,
        help="Use every valid controlled vehicle.",
    )
    parser.add_argument(
        "--no-control-all-vehicles",
        dest="control_all_vehicles",
        action="store_false",
    )
    parser.add_argument(
        "--allow-idm",
        action="store_true",
        default=False,
        help="Allow replay obstacles to hand over to IDM instead of disappearing when replay ends.",
    )
    parser.add_argument(
        "--no-allow-idm",
        dest="allow_idm",
        action="store_false",
    )
    parser.add_argument("--sample-controlled", type=int, default=5)
    parser.add_argument("--sample-obstacles", type=int, default=8)
    parser.add_argument("--out-dir", default="expert_compare_outputs/replay_diagnostics")
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
        action_mode="teleport",
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.prebuilt_split),
        percentage_controlled_vehicles=float(args.percentage_controlled_vehicles),
        control_all_vehicles=bool(args.control_all_vehicles),
        max_surrounding=args.max_surrounding,
        observation_config=observation_config(),
        simulation_frequency=10,
        policy_frequency=10,
        max_episode_steps=int(args.max_steps),
        seed=None,
        simulation_period={"episode_name": str(args.episode_name)},
        scene_dataset_collection_mode=True,
        allow_idm=bool(args.allow_idm),
    )
    cfg["expert_test_mode"] = False
    cfg["disable_controlled_vehicle_collisions"] = True
    cfg["terminate_when_all_controlled_crashed"] = False
    cfg["controlled_vehicle_min_occupancy"] = float(args.controlled_min_occupancy)
    return cfg


def load_prebuilt_episode(
    episode_root: str,
    scene: str,
    split: str,
    episode_name: str,
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    prebuilt_dir = Path(episode_root) / scene / "prebuilt"
    traj_all = np.load(prebuilt_dir / f"trajectory_{split}.npy", allow_pickle=True).item()
    valid_ids = np.load(prebuilt_dir / f"veh_ids_{split}.npy", allow_pickle=True).item()
    veh_dict = traj_all[str(episode_name)]
    valid = {int(v) for v in valid_ids.get(str(episode_name), [])}
    return veh_dict, valid


def vehicle_active_steps(traj: np.ndarray) -> list[int]:
    return [idx for idx, row in enumerate(np.asarray(traj, dtype=float)) if trajectory_row_is_active(row)]


def build_raw_stats(
    veh_dict: dict[int, dict[str, Any]],
    controlled_ids: set[int],
    valid_ids: set[int],
    min_occupancy: float,
) -> dict[str, Any]:
    any_traj = next(iter(veh_dict.values()))["trajectory"]
    horizon = int(np.asarray(any_traj).shape[0])
    raw_active_all = np.zeros(horizon, dtype=int)
    raw_active_obstacles = np.zeros(horizon, dtype=int)
    rows: list[dict[str, Any]] = []

    for vid, meta in veh_dict.items():
        vehicle_id = int(vid)
        traj = np.asarray(meta["trajectory"], dtype=float)
        active_idx = vehicle_active_steps(traj)
        if active_idx:
            raw_active_all[active_idx] += 1
            if vehicle_id not in controlled_ids:
                raw_active_obstacles[active_idx] += 1
            start_idx = int(active_idx[0])
            end_idx = int(active_idx[-1])
        else:
            start_idx = None
            end_idx = None

        span_start, span_end, span_len = longest_continuous_active_span_bounds(traj)
        length = float(meta.get("length", 0.0))
        width = float(meta.get("width", 0.0))
        size_ok = 10.0 <= length <= 22.0 and 4.0 <= width <= 8.0
        continuity_ok = trajectory_has_min_continuous_occupancy(
            traj,
            min_presence_ratio=min_occupancy,
        )
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "continuous_start": span_start,
                "continuous_end": span_end,
                "continuous_len": int(span_len),
                "length": length,
                "width": width,
                "size_ok": bool(size_ok),
                "continuity_ok": bool(continuity_ok),
                "valid_controlled": vehicle_id in valid_ids,
                "selected_controlled": vehicle_id in controlled_ids,
            }
        )

    return {
        "horizon": horizon,
        "raw_active_all": raw_active_all,
        "raw_active_obstacles": raw_active_obstacles,
        "rows": rows,
    }


def choose_sample_ids(
    rows: list[dict[str, Any]],
    controlled_ids: list[int],
    sample_controlled: int,
    sample_obstacles: int,
) -> tuple[list[int], list[int]]:
    controlled_sample = controlled_ids[: max(1, sample_controlled)]

    obstacle_candidates = [
        row
        for row in rows
        if not row["selected_controlled"] and row["start_idx"] is not None
    ]
    obstacle_candidates.sort(
        key=lambda row: (
            -int(row["continuous_len"]),
            -int(row["end_idx"]),
            int(row["vehicle_id"]),
        )
    )
    obstacle_sample = [int(row["vehicle_id"]) for row in obstacle_candidates[: max(1, sample_obstacles)]]
    return controlled_sample, obstacle_sample


def run_env_diagnostics(
    args: argparse.Namespace,
    controlled_sample: list[int],
    obstacle_sample: list[int],
) -> dict[str, Any]:
    register_ngsim_env()
    env = gym.make(ENV_ID, config=build_config(args))
    replay_positions = {vehicle_id: [] for vehicle_id in controlled_sample + obstacle_sample}
    replay_active_steps = {vehicle_id: [] for vehicle_id in controlled_sample + obstacle_sample}
    replay_obstacle_active_counts: list[int] = []
    replay_obstacle_total_counts: list[int] = []
    replay_controlled_active_counts: list[int] = []

    try:
        _obs, _info = env.reset(seed=int(args.seed))
        del _obs, _info
        base = env.unwrapped

        for _step in range(int(args.max_steps)):
            action = tuple(1 for _ in range(len(base.controlled_vehicles)))
            _obs, _reward, terminated, truncated, info = env.step(action)
            del _obs, _reward

            step_index = int(base.steps)
            alive_controlled = {int(v) for v in info.get("alive_controlled_vehicle_ids", []) if v is not None}
            replay_controlled_active_counts.append(len(alive_controlled))

            obstacle_total = 0
            obstacle_active = 0
            road_ids_seen: set[int] = set()
            for vehicle in base.road.vehicles:
                vehicle_id = int(getattr(vehicle, "vehicle_ID", -1))
                road_ids_seen.add(vehicle_id)
                if vehicle in base.controlled_vehicles:
                    if vehicle_id in replay_positions and bool(
                        getattr(vehicle, "scene_collection_is_active", False)
                    ):
                        replay_positions[vehicle_id].append(np.asarray(vehicle.position, dtype=float).copy())
                        replay_active_steps[vehicle_id].append(step_index)
                    continue

                obstacle_total += 1
                is_active = bool(getattr(vehicle, "appear", True)) and not bool(
                    getattr(vehicle, "remove_from_road", False)
                )
                obstacle_active += int(is_active)
                if vehicle_id in replay_positions and is_active:
                    replay_positions[vehicle_id].append(np.asarray(vehicle.position, dtype=float).copy())
                    replay_active_steps[vehicle_id].append(step_index)

            replay_obstacle_total_counts.append(obstacle_total)
            replay_obstacle_active_counts.append(obstacle_active)

            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "replay_positions": replay_positions,
        "replay_active_steps": replay_active_steps,
        "replay_obstacle_total_counts": np.asarray(replay_obstacle_total_counts, dtype=int),
        "replay_obstacle_active_counts": np.asarray(replay_obstacle_active_counts, dtype=int),
        "replay_controlled_active_counts": np.asarray(replay_controlled_active_counts, dtype=int),
    }


def plot_counts(
    out_dir: Path,
    raw_stats: dict[str, Any],
    replay: dict[str, Any],
) -> None:
    steps_raw = np.arange(raw_stats["horizon"], dtype=int)
    steps_rep = np.arange(1, len(replay["replay_obstacle_active_counts"]) + 1, dtype=int)

    plt.figure(figsize=(11, 5))
    plt.plot(steps_raw, raw_stats["raw_active_all"], label="Raw Active All Vehicles", linewidth=2.0)
    plt.plot(steps_raw, raw_stats["raw_active_obstacles"], label="Raw Active Obstacles", linewidth=2.0)
    plt.plot(steps_rep, replay["replay_obstacle_active_counts"], label="Replay Active Obstacles", linewidth=2.0)
    plt.plot(steps_rep, replay["replay_obstacle_total_counts"], label="Replay Total Obstacles On Road", linewidth=1.5)
    plt.plot(steps_rep, replay["replay_controlled_active_counts"], label="Replay Active Controlled", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Vehicle Count")
    plt.title("Raw Scene Occupancy vs Replay Occupancy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "counts_vs_step.png", dpi=180)
    plt.close()


def plot_intervals(
    out_dir: Path,
    rows: list[dict[str, Any]],
    replay_active_steps: dict[int, list[int]],
    controlled_sample: list[int],
    obstacle_sample: list[int],
) -> None:
    row_map = {int(row["vehicle_id"]): row for row in rows}
    ordered_ids = controlled_sample + obstacle_sample
    labels = []

    plt.figure(figsize=(12, 0.55 * max(len(ordered_ids), 4) + 2))
    y_positions = np.arange(len(ordered_ids), dtype=float)

    for y, vehicle_id in zip(y_positions, ordered_ids):
        row = row_map[vehicle_id]
        label_prefix = "ctrl" if vehicle_id in controlled_sample else "obs"
        labels.append(f"{label_prefix}:{vehicle_id}")

        if row["start_idx"] is not None and row["end_idx"] is not None:
            plt.hlines(y + 0.12, row["start_idx"], row["end_idx"], colors="tab:blue", linewidth=3.0)

        rep_steps = replay_active_steps.get(vehicle_id, [])
        if rep_steps:
            plt.hlines(y - 0.12, min(rep_steps), max(rep_steps), colors="tab:orange", linewidth=3.0)

    plt.yticks(y_positions, labels)
    plt.xlabel("Step")
    plt.title("Raw Active Interval (blue) vs Replay Active Interval (orange)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "intervals_raw_vs_replay.png", dpi=180)
    plt.close()


def plot_trajectories(
    out_dir: Path,
    veh_dict: dict[int, dict[str, Any]],
    replay_positions: dict[int, list[np.ndarray]],
    controlled_sample: list[int],
    obstacle_sample: list[int],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
    for ax, ids, title in zip(
        axes,
        [controlled_sample, obstacle_sample],
        ["Controlled Trajectories", "Obstacle Trajectories"],
    ):
        for vehicle_id in ids:
            raw_traj = np.asarray(veh_dict[vehicle_id]["trajectory"], dtype=float)
            active = np.asarray([trajectory_row_is_active(row) for row in raw_traj], dtype=bool)
            raw_xy = raw_traj[active, :2]
            rep_xy = np.asarray(replay_positions.get(vehicle_id, []), dtype=float)

            if raw_xy.size:
                ax.plot(raw_xy[:, 0], raw_xy[:, 1], linewidth=1.7, label=f"raw {vehicle_id}")
            if rep_xy.size:
                ax.plot(rep_xy[:, 0], rep_xy[:, 1], linestyle="--", linewidth=1.7, label=f"replay {vehicle_id}")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_xy_raw_vs_replay.png", dpi=180)
    plt.close()


def plot_size_distribution(
    out_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    def _category(row: dict[str, Any]) -> str:
        if row["valid_controlled"]:
            return "valid_controlled"
        if row["continuity_ok"] and not row["size_ok"]:
            return "continuous_but_size_filtered"
        if row["size_ok"] and not row["continuity_ok"]:
            return "size_ok_but_short_or_discontinuous"
        return "other_invalid"

    categories = {
        "valid_controlled": ("tab:blue", "o"),
        "continuous_but_size_filtered": ("tab:red", "x"),
        "size_ok_but_short_or_discontinuous": ("tab:orange", "^"),
        "other_invalid": ("0.7", "."),
    }

    plt.figure(figsize=(8, 6))
    for category, (color, marker) in categories.items():
        subset = [row for row in rows if _category(row) == category]
        if not subset:
            continue
        plt.scatter(
            [row["length"] for row in subset],
            [row["width"] for row in subset],
            c=color,
            marker=marker,
            alpha=0.85,
            label=f"{category} (n={len(subset)})",
        )

    for row in rows:
        if row["continuity_ok"] and not row["valid_controlled"]:
            plt.annotate(
                str(row["vehicle_id"]),
                (row["length"], row["width"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    plt.axvspan(10.0, 22.0, color="tab:green", alpha=0.08)
    plt.axhspan(4.0, 8.0, color="tab:green", alpha=0.08)
    plt.xlabel("Length")
    plt.ylabel("Width")
    plt.title("Vehicle Length/Width Distribution and Controlled Eligibility")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "vehicle_size_distribution.png", dpi=180)
    plt.close()


def print_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    replay: dict[str, Any],
    controlled_sample: list[int],
    obstacle_sample: list[int],
) -> None:
    long_lived_invalid = [
        row
        for row in rows
        if row["continuity_ok"] and not row["valid_controlled"]
    ]
    print(f"episode={args.episode_name} scene={args.scene} split={args.prebuilt_split}")
    print(f"selected_controlled_ids={controlled_sample}")
    print(f"sample_obstacle_ids={obstacle_sample}")
    print(
        f"replay_obstacle_active_start={int(replay['replay_obstacle_active_counts'][0])} "
        f"replay_obstacle_active_end={int(replay['replay_obstacle_active_counts'][-1])}"
    )
    print(
        f"replay_obstacle_total_start={int(replay['replay_obstacle_total_counts'][0])} "
        f"replay_obstacle_total_end={int(replay['replay_obstacle_total_counts'][-1])}"
    )
    print(f"long_lived_invalid_count={len(long_lived_invalid)}")
    for row in long_lived_invalid[:20]:
        print(
            "  long_lived_invalid "
            f"id={row['vehicle_id']} length={row['length']:.2f} width={row['width']:.2f} "
            f"size_ok={row['size_ok']} continuous_len={row['continuous_len']}"
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / args.scene / args.prebuilt_split / args.episode_name
    out_dir.mkdir(parents=True, exist_ok=True)

    veh_dict, valid_ids = load_prebuilt_episode(
        episode_root=args.episode_root,
        scene=args.scene,
        split=args.prebuilt_split,
        episode_name=args.episode_name,
    )

    register_ngsim_env()
    probe_env = gym.make(ENV_ID, config=build_config(args))
    try:
        _obs, _info = probe_env.reset(seed=int(args.seed))
        del _obs, _info
        controlled_ids = [int(v.vehicle_ID) for v in probe_env.unwrapped.controlled_vehicles]
    finally:
        probe_env.close()

    raw_stats = build_raw_stats(
        veh_dict=veh_dict,
        controlled_ids=set(controlled_ids),
        valid_ids=valid_ids,
        min_occupancy=float(args.controlled_min_occupancy),
    )
    controlled_sample, obstacle_sample = choose_sample_ids(
        rows=raw_stats["rows"],
        controlled_ids=controlled_ids,
        sample_controlled=int(args.sample_controlled),
        sample_obstacles=int(args.sample_obstacles),
    )
    replay = run_env_diagnostics(args, controlled_sample, obstacle_sample)

    plot_counts(out_dir, raw_stats, replay)
    plot_intervals(out_dir, raw_stats["rows"], replay["replay_active_steps"], controlled_sample, obstacle_sample)
    plot_trajectories(out_dir, veh_dict, replay["replay_positions"], controlled_sample, obstacle_sample)
    plot_size_distribution(out_dir, raw_stats["rows"])
    print_summary(args, raw_stats["rows"], replay, controlled_sample, obstacle_sample)
    print(f"saved_plots={out_dir}")


if __name__ == "__main__":
    main()
