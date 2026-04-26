#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from highway_env.ngsim_utils.road.gen_road import create_japanese_road, create_ngsim_101_road
from highway_env.ngsim_utils.road.lane_mapping import target_lane_index_from_lane_id
from highway_env.ngsim_utils.data.trajectory_gen import (
    process_raw_trajectory,
    trajectory_row_is_active,
)


ROAD_BUILDERS = {
    "us-101": create_ngsim_101_road,
    "japanese": create_japanese_road,
}


@dataclass
class SceneStats:
    scene: str
    split: str
    num_episodes: int
    num_vehicles: int
    speed_values: np.ndarray
    lateral_offsets: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan prebuilt replay data and recommend a shared discrete speed ladder "
            "and lateral-offset action config that can be used across US-101 and Japanese scenes."
        )
    )
    parser.add_argument(
        "--episode-root",
        default="highway_env/data/processed_20s",
        help="Root folder containing <scene>/prebuilt/trajectory_<split>.npy files.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["us-101", "japanese"],
        choices=["us-101", "japanese"],
        help="Scenes to include in the joint recommendation.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Prebuilt split to analyze.",
    )
    parser.add_argument(
        "--speed-step-candidates",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        help="Candidate shared speed ladder steps in m/s.",
    )
    parser.add_argument(
        "--speed-max-percentile",
        type=float,
        default=99.5,
        help="Robust percentile used to size the shared speed ladder upper bound.",
    )
    parser.add_argument(
        "--speed-complexity-penalty",
        type=float,
        default=0.02,
        help="Penalty per speed bin to avoid overfitting with very fine ladders.",
    )
    parser.add_argument(
        "--offset-step-candidates",
        nargs="+",
        type=float,
        default=[0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30],
        help="Candidate lateral offset steps in meters.",
    )
    parser.add_argument(
        "--offset-max-candidates",
        nargs="+",
        type=float,
        default=[0.6, 0.8, 1.0, 1.2, 1.5],
        help="Candidate absolute lateral offset caps in meters.",
    )
    parser.add_argument(
        "--offset-complexity-penalty",
        type=float,
        default=0.01,
        help="Penalty per lateral bin to avoid overfitting with very fine steering grids.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the recommendation as JSON.",
    )
    return parser.parse_args()


def _valid_rows(traj: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    if traj.shape[1] >= 4:
        return np.asarray([trajectory_row_is_active(row) for row in traj], dtype=bool)
    return ~np.all(np.isclose(traj, 0.0), axis=1)


def _load_prebuilt(scene: str, split: str, episode_root: str) -> dict[str, dict[Any, Any]]:
    path = os.path.join(os.path.abspath(episode_root), scene, "prebuilt", f"trajectory_{split}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prebuilt trajectory file: {path}")
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected prebuilt format in {path}")
    return data


def collect_scene_stats(scene: str, split: str, episode_root: str) -> SceneStats:
    traj_all = _load_prebuilt(scene=scene, split=split, episode_root=episode_root)
    road = ROAD_BUILDERS[scene]()

    speed_values: list[np.ndarray] = []
    lateral_offsets: list[np.ndarray] = []
    num_vehicles = 0

    for _episode_name, vehicles in traj_all.items():
        for _vehicle_id, meta in vehicles.items():
            num_vehicles += 1
            traj = process_raw_trajectory(meta["trajectory"], scene)
            valid = _valid_rows(traj)
            if not np.any(valid):
                continue

            traj_valid = np.asarray(traj[valid], dtype=float)
            speeds = traj_valid[:, 2]
            speeds = speeds[np.isfinite(speeds) & (speeds >= 0.0)]
            if speeds.size:
                speed_values.append(speeds.astype(np.float64, copy=False))

            offsets_vehicle: list[float] = []
            for x, y, _speed, lane_id in traj_valid:
                lane_index = target_lane_index_from_lane_id(road, scene, float(x), int(lane_id))
                if lane_index is None:
                    continue
                try:
                    lane = road.get_lane(lane_index)
                    s, r = lane.local_coordinates(np.array([x, y], dtype=float))
                    lane_width = float(lane.width_at(s))
                except Exception:
                    continue
                # Reject gross off-lane points caused by stale lane ids around merges
                # or geometry mismatches. The discrete steer bias is a within-lane
                # setpoint, so these outliers are not informative for tuning it.
                if (
                    np.isfinite(r)
                    and np.isfinite(lane_width)
                    and abs(r) <= (0.5 * lane_width + 1.0)
                ):
                    offsets_vehicle.append(float(r))
            if offsets_vehicle:
                lateral_offsets.append(np.asarray(offsets_vehicle, dtype=np.float64))

    if not speed_values:
        raise RuntimeError(f"No valid speeds found for scene={scene!r} split={split!r}.")
    if not lateral_offsets:
        raise RuntimeError(f"No valid lateral offsets found for scene={scene!r} split={split!r}.")

    return SceneStats(
        scene=scene,
        split=split,
        num_episodes=len(traj_all),
        num_vehicles=num_vehicles,
        speed_values=np.concatenate(speed_values, axis=0),
        lateral_offsets=np.concatenate(lateral_offsets, axis=0),
    )


def _summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def recommend_speed_ladder(
    speeds: np.ndarray,
    *,
    candidate_steps: list[float],
    max_percentile: float,
    complexity_penalty: float,
) -> dict[str, Any]:
    speeds = np.asarray(speeds, dtype=np.float64)
    robust_max = float(np.percentile(speeds, max_percentile))
    true_max = float(np.max(speeds))

    best: dict[str, Any] | None = None
    for step in candidate_steps:
        if step <= 0:
            continue
        upper = max(robust_max, true_max)
        bins = int(math.ceil(upper / step)) + 1
        ladder = np.arange(bins, dtype=np.float64) * float(step)

        # Compare against nearest ladder point rather than always rounding up.
        idx_hi = np.clip(np.searchsorted(ladder, speeds, side="left"), 0, len(ladder) - 1)
        idx_lo = np.clip(idx_hi - 1, 0, len(ladder) - 1)
        lo = ladder[idx_lo]
        hi = ladder[idx_hi]
        nearest = np.where(np.abs(speeds - lo) <= np.abs(hi - speeds), lo, hi)

        mae = float(np.mean(np.abs(nearest - speeds)))
        score = mae + (complexity_penalty * len(ladder))
        candidate = {
            "step": float(step),
            "num_bins": int(len(ladder)),
            "max_speed": float(ladder[-1]),
            "mean_abs_error": mae,
            "score": float(score),
            "target_speeds": ladder.tolist(),
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("Could not determine a shared speed ladder recommendation.")
    return best


def recommend_lateral_config(
    offsets: np.ndarray,
    *,
    step_candidates: list[float],
    max_candidates: list[float],
    complexity_penalty: float,
) -> dict[str, Any]:
    offsets = np.asarray(offsets, dtype=np.float64)
    best: dict[str, Any] | None = None

    for offset_max in max_candidates:
        if offset_max <= 0:
            continue
        clipped = np.clip(offsets, -offset_max, offset_max)
        clipped_fraction = float(np.mean(np.abs(offsets) > offset_max))

        for step in step_candidates:
            if step <= 0 or step > offset_max:
                continue
            bins_each_side = int(math.floor(offset_max / step))
            if bins_each_side < 1:
                continue

            levels = np.arange(-bins_each_side, bins_each_side + 1, dtype=np.float64) * float(step)
            idx_hi = np.clip(np.searchsorted(levels, clipped, side="left"), 0, len(levels) - 1)
            idx_lo = np.clip(idx_hi - 1, 0, len(levels) - 1)
            lo = levels[idx_lo]
            hi = levels[idx_hi]
            nearest = np.where(np.abs(clipped - lo) <= np.abs(hi - clipped), lo, hi)

            mae = float(np.mean(np.abs(nearest - clipped)))
            score = mae + (complexity_penalty * len(levels))
            candidate = {
                "lateral_offset_step": float(step),
                "lateral_offset_max": float(offset_max),
                "num_levels": int(len(levels)),
                "mean_abs_error": mae,
                "clipped_fraction": clipped_fraction,
                "score": float(score),
                "levels": levels.tolist(),
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

    if best is None:
        raise RuntimeError("Could not determine a shared lateral offset recommendation.")
    return best


def build_report(scene_stats: list[SceneStats], args: argparse.Namespace) -> dict[str, Any]:
    joint_speeds = np.concatenate([stats.speed_values for stats in scene_stats], axis=0)
    joint_offsets = np.concatenate([stats.lateral_offsets for stats in scene_stats], axis=0)

    return {
        "input": {
            "episode_root": os.path.abspath(args.episode_root),
            "scenes": list(args.scenes),
            "split": args.split,
        },
        "per_scene": [
            {
                "scene": stats.scene,
                "split": stats.split,
                "num_episodes": stats.num_episodes,
                "num_vehicles": stats.num_vehicles,
                "speed_summary_mps": _summarize(stats.speed_values),
                "lateral_offset_summary_m": _summarize(stats.lateral_offsets),
            }
            for stats in scene_stats
        ],
        "joint": {
            "speed_summary_mps": _summarize(joint_speeds),
            "lateral_offset_summary_m": _summarize(joint_offsets),
            "recommended_speed_ladder": recommend_speed_ladder(
                joint_speeds,
                candidate_steps=list(args.speed_step_candidates),
                max_percentile=float(args.speed_max_percentile),
                complexity_penalty=float(args.speed_complexity_penalty),
            ),
            "recommended_lateral_config": recommend_lateral_config(
                joint_offsets,
                step_candidates=list(args.offset_step_candidates),
                max_candidates=list(args.offset_max_candidates),
                complexity_penalty=float(args.offset_complexity_penalty),
            ),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print("Joint discrete action recommendation")
    print(
        f"episode_root={report['input']['episode_root']} "
        f"scenes={','.join(report['input']['scenes'])} "
        f"split={report['input']['split']}"
    )
    print("")

    for scene_report in report["per_scene"]:
        print(
            f"[scene] {scene_report['scene']} "
            f"episodes={scene_report['num_episodes']} "
            f"vehicles={scene_report['num_vehicles']}"
        )
        speed = scene_report["speed_summary_mps"]
        lat = scene_report["lateral_offset_summary_m"]
        print(
            "  speed_mps "
            f"p01={speed['p01']:.2f} median={speed['median']:.2f} "
            f"p95={speed['p95']:.2f} p99={speed['p99']:.2f} max={speed['max']:.2f}"
        )
        print(
            "  lateral_offset_m "
            f"p01={lat['p01']:.3f} median={lat['median']:.3f} "
            f"p95={lat['p95']:.3f} p99={lat['p99']:.3f} max={lat['max']:.3f}"
        )
        print("")

    speed_rec = report["joint"]["recommended_speed_ladder"]
    lat_rec = report["joint"]["recommended_lateral_config"]

    print("[joint recommendation]")
    print(
        f"target_speeds_step_mps={speed_rec['step']:.3f} "
        f"num_bins={speed_rec['num_bins']} "
        f"max_speed_mps={speed_rec['max_speed']:.2f} "
        f"mean_abs_quantization_error_mps={speed_rec['mean_abs_error']:.3f}"
    )
    print(f"target_speeds={speed_rec['target_speeds']}")
    print(
        f"lateral_offset_step_m={lat_rec['lateral_offset_step']:.3f} "
        f"lateral_offset_max_m={lat_rec['lateral_offset_max']:.3f} "
        f"num_levels={lat_rec['num_levels']} "
        f"mean_abs_quantization_error_m={lat_rec['mean_abs_error']:.3f} "
        f"clipped_fraction={lat_rec['clipped_fraction']:.4f}"
    )
    print("")
    print("[config snippet]")
    print(
        json.dumps(
            {
                "action_config": {
                    "target_speeds": speed_rec["target_speeds"],
                    "lateral_offset_step": lat_rec["lateral_offset_step"],
                    "lateral_offset_max": lat_rec["lateral_offset_max"],
                }
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    stats = [
        collect_scene_stats(scene=scene, split=args.split, episode_root=args.episode_root)
        for scene in args.scenes
    ]
    report = build_report(stats, args)
    print_report(report)

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("")
        print(f"Saved JSON report to: {output_path}")


if __name__ == "__main__":
    main()
