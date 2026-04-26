#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from highway_env.ngsim_utils.core.constants import (  # noqa: E402
    FEET_PER_METER,
    IDM_PARAMETER_PRESETS,
    SCENE_IDM_PARAMETER_KEY,
)
from highway_env.ngsim_utils.data.trajectory_gen import trajectory_row_is_active  # noqa: E402


US_SCENES = {"us-101", "i-80", "lankershim"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate placeholder IDM/MOBIL parameters directly from prebuilt "
            "processed_20s trajectory caches."
        )
    )
    parser.add_argument(
        "--episode-root",
        default="highway_env/data/processed_20s",
        help="Root folder containing <scene>/prebuilt/trajectory_<split>.npy.",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene to analyze, e.g. us-101 or japanese.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which prebuilt split(s) to use.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episodes to analyze for a quick pass.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the estimated parameter block as JSON.",
    )
    parser.add_argument(
        "--free-gap-m",
        type=float,
        default=60.0,
        help="Minimum leader gap treated as free-flow.",
    )
    parser.add_argument(
        "--max-following-gap-m",
        type=float,
        default=120.0,
        help="Maximum leader gap included in following statistics.",
    )
    parser.add_argument(
        "--steady-dv-mps",
        type=float,
        default=1.0,
        help="Maximum |dv| for steady-following samples.",
    )
    parser.add_argument(
        "--steady-acc-mps2",
        type=float,
        default=0.5,
        help="Maximum |a| for steady-following samples.",
    )
    parser.add_argument(
        "--low-speed-threshold-mps",
        type=float,
        default=2.0,
        help="Speed threshold used when estimating standstill/min-gap spacing.",
    )
    parser.add_argument(
        "--lane-change-window",
        type=int,
        default=1,
        help="Frames before/after a lane change used for simple MOBIL heuristics.",
    )
    return parser.parse_args()


def load_prebuilt_episodes(
    episode_root: str,
    scene: str,
    split: str,
    max_episodes: int | None = None,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    splits = ("train", "val", "test") if split == "all" else (split,)
    merged: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for split_name in splits:
        path = os.path.join(episode_root, scene, "prebuilt", f"trajectory_{split_name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing prebuilt trajectory cache: {path}")
        data = np.load(path, allow_pickle=True).item()
        for episode_name in sorted(data.keys()):
            merged[f"{split_name}:{episode_name}"] = data[episode_name]
            if max_episodes is not None and len(merged) >= max_episodes:
                return merged
    return merged


def scene_to_region_key(scene: str) -> str:
    return SCENE_IDM_PARAMETER_KEY.get(scene, "US")


def normalize_scene_trajectory(scene: str, trajectory: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=float).copy()
    if scene in US_SCENES:
        longitudinal = traj[:, 1] / FEET_PER_METER
        lateral = (traj[:, 0] - 6.0) / FEET_PER_METER
        speed = traj[:, 2] / FEET_PER_METER
    elif scene == "japanese":
        longitudinal = traj[:, 0]
        lateral = traj[:, 1]
        speed = traj[:, 2] / 3.6
    else:
        raise ValueError(f"Unsupported scene={scene!r}")

    traj[:, 0] = longitudinal
    traj[:, 1] = lateral
    traj[:, 2] = speed
    return traj


def normalize_vehicle_dimensions(scene: str, length: float, width: float) -> tuple[float, float]:
    if scene in US_SCENES:
        return float(length) / FEET_PER_METER, float(width) / FEET_PER_METER
    return float(length), float(width)


def infer_dt(episodes: dict[str, dict[int, dict[str, np.ndarray]]], scene: str) -> float:
    dts: list[float] = []
    for vehicle_dict in episodes.values():
        for meta in vehicle_dict.values():
            traj = normalize_scene_trajectory(scene, np.asarray(meta["trajectory"], dtype=float))
            active = np.array([trajectory_row_is_active(row) for row in traj], dtype=bool)
            if np.count_nonzero(active) < 3:
                continue
            ds = np.diff(traj[:, 0])
            v_mid = np.maximum(traj[:-1, 2], 1e-3)
            step_dt = np.abs(ds[active[:-1] & active[1:]]) / v_mid[active[:-1] & active[1:]]
            valid = step_dt[np.isfinite(step_dt) & (step_dt > 0.02) & (step_dt < 1.0)]
            if valid.size:
                dts.extend(valid.tolist())
        if len(dts) > 5000:
            break
    if not dts:
        return 0.1
    return float(np.median(np.asarray(dts, dtype=float)))


def robust_percentile(values: list[float], q: float, default: float | None = None) -> float | None:
    if not values:
        return default
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.percentile(arr, q))


def robust_median(values: list[float], default: float | None = None) -> float | None:
    if not values:
        return default
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.median(arr))


def clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return float(np.clip(value, 0.0, 1.0))


def analyze_prebuilt_scene(
    episodes: dict[str, dict[int, dict[str, np.ndarray]]],
    scene: str,
    *,
    free_gap_m: float,
    max_following_gap_m: float,
    steady_dv_mps: float,
    steady_acc_mps2: float,
    low_speed_threshold_mps: float,
    lane_change_window: int,
) -> tuple[dict, dict]:
    dt = infer_dt(episodes, scene)

    free_speeds: list[float] = []
    positive_accels: list[float] = []
    negative_accels: list[float] = []
    following_gaps: list[float] = []
    steady_headways: list[float] = []
    low_speed_gaps: list[float] = []
    lane_change_intervals: list[float] = []
    lane_change_gains: list[float] = []
    imposed_braking: list[float] = []
    politeness_ratios: list[float] = []

    sample_counts = defaultdict(int)

    for episode_name, vehicle_dict in episodes.items():
        del episode_name
        normalized: dict[int, dict[str, np.ndarray | float]] = {}
        max_steps = 0

        for veh_id, meta in vehicle_dict.items():
            traj = normalize_scene_trajectory(scene, np.asarray(meta["trajectory"], dtype=float))
            length_m, width_m = normalize_vehicle_dimensions(scene, meta["length"], meta["width"])
            active = np.array([trajectory_row_is_active(row) for row in traj], dtype=bool)
            accel = np.zeros(traj.shape[0], dtype=float)
            if traj.shape[0] >= 2:
                accel[1:] = np.diff(traj[:, 2]) / dt
                accel[0] = accel[1]
            normalized[int(veh_id)] = {
                "traj": traj,
                "active": active,
                "accel": accel,
                "length": float(length_m),
                "width": float(width_m),
            }
            max_steps = max(max_steps, traj.shape[0])

        if not normalized:
            continue

        for t in range(max_steps):
            lane_buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
            for veh_id, item in normalized.items():
                traj = item["traj"]
                active = item["active"]
                if t >= len(traj) or not active[t]:
                    continue
                lane_id = int(round(float(traj[t, 3])))
                if lane_id <= 0:
                    continue
                lane_buckets[lane_id].append((float(traj[t, 0]), veh_id))

            leaders: dict[int, int] = {}
            followers: dict[int, int] = {}
            for lane_id, entries in lane_buckets.items():
                del lane_id
                entries.sort(key=lambda pair: pair[0])
                for idx in range(len(entries) - 1):
                    follower_id = entries[idx][1]
                    leader_id = entries[idx + 1][1]
                    leaders[follower_id] = leader_id
                    followers[leader_id] = follower_id

            for veh_id, item in normalized.items():
                traj = item["traj"]
                active = item["active"]
                accel = item["accel"]
                if t >= len(traj) or not active[t]:
                    continue

                speed = float(traj[t, 2])
                acc = float(accel[t])
                leader_id = leaders.get(veh_id)

                if leader_id is None:
                    if speed > 0.0:
                        free_speeds.append(speed)
                        sample_counts["free_speed"] += 1
                    if acc > 0.0:
                        positive_accels.append(acc)
                        sample_counts["positive_accel"] += 1
                    continue

                leader = normalized[leader_id]
                leader_traj = leader["traj"]
                if t >= len(leader_traj) or not leader["active"][t]:
                    continue

                s = float(traj[t, 0])
                s_lead = float(leader_traj[t, 0])
                v_lead = float(leader_traj[t, 2])
                center_gap = s_lead - s
                bumper_gap = center_gap - 0.5 * (float(item["length"]) + float(leader["length"]))

                if not math.isfinite(bumper_gap) or bumper_gap <= 0.0:
                    continue

                following_gaps.append(bumper_gap)
                sample_counts["following_gap"] += 1

                dv = speed - v_lead
                if bumper_gap >= free_gap_m:
                    free_speeds.append(speed)
                    sample_counts["free_speed"] += 1
                    if acc > 0.0:
                        positive_accels.append(acc)
                        sample_counts["positive_accel"] += 1
                elif bumper_gap <= max_following_gap_m:
                    if acc < 0.0:
                        negative_accels.append(abs(acc))
                        sample_counts["negative_accel"] += 1

                    if (
                        speed > low_speed_threshold_mps
                        and abs(dv) <= steady_dv_mps
                        and abs(acc) <= steady_acc_mps2
                    ):
                        steady_headways.append(bumper_gap / max(speed, 1e-3))
                        sample_counts["steady_headway"] += 1

                    if speed <= low_speed_threshold_mps and abs(dv) <= steady_dv_mps:
                        low_speed_gaps.append(bumper_gap)
                        sample_counts["low_speed_gap"] += 1

            # Lane-change heuristics
            for veh_id, item in normalized.items():
                traj = item["traj"]
                active = item["active"]
                accel = item["accel"]
                if t <= 0 or t >= len(traj) - 1:
                    continue
                if not (active[t - 1] and active[t]):
                    continue

                prev_lane = int(round(float(traj[t - 1, 3])))
                curr_lane = int(round(float(traj[t, 3])))
                if prev_lane <= 0 or curr_lane <= 0 or prev_lane == curr_lane:
                    continue

                sample_counts["lane_change"] += 1

                prev_change_times = item.setdefault("change_times", [])
                prev_change_times.append(t)
                if len(prev_change_times) >= 2:
                    lane_change_intervals.append((prev_change_times[-1] - prev_change_times[-2]) * dt)

                before_idx = max(0, t - lane_change_window)
                after_idx = min(len(accel) - 1, t + lane_change_window)
                gain = float(accel[after_idx] - accel[before_idx])
                if math.isfinite(gain):
                    lane_change_gains.append(gain)

                follower_id = None
                new_lane_entries = lane_buckets.get(curr_lane, [])
                if new_lane_entries:
                    ordered = sorted(new_lane_entries, key=lambda pair: pair[0])
                    for idx, (_s_val, candidate_id) in enumerate(ordered):
                        if candidate_id == veh_id and idx > 0:
                            follower_id = ordered[idx - 1][1]
                            break

                if follower_id is not None:
                    follower = normalized[follower_id]
                    follower_accel_arr = follower["accel"]
                    follower_after_idx = min(after_idx, len(follower_accel_arr) - 1)
                    if follower_after_idx < 0:
                        continue
                    follower_accel = float(follower_accel_arr[follower_after_idx])
                    braking = max(0.0, -follower_accel)
                    imposed_braking.append(braking)
                    if gain > 0.0:
                        politeness_ratios.append(braking / max(gain, 1e-3))

    desired_speed = robust_percentile(free_speeds, 85.0, default=30.0)
    min_gap = robust_percentile(low_speed_gaps, 50.0, default=2.0)
    raw_time_headway = robust_percentile(steady_headways, 50.0, default=1.5)
    if min_gap is not None and raw_time_headway is not None:
        time_headway = max(0.1, raw_time_headway - min_gap / max(desired_speed or 1.0, 1.0))
    else:
        time_headway = raw_time_headway

    acceleration = robust_percentile(positive_accels, 90.0, default=1.5)
    comfortable_deceleration = robust_percentile(negative_accels, 85.0, default=2.0)
    lane_change_delay = robust_percentile(lane_change_intervals, 50.0, default=1.0)
    lane_change_min_acc_gain = robust_percentile(
        [gain for gain in lane_change_gains if gain > 0.0],
        25.0,
        default=0.2,
    )
    lane_change_max_braking_imposed = robust_percentile(imposed_braking, 85.0, default=2.0)
    politeness = clip01(robust_percentile(politeness_ratios, 50.0, default=0.2))

    region_key = scene_to_region_key(scene)
    result = deepcopy(IDM_PARAMETER_PRESETS[region_key])
    result["scene"] = scene
    result["dt_seconds"] = dt
    result["idm"] = {
        "desired_speed": round(float(desired_speed), 4) if desired_speed is not None else None,
        "time_headway": round(float(time_headway), 4) if time_headway is not None else None,
        "min_gap": round(float(min_gap), 4) if min_gap is not None else None,
        "acceleration": round(float(acceleration), 4) if acceleration is not None else None,
        "comfortable_deceleration": round(float(comfortable_deceleration), 4)
        if comfortable_deceleration is not None
        else None,
        "delta": 4.0,
    }
    result["mobil"] = {
        "politeness": round(float(politeness), 4) if politeness is not None else None,
        "lane_change_min_acc_gain": round(float(lane_change_min_acc_gain), 4)
        if lane_change_min_acc_gain is not None
        else None,
        "lane_change_max_braking_imposed": round(float(lane_change_max_braking_imposed), 4)
        if lane_change_max_braking_imposed is not None
        else None,
        "lane_change_delay": round(float(lane_change_delay), 4) if lane_change_delay is not None else None,
    }

    diagnostics = {
        "sample_counts": dict(sorted(sample_counts.items())),
        "episodes_analyzed": len(episodes),
        "free_speed_p50": robust_percentile(free_speeds, 50.0),
        "free_speed_p85": robust_percentile(free_speeds, 85.0),
        "following_gap_p50": robust_percentile(following_gaps, 50.0),
        "following_gap_p85": robust_percentile(following_gaps, 85.0),
        "steady_headway_p50": robust_percentile(steady_headways, 50.0),
        "low_speed_gap_p50": robust_percentile(low_speed_gaps, 50.0),
        "positive_accel_p90": robust_percentile(positive_accels, 90.0),
        "negative_accel_p85": robust_percentile(negative_accels, 85.0),
    }
    return result, diagnostics


def print_summary(result: dict, diagnostics: dict) -> None:
    print("\nEstimated IDM/MOBIL placeholder parameters")
    print(json.dumps(result, indent=2, sort_keys=False))
    print("\nDiagnostics")
    print(json.dumps(diagnostics, indent=2, sort_keys=False))


def main() -> None:
    args = parse_args()
    episodes = load_prebuilt_episodes(
        episode_root=args.episode_root,
        scene=args.scene,
        split=args.split,
        max_episodes=args.max_episodes,
    )
    result, diagnostics = analyze_prebuilt_scene(
        episodes,
        args.scene,
        free_gap_m=float(args.free_gap_m),
        max_following_gap_m=float(args.max_following_gap_m),
        steady_dv_mps=float(args.steady_dv_mps),
        steady_acc_mps2=float(args.steady_acc_mps2),
        low_speed_threshold_mps=float(args.low_speed_threshold_mps),
        lane_change_window=int(args.lane_change_window),
    )
    print_summary(result, diagnostics)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "parameters": result,
                    "diagnostics": diagnostics,
                },
                f,
                indent=2,
            )
        print(f"\nSaved estimates to {out_path}")


if __name__ == "__main__":
    main()
