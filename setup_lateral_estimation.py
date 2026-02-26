#!/usr/bin/env python3
"""
Pick a discrete target-speed grid (for MDPVehicle / DiscreteMetaAction) from your
prebuilt NGSIM trajectories.

This script reads the same prebuilt files you load in NGSimEnv.__init__():
  - trajectory_train.npy  (dict: episode -> {veh_id -> meta})
  where meta["trajectory"] is the raw trajectory you pass into process_raw_trajectory()

It then aggregates speeds across:
  - all vehicles, or
  - ego-only vehicles (you can toggle)
and proposes target speed bins using either:
  (A) uniform spacing over a chosen range, or
  (B) quantile-based bins to match the empirical distribution.

Outputs speeds in m/s.

Usage:
  python pick_target_speeds.py \
    --episode-root highway_env/data/processed_10s \
    --scene us-101 \
    --split train \
    --method quantile \
    --n-bins 7 \
    --speed-range 5 35
"""

from __future__ import annotations

import os
import argparse
import numpy as np

# Import your preprocessing exactly like your env does
from highway_env.ngsim_utils.trajectory_gen import process_raw_trajectory


def load_prebuilt(prebuilt_dir: str, split: str):
    veh_ids_path = os.path.join(prebuilt_dir, f"veh_ids_{split}.npy")
    traj_path = os.path.join(prebuilt_dir, f"trajectory_{split}.npy")

    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Missing {traj_path}")
    if not os.path.exists(veh_ids_path):
        raise FileNotFoundError(f"Missing {veh_ids_path}")

    valid_ids_by_episode = np.load(veh_ids_path, allow_pickle=True).item()
    traj_all_by_episode = np.load(traj_path, allow_pickle=True).item()
    episodes = sorted(traj_all_by_episode.keys())
    return valid_ids_by_episode, traj_all_by_episode, episodes


def extract_speeds(
    valid_ids_by_episode: dict,
    traj_all_by_episode: dict,
    episodes: list[str],
    *,
    ego_only: bool,
    max_episodes: int | None,
    max_veh_per_episode: int | None,
    min_speed: float,
    max_speed: float,
):
    speeds = []

    ep_list = episodes if max_episodes is None else episodes[:max_episodes]

    for ep in ep_list:
        traj_all = traj_all_by_episode[ep]

        # If ego_only, we only take speeds for IDs listed in veh_ids_by_episode[ep]
        if ego_only:
            veh_ids = list(valid_ids_by_episode[ep])
        else:
            veh_ids = [vid for vid in traj_all.keys() if isinstance(vid, (int, np.integer))]

        if max_veh_per_episode is not None:
            veh_ids = veh_ids[:max_veh_per_episode]

        for vid in veh_ids:
            if vid not in traj_all:
                continue
            meta = traj_all[vid]
            if "trajectory" not in meta:
                continue

            traj = process_raw_trajectory(meta["trajectory"])  # [T,4] x,y,speed,lane
            if traj is None or len(traj) == 0:
                continue

            v = np.asarray(traj[:, 2], dtype=float)

            # Drop padding/ghost and invalid values
            v = v[np.isfinite(v)]
            v = v[v > 0.0]  # drop zeros from padding
            if v.size == 0:
                continue

            # Optional clamp to avoid extreme tails dominating
            v = v[(v >= min_speed) & (v <= max_speed)]
            if v.size == 0:
                continue

            speeds.append(v)

    if not speeds:
        return np.array([], dtype=float)

    return np.concatenate(speeds, axis=0)


def propose_uniform_bins(v: np.ndarray, n_bins: int, lo: float, hi: float) -> np.ndarray:
    # Uniform bins within [lo, hi], rounded to 0.1 m/s for readability
    bins = np.linspace(lo, hi, n_bins)
    return np.round(bins, 1)


def propose_quantile_bins(v: np.ndarray, n_bins: int, lo: float, hi: float) -> np.ndarray:
    # Quantiles within [lo, hi], then unique+sorted. Guarantees endpoints.
    v = v[(v >= lo) & (v <= hi)]
    if v.size == 0:
        raise ValueError("No speeds remaining after applying speed-range. Widen --speed-range.")

    qs = np.linspace(0.0, 1.0, n_bins)
    bins = np.quantile(v, qs)

    bins[0] = lo
    bins[-1] = hi

    bins = np.round(bins, 1)
    bins = np.unique(bins)

    # If duplicates collapse bins (common with quantiles), fall back to uniform.
    if bins.size < n_bins:
        bins = propose_uniform_bins(v, n_bins, lo, hi)

    return bins


def summarize(v: np.ndarray):
    p = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(v, p)
    out = {f"p{pp:02d}": float(qq) for pp, qq in zip(p, q)}
    out["mean"] = float(np.mean(v))
    out["std"] = float(np.std(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-root", type=str, default="highway_env/data/processed_10s")
    ap.add_argument("--scene", type=str, default="us-101")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--ego-only", action="store_true", help="Use only valid ego IDs per episode")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-veh-per-episode", type=int, default=None)
    ap.add_argument("--speed-range", type=float, nargs=2, default=[0.0, 35.0], metavar=("LO", "HI"))
    ap.add_argument("--clip-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="Optional additional clip applied before speed-range (e.g., 0 40)")
    ap.add_argument("--method", type=str, default="quantile", choices=["quantile", "uniform"])
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    prebuilt_dir = os.path.join(args.episode_root, args.scene, "prebuilt")
    valid_ids_by_episode, traj_all_by_episode, episodes = load_prebuilt(prebuilt_dir, args.split)

    # Two-stage filtering:
    clip_lo, clip_hi = (args.clip_range if args.clip_range is not None else (0.0, float("inf")))
    lo, hi = float(args.speed_range[0]), float(args.speed_range[1])

    v = extract_speeds(
        valid_ids_by_episode,
        traj_all_by_episode,
        episodes,
        ego_only=args.ego_only,
        max_episodes=args.max_episodes,
        max_veh_per_episode=args.max_veh_per_episode,
        min_speed=max(clip_lo, 0.0),
        max_speed=clip_hi,
    )

    if v.size == 0:
        raise RuntimeError("No speeds extracted. Check paths, split, or preprocessing.")

    # Apply final range for bin selection + stats
    v_in = v[(v >= lo) & (v <= hi)]
    if v_in.size == 0:
        raise RuntimeError("No speeds within --speed-range. Widen range or verify units (m/s).")

    stats = summarize(v_in)
    print("Speed stats within speed-range (m/s):")
    for k in ["p00", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100", "mean", "std"]:
        print(f"  {k}: {stats[k]:.3f}")

    if args.method == "quantile":
        bins = propose_quantile_bins(v_in, args.n_bins, lo, hi)
    else:
        bins = propose_uniform_bins(v_in, args.n_bins, lo, hi)

    print("\nSuggested target_speeds (m/s):")
    print(list(map(float, bins)))

    print("\nConfig snippet:")
    print(f'"target_speeds": {list(map(float, bins))},')


if __name__ == "__main__":
    main()
