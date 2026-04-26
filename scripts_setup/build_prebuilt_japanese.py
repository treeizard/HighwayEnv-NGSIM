#!/usr/bin/env python3
"""
Build prebuilt Japanese trajectory caches directly from the filtered Morinomiya
raw-data artifact.

This script turns the filtered `.npy` produced from the `raw_data` pipeline into
the same prebuilt format consumed by `NGSimEnv`:

  <episode_root>/japanese/prebuilt/
    veh_ids_train.npy
    trajectory_train.npy
    veh_ids_val.npy
    trajectory_val.npy
    veh_ids_test.npy
    trajectory_test.npy

It follows the existing notebook / raw-data workflow:
1. load filtered Morinomiya records
2. reconstruct local XY coordinates
3. estimate a curved-road remap using mainline lanes
4. smooth each vehicle trajectory
5. slice into fixed-duration windows
6. split windows into consecutive train / val / test sets
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import datetime as dt
import gc

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RAW_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(RAW_DATA_DIR))

from highway_env.data.curvature_remap import estimate_curvature_remap
from highway_env.ngsim_utils.data.trajectory_gen import (
    trajectory_has_min_continuous_occupancy,
    trajectory_smoothing,
)


MORINOMIYA_START_JST = pd.Timestamp("2020-01-01 09:00:00", tz="Asia/Tokyo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Japanese prebuilt trajectory caches from filtered Morinomiya data."
    )
    parser.add_argument(
        "--input_npy",
        default="raw_data/morinomiya_filtered_800.npy",
        help="Filtered Morinomiya .npy produced by the raw_data preprocessing pipeline.",
    )
    parser.add_argument(
        "--episode_root",
        default="highway_env/data/processed_20s",
        help="Root output folder that will contain <scene>/prebuilt/*.npy.",
    )
    parser.add_argument(
        "--scene",
        default="japanese",
        help="Scene name used in the output folder layout.",
    )
    parser.add_argument(
        "--window_sec",
        type=int,
        default=20,
        help="Episode window size in seconds.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.3333,
        help="Fraction of episode windows assigned to validation from the middle time segment.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.3333,
        help="Fraction of episode windows assigned to test from the latest time segment.",
    )
    parser.add_argument(
        "--presence_ratio_threshold",
        type=float,
        default=0.8,
        help="Minimum fraction of expected frames required for a vehicle to be considered valid.",
    )
    parser.add_argument(
        "--x_m_max",
        type=float,
        default=800.0,
        help="Optional longitudinal crop in meters after local XY conversion.",
    )
    parser.add_argument(
        "--basis_lat",
        type=float,
        default=34.681580,
        help="Reference latitude for local XY conversion.",
    )
    parser.add_argument(
        "--basis_lon",
        type=float,
        default=135.527945,
        help="Reference longitude for local XY conversion.",
    )
    parser.add_argument(
        "--centerline_lanes",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Lane ids used to estimate the centerline for curvature remapping.",
    )
    parser.add_argument(
        "--bin_size_m",
        type=float,
        default=5.0,
        help="Longitudinal bin size used by the centerline estimator.",
    )
    parser.add_argument(
        "--curved_flatten_bin_size_m",
        type=float,
        default=10.0,
        help="Bin size used to suppress residual end-of-road curvature in y_curved.",
    )
    parser.add_argument(
        "--curved_flatten_sample_step",
        type=int,
        default=25,
        help="Use every Nth remapped row to estimate the residual y_curved drift profile.",
    )
    parser.add_argument(
        "--start_clock",
        default="09:00:00",
        help="Start clock time in JST for filtering rows, format HH:MM[:SS].",
    )
    parser.add_argument(
        "--end_clock",
        default="13:00:00",
        help="End clock time in JST for filtering rows, format HH:MM[:SS].",
    )
    return parser.parse_args()


def add_local_xy_fast(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    basis_lat: float = 34.681580,
    basis_lon: float = 135.527945,
) -> pd.DataFrame:
    """Fast local tangent-plane approximation in meters."""
    radius = 6378137.0
    out = df.copy()
    lat0 = np.deg2rad(basis_lat)
    lon0 = np.deg2rad(basis_lon)

    lat = np.deg2rad(pd.to_numeric(out[lat_col], errors="coerce").to_numpy())
    lon = np.deg2rad(pd.to_numeric(out[lon_col], errors="coerce").to_numpy())

    out["x_m"] = radius * (lon - lon0) * np.cos(lat0)
    out["y_m"] = radius * (lat - lat0)
    return out


def parse_clock_time(value: str) -> dt.time:
    """Parse a wall-clock time in HH:MM or HH:MM:SS format."""
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid clock time {value!r}; expected HH:MM[:SS].")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    return dt.time(hour=hour, minute=minute, second=second)


def filter_by_jst_clock(
    df: pd.DataFrame,
    start_clock: dt.time,
    end_clock: dt.time,
) -> pd.DataFrame:
    """
    Keep only rows inside the requested JST clock window.

    The default follows the existing Morinomiya filtering pipeline:
    09:00:00 <= time < 12:00:00.
    """
    if start_clock >= end_clock:
        raise ValueError("--start_clock must be earlier than --end_clock.")

    out = df.copy()
    out["datetime_jst"] = pd.to_datetime(out["datetime_jst"], errors="coerce")
    out = out.dropna(subset=["datetime_jst"]).copy()

    time_mask = (
        (out["datetime_jst"].dt.time >= start_clock)
        & (out["datetime_jst"].dt.time < end_clock)
    )
    return out.loc[time_mask].copy()


def load_filtered_morinomiya(
    npy_path: str,
    basis_lat: float,
    basis_lon: float,
    x_m_max: float | None,
    start_clock: dt.time,
    end_clock: dt.time,
) -> pd.DataFrame:
    """
    Load and clean the filtered Morinomiya record array.

    Memory-saving behavior:
    - apply the JST clock filter immediately after reconstructing timestamps
    - compute local XY only if needed for the early x-range crop
    - apply `x_m_max` before constructing the large pandas DataFrame
    - drop unused raw columns as soon as the spatial filter is done
    """
    arr = np.load(npy_path, allow_pickle=True)
    field_names = list(arr.dtype.names or [])
    if not field_names:
        raise ValueError(
            f"{npy_path} does not contain a structured array with named columns."
        )

    def field_to_numeric(name: str) -> np.ndarray:
        if name not in field_names:
            raise KeyError(name)
        return pd.to_numeric(pd.Series(arr[name], copy=False), errors="coerce").to_numpy()

    numeric_data: dict[str, np.ndarray] = {}
    for col in [
        "vehicle_id",
        "datetime",
        "vehicle_type",
        "velocity",
        "traffic_lane",
        "longitude",
        "latitude",
        "vehicle_length",
        "detected_flag",
        "x_m",
        "y_m",
    ]:
        if col in field_names:
            numeric_data[col] = field_to_numeric(col)

    required_mask = (
        np.isfinite(numeric_data["vehicle_id"])
        & np.isfinite(numeric_data["datetime"])
        & np.isfinite(numeric_data["traffic_lane"])
    )

    if "datetime_jst" in field_names:
        datetime_jst = pd.to_datetime(
            pd.Series(arr["datetime_jst"], copy=False),
            errors="coerce",
        )
    else:
        datetime_jst = pd.Series(
            MORINOMIYA_START_JST + pd.to_timedelta(
                numeric_data["datetime"],
                unit="ms",
            ),
            copy=False,
        )

    datetime_jst_mask = datetime_jst.notna().to_numpy()
    if not np.any(datetime_jst_mask & required_mask):
        raise ValueError("No valid datetime rows remained after parsing JST timestamps.")

    clock_times = datetime_jst.dt.time
    time_mask = (
        (clock_times >= start_clock)
        & (clock_times < end_clock)
    ).to_numpy()
    mask = required_mask & datetime_jst_mask & time_mask

    if "x_m" in numeric_data and "y_m" in numeric_data:
        x_m = numeric_data["x_m"]
        y_m = numeric_data["y_m"]
    else:
        radius = 6378137.0
        lat0 = np.deg2rad(basis_lat)
        lon0 = np.deg2rad(basis_lon)
        lat = np.deg2rad(numeric_data["latitude"])
        lon = np.deg2rad(numeric_data["longitude"])
        x_m = radius * (lon - lon0) * np.cos(lat0)
        y_m = radius * (lat - lat0)

    xy_mask = np.isfinite(x_m) & np.isfinite(y_m)
    mask &= xy_mask

    if x_m_max is not None:
        mask &= x_m <= float(x_m_max)

    if not np.any(mask):
        raise ValueError("No rows remained after applying JST and x_m_max filtering.")

    # Materialize only the post-crop subset needed downstream.
    data = {
        "vehicle_id": numeric_data["vehicle_id"][mask].astype(np.int64, copy=False),
        "datetime": numeric_data["datetime"][mask].astype(np.int64, copy=False),
        "datetime_jst": datetime_jst[mask].reset_index(drop=True),
        "traffic_lane": numeric_data["traffic_lane"][mask].astype(np.int64, copy=False),
        "x_m": x_m[mask],
        "y_m": y_m[mask],
    }
    for col in ["vehicle_type", "velocity", "vehicle_length"]:
        if col in numeric_data:
            data[col] = numeric_data[col][mask]

    df = pd.DataFrame(data)
    del arr
    gc.collect()

    gc.collect()

    return df.sort_values(["vehicle_id", "datetime"]).reset_index(drop=True)


def add_vehicle_width(df: pd.DataFrame) -> pd.DataFrame:
    """Infer Morinomiya vehicle widths using the notebook rules."""
    out = df.copy()
    out["vehicle_width"] = np.nan

    mask_type1 = out["vehicle_type"] == 1
    out.loc[mask_type1 & (out["vehicle_length"] <= 3.4), "vehicle_width"] = 1.48
    out.loc[
        mask_type1
        & (out["vehicle_length"] > 3.4)
        & (out["vehicle_length"] <= 4.7),
        "vehicle_width",
    ] = 1.7
    out.loc[mask_type1 & (out["vehicle_length"] > 4.7), "vehicle_width"] = 2.0
    out.loc[out["vehicle_type"] == 2, "vehicle_width"] = 2.5

    # Sensible fallback for uncommon / missing type codes.
    out.loc[out["vehicle_width"].isna() & (out["vehicle_length"] <= 4.7), "vehicle_width"] = 1.7
    out.loc[out["vehicle_width"].isna() & (out["vehicle_length"] > 4.7), "vehicle_width"] = 2.0
    return out


def smooth_vehicle_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the existing Savitzky-Golay smoothing per vehicle."""
    smoothed_groups: list[pd.DataFrame] = []

    for vehicle_id, group in df.groupby("vehicle_id", sort=True):
        del vehicle_id
        ordered = group.sort_values("datetime").copy()
        traj = ordered[["x_curved", "y_curved", "velocity", "traffic_lane"]].to_numpy(dtype=float)
        smoothed = np.asarray(trajectory_smoothing(traj), dtype=float)

        ordered["x_smooth"] = smoothed[:, 0]
        ordered["y_smooth"] = smoothed[:, 1]
        ordered["v_smooth"] = smoothed[:, 2]
        smoothed_groups.append(ordered)

    if not smoothed_groups:
        raise ValueError("No vehicle trajectories were available for smoothing.")

    return pd.concat(smoothed_groups, ignore_index=True)


def suppress_terminal_curvature(
    df: pd.DataFrame,
    *,
    x_col: str = "x_curved",
    y_col: str = "y_curved",
    bin_size_m: float = 10.0,
    sample_step: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove shared residual lateral drift from the curved-road projection.

    The centerline remap is already close to straight, but sparse downstream
    coverage can leave a small common bend near the road end. We estimate that
    shared bias profile in x_curved bins and subtract it from all rows so the
    prebuilt road stays visually and numerically flatter.
    """
    out = df.copy()
    sample_step = max(1, int(sample_step))
    sample = out.iloc[::sample_step].copy()
    sample = sample.dropna(subset=[x_col, y_col]).copy()
    if sample.empty:
        empty_profile = pd.DataFrame(columns=["x_center", "y_bias"])
        out["y_curved_raw"] = out[y_col]
        return out, empty_profile

    x_vals = pd.to_numeric(sample[x_col], errors="coerce").to_numpy(dtype=float)
    y_vals = pd.to_numeric(sample[y_col], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[finite_mask]
    y_vals = y_vals[finite_mask]
    if x_vals.size == 0:
        empty_profile = pd.DataFrame(columns=["x_center", "y_bias"])
        out["y_curved_raw"] = out[y_col]
        return out, empty_profile

    bin_size = max(1.0, float(bin_size_m))
    bins = np.arange(0.0, float(np.nanmax(x_vals)) + bin_size, bin_size, dtype=float)
    if bins.size < 2:
        empty_profile = pd.DataFrame(columns=["x_center", "y_bias"])
        out["y_curved_raw"] = out[y_col]
        return out, empty_profile

    sample["_bin"] = pd.cut(
        sample[x_col],
        bins=bins,
        labels=False,
        include_lowest=True,
    )
    profile = (
        sample.groupby("_bin", observed=False)[y_col]
        .median()
        .dropna()
        .to_frame("y_median")
    )
    if profile.empty:
        empty_profile = pd.DataFrame(columns=["x_center", "y_bias"])
        out["y_curved_raw"] = out[y_col]
        return out, empty_profile

    profile["x_center"] = bins[:-1][profile.index.to_numpy(dtype=int)] + 0.5 * bin_size
    profile["y_bias"] = (
        profile["y_median"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
    )

    x_bias = profile["x_center"].to_numpy(dtype=float)
    y_bias = profile["y_bias"].to_numpy(dtype=float)
    out["y_curved_raw"] = pd.to_numeric(out[y_col], errors="coerce")
    out[y_col] = out["y_curved_raw"] - np.interp(
        pd.to_numeric(out[x_col], errors="coerce").to_numpy(dtype=float),
        x_bias,
        y_bias,
        left=float(y_bias[0]),
        right=float(y_bias[-1]),
    )
    return out, profile.reset_index(drop=True)[["x_center", "y_bias"]]


def build_episode_dicts(
    df_smooth: pd.DataFrame,
    window_sec: int,
    presence_ratio_threshold: float,
) -> tuple[dict[str, list[np.int64]], dict[str, dict[np.int64, dict[str, np.ndarray]]]]:
    """Build per-window trajectory and valid-id dictionaries."""
    df = df_smooth.copy()

    required_cols = [
        "vehicle_id",
        "vehicle_length",
        "vehicle_width",
        "x_smooth",
        "y_smooth",
        "v_smooth",
        "traffic_lane",
        "datetime_jst",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "vehicle_id",
        "vehicle_length",
        "vehicle_width",
        "x_smooth",
        "y_smooth",
        "v_smooth",
        "traffic_lane",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime_jst"] = pd.to_datetime(df["datetime_jst"], errors="coerce")
    df = df.dropna(subset=["datetime_jst"] + numeric_cols).copy()

    if df.empty:
        raise ValueError("No valid rows remain after smoothing and cleaning.")

    df["vehicle_id"] = df["vehicle_id"].astype(np.int64)
    df["traffic_lane"] = df["traffic_lane"].astype(np.int64)
    df = df.sort_values(["datetime_jst", "vehicle_id"]).copy()

    unique_times = pd.Series(df["datetime_jst"].drop_duplicates().sort_values())
    time_deltas = unique_times.diff().dropna().dt.total_seconds()
    if len(time_deltas) == 0:
        raise ValueError("Not enough timestamps to estimate the sampling interval.")

    nominal_dt = float(time_deltas.median())
    if nominal_dt <= 0:
        raise ValueError("Estimated a non-positive sampling interval.")

    expected_frames = max(int(round(window_sec / nominal_dt)), 1)
    min_presence_frames = int(np.ceil(expected_frames * presence_ratio_threshold))

    print(f"Estimated nominal dt: {nominal_dt:.6f} s")
    print(f"Expected frames per {window_sec}s window: {expected_frames}")
    print(
        f"Presence threshold ({presence_ratio_threshold:.0%}): "
        f"{min_presence_frames} frames"
    )

    start_time = df["datetime_jst"].min().floor(f"{window_sec}s")
    elapsed_sec = (df["datetime_jst"] - start_time).dt.total_seconds()
    df["_window_idx"] = (elapsed_sec // window_sec).astype(int)
    df["_window_start"] = start_time + pd.to_timedelta(df["_window_idx"] * window_sec, unit="s")

    veh_ids_by_episode: dict[str, list[np.int64]] = {}
    trajectories_by_episode: dict[str, dict[np.int64, dict[str, np.ndarray]]] = {}

    for window_start, group in df.groupby("_window_start", sort=True):
        key = f"t{int(window_start.timestamp() * 1000)}"
        traj_dict: dict[np.int64, dict[str, np.ndarray]] = {}
        valid_ids: list[np.int64] = []

        for veh_id, veh_group in group.groupby("vehicle_id", sort=True):
            ordered = veh_group.sort_values("datetime_jst")
            traj = ordered[["x_smooth", "y_smooth", "v_smooth", "traffic_lane"]].to_numpy(dtype=float)
            traj[:, 3] = ordered["traffic_lane"].to_numpy(dtype=np.int64)

            traj_dict[np.int64(veh_id)] = {
                "length": np.float64(ordered["vehicle_length"].iloc[0]),
                "width": np.float64(ordered["vehicle_width"].iloc[0]),
                "trajectory": traj,
            }

            if len(ordered) >= min_presence_frames and trajectory_has_min_continuous_occupancy(
                traj,
                min_presence_ratio=presence_ratio_threshold,
            ):
                valid_ids.append(np.int64(veh_id))

        if traj_dict:
            veh_ids_by_episode[key] = valid_ids
            trajectories_by_episode[key] = traj_dict

    return veh_ids_by_episode, trajectories_by_episode


def split_episode_keys(
    episode_keys: list[str],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[str], list[str], list[str]]:
    """Split episode keys into consecutive train, validation, and test sets."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val_ratio must be in [0, 1).")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("--test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val_ratio + --test_ratio must be < 1.0.")

    ordered = sorted(episode_keys)
    n_total = len(ordered)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = min(max(n_test, 0), n_total)
    n_val = min(max(n_val, 0), n_total - n_test)
    n_train = n_total - n_val - n_test
    train_keys = ordered[:n_train]
    val_keys = ordered[n_train : n_train + n_val]
    test_keys = ordered[n_train + n_val :]
    return train_keys, val_keys, test_keys


def subset_dict(d: dict, keys: list[str]) -> dict:
    return {key: d[key] for key in keys}


def save_split(
    out_dir: str,
    split: str,
    veh_ids_by_episode: dict[str, list[np.int64]],
    trajectories_by_episode: dict[str, dict[np.int64, dict[str, np.ndarray]]],
) -> None:
    out_path_ids = os.path.join(out_dir, f"veh_ids_{split}.npy")
    out_path_traj = os.path.join(out_dir, f"trajectory_{split}.npy")
    np.save(out_path_ids, veh_ids_by_episode)
    np.save(out_path_traj, trajectories_by_episode)
    print(f"Saved {len(veh_ids_by_episode)} episodes to {out_path_ids}")
    print(f"Saved {len(trajectories_by_episode)} episodes to {out_path_traj}")


def main() -> None:
    args = parse_args()
    start_clock = parse_clock_time(args.start_clock)
    end_clock = parse_clock_time(args.end_clock)

    print(f"Loading filtered Morinomiya data from: {args.input_npy}")
    df = load_filtered_morinomiya(
        npy_path=args.input_npy,
        basis_lat=args.basis_lat,
        basis_lon=args.basis_lon,
        x_m_max=args.x_m_max,
        start_clock=start_clock,
        end_clock=end_clock,
    )
    print(
        f"Loaded {len(df)} cleaned rows across {df['vehicle_id'].nunique()} vehicles "
        f"for JST window {args.start_clock} to {args.end_clock}"
    )

    df = add_vehicle_width(df)
    df = df.dropna(subset=["vehicle_length", "vehicle_width", "velocity"]).copy()

    df_curved, centerline_df = estimate_curvature_remap(
        df,
        lanes=list(args.centerline_lanes),
        bin_size_m=args.bin_size_m,
    )
    df["x_curved"] = pd.to_numeric(df_curved["x_curved"], errors="coerce")
    df["y_curved"] = pd.to_numeric(df_curved["y_curved"], errors="coerce")
    df = df.dropna(subset=["x_curved", "y_curved"]).copy()
    df, flatten_profile = suppress_terminal_curvature(
        df,
        x_col="x_curved",
        y_col="y_curved",
        bin_size_m=args.curved_flatten_bin_size_m,
        sample_step=args.curved_flatten_sample_step,
    )

    print(
        "Centerline estimated with "
        f"{len(centerline_df)} samples using lanes {list(args.centerline_lanes)}"
    )
    print(
        "Applied residual curved-road flattening with "
        f"{len(flatten_profile)} bias samples"
    )

    df_smooth = smooth_vehicle_trajectories(df)
    veh_ids_all, traj_all = build_episode_dicts(
        df_smooth=df_smooth,
        window_sec=args.window_sec,
        presence_ratio_threshold=args.presence_ratio_threshold,
    )

    episode_keys = sorted(traj_all.keys())
    if not episode_keys:
        raise RuntimeError("No episodes were created from the filtered Japanese dataset.")

    train_keys, val_keys, test_keys = split_episode_keys(
        episode_keys=episode_keys,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    out_dir = os.path.join(args.episode_root, args.scene, "prebuilt")
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Built {len(episode_keys)} total episodes: "
        f"{len(train_keys)} train, {len(val_keys)} val, {len(test_keys)} test"
    )

    save_split(
        out_dir=out_dir,
        split="train",
        veh_ids_by_episode=subset_dict(veh_ids_all, train_keys),
        trajectories_by_episode=subset_dict(traj_all, train_keys),
    )

    save_split(
        out_dir=out_dir,
        split="val",
        veh_ids_by_episode=subset_dict(veh_ids_all, val_keys),
        trajectories_by_episode=subset_dict(traj_all, val_keys),
    )

    save_split(
        out_dir=out_dir,
        split="test",
        veh_ids_by_episode=subset_dict(veh_ids_all, test_keys),
        trajectories_by_episode=subset_dict(traj_all, test_keys),
    )


if __name__ == "__main__":
    main()
