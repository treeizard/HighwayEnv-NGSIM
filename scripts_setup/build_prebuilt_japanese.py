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
from highway_env.ngsim_utils.road.gen_road import create_japanese_road
from highway_env.ngsim_utils.road.lane_mapping import target_lane_index_from_lane_id


MORINOMIYA_START_JST = pd.Timestamp("2020-01-01 09:00:00", tz="Asia/Tokyo")
JST_TIMEZONE = "Asia/Tokyo"


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
        "--allowed_lane_ids",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Road-supported Japanese lane ids to keep before building trajectories.",
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
        help="Longitudinal bin size used for lane-aware lateral recentering.",
    )
    parser.add_argument(
        "--curved_flatten_sample_step",
        type=int,
        default=25,
        help="Use every Nth remapped row to estimate lane-center correction profiles.",
    )
    parser.add_argument(
        "--max_lane_lateral_m",
        type=float,
        default=1.65,
        help=(
            "Clip each vehicle center to this absolute lateral offset from its "
            "mapped lane center after recentering. Use a negative value to disable."
        ),
    )
    parser.add_argument(
        "--disable_lane_center_alignment",
        action="store_true",
        help="Skip lane-aware recentering and keep the raw Frenet lateral offsets.",
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
    parser.add_argument(
        "--min_episode_count",
        type=int,
        default=0,
        help=(
            "Fail before saving if fewer than this many episode windows are built. "
            "Use this to avoid accidentally overwriting a full cache with a small subset."
        ),
    )
    parser.add_argument(
        "--require_episode",
        nargs="*",
        default=[],
        help=(
            "Episode names that must be present before saving, e.g. "
            "t1577843200000."
        ),
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


def parse_datetime_jst(values) -> pd.Series:
    """Parse possibly mixed-offset timestamps and normalize them to JST."""
    parsed = pd.to_datetime(
        pd.Series(values, copy=False),
        errors="coerce",
        utc=True,
    )
    return parsed.dt.tz_convert(JST_TIMEZONE)


def parse_morinomiya_clock_datetime(values) -> pd.Series:
    """Parse Morinomiya HHMMSSmmm numeric timestamps into JST datetimes."""
    numeric = pd.to_numeric(pd.Series(values, copy=False), errors="coerce").astype("Int64")
    text = numeric.astype("string").str.zfill(9)
    valid = numeric.notna() & text.str.match(r"^\d{9}$", na=False)

    hours = pd.to_numeric(text.str.slice(0, 2), errors="coerce")
    minutes = pd.to_numeric(text.str.slice(2, 4), errors="coerce")
    seconds = pd.to_numeric(text.str.slice(4, 6), errors="coerce")
    millis = pd.to_numeric(text.str.slice(6, 9), errors="coerce")
    valid &= (
        hours.between(0, 23)
        & minutes.between(0, 59)
        & seconds.between(0, 59)
        & millis.between(0, 999)
    )

    total_ms = (
        hours * 3_600_000
        + minutes * 60_000
        + seconds * 1_000
        + millis
    )
    return MORINOMIYA_START_JST.normalize() + pd.to_timedelta(
        total_ms.where(valid),
        unit="ms",
    )


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
        datetime_jst = parse_datetime_jst(arr["datetime_jst"])
    else:
        datetime_jst = parse_morinomiya_clock_datetime(numeric_data["datetime"])

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


def target_japanese_lane_center_y(net, x: float, lane_id: int) -> float:
    """Return the HighwayEnv Japanese lane center y for a processed x/lane id."""
    lane_index = target_lane_index_from_lane_id(net, "japanese", float(x), int(lane_id))
    if lane_index is None:
        return float("nan")
    lane = net.get_lane(lane_index)
    local_s, _local_r = lane.local_coordinates(np.array([float(x), 0.0], dtype=float))
    lane_length = float(getattr(lane, "length", local_s))
    if np.isfinite(lane_length) and lane_length > 0:
        local_s = float(np.clip(local_s, 0.0, lane_length))
    return float(lane.position(local_s, 0.0)[1])


def target_japanese_lane_center_y_array(
    x_values: np.ndarray,
    lane_ids: np.ndarray,
) -> np.ndarray:
    net = create_japanese_road()
    out = np.full(len(x_values), np.nan, dtype=float)
    for i, (x_value, lane_id) in enumerate(zip(x_values, lane_ids)):
        if not np.isfinite(x_value) or not np.isfinite(lane_id):
            continue
        out[i] = target_japanese_lane_center_y(net, float(x_value), int(lane_id))
    return out


def align_lanes_to_japanese_road(
    df: pd.DataFrame,
    *,
    x_col: str = "x_curved",
    y_col: str = "y_curved",
    lane_col: str = "traffic_lane",
    bin_size_m: float = 10.0,
    sample_step: int = 25,
    max_abs_lateral_m: float | None = 1.65,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Align each recorded lane center to the HighwayEnv Japanese road geometry.

    The raw Morinomiya road bends slightly near the downstream end. The Frenet
    projection straightens that bend, but residual lane-specific lateral drift can
    remain when the centerline estimate is sparse or traffic mix changes. This
    correction estimates the median ``raw_y - mapped_lane_center_y`` per lane and
    longitudinal bin, subtracts only that shared lane bias, and preserves each
    vehicle's within-lane lateral deviation. Optional clipping keeps centers
    inside the mapped lane so replay vehicles do not go off-road automatically.
    """
    out = df.copy()
    out["y_curved_raw"] = pd.to_numeric(out[y_col], errors="coerce")

    sample_step = max(1, int(sample_step))
    bin_size = max(1.0, float(bin_size_m))
    sample = out.iloc[::sample_step].dropna(subset=[x_col, y_col, lane_col]).copy()
    if sample.empty:
        return out, pd.DataFrame(columns=["lane_id", "x_center", "y_bias"]), {
            "aligned_rows": 0,
            "clipped_rows": 0,
            "profile_rows": 0,
        }

    sample_x = pd.to_numeric(sample[x_col], errors="coerce").to_numpy(dtype=float)
    sample_lane = pd.to_numeric(sample[lane_col], errors="coerce").to_numpy(dtype=float)
    sample_y = pd.to_numeric(sample[y_col], errors="coerce").to_numpy(dtype=float)
    sample_target = target_japanese_lane_center_y_array(sample_x, sample_lane)
    finite_sample = (
        np.isfinite(sample_x)
        & np.isfinite(sample_y)
        & np.isfinite(sample_lane)
        & np.isfinite(sample_target)
    )
    sample = sample.loc[finite_sample].copy()
    if sample.empty:
        return out, pd.DataFrame(columns=["lane_id", "x_center", "y_bias"]), {
            "aligned_rows": 0,
            "clipped_rows": 0,
            "profile_rows": 0,
        }

    sample["_target_y"] = sample_target[finite_sample]
    sample["_lane_id"] = sample[lane_col].astype(int)
    sample["_lateral_residual"] = pd.to_numeric(sample[y_col], errors="coerce") - sample["_target_y"]

    bins = np.arange(
        0.0,
        float(np.nanmax(pd.to_numeric(sample[x_col], errors="coerce"))) + bin_size,
        bin_size,
        dtype=float,
    )
    if bins.size < 2:
        return out, pd.DataFrame(columns=["lane_id", "x_center", "y_bias"]), {
            "aligned_rows": 0,
            "clipped_rows": 0,
            "profile_rows": 0,
        }

    sample["_bin"] = pd.cut(sample[x_col], bins=bins, labels=False, include_lowest=True)
    profile = (
        sample.groupby(["_lane_id", "_bin"], observed=False)["_lateral_residual"]
        .median()
        .dropna()
        .to_frame("y_bias")
        .reset_index()
    )
    if profile.empty:
        return out, pd.DataFrame(columns=["lane_id", "x_center", "y_bias"]), {
            "aligned_rows": 0,
            "clipped_rows": 0,
            "profile_rows": 0,
        }

    profile["lane_id"] = profile["_lane_id"].astype(int)
    profile["x_center"] = bins[:-1][profile["_bin"].to_numpy(dtype=int)] + 0.5 * bin_size
    profile["y_bias"] = (
        profile.groupby("lane_id", group_keys=False)["y_bias"]
        .transform(lambda s: s.rolling(window=5, center=True, min_periods=1).median())
    )
    profile = profile.sort_values(["lane_id", "x_center"]).reset_index(drop=True)

    aligned_rows = 0
    clipped_rows = 0
    max_abs_lateral = None
    if max_abs_lateral_m is not None and float(max_abs_lateral_m) >= 0.0:
        max_abs_lateral = float(max_abs_lateral_m)

    all_lane_ids = pd.to_numeric(out[lane_col], errors="coerce")
    for lane_id, lane_profile in profile.groupby("lane_id", sort=True):
        row_mask = all_lane_ids == int(lane_id)
        if not row_mask.any():
            continue
        profile_x = lane_profile["x_center"].to_numpy(dtype=float)
        profile_bias = lane_profile["y_bias"].to_numpy(dtype=float)
        if profile_x.size == 0:
            continue

        row_x = pd.to_numeric(out.loc[row_mask, x_col], errors="coerce").to_numpy(dtype=float)
        row_y = pd.to_numeric(out.loc[row_mask, y_col], errors="coerce").to_numpy(dtype=float)
        correction = np.interp(
            row_x,
            profile_x,
            profile_bias,
            left=float(profile_bias[0]),
            right=float(profile_bias[-1]),
        )
        aligned_y = row_y - correction

        if max_abs_lateral is not None:
            target_y = target_japanese_lane_center_y_array(
                row_x,
                np.full_like(row_x, float(lane_id), dtype=float),
            )
            residual = aligned_y - target_y
            clipped = np.isfinite(residual) & (np.abs(residual) > max_abs_lateral)
            clipped_rows += int(clipped.sum())
            aligned_y = np.where(
                np.isfinite(target_y),
                target_y + np.clip(residual, -max_abs_lateral, max_abs_lateral),
                aligned_y,
            )

        out.loc[row_mask, y_col] = aligned_y
        aligned_rows += int(row_mask.sum())

    report_profile = profile[["lane_id", "x_center", "y_bias"]].copy()
    summary = {
        "aligned_rows": int(aligned_rows),
        "clipped_rows": int(clipped_rows),
        "profile_rows": int(len(report_profile)),
    }
    return out, report_profile, summary


def clip_japanese_lateral_to_road(
    df: pd.DataFrame,
    *,
    x_col: str = "x_smooth",
    y_col: str = "y_smooth",
    lane_col: str = "traffic_lane",
    max_abs_lateral_m: float = 1.65,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clip smoothed Japanese vehicle centers inside their mapped road lane."""
    out = df.copy()
    if max_abs_lateral_m is None or float(max_abs_lateral_m) < 0.0:
        return out, {"checked_rows": 0, "clipped_rows": 0}

    x_values = pd.to_numeric(out[x_col], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(out[y_col], errors="coerce").to_numpy(dtype=float)
    lane_ids = pd.to_numeric(out[lane_col], errors="coerce").to_numpy(dtype=float)
    target_y = target_japanese_lane_center_y_array(x_values, lane_ids)
    residual = y_values - target_y
    valid = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(target_y)
    clipped = valid & (np.abs(residual) > float(max_abs_lateral_m))
    y_values = np.where(
        clipped,
        target_y + np.clip(residual, -float(max_abs_lateral_m), float(max_abs_lateral_m)),
        y_values,
    )
    out[y_col] = y_values
    return out, {"checked_rows": int(valid.sum()), "clipped_rows": int(clipped.sum())}


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

    df["datetime_jst"] = parse_datetime_jst(df["datetime_jst"])
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

        window_elapsed = (group["datetime_jst"] - window_start).dt.total_seconds()
        group = group.copy()
        group["_frame_idx"] = np.rint(
            pd.to_numeric(window_elapsed, errors="coerce").to_numpy(dtype=float) / nominal_dt
        ).astype(int)
        group = group[
            (group["_frame_idx"] >= 0)
            & (group["_frame_idx"] < expected_frames)
        ].copy()

        for veh_id, veh_group in group.groupby("vehicle_id", sort=True):
            ordered = (
                veh_group.sort_values("datetime_jst")
                .drop_duplicates("_frame_idx", keep="last")
                .copy()
            )
            if ordered.empty:
                continue

            traj = np.zeros((expected_frames, 4), dtype=float)
            frame_idx = ordered["_frame_idx"].to_numpy(dtype=int)
            values = ordered[
                ["x_smooth", "y_smooth", "v_smooth", "traffic_lane"]
            ].to_numpy(dtype=float)
            values[:, 3] = ordered["traffic_lane"].to_numpy(dtype=np.int64)
            traj[frame_idx] = values

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
    loaded_start = df["datetime_jst"].min()
    loaded_end = df["datetime_jst"].max()
    print(
        f"Loaded {len(df)} cleaned rows across {df['vehicle_id'].nunique()} vehicles "
        f"for JST window {args.start_clock} to {args.end_clock}"
    )
    print(f"Loaded data time range: {loaded_start} to {loaded_end}")

    df = add_vehicle_width(df)
    df = df.dropna(subset=["vehicle_length", "vehicle_width", "velocity"]).copy()
    allowed_lane_ids = {int(lane_id) for lane_id in args.allowed_lane_ids}
    if allowed_lane_ids:
        before_lane_filter = len(df)
        df = df[df["traffic_lane"].astype(int).isin(allowed_lane_ids)].copy()
        print(
            "Filtered unsupported Japanese lane ids: "
            f"kept={len(df)}, dropped={before_lane_filter - len(df)}, "
            f"allowed={sorted(allowed_lane_ids)}"
        )

    df_curved, centerline_df = estimate_curvature_remap(
        df,
        lanes=list(args.centerline_lanes),
        bin_size_m=args.bin_size_m,
    )
    df["x_curved"] = pd.to_numeric(df_curved["x_curved"], errors="coerce")
    df["y_curved"] = pd.to_numeric(df_curved["y_curved"], errors="coerce")
    df = df.dropna(subset=["x_curved", "y_curved"]).copy()
    if args.disable_lane_center_alignment:
        alignment_profile = pd.DataFrame(columns=["lane_id", "x_center", "y_bias"])
        alignment_summary = {"aligned_rows": 0, "clipped_rows": 0, "profile_rows": 0}
    else:
        max_lane_lateral_m = (
            None
            if args.max_lane_lateral_m is None or float(args.max_lane_lateral_m) < 0.0
            else float(args.max_lane_lateral_m)
        )
        df, alignment_profile, alignment_summary = align_lanes_to_japanese_road(
            df,
            x_col="x_curved",
            y_col="y_curved",
            lane_col="traffic_lane",
            bin_size_m=args.curved_flatten_bin_size_m,
            sample_step=args.curved_flatten_sample_step,
            max_abs_lateral_m=max_lane_lateral_m,
        )

    print(
        "Centerline estimated with "
        f"{len(centerline_df)} samples using lanes {list(args.centerline_lanes)}"
    )
    print(
        "Applied lane-aware Japanese road recentering with "
        f"{len(alignment_profile)} bias samples; "
        f"aligned_rows={alignment_summary['aligned_rows']}, "
        f"clipped_rows={alignment_summary['clipped_rows']}"
    )

    df_smooth = smooth_vehicle_trajectories(df)
    if not args.disable_lane_center_alignment and max_lane_lateral_m is not None:
        df_smooth, smooth_clip_summary = clip_japanese_lateral_to_road(
            df_smooth,
            x_col="x_smooth",
            y_col="y_smooth",
            lane_col="traffic_lane",
            max_abs_lateral_m=max_lane_lateral_m,
        )
        print(
            "Applied post-smoothing lane lateral guard: "
            f"checked_rows={smooth_clip_summary['checked_rows']}, "
            f"clipped_rows={smooth_clip_summary['clipped_rows']}"
        )
    veh_ids_all, traj_all = build_episode_dicts(
        df_smooth=df_smooth,
        window_sec=args.window_sec,
        presence_ratio_threshold=args.presence_ratio_threshold,
    )

    episode_keys = sorted(traj_all.keys())
    if not episode_keys:
        raise RuntimeError("No episodes were created from the filtered Japanese dataset.")

    print(f"Episode key range: {episode_keys[0]} to {episode_keys[-1]}")

    if args.min_episode_count and len(episode_keys) < int(args.min_episode_count):
        raise RuntimeError(
            f"Built only {len(episode_keys)} episodes, fewer than "
            f"--min_episode_count={args.min_episode_count}. "
            f"Loaded data covered {loaded_start} to {loaded_end}; refusing to save."
        )

    required_episodes = [str(episode_name) for episode_name in args.require_episode]
    missing_required = [
        episode_name
        for episode_name in required_episodes
        if episode_name not in traj_all
    ]
    if missing_required:
        raise RuntimeError(
            "Required episode(s) missing before save: "
            f"{missing_required}. Loaded data covered {loaded_start} to {loaded_end}; "
            f"built episode range {episode_keys[0]} to {episode_keys[-1]}."
        )

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
