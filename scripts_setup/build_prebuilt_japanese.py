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
import datetime as dt
import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from highway_env.data.curvature_remap import estimate_curvature_remap
from highway_env.ngsim_utils.data.trajectory_gen import (
    trajectory_has_min_continuous_occupancy,
    trajectory_smoothing,
)
from highway_env.ngsim_utils.road.gen_road import (
    JAPANESE_SOURCE_PREPROCESSING_CONTRACT,
    JAPANESE_SOURCE_ROAD_CONTRACT,
    create_japanese_road,
)
from highway_env.ngsim_utils.road.lane_mapping import target_lane_index_from_lane_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("VFI_DATA_ROOT", PROJECT_ROOT / "highway_env" / "data")
).expanduser().resolve()
RAW_DATA_DIR = DATA_ROOT / "raw"


MORINOMIYA_START_JST = pd.Timestamp("2020-01-01 09:00:00", tz="Asia/Tokyo")
JST_TIMEZONE = "Asia/Tokyo"
SOURCE_TIME_SEPARATOR = "__t"
SOURCE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SOURCE_PRESERVING_PREPROCESSING_CONTRACT = JAPANESE_SOURCE_PREPROCESSING_CONTRACT


def source_bound_episode_key(source_file: str, window_start: pd.Timestamp) -> str:
    """Return an episode key that cannot alias a different source recording."""
    source = str(source_file).strip()
    if not SOURCE_NAME_PATTERN.fullmatch(source):
        raise ValueError(
            "Morinomiya source_file must be a nonempty filesystem-safe identifier; "
            f"got {source_file!r}."
        )
    return f"{source}{SOURCE_TIME_SEPARATOR}{int(window_start.timestamp() * 1000)}"


def episode_source_name(episode_key: str) -> str:
    """Extract the source recording bound into a Japanese episode key."""
    source, separator, timestamp = str(episode_key).rpartition(SOURCE_TIME_SEPARATOR)
    if not separator or not source or not timestamp.isdigit():
        raise ValueError(f"Episode key is not source-bound: {episode_key!r}")
    return source


def episode_time_ms(episode_key: str) -> int:
    """Extract a millisecond timestamp from a source-bound or legacy key."""
    key = str(episode_key)
    if SOURCE_TIME_SEPARATOR in key:
        _source, _separator, timestamp = key.rpartition(SOURCE_TIME_SEPARATOR)
    elif key.startswith("t"):
        timestamp = key[1:]
    else:
        timestamp = ""
    if not timestamp.isdigit():
        raise ValueError(f"Episode key does not encode a millisecond timestamp: {episode_key!r}")
    return int(timestamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Japanese prebuilt trajectory caches from filtered Morinomiya data."
    )
    parser.add_argument(
        "--input_npy",
        default=str(RAW_DATA_DIR / "morinomiya_filtered_800.npy"),
        help="Filtered Morinomiya .npy produced by the raw_data preprocessing pipeline.",
    )
    parser.add_argument(
        "--episode_root",
        default=str(DATA_ROOT / "processed_20s"),
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
        "--preprocessing-contract",
        choices=("legacy_warped_v1", SOURCE_PRESERVING_PREPROCESSING_CONTRACT),
        default="legacy_warped_v1",
        help=(
            "Use the legacy curved-coordinate/recentering path only as a negative "
            "control. The v3 contract preserves source trajectories under one rigid "
            "shared SE(2) registration for all recordings and fits the road from train rows."
        ),
    )
    parser.add_argument(
        "--minimum_vehicle_center_separation_m",
        type=float,
        default=1.0,
        help=(
            "Quarantine both source trajectories if their recorded centers are "
            "closer than this physically impossible distance at the same source "
            "timestamp. Use a negative value to disable."
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
    return parsed.dt.tz_convert(JST_TIMEZONE).astype("datetime64[ns, Asia/Tokyo]")


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
    parsed = MORINOMIYA_START_JST.normalize() + pd.to_timedelta(
        total_ms.where(valid),
        unit="ms",
    )
    return parsed.astype("datetime64[ns, Asia/Tokyo]")


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
    if "source_file" not in field_names:
        raise ValueError(
            "Filtered Morinomiya input has no source_file field. Source provenance "
            "is required to prevent different recordings from sharing an episode."
        )

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
        "kilopost",
        "x_m",
        "y_m",
    ]:
        if col in field_names:
            numeric_data[col] = pd.to_numeric(
                pd.Series(arr[col], copy=False),
                errors="coerce",
            ).to_numpy()

    required_mask = (
        np.isfinite(numeric_data["vehicle_id"])
        & np.isfinite(numeric_data["datetime"])
        & np.isfinite(numeric_data["traffic_lane"])
    )
    source_files = np.asarray(arr["source_file"]).astype(str, copy=False)

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

    missing_source = mask & (np.char.strip(source_files) == "")
    if np.any(missing_source):
        raise ValueError(
            f"{int(missing_source.sum())} retained Morinomiya rows have no source_file; "
            "refusing to construct ambiguous episodes."
        )

    # Materialize only the post-crop subset needed downstream.
    data = {
        "vehicle_id": numeric_data["vehicle_id"][mask].astype(np.int64, copy=False),
        "datetime": numeric_data["datetime"][mask].astype(np.int64, copy=False),
        "datetime_jst": datetime_jst[mask].reset_index(drop=True),
        "traffic_lane": numeric_data["traffic_lane"][mask].astype(np.int64, copy=False),
        "source_file": source_files[mask],
        "x_m": x_m[mask],
        "y_m": y_m[mask],
    }
    for col in ["vehicle_type", "velocity", "vehicle_length", "detected_flag", "kilopost"]:
        if col in numeric_data:
            data[col] = numeric_data[col][mask]

    df = pd.DataFrame(data)
    del arr
    gc.collect()

    gc.collect()

    return df.sort_values(["source_file", "datetime", "vehicle_id"]).reset_index(drop=True)


def quarantine_near_coincident_trajectories(
    df: pd.DataFrame,
    *,
    minimum_center_separation_m: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove source trajectories participating in an impossible co-location.

    The gate uses original local XY coordinates before curvature remapping,
    recentering, smoothing, or episode slicing. Both track IDs are quarantined
    for the full source recording so a fragmented/duplicated physical track
    cannot leak across a later split boundary.
    """
    threshold = float(minimum_center_separation_m)
    if threshold < 0.0:
        return df.copy(), {
            "enabled": False,
            "minimum_center_separation_m": threshold,
            "pair_count": 0,
            "quarantined_identity_count": 0,
            "removed_row_count": 0,
            "sources": {},
            "pairs": [],
        }
    if threshold <= 0.0:
        raise ValueError("minimum_center_separation_m must be positive or negative to disable.")

    required = {"source_file", "vehicle_id", "datetime", "x_m", "y_m"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Near-coincident trajectory audit lacks columns: {missing}")

    ordered = df.sort_values(["source_file", "datetime", "vehicle_id"]).reset_index(
        drop=True
    )
    sources = ordered["source_file"].astype(str).to_numpy()
    vehicle_ids = ordered["vehicle_id"].to_numpy(dtype=np.int64)
    timestamps = ordered["datetime"].to_numpy(dtype=np.int64)
    x_values = ordered["x_m"].to_numpy(dtype=float)
    y_values = ordered["y_m"].to_numpy(dtype=float)

    boundaries = np.flatnonzero(
        (sources[1:] != sources[:-1]) | (timestamps[1:] != timestamps[:-1])
    ) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(ordered)])))

    pair_records: dict[tuple[str, int, int], dict[str, object]] = {}
    quarantined_by_source: dict[str, set[int]] = {}
    for start, end in zip(starts.tolist(), ends.tolist()):
        x_order = np.argsort(x_values[start:end], kind="stable") + start
        for position, left_index in enumerate(x_order):
            right_position = position + 1
            while (
                right_position < len(x_order)
                and x_values[x_order[right_position]] - x_values[left_index] < threshold
            ):
                right_index = int(x_order[right_position])
                left_id = int(vehicle_ids[left_index])
                right_id = int(vehicle_ids[right_index])
                if left_id != right_id:
                    distance = float(
                        np.hypot(
                            x_values[right_index] - x_values[left_index],
                            y_values[right_index] - y_values[left_index],
                        )
                    )
                    if distance < threshold:
                        source = str(sources[left_index])
                        first_id, second_id = sorted((left_id, right_id))
                        key = (source, first_id, second_id)
                        record = pair_records.setdefault(
                            key,
                            {
                                "source_file": source,
                                "vehicle_id_a": first_id,
                                "vehicle_id_b": second_id,
                                "event_count": 0,
                                "minimum_center_distance_m": float("inf"),
                                "first_timestamp": int(timestamps[left_index]),
                                "last_timestamp": int(timestamps[left_index]),
                            },
                        )
                        record["event_count"] = int(record["event_count"]) + 1
                        record["minimum_center_distance_m"] = min(
                            float(record["minimum_center_distance_m"]),
                            distance,
                        )
                        record["first_timestamp"] = min(
                            int(record["first_timestamp"]),
                            int(timestamps[left_index]),
                        )
                        record["last_timestamp"] = max(
                            int(record["last_timestamp"]),
                            int(timestamps[left_index]),
                        )
                        quarantined_by_source.setdefault(source, set()).update(
                            (first_id, second_id)
                        )
                right_position += 1

    quarantined_keys = {
        (source, int(vehicle_id))
        for source, vehicle_ids_for_source in quarantined_by_source.items()
        for vehicle_id in vehicle_ids_for_source
    }
    remove_mask = np.fromiter(
        (
            (str(source), int(vehicle_id)) in quarantined_keys
            for source, vehicle_id in zip(sources, vehicle_ids)
        ),
        dtype=bool,
        count=len(ordered),
    )
    retained = ordered.loc[~remove_mask].copy()
    if retained.empty:
        raise RuntimeError("Near-coincident trajectory quarantine removed every row.")
    source_report = {}
    for source in sorted(set(sources)):
        source_mask = sources == source
        source_quarantined = sorted(quarantined_by_source.get(source, set()))
        source_report[source] = {
            "input_identity_count": int(np.unique(vehicle_ids[source_mask]).size),
            "quarantined_identity_count": len(source_quarantined),
            "quarantined_vehicle_ids": source_quarantined,
            "input_row_count": int(source_mask.sum()),
            "removed_row_count": int(
                (source_mask & np.isin(vehicle_ids, source_quarantined)).sum()
            ),
        }
    report = {
        "enabled": True,
        "identity_key_format": "<source_file>,<vehicle_id>",
        "minimum_center_separation_m": threshold,
        "pair_count": len(pair_records),
        "quarantined_identity_count": len(quarantined_keys),
        "removed_row_count": int(len(ordered) - len(retained)),
        "sources": source_report,
        "pairs": [pair_records[key] for key in sorted(pair_records)],
    }
    return retained.reset_index(drop=True), report


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
    if "source_file" not in df.columns:
        raise ValueError("source_file is required before smoothing Japanese trajectories.")
    smoothed_groups: list[pd.DataFrame] = []

    for (_source_file, _vehicle_id), group in df.groupby(
        ["source_file", "vehicle_id"], sort=True
    ):
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


def source_training_row_mask(
    df: pd.DataFrame,
    *,
    window_sec: int,
    val_ratio: float,
    test_ratio: float,
) -> np.ndarray:
    """Return the predeclared chronological train section within each source."""
    if not 0.0 <= float(val_ratio) < 1.0 or not 0.0 <= float(test_ratio) < 1.0:
        raise ValueError("Validation/test ratios must be in [0, 1).")
    if float(val_ratio) + float(test_ratio) >= 1.0:
        raise ValueError("Validation and test ratios must sum to less than one.")
    result = np.zeros(len(df), dtype=bool)
    timestamps = parse_datetime_jst(df["datetime_jst"])
    for source, indices in df.groupby("source_file", sort=True).groups.items():
        row_indices = np.asarray(indices, dtype=np.int64)
        source_times = timestamps.iloc[row_indices]
        start = source_times.min().floor(f"{int(window_sec)}s")
        windows = np.floor(
            (source_times - start).dt.total_seconds().to_numpy(dtype=float)
            / float(window_sec)
        ).astype(np.int64)
        unique_windows = np.unique(windows)
        train_windows, _val_windows, _test_windows = split_episode_keys(
            [f"t{int(value)}" for value in unique_windows],
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
        )
        selected = {int(value[1:]) for value in train_windows}
        if not selected:
            raise RuntimeError(f"Source {source!r} has no training windows for registration.")
        result[row_indices] = np.isin(windows, np.asarray(sorted(selected), dtype=np.int64))
    return result


def rigid_register_shared_coordinates(
    df: pd.DataFrame,
    *,
    fit_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply one train-fitted rigid SE(2) registration shared by every source.

    Morinomiya longitude/latitude values are already WGS84 coordinates on the
    same physical road.  Independent source translations would manufacture
    multiple displaced copies of that road, so the transform is fitted once
    and applied to all rows.  A shared rigid transform preserves every
    intra-source and inter-source Euclidean distance.
    """
    if len(fit_mask) != len(df):
        raise ValueError("Registration fit mask length does not match Japanese rows.")
    out = df.copy()
    fit_mask = np.asarray(fit_mask, dtype=bool)
    fit_rows = out.loc[
        fit_mask & out["traffic_lane"].astype(int).isin((1, 2)),
        ["source_file", "x_m", "y_m"],
    ].copy()
    fit_counts = {
        str(source): int(count)
        for source, count in fit_rows.groupby("source_file", sort=True).size().items()
    }
    all_sources = sorted(out["source_file"].astype(str).unique().tolist())
    missing = [source for source in all_sources if fit_counts.get(source, 0) < 100]
    if missing:
        raise RuntimeError(
            "Shared Japanese registration lacks 100 mainline train rows for sources: "
            f"{missing}; counts={fit_counts}."
        )
    fit_xy = fit_rows[["x_m", "y_m"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(fit_xy)):
        raise ValueError("Japanese shared registration coordinates are non-finite.")
    low_x, high_x = np.quantile(fit_xy[:, 0], [0.05, 0.95])
    low = np.median(fit_xy[fit_xy[:, 0] <= low_x], axis=0)
    high = np.median(fit_xy[fit_xy[:, 0] >= high_x], axis=0)
    direction = high - low
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm < 100.0:
        raise RuntimeError("Japanese shared registration has insufficient road span.")
    angle = float(np.arctan2(direction[1], direction[0]))
    cosine = float(np.cos(-angle))
    sine = float(np.sin(-angle))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    all_xy = out[["x_m", "y_m"]].to_numpy(dtype=float)
    rotated = all_xy @ rotation.T
    fit_rotated = fit_xy @ rotation.T
    x_origin = float(np.quantile(fit_rotated[:, 0], 0.005))
    y_origin = float(np.median(fit_rotated[:, 1]))
    registered = rotated - np.asarray([x_origin, y_origin], dtype=float)
    out["x_registered"] = registered[:, 0]
    out["y_registered"] = registered[:, 1]
    if not np.all(np.isfinite(out[["x_registered", "y_registered"]].to_numpy(dtype=float))):
        raise RuntimeError("Rigid Japanese registration left non-finite output coordinates.")
    receipt: dict[str, object] = {
        "fit_split": "train",
        "fit_sources": all_sources,
        "fit_row_count": int(len(fit_rows)),
        "fit_rows_by_source": fit_counts,
        "rotation_rad": float(-angle),
        "translation_after_rotation_m": [-x_origin, -y_origin],
        "transform_type": "rigid_se2",
        "transform_scope": "one_shared_transform_all_sources",
        "preserves_inter_source_geometry": True,
        "lane_specific_transform": False,
        "nonlinear_warp": False,
    }
    return out, receipt


def audit_registered_kilopost(
    df: pd.DataFrame,
    *,
    fit_mask: np.ndarray,
) -> dict[str, object]:
    """Audit registered x against the provider's road-distance coordinate.

    Kilopost is retained as provenance only.  It is not written into the four-
    column simulator trajectory and is never exposed to the policy.
    """
    required = {"source_file", "kilopost", "x_registered"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Japanese kilopost audit lacks columns: {missing}")
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if len(fit_mask) != len(df):
        raise ValueError("Kilopost audit fit mask length mismatch.")
    kilopost = pd.to_numeric(df["kilopost"], errors="coerce").to_numpy(dtype=float)
    x_registered = pd.to_numeric(
        df["x_registered"], errors="coerce"
    ).to_numpy(dtype=float)
    finite = np.isfinite(kilopost) & np.isfinite(x_registered)
    fit = finite & fit_mask
    if np.count_nonzero(fit) < 100:
        raise RuntimeError("Japanese kilopost audit has insufficient train support.")
    offset = float(np.median(kilopost[fit] - x_registered[fit]))

    def summarize(mask: np.ndarray) -> dict[str, object]:
        selected = finite & mask
        residual = kilopost[selected] - x_registered[selected] - offset
        return {
            "row_count": int(np.count_nonzero(selected)),
            "absolute_error_q50_m": float(np.quantile(np.abs(residual), 0.50)),
            "absolute_error_q95_m": float(np.quantile(np.abs(residual), 0.95)),
            "absolute_error_q99_m": float(np.quantile(np.abs(residual), 0.99)),
            "pearson_correlation": float(
                np.corrcoef(kilopost[selected], x_registered[selected])[0, 1]
            ),
        }

    by_source = {}
    sources = df["source_file"].astype(str).to_numpy()
    for source in sorted(set(sources)):
        by_source[source] = summarize(sources == source)
    return {
        "fit_split": "train",
        "provider_field": "kilopost",
        "provider_definition": "distance_from_starting_point_of_expressway_route_m",
        "policy_visible": False,
        "fitted_kilopost_minus_registered_x_offset_m": offset,
        "train": summarize(fit_mask),
        "all_retained_rows": summarize(np.ones(len(df), dtype=bool)),
        "by_source_all_retained_rows": by_source,
    }


def source_preserving_trajectory_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Bind raw registered x/y/speed to the existing episode writer columns."""
    out = df.copy()
    speed = pd.to_numeric(out["velocity"], errors="coerce").to_numpy(dtype=float)
    invalid_speed = ~np.isfinite(speed) | (speed < 0.0)
    invalid_coordinate = ~np.all(
        np.isfinite(out[["x_registered", "y_registered"]].to_numpy(dtype=float)),
        axis=1,
    )
    rejected = invalid_speed | invalid_coordinate
    report = {
        "input_rows": int(len(out)),
        "negative_speed_rows": int(np.count_nonzero(np.isfinite(speed) & (speed < 0.0))),
        "nonfinite_speed_rows": int(np.count_nonzero(~np.isfinite(speed))),
        "nonfinite_coordinate_rows": int(np.count_nonzero(invalid_coordinate)),
        "rejected_rows": int(np.count_nonzero(rejected)),
    }
    out = out.loc[~rejected].copy()
    out["x_smooth"] = out["x_registered"].astype(float)
    out["y_smooth"] = out["y_registered"].astype(float)
    out["v_smooth"] = pd.to_numeric(out["velocity"], errors="raise").astype(float)
    report["output_rows"] = int(len(out))
    return out, report


def _binned_lane_centerline(
    rows: pd.DataFrame,
    *,
    lane_id: int,
    bin_size_m: float,
    endpoint_padding_m: float = 0.0,
) -> list[list[float]]:
    lane = rows[rows["traffic_lane"].astype(int) == int(lane_id)][
        ["x_registered", "y_registered"]
    ].copy()
    if len(lane) < 100:
        raise RuntimeError(f"Lane {lane_id} has insufficient train rows for road fitting.")
    x = lane["x_registered"].to_numpy(dtype=float)
    # The upstream CSV ingestion already enforces finite physical support.
    # Quantile trimming here silently shortened the road and made otherwise
    # centered validation vehicles appear off-road near both endpoints.
    lower = float(np.min(x))
    upper = float(np.max(x))
    bins = np.arange(lower, upper + float(bin_size_m), float(bin_size_m))
    if len(bins) < 3:
        raise RuntimeError(f"Lane {lane_id} has insufficient longitudinal road support.")
    lane["_bin"] = pd.cut(
        lane["x_registered"], bins=bins, labels=False, include_lowest=True
    )
    profile = (
        lane.groupby("_bin", observed=False)[["x_registered", "y_registered"]]
        .median()
        .dropna()
        .sort_values("x_registered")
    )
    profile = profile.loc[
        ~profile["x_registered"].duplicated(keep="first")
    ]
    if len(profile) < 3:
        raise RuntimeError(f"Lane {lane_id} road fit produced fewer than three points.")
    values = profile.to_numpy(dtype=float)
    if values[0, 0] > lower:
        values = np.vstack(([lower, values[0, 1]], values))
    if values[-1, 0] < upper:
        values = np.vstack((values, [upper, values[-1, 1]]))
    padding = float(endpoint_padding_m)
    if padding < 0.0:
        raise ValueError("Japanese road endpoint padding cannot be negative.")
    if padding > 0.0:
        start_slope = float(
            (values[1, 1] - values[0, 1]) / (values[1, 0] - values[0, 0])
        )
        end_slope = float(
            (values[-1, 1] - values[-2, 1])
            / (values[-1, 0] - values[-2, 0])
        )
        padded_start_x = float(values[0, 0] - padding)
        padded_end_x = float(values[-1, 0] + padding)
        values = np.vstack(
            (
                [padded_start_x, values[0, 1] - padding * start_slope],
                values,
                [padded_end_x, values[-1, 1] + padding * end_slope],
            )
        )
    return values.tolist()


def _estimate_nominal_lane_width(
    train: pd.DataFrame,
    *,
    bin_size_m: float,
    quantile: float = 0.995,
) -> tuple[float, dict[str, object]]:
    """Estimate physical lane width from stable mainline lane separation.

    Median vehicle positions can sit inward of the painted lane centers.  The
    high, robust envelope of per-source binned lane-center separations recovers
    the nominal width without using validation/test rows or merge trajectories.
    """
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("Japanese lane-width quantile must lie in (0.5, 1).")
    samples = []
    counts: dict[str, int] = {}
    for source, source_rows in train.groupby("source_file", sort=True):
        mainline = source_rows[
            source_rows["traffic_lane"].astype(int).isin((1, 2))
        ].copy()
        mainline["_width_bin"] = np.floor(
            mainline["x_registered"].to_numpy(dtype=float) / float(bin_size_m)
        ).astype(np.int64)
        centers = mainline.groupby(
            ["_width_bin", "traffic_lane"], observed=False
        )["y_registered"].median().unstack()
        if 1 not in centers or 2 not in centers:
            raise RuntimeError(f"Source {source!r} lacks both mainline lanes.")
        separation = (centers[1] - centers[2]).abs().dropna().to_numpy(dtype=float)
        if len(separation) < 20:
            raise RuntimeError(
                f"Source {source!r} has insufficient lane-width bins: {len(separation)}."
            )
        samples.extend(separation.tolist())
        counts[str(source)] = int(len(separation))
    values = np.asarray(samples, dtype=float)
    lane_width = float(np.quantile(values, float(quantile)))
    if not 2.5 <= lane_width <= 5.0:
        raise RuntimeError(f"Train-fitted Japanese lane width is implausible: {lane_width:g} m.")
    receipt = {
        "contract": "per_source_binned_mainline_center_separation_q995_v1",
        "fit_split": "train",
        "merge_rows_used": False,
        "validation_or_test_rows_used": False,
        "bin_size_m": float(bin_size_m),
        "quantile": float(quantile),
        "sample_count": int(len(values)),
        "sample_counts_by_source": counts,
        "sample_quantiles_m": {
            str(value): float(np.quantile(values, value))
            for value in (0.5, 0.9, 0.95, 0.99, 0.995)
        },
        "estimated_lane_width_m": lane_width,
    }
    return lane_width, receipt


def fit_source_derived_japanese_road(
    df: pd.DataFrame,
    *,
    fit_mask: np.ndarray,
    bin_size_m: float = 10.0,
) -> dict[str, object]:
    """Fit a curved two-mainline-plus-merge road without changing any trajectory."""
    train = df.loc[np.asarray(fit_mask, dtype=bool)].copy()
    train_lane_row_counts = {
        str(lane_id): int(
            np.count_nonzero(train["traffic_lane"].astype(int).to_numpy() == lane_id)
        )
        for lane_id in (1, 2, 3)
    }
    insufficient = {
        lane_id: count
        for lane_id, count in train_lane_row_counts.items()
        if count < 100
    }
    if insufficient:
        raise RuntimeError(
            "Train-only Japanese road fitting lacks lane support: "
            f"counts={train_lane_row_counts}, required_per_lane=100."
        )
    if "vehicle_length" in train:
        vehicle_lengths = pd.to_numeric(
            train["vehicle_length"], errors="coerce"
        ).to_numpy(dtype=float)
        vehicle_lengths = vehicle_lengths[
            np.isfinite(vehicle_lengths) & (vehicle_lengths > 0.0)
        ]
        endpoint_padding = 0.5 * float(np.quantile(vehicle_lengths, 0.999))
    else:
        endpoint_padding = 0.0
    lane_centerlines = {
        str(lane_id): _binned_lane_centerline(
            train,
            lane_id=lane_id,
            bin_size_m=float(bin_size_m),
            endpoint_padding_m=endpoint_padding,
        )
        for lane_id in (1, 2, 3)
    }
    main_1 = np.asarray(lane_centerlines["1"], dtype=float)
    main_2 = np.asarray(lane_centerlines["2"], dtype=float)
    merge = np.asarray(lane_centerlines["3"], dtype=float)
    x_start = min(float(main_1[0, 0]), float(main_2[0, 0]))
    x_end = max(float(main_1[-1, 0]), float(main_2[-1, 0]))
    merge_start = float(np.quantile(merge[:, 0], 0.25))
    merge_end = float(merge[-1, 0])
    if not x_start < merge_start < merge_end < x_end:
        raise RuntimeError("Train-fitted Japanese merge bounds are not ordered.")
    lane_width, lane_width_receipt = _estimate_nominal_lane_width(
        train,
        bin_size_m=float(bin_size_m),
    )
    return {
        "schema_version": 1,
        "contract": JAPANESE_SOURCE_ROAD_CONTRACT,
        "coordinate_contract": SOURCE_PRESERVING_PREPROCESSING_CONTRACT,
        "fit_split": "train",
        "test_rows_used": False,
        "fit_sources": sorted(train["source_file"].astype(str).unique().tolist()),
        "fit_row_count": int(len(train)),
        "fit_lane_row_counts": train_lane_row_counts,
        "lane_width_m": lane_width,
        "lane_width_estimator": lane_width_receipt,
        "centerline_endpoint_padding_m": endpoint_padding,
        "centerline_endpoint_padding_contract": (
            "half_train_vehicle_length_q999_for_full_recorded_footprint_v1"
        ),
        "x_start_m": x_start,
        "x_end_m": x_end,
        "merge_start_x_m": merge_start,
        "merge_end_x_m": merge_end,
        "lane_centerlines": lane_centerlines,
        "trajectory_coordinates_modified_to_fit_road": False,
    }


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
        "source_file",
        "vehicle_id",
        "vehicle_length",
        "vehicle_width",
        "x_smooth",
        "y_smooth",
        "v_smooth",
        "traffic_lane",
        "kilopost",
        "detected_flag",
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
        "kilopost",
        "detected_flag",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_file"] = df["source_file"].astype(str).str.strip()
    if (df["source_file"] == "").any():
        raise ValueError("Japanese trajectory rows contain an empty source_file.")
    df["datetime_jst"] = parse_datetime_jst(df["datetime_jst"])
    df = df.dropna(subset=["datetime_jst"] + numeric_cols).copy()

    if df.empty:
        raise ValueError("No valid rows remain after smoothing and cleaning.")

    df["vehicle_id"] = df["vehicle_id"].astype(np.int64)
    df["traffic_lane"] = df["traffic_lane"].astype(np.int64)
    veh_ids_by_episode: dict[str, list[np.int64]] = {}
    trajectories_by_episode: dict[str, dict[np.int64, dict[str, np.ndarray]]] = {}

    for source_file, source_rows in df.groupby("source_file", sort=True):
        source = str(source_file)
        if not SOURCE_NAME_PATTERN.fullmatch(source):
            raise ValueError(f"Unsafe or ambiguous source_file identifier: {source!r}")
        source_rows = source_rows.sort_values(["datetime_jst", "vehicle_id"]).copy()
        unique_times = pd.Series(
            source_rows["datetime_jst"].drop_duplicates().sort_values()
        )
        time_deltas = unique_times.diff().dropna().dt.total_seconds()
        if len(time_deltas) == 0:
            raise ValueError(f"Not enough timestamps to estimate sampling for {source!r}.")
        nominal_dt = float(time_deltas.median())
        if nominal_dt <= 0:
            raise ValueError(f"Estimated a non-positive sampling interval for {source!r}.")
        expected_frames = max(int(round(window_sec / nominal_dt)), 1)
        min_presence_frames = int(np.ceil(expected_frames * presence_ratio_threshold))

        print(
            f"{source}: nominal_dt={nominal_dt:.6f}s, "
            f"frames_per_window={expected_frames}, "
            f"minimum_presence_frames={min_presence_frames}"
        )

        start_time = source_rows["datetime_jst"].min().floor(f"{window_sec}s")
        elapsed_sec = (source_rows["datetime_jst"] - start_time).dt.total_seconds()
        source_rows["_window_idx"] = (elapsed_sec // window_sec).astype(int)
        source_rows["_window_start"] = start_time + pd.to_timedelta(
            source_rows["_window_idx"] * window_sec, unit="s"
        )

        for window_start, group in source_rows.groupby("_window_start", sort=True):
            key = source_bound_episode_key(source, window_start)
            if key in trajectories_by_episode:
                raise RuntimeError(f"Duplicate source-bound episode key: {key}")
            traj_dict: dict[np.int64, dict[str, np.ndarray]] = {}
            valid_ids: list[np.int64] = []

            window_elapsed = (group["datetime_jst"] - window_start).dt.total_seconds()
            group = group.copy()
            group["_frame_idx"] = np.rint(
                pd.to_numeric(window_elapsed, errors="coerce").to_numpy(dtype=float)
                / nominal_dt
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
                provider_observation_mask = np.full(expected_frames, -1, dtype=np.int8)
                frame_idx = ordered["_frame_idx"].to_numpy(dtype=int)
                values = ordered[
                    ["x_smooth", "y_smooth", "v_smooth", "traffic_lane"]
                ].to_numpy(dtype=float)
                values[:, 3] = ordered["traffic_lane"].to_numpy(dtype=np.int64)
                traj[frame_idx] = values
                detected = ordered["detected_flag"].to_numpy(dtype=np.int64)
                if not np.isin(detected, (0, 1)).all():
                    raise ValueError(
                        f"Invalid Morinomiya detected_flag: {source}/{veh_id}"
                    )
                provider_observation_mask[frame_idx] = detected.astype(np.int8)

                traj_dict[np.int64(veh_id)] = {
                    "source_file": source,
                    "length": np.float64(ordered["vehicle_length"].iloc[0]),
                    "width": np.float64(ordered["vehicle_width"].iloc[0]),
                    "trajectory": traj,
                    "provider_observation_mask": provider_observation_mask,
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

    ordered = sorted(episode_keys, key=lambda key: (episode_time_ms(key), str(key)))
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


def split_episode_keys_by_source(
    episode_keys: list[str],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[str], list[str], list[str]]:
    """Chronologically split every source session, then aggregate the splits."""
    by_source: dict[str, list[str]] = {}
    for key in episode_keys:
        by_source.setdefault(episode_source_name(key), []).append(str(key))
    if not by_source:
        raise ValueError("No source-bound Japanese episode keys were provided.")

    combined = {"train": [], "val": [], "test": []}
    for source, source_keys in sorted(by_source.items()):
        train_keys, val_keys, test_keys = split_episode_keys(
            source_keys,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        source_splits = {"train": train_keys, "val": val_keys, "test": test_keys}
        empty = [split for split, keys in source_splits.items() if not keys]
        if empty:
            raise RuntimeError(
                f"Source {source!r} has insufficient episode support; empty splits: {empty}."
            )
        for split, keys in source_splits.items():
            combined[split].extend(keys)

    return tuple(
        sorted(combined[split], key=lambda key: (episode_source_name(key), episode_time_ms(key)))
        for split in ("train", "val", "test")
    )


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

    df, quarantine_report = quarantine_near_coincident_trajectories(
        df,
        minimum_center_separation_m=args.minimum_vehicle_center_separation_m,
    )
    print(
        "Near-coincident raw trajectory quarantine: "
        f"pairs={quarantine_report['pair_count']}, "
        f"identities={quarantine_report['quarantined_identity_count']}, "
        f"rows={quarantine_report['removed_row_count']}"
    )

    road_geometry = None
    registration_receipt: dict[str, object] = {}
    kilopost_receipt: dict[str, object] | None = None
    source_state_rejections = {
        "input_rows": int(len(df)),
        "negative_speed_rows": 0,
        "nonfinite_speed_rows": 0,
        "nonfinite_coordinate_rows": 0,
        "rejected_rows": 0,
        "output_rows": int(len(df)),
    }
    if args.preprocessing_contract == SOURCE_PRESERVING_PREPROCESSING_CONTRACT:
        df = df.reset_index(drop=True)
        fit_mask = source_training_row_mask(
            df,
            window_sec=args.window_sec,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        df["_road_fit_train"] = fit_mask
        df, registration_receipt = rigid_register_shared_coordinates(
            df,
            fit_mask=fit_mask,
        )
        kilopost_receipt = audit_registered_kilopost(df, fit_mask=fit_mask)
        df_smooth, source_state_rejections = source_preserving_trajectory_rows(df)
        road_geometry = fit_source_derived_japanese_road(
            df_smooth,
            fit_mask=df_smooth["_road_fit_train"].to_numpy(dtype=bool),
            bin_size_m=args.bin_size_m,
        )
        alignment_summary = {"aligned_rows": 0, "clipped_rows": 0, "profile_rows": 0}
        smooth_clip_summary = {"checked_rows": 0, "clipped_rows": 0}
        print(
            "Applied source-preserving train-fitted shared rigid SE(2) registration; "
            "trajectory recentering=0, smoothing=0, clipping=0."
        )
    else:
        df_curved, centerline_df = estimate_curvature_remap(
            df,
            lanes=list(args.centerline_lanes),
            bin_size_m=args.bin_size_m,
        )
        df["x_curved"] = pd.to_numeric(df_curved["x_curved"], errors="coerce")
        df["y_curved"] = pd.to_numeric(df_curved["y_curved"], errors="coerce")
        df = df.dropna(subset=["x_curved", "y_curved"]).copy()
        max_lane_lateral_m = (
            None
            if args.max_lane_lateral_m is None or float(args.max_lane_lateral_m) < 0.0
            else float(args.max_lane_lateral_m)
        )
        if args.disable_lane_center_alignment:
            alignment_profile = pd.DataFrame(columns=["lane_id", "x_center", "y_bias"])
            alignment_summary = {"aligned_rows": 0, "clipped_rows": 0, "profile_rows": 0}
        else:
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
        smooth_clip_summary = {"checked_rows": 0, "clipped_rows": 0}
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

    train_keys, val_keys, test_keys = split_episode_keys_by_source(
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

    split_keys = {"train": train_keys, "val": val_keys, "test": test_keys}
    source_manifest: dict[str, dict[str, int]] = {}
    for source in sorted({episode_source_name(key) for key in episode_keys}):
        source_manifest[source] = {
            split: sum(episode_source_name(key) == source for key in keys)
            for split, keys in split_keys.items()
        }
    manifest = {
        "schema_version": 2,
        "contract": "source_bound_episode_key_and_trajectory_metadata_v1",
        "preprocessing_contract": str(args.preprocessing_contract),
        "policy_visible_future_rows": False
        if args.preprocessing_contract == SOURCE_PRESERVING_PREPROCESSING_CONTRACT
        else True,
        "trajectory_recentered_rows": int(alignment_summary["aligned_rows"]),
        "trajectory_clipped_rows": int(
            alignment_summary["clipped_rows"] + smooth_clip_summary["clipped_rows"]
        ),
        "trajectory_centered_smoothing_applied": bool(
            args.preprocessing_contract != SOURCE_PRESERVING_PREPROCESSING_CONTRACT
        ),
        "rigid_shared_registration": registration_receipt,
        "kilopost_registration_audit": kilopost_receipt,
        "provider_observation_metadata": {
            "field": "detected_flag",
            "meaning": {"1": "image_detected", "0": "source_interpolated", "-1": "absent"},
            "stored_per_vehicle_key": "provider_observation_mask",
            "policy_visible": False,
            "rows_removed_because_interpolated": 0,
        },
        "source_state_rejections": source_state_rejections,
        "road_geometry_contract": (
            road_geometry.get("contract") if road_geometry is not None else None
        ),
        "input_npy": str(Path(args.input_npy).expanduser().resolve()),
        "episode_key_format": "<source_file>__t<unix_time_ms>",
        "source_episode_counts_by_split": source_manifest,
        "total_episode_counts_by_split": {
            split: len(keys) for split, keys in split_keys.items()
        },
    }
    Path(out_dir, "SOURCE_SESSION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(out_dir, "QUARANTINED_TRAJECTORIES.json").write_text(
        json.dumps(quarantine_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if road_geometry is not None:
        Path(out_dir, "ROAD_GEOMETRY.json").write_text(
            json.dumps(road_geometry, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
