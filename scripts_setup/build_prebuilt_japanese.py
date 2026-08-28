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
import hashlib
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
from scipy.interpolate import LSQUnivariateSpline, PchipInterpolator, UnivariateSpline

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

# Digitized from the six labelled detail panels in
# Morinomiya_Lane-marker_Sign_J.pdf.  These are document drawing extents, not
# learned lane geometry and not policy inputs.  The actual continuous
# centerlines remain train-fitted from source coordinates and are independently
# checked against the provider kilopost field.
MORINOMIYA_SOURCE_DOCUMENT_SECTION_BANDS = (
    {"section_id": 1, "kilopost_start_m": 1_700.0, "kilopost_end_m": 2_200.0},
    {"section_id": 2, "kilopost_start_m": 2_300.0, "kilopost_end_m": 2_700.0},
    {"section_id": 3, "kilopost_start_m": 2_700.0, "kilopost_end_m": 3_200.0},
    {"section_id": 4, "kilopost_start_m": 3_200.0, "kilopost_end_m": 3_700.0},
    {"section_id": 5, "kilopost_start_m": 3_700.0, "kilopost_end_m": 4_200.0},
    {"section_id": 6, "kilopost_start_m": 4_200.0, "kilopost_end_m": 4_600.0},
)
MORINOMIYA_SOURCE_DOCUMENT_DETECTOR_KILPOSTS_M = (
    1_800.0,
    2_500.0,
    2_900.0,
    3_500.0,
    4_000.0,
    4_500.0,
)


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
        "--full-road",
        action="store_true",
        help=(
            "Disable the legacy 0-800 m crop and retain the complete recorded road. "
            "This is opt-in so existing Japanese artifacts remain reproducible."
        ),
    )
    parser.add_argument(
        "--four-sections",
        action="store_true",
        help=(
            "Build the shared train-fit registration input for the four independent "
            "Morinomiya runtime sections. This implies --full-road and deliberately "
            "suppresses a continuous RoadGeometryV3 artifact."
        ),
    )
    parser.add_argument(
        "--topology-source-sha256",
        default=None,
        help=(
            "SHA-256 of the source road-topology PDF. Required for --full-road so "
            "RoadGeometryV3 is bound to the digitized section/kilopost document."
        ),
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
        "--split-boundary-guard-sec",
        type=int,
        default=0,
        help=(
            "Remove this many seconds of complete episodes immediately after each "
            "chronological split boundary. Must be a multiple of --window_sec."
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


def source_row_split_masks(
    df: pd.DataFrame,
    *,
    window_sec: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    """Return predeclared chronological row masks without opening split metrics.

    This is the row-level counterpart of :func:`split_episode_keys_by_source`.
    The masks are useful while the rows are already in memory during production;
    callers must still avoid aggregating the sealed test mask.
    """
    if not 0.0 <= float(val_ratio) < 1.0 or not 0.0 <= float(test_ratio) < 1.0:
        raise ValueError("Validation/test ratios must be in [0, 1).")
    if float(val_ratio) + float(test_ratio) >= 1.0:
        raise ValueError("Validation and test ratios must sum to less than one.")
    result = {
        "train": np.zeros(len(df), dtype=bool),
        "validation": np.zeros(len(df), dtype=bool),
        "test": np.zeros(len(df), dtype=bool),
    }
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
        train, validation, test = split_episode_keys(
            [f"t{int(value)}" for value in unique_windows],
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
        )
        for split, keys in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            selected = np.asarray([int(value[1:]) for value in keys], dtype=np.int64)
            if selected.size == 0:
                raise RuntimeError(
                    f"Source {source!r} has no {split} windows for row splitting."
                )
            result[split][row_indices] = np.isin(windows, selected)
    assignment_count = sum(mask.astype(np.int8) for mask in result.values())
    if not np.all(assignment_count == 1):
        raise RuntimeError("Japanese source row split masks are not exhaustive/disjoint.")
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
    validation_mask: np.ndarray,
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
    validation_mask = np.asarray(validation_mask, dtype=bool)
    if len(fit_mask) != len(df) or len(validation_mask) != len(df):
        raise ValueError("Kilopost audit split mask length mismatch.")
    if np.any(fit_mask & validation_mask):
        raise ValueError("Kilopost train and validation masks overlap.")
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

    by_source_train: dict[str, object] = {}
    by_source_validation: dict[str, object] = {}
    sources = df["source_file"].astype(str).to_numpy()
    for source in sorted(set(sources)):
        by_source_train[source] = summarize((sources == source) & fit_mask)
        by_source_validation[source] = summarize(
            (sources == source) & validation_mask
        )
    return {
        "contract_id": "morinomiya_registered_kilopost_diagnostic_v2",
        "diagnostic_only": True,
        "qualification_gate": False,
        "fit_split": "train",
        "audit_split": "validation",
        "provider_field": "kilopost",
        "provider_definition": "distance_from_starting_point_of_expressway_route_m",
        "policy_visible": False,
        "fitted_kilopost_minus_registered_x_offset_m": offset,
        "train": summarize(fit_mask),
        "validation": summarize(validation_mask),
        "by_source_train": by_source_train,
        "by_source_validation": by_source_validation,
        "test_rows_aggregated": False,
        "test_rows_opened_for_metric": False,
        "all_retained_aggregate_present": False,
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


def add_causal_source_headings(
    df: pd.DataFrame,
    *,
    road_geometry: dict[str, object],
) -> pd.DataFrame:
    """Attach causal headings before 20-second episode partitioning.

    Exact prior source motion is preferred.  Track starts, gaps, low reported
    speed, and sub-threshold displacement use the current train-fit lane tangent;
    no future actor row is inspected.  Computing this on the unpartitioned table
    makes the result invariant to episode boundaries.
    """
    out = df.copy().reset_index(drop=True)
    required = {
        "source_file",
        "vehicle_id",
        "datetime_jst",
        "x_smooth",
        "y_smooth",
        "v_smooth",
        "traffic_lane",
        "detected_flag",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Causal Japanese heading input lacks columns: {missing}")
    lane_tangent_support: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for raw_lane, raw_points in dict(road_geometry["lane_centerlines"]).items():
        points = np.asarray(raw_points, dtype=np.float64)
        order = np.argsort(points[:, 0], kind="stable")
        points = points[order]
        segment = np.diff(points, axis=0)
        midpoint_x = 0.5 * (points[:-1, 0] + points[1:, 0])
        angle = np.arctan2(segment[:, 1], segment[:, 0])
        if not len(angle) or not np.all(np.isfinite(angle)):
            raise RuntimeError(f"Train-fit lane {raw_lane} has no finite tangent.")
        lane_tangent_support[int(raw_lane)] = (midpoint_x, np.unwrap(angle))
    heading = np.zeros(len(out), dtype=np.float64)
    derivation = np.zeros(len(out), dtype=np.int8)
    for (_source, _vehicle), indices in out.groupby(
        ["source_file", "vehicle_id"], sort=True
    ).groups.items():
        ordered = out.loc[indices].sort_values("datetime_jst", kind="stable")
        row_indices = ordered.index.to_numpy(dtype=np.int64)
        xy = ordered[["x_smooth", "y_smooth"]].to_numpy(dtype=np.float64)
        timestamps_ns = parse_datetime_jst(ordered["datetime_jst"]).astype(
            "int64"
        ).to_numpy(dtype=np.int64)
        speed_mps = pd.to_numeric(
            ordered["v_smooth"], errors="raise"
        ).to_numpy(dtype=np.float64) / 3.6
        provider = pd.to_numeric(
            ordered["detected_flag"], errors="raise"
        ).to_numpy(dtype=np.int8)
        raw_lane = pd.to_numeric(
            ordered["traffic_lane"], errors="raise"
        ).to_numpy(dtype=np.int64)
        fallback = np.full(len(ordered), np.nan, dtype=np.float64)
        for lane_id in np.unique(raw_lane):
            selected = raw_lane == lane_id
            support = lane_tangent_support.get(int(lane_id))
            if support is None:
                continue
            fallback[selected] = np.interp(
                xy[selected, 0],
                support[0],
                support[1],
                left=float(support[1][0]),
                right=float(support[1][-1]),
            )
        delta = np.zeros_like(xy)
        delta[1:] = np.diff(xy, axis=0)
        displacement = np.linalg.norm(delta, axis=1)
        exact_prior = np.zeros(len(ordered), dtype=bool)
        exact_prior[1:] = np.diff(timestamps_ns) == 100_000_000
        source_motion = exact_prior & (speed_mps >= 0.2) & (displacement >= 0.02)
        source_motion[1:] &= (provider[1:] == 1) & (provider[:-1] == 1)
        source_motion[0] = False
        values = fallback.copy()
        values[source_motion] = np.arctan2(
            delta[source_motion, 1], delta[source_motion, 0]
        )
        codes = np.where(source_motion, 1, 2).astype(np.int8)
        for index in np.flatnonzero(~np.isfinite(values)):
            if index > 0 and np.isfinite(values[index - 1]):
                values[index] = values[index - 1]
                codes[index] = 3
        if not np.all(np.isfinite(values)) or not np.all(np.isin(codes, (1, 2, 3))):
            raise RuntimeError(
                f"No causal source heading for Japanese track {_source}/{_vehicle}."
            )
        heading[row_indices] = (values + np.pi) % (2.0 * np.pi) - np.pi
        derivation[row_indices] = codes
    if np.any(derivation == 0):
        raise RuntimeError("Causal Japanese heading derivation left unassigned rows.")
    out["heading_rad"] = heading
    out["heading_valid"] = True
    out["heading_derivation"] = derivation
    return out


def _source_binned_lane_profiles(
    train: pd.DataFrame,
    *,
    lane_ids: tuple[int, ...],
    bin_size_m: float,
) -> dict[str, dict[int, pd.DataFrame]]:
    """Return source-balanced train medians without pooling row counts."""
    profiles: dict[str, dict[int, pd.DataFrame]] = {}
    for source, source_rows in train.groupby("source_file", sort=True):
        source_profiles: dict[int, pd.DataFrame] = {}
        for lane_id in lane_ids:
            lane = source_rows[
                source_rows["traffic_lane"].astype(int) == int(lane_id)
            ][["x_registered", "y_registered"]].copy()
            if len(lane) < 100:
                # The merge lane exists only in the upstream source panels;
                # mainline support remains mandatory in every recording.
                if int(lane_id) == 3:
                    continue
                raise RuntimeError(
                    f"Source {source!r} lane {lane_id} has insufficient train rows."
                )
            lane["_fit_bin"] = np.floor(
                lane["x_registered"].to_numpy(dtype=float) / float(bin_size_m)
            ).astype(np.int64)
            profile = lane.groupby("_fit_bin", observed=False).median().sort_index()
            if len(profile) < 3:
                raise RuntimeError(
                    f"Source {source!r} lane {lane_id} has insufficient fit bins."
                )
            source_profiles[int(lane_id)] = profile
        profiles[str(source)] = source_profiles
    return profiles


def _mainline_midpoint_profiles(
    profiles: dict[str, dict[int, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for source, source_profiles in profiles.items():
        paired = source_profiles[1].join(
            source_profiles[2], how="inner", lsuffix="_1", rsuffix="_2"
        )
        if len(paired) < 20:
            raise RuntimeError(
                f"Source {source!r} has insufficient paired mainline fit bins: {len(paired)}."
            )
        result[source] = pd.DataFrame(
            {
                "x": 0.5 * (paired["x_registered_1"] + paired["x_registered_2"]),
                "midpoint_y": 0.5
                * (paired["y_registered_1"] + paired["y_registered_2"]),
                "separation": (
                    paired["y_registered_1"] - paired["y_registered_2"]
                ).abs(),
            },
            index=paired.index,
        )
    return result


def _equal_source_profile(
    profiles: dict[str, pd.DataFrame],
    *,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for source, profile in profiles.items():
        for fit_bin, row in profile.iterrows():
            rows.append(
                {
                    "source": source,
                    "fit_bin": int(fit_bin),
                    "x": float(row["x"]),
                    "value": float(row[value_column]),
                }
            )
    table = pd.DataFrame(rows)
    aggregated = table.groupby("fit_bin", observed=False)[["x", "value"]].median()
    aggregated = aggregated.sort_values("x")
    x = aggregated["x"].to_numpy(dtype=np.float64)
    values = aggregated["value"].to_numpy(dtype=np.float64)
    unique = np.concatenate(([True], np.diff(x) > 1.0e-9))
    return x[unique], values[unique]


def _robust_lsq_spline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    knot_spacing_m: float,
):
    """Fit a deterministic Huber-reweighted cubic spline."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 4 or np.any(np.diff(x) <= 0.0):
        raise RuntimeError("Japanese shared spine requires four ordered fit points.")
    degree = min(3, len(x) - 1)
    knots = np.arange(
        float(x[0]) + float(knot_spacing_m),
        float(x[-1]),
        float(knot_spacing_m),
    )
    knots = knots[(knots > x[degree]) & (knots < x[-degree - 1])]
    weights = np.ones(len(x), dtype=np.float64)
    spline = None
    for _iteration in range(5):
        if len(knots):
            spline = LSQUnivariateSpline(x, y, knots, w=weights, k=degree, ext=0)
        else:
            spline = UnivariateSpline(x, y, w=weights, k=degree, s=len(x) * 0.01)
        residual = y - spline(x)
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if scale <= 1.0e-9:
            break
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1.0e-12))
    return spline


def _select_mainline_spline(
    midpoint_profiles: dict[str, pd.DataFrame],
    *,
    candidate_knot_spacings_m: tuple[float, ...] = (25.0, 50.0, 100.0),
) -> tuple[object, dict[str, object]]:
    """Select spline smoothness with source-held-out train folds only."""
    diagnostics = []
    sources = sorted(midpoint_profiles)
    for spacing in candidate_knot_spacings_m:
        fold_errors = []
        held_out_sources = sources if len(sources) > 1 else [sources[0]]
        for held_out in held_out_sources:
            fit_profiles = {
                source: profile
                for source, profile in midpoint_profiles.items()
                if len(sources) == 1 or source != held_out
            }
            x_fit, y_fit = _equal_source_profile(
                fit_profiles, value_column="midpoint_y"
            )
            spline = _robust_lsq_spline(
                x_fit, y_fit, knot_spacing_m=float(spacing)
            )
            held = midpoint_profiles[held_out]
            held_x = held["x"].to_numpy(dtype=np.float64)
            held_y = held["midpoint_y"].to_numpy(dtype=np.float64)
            supported = (held_x >= x_fit[0]) & (held_x <= x_fit[-1])
            fold_errors.append(
                float(np.median(np.abs(held_y[supported] - spline(held_x[supported]))))
            )
        x_all, y_all = _equal_source_profile(
            midpoint_profiles, value_column="midpoint_y"
        )
        final_candidate = _robust_lsq_spline(
            x_all, y_all, knot_spacing_m=float(spacing)
        )
        probe = np.arange(x_all[0], x_all[-1] + 1.0e-9, 5.0)
        heading = np.arctan(final_candidate.derivative()(probe))
        max_heading_step_deg = float(
            np.degrees(np.max(np.abs(np.diff(np.unwrap(heading)))))
        )
        diagnostics.append(
            {
                "knot_spacing_m": float(spacing),
                "source_fold_median_absolute_errors_m": fold_errors,
                "mean_fold_median_absolute_error_m": float(np.mean(fold_errors)),
                "fold_standard_error_m": (
                    float(np.std(fold_errors, ddof=1) / np.sqrt(len(fold_errors)))
                    if len(fold_errors) > 1
                    else 0.0
                ),
                "max_heading_change_per_5m_deg": max_heading_step_deg,
                "smoothness_gate_passed": max_heading_step_deg <= 2.0,
            }
        )
    admissible = [item for item in diagnostics if item["smoothness_gate_passed"]]
    if not admissible:
        raise RuntimeError("No Morinomiya shared-spine candidate passed smoothness.")
    best = min(admissible, key=lambda item: item["mean_fold_median_absolute_error_m"])
    threshold = (
        float(best["mean_fold_median_absolute_error_m"])
        + float(best["fold_standard_error_m"])
    )
    selected = max(
        (
            item
            for item in admissible
            if float(item["mean_fold_median_absolute_error_m"]) <= threshold + 1.0e-12
        ),
        key=lambda item: item["knot_spacing_m"],
    )
    x_all, y_all = _equal_source_profile(midpoint_profiles, value_column="midpoint_y")
    spline = _robust_lsq_spline(
        x_all,
        y_all,
        knot_spacing_m=float(selected["knot_spacing_m"]),
    )
    return spline, {
        "contract": "equal_source_train_only_shared_spine_loso_one_se_v1",
        "fit_split": "train",
        "validation_or_test_rows_used": False,
        "candidate_knot_spacings_m": list(candidate_knot_spacings_m),
        "candidate_diagnostics": diagnostics,
        "selected_knot_spacing_m": float(selected["knot_spacing_m"]),
        "selection_rule": "smoothest_within_one_standard_error_of_minimum_loso_mae",
        "fit_sources": sources,
    }


def _estimate_nominal_lane_width(
    midpoint_profiles: dict[str, pd.DataFrame],
    *,
    bin_size_m: float,
) -> tuple[float, dict[str, object]]:
    """Use the median of per-source medians so each recording has equal weight."""
    by_source = {
        source: float(np.median(profile["separation"].to_numpy(dtype=np.float64)))
        for source, profile in midpoint_profiles.items()
    }
    source_values = np.asarray(list(by_source.values()), dtype=np.float64)
    lane_width = float(np.median(source_values))
    if not 2.5 <= lane_width <= 5.0:
        raise RuntimeError(f"Train-fitted Japanese lane width is implausible: {lane_width:g} m.")
    return lane_width, {
        "contract": "equal_source_median_of_binned_mainline_medians_v1",
        "fit_split": "train",
        "merge_rows_used": False,
        "validation_or_test_rows_used": False,
        "bin_size_m": float(bin_size_m),
        "source_count": int(len(by_source)),
        "per_source_median_separation_m": by_source,
        "source_median_quantiles_m": {
            str(q): float(np.quantile(source_values, q)) for q in (0.0, 0.5, 1.0)
        },
        "estimated_lane_width_m": lane_width,
    }


def _isotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares non-increasing projection using pooled adjacent violators."""
    blocks: list[list[float]] = []
    for value in np.asarray(values, dtype=np.float64):
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            right_value, right_weight = blocks.pop()
            left_value, left_weight = blocks.pop()
            weight = left_weight + right_weight
            blocks.append(
                [
                    (left_value * left_weight + right_value * right_weight) / weight,
                    weight,
                ]
            )
    return np.concatenate(
        [np.full(int(weight), value, dtype=np.float64) for value, weight in blocks]
    )


def _registered_position_sha256(df: pd.DataFrame) -> str:
    positions = np.ascontiguousarray(
        df[["x_registered", "y_registered"]].to_numpy(dtype=np.float64)
    )
    return hashlib.sha256(positions.tobytes(order="C")).hexdigest()


def fit_source_derived_japanese_road(
    df: pd.DataFrame,
    *,
    fit_mask: np.ndarray,
    bin_size_m: float = 10.0,
    topology_source_sha256: str | None = None,
) -> dict[str, object]:
    """Fit one smooth, source-balanced road without changing any trajectory."""
    train = df.loc[np.asarray(fit_mask, dtype=bool)].copy()
    source_position_sha256_before = _registered_position_sha256(train)
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
    source_profiles = _source_binned_lane_profiles(
        train,
        lane_ids=(1, 2, 3),
        bin_size_m=float(bin_size_m),
    )
    midpoint_profiles = _mainline_midpoint_profiles(source_profiles)
    lane_width, lane_width_receipt = _estimate_nominal_lane_width(
        midpoint_profiles,
        bin_size_m=float(bin_size_m),
    )
    spine, spine_receipt = _select_mainline_spline(midpoint_profiles)

    mainline_rows = train[train["traffic_lane"].astype(int).isin((1, 2))]
    x_start = float(mainline_rows["x_registered"].min()) - endpoint_padding
    x_end = float(mainline_rows["x_registered"].max()) + endpoint_padding
    main_x = np.arange(x_start, x_end, float(bin_size_m), dtype=np.float64)
    if not len(main_x) or main_x[-1] < x_end:
        main_x = np.append(main_x, x_end)
    spine_y = np.asarray(spine(main_x), dtype=np.float64)
    spine_slope = np.asarray(spine.derivative()(main_x), dtype=np.float64)
    normal_scale = np.sqrt(1.0 + spine_slope**2)
    normal = np.column_stack((-spine_slope / normal_scale, 1.0 / normal_scale))
    spine_points = np.column_stack((main_x, spine_y))
    main_1 = spine_points + 0.5 * lane_width * normal
    main_2 = spine_points - 0.5 * lane_width * normal
    if np.any(np.diff(main_1[:, 0]) <= 0.0) or np.any(np.diff(main_2[:, 0]) <= 0.0):
        raise RuntimeError("Normal-offset Morinomiya main lanes are not x-monotone.")

    lane3_profiles: dict[str, pd.DataFrame] = {}
    for source, profiles in source_profiles.items():
        if 3 not in profiles:
            continue
        lane3 = profiles[3]
        lane3_profiles[source] = pd.DataFrame(
            {
                "x": lane3["x_registered"],
                "lateral_y": lane3["y_registered"],
            },
            index=lane3.index,
        )
    if not lane3_profiles:
        raise RuntimeError("Morinomiya lane 3 has no source with sufficient train support.")
    lane3_x, lane3_y = _equal_source_profile(
        lane3_profiles,
        value_column="lateral_y",
    )
    raw_lane3_x = train.loc[
        train["traffic_lane"].astype(int) == 3, "x_registered"
    ].to_numpy(dtype=np.float64)
    merge_start = float(np.quantile(raw_lane3_x, 0.25))
    merge_end = float(np.max(raw_lane3_x))
    if not x_start < merge_start < merge_end < x_end:
        raise RuntimeError("Train-fitted Japanese merge bounds are not ordered.")
    approach_support = lane3_x <= merge_start
    if np.count_nonzero(approach_support) < 3:
        raise RuntimeError("Morinomiya lane 3 has insufficient pre-merge support.")
    approach_x = lane3_x[approach_support]
    lane1_at_approach = np.interp(approach_x, main_1[:, 0], main_1[:, 1])
    approach_gap = _isotonic_nonincreasing(
        lane3_y[approach_support] - lane1_at_approach
    )
    if np.any(approach_gap <= 0.0):
        raise RuntimeError("Morinomiya lane 3 must remain outside lane 1 before merging.")
    if approach_x[-1] < merge_start:
        approach_x = np.append(approach_x, merge_start)
        approach_gap = np.append(approach_gap, approach_gap[-1])
    else:
        approach_x[-1] = merge_start
    # A flat final support interval gives the approach and taper the same
    # derivative at their junction. The taper then uses the standard monotone
    # smoothstep and reaches lane 1 with matching position and tangent.
    approach_gap[-1] = approach_gap[-2]
    gap_interpolator = PchipInterpolator(approach_x, approach_gap, extrapolate=True)
    merge_approach_start = float(np.min(raw_lane3_x)) - endpoint_padding
    approach_grid = np.arange(
        merge_approach_start,
        merge_start,
        float(bin_size_m),
        dtype=np.float64,
    )
    approach_grid = np.append(approach_grid, merge_start)
    fitted_approach_gap = np.asarray(gap_interpolator(approach_grid), dtype=np.float64)
    if np.any(fitted_approach_gap <= 0.0) or np.any(np.diff(fitted_approach_gap) > 1.0e-8):
        raise RuntimeError("Morinomiya lane-3 approach gap is not positive monotone.")
    taper_grid = np.arange(
        merge_start + float(bin_size_m),
        merge_end,
        float(bin_size_m),
        dtype=np.float64,
    )
    taper_grid = np.append(taper_grid, merge_end)
    taper_t = (taper_grid - merge_start) / (merge_end - merge_start)
    taper_gap = approach_gap[-1] * (1.0 - 3.0 * taper_t**2 + 2.0 * taper_t**3)
    merge_x = np.concatenate((approach_grid, taper_grid))
    merge_gap = np.concatenate((fitted_approach_gap, taper_gap))
    merge_y = np.interp(merge_x, main_1[:, 0], main_1[:, 1]) + merge_gap
    merge = np.column_stack((merge_x, merge_y))
    lane_centerlines = {
        "1": main_1.tolist(),
        "2": main_2.tolist(),
        "3": merge.tolist(),
    }
    paired_spacing = np.linalg.norm(main_1 - main_2, axis=1)
    separation_error = np.abs(paired_spacing - lane_width)
    main_heading = np.arctan2(np.diff(main_1[:, 1]), np.diff(main_1[:, 0]))
    lane3_heading = np.arctan2(np.diff(merge[:, 1]), np.diff(merge[:, 0]))
    max_main_heading_step_deg = float(
        np.degrees(np.max(np.abs(np.diff(np.unwrap(main_heading)))))
    )
    max_lane3_heading_step_deg = float(
        np.degrees(np.max(np.abs(np.diff(np.unwrap(lane3_heading)))))
    )
    source_position_sha256_after = _registered_position_sha256(train)
    geometry_fit_receipt = {
        "contract_id": "morinomiya_shared_spine_parallel_offsets_merge_fit_v1",
        "fit_split": "train",
        "validation_or_test_rows_used": False,
        "mainline_spine": spine_receipt,
        "lane_spacing": {
            "construction": "exact_normal_offsets_from_shared_spine",
            "nominal_width_m": lane_width,
            "median_absolute_error_m": float(np.median(separation_error)),
            "maximum_absolute_error_m": float(np.max(separation_error)),
            "envelope_overlap_max_m": float(np.max(np.maximum(0.0, lane_width - paired_spacing))),
        },
        "smoothness": {
            "mainline_max_heading_change_per_sample_deg": max_main_heading_step_deg,
            "lane3_max_heading_change_per_sample_deg": max_lane3_heading_step_deg,
            "sample_spacing_m": float(bin_size_m),
        },
        "lane3_merge": {
            "contract": "positive_isotonic_approach_gap_and_c1_smoothstep_taper_v1",
            "merge_start_x_m": merge_start,
            "merge_end_x_m": merge_end,
            "approach_gap_nonincreasing": bool(np.all(np.diff(merge_gap) <= 1.0e-8)),
            "junction_position_error_m": float(abs(merge_gap[-1])),
            "junction_tangent_error_rad": 0.0,
        },
        "source_position_invariance": {
            "trajectory_coordinates_modified": False,
            "train_registered_xy_sha256_before": source_position_sha256_before,
            "train_registered_xy_sha256_after": source_position_sha256_after,
            "passed": source_position_sha256_before == source_position_sha256_after,
            "validation_or_test_rows_hashed": False,
        },
    }
    if max_main_heading_step_deg > 2.0:
        raise RuntimeError("Selected Morinomiya mainline exceeds the 2 degree smoothness gate.")
    if not geometry_fit_receipt["source_position_invariance"]["passed"]:
        raise RuntimeError("Morinomiya road fitting changed registered source positions.")

    def segment(points: list[list[float]], lower: float, upper: float) -> list[list[float]]:
        values = np.asarray(points, dtype=float)
        interior = values[(values[:, 0] > lower) & (values[:, 0] < upper)]
        endpoints = np.asarray(
            [
                [lower, np.interp(lower, values[:, 0], values[:, 1])],
                [upper, np.interp(upper, values[:, 0], values[:, 1])],
            ],
            dtype=float,
        )
        return np.vstack((endpoints[:1], interior, endpoints[1:])).tolist()

    if merge_approach_start >= merge_start:
        merge_approach_start = float(np.nextafter(merge_start, -np.inf))
    coordinate_frame = {
        "frame_id": "morinomiya_shared_rigid_se2_v1",
        "units": "m",
        "x_axis": "registered_downstream",
        "y_axis": "registered_left_lateral",
    }

    def lane(
        lane_id: str,
        raw_lane_id: int,
        points: list[list[float]],
        successors: list[str],
        *,
        forbidden: bool = False,
    ) -> dict[str, object]:
        return {
            "lane_id": lane_id,
            "polyline_xy_m": points,
            "width_m": lane_width,
            "raw_lane_ids": [raw_lane_id],
            "successors": successors,
            "forbidden": forbidden,
        }

    return {
        "schema_version": 3,
        "contract_id": "road_geometry_v3",
        "contract": JAPANESE_SOURCE_ROAD_CONTRACT,
        "site_id": "morinomiya",
        "coordinate_frame": coordinate_frame,
        "coordinate_contract": SOURCE_PRESERVING_PREPROCESSING_CONTRACT,
        "fit_split": "train",
        "test_rows_used": False,
        "fit_sources": sorted(train["source_file"].astype(str).unique().tolist()),
        "fit_row_count": int(len(train)),
        "fit_lane_row_counts": train_lane_row_counts,
        "lane_width_m": lane_width,
        "lane_width_estimator": lane_width_receipt,
        "geometry_fit_receipt": geometry_fit_receipt,
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
        "source_document_topology": {
            "contract_id": "morinomiya_source_document_section_kilopost_v1",
            "document_name": "Morinomiya_Lane-marker_Sign_J.pdf",
            "document_sha256": topology_source_sha256,
            "digitized_section_drawing_bands": [
                dict(item) for item in MORINOMIYA_SOURCE_DOCUMENT_SECTION_BANDS
            ],
            "digitized_detector_kiloposts_m": list(
                MORINOMIYA_SOURCE_DOCUMENT_DETECTOR_KILPOSTS_M
            ),
            "geometry_use": (
                "continuous train-fitted lane polylines constrained by documented "
                "two-mainline-plus-section-1-merge topology"
            ),
            "policy_visible": False,
            "test_rows_used": False,
        },
        "nodes": [
            {"node_id": node_id}
            for node_id in ("a", "b", "c", "d", "j")
        ],
        "edges": [
            {
                "edge_id": "main_before_merge",
                "from_node": "a",
                "to_node": "b",
                "lanes": [
                    lane("lane_2_before", 2, segment(lane_centerlines["2"], x_start, merge_start), ["lane_2_merge"]),
                    lane("lane_1_before", 1, segment(lane_centerlines["1"], x_start, merge_start), ["lane_1_merge"]),
                ],
            },
            {
                "edge_id": "main_merge",
                "from_node": "b",
                "to_node": "c",
                "lanes": [
                    lane("lane_2_merge", 2, segment(lane_centerlines["2"], merge_start, merge_end), ["lane_2_after"]),
                    lane("lane_1_merge", 1, segment(lane_centerlines["1"], merge_start, merge_end), ["lane_1_after"]),
                    lane("lane_3_merge", 3, segment(lane_centerlines["3"], merge_start, merge_end), ["lane_1_after"], forbidden=True),
                ],
            },
            {
                "edge_id": "main_after_merge",
                "from_node": "c",
                "to_node": "d",
                "lanes": [
                    lane("lane_2_after", 2, segment(lane_centerlines["2"], merge_end, x_end), []),
                    lane("lane_1_after", 1, segment(lane_centerlines["1"], merge_end, x_end), []),
                ],
            },
            {
                "edge_id": "merge_approach",
                "from_node": "j",
                "to_node": "b",
                "lanes": [
                    lane("lane_3_approach", 3, segment(lane_centerlines["3"], merge_approach_start, merge_start), ["lane_3_merge"], forbidden=True),
                ],
            },
        ],
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
                heading_rad = np.zeros(expected_frames, dtype=np.float64)
                heading_valid_mask = np.zeros(expected_frames, dtype=bool)
                heading_derivation = np.full(expected_frames, -1, dtype=np.int8)
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

                if {
                    "heading_rad",
                    "heading_valid",
                    "heading_derivation",
                }.issubset(ordered.columns):
                    source_heading = ordered["heading_rad"].to_numpy(dtype=np.float64)
                    source_valid = ordered["heading_valid"].to_numpy(dtype=bool)
                    source_derivation = ordered["heading_derivation"].to_numpy(
                        dtype=np.int8
                    )
                    if (
                        not np.all(np.isfinite(source_heading))
                        or not np.all(source_valid)
                        or not np.all(np.isin(source_derivation, (1, 2, 3)))
                    ):
                        raise ValueError(
                            f"Invalid causal heading metadata: {source}/{veh_id}"
                        )
                    heading_rad[frame_idx] = source_heading
                    heading_valid_mask[frame_idx] = source_valid
                    heading_derivation[frame_idx] = source_derivation

                traj_dict[np.int64(veh_id)] = {
                    "source_file": source,
                    "length": np.float64(ordered["vehicle_length"].iloc[0]),
                    "width": np.float64(ordered["vehicle_width"].iloc[0]),
                    "trajectory": traj,
                    "provider_observation_mask": provider_observation_mask,
                    **(
                        {
                            "heading_rad": heading_rad,
                            "heading_valid_mask": heading_valid_mask,
                            "heading_derivation": heading_derivation,
                        }
                        if "heading_rad" in ordered.columns
                        else {}
                    ),
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


def enforce_split_isolation(
    *,
    split_keys: dict[str, list[str]],
    veh_ids_by_episode: dict[str, list[np.int64]],
    trajectories_by_episode: dict[str, dict[np.int64, dict[str, np.ndarray]]],
    window_sec: int,
    boundary_guard_sec: int,
) -> tuple[
    dict[str, list[str]],
    dict[str, list[np.int64]],
    dict[str, dict[np.int64, dict[str, np.ndarray]]],
    dict[str, object],
]:
    """Apply temporal guards and quarantine identities spanning split boundaries.

    Identity is explicitly source-bound even though the current preparer also assigns
    disjoint numeric-id ranges.  Quarantine removes the complete identity from every
    split, not only the boundary row, so no trajectory context can leak.
    """
    if boundary_guard_sec < 0 or boundary_guard_sec % int(window_sec) != 0:
        raise ValueError(
            "--split-boundary-guard-sec must be non-negative and a multiple of --window_sec."
        )
    guarded = {split: list(keys) for split, keys in split_keys.items()}
    removed_guard_keys: dict[str, list[str]] = {"val": [], "test": []}
    guard_episodes = boundary_guard_sec // int(window_sec)
    if guard_episodes:
        for downstream in ("val", "test"):
            kept: list[str] = []
            by_source: dict[str, list[str]] = {}
            for key in guarded[downstream]:
                by_source.setdefault(episode_source_name(key), []).append(key)
            for source, keys in sorted(by_source.items()):
                ordered = sorted(keys, key=episode_time_ms)
                dropped = ordered[:guard_episodes]
                remaining = ordered[guard_episodes:]
                if not remaining:
                    raise RuntimeError(
                        f"Boundary guard removes every {downstream} episode for {source}."
                    )
                removed_guard_keys[downstream].extend(dropped)
                kept.extend(remaining)
            guarded[downstream] = sorted(
                kept, key=lambda key: (episode_source_name(key), episode_time_ms(key))
            )

    identities_by_split: dict[str, set[tuple[str, int]]] = {}
    for split, keys in guarded.items():
        identities_by_split[split] = {
            (episode_source_name(key), int(vehicle_id))
            for key in keys
            for vehicle_id in trajectories_by_episode[key]
        }
    overlapping = (
        (identities_by_split["train"] & identities_by_split["val"])
        | (identities_by_split["train"] & identities_by_split["test"])
        | (identities_by_split["val"] & identities_by_split["test"])
    )

    filtered_ids: dict[str, list[np.int64]] = {}
    filtered_trajectories: dict[str, dict[np.int64, dict[str, np.ndarray]]] = {}
    removed_rows = 0
    removed_vehicle_episode_slots = 0
    removed_empty_episodes: list[str] = []
    for split in ("train", "val", "test"):
        retained_keys: list[str] = []
        for key in guarded[split]:
            source = episode_source_name(key)
            episode_trajectories = {
                vehicle_id: metadata
                for vehicle_id, metadata in trajectories_by_episode[key].items()
                if (source, int(vehicle_id)) not in overlapping
            }
            removed = len(trajectories_by_episode[key]) - len(episode_trajectories)
            removed_vehicle_episode_slots += removed
            for vehicle_id, metadata in trajectories_by_episode[key].items():
                if (source, int(vehicle_id)) in overlapping:
                    provider = np.asarray(metadata["provider_observation_mask"])
                    removed_rows += int((provider >= 0).sum())
            if not episode_trajectories:
                removed_empty_episodes.append(key)
                continue
            retained_keys.append(key)
            filtered_trajectories[key] = episode_trajectories
            filtered_ids[key] = [
                np.int64(vehicle_id)
                for vehicle_id in veh_ids_by_episode[key]
                if (source, int(vehicle_id)) not in overlapping
                and vehicle_id in episode_trajectories
            ]
        guarded[split] = retained_keys

    final_identities = {
        split: {
            (episode_source_name(key), int(vehicle_id))
            for key in guarded[split]
            for vehicle_id in filtered_trajectories[key]
        }
        for split in ("train", "val", "test")
    }
    if any(
        final_identities[left] & final_identities[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("Vehicle identity quarantine failed to make splits disjoint.")
    if len(set().union(*(set(keys) for keys in guarded.values()))) != sum(
        len(keys) for keys in guarded.values()
    ):
        raise RuntimeError("Episode keys overlap after split isolation.")

    receipt: dict[str, object] = {
        "contract_id": "source_identity_disjoint_chronological_split_v1",
        "boundary_guard_sec": int(boundary_guard_sec),
        "guard_episode_count_per_boundary_source": int(guard_episodes),
        "guard_removed_episode_keys": removed_guard_keys,
        "quarantined_cross_split_identity_count": int(len(overlapping)),
        "quarantined_cross_split_identity_examples": [
            {"source_file": source, "vehicle_id": vehicle_id}
            for source, vehicle_id in sorted(overlapping)[:100]
        ],
        "removed_vehicle_episode_slots": int(removed_vehicle_episode_slots),
        "removed_active_rows": int(removed_rows),
        "removed_empty_episode_keys": sorted(removed_empty_episodes),
        "vehicle_identity_overlap_counts_after": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
        "episode_key_overlap_count_after": 0,
    }
    return guarded, filtered_ids, filtered_trajectories, receipt


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
    if args.four_sections:
        args.full_road = True
    if args.full_road:
        args.x_m_max = None
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
    lane_row_counts_before = {
        str(int(lane_id)): int(count)
        for lane_id, count in df["traffic_lane"].value_counts().sort_index().items()
    }
    x_before = pd.to_numeric(df["x_m"], errors="coerce").to_numpy(dtype=float)
    spatial_band_row_counts = {
        "0_800": int(((x_before >= 0.0) & (x_before <= 800.0)).sum()),
        "800_1600": int(((x_before > 800.0) & (x_before <= 1600.0)).sum()),
        "1600_2400": int(((x_before > 1600.0) & (x_before <= 2400.0)).sum()),
        "2400_end": int((x_before > 2400.0).sum()),
    }
    allowed_lane_ids = {int(lane_id) for lane_id in args.allowed_lane_ids}
    unsupported_lane_row_counts: dict[str, int] = {}
    if allowed_lane_ids:
        before_lane_filter = len(df)
        unsupported = df[~df["traffic_lane"].astype(int).isin(allowed_lane_ids)]
        unsupported_lane_row_counts = {
            str(int(lane_id)): int(count)
            for lane_id, count in unsupported["traffic_lane"].value_counts().sort_index().items()
        }
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
        if args.full_road and not re.fullmatch(
            r"[0-9a-f]{64}", str(args.topology_source_sha256 or "")
        ):
            raise ValueError(
                "--full-road requires --topology-source-sha256 as 64 lowercase hex."
            )
        df = df.reset_index(drop=True)
        row_split_masks = source_row_split_masks(
            df,
            window_sec=args.window_sec,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        fit_mask = row_split_masks["train"]
        df["_road_fit_train"] = fit_mask
        df, registration_receipt = rigid_register_shared_coordinates(
            df,
            fit_mask=fit_mask,
        )
        kilopost_receipt = audit_registered_kilopost(
            df,
            fit_mask=fit_mask,
            validation_mask=row_split_masks["validation"],
        )
        df_smooth, source_state_rejections = source_preserving_trajectory_rows(df)
        road_geometry = fit_source_derived_japanese_road(
            df_smooth,
            fit_mask=df_smooth["_road_fit_train"].to_numpy(dtype=bool),
            bin_size_m=args.bin_size_m,
            topology_source_sha256=args.topology_source_sha256,
        )
        if args.four_sections:
            df_smooth = add_causal_source_headings(
                df_smooth,
                road_geometry=road_geometry,
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
    split_keys, veh_ids_all, traj_all, split_isolation_receipt = enforce_split_isolation(
        split_keys={"train": train_keys, "val": val_keys, "test": test_keys},
        veh_ids_by_episode=veh_ids_all,
        trajectories_by_episode=traj_all,
        window_sec=args.window_sec,
        boundary_guard_sec=args.split_boundary_guard_sec,
    )
    train_keys, val_keys, test_keys = (
        split_keys["train"], split_keys["val"], split_keys["test"]
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
            road_geometry.get("contract")
            if road_geometry is not None and not args.four_sections
            else None
        ),
        "four_section_shared_fit": bool(args.four_sections),
        "heading_contract": (
            "causal_source_motion_or_train_road_tangent_v1"
            if args.four_sections
            else None
        ),
        "heading_derived_before_episode_partition": bool(args.four_sections),
        "heading_future_actor_rows_used": False if args.four_sections else None,
        "full_recorded_road": bool(args.full_road),
        "spatial_crop_x_m_max": args.x_m_max,
        "spatial_band_row_counts_before_lane_quarantine": spatial_band_row_counts,
        "lane_row_counts_before_quarantine": lane_row_counts_before,
        "row_quarantine": {
            "unsupported_lane_ids": unsupported_lane_row_counts,
            "unsupported_lane_row_count": int(sum(unsupported_lane_row_counts.values())),
            "lane_9_rows_preserved_in_canonical_storage": int(
                lane_row_counts_before.get("9", 0)
            ),
            "lane_9_training_eligible": False,
            "near_coincident_identity_quarantine": quarantine_report,
        },
        "split_isolation": split_isolation_receipt,
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
    Path(out_dir, "ROW_QUARANTINE.json").write_text(
        json.dumps(manifest["row_quarantine"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if road_geometry is not None:
        if args.four_sections:
            shared_fit = {
                "schema_version": 1,
                "contract_id": "morinomiya_shared_registration_train_road_fit_v1",
                "site_id": "morinomiya",
                "fit_split": "train",
                "test_rows_used": False,
                "continuous_runtime_road_geometry_emitted": False,
                "shared_registered_coordinate_frame": dict(
                    road_geometry["coordinate_frame"]
                ),
                "rigid_shared_registration": registration_receipt,
                "train_fit_road_metadata": {
                    key: road_geometry[key]
                    for key in (
                        "contract",
                        "coordinate_contract",
                        "fit_sources",
                        "fit_row_count",
                        "fit_lane_row_counts",
                        "lane_width_m",
                        "lane_width_estimator",
                        "geometry_fit_receipt",
                        "centerline_endpoint_padding_m",
                        "centerline_endpoint_padding_contract",
                        "x_start_m",
                        "x_end_m",
                        "merge_start_x_m",
                        "merge_end_x_m",
                        "lane_centerlines",
                        "source_document_topology",
                    )
                },
                "kilopost_registration_audit": kilopost_receipt,
            }
            Path(out_dir, "SHARED_REGISTRATION.json").write_text(
                json.dumps(shared_fit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
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
