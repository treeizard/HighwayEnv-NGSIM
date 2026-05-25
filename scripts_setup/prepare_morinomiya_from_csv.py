#!/usr/bin/env python3
"""Prepare a clean filtered Morinomiya trajectory array from raw CSV files.

The raw Morinomiya folders contain one trajectory CSV per source recording. Local
vehicle IDs are not globally unique across those recordings, and some previously
generated merged arrays contained blank padded rows. This script reads the CSVs
directly, keeps the local ID for auditability, and writes a globally unique
``vehicle_id`` by prefixing each source.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAJECTORY_COLUMNS = [
    "vehicle_id_local",
    "datetime",
    "vehicle_type",
    "velocity",
    "traffic_lane",
    "longitude",
    "latitude",
    "kilopost",
    "vehicle_length",
    "detected_flag",
]
JST_TIMEZONE = "Asia/Tokyo"
DEFAULT_BASE_DATE = "2020-01-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="raw_data/Morinomiya")
    parser.add_argument(
        "--output-npy",
        default="raw_data/morinomiya_filtered_800_from_csv.npy",
    )
    parser.add_argument(
        "--summary-json",
        default="artifacts/morinomiya_processing/morinomiya_filtered_800_summary.json",
    )
    parser.add_argument(
        "--availability-csv",
        default="artifacts/morinomiya_processing/morinomiya_availability_by_source.csv",
    )
    parser.add_argument("--start-clock", default="09:00:00")
    parser.add_argument("--end-clock", default="13:00:00")
    parser.add_argument("--base-date", default=DEFAULT_BASE_DATE)
    parser.add_argument("--x-m-min", type=float, default=0.0)
    parser.add_argument("--x-m-max", type=float, default=800.0)
    parser.add_argument("--basis-lat", type=float, default=34.681580)
    parser.add_argument("--basis-lon", type=float, default=135.527945)
    parser.add_argument("--id-source-stride", type=int, default=10_000_000)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--availability-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_clock_time(value: str) -> dt.time:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid clock time {value!r}; expected HH:MM[:SS].")
    return dt.time(
        hour=int(parts[0]),
        minute=int(parts[1]),
        second=int(parts[2]) if len(parts) == 3 else 0,
    )


def clock_int_to_timedelta(values: pd.Series) -> pd.Series:
    """Convert HHMMSSmmm integers such as 94000500 or 100000600."""
    numeric = pd.to_numeric(values, errors="coerce").astype("Int64")
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
    out = pd.to_timedelta(total_ms.where(valid), unit="ms")
    return out


def clock_int_to_jst(values: pd.Series, base_date: str) -> pd.Series:
    base = pd.Timestamp(base_date, tz=JST_TIMEZONE)
    return base + clock_int_to_timedelta(values)


def add_local_xy(
    df: pd.DataFrame,
    *,
    basis_lat: float,
    basis_lon: float,
) -> pd.DataFrame:
    radius = 6378137.0
    lat0 = np.deg2rad(basis_lat)
    lon0 = np.deg2rad(basis_lon)
    lat = np.deg2rad(pd.to_numeric(df["latitude"], errors="coerce").to_numpy())
    lon = np.deg2rad(pd.to_numeric(df["longitude"], errors="coerce").to_numpy())
    df = df.copy()
    df["x_m"] = radius * (lon - lon0) * np.cos(lat0)
    df["y_m"] = radius * (lat - lat0)
    return df


def discover_trajectory_csvs(raw_root: Path) -> list[Path]:
    return sorted(raw_root.glob("L003_F*_ALL/L003_F*_TRAJECTORY/L003_F*_trajectory.csv"))


def source_name_from_path(path: Path) -> str:
    return path.stem.replace("_trajectory", "")


def source_offset(source_index: int, stride: int) -> int:
    return int(source_index) * int(stride)


def frame_filter(
    df: pd.DataFrame,
    *,
    start_clock: dt.time,
    end_clock: dt.time,
    base_date: str,
    basis_lat: float,
    basis_lon: float,
    x_m_min: float | None,
    x_m_max: float | None,
) -> pd.DataFrame:
    for column in TRAJECTORY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["vehicle_id_local", "datetime", "traffic_lane", "longitude", "latitude"]).copy()
    if df.empty:
        return df

    df["datetime_jst"] = clock_int_to_jst(df["datetime"], base_date)
    df = df.dropna(subset=["datetime_jst"]).copy()
    if df.empty:
        return df

    clock_times = df["datetime_jst"].dt.time
    time_mask = (clock_times >= start_clock) & (clock_times < end_clock)
    df = df.loc[time_mask].copy()
    if df.empty:
        return df

    df = add_local_xy(df, basis_lat=basis_lat, basis_lon=basis_lon)
    spatial_mask = np.isfinite(df["x_m"]) & np.isfinite(df["y_m"])
    if x_m_min is not None:
        spatial_mask &= df["x_m"] >= float(x_m_min)
    if x_m_max is not None:
        spatial_mask &= df["x_m"] <= float(x_m_max)
    return df.loc[spatial_mask].copy()


def summarize_source(
    path: Path,
    *,
    source_index: int,
    args: argparse.Namespace,
    start_clock: dt.time,
    end_clock: dt.time,
) -> tuple[pd.DataFrame, dict]:
    source_name = source_name_from_path(path)
    kept_chunks: list[pd.DataFrame] = []
    raw_rows = 0
    time_rows = 0
    x_rows = 0
    raw_min = None
    raw_max = None

    for chunk in pd.read_csv(
        path,
        header=None,
        names=TRAJECTORY_COLUMNS,
        chunksize=int(args.chunksize),
    ):
        raw_rows += len(chunk)
        chunk_dt = clock_int_to_jst(chunk["datetime"], args.base_date)
        valid_dt = chunk_dt.dropna()
        if not valid_dt.empty:
            chunk_min = valid_dt.min()
            chunk_max = valid_dt.max()
            raw_min = chunk_min if raw_min is None else min(raw_min, chunk_min)
            raw_max = chunk_max if raw_max is None else max(raw_max, chunk_max)

        filtered = frame_filter(
            chunk,
            start_clock=start_clock,
            end_clock=end_clock,
            base_date=args.base_date,
            basis_lat=float(args.basis_lat),
            basis_lon=float(args.basis_lon),
            x_m_min=args.x_m_min,
            x_m_max=args.x_m_max,
        )
        if filtered.empty:
            continue

        time_rows += len(filtered)
        x_rows += len(filtered)
        filtered["source_file"] = source_name
        filtered["vehicle_id_local"] = filtered["vehicle_id_local"].astype(np.int64)
        filtered["vehicle_id"] = (
            source_offset(source_index, int(args.id_source_stride))
            + filtered["vehicle_id_local"].astype(np.int64)
        )
        kept_chunks.append(filtered)

    if kept_chunks:
        kept = pd.concat(kept_chunks, ignore_index=True)
    else:
        kept = pd.DataFrame(columns=["vehicle_id", *TRAJECTORY_COLUMNS, "datetime_jst", "source_file", "x_m", "y_m"])

    summary = {
        "source_file": source_name,
        "source_index": int(source_index),
        "vehicle_id_offset": source_offset(source_index, int(args.id_source_stride)),
        "raw_rows": int(raw_rows),
        "filtered_rows": int(x_rows),
        "raw_time_min": str(raw_min) if raw_min is not None else None,
        "raw_time_max": str(raw_max) if raw_max is not None else None,
        "filtered_time_min": str(kept["datetime_jst"].min()) if not kept.empty else None,
        "filtered_time_max": str(kept["datetime_jst"].max()) if not kept.empty else None,
        "filtered_vehicle_count": int(kept["vehicle_id"].nunique()) if not kept.empty else 0,
        "filtered_local_vehicle_count": int(kept["vehicle_id_local"].nunique()) if not kept.empty else 0,
    }
    return kept, summary


def conflict_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "duplicate_source_local_time_groups": 0,
            "conflicting_source_local_time_groups": 0,
            "duplicate_global_time_groups": 0,
        }

    grouped = df.groupby(["source_file", "vehicle_id_local", "datetime"], sort=False)
    group_sizes = grouped.size()
    duplicate_groups = group_sizes[group_sizes > 1]
    conflicting_groups = 0
    if not duplicate_groups.empty:
        subset_index = duplicate_groups.index
        dup_rows = df.set_index(["source_file", "vehicle_id_local", "datetime"]).loc[subset_index].reset_index()
        spreads = dup_rows.groupby(["source_file", "vehicle_id_local", "datetime"], sort=False).agg(
            x_span=("x_m", lambda s: float(s.max() - s.min())),
            y_span=("y_m", lambda s: float(s.max() - s.min())),
        )
        conflicting_groups = int(((spreads["x_span"] > 2.0) | (spreads["y_span"] > 2.0)).sum())

    duplicate_global_time = int(
        (df.groupby(["vehicle_id", "datetime"], sort=False).size() > 1).sum()
    )
    return {
        "duplicate_source_local_time_groups": int(len(duplicate_groups)),
        "conflicting_source_local_time_groups": conflicting_groups,
        "duplicate_global_time_groups": duplicate_global_time,
    }


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = ["source_file", "vehicle_id_local", "datetime", "detected_flag"]
    df = df.sort_values(sort_cols, ascending=[True, True, True, False]).copy()
    df = df.drop_duplicates(["source_file", "vehicle_id_local", "datetime"], keep="first")
    return df.sort_values(["datetime_jst", "source_file", "vehicle_id_local"]).reset_index(drop=True)


def to_record_array(df: pd.DataFrame) -> np.ndarray:
    dtype = np.dtype(
        [
            ("vehicle_id", "i8"),
            ("vehicle_id_local", "i8"),
            ("datetime", "i8"),
            ("datetime_jst", "U32"),
            ("vehicle_type", "i8"),
            ("velocity", "f8"),
            ("traffic_lane", "i8"),
            ("longitude", "f8"),
            ("latitude", "f8"),
            ("kilopost", "f8"),
            ("vehicle_length", "f8"),
            ("detected_flag", "i8"),
            ("source_file", "U16"),
            ("x_m", "f8"),
            ("y_m", "f8"),
        ]
    )
    records = np.empty(len(df), dtype=dtype)
    records["vehicle_id"] = df["vehicle_id"].to_numpy(dtype=np.int64)
    records["vehicle_id_local"] = df["vehicle_id_local"].to_numpy(dtype=np.int64)
    records["datetime"] = df["datetime"].to_numpy(dtype=np.int64)
    records["datetime_jst"] = df["datetime_jst"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z").to_numpy(dtype="U32")
    records["vehicle_type"] = df["vehicle_type"].fillna(0).to_numpy(dtype=np.int64)
    records["velocity"] = df["velocity"].to_numpy(dtype=float)
    records["traffic_lane"] = df["traffic_lane"].to_numpy(dtype=np.int64)
    records["longitude"] = df["longitude"].to_numpy(dtype=float)
    records["latitude"] = df["latitude"].to_numpy(dtype=float)
    records["kilopost"] = df["kilopost"].to_numpy(dtype=float)
    records["vehicle_length"] = df["vehicle_length"].to_numpy(dtype=float)
    records["detected_flag"] = df["detected_flag"].fillna(0).to_numpy(dtype=np.int64)
    records["source_file"] = df["source_file"].astype(str).to_numpy(dtype="U16")
    records["x_m"] = df["x_m"].to_numpy(dtype=float)
    records["y_m"] = df["y_m"].to_numpy(dtype=float)
    return records


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_npy = Path(args.output_npy)
    summary_json = Path(args.summary_json)
    availability_csv = Path(args.availability_csv)

    if output_npy.exists() and not args.overwrite and not args.availability_only:
        raise FileExistsError(f"{output_npy} already exists; pass --overwrite to replace it.")

    start_clock = parse_clock_time(args.start_clock)
    end_clock = parse_clock_time(args.end_clock)
    if start_clock >= end_clock:
        raise ValueError("--start-clock must be earlier than --end-clock.")

    csv_paths = discover_trajectory_csvs(raw_root)
    if not csv_paths:
        raise FileNotFoundError(f"No trajectory CSVs found under {raw_root}")

    all_chunks: list[pd.DataFrame] = []
    source_summaries: list[dict] = []
    for source_index, path in enumerate(csv_paths):
        kept, summary = summarize_source(
            path,
            source_index=source_index,
            args=args,
            start_clock=start_clock,
            end_clock=end_clock,
        )
        source_summaries.append(summary)
        if not kept.empty:
            all_chunks.append(kept)
        print(
            f"{summary['source_file']}: raw={summary['raw_rows']} "
            f"kept={summary['filtered_rows']} "
            f"time={summary['filtered_time_min']}..{summary['filtered_time_max']} "
            f"offset={summary['vehicle_id_offset']}"
        )

    combined = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    conflicts_before = conflict_summary(combined)
    cleaned = clean_rows(combined)
    conflicts_after = conflict_summary(cleaned)

    availability_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_summaries).to_csv(availability_csv, index=False)

    summary = {
        "raw_root": str(raw_root),
        "output_npy": str(output_npy),
        "start_clock": args.start_clock,
        "end_clock": args.end_clock,
        "base_date": args.base_date,
        "x_m_min": args.x_m_min,
        "x_m_max": args.x_m_max,
        "source_count": len(source_summaries),
        "rows_before_duplicate_drop": int(len(combined)),
        "rows_after_duplicate_drop": int(len(cleaned)),
        "time_min": str(cleaned["datetime_jst"].min()) if not cleaned.empty else None,
        "time_max": str(cleaned["datetime_jst"].max()) if not cleaned.empty else None,
        "global_vehicle_count": int(cleaned["vehicle_id"].nunique()) if not cleaned.empty else 0,
        "conflicts_before_duplicate_drop": conflicts_before,
        "conflicts_after_duplicate_drop": conflicts_after,
        "sources": source_summaries,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.availability_only:
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        records = to_record_array(cleaned)
        np.save(output_npy, records)
        print(f"Saved {len(records)} rows to {output_npy}")

    print(f"Wrote availability CSV: {availability_csv}")
    print(f"Wrote summary JSON: {summary_json}")
    print(json.dumps({k: summary[k] for k in summary if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
