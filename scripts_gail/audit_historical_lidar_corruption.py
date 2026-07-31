#!/usr/bin/env python3
"""Quantify legacy lidar-space violations without opening a test split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DOMAINS = ("us", "japanese")
ALLOWED_SPLITS = ("train", "val")
LIDAR_CELLS = 128
LIDAR_WIDTH = LIDAR_CELLS * 2
BOUND_TOLERANCE = 1.0e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Manifest has no episodes: {manifest_path}")

    arrays = {
        "observations": {
            "rows": 0,
            "violating_rows": 0,
            "violating_cells": 0,
            "maximum_absolute_value": 0.0,
            "trajectory_ids": set(),
        },
        "next_observations": {
            "rows": 0,
            "violating_rows": 0,
            "violating_cells": 0,
            "maximum_absolute_value": 0.0,
            "trajectory_ids": set(),
        },
    }
    files: list[dict[str, Any]] = []
    for episode in episodes:
        episode_name = str(episode["episode_name"])
        path = root / str(episode["dataset_file"])
        with np.load(path, allow_pickle=False) as payload:
            vehicle_ids = np.asarray(payload["vehicle_ids"], dtype=np.int64)
            per_file: dict[str, Any] = {}
            for array_name, accumulator in arrays.items():
                raw = np.asarray(payload[array_name])
                if raw.ndim != 2 or raw.shape[1] < LIDAR_WIDTH:
                    raise ValueError(
                        f"{path} {array_name} has invalid shape {raw.shape}."
                    )
                speeds = raw[:, 1:LIDAR_WIDTH:2].astype(
                    np.float64,
                    copy=False,
                )
                violations = np.abs(speeds) > 1.0 + BOUND_TOLERANCE
                row_violations = np.any(violations, axis=1)
                violating_rows = int(np.count_nonzero(row_violations))
                violating_cells = int(np.count_nonzero(violations))
                maximum = (
                    float(np.max(np.abs(speeds))) if speeds.size else 0.0
                )
                trajectory_ids = {
                    f"{episode_name}:{int(vehicle_id)}"
                    for vehicle_id in vehicle_ids[row_violations]
                }
                accumulator["rows"] += int(len(raw))
                accumulator["violating_rows"] += violating_rows
                accumulator["violating_cells"] += violating_cells
                accumulator["maximum_absolute_value"] = max(
                    float(accumulator["maximum_absolute_value"]),
                    maximum,
                )
                accumulator["trajectory_ids"].update(trajectory_ids)
                per_file[array_name] = {
                    "rows": int(len(raw)),
                    "violating_rows": violating_rows,
                    "violating_cells": violating_cells,
                    "maximum_absolute_value": maximum,
                    "violating_trajectories": len(trajectory_ids),
                }
        files.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "arrays": per_file,
            }
        )

    serialized_arrays: dict[str, Any] = {}
    for name, accumulator in arrays.items():
        rows = int(accumulator["rows"])
        violating_rows = int(accumulator["violating_rows"])
        trajectory_ids = sorted(accumulator.pop("trajectory_ids"))
        serialized_arrays[name] = {
            **accumulator,
            "violating_row_fraction": (
                float(violating_rows / rows) if rows else 0.0
            ),
            "violating_trajectory_count": len(trajectory_ids),
            "violating_trajectory_ids_sha256": hashlib.sha256(
                json.dumps(
                    trajectory_ids,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "episode_count": len(episodes),
        "declared_lidar_relative_speed_range": [-1.0, 1.0],
        "bound_tolerance": BOUND_TOLERANCE,
        "arrays": serialized_arrays,
        "files": files,
        "status": (
            "failed_declared_lidar_space"
            if any(
                int(record["violating_cells"]) > 0
                for record in serialized_arrays.values()
            )
            else "passed"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALLOWED_SPLITS,
        default=list(ALLOWED_SPLITS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit receipt: {output}")
    splits = tuple(dict.fromkeys(str(value) for value in args.splits))
    if not splits or len(splits) != len(args.splits):
        raise ValueError("Splits must be a non-empty, duplicate-free subset.")
    collection_root = args.collection_root.resolve()
    records = {
        f"{domain}/{split}": audit_split(
            collection_root / domain / split
        )
        for domain in DOMAINS
        for split in splits
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "historical corruption diagnosis only; no filtering or repair of "
            "the source collection"
        ),
        "collection_root": str(collection_root),
        "audited_splits": list(splits),
        "test_data_status": "not_opened",
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "status": (
            "failed_declared_lidar_space"
            if any(
                record["status"] != "passed"
                for record in records.values()
            )
            else "passed"
        ),
        "splits": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
