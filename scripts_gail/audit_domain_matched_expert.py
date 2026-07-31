#!/usr/bin/env python3
"""Audit all domain/split expert files against one explicit comparison contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts_gail.ps_gail.contracts import (
    assert_compatible_action_contracts,
    assert_compatible_observation_contracts,
    policy_observation_contract,
    runtime_continuous_action_contract,
    validate_declared_raw_observation_space,
    validate_expert_action_contract,
)


DOMAINS = ("us", "japanese")
SPLITS = ("train", "val", "test")
NORMALIZED_ACTION_COLUMNS = ("acceleration_norm", "steering_norm")
PHYSICAL_ACTION_COLUMNS = ("steering_rad", "acceleration_mps2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def summarize_raw_observation_receipts(
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    field_extrema: dict[str, dict[str, float]] = {}
    rows_per_array = 0
    for file_receipt in receipts:
        rows_per_array += int(file_receipt["observations"]["row_count"])
        for array_name in ("observations", "next_observations"):
            for field_name, field in file_receipt[array_name]["fields"].items():
                extrema = field_extrema.setdefault(
                    field_name,
                    {
                        "minimum": float(field["minimum"]),
                        "maximum": float(field["maximum"]),
                    },
                )
                extrema["minimum"] = min(
                    extrema["minimum"],
                    float(field["minimum"]),
                )
                extrema["maximum"] = max(
                    extrema["maximum"],
                    float(field["maximum"]),
                )
    return {
        "status": "passed",
        "files_checked": len(receipts),
        "rows_checked_per_array": rows_per_array,
        "arrays_checked": ["observations", "next_observations"],
        "field_extrema_across_both_arrays": field_extrema,
    }


def audit_split(root: Path, *, reference_observation: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    validation_path = root / "action_contract_validation.json"
    manifest = read_json(manifest_path)
    validation = read_json(validation_path)
    if validation.get("status") != "complete":
        raise ValueError(f"Collection validation is not complete: {validation_path}")

    manifest_action = manifest.get("continuous_action_contract")
    manifest_observation = manifest.get("policy_observation_contract")
    if not isinstance(manifest_action, dict):
        raise ValueError(f"Missing manifest action contract: {manifest_path}")
    if not isinstance(manifest_observation, dict):
        raise ValueError(f"Missing manifest observation contract: {manifest_path}")
    assert_compatible_action_contracts(
        runtime_continuous_action_contract(),
        manifest_action,
    )
    assert_compatible_observation_contracts(
        reference_observation,
        manifest_observation,
    )

    episode_rows = manifest.get("episodes")
    if not isinstance(episode_rows, list) or not episode_rows:
        raise ValueError(f"Manifest has no episode records: {manifest_path}")
    files = [root / str(row["dataset_file"]) for row in episode_rows]
    if len(files) != len(set(files)):
        raise ValueError(f"Manifest repeats dataset files: {manifest_path}")

    rows = 0
    file_records: list[dict[str, Any]] = []
    action_sum = np.zeros(2, dtype=np.float64)
    action_square_sum = np.zeros(2, dtype=np.float64)
    physical_action_sum = np.zeros(2, dtype=np.float64)
    physical_action_square_sum = np.zeros(2, dtype=np.float64)
    normalized_saturation_count = np.zeros(2, dtype=np.int64)
    behavior_counts = {
        "hard_brake_acceleration_le_minus_1p5_mps2": 0,
        "positive_acceleration_ge_1p5_mps2": 0,
        "lateral_abs_steering_ge_0p008_rad": 0,
        "sharp_lateral_abs_steering_ge_0p08_rad": 0,
        "near_zero_abs_steering_lt_0p001_rad": 0,
    }
    trajectory_lengths: list[int] = []
    terminal_trajectory_count = 0
    trajectory_gap_count = 0
    vehicle_id_values: set[int] = set()
    raw_observation_receipts: list[dict[str, Any]] = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        # The collection writer stores metadata_json as a zero-dimensional
        # object array for compatibility with the established expert-data
        # loaders. These are freshly collected, manifest-enumerated project
        # artifacts, so load the object scalar using the same contract as the
        # training loader. The numeric arrays are still copied into explicit
        # dtypes and validated below.
        with np.load(path, allow_pickle=True) as payload:
            normalized = np.asarray(
                payload["actions_continuous_env"],
                dtype=np.float64,
            )
            physical = np.asarray(
                payload["actions_steering_acceleration"],
                dtype=np.float64,
            )
            observations = np.asarray(payload["observations"])
            next_observations = np.asarray(payload["next_observations"])
            trajectory_states = np.asarray(
                payload["trajectory_states"],
                dtype=np.float64,
            )
            rewards = np.asarray(payload["rewards"], dtype=np.float64)
            dones = np.asarray(payload["dones"], dtype=bool)
            vehicle_ids = np.asarray(payload["vehicle_ids"], dtype=np.int64)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int64)
            metadata = json.loads(str(payload["metadata_json"].item()))
        file_action = validate_expert_action_contract(
            normalized,
            physical,
            recorded_contract=metadata.get("continuous_action_contract"),
            require_runtime_match=True,
        )
        assert_compatible_action_contracts(manifest_action, file_action)
        file_observation = metadata.get("policy_observation_contract")
        if not isinstance(file_observation, dict):
            raise ValueError(f"Missing explicit observation contract: {path}")
        assert_compatible_observation_contracts(
            manifest_observation,
            file_observation,
        )
        observation_receipt = validate_declared_raw_observation_space(
            observations,
            file_observation,
            context=f"{path} observations",
        )
        next_observation_receipt = validate_declared_raw_observation_space(
            next_observations,
            file_observation,
            context=f"{path} next_observations",
        )
        raw_observation_receipts.append(
            {
                "observations": observation_receipt,
                "next_observations": next_observation_receipt,
            }
        )
        if next_observations.shape != observations.shape:
            raise ValueError(
                f"Observation transition mismatch {observations.shape} -> "
                f"{next_observations.shape}: {path}"
            )
        aligned_lengths = {
            len(observations),
            len(trajectory_states),
            len(normalized),
            len(physical),
            len(rewards),
            len(dones),
            len(vehicle_ids),
            len(timesteps),
        }
        if len(aligned_lengths) != 1:
            raise ValueError(f"Transition arrays are not row-aligned: {path}")
        numeric_arrays = {
            "observations": observations,
            "next_observations": next_observations,
            "trajectory_states": trajectory_states,
            "actions_continuous_env": normalized,
            "actions_steering_acceleration": physical,
            "rewards": rewards,
        }
        for name, values in numeric_arrays.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path} contains non-finite {name}.")
        if np.any(np.abs(normalized) > 1.0 + 1.0e-6):
            raise ValueError(f"{path} has normalized actions outside [-1, 1].")

        action_sum += normalized.sum(axis=0, dtype=np.float64)
        action_square_sum += np.square(normalized).sum(axis=0, dtype=np.float64)
        physical_action_sum += physical.sum(axis=0, dtype=np.float64)
        physical_action_square_sum += np.square(physical).sum(
            axis=0,
            dtype=np.float64,
        )
        normalized_saturation_count += np.sum(
            np.abs(normalized) >= 1.0 - 1.0e-6,
            axis=0,
            dtype=np.int64,
        )
        steering = physical[:, 0]
        acceleration = physical[:, 1]
        behavior_counts[
            "hard_brake_acceleration_le_minus_1p5_mps2"
        ] += int(np.sum(acceleration <= -1.5))
        behavior_counts[
            "positive_acceleration_ge_1p5_mps2"
        ] += int(np.sum(acceleration >= 1.5))
        behavior_counts[
            "lateral_abs_steering_ge_0p008_rad"
        ] += int(np.sum(np.abs(steering) >= 0.008))
        behavior_counts[
            "sharp_lateral_abs_steering_ge_0p08_rad"
        ] += int(np.sum(np.abs(steering) >= 0.08))
        behavior_counts[
            "near_zero_abs_steering_lt_0p001_rad"
        ] += int(np.sum(np.abs(steering) < 0.001))

        vehicle_id_values.update(int(value) for value in np.unique(vehicle_ids))
        for vehicle_id in np.unique(vehicle_ids):
            indices = np.flatnonzero(vehicle_ids == vehicle_id)
            order = np.argsort(timesteps[indices], kind="stable")
            indices = indices[order]
            if not len(indices):
                continue
            differences = np.diff(timesteps[indices])
            boundaries = np.flatnonzero(differences != 1) + 1
            segments = np.split(indices, boundaries)
            trajectory_gap_count += int(len(boundaries))
            for segment in segments:
                if not len(segment):
                    continue
                trajectory_lengths.append(int(len(segment)))
                terminal_trajectory_count += int(bool(dones[int(segment[-1])]))
        rows += int(len(observations))
        file_records.append(
            {
                "file": path.name,
                "rows": int(len(observations)),
                "sha256": sha256_file(path),
            }
        )

    if rows != int(manifest.get("num_samples", -1)):
        raise ValueError(
            f"Manifest sample count {manifest.get('num_samples')} != audited {rows}: "
            f"{manifest_path}"
        )
    if len(files) != int(manifest.get("num_episodes", -1)):
        raise ValueError(
            f"Manifest episode count {manifest.get('num_episodes')} != audited "
            f"{len(files)}: {manifest_path}"
        )
    if not trajectory_lengths:
        raise ValueError(f"No trajectory segments were found: {root}")
    action_mean = action_sum / rows
    action_std = np.sqrt(
        np.maximum(action_square_sum / rows - np.square(action_mean), 0.0)
    )
    physical_action_mean = physical_action_sum / rows
    physical_action_std = np.sqrt(
        np.maximum(
            physical_action_square_sum / rows
            - np.square(physical_action_mean),
            0.0,
        )
    )
    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "action_contract_validation_sha256": sha256_file(validation_path),
        "episode_count": len(files),
        "row_count": rows,
        "numeric_completeness": {
            "all_required_arrays_row_aligned": True,
            "all_numeric_arrays_finite": True,
            "normalized_actions_within_bounds": True,
            "declared_raw_observation_space_passed": True,
        },
        "raw_observation_space_validation": (
            summarize_raw_observation_receipts(raw_observation_receipts)
        ),
        "trajectory_coverage": {
            "trajectory_segment_count": len(trajectory_lengths),
            "unexpected_timestep_gap_count": trajectory_gap_count,
            "length_steps": {
                "minimum": int(np.min(trajectory_lengths)),
                "median": float(np.median(trajectory_lengths)),
                "p95": float(np.percentile(trajectory_lengths, 95)),
                "maximum": int(np.max(trajectory_lengths)),
            },
            "terminal_last_row_fraction": (
                float(terminal_trajectory_count) / len(trajectory_lengths)
            ),
        },
        "action_coverage": {
            "normalized_columns": list(NORMALIZED_ACTION_COLUMNS),
            "normalized_mean": action_mean.tolist(),
            "normalized_std": action_std.tolist(),
            "normalized_saturation_fraction": (
                normalized_saturation_count / rows
            ).tolist(),
            "physical_columns": list(PHYSICAL_ACTION_COLUMNS),
            "physical_mean": physical_action_mean.tolist(),
            "physical_std": physical_action_std.tolist(),
            "behavior_fraction": {
                name: float(count) / rows
                for name, count in behavior_counts.items()
            },
        },
        "vehicle_ids": sorted(vehicle_id_values),
        "continuous_action_contract": manifest_action,
        "policy_observation_contract": manifest_observation,
        "files": file_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help=(
            "Explicit split subset to open, for example '--splits train val'. "
            "Comma-separated values are also accepted. Omitted splits are not "
            "resolved, listed or opened."
        ),
    )
    return parser.parse_args()


def normalize_requested_splits(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    requested: list[str] = []
    for value in values:
        requested.extend(
            part.strip()
            for part in str(value).split(",")
            if part.strip()
        )
    invalid = sorted(set(requested).difference(SPLITS))
    if invalid:
        raise ValueError(
            f"Unsupported audit splits {invalid}; expected a subset of {SPLITS}."
        )
    if not requested:
        raise ValueError("At least one audit split must be requested.")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Audit splits must not repeat: {requested}.")
    return tuple(requested)


def audit_collection(
    collection_root: Path,
    *,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    collection_root = collection_root.resolve()
    reference_observation = policy_observation_contract(
        lidar_cells=128,
        maximum_range=64.0,
    )
    split_records: dict[str, Any] = {}
    reference_action: dict[str, Any] | None = None
    for domain in DOMAINS:
        for split in splits:
            key = f"{domain}/{split}"
            record = audit_split(
                collection_root / domain / split,
                reference_observation=reference_observation,
            )
            action_contract = record["continuous_action_contract"]
            if reference_action is None:
                reference_action = action_contract
            else:
                assert_compatible_action_contracts(
                    reference_action,
                    action_contract,
                )
            split_records[key] = record

    pairwise_split_checks: dict[str, Any] = {}
    for domain in DOMAINS:
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                left_record = split_records[f"{domain}/{left}"]
                right_record = split_records[f"{domain}/{right}"]
                left_names = {row["file"] for row in left_record["files"]}
                right_names = {row["file"] for row in right_record["files"]}
                left_hashes = {row["sha256"] for row in left_record["files"]}
                right_hashes = {row["sha256"] for row in right_record["files"]}
                left_vehicles = set(left_record["vehicle_ids"])
                right_vehicles = set(right_record["vehicle_ids"])
                pairwise_split_checks[f"{domain}/{left}_vs_{right}"] = {
                    "episode_filename_overlap_count": len(
                        left_names.intersection(right_names)
                    ),
                    "exact_file_sha256_overlap_count": len(
                        left_hashes.intersection(right_hashes)
                    ),
                    "raw_vehicle_id_overlap_count": len(
                        left_vehicles.intersection(right_vehicles)
                    ),
                    "qualification": (
                        "time_window_disjoint_not_vehicle_disjoint"
                    ),
                }

    all_hashes = [
        row["sha256"]
        for record in split_records.values()
        for row in record["files"]
    ]
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError("Exact expert episode content is duplicated across splits.")
    if any(
        check["episode_filename_overlap_count"]
        or check["exact_file_sha256_overlap_count"]
        for check in pairwise_split_checks.values()
    ):
        raise ValueError(
            "Expert episode leakage detected within the explicitly audited "
            f"splits {list(splits)}."
        )

    return {
        "schema_version": 1,
        "status": "passed",
        "collection_root": str(collection_root),
        "audited_splits": list(splits),
        "not_opened_splits": [
            split for split in SPLITS if split not in splits
        ],
        "test_data_status": (
            "audited" if "test" in splits else "not_opened"
        ),
        "domain_split_count": len(split_records),
        "continuous_action_contract": reference_action,
        "policy_observation_contract": reference_observation,
        "total_episode_count": sum(
            int(record["episode_count"]) for record in split_records.values()
        ),
        "total_row_count": sum(
            int(record["row_count"]) for record in split_records.values()
        ),
        "all_episode_sha256_unique": True,
        "split_independence": {
            "method": "collected_time_window",
            "episode_and_exact_content_disjoint": True,
            "vehicle_disjoint": False,
            "pairwise": pairwise_split_checks,
        },
        "splits": split_records,
    }


def main() -> None:
    args = parse_args()
    collection_root = args.collection_root.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite expert audit: {output}")
    splits = normalize_requested_splits(args.splits)
    result = audit_collection(collection_root, splits=splits)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
