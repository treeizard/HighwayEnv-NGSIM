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
    validate_expert_action_contract,
)


DOMAINS = ("us", "japanese")
SPLITS = ("train", "val", "test")


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
        if observations.ndim != 2 or observations.shape[1] != 323:
            raise ValueError(f"Unexpected raw observations {observations.shape}: {path}")
        if next_observations.shape != observations.shape:
            raise ValueError(
                f"Observation transition mismatch {observations.shape} -> "
                f"{next_observations.shape}: {path}"
            )
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
    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "action_contract_validation_sha256": sha256_file(validation_path),
        "episode_count": len(files),
        "row_count": rows,
        "continuous_action_contract": manifest_action,
        "policy_observation_contract": manifest_observation,
        "files": file_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_root = args.collection_root.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite expert audit: {output}")

    reference_observation = policy_observation_contract(
        lidar_cells=128,
        maximum_range=64.0,
    )
    split_records: dict[str, Any] = {}
    reference_action: dict[str, Any] | None = None
    for domain in DOMAINS:
        for split in SPLITS:
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

    result = {
        "schema_version": 1,
        "status": "passed",
        "collection_root": str(collection_root),
        "domain_split_count": len(split_records),
        "continuous_action_contract": reference_action,
        "policy_observation_contract": reference_observation,
        "total_episode_count": sum(
            int(record["episode_count"]) for record in split_records.values()
        ),
        "total_row_count": sum(
            int(record["row_count"]) for record in split_records.values()
        ),
        "splits": split_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
