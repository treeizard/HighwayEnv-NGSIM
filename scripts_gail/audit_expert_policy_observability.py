#!/usr/bin/env python3
"""Audit whether stored expert actions use information absent from BC inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts_gail.ps_gail.data import load_expert_transition_data


DOMAINS = ("us", "japanese")
SPLITS = {
    "train": ("train", 300_000),
    "validation": ("val", 100_000),
    "test": ("test", 100_000),
}
HORIZONS = (1, 3, 5, 10, 15, 20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_segments(transitions: Any) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, trajectory_id in enumerate(transitions.trajectory_ids):
        grouped.setdefault(str(trajectory_id), []).append(index)
    timesteps = np.asarray(transitions.timesteps, dtype=np.int64)
    dones = np.asarray(transitions.dones, dtype=bool)
    segments: list[np.ndarray] = []
    for indices_list in grouped.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        indices = indices[np.argsort(timesteps[indices], kind="stable")]
        boundaries = np.flatnonzero(
            dones[indices[:-1]]
            | (np.diff(timesteps[indices]) != 1)
        ) + 1
        segments.extend(
            segment
            for segment in np.split(indices, boundaries)
            if len(segment)
        )
    return segments


def future_bearing_evidence(transitions: Any) -> dict[str, Any]:
    positions = np.asarray(
        transitions.trajectory_states[:, :2],
        dtype=np.float64,
    )
    headings = np.asarray(
        transitions.policy_observations[:, -1],
        dtype=np.float64,
    )
    steering = np.asarray(
        transitions.actions_continuous_env[:, 1],
        dtype=np.float64,
    )
    segments = trajectory_segments(transitions)
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        bearing_parts: list[np.ndarray] = []
        steering_parts: list[np.ndarray] = []
        for segment in segments:
            if len(segment) <= horizon:
                continue
            current = segment[:-horizon]
            future = segment[horizon:]
            displacement = positions[future] - positions[current]
            bearing = (
                np.arctan2(displacement[:, 1], displacement[:, 0])
                - headings[current]
            )
            bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi
            bearing_parts.append(bearing)
            steering_parts.append(steering[current])
        bearings = np.concatenate(bearing_parts)
        steering_values = np.concatenate(steering_parts)
        correlation = (
            float(np.corrcoef(bearings, steering_values)[0, 1])
            if np.std(bearings) > 0.0 and np.std(steering_values) > 0.0
            else 0.0
        )
        result[str(horizon)] = {
            "paired_rows": int(len(bearings)),
            "future_bearing_steering_correlation": correlation,
            "future_bearing_std_rad": float(np.std(bearings)),
            "steering_std_normalized": float(np.std(steering_values)),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20260716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite observability audit: {output}")
    collection_root = args.collection_root.resolve()
    component_root = Path(__file__).resolve().parents[1]
    expert_mixin = (
        component_root
        / "highway_env"
        / "ngsim_utils"
        / "expert"
        / "ngsim_expert_mixin.py"
    )
    tracker_source = (
        component_root
        / "highway_env"
        / "ngsim_utils"
        / "expert"
        / "trajectory_to_action.py"
    )

    domain_results: dict[str, Any] = {}
    policy_contract: dict[str, Any] | None = None
    for domain in DOMAINS:
        domain_results[domain] = {}
        for offset, (split, (directory, limit)) in enumerate(SPLITS.items()):
            transitions = load_expert_transition_data(
                str(collection_root / domain / directory),
                max_samples=int(limit),
                seed=int(args.data_seed) + offset,
                trajectory_frame="relative",
            )
            if policy_contract is None:
                policy_contract = transitions.metadata.get(
                    "policy_observation_contract"
                )
            domain_results[domain][split] = {
                "loaded_rows": int(len(transitions.policy_observations)),
                "source_rows": int(
                    transitions.metadata["num_source_samples"]
                ),
                "future_path_proxy": future_bearing_evidence(transitions),
            }

    result = {
        "schema_version": 1,
        "status": "failed_observability_equivalence",
        "collection_root": str(collection_root),
        "expert_label_generator": {
            "kind": "time_anchored_pure_pursuit_tracker",
            "expert_mixin_source": str(expert_mixin),
            "expert_mixin_sha256": sha256_file(expert_mixin),
            "tracker_source": str(tracker_source),
            "tracker_source_sha256": sha256_file(tracker_source),
            "steering_inputs": [
                "current_position",
                "current_heading",
                "current_speed",
                "hidden_future_reference_xy",
                "hidden_tracker_time_index",
                "hidden_previous_steering",
            ],
        },
        "bc_policy_observation_contract": policy_contract,
        "missing_from_bc_policy_observation": [
            "future_reference_xy",
            "tracker_time_index",
            "previous_tracker_steering_state",
        ],
        "empirical_cross_check": {
            "proxy": (
                "bearing from current pose to the same trajectory at a future "
                "horizon; this is diagnostic only and is not a deployable input"
            ),
            "domains": domain_results,
        },
        "conclusion": (
            "The expert and BC policy are not conditioned on equivalent "
            "information. Offline BC can support an observation-conditioned "
            "imitation claim only after the expert is restricted to the policy "
            "inputs, or a deployable route/waypoint input is added to both."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
