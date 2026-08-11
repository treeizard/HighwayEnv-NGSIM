import argparse
import json
from pathlib import Path

import numpy as np
from highway_env.imitation.observation_aligned_reflex import (
    observation_aligned_reflex_contract,
)
from policy.data.observation_aligned_aggregate import (
    build,
    select_trajectory_rows,
)
from policy.data.observation_aligned_dagger import dagger_reset_seed


def _write_episode(
    path: Path,
    *,
    episode_name: str,
    vehicle_ids: list[int],
) -> None:
    rows = len(vehicle_ids)
    metadata = {
        "schema_version": 3,
        "scene": "us-101",
        "prebuilt_split": "train",
        "episode_name": episode_name,
        "target_generation_contract": observation_aligned_reflex_contract(),
    }
    actions = np.linspace(-0.2, 0.2, rows * 2, dtype=np.float32).reshape(rows, 2)
    arrays = {
        "observations": np.zeros((rows, 323), dtype=np.float32),
        "next_observations": np.zeros((rows, 323), dtype=np.float32),
        "trajectory_states": np.zeros((rows, 3), dtype=np.float32),
        "vehicle_ids": np.asarray(vehicle_ids, dtype=np.int64),
        "timesteps": np.arange(rows, dtype=np.int64),
        "dones": np.zeros(rows, dtype=bool),
        "rewards": np.zeros(rows, dtype=np.float32),
        "actions_continuous_env": actions,
        "actions_steering_acceleration": np.stack(
            [actions[:, 1] * (np.pi / 4.0), actions[:, 0] * 5.0],
            axis=1,
        ).astype(np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    np.savez_compressed(path, **arrays)


def _write_manifest(
    root: Path,
    *,
    kind: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "dataset_kind": kind,
        "scene": "us-101",
        "prebuilt_split": "train",
        "episodes": rows,
        "num_episodes": len(rows),
        "num_samples": sum(int(row["num_samples"]) for row in rows),
        "target_generation_contract": observation_aligned_reflex_contract(),
        "test_split_opened": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def test_dagger_reset_seed_is_policy_seed_independent():
    assert dagger_reset_seed(scenario_seed=1000, episode_index=7) == 1007


def test_trajectory_selection_is_deterministic_and_keeps_whole_trajectories(
    tmp_path: Path,
):
    root = tmp_path / "base"
    root.mkdir()
    source = root / "episode.npz"
    _write_episode(
        source,
        episode_name="train_a",
        vehicle_ids=[1, 1, 1, 2, 2, 3, 3, 3, 3],
    )
    manifest = _write_manifest(
        root,
        kind="observation_aligned_synthetic_reflex_collection",
        rows=[
            {
                "dataset_file": source.name,
                "episode_name": "train_a",
                "num_samples": 9,
            }
        ],
    )
    first = select_trajectory_rows(root, manifest, row_budget=4, seed=17)
    second = select_trajectory_rows(root, manifest, row_budget=4, seed=17)
    np.testing.assert_array_equal(first[source], second[source])
    selected_ids = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3])[first[source]]
    for vehicle_id in np.unique(selected_ids):
        assert np.count_nonzero(selected_ids == vehicle_id) in {2, 3, 4}


def test_aggregate_contains_bounded_base_and_all_train_only_dagger_rows(
    tmp_path: Path,
):
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    output = tmp_path / "aggregate"
    base.mkdir()
    dagger.mkdir()
    _write_episode(
        base / "base_a.npz",
        episode_name="train_a",
        vehicle_ids=[1, 1, 1, 2, 2, 2],
    )
    _write_episode(
        dagger / "dagger_a.npz",
        episode_name="train_a__dagger_1000",
        vehicle_ids=[10, 10, 10, 11, 11],
    )
    _write_manifest(
        base,
        kind="observation_aligned_synthetic_reflex_collection",
        rows=[
            {
                "dataset_file": "base_a.npz",
                "episode_name": "train_a",
                "num_samples": 6,
            }
        ],
    )
    _write_manifest(
        dagger,
        kind="observation_aligned_dagger_collection",
        rows=[
            {
                "dataset_file": "dagger_a.npz",
                "episode_name": "train_a__dagger_1000",
                "num_samples": 5,
            }
        ],
    )
    receipt = build(
        argparse.Namespace(
            base_root=base,
            dagger_root=dagger,
            base_row_budget=3,
            seed=5,
            out_root=output,
        )
    )
    assert receipt["test_split_opened"] is False
    assert receipt["selected_base_rows"] == 3
    assert receipt["dagger_rows"] == 5
    assert receipt["num_samples"] == 8
    assert {row["aggregation_role"] for row in receipt["episodes"]} == {
        "base_subset",
        "on_policy_teacher_label",
    }
