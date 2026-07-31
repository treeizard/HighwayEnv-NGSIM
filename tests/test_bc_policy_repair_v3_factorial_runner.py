from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts_gail.run_bc_policy_repair_v3_factorial import (
    EXPECTED_COLLECTION_ID,
    build_data_lock,
    build_training_command,
    enumerate_cells,
    expert_split_roots,
    prebuilt_train_validation_files,
    read_json,
    validate_audit_receipt,
    validate_study_config,
)


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    COMPONENT_ROOT
    / "configs"
    / "bc_policy_repair_v3_factorial_seed0.json"
)


def locked_config() -> dict[str, object]:
    return read_json(CONFIG_PATH)


def test_locked_v3_factorial_has_eight_unique_seed0_cells() -> None:
    config = locked_config()
    validate_study_config(config)
    cells = enumerate_cells(config)

    assert len(cells) == len({cell.cell_id for cell in cells}) == 8
    assert {cell.seed for cell in cells} == {0}
    assert {
        (cell.policy_model, cell.depth, cell.action_loss_weighting)
        for cell in cells
    } == {
        ("recurrent_gru", 1, "fixed"),
        ("recurrent_gru", 1, "inverse_variance"),
        ("recurrent_transformer", 2, "fixed"),
        ("recurrent_transformer", 2, "inverse_variance"),
    }
    assert all(
        f"weighting_{cell.action_loss_weighting}" in cell.cell_id
        for cell in cells
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("runtime_invariants", "recurrent_warmup_mode"), "bounded_raw_history"),
        (("runtime_invariants", "evaluation_scenario_seed"), 9),
        (("runtime_invariants", "policy_action_substitution"), True),
        (
            (
                "evaluation",
                "expert_replay_qualification",
                "maximum_vehicle_crash_rate_gap",
            ),
            0.0,
        ),
        (("factorial_design", "policy_seeds"), [0, 1]),
    ],
)
def test_locked_v3_factorial_rejects_protocol_drift(
    path: tuple[str, ...],
    value: object,
) -> None:
    config = json.loads(json.dumps(locked_config()))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="Invalid prospective"):
        validate_study_config(config)


def test_every_training_command_is_explicitly_validation_only_and_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = locked_config()
    expert_root = tmp_path / EXPECTED_COLLECTION_ID
    episode_root = tmp_path / "processed_20s"
    cells = enumerate_cells(config)

    for cell in cells:
        command = build_training_command(
            config,
            cell,
            python=Path(sys.executable),
            expert_root=expert_root,
            episode_root=episode_root,
            cell_root=tmp_path / "out" / cell.cell_id,
            device="cpu",
        )
        assert "--expert-test-data" not in command
        assert command[command.index("--test-evaluation-mode") + 1] == "deferred"
        assert command[command.index("--evaluation-split") + 1] == "val"
        assert command[command.index("--recurrent-warmup-mode") + 1] == "full_prefix"
        assert command[command.index("--evaluation-scenario-seed") + 1] == "20260716"
        assert command[
            command.index("--maximum-expert-replay-crash-rate-gap") + 1
        ] == "0.05"
        assert command[
            command.index("--maximum-expert-replay-offroad-rate-gap") + 1
        ] == "0.05"
        assert "--evaluation-enable-collision" in command
        assert "--matched-evaluation" in command
        assert "--evaluation-action-override" not in command
        assert "--policy-action-substitution" not in command

        # Parse the constructed argument vector with the real training parser.
        # This is static CLI validation: it opens no data and starts no model.
        from scripts_gail.train_recurrent_bc_policy import parse_args

        monkeypatch.setattr(
            sys,
            "argv",
            ["train_recurrent_bc_policy.py", *command[3:]],
        )
        parsed = parse_args()
        assert parsed.expert_test_data == ""
        assert parsed.test_evaluation_mode == "deferred"
        assert parsed.recurrent_warmup_mode == "full_prefix"
        assert parsed.evaluation_scenario_seed == 20260716
        assert parsed.action_loss_weighting == cell.action_loss_weighting
        assert parsed.checkpoint_selection_rule == "validation_loss"
        assert parsed.evaluation_enable_collision is True


def test_only_explicit_train_and_val_paths_are_constructed(tmp_path: Path) -> None:
    expert_root = tmp_path / EXPECTED_COLLECTION_ID
    roots = expert_split_roots(expert_root)

    assert set(roots) == {"us", "japanese"}
    assert all(set(domain_roots) == {"train", "val"} for domain_roots in roots.values())
    assert not any(path.name == "test" for domain_roots in roots.values() for path in domain_roots.values())

    prebuilt = prebuilt_train_validation_files(tmp_path / "processed_20s")
    assert len(prebuilt) == 8
    assert all(path.name.endswith(("_train.npy", "_val.npy")) for path in prebuilt)
    assert not any("_test.npy" in path.name for path in prebuilt)


def test_audit_receipt_must_keep_test_unopened(tmp_path: Path) -> None:
    expert_root = tmp_path / EXPECTED_COLLECTION_ID
    split_records = {}
    for domain in ("us", "japanese"):
        for split in ("train", "val"):
            key = f"{domain}/{split}"
            split_records[key] = {
                "root": str((expert_root / key).resolve()),
                "numeric_completeness": {
                    "declared_raw_observation_space_passed": True,
                },
            }
    receipt = {
        "status": "passed",
        "collection_root": str(expert_root.resolve()),
        "audited_splits": ["train", "val"],
        "not_opened_splits": ["test"],
        "test_data_status": "not_opened",
        "domain_split_count": 4,
        "all_episode_sha256_unique": True,
        "continuous_action_contract": {
            "normalized_columns": [
                "acceleration_norm",
                "steering_norm",
            ],
            "physical_columns": [
                "steering_rad",
                "acceleration_mps2",
            ],
            "physical_index_for_normalized": [1, 0],
            "scales": [5.0, 0.7853981633974483],
            "maximum_absolute_residual": 0.0,
        },
        "policy_observation_contract": {
            "schema_version": 2,
            "raw_observation_dim": 323,
            "policy_observation_dim": 322,
            "lidar_cells": 128,
            "lane_camera_cells": 21,
            "maximum_range_m": 64.0,
        },
        "split_independence": {
            "episode_and_exact_content_disjoint": True,
        },
        "splits": split_records,
    }
    validate_audit_receipt(receipt, expert_root=expert_root)

    bad_action = json.loads(json.dumps(receipt))
    bad_action["continuous_action_contract"]["scales"][0] = 10.0
    with pytest.raises(ValueError, match="action scales"):
        validate_audit_receipt(bad_action, expert_root=expert_root)

    bad_observation = json.loads(json.dumps(receipt))
    bad_observation["policy_observation_contract"]["policy_observation_dim"] = 323
    with pytest.raises(ValueError, match="observation dimensions"):
        validate_audit_receipt(bad_observation, expert_root=expert_root)

    opened_test = json.loads(json.dumps(receipt))
    opened_test["audited_splits"] = ["train", "val", "test"]
    opened_test["test_data_status"] = "audited"
    with pytest.raises(ValueError, match="test"):
        validate_audit_receipt(opened_test, expert_root=expert_root)


def test_synthetic_source_and_data_lock_detects_drift(tmp_path: Path) -> None:
    expert_root = tmp_path / EXPECTED_COLLECTION_ID
    for domain in ("us", "japanese"):
        for split in ("train", "val"):
            root = expert_root / domain / split
            root.mkdir(parents=True)
            (root / "manifest.json").write_text("{}\n", encoding="utf-8")
            (root / "episode.npz").write_bytes(f"{domain}-{split}".encode())

    episode_root = tmp_path / "processed_20s"
    for path in prebuilt_train_validation_files(episode_root):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    audit = expert_root / "collection_contract_audit_train_val.json"
    audit.write_text('{"status":"passed"}\n', encoding="utf-8")

    before = build_data_lock(
        expert_root=expert_root,
        episode_root=episode_root,
        audit_path=audit,
    )
    assert before["test_data_status"] == "not_constructed_not_listed_not_opened"
    assert set(before["expert_split_trees"]) == {
        "us/train",
        "us/val",
        "japanese/train",
        "japanese/val",
    }

    (expert_root / "us" / "train" / "episode.npz").write_bytes(b"changed")
    after = build_data_lock(
        expert_root=expert_root,
        episode_root=episode_root,
        audit_path=audit,
    )
    assert before != after
