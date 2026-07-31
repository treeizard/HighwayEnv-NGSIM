#!/usr/bin/env python3
"""Run the prospective v3 BC architecture-by-loss validation screen.

This is deliberately a validation-only engineering runner.  It has no test
source argument, constructs only train/validation paths, executes every
prespecified cell even when its metrics are poor, and never modifies a policy
action.  Publication qualification and locked-test evaluation are separate
future stages.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


EXPECTED_COLLECTION_ID = "domain_matched_accel5_v3_paddingfix"
EXPECTED_DOMAIN_SCENES = {
    "us": "us-101",
    "japanese": "japanese",
}
EXPECTED_FACTORIAL_ARMS = {
    ("recurrent_gru", 1, "fixed"),
    ("recurrent_gru", 1, "inverse_variance"),
    ("recurrent_transformer", 2, "fixed"),
    ("recurrent_transformer", 2, "inverse_variance"),
}
TRAINING_SPLITS = ("train", "val")


@dataclass(frozen=True)
class TrialCell:
    cell_id: str
    domain: str
    scene: str
    arm_id: str
    policy_model: str
    depth: int
    action_loss_weighting: str
    seed: int


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(
    condition: bool,
    message: str,
    errors: list[str],
) -> None:
    if not condition:
        errors.append(message)


def validate_study_config(config: dict[str, Any]) -> None:
    """Reject drift from the prespecified seed-0 factorial protocol."""

    errors: list[str] = []
    _require(config.get("schema_version") == 1, "schema_version must be 1", errors)
    _require(config.get("status") == "locked", "status must be locked", errors)
    _require(
        config.get("evidence_level") == "engineering_discovery_validation_only",
        "evidence_level must remain engineering_discovery_validation_only",
        errors,
    )
    _require(
        config.get("scientific_claim_allowed") is False,
        "seed-0 discovery must not allow a scientific claim",
        errors,
    )

    collection = config.get("collection")
    _require(isinstance(collection, dict), "collection must be an object", errors)
    if isinstance(collection, dict):
        _require(
            collection.get("id") == EXPECTED_COLLECTION_ID,
            f"collection.id must be {EXPECTED_COLLECTION_ID}",
            errors,
        )
        _require(
            collection.get("audited_splits") == ["train", "val"],
            "audited_splits must be exactly train and val",
            errors,
        )
        _require(
            collection.get("not_opened_splits") == ["test"],
            "test must remain the sole not-opened split",
            errors,
        )
        _require(
            collection.get("test_data_status") == "not_opened",
            "test_data_status must be not_opened",
            errors,
        )
        domain_scenes = {
            str(row.get("id")): str(row.get("scene"))
            for row in collection.get("domains", [])
            if isinstance(row, dict)
        }
        _require(
            domain_scenes == EXPECTED_DOMAIN_SCENES,
            f"domain/scene mapping must be {EXPECTED_DOMAIN_SCENES}",
            errors,
        )

    design = config.get("factorial_design")
    _require(isinstance(design, dict), "factorial_design must be an object", errors)
    if isinstance(design, dict):
        _require(
            design.get("policy_seeds") == [0],
            "initial discovery must use exactly policy seed 0",
            errors,
        )
        _require(
            design.get("phase") == "seed0_discovery",
            "phase must be seed0_discovery",
            errors,
        )
        _require(
            design.get("all_cells_mandatory") is True,
            "all factorial cells must be mandatory",
            errors,
        )
        _require(
            design.get("stop_on_poor_metric") is False,
            "poor metrics must not terminate independent cells",
            errors,
        )
        _require(
            design.get("selection_split") == "val",
            "selection split must be val",
            errors,
        )
        _require(
            design.get("winner_selection_allowed") is False,
            "seed-0 winner selection must remain disabled",
            errors,
        )
        arms = design.get("arms")
        _require(
            isinstance(arms, list) and len(arms) == 4,
            "factorial_design.arms must contain four arms",
            errors,
        )
        if isinstance(arms, list):
            realized = {
                (
                    str(arm.get("policy_model")),
                    int(arm.get("depth", -1)),
                    str(arm.get("action_loss_weighting")),
                )
                for arm in arms
                if isinstance(arm, dict)
            }
            arm_ids = [
                str(arm.get("id"))
                for arm in arms
                if isinstance(arm, dict)
            ]
            _require(
                realized == EXPECTED_FACTORIAL_ARMS,
                f"factorial arms must be exactly {sorted(EXPECTED_FACTORIAL_ARMS)}",
                errors,
            )
            _require(
                len(arm_ids) == len(set(arm_ids)) == 4,
                "factorial arm IDs must be unique",
                errors,
            )

    invariants = config.get("runtime_invariants")
    _require(
        isinstance(invariants, dict),
        "runtime_invariants must be an object",
        errors,
    )
    if isinstance(invariants, dict):
        expected_invariants = {
            "policy_action_execution": (
                "network_output_passed_unchanged_to_env_step"
            ),
            "policy_action_clamping": False,
            "policy_action_substitution": False,
            "collision_physics_enabled": True,
            "collision_termination_enabled": False,
            "test_evaluation_mode": "deferred",
            "evaluation_split": "val",
            "recurrent_warmup_mode": "full_prefix",
            "evaluation_scenario_seed": 20260716,
            "observation_standardization_clip": 0.0,
        }
        for key, expected in expected_invariants.items():
            _require(
                invariants.get(key) == expected,
                f"runtime_invariants.{key} must be {expected!r}",
                errors,
            )

    data = config.get("data")
    _require(isinstance(data, dict), "data must be an object", errors)
    if isinstance(data, dict):
        _require(
            data.get("continuous_action_columns")
            == ["acceleration_norm", "steering_norm"],
            "continuous action order must be acceleration then steering",
            errors,
        )
        _require(
            data.get("continuous_action_scales")
            == [5.0, 0.7853981633974483],
            "continuous action scales must remain [5, pi/4]",
            errors,
        )
        _require(
            data.get("policy_observation_dim") == 322,
            "policy observation dimension must be 322",
            errors,
        )
        _require(
            data.get("require_explicit_data_contracts") is True,
            "explicit data contracts are mandatory",
            errors,
        )
        _require(
            data.get("max_test_samples") == 0,
            "max_test_samples must be zero",
            errors,
        )

    optimization = config.get("optimization")
    _require(
        isinstance(optimization, dict),
        "optimization must be an object",
        errors,
    )
    if isinstance(optimization, dict):
        _require(
            optimization.get("action_loss_weights") == [1.0, 1.0],
            "base action-loss weights must be [1, 1] in every arm",
            errors,
        )
        _require(
            optimization.get("checkpoint_selection_rule")
            == "validation_loss",
            (
                "checkpoint selection must use the same unweighted validation "
                "MSE in every factorial arm"
            ),
            errors,
        )

    gates = config.get("gates")
    _require(isinstance(gates, dict), "gates must be an object", errors)
    if isinstance(gates, dict):
        _require(
            gates.get("role") == "report_only_during_seed0_discovery",
            "seed-0 gates must be report-only",
            errors,
        )

    evaluation = config.get("evaluation")
    _require(isinstance(evaluation, dict), "evaluation must be an object", errors)
    if isinstance(evaluation, dict):
        _require(
            evaluation.get("matched_evaluation") is True,
            "matched validation evaluation is required",
            errors,
        )
        _require(
            evaluation.get("vehicle_mode") == "single",
            "vehicle_mode must be single",
            errors,
        )
        qualification = evaluation.get("expert_replay_qualification")
        _require(
            isinstance(qualification, dict),
            "expert_replay_qualification must be an object",
            errors,
        )
        if isinstance(qualification, dict):
            for key in (
                "maximum_vehicle_crash_rate_gap",
                "maximum_vehicle_offroad_rate_gap",
            ):
                value = qualification.get(key)
                _require(
                    isinstance(value, (float, int))
                    and math.isclose(float(value), 0.05, abs_tol=1.0e-12),
                    f"{key} must be 0.05",
                    errors,
                )
            _require(
                qualification.get("required_for_policy_realism") is True,
                "expert-relative qualification must remain required",
                errors,
            )

    reporting = config.get("reporting")
    _require(isinstance(reporting, dict), "reporting must be an object", errors)
    if isinstance(reporting, dict):
        _require(
            reporting.get("report_all_cells") is True,
            "all cells must be reported",
            errors,
        )
        _require(
            reporting.get("publication_qualification_in_this_phase") is False,
            "this phase cannot grant publication qualification",
            errors,
        )

    if errors:
        raise ValueError(
            "Invalid prospective v3 factorial configuration:\n- "
            + "\n- ".join(errors)
        )


def enumerate_cells(config: dict[str, Any]) -> list[TrialCell]:
    validate_study_config(config)
    domains = config["collection"]["domains"]
    arms = config["factorial_design"]["arms"]
    seeds = config["factorial_design"]["policy_seeds"]
    cells: list[TrialCell] = []
    for domain_row in domains:
        domain = str(domain_row["id"])
        scene = str(domain_row["scene"])
        for arm in arms:
            arm_id = str(arm["id"])
            weighting = str(arm["action_loss_weighting"])
            for seed in seeds:
                cell_id = (
                    f"{domain}__{arm_id}__weighting_{weighting}"
                    f"__seed_{int(seed)}"
                )
                cells.append(
                    TrialCell(
                        cell_id=cell_id,
                        domain=domain,
                        scene=scene,
                        arm_id=arm_id,
                        policy_model=str(arm["policy_model"]),
                        depth=int(arm["depth"]),
                        action_loss_weighting=weighting,
                        seed=int(seed),
                    )
                )
    if len(cells) != 8 or len({cell.cell_id for cell in cells}) != 8:
        raise RuntimeError("Expected eight unique seed-0 factorial cells.")
    return cells


def expert_split_roots(
    expert_root: Path,
) -> dict[str, dict[str, Path]]:
    """Construct train/val paths only.  There is intentionally no test key."""

    resolved = expert_root.resolve()
    if resolved.name != EXPECTED_COLLECTION_ID:
        raise ValueError(
            f"Expert root must end in {EXPECTED_COLLECTION_ID}: {resolved}"
        )
    return {
        domain: {
            split: resolved / domain / split
            for split in TRAINING_SPLITS
        }
        for domain in EXPECTED_DOMAIN_SCENES
    }


def validate_audit_receipt(
    payload: dict[str, Any],
    *,
    expert_root: Path,
) -> None:
    expected_root = expert_root.resolve()
    errors: list[str] = []
    _require(payload.get("status") == "passed", "audit status must be passed", errors)
    _require(
        payload.get("collection_root") == str(expected_root),
        "audit collection_root does not match the requested v3 root",
        errors,
    )
    _require(
        payload.get("audited_splits") == ["train", "val"],
        "audit must have opened exactly train and val",
        errors,
    )
    _require(
        payload.get("not_opened_splits") == ["test"],
        "audit must report test as not opened",
        errors,
    )
    _require(
        payload.get("test_data_status") == "not_opened",
        "audit test_data_status must be not_opened",
        errors,
    )
    _require(
        payload.get("domain_split_count") == 4,
        "audit must contain four domain/split cells",
        errors,
    )
    _require(
        payload.get("all_episode_sha256_unique") is True,
        "audit must confirm unique episode hashes",
        errors,
    )
    action_contract = payload.get("continuous_action_contract")
    _require(
        isinstance(action_contract, dict),
        "audit continuous_action_contract must be an object",
        errors,
    )
    if isinstance(action_contract, dict):
        _require(
            action_contract.get("normalized_columns")
            == ["acceleration_norm", "steering_norm"],
            "audit normalized action order must be acceleration then steering",
            errors,
        )
        _require(
            action_contract.get("physical_columns")
            == ["steering_rad", "acceleration_mps2"],
            "audit physical action order must be steering then acceleration",
            errors,
        )
        _require(
            action_contract.get("physical_index_for_normalized") == [1, 0],
            "audit normalized-to-physical action mapping must be [1, 0]",
            errors,
        )
        scales = action_contract.get("scales")
        _require(
            isinstance(scales, list)
            and len(scales) == 2
            and math.isclose(float(scales[0]), 5.0, abs_tol=1.0e-8)
            and math.isclose(
                float(scales[1]),
                math.pi / 4.0,
                abs_tol=1.0e-8,
            ),
            "audit action scales must be [5, pi/4]",
            errors,
        )
        residual = action_contract.get("maximum_absolute_residual")
        _require(
            isinstance(residual, (float, int))
            and math.isfinite(float(residual))
            and float(residual) <= 1.0e-6,
            "audit action-contract residual must be finite and <= 1e-6",
            errors,
        )
    observation_contract = payload.get("policy_observation_contract")
    _require(
        isinstance(observation_contract, dict),
        "audit policy_observation_contract must be an object",
        errors,
    )
    if isinstance(observation_contract, dict):
        _require(
            observation_contract.get("schema_version") == 2,
            "audit observation contract must use bounded-sensor schema 2",
            errors,
        )
        _require(
            observation_contract.get("raw_observation_dim") == 323
            and observation_contract.get("policy_observation_dim") == 322,
            "audit observation dimensions must be raw 323 and policy 322",
            errors,
        )
        _require(
            observation_contract.get("lidar_cells") == 128
            and observation_contract.get("lane_camera_cells") == 21
            and math.isclose(
                float(observation_contract.get("maximum_range_m", math.nan)),
                64.0,
                abs_tol=1.0e-12,
            ),
            "audit sensor geometry must be lidar 128, camera 21, range 64 m",
            errors,
        )
    split_independence = payload.get("split_independence")
    _require(
        isinstance(split_independence, dict)
        and split_independence.get(
            "episode_and_exact_content_disjoint"
        )
        is True,
        "audit must confirm episode/content-disjoint train and val",
        errors,
    )
    split_records = payload.get("splits")
    _require(isinstance(split_records, dict), "audit splits must be an object", errors)
    if isinstance(split_records, dict):
        expected_keys = {
            f"{domain}/{split}"
            for domain in EXPECTED_DOMAIN_SCENES
            for split in TRAINING_SPLITS
        }
        _require(
            set(split_records) == expected_keys,
            f"audit split keys must be {sorted(expected_keys)}",
            errors,
        )
        for key in sorted(expected_keys.intersection(split_records)):
            record = split_records[key]
            expected_split_root = expected_root / key
            numeric = (
                record.get("numeric_completeness")
                if isinstance(record, dict)
                else None
            )
            _require(
                isinstance(record, dict)
                and record.get("root") == str(expected_split_root),
                f"audit root mismatch for {key}",
                errors,
            )
            _require(
                isinstance(numeric, dict)
                and numeric.get("declared_raw_observation_space_passed") is True,
                f"raw observation-space audit did not pass for {key}",
                errors,
            )
    if errors:
        raise ValueError(
            "Invalid v3 train/validation audit receipt:\n- "
            + "\n- ".join(errors)
        )


def file_set_receipt(paths: Iterable[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted({item.resolve() for item in paths}, key=str):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"Provenance requires a regular non-symlink file: {path}"
            )
        records.append(
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    receipt = {
        "algorithm": (
            "sha256 over complete files and canonical JSON over sorted records"
        ),
        "file_count": len(records),
        "files": records,
    }
    receipt["set_sha256"] = canonical_sha256(records)
    return receipt


def data_tree_receipt(root: Path) -> dict[str, Any]:
    """Hash one explicit train or val tree without listing its parent."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing explicit data tree: {resolved}")
    records: list[dict[str, Any]] = []
    for path in sorted(
        resolved.rglob("*"),
        key=lambda item: item.relative_to(resolved).as_posix(),
    ):
        relative = path.relative_to(resolved).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"Data lock refuses symlink: {resolved}/{relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                f"Data lock found non-regular entry: {resolved}/{relative}"
            )
        records.append(
            {
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise RuntimeError(f"Data tree is empty: {resolved}")
    receipt = {
        "root": str(resolved),
        "algorithm": (
            "sha256 over every regular file in this explicit split tree"
        ),
        "file_count": len(records),
        "total_size_bytes": sum(row["size_bytes"] for row in records),
        "files": records,
    }
    receipt["tree_sha256"] = canonical_sha256(records)
    return receipt


def prebuilt_train_validation_files(episode_root: Path) -> list[Path]:
    """Name exact train/val files; never enumerate or resolve a test path."""

    root = episode_root.resolve()
    paths: list[Path] = []
    for scene in EXPECTED_DOMAIN_SCENES.values():
        prebuilt = root / scene / "prebuilt"
        for split in TRAINING_SPLITS:
            paths.extend(
                [
                    prebuilt / f"veh_ids_{split}.npy",
                    prebuilt / f"trajectory_{split}.npy",
                ]
            )
    return paths


def source_paths(component_root: Path, config_path: Path) -> list[Path]:
    relative_paths = [
        "scripts_gail/run_bc_policy_repair_v3_factorial.py",
        "scripts_gail/train_recurrent_bc_policy.py",
        "scripts_gail/ps_gail/recurrent_bc.py",
        "scripts_gail/ps_gail/models.py",
        "scripts_gail/ps_gail/config.py",
        "scripts_gail/ps_gail/data.py",
        "scripts_gail/ps_gail/checkpoints.py",
        "scripts_gail/ps_gail/contracts.py",
        "scripts_gail/ps_gail/envs.py",
        "scripts_gail/ps_gail/observations.py",
        "scripts_gail/ps_gail/trainer.py",
        "scripts_gail/ps_gail/validation.py",
        "scripts_gail/ps_gail/training/evaluation.py",
        "scripts_gail/ps_gail/training/policy.py",
        "highway_env/utils.py",
        "highway_env/imitation/expert_dataset.py",
        "highway_env/envs/common/abstract.py",
        "highway_env/envs/common/action.py",
        "highway_env/envs/common/observation.py",
        "highway_env/envs/common/observations/lidar.py",
        "highway_env/envs/ngsim_env.py",
        "highway_env/road/lane.py",
        "highway_env/road/road.py",
        "highway_env/vehicle/kinematics.py",
        "highway_env/vehicle/controller.py",
        "highway_env/vehicle/behavior.py",
        "highway_env/ngsim_utils/core/constants.py",
        "highway_env/ngsim_utils/data/prebuilt.py",
        "highway_env/ngsim_utils/data/trajectory_gen.py",
        "highway_env/ngsim_utils/vehicles/ego.py",
        "highway_env/ngsim_utils/vehicles/replay.py",
    ]
    return [component_root / relative for relative in relative_paths] + [
        config_path.resolve()
    ]


def build_data_lock(
    *,
    expert_root: Path,
    episode_root: Path,
    audit_path: Path,
) -> dict[str, Any]:
    roots = expert_split_roots(expert_root)
    return {
        "expert_split_trees": {
            f"{domain}/{split}": data_tree_receipt(path)
            for domain, domain_roots in roots.items()
            for split, path in domain_roots.items()
        },
        "prebuilt_train_validation_files": file_set_receipt(
            prebuilt_train_validation_files(episode_root)
        ),
        "audit_receipt": file_set_receipt([audit_path]),
        "test_data_status": "not_constructed_not_listed_not_opened",
    }


def verify_audit_file_binding(
    audit: dict[str, Any],
    *,
    expert_root: Path,
) -> None:
    """Match the passed audit hashes to the bytes that training will open."""

    for key, record in sorted(audit["splits"].items()):
        split_root = expert_root.resolve() / key
        manifest_path = split_root / "manifest.json"
        validation_path = split_root / "action_contract_validation.json"
        if sha256_file(manifest_path) != record.get("manifest_sha256"):
            raise RuntimeError(f"Manifest changed after audit: {manifest_path}")
        if (
            sha256_file(validation_path)
            != record.get("action_contract_validation_sha256")
        ):
            raise RuntimeError(
                f"Action/observation validation changed after audit: {validation_path}"
            )
        expected_files = {
            str(row["file"]): str(row["sha256"])
            for row in record.get("files", [])
        }
        actual_files = {
            name: sha256_file(split_root / name)
            for name in sorted(expected_files)
        }
        if actual_files != expected_files:
            raise RuntimeError(f"Expert dataset changed after audit: {split_root}")


def scenario_pairing_receipt(
    config: dict[str, Any],
    *,
    episode_root: Path,
) -> dict[str, Any]:
    """Materialize the exact validation scenario lists shared by every arm."""

    from scripts_gail.ps_gail.config import PSGAILConfig
    from scripts_gail.ps_gail.training.evaluation import _evaluation_scenarios

    seed = int(config["runtime_invariants"]["evaluation_scenario_seed"])
    episodes = int(config["evaluation"]["validation_episodes"])
    domains: dict[str, Any] = {}
    for domain, scene in EXPECTED_DOMAIN_SCENES.items():
        cfg = PSGAILConfig(
            episode_root=str(episode_root.resolve()),
            scene=scene,
            prebuilt_split="val",
            evaluation_scenario_seed=seed,
        )
        scenarios = [
            {"episode_name": name, "vehicle_id": int(vehicle_id)}
            for name, vehicle_id in _evaluation_scenarios(
                cfg,
                split="val",
                episodes=episodes,
            )
        ]
        domains[domain] = {
            "scene": scene,
            "split": "val",
            "scenario_seed": seed,
            "requested_episodes": episodes,
            "scenarios": scenarios,
            "scenario_bank_sha256": canonical_sha256(scenarios),
        }
    return {
        "pairing_rule": "identical domain-specific val bank across every arm",
        "domains": domains,
        "test_data_status": "not_opened",
    }


def _csv(values: Iterable[object]) -> str:
    return ",".join(str(value) for value in values)


def build_training_command(
    config: dict[str, Any],
    cell: TrialCell,
    *,
    python: Path,
    expert_root: Path,
    episode_root: Path,
    cell_root: Path,
    device: str,
) -> list[str]:
    """Build a shell-free command; no policy-action transform is available."""

    roots = expert_split_roots(expert_root)[cell.domain]
    data = config["data"]
    architecture = config["architecture"]
    optimization = config["optimization"]
    gates = config["gates"]
    evaluation = config["evaluation"]
    invariants = config["runtime_invariants"]
    qualification = evaluation["expert_replay_qualification"]
    command = [
        str(python.resolve()),
        "-m",
        "scripts_gail.train_recurrent_bc_policy",
        "--policy-model",
        cell.policy_model,
        "--expert-data",
        str(roots["train"]),
        "--expert-validation-data",
        str(roots["val"]),
        "--out-dir",
        str(cell_root),
        "--domain",
        cell.domain,
        "--scene",
        cell.scene,
        "--episode-root",
        str(episode_root.resolve()),
        "--prebuilt-split",
        "train",
        "--seed",
        str(cell.seed),
        "--data-seed",
        str(data["data_seed"]),
        "--split-seed",
        str(data["split_seed"]),
        "--max-expert-samples",
        str(data["max_expert_samples"]),
        "--max-validation-samples",
        str(data["max_validation_samples"]),
        "--max-test-samples",
        "0",
        "--epochs",
        str(optimization["epochs"]),
        "--checkpoint-purpose",
        "policy",
        "--checkpoint-selection-rule",
        str(optimization["checkpoint_selection_rule"]),
        "--test-evaluation-mode",
        "deferred",
        "--early-stopping-patience",
        str(optimization["early_stopping_patience"]),
        "--early-stopping-min-epochs",
        str(optimization["early_stopping_min_epochs"]),
        "--early-stopping-min-delta-relative",
        str(optimization["early_stopping_min_delta_relative"]),
        "--learning-rate",
        str(optimization["learning_rate"]),
        "--weight-decay",
        str(optimization["weight_decay"]),
        "--max-grad-norm",
        str(optimization["max_grad_norm"]),
        "--train-fraction",
        str(data["train_fraction"]),
        "--validation-fraction",
        str(data["validation_fraction"]),
        "--action-loss-weights",
        _csv(optimization["action_loss_weights"]),
        "--action-loss-weighting",
        cell.action_loss_weighting,
        "--correlation-loss-weight",
        str(optimization["correlation_loss_weight"]),
        "--variance-loss-weight",
        str(optimization["variance_loss_weight"]),
        "--training-min-prediction-std-ratios",
        _csv(optimization["training_min_prediction_std_ratios"]),
        "--mirror-augmentation-probability",
        str(optimization["mirror_augmentation_probability"]),
        "--min-validation-skill",
        str(gates["min_validation_skill"]),
        "--max-validation-mae",
        str(gates["max_validation_mae"]),
        "--learning-action-index",
        str(gates["learning_action_index"]),
        "--min-learning-action-std-ratio",
        str(gates["min_learning_action_std_ratio"]),
        "--min-learning-action-correlation",
        str(gates["min_learning_action_correlation"]),
        "--learning-action-indices",
        _csv(gates["learning_action_indices"]),
        "--min-learning-action-std-ratios",
        _csv(gates["min_learning_action_std_ratios"]),
        "--min-learning-action-correlations",
        _csv(gates["min_learning_action_correlations"]),
        "--require-explicit-data-contracts",
        "--hidden-size",
        str(architecture["hidden_size"]),
        "--transformer-layers",
        str(cell.depth),
        "--transformer-heads",
        str(architecture["transformer_heads"]),
        "--transformer-dropout",
        str(architecture["transformer_dropout"]),
        "--transformer-norm-first",
        "--transformer-observation-normalization",
        "--policy-observation-standardization-clip",
        str(invariants["observation_standardization_clip"]),
        "--transformer-observation-tokenization",
        str(architecture["observation_tokenization"]),
        "--policy-head-init-std",
        str(architecture["policy_head_init_std"]),
        "--memory-tokens",
        str(architecture["memory_tokens"]),
        "--memory-context-length",
        str(architecture["memory_context_length"]),
        "--sequence-length",
        str(optimization["sequence_length"]),
        "--recurrent-warmup-mode",
        "full_prefix",
        "--sequences-per-batch",
        str(optimization["sequences_per_batch"]),
        "--micro-batch-sequences",
        str(optimization["micro_batch_sequences"]),
        "--evaluation-episodes",
        str(evaluation["episodes"]),
        "--validation-evaluation-episodes",
        str(evaluation["validation_episodes"]),
        "--evaluation-scenario-seed",
        str(invariants["evaluation_scenario_seed"]),
        "--evaluation-split",
        "val",
        "--evaluation-enable-collision",
        "--matched-evaluation",
        "--evaluation-vehicle-mode",
        str(evaluation["vehicle_mode"]),
        "--min-rollout-steps",
        str(evaluation["min_rollout_steps"]),
        "--max-crash-fraction",
        str(evaluation["max_crash_fraction"]),
        "--max-offroad-fraction",
        str(evaluation["max_offroad_fraction"]),
        "--expert-replay-qualification-required",
        "--maximum-expert-replay-crash-rate-gap",
        str(qualification["maximum_vehicle_crash_rate_gap"]),
        "--maximum-expert-replay-offroad-rate-gap",
        str(qualification["maximum_vehicle_offroad_rate_gap"]),
        "--no-render-video",
        "--capability-failure-mode",
        "report",
        "--device",
        str(device),
    ]
    if "--expert-test-data" in command:
        raise RuntimeError("Validation-only command unexpectedly contains test data.")
    return command


def validate_cell_summary(
    summary: dict[str, Any],
    *,
    cell: TrialCell,
    cell_root: Path,
    expert_root: Path,
) -> dict[str, Any]:
    qualification = summary.get("expert_replay_qualification") or {}
    source_integrity = (
        (summary.get("training_data_contract") or {})
        .get("source_root_integrity", {})
        .get("sources", {})
    )
    expected_roots = expert_split_roots(expert_root)[cell.domain]
    checks = {
        "training_artifact_complete": (
            summary.get("training_artifact_complete") is True
        ),
        "domain_exact": summary.get("domain") == cell.domain,
        "scene_exact": summary.get("scene") == cell.scene,
        "seed_exact": summary.get("seed") == cell.seed,
        "policy_model_exact": summary.get("policy_model") == cell.policy_model,
        "depth_exact": summary.get("transformer_layers") == cell.depth,
        "weighting_exact": (
            summary.get("action_loss_weighting")
            == cell.action_loss_weighting
        ),
        "configured_weights_exact": (
            summary.get("configured_action_loss_weights") == [1.0, 1.0]
        ),
        "full_prefix_exact": (
            summary.get("recurrent_warmup_mode") == "full_prefix"
        ),
        "checkpoint_selection_rule_exact": (
            summary.get("checkpoint_selection_rule")
            == "validation_loss"
        ),
        "checkpoint_selection_metric_exact": (
            summary.get("checkpoint_selection_metric")
            == "unweighted_validation_mse"
        ),
        "scenario_seed_exact": (
            summary.get("evaluation_scenario_seed") == 20260716
        ),
        "test_mode_deferred": (
            summary.get("test_evaluation_mode") == "deferred"
        ),
        "offline_test_not_evaluated": (
            summary.get("offline_test_evaluated") is False
        ),
        "offline_test_metrics_null": (
            summary.get("test_mse") is None
            and summary.get("test_mae") is None
        ),
        "closed_loop_test_pending": (
            (summary.get("held_out_evaluation") or {}).get("status")
            == "pending_deferred"
        ),
        "collision_physics_enabled": (
            summary.get("validation_collision_physics_enabled") is True
        ),
        "native_action_echo_exact": math.isclose(
            float(
                (summary.get("validation_rollouts") or {}).get(
                    "validation/normalized_action_echo_exact_rate",
                    math.nan,
                )
            ),
            1.0,
            abs_tol=0.0,
        ),
        "no_unexpected_native_action_override": (
            float(
                (summary.get("validation_rollouts") or {}).get(
                    "validation/unexpected_action_override_count",
                    math.nan,
                )
            )
            == 0.0
        ),
        "expert_reference_required": (
            qualification.get("required_for_policy_realism") is True
        ),
        "expert_crash_gap_exact": math.isclose(
            float(qualification.get("maximum_vehicle_crash_rate_gap", math.nan)),
            0.05,
            abs_tol=1.0e-12,
        ),
        "expert_offroad_gap_exact": math.isclose(
            float(
                qualification.get(
                    "maximum_vehicle_offroad_rate_gap",
                    math.nan,
                )
            ),
            0.05,
            abs_tol=1.0e-12,
        ),
        "policy_realism_status_recorded": (
            isinstance(summary.get("policy_realism_qualified"), bool)
        ),
        "train_source_exact": (
            (source_integrity.get("train") or {}).get("root")
            == str(expected_roots["train"].resolve())
        ),
        "validation_source_exact": (
            (source_integrity.get("validation") or {}).get("root")
            == str(expected_roots["val"].resolve())
        ),
        "test_source_not_opened": (
            (summary.get("expert_data") or {}).get("test", {}).get("status")
            == "pending_deferred_not_opened"
        ),
    }

    checkpoint = cell_root / "best.pt"
    sidecar = cell_root / "best.pt.sha256"
    actual_checkpoint_sha256 = (
        sha256_file(checkpoint) if checkpoint.is_file() else None
    )
    sidecar_fields = (
        sidecar.read_text(encoding="utf-8").split()
        if sidecar.is_file()
        else []
    )
    checks["checkpoint_digest_exact"] = bool(
        actual_checkpoint_sha256
        and summary.get("checkpoint_sha256") == actual_checkpoint_sha256
        and sidecar_fields == [actual_checkpoint_sha256, "best.pt"]
    )
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha256,
    }


def compact_cell_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    validation_rollouts = summary.get("validation_rollouts") or {}
    return {
        "validation_mse": summary.get("validation_mse"),
        "validation_mae": summary.get("validation_mae"),
        "validation_skill": summary.get("validation_skill"),
        "validation_prediction_target_correlation": summary.get(
            "validation_prediction_target_correlation"
        ),
        "validation_prediction_std_ratio": summary.get(
            "validation_prediction_std_ratio"
        ),
        "effective_action_loss_weights": summary.get("action_loss_weights"),
        "training_action_variance": summary.get("training_action_variance"),
        "paper_validation_score": summary.get("paper_validation_score"),
        "paper_validation_cost": summary.get("paper_validation_cost"),
        "validation_vehicle_crash_rate": validation_rollouts.get(
            "validation/vehicle_crash_rate"
        ),
        "validation_vehicle_offroad_rate": validation_rollouts.get(
            "validation/vehicle_offroad_rate"
        ),
        "validation_rollout_horizon_coverage_20s": validation_rollouts.get(
            "validation/rollout_horizon_coverage_20s"
        ),
        "validation_background_idm_handover_count": validation_rollouts.get(
            "validation/background_idm_handover_count"
        ),
        "validation_normalized_action_echo_exact_rate": (
            validation_rollouts.get(
                "validation/normalized_action_echo_exact_rate"
            )
        ),
        "validation_post_step_action_state_exact_rate": (
            validation_rollouts.get(
                "validation/post_step_action_state_exact_rate"
            )
        ),
        "validation_crash_physics_action_override_count": (
            validation_rollouts.get(
                "validation/crash_physics_action_override_count"
            )
        ),
        "validation_speed_bound_action_override_count": (
            validation_rollouts.get(
                "validation/speed_bound_action_override_count"
            )
        ),
        "validation_unexpected_action_override_count": (
            validation_rollouts.get(
                "validation/unexpected_action_override_count"
            )
        ),
        "validation_selection_eligible": summary.get(
            "validation_selection_eligible"
        ),
        "policy_realism_qualified": summary.get("policy_realism_qualified"),
    }


def current_provenance(
    *,
    component_root: Path,
    config_path: Path,
    expert_root: Path,
    episode_root: Path,
    audit_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_files": file_set_receipt(
            source_paths(component_root, config_path)
        ),
        "data": build_data_lock(
            expert_root=expert_root,
            episode_root=episode_root,
            audit_path=audit_path,
        ),
    }


def verify_runtime_imports(component_root: Path, python: Path) -> dict[str, str]:
    code = """
import json
from pathlib import Path
import highway_env
import scripts_gail.train_recurrent_bc_policy as trainer
print(json.dumps({
    "highway_env": str(Path(highway_env.__file__).resolve()),
    "trainer": str(Path(trainer.__file__).resolve()),
}, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(component_root)
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    result = subprocess.run(
        [str(python.resolve()), "-c", code],
        cwd=component_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    expected = {
        "highway_env": str(
            (component_root / "highway_env/__init__.py").resolve()
        ),
        "trainer": str(
            (
                component_root
                / "scripts_gail/train_recurrent_bc_policy.py"
            ).resolve()
        ),
    }
    if payload != expected:
        raise RuntimeError(
            f"Repository-first import check failed: {payload} != {expected}"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path)
    parser.add_argument("--audit-receipt", type=Path)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the locked config and print the eight cells without data access.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    component_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_study_config(config)
    cells = enumerate_cells(config)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "config_valid",
                    "study_id": config["study_id"],
                    "cell_count": len(cells),
                    "cells": [asdict(cell) for cell in cells],
                    "test_data_status": "not_constructed_not_opened",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    missing = [
        name
        for name in (
            "expert_root",
            "audit_receipt",
            "episode_root",
            "run_root",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Execution requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )

    expert_root = args.expert_root.resolve()
    audit_path = args.audit_receipt.resolve()
    episode_root = args.episode_root.resolve()
    run_root = args.run_root.resolve()
    python = args.python.resolve()
    if not run_root.is_absolute():
        raise ValueError("run-root must be absolute")
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {run_root}")
    if audit_path.parent != expert_root:
        raise ValueError(
            "Audit receipt must be a direct child of the exact v3 expert root."
        )

    expert_split_roots(expert_root)
    audit = read_json(audit_path)
    validate_audit_receipt(audit, expert_root=expert_root)
    verify_audit_file_binding(audit, expert_root=expert_root)
    import_receipt = verify_runtime_imports(component_root, python)
    baseline = current_provenance(
        component_root=component_root,
        config_path=config_path,
        expert_root=expert_root,
        episode_root=episode_root,
        audit_path=audit_path,
    )
    pairing = scenario_pairing_receipt(
        config,
        episode_root=episode_root,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "logs").mkdir()
    (run_root / "cells").mkdir()
    (run_root / "receipts").mkdir()
    manifest = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "engineering_discovery_validation_only",
        "scientific_claim_allowed": False,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_payload": config,
        "cells": [asdict(cell) for cell in cells],
        "expected_cell_count": len(cells),
        "all_cells_mandatory": True,
        "runtime_imports": import_receipt,
        "scenario_pairing": pairing,
        "provenance_lock": baseline,
        "provenance_lock_sha256": canonical_sha256(baseline),
        "test_access": {
            "expert_test_argument_supported_by_this_runner": False,
            "constructed_splits": ["train", "val"],
            "test_data_status": "not_constructed_not_listed_not_opened",
        },
        "policy_action_contract": {
            "execution": "network_output_passed_unchanged_to_env_step",
            "clamping": False,
            "substitution": False,
            "evidence_role": (
                "source-locked execution invariant; final qualification must "
                "retain its independent runtime receipt"
            ),
        },
    }
    write_json(run_root / "RUN_MANIFEST.json", manifest)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": (
                str(component_root)
                + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
            ),
            "NGSIM_ACCELERATION_LIMIT_MPS2": "5.0",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "2"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "2"),
            "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", "2"),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": env.get(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True",
            ),
        }
    )

    records: list[dict[str, Any]] = []
    fatal_drift: str | None = None
    for cell in cells:
        current_before = current_provenance(
            component_root=component_root,
            config_path=config_path,
            expert_root=expert_root,
            episode_root=episode_root,
            audit_path=audit_path,
        )
        if current_before != baseline:
            fatal_drift = f"Source/data drift detected before {cell.cell_id}"
            break

        cell_root = run_root / "cells" / cell.cell_id
        log_path = run_root / "logs" / f"{cell.cell_id}.log"
        command = build_training_command(
            config,
            cell,
            python=python,
            expert_root=expert_root,
            episode_root=episode_root,
            cell_root=cell_root,
            device=str(args.device),
        )
        print(f"[v3-factorial] starting {cell.cell_id}", flush=True)
        started = datetime.now(timezone.utc)
        with log_path.open("wb") as log_handle:
            completed = subprocess.run(
                command,
                cwd=component_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finished = datetime.now(timezone.utc)
        record: dict[str, Any] = {
            **asdict(cell),
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "command": command,
            "log": str(log_path),
            "return_code": int(completed.returncode),
        }
        if completed.returncode == 0:
            summary_path = cell_root / "summary.json"
            try:
                summary = read_json(summary_path)
                validation = validate_cell_summary(
                    summary,
                    cell=cell,
                    cell_root=cell_root,
                    expert_root=expert_root,
                )
                record.update(
                    {
                        "status": (
                            "completed"
                            if validation["passed"]
                            else "completed_with_receipt_failure"
                        ),
                        "summary": str(summary_path),
                        "summary_sha256": sha256_file(summary_path),
                        "receipt_validation": validation,
                        "metrics": compact_cell_metrics(summary),
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "completed_with_receipt_exception",
                        "receipt_exception": f"{type(exc).__name__}: {exc}",
                    }
                )
        else:
            record["status"] = "runtime_failed_preserved"

        current_after = current_provenance(
            component_root=component_root,
            config_path=config_path,
            expert_root=expert_root,
            episode_root=episode_root,
            audit_path=audit_path,
        )
        record["provenance_equal_after_cell"] = current_after == baseline
        records.append(record)
        write_json(
            run_root / "receipts" / f"{cell.cell_id}.json",
            record,
        )
        print(
            f"[v3-factorial] {cell.cell_id}: {record['status']}",
            flush=True,
        )
        if current_after != baseline:
            fatal_drift = f"Source/data drift detected after {cell.cell_id}"
            break

    completed_ids = {str(record["cell_id"]) for record in records}
    missing_cells = [
        cell.cell_id for cell in cells if cell.cell_id not in completed_ids
    ]
    successful = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("provenance_equal_after_cell") is True
    ]
    completion = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete"
            if len(successful) == len(cells) and fatal_drift is None
            else "complete_with_failures"
        ),
        "evidence_level": "engineering_discovery_validation_only",
        "scientific_claim_allowed": False,
        "g0_status": (
            "not_evaluated_pending_replication_expert_reference_and_locked_test"
        ),
        "winner_selected": False,
        "expected_cell_count": len(cells),
        "reported_cell_count": len(records),
        "successful_cell_count": len(successful),
        "missing_cells": missing_cells,
        "fatal_provenance_drift": fatal_drift,
        "test_data_status": "not_constructed_not_listed_not_opened",
        "policy_action_execution": (
            "network_output_passed_unchanged_to_env_step"
        ),
        "records": records,
    }
    write_json(run_root / "COMPLETION.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0 if completion["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
