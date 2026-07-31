#!/usr/bin/env python3
"""Train an interpretation-ready recurrent-transformer BC policy."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from scripts_gail.pretrain_continuous_bc_policy import (
    build_policy_for_env,
    default_scenario_from_expert_folder,
    make_selected_replay_env,
    render_selected_replay,
)
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.contracts import (
    assert_compatible_action_contracts,
    assert_compatible_observation_contracts,
    validate_training_data_contracts,
)
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.recurrent_bc import (
    PreparedRecurrentBCData,
    prepare_recurrent_bc_data_from_explicit_splits,
    train_recurrent_behavior_clone,
)
from scripts_gail.ps_gail.trainer import (
    evaluate_policy_matched_trajectories,
    resolve_device,
)
from scripts_gail.ps_gail.validation import (
    PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
    action_learning_gate,
    closed_loop_policy_quality,
    paper_driver_model_validation_overrides,
    scored_validation_metrics,
)
from scripts_gail.train_simple_ps_gail import evaluate_policy_survival


CHECKPOINT_SCHEMA_VERSION = 1
EXPECTED_SCENE_BY_DOMAIN = {
    "us": "us-101",
    "japanese": "japanese",
}


def _comma_separated_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _comma_separated_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _valid_two_action_diagnostic(values: Any) -> bool:
    """Require one finite diagnostic value for each policy action."""
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return False
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == (2,) and np.isfinite(array).all())


def _split_manifest_is_valid(
    path: Path,
    *,
    test_evaluated: bool,
) -> bool:
    """Validate split names, trajectory identifiers, and disjointness."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or set(payload) != {
        "train",
        "validation",
        "test",
    }:
        return False
    split_sets: dict[str, set[str]] = {}
    canonical_sets: dict[str, set[str]] = {}
    known_prefixes = tuple(
        f"{name}:" for name in ("train", "validation", "test")
    )
    for split in ("train", "validation", "test"):
        identifiers = payload[split]
        if (
            not isinstance(identifiers, list)
            or any(
                not isinstance(value, str) or not value
                for value in identifiers
            )
            or len(set(identifiers)) != len(identifiers)
        ):
            return False
        split_sets[split] = set(identifiers)
        canonical: set[str] = set()
        expected_prefix = f"{split}:"
        for identifier in identifiers:
            if identifier.startswith(expected_prefix):
                identifier = identifier[len(expected_prefix) :]
            elif identifier.startswith(known_prefixes):
                return False
            if (
                not identifier
                or identifier.startswith("fallback_basename/")
                or identifier in canonical
            ):
                return False
            canonical.add(identifier)
        canonical_sets[split] = canonical
    if not split_sets["train"] or not split_sets["validation"]:
        return False
    if test_evaluated and not split_sets["test"]:
        return False
    if any(
        canonical_sets[left] & canonical_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        return False
    return True


def _expert_root_manifest_identity(
    root: Path,
    *,
    expected_scene: str,
    expected_split: str,
) -> dict[str, object]:
    """Validate one complete collection root before optimization starts."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Expert source root not found: {resolved}")
    manifest_path = resolved / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Expert source has no readable manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Expert source manifest is not an object: {manifest_path}")
    scene = str(manifest.get("scene") or "")
    split = str(manifest.get("prebuilt_split") or "")
    if scene != str(expected_scene):
        raise ValueError(
            f"Expert source scene mismatch: {scene!r} != "
            f"{expected_scene!r} at {manifest_path}."
        )
    if split != str(expected_split):
        raise ValueError(
            f"Expert source split mismatch: {split!r} != "
            f"{expected_split!r} at {manifest_path}."
        )
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Expert source manifest has no episodes: {manifest_path}")
    canonical_ids: list[str] = []
    dataset_files: list[str] = []
    for row in episodes:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid episode record in {manifest_path}.")
        episode_name = str(row.get("episode_name") or "").strip()
        dataset_file = str(row.get("dataset_file") or "").strip()
        if not episode_name or not dataset_file:
            raise ValueError(
                f"Episode record lacks episode_name/dataset_file: {manifest_path}"
            )
        dataset_path = (resolved / dataset_file).resolve()
        if dataset_path.parent != resolved or not dataset_path.is_file():
            raise ValueError(
                "Expert manifest dataset_file must name an existing direct "
                f"child of its root: {dataset_file!r}."
            )
        canonical_ids.append(f"{scene}/{episode_name}")
        dataset_files.append(dataset_file)
    if (
        len(canonical_ids) != len(set(canonical_ids))
        or len(dataset_files) != len(set(dataset_files))
    ):
        raise ValueError(
            f"Expert source repeats episode identities or files: {manifest_path}"
        )
    return {
        "root": str(resolved),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "scene": scene,
        "split": split,
        "episode_count": len(canonical_ids),
        "canonical_episode_ids": sorted(canonical_ids),
    }


def validate_explicit_expert_source_roots(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Fail before training on domain/scene drift or source overlap."""

    domain = str(args.domain).strip().lower()
    if domain not in EXPECTED_SCENE_BY_DOMAIN:
        raise ValueError(f"Unsupported BC domain: {domain!r}.")
    expected_scene = EXPECTED_SCENE_BY_DOMAIN[domain]
    if str(args.scene) != expected_scene:
        raise ValueError(
            f"Domain {domain!r} requires scene {expected_scene!r}, got "
            f"{args.scene!r}."
        )
    test_mode = str(
        getattr(args, "test_evaluation_mode", "deferred")
    ).strip().lower()
    validation_raw = str(
        getattr(args, "expert_validation_data", "") or ""
    )
    test_raw = str(getattr(args, "expert_test_data", "") or "")
    sources: dict[str, Path] = {
        "train": Path(args.expert_data),
    }
    if validation_raw:
        sources["validation"] = Path(validation_raw)
    if test_mode == "evaluate":
        if not validation_raw or not test_raw:
            raise ValueError(
                "Final evaluation requires explicit train, validation and "
                "test source roots."
            )
        sources["test"] = Path(test_raw)
    elif test_raw:
        raise ValueError(
            "Deferred test mode refuses an expert test source root."
        )

    source_names = sorted(sources)
    for left_index, left in enumerate(source_names):
        left_root = sources[left].resolve()
        for right in source_names[left_index + 1 :]:
            right_root = sources[right].resolve()
            same_source = left_root == right_root
            if not same_source and left_root.exists() and right_root.exists():
                try:
                    same_source = os.path.samefile(left_root, right_root)
                except OSError:
                    same_source = False
            if same_source:
                raise ValueError(
                    f"Expert {left}/{right} roots resolve to the same source."
                )

    records = {
        split: _expert_root_manifest_identity(
            root,
            expected_scene=expected_scene,
            expected_split=("val" if split == "validation" else split),
        )
        for split, root in sources.items()
    }
    split_names = sorted(records)
    for left_index, left in enumerate(split_names):
        left_root = Path(str(records[left]["root"]))
        left_ids = set(records[left]["canonical_episode_ids"])
        for right in split_names[left_index + 1 :]:
            right_root = Path(str(records[right]["root"]))
            same_source = left_root == right_root
            if not same_source:
                try:
                    same_source = os.path.samefile(left_root, right_root)
                except OSError:
                    same_source = False
            if same_source:
                raise ValueError(
                    f"Expert {left}/{right} roots resolve to the same source."
                )
            overlap = sorted(
                left_ids.intersection(records[right]["canonical_episode_ids"])
            )
            if overlap:
                raise ValueError(
                    f"Expert {left}/{right} roots overlap in canonical "
                    f"episode identity; first={overlap[0]!r}."
                )
    return {
        "status": "passed",
        "domain": domain,
        "scene": expected_scene,
        "sources": records,
        "canonical_episode_disjoint": True,
        "test_source_opened": bool("test" in records),
    }


def _checkpoint_digest_is_valid(
    checkpoint_path: Path,
    sidecar_path: Path,
    expected_digest: Any,
) -> bool:
    """Match the summary and sidecar digests to the checkpoint bytes."""
    if not checkpoint_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        actual_digest = sha256_file(checkpoint_path)
        sidecar_fields = sidecar_path.read_text(encoding="utf-8").split()
    except (OSError, UnicodeError):
        return False
    return bool(
        isinstance(expected_digest, str)
        and expected_digest == actual_digest
        and len(sidecar_fields) == 2
        and sidecar_fields == [actual_digest, checkpoint_path.name]
    )


def training_artifact_is_complete(summary: dict[str, Any], out_dir: Path) -> bool:
    """Return whether a BC run produced a finite, integrity-checkable artifact."""
    finite_scalars = (
        summary.get("initial_validation_mse"),
        summary.get("validation_mse"),
        summary.get("validation_mae"),
        summary.get("validation_selection_mse"),
    )
    diagnostic_vectors = (
        summary.get("validation_action_mse"),
        summary.get("validation_action_mae"),
        summary.get("validation_prediction_std"),
        summary.get("validation_target_std"),
        summary.get("validation_prediction_std_ratio"),
        summary.get("validation_prediction_target_correlation"),
        summary.get("validation_prediction_saturation_fraction"),
    )
    checkpoint_path = out_dir / "best.pt"
    split_manifest_path = out_dir / "split_manifest.json"
    data_contract = summary.get("training_data_contract")
    source_identity_passed = bool(
        not isinstance(data_contract, dict)
        or not bool(data_contract.get("explicit_contracts_required"))
        or data_contract.get(
            "source_stable_trajectory_identity_passed"
        )
        is True
    )
    return bool(
        summary.get("checkpoint_saved")
        and _checkpoint_digest_is_valid(
            checkpoint_path,
            out_dir / "best.pt.sha256",
            summary.get("checkpoint_sha256"),
        )
        and _split_manifest_is_valid(
            split_manifest_path,
            test_evaluated=bool(summary.get("offline_test_evaluated")),
        )
        and source_identity_passed
        and all(value is not None and np.isfinite(float(value)) for value in finite_scalars)
        and all(_valid_two_action_diagnostic(values) for values in diagnostic_vectors)
    )


def validate_recurrent_bc_source_contracts(
    metadata: dict[str, Any],
    *,
    lidar_cells: int,
    maximum_range: float,
    require_explicit: bool,
    expected_scene: str | None = None,
) -> dict[str, object]:
    """Validate every opened source split and their mutual compatibility."""
    if str(metadata.get("split_method", "")).startswith(
        "explicit_source_directories"
    ):
        sources = metadata.get("sources")
        if not isinstance(sources, dict) or not {
            "train",
            "validation",
        }.issubset(sources):
            raise ValueError(
                "Explicit recurrent BC metadata must include train and "
                "validation source metadata."
            )
    else:
        sources = {"train": metadata}

    contracts: dict[str, dict[str, object]] = {}
    identity_quality: dict[str, object] = {}
    for split, source_metadata in sources.items():
        episode_scenes = {
            str(item.get("scene") or "")
            for item in source_metadata.get("episodes", [])
            if isinstance(item, dict)
        }
        if expected_scene is not None and episode_scenes != {
            str(expected_scene)
        }:
            raise ValueError(
                f"Recurrent BC {split!r} source scene set "
                f"{sorted(episode_scenes)} does not equal "
                f"{expected_scene!r}."
            )
        identity_quality[split] = source_metadata.get(
            "trajectory_id_identity_quality"
        )
        if (
            bool(require_explicit)
            and identity_quality[split] != "scene_and_episode_name"
        ):
            raise ValueError(
                f"Recurrent BC {split!r} source lacks source-stable "
                "scene-and-episode trajectory identity."
            )
        try:
            contracts[split] = validate_training_data_contracts(
                source_metadata,
                lidar_cells=int(lidar_cells),
                maximum_range=float(maximum_range),
                require_explicit=bool(require_explicit),
            )
        except ValueError as exc:
            raise ValueError(
                f"Recurrent BC {split!r} source contract is invalid."
            ) from exc
    reference = contracts["train"]
    for split, candidate in contracts.items():
        if split == "train":
            continue
        try:
            assert_compatible_action_contracts(
                reference["continuous_action"],
                candidate["continuous_action"],
            )
            assert_compatible_observation_contracts(
                reference["policy_observation"],
                candidate["policy_observation"],
            )
        except ValueError as exc:
            raise ValueError(
                f"Recurrent BC {split!r} contract differs from train."
            ) from exc
    return {
        **reference,
        "validated_source_splits": sorted(contracts),
        "cross_source_contracts_matched": True,
        "trajectory_id_identity_quality": identity_quality,
        "source_stable_trajectory_identity_passed": bool(
            all(
                value == "scene_and_episode_name"
                for value in identity_quality.values()
            )
        ),
    }


def recurrent_bc_row_sampling_receipt(
    metadata: dict[str, Any],
    *,
    requested_train_rows: int,
    requested_validation_rows: int,
    requested_test_rows: int,
) -> dict[str, object]:
    """Report trajectory-preserving row targets separately from loaded rows."""
    if str(metadata.get("split_method", "")).startswith(
        "explicit_source_directories"
    ):
        sources = dict(metadata.get("sources") or {})
        actual = {
            split: (
                int(source["num_samples"])
                if isinstance(source, dict)
                and source.get("num_samples") is not None
                else None
            )
            for split, source in sources.items()
        }
    else:
        actual = {
            "train_source_before_internal_split": (
                int(metadata["num_samples"])
                if metadata.get("num_samples") is not None
                else None
            )
        }
    return {
        "sampling_semantics": (
            "trajectory_preserving_row_target_may_overshoot_to_keep_complete_trajectories"
        ),
        "requested_row_targets": {
            "train": int(requested_train_rows),
            "validation": int(requested_validation_rows),
            "test": int(requested_test_rows),
        },
        "actual_loaded_rows": actual,
    }


def prepare_fresh_output_directory(path: Path) -> Path:
    """Create or validate an empty output directory; resume is unsupported."""
    resolved = path.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise FileExistsError(
                f"BC output path exists and is not a directory: {resolved}"
            )
        if next(resolved.iterdir(), None) is not None:
            raise FileExistsError(
                "Refusing to reuse non-empty BC output directory because no "
                f"resume protocol is implemented: {resolved}"
            )
    else:
        resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def collision_physics_evaluation_status(
    *,
    configured: bool,
    evaluated: bool,
) -> tuple[bool | None, str]:
    """Separate a configured collision mode from an executed test."""
    if not evaluated:
        return None, "configured_not_evaluated"
    enabled = bool(configured)
    return enabled, "enabled" if enabled else "disabled"


def validation_selection_is_eligible(
    *,
    training_artifact_complete: bool,
    metric_capability_passed: bool,
    validation_rollout_capability_passed: bool,
    validation_contract_passed: bool,
    collision_physics_enabled: bool,
) -> bool:
    """Require validation quality and collision physics without using test data."""
    return bool(
        training_artifact_complete
        and metric_capability_passed
        and validation_rollout_capability_passed
        and validation_contract_passed
        and collision_physics_enabled
    )


def final_test_qualification_is_eligible(
    *,
    test_evaluation_enabled: bool,
    held_out_metric_capability_passed: bool,
    rollout_capability_passed: bool,
    test_contract_passed: bool,
    collision_physics_enabled: bool,
) -> bool:
    """Require a locked test, its contracts, and enabled collision physics."""
    return bool(
        test_evaluation_enabled
        and held_out_metric_capability_passed
        and rollout_capability_passed
        and test_contract_passed
        and collision_physics_enabled
    )


def survival_quality_passed(
    stats: dict[str, Any],
    *,
    min_rollout_steps: int,
    max_collision_fraction: float,
    max_offroad_fraction: float,
) -> bool:
    """Gate physical collisions and off-road events as distinct outcomes."""
    required = (
        "bc_eval/mean_episode_length",
        "bc_eval/collision_episode_fraction",
        "bc_eval/offroad_episode_fraction",
    )
    if not stats or any(key not in stats for key in required):
        return False
    values = [float(stats[key]) for key in required]
    return bool(
        all(np.isfinite(value) for value in values)
        and values[0] >= float(min_rollout_steps)
        and values[1] <= float(max_collision_fraction)
        and values[2] <= float(max_offroad_fraction)
    )


def evaluation_completion_flags(
    *,
    matched_evaluation: bool,
    matched_validation_evaluated: bool,
    matched_test_evaluated: bool,
    validation_survival_evaluated: bool,
    test_survival_evaluated: bool,
) -> dict[str, bool]:
    """Keep matched and unmatched evaluator completion labels disjoint."""
    return {
        "matched_evaluation_complete": bool(
            matched_evaluation
            and matched_validation_evaluated
            and matched_test_evaluated
        ),
        "validation_survival_evaluation_complete": bool(
            not matched_evaluation and validation_survival_evaluated
        ),
        "test_survival_evaluation_complete": bool(
            not matched_evaluation and test_survival_evaluated
        ),
    }


def paper_validation_overrides_for_vehicle_mode(
    vehicle_mode: str,
) -> dict[str, Any]:
    """Apply the requested evaluator scope without duplicate keyword overrides."""
    mode = str(vehicle_mode)
    if mode not in {"single", "training_count", "all"}:
        raise ValueError(f"Unsupported evaluation vehicle mode: {mode!r}.")
    overrides = dict(paper_driver_model_validation_overrides())
    overrides["validation_vehicle_mode"] = mode
    overrides["test_vehicle_mode"] = mode
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-data", required=True)
    parser.add_argument(
        "--expert-validation-data",
        default="",
        help=(
            "Optional independently collected validation root. Deferred mode "
            "loads this with training data only; final evaluate mode also "
            "requires --expert-test-data."
        ),
    )
    parser.add_argument(
        "--expert-test-data",
        default="",
        help=(
            "Optional independently collected final-test root. It is accepted "
            "and loaded only in final evaluate mode."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--domain", required=True, choices=["us", "japanese"])
    parser.add_argument("--scene", required=True, choices=["us-101", "japanese"])
    parser.add_argument("--episode-root", default="data/highway_env/processed_20s")
    parser.add_argument("--prebuilt-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=20260716)
    parser.add_argument("--split-seed", type=int, default=20260716)
    parser.add_argument("--max-expert-samples", type=int, default=300_000)
    parser.add_argument("--max-validation-samples", type=int, default=100_000)
    parser.add_argument("--max-test-samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--checkpoint-purpose",
        choices=["policy", "warm_start"],
        default="policy",
        help="Use warm_start for a deliberately under-trained GAIL/AIRL stabilizer.",
    )
    parser.add_argument("--max-warmup-epochs", type=int, default=5)
    parser.add_argument("--min-warmup-relative-improvement", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=0,
        help=(
            "Do not apply patience-based early stopping before this epoch. "
            "Set patience to zero to disable early stopping."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta-relative",
        type=float,
        default=0.001,
        help=(
            "Reset early-stopping patience only after at least this relative "
            "validation-loss improvement. Best-checkpoint selection still "
            "retains every strict validation-loss improvement."
        ),
    )
    parser.add_argument("--min-validation-skill", type=float, default=0.05)
    parser.add_argument("--max-validation-mae", type=float, default=0.35)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--action-loss-weights",
        type=_comma_separated_floats,
        default=[1.0, 1.0],
        help="Comma-separated BC optimization weights in [acceleration, steering] order.",
    )
    parser.add_argument(
        "--action-loss-weighting",
        choices=["fixed", "inverse_variance"],
        default="fixed",
        help=(
            "Use fixed weights directly or divide them by each training-split "
            "action variance before normalization."
        ),
    )
    parser.add_argument(
        "--correlation-loss-weight",
        type=float,
        default=0.0,
        help="Weight on the per-micro-batch action correlation anti-collapse loss.",
    )
    parser.add_argument(
        "--variance-loss-weight",
        type=float,
        default=0.0,
        help="Weight on the one-sided prediction standard-deviation anti-collapse loss.",
    )
    parser.add_argument(
        "--training-min-prediction-std-ratios",
        type=_comma_separated_floats,
        default=[],
        help="Per-action training targets for the one-sided prediction variance loss.",
    )
    parser.add_argument(
        "--mirror-augmentation-probability",
        type=float,
        default=0.0,
        help=(
            "Probability of reflecting each training sequence across the "
            "world x-axis. Validation and test data are never augmented."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-rule",
        choices=["qualification_then_loss", "validation_loss"],
        default="validation_loss",
        help=(
            "Select by validation loss by default and report qualification "
            "independently. The qualification_then_loss option is retained "
            "only for explicit legacy reproduction."
        ),
    )
    parser.add_argument(
        "--test-evaluation-mode",
        choices=["deferred", "evaluate"],
        default="deferred",
        help=(
            "Keep offline and closed-loop test qualification pending during "
            "validation-based method/epoch/seed selection, or explicitly open "
            "the test split for a final locked evaluation."
        ),
    )
    parser.add_argument(
        "--learning-action-index",
        type=int,
        default=0,
        help="Action dimension used by anti-collapse prediction-variance and correlation gates.",
    )
    parser.add_argument(
        "--min-learning-action-std-ratio",
        type=float,
        default=0.0,
        help="Minimum validation prediction/target standard-deviation ratio for the learning action.",
    )
    parser.add_argument(
        "--min-learning-action-correlation",
        type=float,
        default=-1.0,
        help="Minimum validation prediction/target correlation for the learning action.",
    )
    parser.add_argument(
        "--learning-action-indices",
        type=_comma_separated_ints,
        default=[],
        help="Optional comma-separated action dimensions that must all pass.",
    )
    parser.add_argument(
        "--min-learning-action-std-ratios",
        type=_comma_separated_floats,
        default=[],
        help="Per-dimension minimum prediction/target standard-deviation ratios.",
    )
    parser.add_argument(
        "--min-learning-action-correlations",
        type=_comma_separated_floats,
        default=[],
        help="Per-dimension minimum prediction/target correlations.",
    )
    parser.add_argument(
        "--require-explicit-data-contracts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--policy-model",
        choices=["recurrent_transformer", "recurrent_gru"],
        default="recurrent_transformer",
        help=(
            "Recurrent architecture to train. The GRU is an explicit diagnostic "
            "control; it does not replace the manuscript Transformer."
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--transformer-layers",
        type=int,
        required=True,
        choices=[1, 2, 3],
    )
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument(
        "--transformer-norm-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pre-norm transformer encoder blocks for optimization stability.",
    )
    parser.add_argument(
        "--transformer-observation-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit and persist a train-split-only standardizer for policy observations.",
    )
    parser.add_argument(
        "--policy-observation-standardization-clip",
        type=float,
        default=0.0,
        help=(
            "Optional symmetric clip after train-split standardization. "
            "Zero disables clipping; a positive value is an explicit "
            "legacy/ablation setting."
        ),
    )
    parser.add_argument(
        "--transformer-observation-tokenization",
        choices=["semantic", "dense_temporal"],
        default="semantic",
        help=(
            "Use sensor-wise current tokens or one normalized observation token "
            "with transformer attention over temporal memory."
        ),
    )
    parser.add_argument(
        "--policy-head-init-std",
        type=float,
        default=-1.0,
        help="Reinitialize the continuous policy head with this std; negative preserves framework initialization.",
    )
    parser.add_argument("--memory-tokens", type=int, default=8)
    parser.add_argument("--memory-context-length", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument(
        "--recurrent-warmup-mode",
        choices=["full_prefix", "bounded_raw_history"],
        default="full_prefix",
        help=(
            "Reconstruct recurrent state from the complete causal prefix. "
            "bounded_raw_history is retained for labelled legacy diagnostics."
        ),
    )
    parser.add_argument("--sequences-per-batch", type=int, default=16)
    parser.add_argument("--micro-batch-sequences", type=int, default=4)

    parser.add_argument("--render-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-steps", type=int, default=200)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument(
        "--evaluation-scenario-seed",
        type=int,
        default=20260716,
        help=(
            "Protocol seed shared across policy seeds for paired scenario "
            "selection and environment resets."
        ),
    )
    parser.add_argument(
        "--validation-evaluation-episodes",
        type=int,
        default=3,
        help="All-vehicle validation episodes used for shared BC/GAIL ranking.",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=["val", "test"],
        default="val",
        help=(
            "Prebuilt replay split used by the unmatched post-training BC "
            "evaluation. Deferred test mode requires val."
        ),
    )
    parser.add_argument(
        "--evaluation-enable-collision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable collision physics during post-training evaluation (disabled by default).",
    )
    parser.add_argument(
        "--matched-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the same fixed-horizon matched-trajectory evaluator used by GAIL.",
    )
    parser.add_argument(
        "--evaluation-vehicle-mode",
        choices=["single", "training_count", "all"],
        default="single",
    )
    parser.add_argument("--min-rollout-steps", type=int, default=100)
    parser.add_argument("--max-crash-fraction", type=float, default=0.34)
    parser.add_argument("--max-offroad-fraction", type=float, default=0.34)
    parser.add_argument(
        "--expert-replay-qualification-required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep policy_realism_qualified false until a frozen same-scenario "
            "expert-replay reference has been attached and audited."
        ),
    )
    parser.add_argument(
        "--maximum-expert-replay-crash-rate-gap",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum-expert-replay-offroad-rate-gap",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--capability-failure-mode",
        choices=["error", "report"],
        default="error",
        help=(
            "For full policy training, error exits non-zero only when a finite "
            "checkpoint artifact is not produced. Offline and closed-loop "
            "quality are always reported. Warm starts retain their stabilization "
            "gate. Code and data failures always exit non-zero."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_worktree_provenance(path: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    )
    return {
        "head": git_head(path),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_short": status.stdout.splitlines() if status.returncode == 0 else [],
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
    }


def expert_provenance(path: Path) -> dict[str, Any]:
    manifest = path / "manifest.json" if path.is_dir() else None
    payload: dict[str, Any] = {"path": str(path.resolve())}
    if manifest is not None and manifest.is_file():
        payload["manifest_path"] = str(manifest.resolve())
        payload["manifest_sha256"] = sha256_file(manifest)
        with manifest.open(encoding="utf-8") as handle:
            data = json.load(handle)
        payload.update(
            {
                "schema_version": data.get("schema_version"),
                "scene": data.get("scene"),
                "prebuilt_split": data.get("prebuilt_split"),
                "num_episodes": data.get("num_episodes"),
                "num_samples": data.get("num_samples"),
            }
        )
    return payload


def expert_split_provenance(args: argparse.Namespace) -> dict[str, Any]:
    validation_path = str(getattr(args, "expert_validation_data", "") or "")
    test_path = str(getattr(args, "expert_test_data", "") or "")
    test_evaluation_mode = str(
        getattr(args, "test_evaluation_mode", "deferred")
    ).lower()
    if test_path and not validation_path:
        raise ValueError(
            "--expert-test-data requires --expert-validation-data."
        )
    if test_evaluation_mode == "deferred" and test_path:
        raise ValueError(
            "Deferred test mode refuses --expert-test-data so the external "
            "test root cannot be opened during selection."
        )
    if (
        test_evaluation_mode == "evaluate"
        and validation_path
        and not test_path
    ):
        raise ValueError(
            "Final evaluate mode requires --expert-test-data when an explicit "
            "validation root is supplied."
        )
    if not validation_path:
        return {
            "method": "internal_trajectory_split_from_training_root",
            "train": expert_provenance(Path(args.expert_data)),
        }
    if test_evaluation_mode == "deferred":
        return {
            "method": "explicit_collected_train_validation_test_deferred",
            "train": expert_provenance(Path(args.expert_data)),
            "validation": expert_provenance(Path(validation_path)),
            "test": {"status": "pending_deferred_not_opened"},
        }
    return {
        "method": "explicit_collected_train_validation_test_directories",
        "train": expert_provenance(Path(args.expert_data)),
        "validation": expert_provenance(Path(validation_path)),
        "test": expert_provenance(Path(test_path)),
    }


def make_config(args: argparse.Namespace) -> PSGAILConfig:
    policy_model = str(getattr(args, "policy_model", "recurrent_transformer"))
    return PSGAILConfig(
        expert_data=str(Path(args.expert_data).resolve()),
        run_name=f"bc_{args.domain}_{policy_model}_{args.transformer_layers}layer_seed_{args.seed}",
        scene=str(args.scene),
        action_mode="continuous",
        episode_root=str(Path(args.episode_root).resolve()),
        prebuilt_split=str(args.prebuilt_split),
        seed=int(args.seed),
        max_expert_samples=int(args.max_expert_samples),
        require_explicit_data_contracts=bool(
            args.require_explicit_data_contracts
        ),
        trajectory_frame="relative",
        max_surrounding="all",
        control_all_vehicles=False,
        percentage_controlled_vehicles=1.0,
        allow_idm=True,
        cells=128,
        maximum_range=64.0,
        simulation_frequency=10,
        policy_frequency=10,
        max_episode_steps=200,
        evaluation_scenario_seed=int(
            getattr(args, "evaluation_scenario_seed", 20260716)
        ),
        road_query_mode="spatial",
        collision_check_mode="broadphase",
        record_replay_diagnostics=False,
        sensor_road_edge_mode="batched",
        reuse_pre_reset_spaces=True,
        policy_model=policy_model,
        hidden_size=int(args.hidden_size),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_dropout=float(args.transformer_dropout),
        transformer_norm_first=bool(args.transformer_norm_first),
        transformer_observation_normalization=bool(
            args.transformer_observation_normalization
        ),
        policy_observation_standardization_clip=float(
            args.policy_observation_standardization_clip
        ),
        transformer_observation_tokenization=str(
            args.transformer_observation_tokenization
        ),
        transformer_memory_tokens=int(args.memory_tokens),
        transformer_memory_context_length=int(args.memory_context_length),
        transformer_recurrent_sequence_length=int(args.sequence_length),
        recurrent_bc_warmup_mode=str(
            getattr(args, "recurrent_warmup_mode", "full_prefix")
        ),
        transformer_recurrent_sequences_per_batch=int(args.sequences_per_batch),
        transformer_recurrent_micro_batch_sequences=int(args.micro_batch_sequences),
        transformer_use_causal_attention=True,
        evaluation_num_workers=1,
        evaluation_worker_threads=2,
        **paper_validation_overrides_for_vehicle_mode(
            args.evaluation_vehicle_mode
        ),
        bc_pretrain_epochs=int(args.epochs),
        bc_pretrain_learning_rate=float(args.learning_rate),
        bc_pretrain_weight_decay=float(args.weight_decay),
        device=str(args.device),
    )


def checkpoint_payload(
    policy: torch.nn.Module,
    cfg: PSGAILConfig,
    *,
    args: argparse.Namespace,
    obs_dim: int,
    action_dim: int,
    training_summary: dict[str, Any],
    training_data_contract: dict[str, object],
    split_trajectory_ids: dict[str, list[str]],
    scenario: tuple[str | None, int | None],
) -> dict[str, Any]:
    component_root = Path(__file__).resolve().parents[1]
    project_root = Path(os.environ.get("VFI_PROJECT_ROOT", component_root)).resolve()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": (
            "behaviour_cloning_warm_start"
            if args.checkpoint_purpose == "warm_start"
            else "behaviour_cloning_best"
        ),
        "policy_state_dict": policy.state_dict(),
        "config": vars(cfg),
        "policy_architecture": {
            "policy_model": str(cfg.policy_model),
            "obs_dim": int(obs_dim),
            "hidden_size": int(cfg.hidden_size),
            "action_mode": "continuous",
            "continuous_action_dim": int(action_dim),
            "transformer_layers": int(cfg.transformer_layers),
            "transformer_heads": int(cfg.transformer_heads),
            "transformer_dropout": float(cfg.transformer_dropout),
            "transformer_norm_first": bool(cfg.transformer_norm_first),
            "transformer_observation_normalization": bool(
                cfg.transformer_observation_normalization
            ),
            "policy_observation_standardization_clip": float(
                cfg.policy_observation_standardization_clip
            ),
            "transformer_observation_tokenization": str(
                cfg.transformer_observation_tokenization
            ),
            "policy_head_init_std": float(args.policy_head_init_std),
            "transformer_memory_tokens": int(cfg.transformer_memory_tokens),
            "transformer_memory_context_length": int(cfg.transformer_memory_context_length),
            "transformer_use_causal_attention": True,
        },
        "bc_stats": training_summary,
        "training_data_contract": training_data_contract,
        "policy_output_action_contract": training_data_contract[
            "continuous_action"
        ],
        "policy_observation_contract": training_data_contract[
            "policy_observation"
        ],
        "data_split": {
            "method": expert_split_provenance(args)["method"],
            "seed": int(args.split_seed),
            "trajectory_ids": split_trajectory_ids,
        },
        "expert_data": expert_split_provenance(args),
        "provenance": {
            "component_git": git_worktree_provenance(component_root),
            "project_git": git_worktree_provenance(project_root),
            "source_sha256": {
                "recurrent_bc.py": sha256_file(component_root / "scripts_gail" / "ps_gail" / "recurrent_bc.py"),
                "train_recurrent_bc_policy.py": sha256_file(Path(__file__).resolve()),
                "pretrain_continuous_bc_policy.py": sha256_file(
                    component_root
                    / "scripts_gail"
                    / "pretrain_continuous_bc_policy.py"
                ),
                "train_simple_ps_gail.py": sha256_file(
                    component_root / "scripts_gail" / "train_simple_ps_gail.py"
                ),
                "training/evaluation.py": sha256_file(
                    component_root
                    / "scripts_gail"
                    / "ps_gail"
                    / "training"
                    / "evaluation.py"
                ),
                "training/policy.py": sha256_file(
                    component_root
                    / "scripts_gail"
                    / "ps_gail"
                    / "training"
                    / "policy.py"
                ),
                "training/ppo.py": sha256_file(
                    component_root
                    / "scripts_gail"
                    / "ps_gail"
                    / "training"
                    / "ppo.py"
                ),
                "ps_gail/envs.py": sha256_file(
                    component_root
                    / "scripts_gail"
                    / "ps_gail"
                    / "envs.py"
                ),
                "models.py": sha256_file(component_root / "scripts_gail" / "ps_gail" / "models.py"),
                "config.py": sha256_file(
                    component_root / "scripts_gail" / "ps_gail" / "config.py"
                ),
                "data.py": sha256_file(
                    component_root / "scripts_gail" / "ps_gail" / "data.py"
                ),
                "checkpoints.py": sha256_file(
                    component_root
                    / "scripts_gail"
                    / "ps_gail"
                    / "checkpoints.py"
                ),
                "contracts.py": sha256_file(
                    component_root
                    / "scripts_gail"
                    / "ps_gail"
                    / "contracts.py"
                ),
                "highway_env/envs/common/action.py": sha256_file(
                    component_root
                    / "highway_env"
                    / "envs"
                    / "common"
                    / "action.py"
                ),
                "highway_env/envs/ngsim_env.py": sha256_file(
                    component_root / "highway_env" / "envs" / "ngsim_env.py"
                ),
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "command": [str(value) for value in sys.argv],
        },
        "default_video_scenario": {"episode_name": scenario[0], "vehicle_id": scenario[1]},
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append and flush one progress record so interrupted runs retain history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_checkpoint_artifacts(
    checkpoint_path: Path,
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    """Atomically persist a metric-qualified checkpoint and its integrity metadata."""
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    (checkpoint_path.parent / f"{checkpoint_path.name}.sha256").write_text(
        f"{checkpoint_sha256}  {checkpoint_path.name}\n",
        encoding="utf-8",
    )
    summary["checkpoint_saved"] = True
    summary["checkpoint_sha256"] = checkpoint_sha256
    return checkpoint_sha256


def run_training(
    args: argparse.Namespace,
    *,
    transitions: Any | None = None,
    prepared_data: PreparedRecurrentBCData | None = None,
) -> dict[str, Any]:
    """Run one BC cell, optionally reusing already-loaded and prepared data."""
    test_evaluation_mode = str(
        getattr(args, "test_evaluation_mode", "deferred")
    ).lower()
    if test_evaluation_mode not in {"deferred", "evaluate"}:
        raise ValueError(
            "test_evaluation_mode must be 'deferred' or 'evaluate'; "
            f"got {test_evaluation_mode!r}."
        )
    if (
        test_evaluation_mode == "deferred"
        and not bool(args.matched_evaluation)
        and str(args.evaluation_split) != "val"
    ):
        raise ValueError(
            "Deferred test mode requires --evaluation-split val for unmatched "
            "evaluation; refusing to open the test split during selection."
        )
    if args.checkpoint_purpose == "warm_start" and not (
        1 <= int(args.epochs) <= int(args.max_warmup_epochs)
    ):
        raise ValueError(
            "A BC warm start must use between 1 and "
            f"{int(args.max_warmup_epochs)} epochs; got {int(args.epochs)}."
        )
    out_dir = prepare_fresh_output_directory(Path(args.out_dir))
    checkpoint_path = out_dir / "best.pt"
    source_root_integrity = validate_explicit_expert_source_roots(args)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = resolve_device(str(args.device))
    cfg = make_config(args)

    scenario = default_scenario_from_expert_folder(str(args.expert_data))
    if not scenario[0] or scenario[1] is None:
        raise RuntimeError("Could not infer an episode and vehicle from the expert dataset.")
    environment = make_selected_replay_env(
        cfg,
        episode_name=str(scenario[0]),
        vehicle_id=int(scenario[1]),
        render_mode=None,
    )
    try:
        policy, obs_dim, action_dim = build_policy_for_env(cfg, environment, device)
    finally:
        environment.close()
    cfg.continuous_action_dim = int(action_dim)

    if float(args.policy_head_init_std) >= 0.0:
        policy_head = getattr(policy, "policy_head", None)
        if not isinstance(policy_head, torch.nn.Linear):
            raise TypeError("Policy-head initialization requires a linear continuous policy head.")
        torch.nn.init.normal_(policy_head.weight, mean=0.0, std=float(args.policy_head_init_std))
        torch.nn.init.zeros_(policy_head.bias)
    # The policy has already been constructed, so recording the realized
    # initialization here cannot trigger a second initialization or change RNG.
    cfg.policy_head_init_std = float(args.policy_head_init_std)

    if transitions is None:
        training_transitions = load_expert_transition_data(
            str(args.expert_data),
            max_samples=int(args.max_expert_samples),
            seed=int(args.data_seed),
            trajectory_frame="relative",
        )
        validation_path = str(
            getattr(args, "expert_validation_data", "") or ""
        )
        test_path = str(getattr(args, "expert_test_data", "") or "")
        if test_path and not validation_path:
            raise ValueError(
                "--expert-test-data requires --expert-validation-data."
            )
        if test_evaluation_mode == "deferred" and test_path:
            raise ValueError(
                "Deferred test mode refuses --expert-test-data so the external "
                "test root cannot be opened during selection."
            )
        if (
            test_evaluation_mode == "evaluate"
            and validation_path
            and not test_path
        ):
            raise ValueError(
                "Final evaluate mode requires --expert-test-data when an "
                "explicit validation root is supplied."
            )
        if validation_path:
            validation_transitions = load_expert_transition_data(
                validation_path,
                max_samples=int(
                    getattr(args, "max_validation_samples", 100_000)
                ),
                seed=int(args.data_seed) + 1,
                trajectory_frame="relative",
            )
            explicit_splits = {
                "train": training_transitions,
                "validation": validation_transitions,
            }
            if test_evaluation_mode == "evaluate":
                explicit_splits["test"] = load_expert_transition_data(
                    test_path,
                    max_samples=int(
                        getattr(args, "max_test_samples", 100_000)
                    ),
                    seed=int(args.data_seed) + 2,
                    trajectory_frame="relative",
                )
            prepared_data = prepare_recurrent_bc_data_from_explicit_splits(
                explicit_splits,
                sequence_length=int(args.sequence_length),
                context_length=int(args.memory_context_length),
                warmup_mode=str(
                    getattr(args, "recurrent_warmup_mode", "full_prefix")
                ),
            )
            transitions = prepared_data.transitions
        else:
            transitions = training_transitions
    training_data_contract = validate_recurrent_bc_source_contracts(
        transitions.metadata,
        lidar_cells=int(cfg.cells),
        maximum_range=float(cfg.maximum_range),
        require_explicit=bool(args.require_explicit_data_contracts),
        expected_scene=str(cfg.scene),
    )
    training_data_contract["source_root_integrity"] = source_root_integrity
    row_sampling_receipt = recurrent_bc_row_sampling_receipt(
        transitions.metadata,
        requested_train_rows=int(args.max_expert_samples),
        requested_validation_rows=int(
            getattr(args, "max_validation_samples", 100_000)
        ),
        requested_test_rows=int(
            getattr(args, "max_test_samples", 100_000)
        ),
    )
    if int(transitions.policy_observations.shape[1]) != int(obs_dim):
        raise RuntimeError(
            f"Expert and environment observation dimensions differ: "
            f"{transitions.policy_observations.shape[1]} != {obs_dim}."
        )
    if int(transitions.actions_continuous_env.shape[1]) != int(action_dim):
        raise RuntimeError(
            f"Expert and environment action dimensions differ: "
            f"{transitions.actions_continuous_env.shape[1]} != {action_dim}."
        )

    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    result = train_recurrent_behavior_clone(
        policy,
        transitions,
        device=device,
        seed=int(args.seed),
        split_seed=int(args.split_seed),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        sequence_length=int(args.sequence_length),
        sequences_per_batch=int(args.sequences_per_batch),
        micro_batch_sequences=int(args.micro_batch_sequences),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        max_grad_norm=float(args.max_grad_norm),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_epochs=int(
            getattr(args, "early_stopping_min_epochs", 0)
        ),
        early_stopping_min_delta_relative=float(
            getattr(args, "early_stopping_min_delta_relative", 0.001)
        ),
        selection_min_validation_skill=float(args.min_validation_skill),
        action_loss_weights=list(args.action_loss_weights),
        action_loss_weighting=str(args.action_loss_weighting),
        correlation_loss_weight=float(args.correlation_loss_weight),
        variance_loss_weight=float(args.variance_loss_weight),
        minimum_prediction_std_ratios=(
            list(args.training_min_prediction_std_ratios)
            if args.training_min_prediction_std_ratios
            else None
        ),
        selection_min_prediction_std_ratios=(
            list(args.min_learning_action_std_ratios)
            if args.min_learning_action_std_ratios
            else None
        ),
        selection_min_prediction_correlations=(
            list(args.min_learning_action_correlations)
            if args.min_learning_action_correlations
            else None
        ),
        checkpoint_selection_rule=str(args.checkpoint_selection_rule),
        mirror_augmentation_probability=float(
            args.mirror_augmentation_probability
        ),
        warmup_mode=str(
            getattr(args, "recurrent_warmup_mode", "full_prefix")
        ),
        evaluate_test=test_evaluation_mode == "evaluate",
        epoch_callback=lambda row: append_jsonl(metrics_path, row),
        prepared_data=prepared_data,
    )
    policy.load_state_dict(result.best_state_dict, strict=True)
    policy.eval()
    summary = {
        **result.summary,
        "domain": str(args.domain),
        "scene": str(args.scene),
        "seed": int(args.seed),
        "evaluation_scenario_seed": int(
            getattr(args, "evaluation_scenario_seed", 20260716)
        ),
        "policy_model": str(args.policy_model),
        "checkpoint_purpose": str(args.checkpoint_purpose),
        "transformer_layers": int(args.transformer_layers),
        "checkpoint": str(checkpoint_path),
        "expert_data": expert_split_provenance(args),
        "offline_split_method": expert_split_provenance(args)["method"],
        "capability_threshold": float(args.min_validation_skill),
        "max_validation_mae": float(args.max_validation_mae),
        "validation_history": str(metrics_path),
        "validation_history_epochs": len(result.history),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "transformer_dropout": float(args.transformer_dropout),
        "transformer_norm_first": bool(args.transformer_norm_first),
        "transformer_observation_normalization": bool(
            args.transformer_observation_normalization
        ),
        "policy_observation_standardization_clip": float(
            args.policy_observation_standardization_clip
        ),
        "transformer_observation_tokenization": str(
            args.transformer_observation_tokenization
        ),
        "policy_head_init_std": float(args.policy_head_init_std),
        "recurrent_warmup_mode": str(
            getattr(args, "recurrent_warmup_mode", "full_prefix")
        ),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_epochs": int(
            getattr(args, "early_stopping_min_epochs", 0)
        ),
        "early_stopping_min_delta_relative": float(
            getattr(args, "early_stopping_min_delta_relative", 0.001)
        ),
        "configured_action_loss_weights": list(args.action_loss_weights),
        "action_loss_weighting": str(args.action_loss_weighting),
        "correlation_loss_weight": float(args.correlation_loss_weight),
        "variance_loss_weight": float(args.variance_loss_weight),
        "mirror_augmentation_probability": float(
            args.mirror_augmentation_probability
        ),
        "checkpoint_selection_rule": str(
            args.checkpoint_selection_rule
        ),
        "test_evaluation_mode": test_evaluation_mode,
        "training_min_prediction_std_ratios": list(
            args.training_min_prediction_std_ratios
        ),
        "training_data_contract": training_data_contract,
        "policy_output_action_contract": training_data_contract[
            "continuous_action"
        ],
        "policy_observation_contract": training_data_contract[
            "policy_observation"
        ],
        "row_sampling_receipt": row_sampling_receipt,
    }

    learning_action_indices = (
        list(args.learning_action_indices)
        if args.learning_action_indices
        else [int(args.learning_action_index)]
    )
    minimum_std_ratios = (
        list(args.min_learning_action_std_ratios)
        if args.min_learning_action_std_ratios
        else [float(args.min_learning_action_std_ratio)]
    )
    minimum_correlations = (
        list(args.min_learning_action_correlations)
        if args.min_learning_action_correlations
        else [float(args.min_learning_action_correlation)]
    )
    validation_learning_gate = action_learning_gate(
        split="validation",
        prediction_std_ratios=result.summary[
            "validation_prediction_std_ratio"
        ],
        prediction_target_correlations=result.summary[
            "validation_prediction_target_correlation"
        ],
        action_indices=learning_action_indices,
        minimum_std_ratios=minimum_std_ratios,
        minimum_correlations=minimum_correlations,
    )
    learning_signal_passed = bool(validation_learning_gate["passed"])
    metric_capability = bool(
        result.summary["validation_skill"] >= float(args.min_validation_skill)
        and result.summary["validation_mae"] <= float(args.max_validation_mae)
        and learning_signal_passed
    )
    if test_evaluation_mode == "evaluate":
        held_out_learning_gate = action_learning_gate(
            split="test",
            prediction_std_ratios=result.summary["test_prediction_std_ratio"],
            prediction_target_correlations=result.summary[
                "test_prediction_target_correlation"
            ],
            action_indices=learning_action_indices,
            minimum_std_ratios=minimum_std_ratios,
            minimum_correlations=minimum_correlations,
        )
        held_out_learning_signal_passed = bool(
            held_out_learning_gate["passed"]
        )
        held_out_metric_capability = bool(
            np.isfinite(float(result.summary["test_mse"]))
            and np.isfinite(float(result.summary["test_mae"]))
            and float(result.summary["test_mae"])
            <= float(args.max_validation_mae)
            and held_out_learning_signal_passed
        )
    else:
        held_out_learning_gate = {
            "split": "test",
            "status": "pending_deferred",
            "passed": False,
            "actions": [],
        }
        held_out_learning_signal_passed = False
        held_out_metric_capability = False
    summary["learning_signal_passed"] = learning_signal_passed
    summary["learning_signal_gate"] = validation_learning_gate
    summary["validation_metric_capability_passed"] = metric_capability
    summary["held_out_learning_signal_passed"] = held_out_learning_signal_passed
    summary["held_out_learning_signal_gate"] = held_out_learning_gate
    summary["held_out_metric_capability_passed"] = held_out_metric_capability
    summary["final_test_qualification_status"] = (
        "evaluated" if test_evaluation_mode == "evaluate" else "pending_deferred"
    )
    summary["final_test_qualification_passed"] = bool(
        test_evaluation_mode == "evaluate" and held_out_metric_capability
    )
    summary["offline_capability_passed"] = bool(
        test_evaluation_mode == "evaluate"
        and metric_capability
        and held_out_metric_capability
    )
    warm_start_passed = bool(
        args.checkpoint_purpose == "warm_start"
        and np.isfinite(result.summary["initial_validation_mse"])
        and np.isfinite(result.summary["validation_mse"])
        and result.summary["relative_validation_improvement"]
        >= float(args.min_warmup_relative_improvement)
    )
    summary["metric_capability_passed"] = metric_capability
    summary["warm_start_passed"] = warm_start_passed
    summary["warm_start_gate"] = {
        "configured_epochs": int(args.epochs),
        "maximum_epochs": int(args.max_warmup_epochs),
        "relative_validation_improvement": float(result.summary["relative_validation_improvement"]),
        "minimum_relative_validation_improvement": float(args.min_warmup_relative_improvement),
    }
    summary["checkpoint_saved"] = False
    summary["checkpoint_eligibility"] = (
        "warm_start_passed" if args.checkpoint_purpose == "warm_start" else "full_policy_best_validation"
    )
    checkpoint_eligible = warm_start_passed if args.checkpoint_purpose == "warm_start" else True
    write_json(out_dir / "split_manifest.json", result.split_trajectory_ids)

    # A full BC study retains its best-validation model even when capability
    # gates reject it; those gates control promotion, not artifact retention.
    # Short GAIL/AIRL warm-ups still have their own stabilization gate.
    if checkpoint_eligible:
        payload = checkpoint_payload(
            policy,
            cfg,
            args=args,
            obs_dim=obs_dim,
            action_dim=action_dim,
            training_summary={
                **result.summary,
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "max_grad_norm": float(args.max_grad_norm),
                "transformer_dropout": float(args.transformer_dropout),
                "transformer_norm_first": bool(args.transformer_norm_first),
                "transformer_observation_normalization": bool(
                    args.transformer_observation_normalization
                ),
                "policy_observation_standardization_clip": float(
                    args.policy_observation_standardization_clip
                ),
                "transformer_observation_tokenization": str(
                    args.transformer_observation_tokenization
                ),
                "policy_head_init_std": float(args.policy_head_init_std),
                "early_stopping_patience": int(args.early_stopping_patience),
                "early_stopping_min_epochs": int(
                    getattr(args, "early_stopping_min_epochs", 0)
                ),
                "early_stopping_min_delta_relative": float(
                    getattr(
                        args,
                        "early_stopping_min_delta_relative",
                        0.001,
                    )
                ),
                "learning_signal_passed": learning_signal_passed,
                "learning_signal_gate": dict(summary["learning_signal_gate"]),
                "held_out_learning_signal_passed": (
                    held_out_learning_signal_passed
                ),
                "held_out_learning_signal_gate": dict(
                    summary["held_out_learning_signal_gate"]
                ),
                "offline_capability_passed": bool(
                    summary["offline_capability_passed"]
                ),
                "configured_action_loss_weights": list(
                    args.action_loss_weights
                ),
                "action_loss_weighting": str(args.action_loss_weighting),
                "correlation_loss_weight": float(args.correlation_loss_weight),
                "variance_loss_weight": float(args.variance_loss_weight),
                "mirror_augmentation_probability": float(
                    args.mirror_augmentation_probability
                ),
                "checkpoint_selection_rule": str(
                    args.checkpoint_selection_rule
                ),
                "test_evaluation_mode": test_evaluation_mode,
                "training_min_prediction_std_ratios": list(
                    args.training_min_prediction_std_ratios
                ),
                "row_sampling_receipt": row_sampling_receipt,
            },
            training_data_contract=training_data_contract,
            split_trajectory_ids=result.split_trajectory_ids,
            scenario=scenario,
        )
        save_checkpoint_artifacts(checkpoint_path, payload, summary)
    write_json(out_dir / "summary.json", summary)

    evaluation_cfg = replace(
        cfg,
        prebuilt_split=str(args.evaluation_split),
        enable_collision=bool(args.evaluation_enable_collision),
        bc_pretrain_eval_deterministic=True,
    )
    test_evaluation_enabled = test_evaluation_mode == "evaluate"
    survival_stats: dict[str, Any] = {}
    validation_survival_stats: dict[str, Any] = {}
    validation_stats: dict[str, Any] = {}
    matched_validation_contract_passed = False
    matched_test_contract_passed = False
    validation_evaluation_contract_passed = False
    test_evaluation_contract_passed = False
    matched_evaluation_contract_passed = False
    matched_evaluation_complete = False
    matched_closed_loop_quality_passed = False
    validation_quality_gate: dict[str, Any] | None = None
    test_quality_gate: dict[str, Any] | None = None
    if bool(args.matched_evaluation):
        # BC and GAIL use this exact fixed-horizon, collision-enabled evaluator
        # and scoring function. Collision termination is ignored so poor BC
        # behavior remains measurable through the requested horizon.
        matched_cfg = replace(
            evaluation_cfg,
            enable_collision=True,
            **paper_validation_overrides_for_vehicle_mode(
                args.evaluation_vehicle_mode
            ),
        )
        validation_stats = evaluate_policy_matched_trajectories(
            policy,
            matched_cfg,
            device,
            split="val",
            episodes=int(args.validation_evaluation_episodes),
            prefix="validation",
        )
        validation_stats, validation_cost, validation_score = (
            scored_validation_metrics(
                validation_stats,
                matched_cfg,
                prefix="validation",
            )
        )
        test_stats = (
            evaluate_policy_matched_trajectories(
                policy,
                matched_cfg,
                device,
                split="test",
                episodes=int(args.evaluation_episodes),
                prefix="test",
            )
            if test_evaluation_enabled
            else None
        )
        score_horizon = int(matched_cfg.validation_score_horizon_seconds)
        validation_reference_coverage = float(
            validation_stats.get(
                f"validation/horizon_coverage_{score_horizon}s",
                float("nan"),
            )
        )
        validation_rollout_coverage = float(
            validation_stats.get(
                f"validation/rollout_horizon_coverage_{score_horizon}s",
                validation_reference_coverage,
            )
        )
        matched_validation_contract_passed = bool(
            np.isfinite(validation_score)
            and np.isfinite(validation_rollout_coverage)
            and validation_rollout_coverage
            >= float(matched_cfg.validation_min_horizon_coverage)
        )
        validation_quality_gate = closed_loop_policy_quality(
            validation_stats,
            prefix="validation",
            max_vehicle_crash_rate=float(args.max_crash_fraction),
            max_vehicle_offroad_rate=float(args.max_offroad_fraction),
            score_horizon_seconds=score_horizon,
            min_horizon_coverage=float(
                matched_cfg.validation_min_horizon_coverage
            ),
        )
        if test_stats is not None:
            test_reference_coverage = float(
                test_stats.get(
                    f"test/horizon_coverage_{score_horizon}s",
                    float("nan"),
                )
            )
            test_rollout_coverage = float(
                test_stats.get(
                    f"test/rollout_horizon_coverage_{score_horizon}s",
                    test_reference_coverage,
                )
            )
            matched_test_contract_passed = bool(
                np.isfinite(test_rollout_coverage)
                and test_rollout_coverage
                >= float(matched_cfg.validation_min_horizon_coverage)
            )
            test_quality_gate = closed_loop_policy_quality(
                test_stats,
                prefix="test",
                max_vehicle_crash_rate=float(args.max_crash_fraction),
                max_vehicle_offroad_rate=float(args.max_offroad_fraction),
                score_horizon_seconds=score_horizon,
                min_horizon_coverage=float(
                    matched_cfg.validation_min_horizon_coverage
                ),
            )
            matched_evaluation_complete = bool(
                validation_stats
                and test_stats
                and np.isfinite(validation_rollout_coverage)
                and np.isfinite(test_rollout_coverage)
            )
        matched_evaluation_contract_passed = bool(
            test_evaluation_enabled
            and matched_validation_contract_passed
            and matched_test_contract_passed
        )
        validation_evaluation_contract_passed = (
            matched_validation_contract_passed
        )
        test_evaluation_contract_passed = matched_test_contract_passed
        matched_closed_loop_quality_passed = bool(
            matched_evaluation_contract_passed
            and validation_quality_gate["passed"]
            and test_quality_gate is not None
            and test_quality_gate["passed"]
        )
        summary["gail_aligned_matched_evaluation"] = {
            "validation_split": "val",
            "test_split": "test",
            "collision_physics_enabled": True,
            "collision_termination_enabled": False,
            "validation_vehicle_mode": str(
                matched_cfg.validation_vehicle_mode
            ),
            "test_vehicle_mode": str(matched_cfg.test_vehicle_mode),
            "validation_episodes": int(args.validation_evaluation_episodes),
            "test_episodes": int(args.evaluation_episodes),
            "score_horizon_seconds": score_horizon,
            "minimum_horizon_coverage": float(
                matched_cfg.validation_min_horizon_coverage
            ),
            "coverage_semantics": {
                "rollout_horizon_coverage": (
                    "fraction of requested vehicle rollouts that executed the "
                    "requested number of policy steps"
                ),
                "reference_horizon_coverage": (
                    "fraction with a paired expert reference sample available "
                    "for horizon-specific RMSE"
                ),
                "legacy_horizon_coverage": "reference_horizon_coverage",
            },
            "checkpoint_selection": {
                "framework": PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
                "validation_cost": float(validation_cost),
                "validation_score": float(validation_score),
                "components": {
                    key: value
                    for key, value in validation_stats.items()
                    if key.startswith("validation/score_component_")
                },
            },
            "validation_metrics": validation_stats,
            "test_metrics": test_stats,
            "test_status": (
                "evaluated" if test_stats is not None else "pending_deferred"
            ),
            "validation_contract_passed": matched_validation_contract_passed,
            "test_contract_passed": matched_test_contract_passed,
            "evaluation_complete": matched_evaluation_complete,
            "contract_passed": matched_evaluation_contract_passed,
            "closed_loop_quality_gate": {
                "validation": validation_quality_gate,
                "test": test_quality_gate,
            },
            "passed": matched_closed_loop_quality_passed,
        }
        summary["paper_validation_cost"] = float(validation_cost)
        summary["paper_validation_score"] = float(validation_score)
        summary["validation_rollouts"] = validation_stats
        if test_stats is not None:
            summary["held_out_rollouts"] = test_stats
            summary["held_out_evaluation"] = {
                "status": "evaluated",
                "prebuilt_split": "test",
                "collision_physics_enabled": True,
                "collision_termination_enabled": False,
                "vehicle_mode": str(matched_cfg.test_vehicle_mode),
                "episodes": int(args.evaluation_episodes),
                "metrics": test_stats,
            }
        else:
            summary["held_out_rollouts"] = None
            summary["held_out_evaluation"] = {
                "status": "pending_deferred",
                "prebuilt_split": "test",
                "metrics": None,
            }
    else:
        evaluated_survival = evaluate_policy_survival(
            policy,
            evaluation_cfg,
            device,
            episodes=int(args.evaluation_episodes),
            seed_offset=10_000,
        )
        if str(args.evaluation_split) == "val":
            validation_survival_stats = evaluated_survival
            summary["validation_rollouts"] = validation_survival_stats
        else:
            survival_stats = evaluated_survival
        validation_evaluation_contract_passed = bool(
            validation_survival_stats
        )
        test_evaluation_contract_passed = bool(
            test_evaluation_enabled and survival_stats
        )
        matched_validation_contract_passed = False
        matched_test_contract_passed = False
        matched_evaluation_contract_passed = False
        matched_evaluation_complete = False
        if survival_stats:
            summary["held_out_rollouts"] = survival_stats
            summary["held_out_evaluation"] = {
                "status": "evaluated",
                "prebuilt_split": str(args.evaluation_split),
                "collision_physics_enabled": bool(
                    args.evaluation_enable_collision
                ),
                "collision_termination_enabled": False,
                "episodes": int(args.evaluation_episodes),
                "metrics": survival_stats,
            }
        else:
            summary["held_out_rollouts"] = None
            summary["held_out_evaluation"] = {
                "status": "pending_deferred",
                "prebuilt_split": "test",
                "metrics": None,
            }

    completion_flags = evaluation_completion_flags(
        matched_evaluation=bool(args.matched_evaluation),
        matched_validation_evaluated=bool(validation_stats),
        matched_test_evaluated=bool(
            test_evaluation_enabled and summary.get("held_out_rollouts")
        ),
        validation_survival_evaluated=bool(validation_survival_stats),
        test_survival_evaluated=bool(
            test_evaluation_enabled and survival_stats
        ),
    )
    matched_evaluation_complete = completion_flags[
        "matched_evaluation_complete"
    ]

    rollout_stats: dict[str, Any] | None = None
    if bool(args.render_video):
        rollout_stats = render_selected_replay(
            policy,
            cfg,
            episode_name=str(scenario[0]),
            vehicle_id=int(scenario[1]),
            device=device,
            video_path=str(out_dir / "evaluation.mp4"),
            steps=int(args.video_steps),
            deterministic=True,
            screen_width=1200,
            screen_height=608,
            scaling=5.5,
        )
        summary["rollout"] = rollout_stats

    validation_rollout_capability = (
        bool(
            matched_validation_contract_passed
            and validation_quality_gate is not None
            and validation_quality_gate["passed"]
        )
        if bool(args.matched_evaluation)
        else survival_quality_passed(
            validation_survival_stats,
            min_rollout_steps=int(args.min_rollout_steps),
            max_collision_fraction=float(args.max_crash_fraction),
            max_offroad_fraction=float(args.max_offroad_fraction),
        )
    )
    validation_collision_physics_enabled = bool(
        True
        if bool(args.matched_evaluation)
        else args.evaluation_enable_collision
    )
    final_collision_physics_configured = bool(
        True
        if bool(args.matched_evaluation)
        else args.evaluation_enable_collision
    )
    (
        final_collision_physics_enabled,
        final_collision_physics_status,
    ) = collision_physics_evaluation_status(
        configured=final_collision_physics_configured,
        evaluated=test_evaluation_enabled,
    )
    summary["validation_collision_physics_enabled"] = (
        validation_collision_physics_enabled
    )
    summary["final_collision_physics_configured"] = (
        final_collision_physics_configured
    )
    summary["final_collision_physics_enabled"] = (
        final_collision_physics_enabled
    )
    summary["final_collision_physics_status"] = (
        final_collision_physics_status
    )
    rollout_capability = (
        bool(matched_closed_loop_quality_passed)
        if bool(args.matched_evaluation)
        else bool(
            test_evaluation_enabled
            and survival_quality_passed(
                survival_stats,
                min_rollout_steps=int(args.min_rollout_steps),
                max_collision_fraction=float(args.max_crash_fraction),
                max_offroad_fraction=float(args.max_offroad_fraction),
            )
        )
    )
    summary["validation_rollout_capability_passed"] = (
        validation_rollout_capability
    )
    summary["rollout_capability_passed"] = rollout_capability
    summary["matched_validation_contract_passed"] = (
        matched_validation_contract_passed
    )
    summary["matched_test_contract_passed"] = matched_test_contract_passed
    summary["validation_evaluation_contract_passed"] = (
        validation_evaluation_contract_passed
    )
    summary["test_evaluation_contract_passed"] = (
        test_evaluation_contract_passed
    )
    summary["matched_evaluation_complete"] = matched_evaluation_complete
    summary["validation_survival_evaluation_complete"] = completion_flags[
        "validation_survival_evaluation_complete"
    ]
    summary["test_survival_evaluation_complete"] = completion_flags[
        "test_survival_evaluation_complete"
    ]
    summary["matched_evaluation_contract_passed"] = (
        matched_evaluation_contract_passed
    )
    summary["matched_evaluation_passed"] = bool(
        args.matched_evaluation and matched_closed_loop_quality_passed
    )
    summary["closed_loop_quality_gate"] = (
        {
            "validation": validation_quality_gate,
            "test": test_quality_gate,
        }
        if bool(args.matched_evaluation)
        else None
    )
    summary["capability_passed"] = bool(
        summary["offline_capability_passed"]
        and rollout_capability
        and matched_evaluation_contract_passed
        and final_collision_physics_enabled
    )
    summary["closed_loop_quality_passed"] = bool(
        rollout_capability
        and matched_evaluation_contract_passed
        and final_collision_physics_enabled
    )
    summary["closed_loop_evaluation_complete"] = bool(
        matched_evaluation_complete
        if bool(args.matched_evaluation)
        else test_evaluation_enabled and survival_stats
    )
    summary["validation_closed_loop_evaluation_complete"] = bool(
        validation_stats
        if bool(args.matched_evaluation)
        else validation_survival_stats
    )
    expert_replay_required = bool(
        getattr(args, "expert_replay_qualification_required", False)
    )
    expert_replay_status = (
        "pending_frozen_reference"
        if expert_replay_required
        else "not_required_by_recipe"
    )
    expert_replay_gate_passed = not expert_replay_required
    summary["expert_replay_qualification"] = {
        "required_for_policy_realism": expert_replay_required,
        "reference_attached": False,
        "status": expert_replay_status,
        "passed": expert_replay_gate_passed,
        "maximum_vehicle_crash_rate_gap": float(
            getattr(args, "maximum_expert_replay_crash_rate_gap", 0.0)
        ),
        "maximum_vehicle_offroad_rate_gap": float(
            getattr(args, "maximum_expert_replay_offroad_rate_gap", 0.0)
        ),
    }
    summary["training_artifact_complete"] = training_artifact_is_complete(
        summary,
        out_dir,
    )
    summary["validation_selection_eligible"] = (
        validation_selection_is_eligible(
            training_artifact_complete=bool(
                summary["training_artifact_complete"]
            ),
            metric_capability_passed=metric_capability,
            validation_rollout_capability_passed=(
                validation_rollout_capability
            ),
            validation_contract_passed=(
                validation_evaluation_contract_passed
            ),
            collision_physics_enabled=(
                validation_collision_physics_enabled
            ),
        )
    )
    summary["final_test_qualification_passed"] = (
        final_test_qualification_is_eligible(
            test_evaluation_enabled=test_evaluation_enabled,
            held_out_metric_capability_passed=held_out_metric_capability,
            rollout_capability_passed=rollout_capability,
            test_contract_passed=test_evaluation_contract_passed,
            collision_physics_enabled=bool(
                final_collision_physics_enabled
            ),
        )
    )
    summary["final_test_qualification_status"] = (
        (
            "passed"
            if summary["final_test_qualification_passed"]
            else "failed"
        )
        if test_evaluation_enabled
        else "pending_deferred"
    )
    summary["policy_realism_qualification_status"] = (
        (
            "passed"
            if summary["training_artifact_complete"]
            and summary["capability_passed"]
            else "failed_policy_capability"
        )
        if expert_replay_gate_passed
        else expert_replay_status
    )
    summary["policy_realism_qualified"] = bool(
        summary["training_artifact_complete"]
        and summary["capability_passed"]
        and expert_replay_gate_passed
    )
    summary["interpretability_baseline_eligible"] = bool(
        summary["training_artifact_complete"]
        and summary["policy_realism_qualified"]
    )
    summary["benchmark_contract"] = {
        "objective": "behavior_cloning_policy_realism_reference",
        "matrix_completion_gate": "all_training_artifacts_complete",
        "interpretability_eligibility_gate": "policy_realism_qualified",
        "selection_eligibility_gate": "validation_selection_eligible",
        "test_evaluation_mode": test_evaluation_mode,
        "closed_loop_metrics_role": "qualification_non_terminal_for_matrix_execution",
        "shared_validation_framework": PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
        "validation_vehicle_mode": str(args.evaluation_vehicle_mode),
        "test_vehicle_mode": str(args.evaluation_vehicle_mode),
        "collision_termination_enabled": False,
        "closed_loop_metrics_include": [
            "position_rmse_by_horizon",
            "speed_rmse_by_horizon",
            "lane_offset_rmse_by_horizon",
            "vehicle_crash_rate",
            "vehicle_offroad_rate",
            "hard_brake_agent_step_rate",
        ],
        "closed_loop_quality_thresholds": {
            "max_vehicle_crash_rate": float(args.max_crash_fraction),
            "max_vehicle_offroad_rate": float(args.max_offroad_fraction),
        },
        "collision_physics_required_for_selection_and_qualification": True,
        "expert_replay_reference_required": expert_replay_required,
    }
    write_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    requested_gate_passed = (
        warm_start_passed
        if args.checkpoint_purpose == "warm_start"
        else summary["training_artifact_complete"]
    )
    if not requested_gate_passed and args.capability_failure_mode == "error":
        if args.checkpoint_purpose == "warm_start":
            raise RuntimeError(
                "BC warm-start stabilization gate failed: "
                f"relative_validation_improvement={result.summary['relative_validation_improvement']:.4f} "
                f"(minimum {float(args.min_warmup_relative_improvement):.4f})."
            )
        raise RuntimeError(
            "BC training artifact is incomplete or non-finite. Closed-loop "
            "quality metrics are descriptive and cannot trigger this error."
        )

    return summary


def main() -> None:
    run_training(parse_args())

if __name__ == "__main__":
    main()
