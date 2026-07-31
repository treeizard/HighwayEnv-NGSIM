#!/usr/bin/env python3
"""Evaluate one immutable BC checkpoint into a separate qualification receipt.

Development mode is deliberately validation-only.  It rejects every request
that names an offline test root or the simulator test split before loading a
checkpoint or dataset.  Final mode remains disabled until an immutable
protocol artifact and atomic one-shot test ledger are implemented.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.contracts import (
    assert_compatible_action_contracts,
    assert_compatible_observation_contracts,
    validate_training_data_contracts,
)
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.recurrent_bc import (
    build_sequence_windows,
    evaluate_recurrent_bc,
)
from scripts_gail.ps_gail.training.evaluation import (
    _evaluation_protocol_seed,
    _evaluation_episode_names,
    _evaluation_scenarios,
    _evaluation_should_stop,
    _evaluate_policy_matched_trajectories_impl,
    _get_matched_eval_env,
    _parse_evaluation_horizons,
    clear_evaluation_worker_caches,
    evaluate_expert_replay_matched_single_vehicle_floor,
    evaluation_thread_context,
)
from scripts_gail.ps_gail.training.policy import (
    _make_policy_from_state_dict,
    central_critic_observation_dim,
)
from scripts_gail.ps_gail.validation import (
    PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
    action_learning_gate,
    closed_loop_policy_quality,
    paper_driver_model_validation_overrides,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QUALIFICATION_SCHEMA = "bc_checkpoint_qualification_sidecar_v2"
SUPPORTED_VALIDATION_SELECTION_RULES = frozenset(
    {"validation_loss", "qualification_then_loss"}
)
FINAL_TEST_LEDGER_IMPLEMENTED = False
CHECKPOINT_BEHAVIOUR_FIELD_TYPES: dict[str, type] = {
    "policy_model": str,
    "hidden_size": int,
    "action_mode": str,
    "continuous_action_dim": int,
    "transformer_layers": int,
    "transformer_heads": int,
    "transformer_dropout": float,
    "transformer_norm_first": bool,
    "transformer_observation_normalization": bool,
    "policy_observation_standardization_clip": float,
    "transformer_observation_tokenization": str,
    "policy_head_init_std": float,
    "transformer_memory_tokens": int,
    "transformer_memory_context_length": int,
    "transformer_use_causal_attention": bool,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tree_hash_record(root: Path) -> dict[str, object]:
    """Hash every regular file below a dataset root in relative-path order."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {resolved}")
    symlinks = sorted(path for path in resolved.rglob("*") if path.is_symlink())
    if symlinks:
        raise ValueError(
            "Dataset hashing refuses symlinks because their targets are not "
            f"content-addressed by this root: {symlinks[0]}"
        )
    files = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file()
    )
    if not files:
        raise RuntimeError(f"Dataset root contains no regular files: {resolved}")
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(resolved).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        total_bytes += int(size)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(int(size)).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "path": str(resolved),
        "sha256_tree": aggregate.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "aggregation": "sha256(relative_path NUL size NUL file_sha256 LF)",
        "symlinks_followed": False,
    }


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required file not found: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _lexical_root_identity(value: str | Path) -> str:
    """Normalize a recorded root without dereferencing or opening it."""

    return str(Path(value).expanduser().absolute())


def root_manifest_identity(
    root: Path,
    *,
    expected_split: str,
) -> dict[str, Any]:
    """Read split identity metadata without opening any episode payload."""

    resolved = root.resolve()
    record: dict[str, Any] = {
        "root": str(resolved),
        "root_identity": _lexical_root_identity(resolved),
        "expected_split": str(expected_split),
        "metadata_status": "manifest_not_available",
        "scene": None,
        "declared_split": None,
        "canonical_episode_ids": None,
    }
    manifest_path = resolved / "manifest.json"
    if not manifest_path.is_file():
        return record
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Dataset identity manifest is unreadable: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Dataset identity manifest is not an object: {manifest_path}"
        )
    scene = str(manifest.get("scene") or "").strip()
    declared_split = str(manifest.get("prebuilt_split") or "").strip()
    if declared_split != str(expected_split):
        raise ValueError(
            "Dataset identity split mismatch: "
            f"{declared_split!r} != {expected_split!r} at {manifest_path}."
        )
    episodes = manifest.get("episodes")
    if not scene or not isinstance(episodes, list) or not episodes:
        raise ValueError(
            f"Dataset identity manifest lacks scene/episodes: {manifest_path}"
        )
    canonical_ids: list[str] = []
    for row in episodes:
        if not isinstance(row, dict):
            raise ValueError(
                f"Dataset identity manifest has an invalid episode row: "
                f"{manifest_path}"
            )
        episode_name = str(row.get("episode_name") or "").strip()
        if not episode_name:
            raise ValueError(
                "Dataset identity manifest episode lacks episode_name: "
                f"{manifest_path}"
            )
        canonical_ids.append(f"{scene}/{episode_name}")
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError(
            f"Dataset identity manifest repeats canonical episodes: "
            f"{manifest_path}"
        )
    record.update(
        {
            "metadata_status": "manifest_episode_identity_available",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "scene": scene,
            "declared_split": declared_split,
            "canonical_episode_ids": sorted(canonical_ids),
        }
    )
    return record


def checkpoint_source_identities(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract recorded train/validation identities without opening old roots."""

    identities: dict[str, dict[str, Any]] = {}
    training_contract = payload.get("training_data_contract")
    source_integrity = (
        training_contract.get("source_root_integrity")
        if isinstance(training_contract, dict)
        else None
    )
    sources = (
        source_integrity.get("sources")
        if isinstance(source_integrity, dict)
        else None
    )
    if isinstance(sources, dict):
        for role in ("train", "validation"):
            source = sources.get(role)
            if not isinstance(source, dict) or not source.get("root"):
                continue
            canonical_ids = source.get("canonical_episode_ids")
            normalized_ids = (
                [str(value).strip() for value in canonical_ids]
                if isinstance(canonical_ids, list)
                else None
            )
            if normalized_ids is not None and (
                any(not value for value in normalized_ids)
                or len(normalized_ids) != len(set(normalized_ids))
            ):
                raise ValueError(
                    "Checkpoint source identity contains empty or duplicate "
                    f"canonical episodes for {role}."
                )
            identities[f"checkpoint_{role}"] = {
                "root": str(source["root"]),
                "root_identity": _lexical_root_identity(
                    str(source["root"])
                ),
                "metadata_status": (
                    "checkpoint_canonical_episode_identity_available"
                    if normalized_ids is not None
                    else "checkpoint_root_identity_only"
                ),
                "canonical_episode_ids": (
                    sorted(normalized_ids)
                    if normalized_ids is not None
                    else None
                ),
            }
    expert_data = payload.get("expert_data")
    if isinstance(expert_data, dict):
        for role in ("train", "validation"):
            key = f"checkpoint_{role}"
            if key in identities:
                continue
            source = expert_data.get(role)
            if not isinstance(source, dict) or not source.get("path"):
                continue
            identities[key] = {
                "root": str(source["path"]),
                "root_identity": _lexical_root_identity(
                    str(source["path"])
                ),
                "metadata_status": "checkpoint_root_identity_only",
                "canonical_episode_ids": None,
            }
    return identities


def validate_split_identity_contract(
    checkpoint_payload: dict[str, Any],
    *,
    supplied_validation: dict[str, Any],
    supplied_test: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reject same-root or canonical-episode leakage across data roles."""

    identities = {
        **checkpoint_source_identities(checkpoint_payload),
        "supplied_validation": supplied_validation,
    }
    if supplied_test is not None:
        identities["supplied_test"] = supplied_test

    pairwise: dict[str, dict[str, Any]] = {}
    names = sorted(identities)
    validation_alias = frozenset(
        {"checkpoint_validation", "supplied_validation"}
    )
    for left_index, left in enumerate(names):
        left_record = identities[left]
        left_ids = left_record.get("canonical_episode_ids")
        left_set = (
            {str(value) for value in left_ids}
            if isinstance(left_ids, list)
            else None
        )
        for right in names[left_index + 1 :]:
            right_record = identities[right]
            right_ids = right_record.get("canonical_episode_ids")
            right_set = (
                {str(value) for value in right_ids}
                if isinstance(right_ids, list)
                else None
            )
            same_root = (
                left_record.get("root_identity")
                == right_record.get("root_identity")
            )
            overlap = (
                sorted(left_set.intersection(right_set))
                if left_set is not None and right_set is not None
                else None
            )
            pair = frozenset({left, right})
            if pair == validation_alias:
                if same_root and (
                    left_set is not None
                    and right_set is not None
                    and left_set != right_set
                ):
                    raise ValueError(
                        "Checkpoint/supplied validation metadata disagree for "
                        "the same root."
                    )
                if (
                    not same_root
                    and overlap
                    and left_set != right_set
                ):
                    raise ValueError(
                        "Checkpoint/supplied validation roots partially "
                        f"overlap; first={overlap[0]!r}."
                    )
                relationship = (
                    "same_validation_source"
                    if same_root
                    else (
                        "equivalent_validation_episode_identity"
                        if left_set is not None
                        and right_set is not None
                        and left_set == right_set
                        else "distinct_validation_source"
                    )
                )
            else:
                if same_root:
                    raise ValueError(
                        f"Split roots for {left}/{right} identify the same "
                        "source."
                    )
                if overlap:
                    raise ValueError(
                        f"Canonical episode overlap for {left}/{right}; "
                        f"first={overlap[0]!r}."
                    )
                relationship = "disjoint_where_metadata_available"
            pairwise[f"{left}_vs_{right}"] = {
                "same_root": same_root,
                "canonical_episode_metadata_available": (
                    left_set is not None and right_set is not None
                ),
                "canonical_episode_overlap_count": (
                    len(overlap) if overlap is not None else None
                ),
                "relationship": relationship,
            }

    identity_summary = {
        name: {
            "root": record.get("root"),
            "root_identity": record.get("root_identity"),
            "metadata_status": record.get("metadata_status"),
            "canonical_episode_count": (
                len(record["canonical_episode_ids"])
                if isinstance(record.get("canonical_episode_ids"), list)
                else None
            ),
            "canonical_episode_ids_sha256": (
                canonical_sha256(record["canonical_episode_ids"])
                if isinstance(record.get("canonical_episode_ids"), list)
                else None
            ),
        }
        for name, record in sorted(identities.items())
    }
    required_roles = [
        "checkpoint_train",
        "checkpoint_validation",
        "supplied_validation",
    ]
    if supplied_test is not None:
        required_roles.append("supplied_test")
    missing_required_canonical_identities = [
        role
        for role in required_roles
        if role not in identities
        or not isinstance(
            identities[role].get("canonical_episode_ids"),
            list,
        )
        or not identities[role]["canonical_episode_ids"]
    ]
    promotion_eligible = not missing_required_canonical_identities
    return {
        "status": (
            "passed"
            if promotion_eligible
            else "diagnostic_incomplete_canonical_identity"
        ),
        "promotion_eligible": promotion_eligible,
        "missing_required_canonical_identities": (
            missing_required_canonical_identities
        ),
        "rule": (
            "root identity and canonical episode identity disjointness; "
            "checkpoint/supplied validation aliases may match exactly; "
            "all promotion roles require non-empty canonical episode IDs"
        ),
        "identities": identity_summary,
        "pairwise": pairwise,
    }


def reconcile_checkpoint_behaviour_config(
    config_payload: dict[str, Any],
    architecture: dict[str, Any],
) -> dict[str, Any]:
    """Resolve legacy omissions while rejecting every duplicated mismatch.

    ``policy_architecture`` is a redundant integrity record, not permission to
    overwrite a conflicting serialized config. Values used to construct the
    policy must have the exact primitive type emitted by the trainer. A field
    absent from the config may be recovered from the architecture for legacy
    compatibility; a field present in both must be typed-equal.
    """

    resolved = dict(config_payload)
    problems: list[str] = []
    for name, expected_type in CHECKPOINT_BEHAVIOUR_FIELD_TYPES.items():
        config_present = name in config_payload
        architecture_present = name in architecture
        config_value = config_payload.get(name)
        architecture_value = architecture.get(name)

        for source, present, value in (
            ("config", config_present, config_value),
            (
                "policy_architecture",
                architecture_present,
                architecture_value,
            ),
        ):
            if not present:
                continue
            if type(value) is not expected_type:
                problems.append(
                    f"{name} has invalid {source} type "
                    f"{type(value).__name__}; expected "
                    f"{expected_type.__name__}"
                )
            elif expected_type is float and not math.isfinite(value):
                problems.append(
                    f"{name} has non-finite {source} value {value!r}"
                )

        if config_present and architecture_present:
            if (
                type(config_value) is not type(architecture_value)
                or config_value != architecture_value
            ):
                problems.append(
                    f"{name}: config={config_value!r} "
                    f"({type(config_value).__name__}) != "
                    "policy_architecture="
                    f"{architecture_value!r} "
                    f"({type(architecture_value).__name__})"
                )
        elif architecture_present:
            resolved[name] = architecture_value

    if problems:
        raise ValueError(
            "Checkpoint config/policy_architecture typed mismatch: "
            + "; ".join(problems)
        )

    clip_field = "policy_observation_standardization_clip"
    if clip_field not in resolved:
        # Historical normalized policies used an implicit +/-5 clamp. Preserve
        # that exact replay default only when neither redundant record declares
        # the field.
        resolved[clip_field] = 5.0
    return resolved


def _split_csv_floats(raw: str, *, name: str) -> list[float]:
    try:
        values = [
            float(value.strip())
            for value in str(raw).split(",")
            if value.strip()
        ]
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated numbers.") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite numbers.")
    return values


def validate_mode_contract(
    *,
    mode: str,
    offline_test_root: Path | None,
    rollout_split: str | None,
    confirm_open_test: bool,
) -> str:
    """Return the locked split without touching any supplied data path."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"development", "final"}:
        raise ValueError("mode must be 'development' or 'final'.")
    if normalized_mode == "final" and not FINAL_TEST_LEDGER_IMPLEMENTED:
        raise RuntimeError(
            "Final mode is disabled before any test path access until an "
            "immutable protocol artifact and atomic one-shot final-test "
            "ledger are implemented."
        )
    selected_split = (
        str(rollout_split).strip().lower()
        if rollout_split is not None
        else ("val" if normalized_mode == "development" else "test")
    )
    if normalized_mode == "development":
        if offline_test_root is not None:
            raise ValueError(
                "Development mode refuses --offline-test-root; test data must "
                "remain unopened during validation and method selection."
            )
        if selected_split != "val":
            raise ValueError(
                "Development mode requires --rollout-split val and refuses the "
                "simulator test split."
            )
        if bool(confirm_open_test):
            raise ValueError(
                "--confirm-open-test is invalid in development mode."
            )
    else:
        if offline_test_root is None:
            raise ValueError(
                "Final mode requires an explicit --offline-test-root."
            )
        if selected_split != "test":
            raise ValueError(
                "Final mode requires --rollout-split test."
            )
        if not bool(confirm_open_test):
            raise ValueError(
                "Final mode requires --confirm-open-test before any test data "
                "can be opened."
            )
    return selected_split


def threshold_protocol_contract(
    *,
    mode: str,
    protocol_id: str,
    thresholds_frozen: bool,
) -> dict[str, object]:
    """Keep numerical defaults diagnostic until a frozen protocol owns them."""

    identifier = str(protocol_id).strip()
    frozen = bool(thresholds_frozen)
    if frozen and not identifier:
        raise ValueError(
            "--thresholds-frozen requires a non-empty --threshold-protocol-id."
        )
    if str(mode).strip().lower() == "final" and not frozen:
        raise ValueError(
            "Final mode requires --thresholds-frozen and a prespecified "
            "--threshold-protocol-id before test data can be opened."
        )
    return {
        "protocol_id": identifier or None,
        "thresholds_frozen": frozen,
        "qualification_eligible": bool(frozen and identifier),
        "status": (
            "prespecified_frozen"
            if frozen and identifier
            else "diagnostic_thresholds_unfrozen"
        ),
    }


def validate_threshold_values(
    *,
    minimum_validation_skill: float,
    maximum_offline_mae: float,
    minimum_action_std_ratios: list[float],
    minimum_action_correlations: list[float],
    maximum_primary_crash_rate: float,
    maximum_primary_offroad_rate: float,
    minimum_horizon_coverage: float,
    maximum_expert_relative_crash_gap: float,
    maximum_expert_relative_offroad_gap: float,
) -> None:
    scalars = {
        "minimum_validation_skill": minimum_validation_skill,
        "maximum_offline_mae": maximum_offline_mae,
        "maximum_primary_crash_rate": maximum_primary_crash_rate,
        "maximum_primary_offroad_rate": maximum_primary_offroad_rate,
        "minimum_horizon_coverage": minimum_horizon_coverage,
        "maximum_expert_relative_crash_gap": (
            maximum_expert_relative_crash_gap
        ),
        "maximum_expert_relative_offroad_gap": (
            maximum_expert_relative_offroad_gap
        ),
    }
    if not all(math.isfinite(float(value)) for value in scalars.values()):
        raise ValueError("Every qualification threshold must be finite.")
    if float(maximum_offline_mae) < 0.0:
        raise ValueError("maximum_offline_mae cannot be negative.")
    for name, value in (
        ("maximum_primary_crash_rate", maximum_primary_crash_rate),
        ("maximum_primary_offroad_rate", maximum_primary_offroad_rate),
        ("minimum_horizon_coverage", minimum_horizon_coverage),
        (
            "maximum_expert_relative_crash_gap",
            maximum_expert_relative_crash_gap,
        ),
        (
            "maximum_expert_relative_offroad_gap",
            maximum_expert_relative_offroad_gap,
        ),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if len(minimum_action_std_ratios) != 2 or any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in minimum_action_std_ratios
    ):
        raise ValueError(
            "minimum_action_std_ratios must contain two finite nonnegative "
            "values."
        )
    if len(minimum_action_correlations) != 2 or any(
        not math.isfinite(float(value))
        or not -1.0 <= float(value) <= 1.0
        for value in minimum_action_correlations
    ):
        raise ValueError(
            "minimum_action_correlations must contain two values in [-1, 1]."
        )


def threshold_discrimination_contract(
    *,
    maximum_primary_crash_rate: float,
    maximum_primary_offroad_rate: float,
    minimum_horizon_coverage: float,
) -> dict[str, object]:
    """Reject vacuous closed-loop promotion gates without inventing cutoffs.

    Values of one for a maximum failure rate, or zero for minimum coverage,
    cannot distinguish any finite rollout from complete failure. Development
    may still use them to report diagnostics, but they cannot authorize opening
    a locked test or qualify a checkpoint.
    """

    failures: list[str] = []
    if float(maximum_primary_crash_rate) >= 1.0:
        failures.append("maximum_primary_crash_rate_is_vacuous")
    if float(maximum_primary_offroad_rate) >= 1.0:
        failures.append("maximum_primary_offroad_rate_is_vacuous")
    if float(minimum_horizon_coverage) <= 0.0:
        failures.append("minimum_horizon_coverage_is_vacuous")
    return {
        "passed": not failures,
        "status": (
            "discriminating"
            if not failures
            else "diagnostic_only_vacuous_closed_loop_bounds"
        ),
        "failures": failures,
        "rule": (
            "maximum crash/offroad rates must be strictly below one and "
            "minimum horizon coverage must be strictly above zero"
        ),
    }


def _read_hash_sidecar(path: Path) -> tuple[str, str]:
    fields_in_line = path.read_text(encoding="utf-8").strip().split()
    if len(fields_in_line) != 2:
        raise ValueError(
            f"Checkpoint hash sidecar must contain SHA256 and filename: {path}"
        )
    digest, filename = fields_in_line
    return str(digest).lower(), str(filename)


def validate_checkpoint_selection_contract(
    bc_stats: dict[str, Any],
) -> dict[str, str]:
    """Accept only trainer-supported, validation-only checkpoint selection."""

    selection_rule = str(bc_stats.get("checkpoint_selection_rule"))
    if selection_rule not in SUPPORTED_VALIDATION_SELECTION_RULES:
        supported = ", ".join(sorted(SUPPORTED_VALIDATION_SELECTION_RULES))
        raise ValueError(
            "Qualification requires an immutable validation-only checkpoint "
            f"selection rule supported by the trainer ({supported}); got "
            f"{selection_rule!r}."
        )
    if str(bc_stats.get("test_evaluation_mode")) != "deferred":
        raise ValueError(
            "Qualification requires a checkpoint whose training run kept test "
            "evaluation deferred."
        )

    forbidden_populated_test_fields = [
        name
        for name in (
            "test_mse",
            "test_mae",
            "test_action_mse",
            "test_action_mae",
            "test_prediction_std",
            "test_target_std",
            "test_prediction_std_ratio",
            "test_prediction_target_correlation",
            "test_prediction_saturation_fraction",
            "test_samples",
            "test_windows",
        )
        if bc_stats.get(name) is not None
    ]
    if forbidden_populated_test_fields:
        raise ValueError(
            "Qualification rejects checkpoints containing populated test "
            "outcomes despite deferred evaluation; first="
            f"{forbidden_populated_test_fields[0]!r}."
        )
    if bc_stats.get("offline_test_evaluated") not in (None, False):
        raise ValueError(
            "Qualification rejects checkpoints evaluated on the offline test "
            "split during training."
        )
    if bc_stats.get("offline_test_status") not in (
        None,
        "pending_deferred",
    ):
        raise ValueError(
            "Qualification requires offline test status to remain "
            "pending_deferred."
        )
    for field in (
        "held_out_learning_signal_passed",
        "held_out_metric_capability_passed",
        "final_test_qualification_passed",
    ):
        if bc_stats.get(field) not in (None, False):
            raise ValueError(
                "Qualification rejects a checkpoint with a test-informed "
                f"training flag: {field}."
            )
    if bc_stats.get("final_test_qualification_status") not in (
        None,
        "pending_deferred",
    ):
        raise ValueError(
            "Qualification requires final test qualification status to remain "
            "pending_deferred."
        )

    held_out_gate = bc_stats.get("held_out_learning_signal_gate")
    if held_out_gate is not None:
        if not isinstance(held_out_gate, dict) or (
            held_out_gate.get("status") != "pending_deferred"
            or held_out_gate.get("passed") is not False
            or held_out_gate.get("actions") not in (None, [])
        ):
            raise ValueError(
                "Qualification rejects a checkpoint with populated held-out "
                "selection history."
            )

    selected_history = bc_stats.get("selected_history_row")
    if selected_history is not None:
        if not isinstance(selected_history, dict):
            raise ValueError("selected_history_row must be an object.")
        forbidden_history_keys: list[str] = []

        def collect_forbidden_keys(value: Any, *, prefix: str) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    key_text = str(key)
                    path = f"{prefix}.{key_text}" if prefix else key_text
                    if key_text.lower().startswith(("test", "held_out")):
                        forbidden_history_keys.append(path)
                    collect_forbidden_keys(nested, prefix=path)
            elif isinstance(value, (list, tuple)):
                for index, nested in enumerate(value):
                    collect_forbidden_keys(
                        nested,
                        prefix=f"{prefix}[{index}]",
                    )

        collect_forbidden_keys(selected_history, prefix="")
        forbidden_history_keys.sort()
        if forbidden_history_keys:
            raise ValueError(
                "Qualification rejects test-informed checkpoint selection "
                f"history; first={forbidden_history_keys[0]!r}."
            )
    selection_metric = bc_stats.get("checkpoint_selection_metric")
    if selection_metric is not None:
        metric_name = str(selection_metric).strip().lower()
        if "test" in metric_name or "held_out" in metric_name:
            raise ValueError(
                "Qualification rejects a test-informed checkpoint selection "
                "metric."
            )
    return {
        "checkpoint_selection_rule": selection_rule,
        "selection_data_role": "validation_only",
        "training_test_evaluation_mode": "deferred",
    }


def load_verified_checkpoint(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Verify both explicit and stored hashes before deserializing best.pt."""

    checkpoint = checkpoint_path.resolve()
    expected = str(expected_sha256).strip().lower()
    if checkpoint.name != "best.pt":
        raise ValueError(
            f"Qualification accepts only the selected best.pt, got {checkpoint.name!r}."
        )
    if not SHA256_RE.fullmatch(expected):
        raise ValueError("--checkpoint-sha256 must be 64 lowercase hex characters.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    sidecar = checkpoint.with_name("best.pt.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"Checkpoint is missing its immutable hash sidecar: {sidecar}"
        )
    actual = sha256_file(checkpoint)
    stored, stored_name = _read_hash_sidecar(sidecar)
    if stored_name != "best.pt":
        raise ValueError(
            f"Checkpoint hash sidecar names {stored_name!r}, expected 'best.pt'."
        )
    if not SHA256_RE.fullmatch(stored):
        raise ValueError(f"Invalid SHA256 in {sidecar}.")
    if len({expected, stored, actual}) != 1:
        raise RuntimeError(
            "Checkpoint hash mismatch: "
            f"expected={expected} stored={stored} actual={actual}."
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")
    if payload.get("checkpoint_kind") != "behaviour_cloning_best":
        raise ValueError(
            "Qualification requires checkpoint_kind='behaviour_cloning_best'; "
            "warm starts and intermediate artifacts are ineligible."
        )
    if not isinstance(payload.get("policy_state_dict"), dict):
        raise ValueError("Checkpoint has no policy_state_dict.")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint has no serialized PSGAILConfig.")
    architecture = payload.get("policy_architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Checkpoint has no policy_architecture.")
    config_payload = dict(payload["config"])
    reconcile_checkpoint_behaviour_config(
        config_payload,
        architecture,
    )
    bc_stats = payload.get("bc_stats")
    if not isinstance(bc_stats, dict):
        raise ValueError("Checkpoint has no bc_stats.")
    selection_contract = validate_checkpoint_selection_contract(bc_stats)
    clip_field = "policy_observation_standardization_clip"
    if clip_field in architecture:
        clip_value = float(architecture[clip_field])
        clip_provenance = "checkpoint_policy_architecture"
    elif clip_field in config_payload:
        clip_value = float(config_payload[clip_field])
        clip_provenance = "checkpoint_config"
    else:
        # Checkpoints created before this field became explicit always used the
        # hard-coded +/-5 standardized-observation clamp.  Record that replay
        # fallback instead of silently making the legacy artifact look
        # unclipped.
        clip_value = 5.0
        clip_provenance = "legacy_missing_field_replayed_as_5"
    return payload, {
        "path": str(checkpoint),
        "sha256": actual,
        "expected_sha256": expected,
        "stored_hash_sidecar": file_record(sidecar),
        "checkpoint_kind": str(payload["checkpoint_kind"]),
        **selection_contract,
        "policy_model": str(architecture["policy_model"]),
        clip_field: clip_value,
        f"{clip_field}_provenance": clip_provenance,
    }


def config_and_policy_from_checkpoint(
    payload: dict[str, Any],
    *,
    episode_root: Path,
    rollout_split: str,
    device: torch.device,
    evaluation_workers: int,
) -> tuple[PSGAILConfig, torch.nn.Module, int, int]:
    raw_config_payload = dict(payload["config"])
    architecture = dict(payload["policy_architecture"])
    config_payload = reconcile_checkpoint_behaviour_config(
        raw_config_payload,
        architecture,
    )
    known_fields = {item.name for item in fields(PSGAILConfig)}
    cfg = PSGAILConfig(
        **{
            key: value
            for key, value in config_payload.items()
            if key in known_fields
        }
    )
    required_architecture = {
        "policy_model",
        "obs_dim",
        "hidden_size",
        "action_mode",
        "continuous_action_dim",
    }
    missing = sorted(required_architecture - set(architecture))
    if missing:
        raise ValueError(
            f"Checkpoint architecture is missing required fields: {missing}"
        )
    if str(architecture["action_mode"]) != "continuous":
        raise ValueError("BC qualification supports continuous policies only.")
    if int(architecture["continuous_action_dim"]) != 2:
        raise ValueError(
            "BC qualification requires [acceleration, steering] action_dim=2."
        )
    evaluation_overrides = paper_driver_model_validation_overrides()
    evaluation_overrides.update(
        {
            "episode_root": str(episode_root.resolve()),
            "prebuilt_split": str(rollout_split),
            "action_mode": "continuous",
            "enable_collision": True,
            "control_all_vehicles": False,
            "allow_idm": True,
            "evaluation_terminate_when_all_controlled_crashed": False,
            "evaluation_num_workers": max(1, int(evaluation_workers)),
        }
    )
    cfg = replace(
        cfg,
        **evaluation_overrides,
    )
    # The overrides above are assertions about the evaluator, never mutations
    # of policy actions or network parameters.
    cfg.enable_collision = True
    cfg.evaluation_terminate_when_all_controlled_crashed = False
    cfg.validation_vehicle_mode = "single"
    cfg.validation_control_all_vehicles = False
    cfg.test_vehicle_mode = "single"
    cfg.test_control_all_vehicles = False
    policy_obs_dim = int(architecture["obs_dim"])
    critic_obs_dim = central_critic_observation_dim(policy_obs_dim, cfg)
    policy = _make_policy_from_state_dict(
        payload["policy_state_dict"],
        cfg,
        policy_obs_dim,
        critic_obs_dim,
        device,
    )
    return cfg, policy, policy_obs_dim, int(architecture["continuous_action_dim"])


def _all_trajectory_ids(transitions: Any) -> list[str]:
    identifiers = sorted(
        {str(value) for value in np.asarray(transitions.trajectory_ids, dtype=object)}
    )
    if not identifiers:
        raise RuntimeError("Offline evaluation root contains no trajectories.")
    return identifiers


def evaluate_offline_root(
    policy: torch.nn.Module,
    *,
    cfg: PSGAILConfig,
    root: Path,
    split: str,
    device: torch.device,
    max_samples: int,
    seed: int,
    expected_obs_dim: int,
    expected_action_dim: int,
    checkpoint_contract: dict[str, Any],
    checkpoint_validation_baseline_mse: float | None,
    minimum_validation_skill: float,
    maximum_mae: float,
    minimum_action_std_ratios: list[float],
    minimum_action_correlations: list[float],
) -> dict[str, object]:
    transitions = load_expert_transition_data(
        str(root.resolve()),
        max_samples=int(max_samples),
        seed=int(seed),
        trajectory_frame="relative",
    )
    if int(transitions.policy_observations.shape[1]) != int(expected_obs_dim):
        raise ValueError(
            "Offline observation dimension differs from the checkpoint: "
            f"{transitions.policy_observations.shape[1]} != {expected_obs_dim}."
        )
    if int(transitions.actions_continuous_env.shape[1]) != int(
        expected_action_dim
    ):
        raise ValueError(
            "Offline action dimension differs from the checkpoint: "
            f"{transitions.actions_continuous_env.shape[1]} != "
            f"{expected_action_dim}."
        )
    current_contract = validate_training_data_contracts(
        transitions.metadata,
        lidar_cells=int(cfg.cells),
        maximum_range=float(cfg.maximum_range),
        require_explicit=True,
    )
    reference_action = checkpoint_contract.get("continuous_action")
    reference_observation = checkpoint_contract.get("policy_observation")
    if not isinstance(reference_action, dict) or not isinstance(
        reference_observation, dict
    ):
        raise ValueError(
            "Checkpoint is missing its explicit action/observation contracts."
        )
    assert_compatible_action_contracts(
        reference_action,
        current_contract["continuous_action"],
    )
    assert_compatible_observation_contracts(
        reference_observation,
        current_contract["policy_observation"],
    )
    windows = build_sequence_windows(
        transitions,
        _all_trajectory_ids(transitions),
        sequence_length=int(cfg.transformer_recurrent_sequence_length),
        context_length=int(cfg.transformer_memory_context_length),
        warmup_mode=str(cfg.recurrent_bc_warmup_mode),
    )
    metrics = evaluate_recurrent_bc(
        policy,
        transitions,
        windows,
        device=device,
        micro_batch_sequences=int(
            cfg.transformer_recurrent_micro_batch_sequences
        ),
    )
    action_gate = action_learning_gate(
        split=str(split),
        prediction_std_ratios=metrics["prediction_std_ratio"],
        prediction_target_correlations=metrics[
            "prediction_target_correlation"
        ],
        action_indices=list(range(int(expected_action_dim))),
        minimum_std_ratios=minimum_action_std_ratios,
        minimum_correlations=minimum_action_correlations,
    )
    skill: float | None = None
    root_optimal_constant_baseline_mse: float | None = None
    if str(split) == "validation":
        target_actions = np.asarray(
            transitions.actions_continuous_env,
            dtype=np.float64,
        )
        root_optimal_constant_baseline_mse = float(
            np.mean(
                np.square(
                    target_actions
                    - target_actions.mean(axis=0, keepdims=True)
                )
            )
        )
        if (
            not math.isfinite(root_optimal_constant_baseline_mse)
            or root_optimal_constant_baseline_mse <= 0.0
        ):
            raise ValueError(
                "Offline validation root has no finite positive variance for "
                "the conservative optimal-constant skill baseline."
            )
        skill = 1.0 - float(metrics["mse"]) / float(
            root_optimal_constant_baseline_mse
        )
    checks = {
        "finite_mse": math.isfinite(float(metrics["mse"])),
        "finite_mae": math.isfinite(float(metrics["mae"])),
        "maximum_mae": bool(
            math.isfinite(float(metrics["mae"]))
            and float(metrics["mae"]) <= float(maximum_mae)
        ),
        "all_action_learning_gates": bool(action_gate["passed"]),
    }
    if str(split) == "validation":
        checks["minimum_validation_skill"] = bool(
            skill is not None
            and math.isfinite(skill)
            and skill >= float(minimum_validation_skill)
        )
    return {
        "split": str(split),
        "root": str(root.resolve()),
        "metrics": metrics,
        "validation_skill": skill,
        "validation_baseline_mse_source": (
            "supplied_validation_root_optimal_constant_conservative"
            if str(split) == "validation"
            else None
        ),
        "validation_baseline_mse": root_optimal_constant_baseline_mse,
        "checkpoint_training_validation_baseline_mse_diagnostic": (
            float(checkpoint_validation_baseline_mse)
            if checkpoint_validation_baseline_mse is not None
            and math.isfinite(float(checkpoint_validation_baseline_mse))
            else None
        ),
        "thresholds": {
            "minimum_validation_skill": (
                float(minimum_validation_skill)
                if str(split) == "validation"
                else None
            ),
            "maximum_mae": float(maximum_mae),
            "minimum_action_std_ratios": minimum_action_std_ratios,
            "minimum_action_correlations": minimum_action_correlations,
        },
        "action_learning_gate": action_gate,
        "checks": checks,
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "passed": all(checks.values()),
        "data_contract": current_contract,
    }


def evaluate_same_scenario_expert_replay(
    cfg: PSGAILConfig,
    *,
    split: str,
    prefix: str,
    scenarios: list[tuple[str, int]] | None = None,
    episode_names: list[str] | None = None,
) -> dict[str, object]:
    """Run the environment's tracker on the policy evaluator's exact cases."""

    if (scenarios is None) == (episode_names is None):
        raise ValueError(
            "Provide exactly one of scenarios (single ego) or episode_names "
            "(all-vehicle stress)."
        )
    all_vehicle = episode_names is not None
    cases: Iterable[tuple[str, int | None]] = (
        ((str(name), None) for name in episode_names or [])
        if all_vehicle
        else ((str(name), int(vehicle_id)) for name, vehicle_id in scenarios or [])
    )
    horizons = _parse_evaluation_horizons(cfg)
    max_steps = min(
        max(1, int(max(horizons) * int(cfg.policy_frequency))),
        max(1, int(cfg.max_episode_steps)),
    )
    evaluated_cases = 0
    vehicle_episodes = 0
    crashed_vehicle_episodes = 0
    offroad_vehicle_episodes = 0
    collision_agent_steps = 0
    offroad_agent_steps = 0
    controlled_agent_steps = 0
    fully_covered_vehicle_episodes = 0
    records: list[dict[str, object]] = []

    for case_index, (episode_name, vehicle_id) in enumerate(cases):
        env, env_cached = _get_matched_eval_env(
            cfg,
            split=str(split),
            episode_name=str(episode_name),
            vehicle_id=vehicle_id,
            all_vehicle=all_vehicle,
        )
        try:
            _obs, _info = env.reset(
                # Exact reset seed used by both the single-ego and all-vehicle
                # learned-policy evaluator for the corresponding indexed case.
                seed=_evaluation_protocol_seed(cfg)
                + 100_000
                + int(case_index)
            )
            env_config = env.unwrapped.config
            if not bool(env_config.get("expert_test_mode")):
                raise RuntimeError(
                    "Expert baseline lost expert_test_mode after reset."
                )
            if bool(env_config.get("disable_controlled_vehicle_collisions")):
                raise RuntimeError(
                    "Expert baseline has controlled-vehicle collisions disabled."
                )
            controlled = list(
                getattr(env.unwrapped, "controlled_vehicles", ()) or ()
            )
            initial_ids = {
                int(getattr(vehicle, "vehicle_ID", -1))
                for vehicle in controlled
            }
            initial_ids.discard(-1)
            if not initial_ids:
                raise RuntimeError(
                    f"Expert replay spawned no vehicles for {episode_name}."
                )
            if vehicle_id is not None and int(vehicle_id) not in initial_ids:
                raise RuntimeError(
                    f"Selected expert vehicle {vehicle_id} did not spawn in "
                    f"{episode_name}; got {sorted(initial_ids)}."
                )
            crashed_ids: set[int] = set()
            offroad_ids: set[int] = set()
            final_live_ids = set(initial_ids)
            completed_steps = 0
            for _step in range(max_steps):
                live = list(
                    getattr(env.unwrapped, "controlled_vehicles", ()) or ()
                )
                dummy_action = tuple(
                    np.zeros(
                        (int(cfg.continuous_action_dim),),
                        dtype=np.float32,
                    )
                    for _vehicle in live
                )
                _obs, _reward, terminated, truncated, info = env.step(
                    dummy_action
                )
                info_ids = [
                    int(value)
                    for value in list(
                        info.get("controlled_vehicle_ids", []) or []
                    )
                ]
                crash_flags = list(
                    info.get("controlled_vehicle_crashes", []) or []
                )
                offroad_flags = list(
                    info.get("controlled_vehicle_offroad", []) or []
                )
                if len(info_ids) < max(len(crash_flags), len(offroad_flags)):
                    raise RuntimeError(
                        "Expert replay collision/off-road flags have no matching "
                        "controlled vehicle IDs."
                    )
                for flag_index, flag in enumerate(crash_flags):
                    if bool(flag):
                        crashed_ids.add(info_ids[flag_index])
                for flag_index, flag in enumerate(offroad_flags):
                    if bool(flag):
                        offroad_ids.add(info_ids[flag_index])
                collision_agent_steps += sum(bool(flag) for flag in crash_flags)
                offroad_agent_steps += sum(bool(flag) for flag in offroad_flags)
                controlled_agent_steps += max(
                    len(info_ids),
                    len(crash_flags),
                    len(offroad_flags),
                )
                final_live_ids = set(info_ids)
                completed_steps += 1
                if _evaluation_should_stop(
                    cfg,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                ):
                    break
            evaluated_cases += 1
            vehicle_episodes += len(initial_ids)
            crashed_vehicle_episodes += len(crashed_ids & initial_ids)
            offroad_vehicle_episodes += len(offroad_ids & initial_ids)
            if completed_steps >= max_steps:
                fully_covered_vehicle_episodes += len(
                    final_live_ids & initial_ids
                )
            records.append(
                {
                    "episode_name": str(episode_name),
                    "selected_vehicle_id": vehicle_id,
                    "reset_seed": (
                        _evaluation_protocol_seed(cfg)
                        + 100_000
                        + int(case_index)
                    ),
                    "initial_vehicle_ids": sorted(initial_ids),
                    "completed_steps": completed_steps,
                    "crashed_vehicle_ids": sorted(crashed_ids & initial_ids),
                    "offroad_vehicle_ids": sorted(offroad_ids & initial_ids),
                }
            )
        finally:
            if not env_cached:
                env.close()

    if evaluated_cases <= 0 or vehicle_episodes <= 0:
        raise RuntimeError("Expert replay evaluated no cases or vehicles.")
    return {
        "framework": "same_scenario_expert_replay_collision_baseline_v2",
        "scope": (
            "simulator tracker/action replay baseline; not independent "
            "surveyed-road geometry evidence"
        ),
        "split": str(split),
        "vehicle_mode": (
            "all_successfully_spawned" if all_vehicle else "single_ego"
        ),
        "collision_physics_enabled": True,
        "collision_termination_enabled": False,
        "action_source": "environment_expert_tracker_replay",
        "attempted_cases": len(records),
        "evaluated_cases": evaluated_cases,
        "vehicle_episodes": vehicle_episodes,
        "crashed_vehicle_episodes": crashed_vehicle_episodes,
        "offroad_vehicle_episodes": offroad_vehicle_episodes,
        f"{prefix}/vehicle_crash_rate": float(
            crashed_vehicle_episodes / vehicle_episodes
        ),
        f"{prefix}/vehicle_offroad_rate": float(
            offroad_vehicle_episodes / vehicle_episodes
        ),
        f"{prefix}/collision_agent_step_rate": float(
            collision_agent_steps / max(1, controlled_agent_steps)
        ),
        f"{prefix}/offroad_agent_step_rate": float(
            offroad_agent_steps / max(1, controlled_agent_steps)
        ),
        f"{prefix}/horizon_coverage_{int(max(horizons))}s": float(
            fully_covered_vehicle_episodes / vehicle_episodes
        ),
        "cases": records,
    }


def expert_relative_gate(
    policy_metrics: dict[str, object],
    expert_metrics: dict[str, object],
    *,
    policy_prefix: str,
    expert_prefix: str,
    maximum_crash_rate_gap: float,
    maximum_offroad_rate_gap: float,
    minimum_paired_scenarios: int = 24,
    paired_bootstrap_replicates: int = 5_000,
    upper_confidence_quantile: float = 0.95,
    require_paired_cases: bool = True,
) -> dict[str, object]:
    thresholds = {
        "maximum_vehicle_crash_rate_gap": float(maximum_crash_rate_gap),
        "maximum_vehicle_offroad_rate_gap": float(maximum_offroad_rate_gap),
        "minimum_paired_scenarios": int(minimum_paired_scenarios),
        "paired_bootstrap_replicates": int(paired_bootstrap_replicates),
        "upper_confidence_quantile": float(upper_confidence_quantile),
    }
    if not all(
        0.0 <= value <= 1.0
        for value in (
            float(maximum_crash_rate_gap),
            float(maximum_offroad_rate_gap),
        )
    ):
        raise ValueError("Expert-relative gap thresholds must be in [0, 1].")
    if int(minimum_paired_scenarios) < 2:
        raise ValueError("At least two paired scenarios are required.")
    if int(paired_bootstrap_replicates) < 100:
        raise ValueError("At least 100 paired bootstrap replicates are required.")
    if not 0.5 < float(upper_confidence_quantile) < 1.0:
        raise ValueError("Upper confidence quantile must lie in (0.5, 1).")
    policy_crash = float(
        policy_metrics[f"{policy_prefix}/vehicle_crash_rate"]
    )
    expert_crash = float(
        expert_metrics[f"{expert_prefix}/vehicle_crash_rate"]
    )
    policy_offroad = float(
        policy_metrics[f"{policy_prefix}/vehicle_offroad_rate"]
    )
    expert_offroad = float(
        expert_metrics[f"{expert_prefix}/vehicle_offroad_rate"]
    )
    observed = {
        "policy_vehicle_crash_rate": policy_crash,
        "expert_vehicle_crash_rate": expert_crash,
        "vehicle_crash_rate_gap": policy_crash - expert_crash,
        "policy_vehicle_offroad_rate": policy_offroad,
        "expert_vehicle_offroad_rate": expert_offroad,
        "vehicle_offroad_rate_gap": policy_offroad - expert_offroad,
        "policy_vehicle_episodes": float(
            policy_metrics.get(
                f"{policy_prefix}/vehicle_episodes",
                float("nan"),
            )
        ),
        "expert_vehicle_episodes": float(
            expert_metrics.get(
                "vehicle_episodes",
                expert_metrics.get("evaluated_episodes", float("nan")),
            )
        ),
    }
    finite = all(math.isfinite(value) for value in observed.values())

    def indexed_cases(
        metrics: dict[str, object],
    ) -> dict[tuple[str, int, int], dict[str, object]] | None:
        raw = metrics.get("episodes", metrics.get("cases"))
        if not isinstance(raw, list):
            return None
        indexed: dict[tuple[str, int, int], dict[str, object]] = {}
        for row in raw:
            if not isinstance(row, dict):
                raise ValueError("Paired outcome records must be objects.")
            vehicle_id = row.get(
                "ego_vehicle_id",
                row.get("selected_vehicle_id"),
            )
            if (
                not str(row.get("episode_name", "")).strip()
                or vehicle_id is None
                or row.get("reset_seed") is None
            ):
                raise ValueError(
                    "Paired outcome record lacks episode, ego, or reset seed."
                )
            identity = (
                str(row["episode_name"]),
                int(vehicle_id),
                int(row["reset_seed"]),
            )
            if identity in indexed:
                raise ValueError(
                    f"Duplicate paired outcome identity: {identity!r}."
                )
            indexed[identity] = row
        return indexed

    policy_cases = indexed_cases(policy_metrics)
    expert_cases = indexed_cases(expert_metrics)
    pair_records_present = (
        policy_cases is not None and expert_cases is not None
    )
    identity_match = bool(
        pair_records_present
        and set(policy_cases or ()) == set(expert_cases or ())
    )
    paired_count = (
        len(policy_cases or {})
        if identity_match
        else 0
    )
    paired_outcomes: dict[str, object] = {
        "status": (
            "paired"
            if identity_match
            else (
                "identity_mismatch"
                if pair_records_present
                else "case_records_missing"
            )
        ),
        "scenario_count": int(paired_count),
        "identity_sha256": None,
        "crash_rate_gap": None,
        "crash_rate_gap_upper": None,
        "offroad_rate_gap": None,
        "offroad_rate_gap_upper": None,
        "policy_crash_rate_from_cases": None,
        "expert_crash_rate_from_cases": None,
        "policy_offroad_rate_from_cases": None,
        "expert_offroad_rate_from_cases": None,
        "both_initial_collision_count": None,
        "expert_initial_only_count": None,
        "policy_initial_only_count": None,
    }
    if identity_match and policy_cases is not None and expert_cases is not None:
        identities = sorted(policy_cases)
        policy_crash_values = np.asarray(
            [
                policy_cases[key].get("first_collision_step") is not None
                for key in identities
            ],
            dtype=float,
        )
        expert_crash_values = np.asarray(
            [
                expert_cases[key].get("first_collision_step") is not None
                for key in identities
            ],
            dtype=float,
        )
        policy_offroad_values = np.asarray(
            [
                policy_cases[key].get("first_offroad_step") is not None
                for key in identities
            ],
            dtype=float,
        )
        expert_offroad_values = np.asarray(
            [
                expert_cases[key].get("first_offroad_step") is not None
                for key in identities
            ],
            dtype=float,
        )
        crash_differences = policy_crash_values - expert_crash_values
        offroad_differences = policy_offroad_values - expert_offroad_values
        identity_sha256 = canonical_sha256(identities)
        rng = np.random.default_rng(
            int(identity_sha256[:16], 16)
        )
        selected = rng.integers(
            0,
            len(identities),
            size=(
                int(paired_bootstrap_replicates),
                len(identities),
            ),
        )
        crash_bootstrap = crash_differences[selected].mean(axis=1)
        offroad_bootstrap = offroad_differences[selected].mean(axis=1)
        policy_initial = np.asarray(
            [
                policy_cases[key].get("first_collision_step") == 0
                for key in identities
            ],
            dtype=bool,
        )
        expert_initial = np.asarray(
            [
                expert_cases[key].get("first_collision_step") == 0
                for key in identities
            ],
            dtype=bool,
        )
        paired_outcomes.update(
            {
                "identity_sha256": identity_sha256,
                "crash_rate_gap": float(np.mean(crash_differences)),
                "crash_rate_gap_upper": float(
                    np.quantile(
                        crash_bootstrap,
                        float(upper_confidence_quantile),
                    )
                ),
                "offroad_rate_gap": float(np.mean(offroad_differences)),
                "offroad_rate_gap_upper": float(
                    np.quantile(
                        offroad_bootstrap,
                        float(upper_confidence_quantile),
                    )
                ),
                "both_initial_collision_count": int(
                    np.sum(policy_initial & expert_initial)
                ),
                "expert_initial_only_count": int(
                    np.sum(~policy_initial & expert_initial)
                ),
                "policy_initial_only_count": int(
                    np.sum(policy_initial & ~expert_initial)
                ),
                "policy_crash_rate_from_cases": float(
                    np.mean(policy_crash_values)
                ),
                "expert_crash_rate_from_cases": float(
                    np.mean(expert_crash_values)
                ),
                "policy_offroad_rate_from_cases": float(
                    np.mean(policy_offroad_values)
                ),
                "expert_offroad_rate_from_cases": float(
                    np.mean(expert_offroad_values)
                ),
            }
        )
    checks = {
        "finite_rates_and_gaps": finite,
        "same_positive_vehicle_denominator": bool(
            finite
            and observed["policy_vehicle_episodes"] > 0.0
            and observed["policy_vehicle_episodes"]
            == observed["expert_vehicle_episodes"]
        ),
        "maximum_vehicle_crash_rate_gap": bool(
            finite
            and observed["vehicle_crash_rate_gap"]
            <= thresholds["maximum_vehicle_crash_rate_gap"]
        ),
        "maximum_vehicle_offroad_rate_gap": bool(
            finite
            and observed["vehicle_offroad_rate_gap"]
            <= thresholds["maximum_vehicle_offroad_rate_gap"]
        ),
    }
    if require_paired_cases:
        checks.update(
            {
                "paired_case_records_present": pair_records_present,
                "paired_case_identities_match": identity_match,
                "minimum_paired_scenario_count": bool(
                    paired_count >= int(minimum_paired_scenarios)
                ),
                "paired_count_matches_aggregate_denominators": bool(
                    identity_match
                    and float(paired_count)
                    == observed["policy_vehicle_episodes"]
                    == observed["expert_vehicle_episodes"]
                ),
                "paired_case_rates_match_aggregates": bool(
                    identity_match
                    and np.isclose(
                        float(
                            paired_outcomes[
                                "policy_crash_rate_from_cases"
                            ]
                        ),
                        policy_crash,
                    )
                    and np.isclose(
                        float(
                            paired_outcomes[
                                "expert_crash_rate_from_cases"
                            ]
                        ),
                        expert_crash,
                    )
                    and np.isclose(
                        float(
                            paired_outcomes[
                                "policy_offroad_rate_from_cases"
                            ]
                        ),
                        policy_offroad,
                    )
                    and np.isclose(
                        float(
                            paired_outcomes[
                                "expert_offroad_rate_from_cases"
                            ]
                        ),
                        expert_offroad,
                    )
                ),
                "paired_crash_gap_upper": bool(
                    paired_outcomes["crash_rate_gap_upper"] is not None
                    and float(paired_outcomes["crash_rate_gap_upper"])
                    <= float(maximum_crash_rate_gap)
                ),
                "paired_offroad_gap_upper": bool(
                    paired_outcomes["offroad_rate_gap_upper"] is not None
                    and float(paired_outcomes["offroad_rate_gap_upper"])
                    <= float(maximum_offroad_rate_gap)
                ),
            }
        )
    return {
        "comparison": "policy_minus_same_scenario_expert_replay",
        "thresholds": thresholds,
        "observed": observed,
        "paired_outcomes": paired_outcomes,
        "checks": checks,
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "passed": all(checks.values()),
    }


def rollout_data_records(
    *,
    episode_root: Path,
    scene: str,
    split: str,
) -> dict[str, object]:
    prebuilt = episode_root.resolve() / str(scene) / "prebuilt"
    files = {
        "vehicle_ids": file_record(prebuilt / f"veh_ids_{split}.npy"),
        "trajectories": file_record(prebuilt / f"trajectory_{split}.npy"),
    }
    return {
        "episode_root": str(episode_root.resolve()),
        "scene": str(scene),
        "split": str(split),
        "prebuilt_files": files,
        "pair_sha256": canonical_sha256(
            {
                name: record["sha256"]
                for name, record in sorted(files.items())
            }
        ),
    }


def source_records() -> dict[str, dict[str, object]]:
    paths = {
        "qualification_sidecar": Path(__file__).resolve(),
        "recurrent_bc": ROOT / "scripts_gail/ps_gail/recurrent_bc.py",
        "environment_builder": ROOT / "scripts_gail/ps_gail/envs.py",
        "evaluation": ROOT
        / "scripts_gail/ps_gail/training/evaluation.py",
        "policy": ROOT / "scripts_gail/ps_gail/training/policy.py",
        "validation": ROOT / "scripts_gail/ps_gail/validation.py",
        "contracts": ROOT / "scripts_gail/ps_gail/contracts.py",
        "models": ROOT / "scripts_gail/ps_gail/models.py",
        "environment": ROOT / "highway_env/envs/ngsim_env.py",
        "action": ROOT / "highway_env/envs/common/action.py",
        "vehicle_dynamics": ROOT / "highway_env/vehicle/kinematics.py",
        "road_generator": ROOT
        / "highway_env/ngsim_utils/road/gen_road.py",
    }
    return {name: file_record(path) for name, path in paths.items()}


def _device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(value)


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing temporary output: {temporary}"
        )
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_receipt(path: Path, payload: dict[str, Any]) -> dict[str, str]:
    receipt = path.resolve()
    hash_sidecar = receipt.with_name(f"{receipt.name}.sha256")
    if receipt.exists() or hash_sidecar.exists():
        raise FileExistsError(
            f"Refusing to overwrite receipt or hash sidecar: {receipt}"
        )
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    _write_bytes_exclusive(receipt, encoded)
    try:
        _write_bytes_exclusive(
            hash_sidecar,
            f"{digest}  {receipt.name}\n".encode("utf-8"),
        )
    except Exception:
        # The JSON remains a valid immutable receipt even if writing the
        # convenience hash sidecar fails; never modify its contents.
        raise
    return {
        "receipt": str(receipt),
        "receipt_sha256": digest,
        "hash_sidecar": str(hash_sidecar),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--offline-validation-root", type=Path, required=True)
    parser.add_argument("--offline-test-root", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["development", "final"],
        default="development",
    )
    parser.add_argument(
        "--threshold-protocol-id",
        default="",
        help=(
            "Immutable protocol/registry identifier that prespecified every "
            "threshold below. Required with --thresholds-frozen."
        ),
    )
    parser.add_argument(
        "--thresholds-frozen",
        action="store_true",
        help=(
            "Assert thresholds were frozen before these outcomes were opened. "
            "Without this flag, development runs remain diagnostic."
        ),
    )
    parser.add_argument("--confirm-open-test", action="store_true")
    parser.add_argument("--rollout-split", choices=["val", "test"])
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--all-vehicle-stress-episodes", type=int, default=0)
    parser.add_argument("--max-offline-samples", type=int, default=100_000)
    parser.add_argument("--offline-seed", type=int, default=20260716)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-workers", type=int, default=1)
    parser.add_argument("--minimum-validation-skill", type=float, default=0.10)
    parser.add_argument("--maximum-offline-mae", type=float, default=0.35)
    parser.add_argument(
        "--minimum-action-std-ratios",
        default="0.25,0.10",
    )
    parser.add_argument(
        "--minimum-action-correlations",
        default="0.50,0.20",
    )
    parser.add_argument("--maximum-primary-crash-rate", type=float, default=1.0)
    parser.add_argument("--maximum-primary-offroad-rate", type=float, default=1.0)
    parser.add_argument("--minimum-horizon-coverage", type=float, default=0.0)
    parser.add_argument(
        "--maximum-expert-relative-crash-gap",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--maximum-expert-relative-offroad-gap",
        type=float,
        default=0.05,
    )
    parser.add_argument("--minimum-paired-scenarios", type=int, default=24)
    parser.add_argument(
        "--paired-bootstrap-replicates",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--paired-upper-confidence-quantile",
        type=float,
        default=0.95,
    )
    return parser.parse_args(argv)


def evaluate_closed_loop_split(
    policy: torch.nn.Module,
    *,
    cfg: PSGAILConfig,
    device: torch.device,
    split: str,
    episodes: int,
    all_vehicle_stress_episodes: int,
    maximum_primary_crash_rate: float,
    maximum_primary_offroad_rate: float,
    minimum_horizon_coverage: float,
    maximum_expert_relative_crash_gap: float,
    maximum_expert_relative_offroad_gap: float,
    minimum_paired_scenarios: int = 24,
    paired_bootstrap_replicates: int = 5_000,
    paired_upper_confidence_quantile: float = 0.95,
) -> dict[str, Any]:
    """Evaluate one split without selecting or touching any other split."""

    normalized_split = str(split).strip().lower()
    if normalized_split not in {"val", "test"}:
        raise ValueError(f"Unsupported closed-loop split: {split!r}.")
    primary_prefix = (
        "validation_primary"
        if normalized_split == "val"
        else "test_primary"
    )
    expert_prefix = f"{primary_prefix}_expert"
    setattr(cfg, f"{primary_prefix}_vehicle_mode", "single")
    setattr(cfg, f"{primary_prefix}_control_all_vehicles", False)
    clear_evaluation_worker_caches()
    scenarios = _evaluation_scenarios(
        cfg,
        split=normalized_split,
        episodes=int(episodes),
    )
    indexed_scenarios = [
        (index, name, vehicle_id)
        for index, (name, vehicle_id) in enumerate(scenarios)
    ]
    with evaluation_thread_context(cfg):
        policy_metrics = _evaluate_policy_matched_trajectories_impl(
            policy,
            cfg,
            device,
            split=normalized_split,
            episodes=len(scenarios),
            prefix=primary_prefix,
            scenarios=indexed_scenarios,
            include_raw=False,
            include_cases=True,
        )
    clear_evaluation_worker_caches()
    expert_metrics = evaluate_expert_replay_matched_single_vehicle_floor(
        cfg,
        split=normalized_split,
        episodes=len(scenarios),
        prefix=expert_prefix,
        scenarios=indexed_scenarios,
    )
    clear_evaluation_worker_caches()
    absolute_quality = closed_loop_policy_quality(
        policy_metrics,
        prefix=primary_prefix,
        max_vehicle_crash_rate=float(maximum_primary_crash_rate),
        max_vehicle_offroad_rate=float(maximum_primary_offroad_rate),
        score_horizon_seconds=max(_parse_evaluation_horizons(cfg)),
        min_horizon_coverage=float(minimum_horizon_coverage),
    )
    expert_relative_quality = expert_relative_gate(
        policy_metrics,
        expert_metrics,
        policy_prefix=primary_prefix,
        expert_prefix=expert_prefix,
        maximum_crash_rate_gap=float(
            maximum_expert_relative_crash_gap
        ),
        maximum_offroad_rate_gap=float(
            maximum_expert_relative_offroad_gap
        ),
        minimum_paired_scenarios=int(minimum_paired_scenarios),
        paired_bootstrap_replicates=int(paired_bootstrap_replicates),
        upper_confidence_quantile=float(
            paired_upper_confidence_quantile
        ),
        require_paired_cases=True,
    )
    expert_absolute_quality = closed_loop_policy_quality(
        expert_metrics,
        prefix=expert_prefix,
        max_vehicle_crash_rate=float(maximum_primary_crash_rate),
        max_vehicle_offroad_rate=float(maximum_primary_offroad_rate),
        score_horizon_seconds=max(_parse_evaluation_horizons(cfg)),
        min_horizon_coverage=float(minimum_horizon_coverage),
    )

    stress: dict[str, object] = {
        "status": "not_requested",
        "role": "diagnostic_non_promotion_gate",
    }
    if int(all_vehicle_stress_episodes) > 0:
        stress_prefix = f"{primary_prefix}_all_vehicle_stress"
        stress_expert_prefix = f"{stress_prefix}_expert"
        setattr(cfg, f"{stress_prefix}_vehicle_mode", "all")
        setattr(cfg, f"{stress_prefix}_control_all_vehicles", True)
        episode_names = _evaluation_episode_names(
            cfg,
            split=normalized_split,
            episodes=int(all_vehicle_stress_episodes),
        )
        indexed_names = [
            (index, name) for index, name in enumerate(episode_names)
        ]
        with evaluation_thread_context(cfg):
            stress_policy_metrics = (
                _evaluate_policy_matched_trajectories_impl(
                    policy,
                    cfg,
                    device,
                    split=normalized_split,
                    episodes=len(episode_names),
                    prefix=stress_prefix,
                    episode_names=indexed_names,
                    include_raw=False,
                )
            )
        clear_evaluation_worker_caches()
        stress_expert_metrics = evaluate_same_scenario_expert_replay(
            cfg,
            split=normalized_split,
            prefix=stress_expert_prefix,
            episode_names=episode_names,
        )
        clear_evaluation_worker_caches()
        stress = {
            "status": "evaluated",
            "role": "diagnostic_non_promotion_gate",
            "episode_names": episode_names,
            "episode_names_sha256": canonical_sha256(episode_names),
            "policy_metrics": stress_policy_metrics,
            "expert_replay_metrics": stress_expert_metrics,
            "expert_relative": expert_relative_gate(
                stress_policy_metrics,
                stress_expert_metrics,
                policy_prefix=stress_prefix,
                expert_prefix=stress_expert_prefix,
                maximum_crash_rate_gap=float(
                    maximum_expert_relative_crash_gap
                ),
                maximum_offroad_rate_gap=float(
                    maximum_expert_relative_offroad_gap
                ),
                minimum_paired_scenarios=int(minimum_paired_scenarios),
                paired_bootstrap_replicates=int(
                    paired_bootstrap_replicates
                ),
                upper_confidence_quantile=float(
                    paired_upper_confidence_quantile
                ),
                require_paired_cases=False,
            ),
        }

    passed = bool(
        absolute_quality["passed"]
        and expert_absolute_quality["passed"]
        and expert_relative_quality["passed"]
    )
    return {
        "split": normalized_split,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "scenario_sha256": canonical_sha256(scenarios),
        "policy_metrics": policy_metrics,
        "expert_replay_metrics": expert_metrics,
        "absolute_quality": absolute_quality,
        "expert_absolute_quality": expert_absolute_quality,
        "expert_relative_quality": expert_relative_quality,
        "passed": passed,
        "all_vehicle_stress": stress,
    }


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    split = validate_mode_contract(
        mode=str(args.mode),
        offline_test_root=args.offline_test_root,
        rollout_split=args.rollout_split,
        confirm_open_test=bool(args.confirm_open_test),
    )
    threshold_protocol = threshold_protocol_contract(
        mode=str(args.mode),
        protocol_id=str(args.threshold_protocol_id),
        thresholds_frozen=bool(args.thresholds_frozen),
    )
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive.")
    if int(args.all_vehicle_stress_episodes) < 0:
        raise ValueError("--all-vehicle-stress-episodes cannot be negative.")
    if int(args.max_offline_samples) <= 0:
        raise ValueError("--max-offline-samples must be positive.")
    if int(args.evaluation_workers) <= 0:
        raise ValueError("--evaluation-workers must be positive.")
    receipt_path = args.receipt.resolve()
    receipt_hash_path = receipt_path.with_name(f"{receipt_path.name}.sha256")
    receipt_temp_path = receipt_path.with_name(f".{receipt_path.name}.tmp")
    if any(
        path.exists()
        for path in (receipt_path, receipt_hash_path, receipt_temp_path)
    ):
        raise FileExistsError(
            "Refusing to overwrite/reuse an existing receipt, hash, or "
            f"temporary path: {receipt_path}"
        )

    minimum_std = _split_csv_floats(
        args.minimum_action_std_ratios,
        name="--minimum-action-std-ratios",
    )
    minimum_corr = _split_csv_floats(
        args.minimum_action_correlations,
        name="--minimum-action-correlations",
    )
    validate_threshold_values(
        minimum_validation_skill=float(args.minimum_validation_skill),
        maximum_offline_mae=float(args.maximum_offline_mae),
        minimum_action_std_ratios=minimum_std,
        minimum_action_correlations=minimum_corr,
        maximum_primary_crash_rate=float(args.maximum_primary_crash_rate),
        maximum_primary_offroad_rate=float(
            args.maximum_primary_offroad_rate
        ),
        minimum_horizon_coverage=float(args.minimum_horizon_coverage),
        maximum_expert_relative_crash_gap=float(
            args.maximum_expert_relative_crash_gap
        ),
        maximum_expert_relative_offroad_gap=float(
            args.maximum_expert_relative_offroad_gap
        ),
    )
    minimum_paired_scenarios = int(
        getattr(args, "minimum_paired_scenarios", 24)
    )
    paired_bootstrap_replicates = int(
        getattr(args, "paired_bootstrap_replicates", 5_000)
    )
    paired_upper_confidence_quantile = float(
        getattr(args, "paired_upper_confidence_quantile", 0.95)
    )
    if minimum_paired_scenarios < 2:
        raise ValueError("--minimum-paired-scenarios must be at least two.")
    if paired_bootstrap_replicates < 100:
        raise ValueError(
            "--paired-bootstrap-replicates must be at least 100."
        )
    if not 0.5 < paired_upper_confidence_quantile < 1.0:
        raise ValueError(
            "--paired-upper-confidence-quantile must lie in (0.5, 1)."
        )
    threshold_discrimination = threshold_discrimination_contract(
        maximum_primary_crash_rate=float(args.maximum_primary_crash_rate),
        maximum_primary_offroad_rate=float(
            args.maximum_primary_offroad_rate
        ),
        minimum_horizon_coverage=float(args.minimum_horizon_coverage),
    )
    threshold_declaration_eligible = bool(
        threshold_protocol["qualification_eligible"]
        and threshold_discrimination["passed"]
    )

    mode = str(args.mode).strip().lower()
    offline_validation_root = args.offline_validation_root.resolve()
    if not offline_validation_root.is_dir():
        raise FileNotFoundError(
            f"Offline validation root not found: {offline_validation_root}"
        )
    episode_root = args.episode_root.resolve()
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Episode root not found: {episode_root}")

    # Validation and checkpoint eligibility are established before the final
    # branch resolves, stats, hashes, or otherwise inspects the supplied test
    # path. The same opened inputs are recomputed afterward.
    source_before = source_records()
    offline_validation_data_before = tree_hash_record(
        offline_validation_root
    )
    checkpoint_payload, checkpoint_record = load_verified_checkpoint(
        args.checkpoint,
        expected_sha256=str(args.checkpoint_sha256),
    )
    checkpoint_hash_before = str(checkpoint_record["sha256"])
    adjacent_summary = args.checkpoint.resolve().with_name("summary.json")
    adjacent_summary_before = (
        file_record(adjacent_summary) if adjacent_summary.is_file() else None
    )
    device = _device(str(args.device))
    cfg, policy, obs_dim, action_dim = config_and_policy_from_checkpoint(
        checkpoint_payload,
        episode_root=episode_root,
        rollout_split="val",
        device=device,
        evaluation_workers=int(args.evaluation_workers),
    )
    supplied_validation_identity = root_manifest_identity(
        offline_validation_root,
        expected_split="val",
    )
    split_identity_contract = validate_split_identity_contract(
        checkpoint_payload,
        supplied_validation=supplied_validation_identity,
        supplied_test=None,
    )
    validation_rollout_data_before = rollout_data_records(
        episode_root=episode_root,
        scene=str(cfg.scene),
        split="val",
    )

    checkpoint_contract = {
        "continuous_action": checkpoint_payload.get(
            "policy_output_action_contract"
        ),
        "policy_observation": checkpoint_payload.get(
            "policy_observation_contract"
        ),
    }
    bc_stats = dict(checkpoint_payload["bc_stats"])
    offline: dict[str, object] = {
        "validation": evaluate_offline_root(
            policy,
            cfg=cfg,
            root=offline_validation_root,
            split="validation",
            device=device,
            max_samples=int(args.max_offline_samples),
            seed=int(args.offline_seed),
            expected_obs_dim=obs_dim,
            expected_action_dim=action_dim,
            checkpoint_contract=checkpoint_contract,
            checkpoint_validation_baseline_mse=bc_stats.get(
                "validation_baseline_mse"
            ),
            minimum_validation_skill=float(
                args.minimum_validation_skill
            ),
            maximum_mae=float(args.maximum_offline_mae),
            minimum_action_std_ratios=minimum_std,
            minimum_action_correlations=minimum_corr,
        )
    }
    validation_offline_passed = bool(offline["validation"]["passed"])
    validation_closed_loop = evaluate_closed_loop_split(
        policy,
        cfg=cfg,
        device=device,
        split="val",
        episodes=int(args.episodes),
        all_vehicle_stress_episodes=(
            int(args.all_vehicle_stress_episodes)
            if mode == "development"
            else 0
        ),
        maximum_primary_crash_rate=float(
            args.maximum_primary_crash_rate
        ),
        maximum_primary_offroad_rate=float(
            args.maximum_primary_offroad_rate
        ),
        minimum_horizon_coverage=float(args.minimum_horizon_coverage),
        maximum_expert_relative_crash_gap=float(
            args.maximum_expert_relative_crash_gap
        ),
        maximum_expert_relative_offroad_gap=float(
            args.maximum_expert_relative_offroad_gap
        ),
        minimum_paired_scenarios=minimum_paired_scenarios,
        paired_bootstrap_replicates=paired_bootstrap_replicates,
        paired_upper_confidence_quantile=(
            paired_upper_confidence_quantile
        ),
    )
    validation_primary_passed = bool(validation_closed_loop["passed"])
    validation_identity_canonical = bool(
        split_identity_contract["promotion_eligible"]
    )
    validation_sidecar_passed = bool(
        threshold_declaration_eligible
        and validation_identity_canonical
        and validation_offline_passed
        and validation_primary_passed
    )
    test_access_preconditions = {
        "checkpoint_hash_kind_and_selection_verified": True,
        "explicit_final_confirmation": bool(
            mode == "final" and args.confirm_open_test
        ),
        "threshold_declaration_eligible": bool(
            threshold_declaration_eligible
        ),
        "threshold_values_discriminating": bool(
            threshold_discrimination["passed"]
        ),
        "validation_identity_canonical": (
            validation_identity_canonical
        ),
        "validation_offline_passed": validation_offline_passed,
        "validation_primary_closed_loop_passed": (
            validation_primary_passed
        ),
        "validation_sidecar_passed": validation_sidecar_passed,
    }
    test_access_preconditions["all_passed"] = bool(
        test_access_preconditions[
            "checkpoint_hash_kind_and_selection_verified"
        ]
        and test_access_preconditions["explicit_final_confirmation"]
        and test_access_preconditions["threshold_declaration_eligible"]
        and test_access_preconditions["threshold_values_discriminating"]
        and test_access_preconditions["validation_identity_canonical"]
        and test_access_preconditions["validation_offline_passed"]
        and test_access_preconditions[
            "validation_primary_closed_loop_passed"
        ]
        and test_access_preconditions["validation_sidecar_passed"]
    )

    # No expression below this gate may resolve, stat, hash, load, or select a
    # test path/split unless the immutable checkpoint and full validation
    # eligibility have already passed.
    if mode == "final" and not test_access_preconditions["all_passed"]:
        raise RuntimeError(
            "Final test remains unopened because checkpoint/validation "
            "eligibility failed."
        )

    offline_test_root: Path | None = None
    offline_test_data_before: dict[str, object] | None = None
    test_rollout_data_before: dict[str, object] | None = None
    test_cfg: PSGAILConfig | None = None
    test_closed_loop: dict[str, Any] | None = None
    test_data_opened = False
    if mode == "final":
        # validate_mode_contract already requires this object, but it has not
        # been inspected until this explicitly confirmed, validation-eligible
        # branch.
        assert args.offline_test_root is not None
        offline_test_root = args.offline_test_root.resolve()
        if not offline_test_root.is_dir():
            raise FileNotFoundError(
                f"Offline test root not found: {offline_test_root}"
            )
        supplied_test_identity = root_manifest_identity(
            offline_test_root,
            expected_split="test",
        )
        test_data_opened = True
        split_identity_contract = validate_split_identity_contract(
            checkpoint_payload,
            supplied_validation=supplied_validation_identity,
            supplied_test=supplied_test_identity,
        )
        offline_test_data_before = tree_hash_record(offline_test_root)
        test_cfg = replace(cfg, prebuilt_split="test")
        test_rollout_data_before = rollout_data_records(
            episode_root=episode_root,
            scene=str(test_cfg.scene),
            split="test",
        )
        offline["test"] = evaluate_offline_root(
            policy,
            cfg=test_cfg,
            root=offline_test_root,
            split="test",
            device=device,
            max_samples=int(args.max_offline_samples),
            seed=int(args.offline_seed) + 1,
            expected_obs_dim=obs_dim,
            expected_action_dim=action_dim,
            checkpoint_contract=checkpoint_contract,
            checkpoint_validation_baseline_mse=None,
            minimum_validation_skill=float(
                args.minimum_validation_skill
            ),
            maximum_mae=float(args.maximum_offline_mae),
            minimum_action_std_ratios=minimum_std,
            minimum_action_correlations=minimum_corr,
        )
        test_closed_loop = evaluate_closed_loop_split(
            policy,
            cfg=test_cfg,
            device=device,
            split="test",
            episodes=int(args.episodes),
            all_vehicle_stress_episodes=int(
                args.all_vehicle_stress_episodes
            ),
            maximum_primary_crash_rate=float(
                args.maximum_primary_crash_rate
            ),
            maximum_primary_offroad_rate=float(
                args.maximum_primary_offroad_rate
            ),
            minimum_horizon_coverage=float(
                args.minimum_horizon_coverage
            ),
            maximum_expert_relative_crash_gap=float(
                args.maximum_expert_relative_crash_gap
            ),
            maximum_expert_relative_offroad_gap=float(
                args.maximum_expert_relative_offroad_gap
            ),
            minimum_paired_scenarios=minimum_paired_scenarios,
            paired_bootstrap_replicates=paired_bootstrap_replicates,
            paired_upper_confidence_quantile=(
                paired_upper_confidence_quantile
            ),
        )

    test_offline_passed = bool(
        offline.get("test", {}).get("passed", False)
    )
    test_primary_passed = bool(
        test_closed_loop is not None
        and test_closed_loop["passed"]
    )
    final_test_candidate_checks_passed = bool(
        mode == "final"
        and threshold_declaration_eligible
        and validation_sidecar_passed
        and test_offline_passed
        and test_primary_passed
    )
    # This remains a defensive scaffold if final mode is enabled only after an
    # immutable protocol artifact and atomic one-shot ledger are implemented.
    final_test_qualification_passed = False
    primary_closed_loop = (
        test_closed_loop
        if mode == "final"
        else validation_closed_loop
    )
    assert primary_closed_loop is not None
    primary_cfg = test_cfg if test_cfg is not None else cfg
    stress = primary_closed_loop["all_vehicle_stress"]

    checkpoint_hash_after = sha256_file(args.checkpoint.resolve())
    adjacent_summary_after = (
        file_record(adjacent_summary) if adjacent_summary.is_file() else None
    )
    checkpoint_unchanged = checkpoint_hash_after == checkpoint_hash_before
    summary_unchanged = adjacent_summary_before == adjacent_summary_after
    if not checkpoint_unchanged or not summary_unchanged:
        raise RuntimeError(
            "Qualification mutated the checkpoint or adjacent training summary."
        )

    source_after = source_records()
    offline_validation_data_after = tree_hash_record(
        offline_validation_root
    )
    offline_test_data_after = (
        tree_hash_record(offline_test_root)
        if test_data_opened and offline_test_root is not None
        else None
    )
    validation_rollout_data_after = rollout_data_records(
        episode_root=episode_root,
        scene=str(cfg.scene),
        split="val",
    )
    test_rollout_data_after = (
        rollout_data_records(
            episode_root=episode_root,
            scene=str(primary_cfg.scene),
            split="test",
        )
        if test_data_opened
        else None
    )
    immutable_inputs = {
        "source": source_before == source_after,
        "offline_validation": (
            offline_validation_data_before
            == offline_validation_data_after
        ),
        "offline_test": (
            offline_test_data_before == offline_test_data_after
        ),
        "rollout_validation": (
            validation_rollout_data_before
            == validation_rollout_data_after
        ),
        "rollout_test": (
            test_rollout_data_before == test_rollout_data_after
        ),
    }
    if not all(immutable_inputs.values()):
        changed = sorted(
            name for name, unchanged in immutable_inputs.items() if not unchanged
        )
        raise RuntimeError(
            "Qualification inputs changed while evaluation was running: "
            f"{changed}."
        )
    source = {
        "records": source_before,
        "verified_unchanged_after_evaluation": True,
    }
    data_records: dict[str, object] = {
        "offline_validation": {
            **offline_validation_data_before,
            "verified_unchanged_after_evaluation": True,
        },
        "offline_test": (
            {
                **offline_test_data_before,
                "access_status": (
                    "opened_in_this_receipt_after_explicit_confirmation"
                ),
                "verified_unchanged_after_evaluation": True,
            }
            if offline_test_data_before is not None
            else {"status": "locked_not_opened"}
        ),
        "rollout_validation": {
            **validation_rollout_data_before,
            "verified_unchanged_after_evaluation": True,
        },
        "rollout_test": (
            {
                **test_rollout_data_before,
                "access_status": (
                    "opened_in_this_receipt_after_explicit_confirmation"
                ),
                "verified_unchanged_after_evaluation": True,
            }
            if test_rollout_data_before is not None
            else {"status": "locked_not_opened"}
        ),
        "split_identity_contract": split_identity_contract,
        "primary_scenarios": [
            {"episode_name": name, "vehicle_id": int(vehicle_id)}
            for name, vehicle_id in primary_closed_loop["scenarios"]
        ],
        "primary_scenarios_sha256": primary_closed_loop[
            "scenario_sha256"
        ],
    }
    payload = {
        "schema_version": 2,
        "framework": QUALIFICATION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_framework": PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
        "mode": mode,
        "rollout_split": split,
        "test_data_opened": test_data_opened,
        "test_access_status": (
            "opened_in_this_receipt_after_explicit_confirmation"
            if test_data_opened
            else "locked_not_opened"
        ),
        "status": (
            (
                "final_test_candidate_checks_passed_publication_blocked"
                if final_test_candidate_checks_passed
                else "final_test_candidate_checks_failed_publication_blocked"
            )
            if mode == "final"
            else (
                (
                    (
                        "validation_qualification_passed_test_locked"
                        if validation_sidecar_passed
                        else (
                            "diagnostic_incomplete_canonical_identity_test_locked"
                            if not validation_identity_canonical
                            else "validation_qualification_failed_test_locked"
                        )
                    )
                )
                if threshold_declaration_eligible
                else (
                    "diagnostic_vacuous_thresholds_test_locked"
                    if threshold_protocol["qualification_eligible"]
                    else "diagnostic_thresholds_unfrozen_test_locked"
                )
            )
        ),
        "checkpoint": {
            **checkpoint_record,
            "sha256_after_evaluation": checkpoint_hash_after,
            "unchanged": checkpoint_unchanged,
        },
        "adjacent_training_summary": {
            "present": adjacent_summary_before is not None,
            "before": adjacent_summary_before,
            "after": adjacent_summary_after,
            "unchanged": summary_unchanged,
        },
        "source": source,
        "data": data_records,
        "runtime_contract": {
            "device": str(device),
            "scene": str(primary_cfg.scene),
            "policy_model": str(primary_cfg.policy_model),
            "policy_observation_standardization_clip": float(
                primary_cfg.policy_observation_standardization_clip
            ),
            "policy_observation_standardization_clipping_enabled": bool(
                float(
                    primary_cfg.policy_observation_standardization_clip
                )
                > 0.0
            ),
            "action_mode": "continuous",
            "normalized_action_order": [
                "acceleration_norm",
                "steering_norm",
            ],
            "deterministic_policy_actions": True,
            "evaluator_pre_env_action_clamping": False,
            "environment_continuous_action_clip_configured": False,
            "environment_continuous_action_clip_role": (
                "disabled after exact evaluator [-1, 1] validation so an "
                "out-of-contract command fails instead of being silently "
                "changed; the native policy action is passed unchanged to "
                "env.step"
            ),
            "policy_action_substitution": False,
            "policy_parameters_modified": False,
            "policy_network_output_passed_unchanged_to_env_step": True,
            "simulator_post_action_dynamics": (
                "vehicle dynamics may enforce physical speed bounds or "
                "post-collision behavior after env.step receives the "
                "unchanged network action"
            ),
            "collision_physics_enabled": True,
            "collision_termination_enabled": False,
            "background_replay_to_idm_allowed": bool(
                primary_cfg.allow_idm
            ),
            "background_replay_to_idm_handover_metrics_reported": True,
            "primary_vehicle_mode": "single_ego",
            "all_vehicle_stress_role": "diagnostic_non_promotion_gate",
            "horizons_seconds": _parse_evaluation_horizons(primary_cfg),
            "evaluation_scenario_seed": _evaluation_protocol_seed(
                primary_cfg
            ),
        },
        "threshold_protocol": {
            **threshold_protocol,
            "qualification_eligible": threshold_declaration_eligible,
            "declaration_eligible_before_value_check": bool(
                threshold_protocol["qualification_eligible"]
            ),
            "discrimination": threshold_discrimination,
            "immutable_protocol_artifact_verified": False,
            "publication_qualification_eligible": False,
            "values": {
                "minimum_validation_skill": float(
                    args.minimum_validation_skill
                ),
                "maximum_offline_mae": float(args.maximum_offline_mae),
                "minimum_action_std_ratios": minimum_std,
                "minimum_action_correlations": minimum_corr,
                "maximum_primary_crash_rate": float(
                    args.maximum_primary_crash_rate
                ),
                "maximum_primary_offroad_rate": float(
                    args.maximum_primary_offroad_rate
                ),
                "minimum_horizon_coverage": float(
                    args.minimum_horizon_coverage
                ),
                "maximum_expert_relative_crash_gap": float(
                    args.maximum_expert_relative_crash_gap
                ),
                "maximum_expert_relative_offroad_gap": float(
                    args.maximum_expert_relative_offroad_gap
                ),
                "minimum_paired_scenarios": minimum_paired_scenarios,
                "paired_bootstrap_replicates": (
                    paired_bootstrap_replicates
                ),
                "paired_upper_confidence_quantile": (
                    paired_upper_confidence_quantile
                ),
            },
            "warning": (
                (
                    "Metrics are reported, but no pass or promotion is valid "
                    "because threshold provenance is unfrozen."
                )
                if not threshold_protocol["qualification_eligible"]
                else (
                    "Metrics are reported, but no pass or test access is valid "
                    "because the declared closed-loop thresholds are vacuous."
                    if not threshold_discrimination["passed"]
                    else (
                        "The protocol identifier and --thresholds-frozen flag "
                        "are a declaration, not a verified immutable protocol "
                        "artifact; publication qualification remains false."
                        if mode == "final"
                        else None
                    )
                )
            ),
        },
        "offline": offline,
        "validation_closed_loop": {
            key: value
            for key, value in validation_closed_loop.items()
            if key not in {"scenarios", "all_vehicle_stress"}
        },
        "test_closed_loop": (
            {
                key: value
                for key, value in test_closed_loop.items()
                if key not in {"scenarios", "all_vehicle_stress"}
            }
            if test_closed_loop is not None
            else {"status": "locked_not_opened"}
        ),
        "primary_closed_loop": {
            key: value
            for key, value in primary_closed_loop.items()
            if key not in {"scenarios", "all_vehicle_stress"}
        },
        "all_vehicle_stress": stress,
        "qualification": {
            "test_access_preconditions": test_access_preconditions,
            "validation_offline_passed": validation_offline_passed,
            "validation_primary_closed_loop_passed": (
                validation_primary_passed
            ),
            "test_offline_passed": test_offline_passed,
            "test_primary_closed_loop_passed": test_primary_passed,
            "primary_closed_loop_passed": bool(
                primary_closed_loop["passed"]
            ),
            "validation_sidecar_passed": validation_sidecar_passed,
            "final_test_candidate_checks_passed": (
                final_test_candidate_checks_passed
            ),
            "final_test_qualification_passed": (
                final_test_qualification_passed
            ),
            "policy_realism_qualified": False,
            "threshold_protocol_eligible": bool(
                threshold_declaration_eligible
            ),
            "atomic_final_test_ledger_verified": False,
            "immutable_protocol_artifact_verified": False,
            "publication_qualification_allowed": False,
            "locked_test_status": (
                "opened_in_this_receipt_after_explicit_confirmation"
                if test_data_opened
                else "pending_not_opened"
            ),
            "all_vehicle_stress_used_for_promotion": False,
        },
        "limitations": [
            "Expert replay is a simulator/action baseline, not independent "
            "surveyed-road geometry validation.",
            "A development receipt cannot qualify policy realism because the "
            "locked test remains unopened.",
            "A final-mode receipt remains non-publication-qualifying until a "
            "real atomic final-test ledger and a verified immutable protocol "
            "artifact are implemented.",
            "All-vehicle stress is reported separately and is not a substitute "
            "for the prespecified single-ego primary estimand.",
        ],
    }
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_qualification(args)
    record = write_receipt(args.receipt, payload)
    print(json.dumps({**record, "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
