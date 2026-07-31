#!/usr/bin/env python3
"""Train a recipe-declared BC matrix while loading each data split once."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from scripts_gail.archive_bc_checkpoints import archive_bc_checkpoints
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.recurrent_bc import (
    PreparedRecurrentBCData,
    prepare_recurrent_bc_data_from_explicit_splits,
)
from scripts_gail.train_recurrent_bc_policy import run_training, write_json


DOMAINS = ("us", "japanese")
CAUSAL_MOTIF_ARCHITECTURE_CONTRACT_ID = (
    "shared_dense_temporal_recurrent_transformer_depth_only_v2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_active_causal_motif_recipe(
    recipe: dict[str, Any],
) -> Path | None:
    """Revalidate the active architecture and audited data root at runtime."""
    if (
        recipe.get("architecture_contract_id")
        != CAUSAL_MOTIF_ARCHITECTURE_CONTRACT_ID
    ):
        return None
    architecture = dict(recipe.get("architecture") or {})
    expected = {
        "policy_model": "recurrent_transformer",
        "depths": [2, 3],
        "hidden_size": 256,
        "transformer_heads": 4,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "policy_observation_standardization_clip": 0.0,
        "transformer_observation_tokenization": "dense_temporal",
        "policy_head_init_std": 0.01,
        "memory_tokens": 1,
        "memory_context_length": 32,
        "only_permitted_architecture_axis": "depth",
    }
    for field, value in expected.items():
        if architecture.get(field) != value:
            raise ValueError(
                f"Active causal-motif architecture mismatch for {field!r}."
            )
    seeds = architecture.get("policy_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Active causal-motif policy seeds are invalid.")

    data = dict(recipe.get("data") or {})
    collection_id = str(data.get("collection_id") or "")
    collection = dict(data.get("training_collection") or {})
    canonical_root = Path(
        str(collection.get("canonical_root") or "")
    ).resolve()
    audit_artifact = dict(collection.get("audit_artifact") or {})
    audit_path = Path(str(audit_artifact.get("path") or "")).resolve()
    expected_audit_sha256 = str(audit_artifact.get("sha256") or "")
    contract_results_artifact = dict(
        collection.get("contract_results_artifact") or {}
    )
    contract_results_path = Path(
        str(contract_results_artifact.get("path") or "")
    ).resolve()
    expected_contract_results_sha256 = str(
        contract_results_artifact.get("sha256") or ""
    )
    if not collection_id:
        raise ValueError("Active causal-motif recipe has no collection_id.")
    if not canonical_root.is_dir():
        raise ValueError("Active causal-motif canonical collection root is absent.")
    if (
        not audit_path.is_file()
        or len(expected_audit_sha256) != 64
        or sha256_file(audit_path) != expected_audit_sha256
        or not contract_results_path.is_file()
        or len(expected_contract_results_sha256) != 64
        or sha256_file(contract_results_path)
        != expected_contract_results_sha256
    ):
        raise ValueError(
            "Active causal-motif collection audit/results are absent or changed."
        )
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    with contract_results_path.open(encoding="utf-8") as handle:
        contract_results = json.load(handle)
    if (
        Path(str(audit.get("collection_root") or "")).resolve()
        != canonical_root
        or audit.get("test_data_status") != "not_opened"
        or int(audit.get("total_row_count", 0)) <= 0
        or int(audit.get("total_episode_count", 0)) <= 0
        or Path(
            str(contract_results.get("canonical_collection_view") or "")
        ).resolve()
        != canonical_root
        or contract_results.get("test_data_status") != "sealed_not_opened"
        or int(contract_results.get("rows_modified", -1)) != 0
        or int(contract_results.get("rows_excluded", -1)) != 0
    ):
        raise ValueError(
            "Active causal-motif collection audit does not qualify its "
            "canonical root."
        )
    return canonical_root


def validate_active_causal_motif_expert_paths(
    recipe: dict[str, Any],
    expert_paths: dict[str, dict[str, Path]],
) -> None:
    """Require CLI split roots to be the exact roots locked in the recipe."""
    canonical_root = validate_active_causal_motif_recipe(recipe)
    if canonical_root is None:
        return
    expected = {
        domain: {
            "train": (canonical_root / domain / "train").resolve(),
            "validation": (canonical_root / domain / "val").resolve(),
        }
        for domain in DOMAINS
    }
    actual = {
        domain: {
            split: path.resolve()
            for split, path in expert_paths[domain].items()
        }
        for domain in DOMAINS
    }
    if actual != expected:
        raise ValueError(
            "Active causal-motif CLI expert roots differ from the exact "
            "audited training collection."
        )


def read_locked_recipe(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        recipe = json.load(handle)
    schema_version = int(recipe.get("schema_version", -1))
    if schema_version not in {1, 2} or recipe.get("status") != "locked":
        raise ValueError(
            "BC recipe must be a locked supported schema (1 or 2): "
            f"{path}"
        )
    for section in ("architecture", "optimization", "data", "gates", "evaluation"):
        if not isinstance(recipe.get(section), dict):
            raise ValueError(f"Locked BC recipe is missing section {section!r}: {path}")
    recipe_matrix_axes(recipe)
    validate_active_causal_motif_recipe(recipe)
    if str(recipe["evaluation"].get("test_evaluation_mode")) != "deferred":
        raise ValueError(
            "Locked matrix recipe must keep test_evaluation_mode deferred."
        )
    return recipe


def recipe_matrix_axes(
    recipe: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return explicitly locked model depths and policy seeds."""
    architecture = recipe["architecture"]
    depths = tuple(int(value) for value in architecture.get("depths", ()))
    seeds = tuple(
        int(value) for value in architecture.get("policy_seeds", ())
    )
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("Locked BC recipe requires unique non-empty depths.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(
            "Locked BC recipe requires unique non-empty policy_seeds."
        )
    if any(depth < 1 for depth in depths) or any(seed < 0 for seed in seeds):
        raise ValueError("BC depths must be positive and seeds non-negative.")
    return depths, seeds


def matrix_cells(
    *,
    depths: tuple[int, ...],
    seeds: tuple[int, ...],
    limit: int | None = None,
) -> list[tuple[str, int, int]]:
    cells = [
        (domain, depth, seed)
        for domain in DOMAINS
        for depth in depths
        for seed in seeds
    ]
    selected_limit = len(cells) if limit is None or int(limit) == 0 else int(limit)
    if not 1 <= selected_limit <= len(cells):
        raise ValueError(f"model_limit must be between 1 and {len(cells)}; got {limit}")
    return cells[:selected_limit]


def priority_cells_first(
    *,
    depths: tuple[int, ...],
    seeds: tuple[int, ...],
    limit: int | None = None,
) -> list[tuple[str, int, int]]:
    priority_depth = max(depths)
    # Run a previously locally exercised seed for both domains first so early
    # logs are informative. This changes ordering only: no metric or rollout
    # result is allowed to terminate the remaining matrix.
    priority_seed = seeds[0]
    priority = [
        ("us", priority_depth, priority_seed),
        ("japanese", priority_depth, priority_seed),
    ]
    ordered = priority + [
        cell
        for cell in matrix_cells(
            depths=depths,
            seeds=seeds,
        )
        if cell not in priority
    ]
    selected_limit = (
        len(ordered) if limit is None or int(limit) == 0 else int(limit)
    )
    if not 1 <= selected_limit <= len(ordered):
        raise ValueError(f"model_limit must be between 1 and {len(ordered)}; got {limit}")
    return ordered[:selected_limit]


def cell_args(
    recipe: dict[str, Any],
    *,
    domain: str,
    depth: int,
    seed: int,
    expert_train_data: Path,
    expert_validation_data: Path,
    episode_root: Path,
    out_dir: Path,
    device: str,
) -> argparse.Namespace:
    architecture = recipe["architecture"]
    optimization = recipe["optimization"]
    data = recipe["data"]
    gates = recipe["gates"]
    evaluation = recipe["evaluation"]
    scene = "us-101" if domain == "us" else "japanese"
    test_evaluation_mode = str(
        evaluation.get("test_evaluation_mode", "deferred")
    )
    if test_evaluation_mode != "deferred":
        raise ValueError(
            "The matrix is a validation-selection workflow and requires "
            "test_evaluation_mode='deferred'. Run locked test evaluation "
            "separately after candidate selection."
        )
    return argparse.Namespace(
        policy_model=str(architecture.get("policy_model", "recurrent_transformer")),
        expert_data=str(expert_train_data),
        expert_validation_data=str(expert_validation_data),
        expert_test_data="",
        out_dir=str(out_dir),
        domain=domain,
        scene=scene,
        episode_root=str(episode_root),
        prebuilt_split="train",
        seed=int(seed),
        data_seed=int(data["data_seed"]),
        split_seed=int(data["split_seed"]),
        max_expert_samples=int(data["max_expert_samples"]),
        max_validation_samples=int(
            data.get("max_validation_samples", data["max_expert_samples"])
        ),
        max_test_samples=0,
        epochs=int(optimization["epochs"]),
        checkpoint_purpose="policy",
        max_warmup_epochs=5,
        min_warmup_relative_improvement=0.01,
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        train_fraction=float(data["train_fraction"]),
        validation_fraction=float(data["validation_fraction"]),
        early_stopping_patience=int(optimization["early_stopping_patience"]),
        early_stopping_min_epochs=int(
            optimization.get("early_stopping_min_epochs", 0)
        ),
        early_stopping_min_delta_relative=float(
            optimization.get("early_stopping_min_delta_relative", 0.001)
        ),
        min_validation_skill=float(gates["min_validation_skill"]),
        max_validation_mae=float(gates["max_validation_mae"]),
        max_grad_norm=float(optimization["max_grad_norm"]),
        action_loss_weights=list(
            optimization.get("action_loss_weights", [1.0, 1.0])
        ),
        action_loss_weighting=str(
            optimization.get("action_loss_weighting", "fixed")
        ),
        correlation_loss_weight=float(
            optimization.get("correlation_loss_weight", 0.0)
        ),
        variance_loss_weight=float(
            optimization.get("variance_loss_weight", 0.0)
        ),
        training_min_prediction_std_ratios=list(
            optimization.get("training_min_prediction_std_ratios", [])
        ),
        mirror_augmentation_probability=float(
            optimization.get("mirror_augmentation_probability", 0.0)
        ),
        checkpoint_selection_rule=str(
            optimization.get(
                "checkpoint_selection_rule",
                "validation_loss",
            )
        ),
        test_evaluation_mode=test_evaluation_mode,
        learning_action_index=int(gates["learning_action_index"]),
        min_learning_action_std_ratio=float(gates["min_learning_action_std_ratio"]),
        min_learning_action_correlation=float(gates["min_learning_action_correlation"]),
        learning_action_indices=list(
            gates.get("learning_action_indices", [])
        ),
        min_learning_action_std_ratios=list(
            gates.get("min_learning_action_std_ratios", [])
        ),
        min_learning_action_correlations=list(
            gates.get("min_learning_action_correlations", [])
        ),
        require_explicit_data_contracts=bool(
            data.get("require_explicit_data_contracts", False)
        ),
        device=device,
        hidden_size=int(architecture["hidden_size"]),
        transformer_layers=int(depth),
        transformer_heads=int(architecture["transformer_heads"]),
        transformer_dropout=float(architecture["transformer_dropout"]),
        transformer_norm_first=bool(architecture["transformer_norm_first"]),
        transformer_observation_normalization=bool(
            architecture.get("transformer_observation_normalization", False)
        ),
        policy_observation_standardization_clip=float(
            architecture.get(
                "policy_observation_standardization_clip",
                5.0,
            )
        ),
        transformer_observation_tokenization=str(
            architecture.get("transformer_observation_tokenization", "semantic")
        ),
        policy_head_init_std=float(architecture["policy_head_init_std"]),
        memory_tokens=int(architecture["memory_tokens"]),
        memory_context_length=int(architecture["memory_context_length"]),
        sequence_length=int(optimization["sequence_length"]),
        recurrent_warmup_mode=str(
            optimization.get("recurrent_warmup_mode", "full_prefix")
        ),
        sequences_per_batch=int(optimization["sequences_per_batch"]),
        micro_batch_sequences=int(optimization["micro_batch_sequences"]),
        render_video=False,
        video_steps=200,
        evaluation_episodes=int(evaluation["episodes"]),
        evaluation_scenario_seed=int(
            evaluation.get("scenario_seed", 20260716)
        ),
        validation_evaluation_episodes=int(
            evaluation.get("validation_episodes", evaluation["episodes"])
        ),
        evaluation_split=(
            "val"
            if test_evaluation_mode == "deferred"
            else str(evaluation.get("split", "test"))
        ),
        evaluation_enable_collision=bool(
            evaluation.get("enable_collision", False)
        ),
        matched_evaluation=bool(
            evaluation.get("gail_aligned_matched_trajectories", False)
        ),
        evaluation_vehicle_mode=str(
            evaluation.get("vehicle_mode", "single")
        ),
        min_rollout_steps=int(evaluation["min_rollout_steps"]),
        max_crash_fraction=float(evaluation["max_crash_fraction"]),
        max_offroad_fraction=float(evaluation["max_offroad_fraction"]),
        expert_replay_qualification_required=bool(
            dict(evaluation.get("expert_replay_qualification") or {}).get(
                "required_for_policy_realism",
                False,
            )
        ),
        maximum_expert_replay_crash_rate_gap=float(
            dict(evaluation.get("expert_replay_qualification") or {}).get(
                "maximum_vehicle_crash_rate_gap",
                0.0,
            )
        ),
        maximum_expert_replay_offroad_rate_gap=float(
            dict(evaluation.get("expert_replay_qualification") or {}).get(
                "maximum_vehicle_offroad_rate_gap",
                0.0,
            )
        ),
        capability_failure_mode="report",
    )


def load_and_prepare_domains(
    expert_paths: dict[str, dict[str, Path]],
    recipe: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, PreparedRecurrentBCData],
    dict[str, dict[str, int]],
]:
    """Load each independent train/validation source once; never open test."""
    data = recipe["data"]
    optimization = recipe["optimization"]
    architecture = recipe["architecture"]
    if str(
        recipe.get("evaluation", {}).get("test_evaluation_mode", "deferred")
    ) != "deferred":
        raise ValueError(
            "Matrix loading requires deferred test evaluation; test sources "
            "must not be opened during method, epoch, or seed selection."
        )
    transitions: dict[str, dict[str, Any]] = {}
    prepared: dict[str, PreparedRecurrentBCData] = {}
    loader_calls: dict[str, dict[str, int]] = {
        domain: {"train": 0, "validation": 0, "test": 0}
        for domain in DOMAINS
    }
    for domain in DOMAINS:
        domain_paths = expert_paths[domain]
        if set(domain_paths) != {"train", "validation"}:
            raise ValueError(
                f"{domain} matrix sources must contain exactly train and "
                f"validation, never test; got {sorted(domain_paths)}."
            )
        train_path = domain_paths["train"].resolve()
        validation_path = domain_paths["validation"].resolve()
        if train_path == validation_path:
            raise ValueError(
                f"{domain} train and validation sources must be distinct: "
                f"{train_path}"
            )
        train_transitions = load_expert_transition_data(
            str(train_path),
            max_samples=int(data["max_expert_samples"]),
            seed=int(data["data_seed"]),
            trajectory_frame="relative",
        )
        loader_calls[domain]["train"] += 1
        validation_transitions = load_expert_transition_data(
            str(validation_path),
            max_samples=int(
                data.get("max_validation_samples", data["max_expert_samples"])
            ),
            seed=int(data["data_seed"]) + 1,
            trajectory_frame="relative",
        )
        loader_calls[domain]["validation"] += 1
        transitions[domain] = {
            "train": train_transitions,
            "validation": validation_transitions,
        }
        prepared[domain] = prepare_recurrent_bc_data_from_explicit_splits(
            transitions[domain],
            sequence_length=int(optimization["sequence_length"]),
            context_length=int(architecture["memory_context_length"]),
            warmup_mode=str(
                optimization.get("recurrent_warmup_mode", "full_prefix")
            ),
        )
    return transitions, prepared, loader_calls


def select_validation_candidate_checkpoints(
    cell_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select validation candidates without claiming locked-test promotion."""
    selections: list[dict[str, Any]] = []
    groups = sorted(
        {
            (str(record["domain"]), int(record["transformer_layers"]))
            for record in cell_records
        }
    )
    for domain, depth in groups:
        candidates = [
            record
            for record in cell_records
            if str(record["domain"]) == domain
            and int(record["transformer_layers"]) == depth
            and bool(record.get("training_artifact_complete"))
            and record.get("paper_validation_score") is not None
            and math.isfinite(float(record["paper_validation_score"]))
        ]
        primary = [
            record
            for record in candidates
            if bool(record.get("validation_selection_eligible"))
        ]
        if not primary:
            if candidates:
                fallback = max(
                    candidates,
                    key=lambda record: (
                        float(record["paper_validation_score"]),
                        -int(record["seed"]),
                    ),
                )
                fallback["development_fallback_selected"] = True
                fallback["development_fallback_reason"] = (
                    "no_validation_eligible_candidate"
                )
            continue
        selected = max(
            primary,
            key=lambda record: (
                float(record["paper_validation_score"]),
                -int(record["seed"]),
            ),
        )
        selected["validation_candidate_selected"] = True
        selected["validation_candidate_selection_tier"] = (
            "validation_capable_pending_locked_test"
        )
        selections.append(
            {
                "domain": domain,
                "transformer_layers": depth,
                "seed": int(selected["seed"]),
                "relative_path": str(selected["relative_path"]),
                "checkpoint": selected.get("checkpoint"),
                "checkpoint_sha256": selected.get("checkpoint_sha256"),
                "paper_validation_cost": selected.get("paper_validation_cost"),
                "paper_validation_score": selected.get("paper_validation_score"),
                "selection_tier": selected[
                    "validation_candidate_selection_tier"
                ],
            }
        )
    return selections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--us-train-expert", type=Path, required=True)
    parser.add_argument("--us-validation-expert", type=Path, required=True)
    parser.add_argument("--japanese-train-expert", type=Path, required=True)
    parser.add_argument("--japanese-validation-expert", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--checkpoint-archive-root", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--model-limit",
        type=int,
        default=0,
        help="Optional development cap; zero runs every recipe-locked cell.",
    )
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument(
        "--priority-cells-first",
        action="store_true",
        help=(
            "Run the locally exercised depth/seed first in each domain. This "
            "only changes order and never gates the rest of the matrix."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe_path = args.recipe.resolve()
    recipe = read_locked_recipe(recipe_path)
    policy_root = args.policy_root.resolve()
    if policy_root.exists():
        raise FileExistsError(f"Refusing to reuse matrix policy root: {policy_root}")
    policy_root.parent.mkdir(parents=True, exist_ok=True)

    expert_paths = {
        "us": {
            "train": args.us_train_expert.resolve(),
            "validation": args.us_validation_expert.resolve(),
        },
        "japanese": {
            "train": args.japanese_train_expert.resolve(),
            "validation": args.japanese_validation_expert.resolve(),
        },
    }
    validate_active_causal_motif_expert_paths(recipe, expert_paths)
    transitions, prepared, loader_calls = load_and_prepare_domains(expert_paths, recipe)
    cell_records: list[dict[str, Any]] = []
    depths, seeds = recipe_matrix_axes(recipe)
    cells = (
        priority_cells_first(
            depths=depths,
            seeds=seeds,
            limit=int(args.model_limit),
        )
        if bool(args.priority_cells_first)
        else matrix_cells(
            depths=depths,
            seeds=seeds,
            limit=int(args.model_limit),
        )
    )
    progress_path = policy_root / "matrix_progress.json"
    progress_base = {
        "schema_version": 1,
        "study": "bc_domain_depth_load_once_v2",
        "study_id": str(args.study_id),
        "recipe": str(recipe_path),
        "execution": "single_process_serial_models_load_each_domain_once",
        "split_method": "explicit_independent_train_validation_sources",
        "test_source_status": "pending_deferred_not_accepted",
        "expert_sources": {
            domain: {
                split: str(path)
                for split, path in expert_paths[domain].items()
            }
            for domain in DOMAINS
        },
        "priority_cells_first": bool(args.priority_cells_first),
        "recipe_depths": list(depths),
        "recipe_policy_seeds": list(seeds),
        "benchmark": {
            "matrix_completion_gate": "all_training_artifacts_complete",
            "interpretability_eligibility_gate": "policy_realism_qualified",
            "seed_selection_gate": "validation_selection_eligible",
            "test_evaluation_mode": str(
                recipe["evaluation"].get(
                    "test_evaluation_mode",
                    "deferred",
                )
            ),
            "closed_loop_metrics_role": (
                "qualification_non_terminal_for_matrix_execution"
            ),
            "shared_validation_framework": "shared_bc_gail_paper_metrics_v1",
            "validation_vehicle_mode": str(
                recipe["evaluation"].get("vehicle_mode", "single")
            ),
            "collision_termination_enabled": bool(
                recipe["evaluation"].get("terminate_on_collision", True)
            ),
        },
        "loader_calls": loader_calls,
        "loaded_samples": {
            domain: {
                split: int(len(transitions[domain][split].policy_observations))
                for split in ("train", "validation")
            }
            for domain in DOMAINS
        },
        "requested_model_count": len(cells),
    }
    write_json(
        progress_path,
        {
            **progress_base,
            "status": "training",
            "completed_model_count": 0,
            "training_artifact_complete_count": 0,
            "validation_selection_eligible_count": 0,
            "interpretability_baseline_eligible_count": 0,
            "metric_capability_passed_count": 0,
            "held_out_metric_capability_passed_count": 0,
            "offline_capability_passed_count": 0,
            "closed_loop_quality_passed_count": 0,
            "capability_passed_count": 0,
            "policy_realism_qualified_count": 0,
            "cells": [],
        },
    )
    for domain, depth, seed in cells:
        policy_model = str(recipe["architecture"].get("policy_model", "recurrent_transformer"))
        relative = Path(domain) / f"{policy_model}_{depth}layer" / f"policy_seed_{seed}"
        run_dir = policy_root / relative
        summary = run_training(
            cell_args(
                recipe,
                domain=domain,
                depth=depth,
                seed=seed,
                expert_train_data=expert_paths[domain]["train"],
                expert_validation_data=expert_paths[domain]["validation"],
                episode_root=args.episode_root.resolve(),
                out_dir=run_dir,
                device=str(args.device),
            ),
            transitions=prepared[domain].transitions,
            prepared_data=prepared[domain],
        )
        cell_records.append(
            {
                "domain": domain,
                "transformer_layers": depth,
                "seed": seed,
                "relative_path": str(relative),
                "summary": str(run_dir / "summary.json"),
                "checkpoint": summary.get("checkpoint"),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "training_artifact_complete": bool(
                    summary.get("training_artifact_complete")
                ),
                "validation_selection_eligible": bool(
                    summary.get("validation_selection_eligible")
                ),
                "interpretability_baseline_eligible": bool(
                    summary.get("interpretability_baseline_eligible")
                ),
                "metric_capability_passed": bool(summary.get("metric_capability_passed")),
                "held_out_metric_capability_passed": bool(
                    summary.get("held_out_metric_capability_passed")
                ),
                "offline_capability_passed": bool(
                    summary.get("offline_capability_passed")
                ),
                "paper_validation_cost": summary.get("paper_validation_cost"),
                "paper_validation_score": summary.get("paper_validation_score"),
                "closed_loop_evaluation_complete": bool(
                    summary.get("closed_loop_evaluation_complete")
                ),
                "rollout_capability_passed": bool(
                    summary.get("rollout_capability_passed")
                ),
                "matched_evaluation_passed": bool(
                    summary.get("matched_evaluation_passed")
                ),
                "matched_evaluation_complete": bool(
                    summary.get("matched_evaluation_complete")
                ),
                "matched_evaluation_contract_passed": bool(
                    summary.get("matched_evaluation_contract_passed")
                ),
                "closed_loop_quality_passed": bool(
                    summary.get("closed_loop_quality_passed")
                ),
                "capability_passed": bool(summary.get("capability_passed")),
                "policy_realism_qualified": bool(
                    summary.get("policy_realism_qualified")
                ),
                "validation_candidate_selected": False,
                "development_fallback_selected": False,
            }
        )
        write_json(
            progress_path,
            {
                **progress_base,
                "status": "training",
                "completed_model_count": len(cell_records),
                "training_artifact_complete_count": sum(
                    int(record["training_artifact_complete"])
                    for record in cell_records
                ),
                "validation_selection_eligible_count": sum(
                    int(record["validation_selection_eligible"])
                    for record in cell_records
                ),
                "interpretability_baseline_eligible_count": sum(
                    int(record["interpretability_baseline_eligible"])
                    for record in cell_records
                ),
                "metric_capability_passed_count": sum(
                    int(record["metric_capability_passed"]) for record in cell_records
                ),
                "held_out_metric_capability_passed_count": sum(
                    int(record["held_out_metric_capability_passed"])
                    for record in cell_records
                ),
                "offline_capability_passed_count": sum(
                    int(record["offline_capability_passed"])
                    for record in cell_records
                ),
                "closed_loop_quality_passed_count": sum(
                    int(record["closed_loop_quality_passed"])
                    for record in cell_records
                ),
                "capability_passed_count": sum(
                    int(record["capability_passed"]) for record in cell_records
                ),
                "policy_realism_qualified_count": sum(
                    int(record["policy_realism_qualified"])
                    for record in cell_records
                ),
                "cells": cell_records,
            },
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validation_candidate_selections = (
        select_validation_candidate_checkpoints(cell_records)
    )
    completed_sources = [
        policy_root / record["relative_path"]
        for record in cell_records
        if record["training_artifact_complete"]
    ]
    archive: dict[str, Any] | None = None
    if completed_sources and not bool(args.skip_archive):
        archive = archive_bc_checkpoints(
            completed_sources,
            destination_root=args.checkpoint_archive_root.resolve(),
            archive_id=str(args.study_id),
            label=(
                "Complete BC reference checkpoints from the locked shared-"
                "architecture study. Policy realism gates interpretability "
                "eligibility; failed quality gates do not stop independent "
                "matrix cells."
            ),
            qualification_field="training_artifact_complete",
        )
    manifest = {
        "schema_version": 1,
        "study": "bc_domain_depth_load_once_v2",
        "study_id": str(args.study_id),
        "recipe": str(recipe_path),
        "recipe_payload": recipe,
        "execution": "single_process_serial_models_load_each_domain_once",
        "priority_cells_first": bool(args.priority_cells_first),
        "benchmark": progress_base["benchmark"],
        "loader_calls": loader_calls,
        "loaded_samples": {
            domain: {
                split: int(
                    len(transitions[domain][split].policy_observations)
                )
                for split in ("train", "validation")
            }
            for domain in DOMAINS
        },
        "split_method": progress_base["split_method"],
        "test_source_status": progress_base["test_source_status"],
        "expert_sources": progress_base["expert_sources"],
        "recipe_depths": list(depths),
        "recipe_policy_seeds": list(seeds),
        "model_count": len(cell_records),
        "training_artifact_complete_count": sum(
            int(record["training_artifact_complete"]) for record in cell_records
        ),
        "validation_selection_eligible_count": sum(
            int(record["validation_selection_eligible"])
            for record in cell_records
        ),
        "interpretability_baseline_eligible_count": sum(
            int(record["interpretability_baseline_eligible"])
            for record in cell_records
        ),
        "metric_capability_passed_count": sum(
            int(record["metric_capability_passed"]) for record in cell_records
        ),
        "held_out_metric_capability_passed_count": sum(
            int(record["held_out_metric_capability_passed"])
            for record in cell_records
        ),
        "offline_capability_passed_count": sum(
            int(record["offline_capability_passed"])
            for record in cell_records
        ),
        "closed_loop_quality_passed_count": sum(
            int(record["closed_loop_quality_passed"]) for record in cell_records
        ),
        "capability_passed_count": sum(
            int(record["capability_passed"]) for record in cell_records
        ),
        "policy_realism_qualified_count": sum(
            int(record["policy_realism_qualified"])
            for record in cell_records
        ),
        "validation_candidate_selected_count": len(
            validation_candidate_selections
        ),
        "validation_candidate_selections": validation_candidate_selections,
        "development_fallback_selected_count": sum(
            int(record.get("development_fallback_selected", False))
            for record in cell_records
        ),
        "cells": cell_records,
        "checkpoint_archive": archive,
    }
    write_json(policy_root / "matrix_manifest.json", manifest)
    write_json(
        progress_path,
        {
            **progress_base,
            "status": (
                "complete"
                if manifest["training_artifact_complete_count"]
                == manifest["model_count"]
                else "complete_with_artifact_failures"
            ),
            "completed_model_count": len(cell_records),
            "training_artifact_complete_count": manifest[
                "training_artifact_complete_count"
            ],
            "validation_selection_eligible_count": manifest[
                "validation_selection_eligible_count"
            ],
            "interpretability_baseline_eligible_count": manifest[
                "interpretability_baseline_eligible_count"
            ],
            "metric_capability_passed_count": manifest["metric_capability_passed_count"],
            "held_out_metric_capability_passed_count": manifest[
                "held_out_metric_capability_passed_count"
            ],
            "offline_capability_passed_count": manifest[
                "offline_capability_passed_count"
            ],
            "closed_loop_quality_passed_count": manifest[
                "closed_loop_quality_passed_count"
            ],
            "capability_passed_count": manifest["capability_passed_count"],
            "policy_realism_qualified_count": manifest[
                "policy_realism_qualified_count"
            ],
            "validation_candidate_selected_count": manifest[
                "validation_candidate_selected_count"
            ],
            "validation_candidate_selections": validation_candidate_selections,
            "development_fallback_selected_count": manifest[
                "development_fallback_selected_count"
            ],
            "cells": cell_records,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
