#!/usr/bin/env python3
"""Train the 12-cell BC matrix while loading each expert domain exactly once."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch

from scripts_gail.archive_bc_checkpoints import archive_bc_checkpoints
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.recurrent_bc import PreparedRecurrentBCData, prepare_recurrent_bc_data
from scripts_gail.train_recurrent_bc_policy import run_training, write_json


DOMAINS = ("us", "japanese")
DEPTHS = (2, 3)
SEEDS = (0, 1, 2)


def read_locked_recipe(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        recipe = json.load(handle)
    if int(recipe.get("schema_version", -1)) != 1 or recipe.get("status") != "locked":
        raise ValueError(f"BC recipe is not a locked schema-v1 recipe: {path}")
    for section in ("architecture", "optimization", "data", "gates", "evaluation"):
        if not isinstance(recipe.get(section), dict):
            raise ValueError(f"Locked BC recipe is missing section {section!r}: {path}")
    return recipe


def matrix_cells(
    limit: int = 12,
    *,
    depths: tuple[int, ...] = DEPTHS,
) -> list[tuple[str, int, int]]:
    cells = [(domain, depth, seed) for domain in DOMAINS for depth in depths for seed in SEEDS]
    if not 1 <= int(limit) <= len(cells):
        raise ValueError(f"model_limit must be between 1 and {len(cells)}; got {limit}")
    return cells[: int(limit)]


def priority_cells_first(
    limit: int = 12,
    *,
    depths: tuple[int, ...] = DEPTHS,
) -> list[tuple[str, int, int]]:
    priority_depth = max(depths)
    # Run a previously locally exercised seed for both domains first so early
    # logs are informative. This changes ordering only: no metric or rollout
    # result is allowed to terminate the remaining matrix.
    priority = [("us", priority_depth, 0), ("japanese", priority_depth, 0)]
    ordered = priority + [
        cell
        for cell in matrix_cells(
            len(DOMAINS) * len(depths) * len(SEEDS),
            depths=depths,
        )
        if cell not in priority
    ]
    if not 1 <= int(limit) <= len(ordered):
        raise ValueError(f"model_limit must be between 1 and {len(ordered)}; got {limit}")
    return ordered[: int(limit)]


def cell_args(
    recipe: dict[str, Any],
    *,
    domain: str,
    depth: int,
    seed: int,
    expert_data: Path,
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
    return argparse.Namespace(
        policy_model=str(architecture.get("policy_model", "recurrent_transformer")),
        expert_data=str(expert_data),
        out_dir=str(out_dir),
        domain=domain,
        scene=scene,
        episode_root=str(episode_root),
        prebuilt_split="train",
        seed=int(seed),
        data_seed=int(data["data_seed"]),
        split_seed=int(data["split_seed"]),
        max_expert_samples=int(data["max_expert_samples"]),
        epochs=int(optimization["epochs"]),
        checkpoint_purpose="policy",
        max_warmup_epochs=5,
        min_warmup_relative_improvement=0.01,
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        train_fraction=float(data["train_fraction"]),
        validation_fraction=float(data["validation_fraction"]),
        early_stopping_patience=int(optimization["early_stopping_patience"]),
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
        transformer_observation_tokenization=str(
            architecture.get("transformer_observation_tokenization", "semantic")
        ),
        policy_head_init_std=float(architecture["policy_head_init_std"]),
        memory_tokens=int(architecture["memory_tokens"]),
        memory_context_length=int(architecture["memory_context_length"]),
        sequence_length=int(optimization["sequence_length"]),
        sequences_per_batch=int(optimization["sequences_per_batch"]),
        micro_batch_sequences=int(optimization["micro_batch_sequences"]),
        render_video=False,
        video_steps=200,
        evaluation_episodes=int(evaluation["episodes"]),
        validation_evaluation_episodes=int(
            evaluation.get("validation_episodes", evaluation["episodes"])
        ),
        evaluation_split="test",
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
        capability_failure_mode="report",
    )


def load_and_prepare_domains(
    expert_paths: dict[str, Path],
    recipe: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, PreparedRecurrentBCData], dict[str, int]]:
    data = recipe["data"]
    optimization = recipe["optimization"]
    architecture = recipe["architecture"]
    transitions: dict[str, Any] = {}
    prepared: dict[str, PreparedRecurrentBCData] = {}
    loader_calls: dict[str, int] = {domain: 0 for domain in DOMAINS}
    for domain in DOMAINS:
        transitions[domain] = load_expert_transition_data(
            str(expert_paths[domain]),
            max_samples=int(data["max_expert_samples"]),
            seed=int(data["data_seed"]),
            trajectory_frame="relative",
        )
        loader_calls[domain] += 1
        prepared[domain] = prepare_recurrent_bc_data(
            transitions[domain],
            split_seed=int(data["split_seed"]),
            sequence_length=int(optimization["sequence_length"]),
            train_fraction=float(data["train_fraction"]),
            validation_fraction=float(data["validation_fraction"]),
            context_length=int(architecture["memory_context_length"]),
        )
    return transitions, prepared, loader_calls


def select_paper_benchmark_checkpoints(
    cell_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select one validation-ranked seed per domain/depth without dropping others."""
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
            if bool(record.get("interpretability_baseline_eligible"))
        ]
        pool = primary or candidates
        if not pool:
            continue
        selected = max(
            pool,
            key=lambda record: (
                float(record["paper_validation_score"]),
                -int(record["seed"]),
            ),
        )
        selected["paper_benchmark_selected"] = True
        selected["paper_benchmark_selection_tier"] = (
            "offline_capable" if primary else "artifact_fallback"
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
                    "paper_benchmark_selection_tier"
                ],
            }
        )
    return selections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--us-expert", type=Path, required=True)
    parser.add_argument("--japanese-expert", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--checkpoint-archive-root", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--model-limit", type=int, default=12)
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
        "us": args.us_expert.resolve(),
        "japanese": args.japanese_expert.resolve(),
    }
    transitions, prepared, loader_calls = load_and_prepare_domains(expert_paths, recipe)
    cell_records: list[dict[str, Any]] = []
    depths = tuple(int(value) for value in recipe["architecture"].get("depths", DEPTHS))
    cells = (
        priority_cells_first(int(args.model_limit), depths=depths)
        if bool(args.priority_cells_first)
        else matrix_cells(int(args.model_limit), depths=depths)
    )
    progress_path = policy_root / "matrix_progress.json"
    progress_base = {
        "schema_version": 1,
        "study": "bc_domain_depth_load_once_v2",
        "study_id": str(args.study_id),
        "recipe": str(recipe_path),
        "execution": "single_process_serial_models_load_each_domain_once",
        "priority_cells_first": bool(args.priority_cells_first),
        "benchmark": {
            "matrix_completion_gate": "all_training_artifacts_complete",
            "interpretability_eligibility_gate": "offline_imitation_learning",
            "closed_loop_metrics_role": "descriptive_non_terminal",
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
            domain: int(len(transitions[domain].policy_observations)) for domain in DOMAINS
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
            "interpretability_baseline_eligible_count": 0,
            "metric_capability_passed_count": 0,
            "closed_loop_quality_passed_count": 0,
            "capability_passed_count": 0,
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
                expert_data=expert_paths[domain],
                episode_root=args.episode_root.resolve(),
                out_dir=run_dir,
                device=str(args.device),
            ),
            transitions=transitions[domain],
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
                "interpretability_baseline_eligible": bool(
                    summary.get("interpretability_baseline_eligible")
                ),
                "metric_capability_passed": bool(summary.get("metric_capability_passed")),
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
                "closed_loop_quality_passed": bool(
                    summary.get("closed_loop_quality_passed")
                ),
                "capability_passed": bool(summary.get("capability_passed")),
                "paper_benchmark_selected": False,
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
                "interpretability_baseline_eligible_count": sum(
                    int(record["interpretability_baseline_eligible"])
                    for record in cell_records
                ),
                "metric_capability_passed_count": sum(
                    int(record["metric_capability_passed"]) for record in cell_records
                ),
                "closed_loop_quality_passed_count": sum(
                    int(record["closed_loop_quality_passed"])
                    for record in cell_records
                ),
                "capability_passed_count": sum(
                    int(record["capability_passed"]) for record in cell_records
                ),
                "cells": cell_records,
            },
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    paper_benchmark_selections = select_paper_benchmark_checkpoints(cell_records)
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
                "architecture study. Offline imitation gates label primary "
                "interpretability eligibility; closed-loop metrics are "
                "descriptive and non-terminal."
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
            domain: int(len(transitions[domain].policy_observations)) for domain in DOMAINS
        },
        "model_count": len(cell_records),
        "training_artifact_complete_count": sum(
            int(record["training_artifact_complete"]) for record in cell_records
        ),
        "interpretability_baseline_eligible_count": sum(
            int(record["interpretability_baseline_eligible"])
            for record in cell_records
        ),
        "metric_capability_passed_count": sum(
            int(record["metric_capability_passed"]) for record in cell_records
        ),
        "closed_loop_quality_passed_count": sum(
            int(record["closed_loop_quality_passed"]) for record in cell_records
        ),
        "capability_passed_count": sum(
            int(record["capability_passed"]) for record in cell_records
        ),
        "paper_benchmark_selected_count": len(paper_benchmark_selections),
        "paper_benchmark_selections": paper_benchmark_selections,
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
            "interpretability_baseline_eligible_count": manifest[
                "interpretability_baseline_eligible_count"
            ],
            "metric_capability_passed_count": manifest["metric_capability_passed_count"],
            "closed_loop_quality_passed_count": manifest[
                "closed_loop_quality_passed_count"
            ],
            "capability_passed_count": manifest["capability_passed_count"],
            "paper_benchmark_selected_count": manifest[
                "paper_benchmark_selected_count"
            ],
            "paper_benchmark_selections": paper_benchmark_selections,
            "cells": cell_records,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
