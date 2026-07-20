#!/usr/bin/env python3
"""Train the 12-cell BC matrix while loading each expert domain exactly once."""

from __future__ import annotations

import argparse
import gc
import json
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


def matrix_cells(limit: int = 12) -> list[tuple[str, int, int]]:
    cells = [(domain, depth, seed) for domain in DOMAINS for depth in DEPTHS for seed in SEEDS]
    if not 1 <= int(limit) <= len(cells):
        raise ValueError(f"model_limit must be between 1 and {len(cells)}; got {limit}")
    return cells[: int(limit)]


def confirmation_first_cells(limit: int = 12) -> list[tuple[str, int, int]]:
    confirmations = [("us", 3, 0), ("japanese", 3, 1)]
    ordered = confirmations + [cell for cell in matrix_cells() if cell not in confirmations]
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
        learning_action_index=int(gates["learning_action_index"]),
        min_learning_action_std_ratio=float(gates["min_learning_action_std_ratio"]),
        min_learning_action_correlation=float(gates["min_learning_action_correlation"]),
        device=device,
        hidden_size=int(architecture["hidden_size"]),
        transformer_layers=int(depth),
        transformer_heads=int(architecture["transformer_heads"]),
        transformer_dropout=float(architecture["transformer_dropout"]),
        transformer_norm_first=bool(architecture["transformer_norm_first"]),
        policy_head_init_std=float(architecture["policy_head_init_std"]),
        memory_tokens=int(architecture["memory_tokens"]),
        memory_context_length=int(architecture["memory_context_length"]),
        sequence_length=int(optimization["sequence_length"]),
        sequences_per_batch=int(optimization["sequences_per_batch"]),
        micro_batch_sequences=int(optimization["micro_batch_sequences"]),
        render_video=False,
        video_steps=200,
        evaluation_episodes=int(evaluation["episodes"]),
        evaluation_split="test",
        evaluation_enable_collision=False,
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
    parser.add_argument("--require-confirmation", action="store_true")
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
    cells = (
        confirmation_first_cells(int(args.model_limit))
        if bool(args.require_confirmation)
        else matrix_cells(int(args.model_limit))
    )
    progress_path = policy_root / "matrix_progress.json"
    progress_base = {
        "schema_version": 1,
        "study": "bc_domain_depth_load_once_v2",
        "study_id": str(args.study_id),
        "recipe": str(recipe_path),
        "execution": "single_process_serial_models_load_each_domain_once",
        "confirmation_first": bool(args.require_confirmation),
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
            "metric_capability_passed_count": 0,
            "cells": [],
        },
    )
    for cell_index, (domain, depth, seed) in enumerate(cells):
        relative = Path(domain) / f"recurrent_transformer_{depth}layer" / f"policy_seed_{seed}"
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
                "metric_capability_passed": bool(summary.get("metric_capability_passed")),
            }
        )
        write_json(
            progress_path,
            {
                **progress_base,
                "status": "training",
                "completed_model_count": len(cell_records),
                "metric_capability_passed_count": sum(
                    int(record["metric_capability_passed"]) for record in cell_records
                ),
                "cells": cell_records,
            },
        )
        if bool(args.require_confirmation) and cell_index < 2 and not bool(summary.get("metric_capability_passed")):
            failure = {
                "schema_version": 1,
                "study": "bc_domain_depth_load_once_v2",
                "study_id": str(args.study_id),
                "status": "confirmation_failed",
                "failed_confirmation": cell_records[-1],
                "completed_cells": cell_records,
                "loader_calls": loader_calls,
                "recipe": str(recipe_path),
            }
            write_json(policy_root / "confirmation_failure.json", failure)
            raise RuntimeError(
                f"Three-layer confirmation failed for domain={domain} seed={seed}; "
                "remaining matrix cells were not started."
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    qualified_sources = [
        policy_root / record["relative_path"]
        for record in cell_records
        if record["metric_capability_passed"]
    ]
    archive: dict[str, Any] | None = None
    if qualified_sources and not bool(args.skip_archive):
        archive = archive_bc_checkpoints(
            qualified_sources,
            destination_root=args.checkpoint_archive_root.resolve(),
            archive_id=str(args.study_id),
            label="Learning-qualified checkpoints from the locked common-recipe BC depth study.",
        )
    manifest = {
        "schema_version": 1,
        "study": "bc_domain_depth_load_once_v2",
        "study_id": str(args.study_id),
        "recipe": str(recipe_path),
        "recipe_payload": recipe,
        "execution": "single_process_serial_models_load_each_domain_once",
        "confirmation_first": bool(args.require_confirmation),
        "loader_calls": loader_calls,
        "loaded_samples": {
            domain: int(len(transitions[domain].policy_observations)) for domain in DOMAINS
        },
        "model_count": len(cell_records),
        "metric_capability_passed_count": sum(
            int(record["metric_capability_passed"]) for record in cell_records
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
                "learning_qualified"
                if manifest["metric_capability_passed_count"] == manifest["model_count"]
                else "trained_with_learning_failures"
            ),
            "completed_model_count": len(cell_records),
            "metric_capability_passed_count": manifest["metric_capability_passed_count"],
            "cells": cell_records,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
