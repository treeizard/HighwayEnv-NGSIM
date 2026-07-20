#!/usr/bin/env python3
"""Screen and lock a cross-domain recurrent IQ-Learn production recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from scripts_gail.iq_study import CONFIRMATIONS, bc_checkpoint, expert_for_domain, trainer_command
from scripts_gail.train_recurrent_iq_learn import IQ_REFERENCE_COMMIT, IQ_REFERENCE_REPOSITORY, write_json


def candidates() -> list[dict[str, float]]:
    result = []
    for policy_lr, bc_coef in ((1.0e-5, 10.0), (3.0e-6, 30.0)):
        for q_lr in (3.0e-5, 1.0e-4):
            for temperature in (0.001, 0.01):
                result.append({
                    "policy_learning_rate": policy_lr,
                    "bc_coef": bc_coef,
                    "q_learning_rate": q_lr,
                    "entropy_temperature": temperature,
                })
    return result


def pilot_recipe(candidate: dict[str, float], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1, "status": "pilot", "recipe_id": "iq_pilot_candidate",
        "architecture": {
            "hidden_size": 256, "transformer_heads": 4, "transformer_dropout": 0.0,
            "transformer_norm_first": False, "memory_tokens": 8,
            "memory_context_length": 32,
        },
        "optimization": {
            "updates": int(args.pilot_updates), "minimum_joint_updates": int(args.pilot_joint_updates),
            "eval_every": int(args.pilot_eval_every), "early_stopping_evaluations": 6,
            **candidate, "gamma": 0.95, "chi2_alpha": 0.5,
            "chi2_regularization": "expert", "target_tau": 0.005, "target_q_clip": 20.0,
            "q_only_updates": int(args.pilot_q_only_updates), "max_grad_norm": 5.0,
            "max_q_abs": 50.0, "initial_log_std": -2.5, "log_std_min": -5.0,
            "log_std_max": -1.5, "training_context_length": 8, "sequence_length": 8,
            "sequences_per_update": 4, "micro_batch_sequences": 16,
        },
        "replay": {
            "initial_policy_replay": int(args.pilot_initial_replay), "capacity": 100_000,
            "collect_steps": int(args.pilot_collect_steps), "collect_every": 50,
            "training_enable_collision": False,
        },
        "data": {"max_expert_samples": 300_000, "data_seed": 20260716},
        "gates": {
            "min_validation_skill": 0.10, "max_initial_skill_regression": 0.02,
            "max_validation_mae": 0.35, "learning_action_index": 0,
            "min_learning_action_std_ratio": 0.25, "min_learning_action_correlation": 0.50,
        },
        "evaluation": {
            "episodes": 3, "validation_sequence_length": 32,
            "min_rollout_steps": 100, "max_crash_fraction": 0.34,
            "max_offroad_fraction": 0.34, "max_collision_proxy_fraction": 0.34,
            "max_mean_length_regression": 20.0, "max_fraction_regression": 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-root", type=Path, required=True)
    parser.add_argument("--us-expert", type=Path, required=True)
    parser.add_argument("--japanese-expert", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--locked-recipe-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot-updates", type=int, default=120)
    parser.add_argument("--pilot-q-only-updates", type=int, default=20)
    parser.add_argument("--pilot-joint-updates", type=int, default=100)
    parser.add_argument("--pilot-eval-every", type=int, default=40)
    parser.add_argument("--pilot-initial-replay", type=int, default=512)
    parser.add_argument("--pilot-collect-steps", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    if out_root.exists():
        raise FileExistsError(f"Refusing to reuse IQ pilot root: {out_root}")
    out_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates()):
        recipe = pilot_recipe(candidate, args)
        cell_records = []
        for domain, depth, seed in CONFIRMATIONS:
            initial = bc_checkpoint(args.bc_root.resolve(), domain, depth, seed)
            if not initial.is_file():
                raise FileNotFoundError(f"Matched BC pilot initializer is missing: {initial}")
            run_dir = out_root / f"candidate_{candidate_index:02d}" / f"{domain}_{depth}layer_seed{seed}"
            command = trainer_command(
                recipe, domain=domain, depth=depth, seed=seed,
                expert_data=expert_for_domain(domain, args.us_expert.resolve(), args.japanese_expert.resolve()),
                episode_root=args.episode_root.resolve(), initial_checkpoint=initial,
                out_dir=run_dir, device=str(args.device), capability_failure_mode="report",
            )
            completed = subprocess.run(command, check=False)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
            cell_records.append({
                "domain": domain, "depth": depth, "seed": seed, "returncode": completed.returncode,
                "summary": str(summary_path), "capability_passed": bool(summary.get("capability_passed")),
                "validation_skill": float((summary.get("best_validation") or {}).get("validation_skill", float("-inf"))),
                "test_mean_episode_length": float((summary.get("held_out_test_rollouts") or {}).get("bc_eval/mean_episode_length", 0.0)),
            })
        passed = all(record["capability_passed"] for record in cell_records)
        records.append({"candidate_index": candidate_index, "candidate": candidate, "passed": passed, "cells": cell_records})
        write_json(out_root / "pilot_progress.json", {"schema_version": 1, "status": "screening", "candidates": records})
    survivors = [record for record in records if record["passed"]]
    if not survivors:
        write_json(out_root / "pilot_failure.json", {"schema_version": 1, "status": "no_cross_domain_survivor", "candidates": records})
        raise RuntimeError("No IQ-Learn pilot candidate passed both cross-domain confirmation cells.")
    selected = max(
        survivors,
        key=lambda record: (
            min(cell["validation_skill"] for cell in record["cells"]),
            sum(cell["test_mean_episode_length"] for cell in record["cells"]) / len(record["cells"]),
            -int(record["candidate_index"]),
        ),
    )
    locked = pilot_recipe(selected["candidate"], args)
    locked["status"] = "locked"
    locked["recipe_id"] = f"online_iq_reference_pilot_{selected['candidate_index']:02d}"
    locked["optimization"].update({
        "updates": 2_000, "minimum_joint_updates": 800, "eval_every": 200,
        "q_only_updates": 200, "early_stopping_evaluations": 6,
    })
    locked["replay"].update({"initial_policy_replay": 1_024, "collect_steps": 512})
    locked["selection"] = {
        "pilot_root": str(out_root), "selected_candidate": selected,
        "rule": "maximize worst-domain validation skill, then mean held-out episode length",
    }
    locked["reference"] = {"repository": IQ_REFERENCE_REPOSITORY, "commit": IQ_REFERENCE_COMMIT}
    write_json(args.locked_recipe_out.resolve(), locked)
    write_json(out_root / "pilot_selection.json", locked)
    print(json.dumps(locked, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
