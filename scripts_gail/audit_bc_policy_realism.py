#!/usr/bin/env python3
"""Re-audit BC artifacts without rewriting their immutable training summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_gail.ps_gail.validation import (
    action_learning_gate,
    closed_loop_policy_quality,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _learning_thresholds(recipe: dict[str, Any]) -> tuple[list[int], list[float], list[float]]:
    gates = dict(recipe["gates"])
    indices = list(gates.get("learning_action_indices") or [gates["learning_action_index"]])
    std_ratios = list(
        gates.get("min_learning_action_std_ratios")
        or [gates["min_learning_action_std_ratio"]]
    )
    correlations = list(
        gates.get("min_learning_action_correlations")
        or [gates["min_learning_action_correlation"]]
    )
    return (
        [int(value) for value in indices],
        [float(value) for value in std_ratios],
        [float(value) for value in correlations],
    )


def audit_cell(
    record: dict[str, Any],
    *,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    summary_path = Path(str(record["summary"])).resolve()
    summary = read_json(summary_path)
    action_indices, minimum_std_ratios, minimum_correlations = (
        _learning_thresholds(recipe)
    )
    validation_learning = action_learning_gate(
        split="validation",
        prediction_std_ratios=summary["validation_prediction_std_ratio"],
        prediction_target_correlations=summary[
            "validation_prediction_target_correlation"
        ],
        action_indices=action_indices,
        minimum_std_ratios=minimum_std_ratios,
        minimum_correlations=minimum_correlations,
    )
    test_learning = action_learning_gate(
        split="test",
        prediction_std_ratios=summary["test_prediction_std_ratio"],
        prediction_target_correlations=summary[
            "test_prediction_target_correlation"
        ],
        action_indices=action_indices,
        minimum_std_ratios=minimum_std_ratios,
        minimum_correlations=minimum_correlations,
    )
    gates = dict(recipe["gates"])
    offline_validation_passed = bool(
        np.isfinite(float(summary["validation_skill"]))
        and float(summary["validation_skill"]) >= float(gates["min_validation_skill"])
        and np.isfinite(float(summary["validation_mae"]))
        and float(summary["validation_mae"]) <= float(gates["max_validation_mae"])
        and validation_learning["passed"]
    )
    offline_test_passed = bool(
        np.isfinite(float(summary["test_mse"]))
        and np.isfinite(float(summary["test_mae"]))
        and float(summary["test_mae"]) <= float(gates["max_validation_mae"])
        and test_learning["passed"]
    )

    matched = dict(summary.get("gail_aligned_matched_evaluation") or {})
    validation_metrics = dict(matched.get("validation_metrics") or {})
    test_metrics = dict(matched.get("test_metrics") or {})
    evaluation = dict(recipe["evaluation"])
    horizon = int(matched.get("score_horizon_seconds", 20))
    minimum_coverage = float(matched.get("minimum_horizon_coverage", 0.0))
    validation_quality = closed_loop_policy_quality(
        validation_metrics,
        prefix="validation",
        max_vehicle_crash_rate=float(evaluation["max_crash_fraction"]),
        max_vehicle_offroad_rate=float(evaluation["max_offroad_fraction"]),
        score_horizon_seconds=horizon,
        min_horizon_coverage=minimum_coverage,
    )
    test_quality = closed_loop_policy_quality(
        test_metrics,
        prefix="test",
        max_vehicle_crash_rate=float(evaluation["max_crash_fraction"]),
        max_vehicle_offroad_rate=float(evaluation["max_offroad_fraction"]),
        score_horizon_seconds=horizon,
        min_horizon_coverage=minimum_coverage,
    )
    closed_loop_passed = bool(
        validation_quality["passed"] and test_quality["passed"]
    )
    offline_passed = bool(offline_validation_passed and offline_test_passed)
    qualified = bool(offline_passed and closed_loop_passed)
    return {
        "domain": str(record["domain"]),
        "transformer_layers": int(record["transformer_layers"]),
        "seed": int(record["seed"]),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "training_artifact_complete": bool(
            summary.get("training_artifact_complete")
        ),
        "offline_validation_gate": {
            "passed": offline_validation_passed,
            "validation_skill": float(summary["validation_skill"]),
            "maximum_validation_mae": float(gates["max_validation_mae"]),
            "validation_mae": float(summary["validation_mae"]),
            "action_learning": validation_learning,
        },
        "offline_test_gate": {
            "passed": offline_test_passed,
            "maximum_test_mae": float(gates["max_validation_mae"]),
            "test_mae": float(summary["test_mae"]),
            "action_learning": test_learning,
        },
        "offline_capability_passed": offline_passed,
        "closed_loop_quality_gate": {
            "validation": validation_quality,
            "test": test_quality,
            "passed": closed_loop_passed,
        },
        "policy_realism_qualified": qualified,
        "historical_flags": {
            "metric_capability_passed": bool(
                summary.get("metric_capability_passed")
            ),
            "closed_loop_quality_passed": bool(
                summary.get("closed_loop_quality_passed")
            ),
            "capability_passed": bool(summary.get("capability_passed")),
            "interpretability_baseline_eligible": bool(
                summary.get("interpretability_baseline_eligible")
            ),
        },
        "historical_false_positive": bool(
            not qualified
            and (
                summary.get("closed_loop_quality_passed")
                or summary.get("capability_passed")
            )
        ),
    }


def audit_matrix(matrix_manifest_path: Path) -> dict[str, Any]:
    manifest_path = matrix_manifest_path.resolve()
    manifest = read_json(manifest_path)
    recipe = dict(manifest.get("recipe_payload") or {})
    if not recipe:
        recipe_path = Path(str(manifest["recipe"])).resolve()
        recipe = read_json(recipe_path)
    cells = [
        audit_cell(dict(record), recipe=recipe)
        for record in list(manifest.get("cells") or [])
    ]
    if not cells:
        raise ValueError(f"No matrix cells found in {manifest_path}.")
    qualified = [cell for cell in cells if cell["policy_realism_qualified"]]
    return {
        "schema_version": 1,
        "audit": "bc_policy_realism_fail_closed_v1",
        "source_matrix_manifest": str(manifest_path),
        "source_matrix_manifest_sha256": sha256_file(manifest_path),
        "source_study_id": manifest.get("study_id"),
        "model_count": len(cells),
        "training_artifact_complete_count": sum(
            int(cell["training_artifact_complete"]) for cell in cells
        ),
        "offline_capability_passed_count": sum(
            int(cell["offline_capability_passed"]) for cell in cells
        ),
        "closed_loop_quality_passed_count": sum(
            int(cell["closed_loop_quality_gate"]["passed"]) for cell in cells
        ),
        "policy_realism_qualified_count": len(qualified),
        "historical_false_positive_count": sum(
            int(cell["historical_false_positive"]) for cell in cells
        ),
        "policy_qualification_status": (
            "passed" if len(qualified) == len(cells) else "failed"
        ),
        "environment_realism": {
            "us": {
                "status": "pending",
                "reason": (
                    "No same-scenario expert-replay collision baseline is "
                    "attached to this historical matrix."
                ),
            },
            "japanese": {
                "status": "not_independently_validated",
                "reason": (
                    "The hand-built road and lane-aware recentering/clipping "
                    "share the same topology; zero off-road replay is not an "
                    "independent geometry validation."
                ),
            },
        },
        "qualified_models": [
            {
                "domain": cell["domain"],
                "transformer_layers": cell["transformer_layers"],
                "seed": cell["seed"],
            }
            for cell in qualified
        ],
        "cells": cells,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.out.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_path}")
    audit = audit_matrix(args.matrix_manifest)
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
