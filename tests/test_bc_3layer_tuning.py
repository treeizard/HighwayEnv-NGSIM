from __future__ import annotations

import json
from pathlib import Path

from scripts_gail.select_bc_3layer_tuning import select_tuning_candidate
from scripts_gail.diagnose_bc_3layer import infer_diagnosis


def write_summary(
    path: Path,
    *,
    skill: float,
    std_ratio: float,
    correlation: float,
    passed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "transformer_layers": 3,
                "domain": "japanese",
                "seed": 0,
                "learning_rate": 3.0e-4,
                "max_grad_norm": 1.0,
                "transformer_dropout": 0.0,
                "validation_skill": skill,
                "validation_mae": 0.05,
                "learning_signal_passed": passed,
                "metric_capability_passed": passed,
                "learning_signal_gate": {
                    "prediction_std_ratio": std_ratio,
                    "prediction_target_correlation": correlation,
                },
                "checkpoint": str(path.parent / "best.pt"),
                "checkpoint_sha256": "digest",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_tuning_selection_rejects_collapse_and_requires_all_confirmations(tmp_path):
    screen = tmp_path / "screen"
    write_summary(screen / "collapsed" / "summary.json", skill=0.002, std_ratio=0.01, correlation=0.1, passed=False)
    write_summary(screen / "learned" / "summary.json", skill=0.8, std_ratio=0.9, correlation=0.95, passed=True)

    selection = select_tuning_candidate(
        screen,
        expected_candidates=["collapsed", "learned"],
    )
    assert selection["screen_pass_count"] == 1
    assert selection["selected_candidate"]["trial_id"] == "learned"
    assert selection["promotion_status"] == "pending_confirmation"

    confirmation = tmp_path / "confirmation"
    write_summary(confirmation / "us_seed_0" / "summary.json", skill=0.7, std_ratio=0.8, correlation=0.9, passed=True)
    write_summary(
        confirmation / "japanese_seed_1" / "summary.json",
        skill=0.001,
        std_ratio=0.01,
        correlation=0.0,
        passed=False,
    )
    rejected = select_tuning_candidate(
        screen,
        expected_candidates=["collapsed", "learned"],
        confirmation_root=confirmation,
        expected_confirmations=["us_seed_0", "japanese_seed_1"],
    )
    assert rejected["confirmation_pass_count"] == 1
    assert rejected["promotion_status"] == "rejected"


def test_diagnosis_attributes_minimal_passing_change():
    selected = {"candidate_id": "baseline_no_early_stop"}
    assert "premature early stopping" in infer_diagnosis([], selected)
    assert "do not launch" in infer_diagnosis([], None)
