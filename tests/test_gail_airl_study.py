from __future__ import annotations

import json
import sys

import pytest
import torch

from scripts_gail import build_gail_airl_study
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.experiment import (
    config_hash,
    policy_relative_l2_delta,
    resolve_algorithm_variant,
    write_evaluation_summary,
    write_run_manifest,
)
from scripts_gail.ps_gail.health import (
    TrainingHealthMonitor,
    partition_health_reasons,
)
from scripts_gail.ps_gail.monitoring import WandbMonitor
from scripts_gail.ps_gail.study import (
    assess_evaluation_summary,
    build_screening_trials,
    safety_thresholds,
    write_study_files,
)


def test_explicit_algorithm_variants_lock_canonical_contracts():
    gail = resolve_algorithm_variant(
        PSGAILConfig(
            algorithm_variant="gail_bce",
            discriminator_loss="wgan_gp",
            normalize_gail_reward=True,
        ),
        trainer="gail",
    )
    assert gail.discriminator_loss == "bce"
    assert not gail.normalize_gail_reward

    airl = resolve_algorithm_variant(
        PSGAILConfig(
            algorithm_variant="airl_bce",
            discriminator_loss="wgan_gp",
            airl_policy_reward_mode="shaped",
        ),
        trainer="airl",
    )
    assert airl.discriminator_loss == "airl_bce"
    assert airl.airl_policy_reward_mode == "discriminator"
    assert not airl.normalize_gail_reward


def test_algorithm_variant_rejects_wrong_trainer_and_unknown_name():
    with pytest.raises(ValueError, match="cannot run"):
        resolve_algorithm_variant(
            PSGAILConfig(algorithm_variant="airl_bce"),
            trainer="gail",
        )
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_algorithm_variant(
            PSGAILConfig(algorithm_variant="invented"),
            trainer="gail",
        )


def test_manifest_is_stable_and_summary_serializes_nonfinite_as_null(tmp_path):
    cfg = PSGAILConfig(
        algorithm_variant="gail_bce",
        seed=7,
        expert_data=str(tmp_path / "expert"),
        episode_root=str(tmp_path / "episodes"),
    )
    assert config_hash(cfg) == config_hash(cfg)
    manifest_path = write_run_manifest(str(tmp_path / "run"), cfg, trainer="gail")
    summary_path = write_evaluation_summary(
        str(tmp_path / "run"),
        cfg,
        trainer="gail",
        best_validation_score=float("-inf"),
        best_validation_round=0,
        final_validation_metrics={"validation/score": float("nan")},
        stress_metrics={},
        test_metrics={},
    )
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    summary = json.loads((tmp_path / "run" / "evaluation_summary.json").read_text())
    assert manifest_path.endswith("run_manifest.json")
    assert summary_path.endswith("evaluation_summary.json")
    assert manifest["algorithm_variant"] == "gail_bce"
    assert manifest["config_hash"] == summary["config_hash"]
    assert summary["best_validation_score"] is None
    assert summary["initial_validation"] == {}
    assert summary["final_validation"]["validation/score"] is None


def test_screening_study_has_preregistered_balanced_24_trials_per_method(tmp_path):
    trials = build_screening_trials(
        expert_data="expert",
        episode_root="episodes",
        bc_checkpoint="bc.pt",
        bc_policy_config={"transformer_layers": 3, "hidden_size": 192},
    )
    assert len(trials) == 48
    for method in ("gail", "airl"):
        method_trials = [trial for trial in trials if trial.method == method]
        assert len(method_trials) == 24
        assert sum(trial.phase == "factorial" for trial in method_trials) == 12
        assert sum(trial.phase == "objective" for trial in method_trials) == 4
        assert sum(trial.phase == "hpo" for trial in method_trials) == 8
    factorial = [trial for trial in trials if trial.phase == "factorial"]
    assert {trial.initialization for trial in factorial} == {"cold", "bc"}
    assert {trial.collision_training for trial in factorial} == {"full", "soft"}
    curriculum = next(trial for trial in trials if trial.collision_training == "curriculum")
    assert curriculum.arguments["transformer_layers"] == 3
    assert curriculum.arguments["hidden_size"] == 192
    assert "soft" in curriculum.arguments["collision_mode_schedule"]
    assert "mixed" in curriculum.arguments["collision_mode_schedule"]
    assert curriculum.arguments["test_episodes"] == 30
    assert curriculum.arguments["test_vehicle_mode"] == "training_count"
    assert curriculum.arguments["evaluate_initial_policy"] is True
    assert curriculum.arguments["warmup_rounds"] == 5
    assert curriculum.arguments["warmup_learning_rate"] == pytest.approx(5.0e-6)
    assert curriculum.arguments["value_clip_range"] == pytest.approx(0.2)
    assert curriculum.arguments["validation_max_score_drop"] == pytest.approx(5.0)
    assert curriculum.arguments["validation_regression_patience"] == 2
    assert all(trial.arguments["policy_bc_regularization_coef"] == 0.0 for trial in trials)
    paths = write_study_files(str(tmp_path), trials)
    assert len((tmp_path / "commands.txt").read_text().splitlines()) == 48
    assert json.loads((tmp_path / "trials.json").read_text())["trial_count"] == 48
    assert paths["manifest"].endswith("trials.json")


def test_safety_gate_uses_expert_relative_floors_and_rejects_nonfinite():
    assert safety_thresholds(expert_crash_rate=0.0, expert_offroad_rate=0.0) == (0.05, 0.02)
    eligible = assess_evaluation_summary(
        {
            "test": {
                "test/vehicle_crash_rate": 0.04,
                "test/vehicle_offroad_rate": 0.01,
                "test/rmse_position_20s": 12.0,
                "test/rmse_speed_20s": 2.0,
                "test/rmse_lane_offset_20s": 0.5,
            }
        },
        expert_crash_rate=0.0,
        expert_offroad_rate=0.0,
    )
    assert eligible["eligible"]
    rejected = assess_evaluation_summary(
        {
            "test": {
                "test/vehicle_crash_rate": 0.0,
                "test/vehicle_offroad_rate": 0.0,
                "test/rmse_position_20s": None,
                "test/rmse_speed_20s": 1.0,
                "test/rmse_lane_offset_20s": 0.1,
            }
        },
        expert_crash_rate=0.0,
        expert_offroad_rate=0.0,
    )
    assert not rejected["eligible"]
    assert not rejected["finite"]


def test_training_health_gate_uses_consecutive_failures():
    cfg = PSGAILConfig(
        target_kl=0.01,
        health_kl_patience=2,
        health_discriminator_patience=2,
        health_reward_std_patience=2,
    )
    monitor = TrainingHealthMonitor()
    first = monitor.observe(
        cfg,
        approx_kl=0.02,
        expert_accuracy=0.99,
        generator_accuracy=0.99,
        reward_std=1.0e-5,
        action_std=0.2,
    )
    assert first == []
    second = monitor.observe(
        cfg,
        approx_kl=0.02,
        expert_accuracy=0.99,
        generator_accuracy=0.99,
        reward_std=1.0e-5,
        action_std=0.2,
    )
    assert set(second) == {
        "target_kl_repeatedly_exceeded",
        "discriminator_saturated",
        "adversarial_reward_collapsed",
    }
    fatal, warnings = partition_health_reasons(second)
    assert fatal == [
        "target_kl_repeatedly_exceeded",
        "adversarial_reward_collapsed",
    ]
    assert warnings == ["discriminator_saturated"]
    assert monitor.observe(
        cfg,
        approx_kl=0.0,
        expert_accuracy=0.5,
        generator_accuracy=0.5,
        reward_std=1.0,
        action_std=0.2,
        extra_metrics={"value_loss": float("nan")},
    ) == ["nonfinite:value_loss"]

    validation_monitor = TrainingHealthMonitor()
    validation_cfg = PSGAILConfig(
        validation_max_score_drop=5.0,
        validation_regression_patience=2,
    )
    assert validation_monitor.observe_validation(validation_cfg, score=-6.0, best_score=0.0) == []
    assert validation_monitor.observe_validation(
        validation_cfg,
        score=-7.0,
        best_score=0.0,
    ) == ["validation_score_repeatedly_regressed"]


def test_learning_health_gate_requires_post_initializer_improvement():
    cfg = PSGAILConfig(
        health_learning_gate_round=20,
        health_min_best_round=20,
        health_min_relative_validation_improvement=0.02,
    )
    monitor = TrainingHealthMonitor()
    assert monitor.observe_learning(
        cfg,
        round_idx=19,
        initial_score=-100.0,
        best_score=-100.0,
        best_round=0,
    ) == []
    assert monitor.observe_learning(
        cfg,
        round_idx=20,
        initial_score=-100.0,
        best_score=-99.0,
        best_round=20,
    ) == ["validation_did_not_meet_learning_improvement_gate"]
    assert monitor.observe_learning(
        cfg,
        round_idx=20,
        initial_score=-100.0,
        best_score=-97.9,
        best_round=20,
    ) == []


def test_policy_delta_and_durable_compact_metric_reporting(tmp_path):
    initializer = {"weight": torch.tensor([1.0, 2.0])}
    selected = {"weight": torch.tensor([2.0, 2.0])}
    assert policy_relative_l2_delta(selected, initializer) == pytest.approx(1.0 / 5.0**0.5)

    cfg = PSGAILConfig(
        wandb_mode="disabled",
        discriminator_loss="bce",
        wandb_compact_metrics=True,
    )
    monitor = WandbMonitor(cfg, str(tmp_path), trainer="gail")
    monitor.log(
        {
            "rollout/mean_reward": 2.0,
            "rollout/crash_agent_fraction": 0.1,
            "discriminator/expert_acc": 0.75,
            "misdefined/noise": 999.0,
            "nonfinite": float("nan"),
        },
        step=1,
    )
    monitor.log({"validation/score": -3.0}, step=1)
    compact = monitor._compact_metrics(monitor._pending_metrics)
    assert compact["reward/training_mean"] == pytest.approx(2.0)
    assert compact["rollout/crash_transition_fraction"] == pytest.approx(0.1)
    assert compact["discriminator/expert_accuracy"] == pytest.approx(0.75)
    assert "misdefined/noise" not in compact
    assert "nonfinite" not in compact
    monitor.finish()

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["step"] == 1
    assert rows[0]["validation/score"] == pytest.approx(-3.0)
    assert rows[0]["nonfinite"] is None


def test_study_cli_requires_and_propagates_validated_bc_architecture(tmp_path, monkeypatch):
    bc_dir = tmp_path / "bc"
    bc_dir.mkdir()
    checkpoint = bc_dir / "best.pt"
    torch.save(
        {
            "checkpoint_kind": "behaviour_cloning_warm_start",
            "config": {
                "policy_model": "recurrent_transformer",
                "hidden_size": 192,
                "transformer_layers": 3,
                "transformer_heads": 4,
            },
            "policy_state_dict": {},
        },
        checkpoint,
    )
    (bc_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_purpose": "warm_start",
                "checkpoint_saved": True,
                "warm_start_passed": True,
                "configured_epochs": 2,
                "relative_validation_improvement": 0.1,
            }
        )
    )
    output = tmp_path / "study"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_gail_airl_study",
            "--bc-checkpoint",
            str(checkpoint),
            "--expert-data",
            str(tmp_path / "expert"),
            "--episode-root",
            str(tmp_path / "episodes"),
            "--output-dir",
            str(output),
        ],
    )

    build_gail_airl_study.main()

    payload = json.loads((output / "trials.json").read_text())
    assert payload["trial_count"] == 48
    assert all(row["arguments"]["hidden_size"] == 192 for row in payload["trials"])
    assert all(row["arguments"]["transformer_layers"] == 3 for row in payload["trials"])
