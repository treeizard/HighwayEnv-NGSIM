from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts_gail.finalize_bc_domain_depth_study import finalize_study, sha256_file
from scripts_gail.train_recurrent_bc_policy import (
    append_jsonl,
    save_checkpoint_artifacts,
    training_artifact_is_complete,
)
from scripts_gail.train_simple_ps_gail import _collision_only_flags, _collision_proxy_flags
from scripts_gail.ps_gail.validation import (
    paper_driver_model_validation_overrides,
)


ROOT = Path(__file__).resolve().parents[1]


def test_collision_reporting_separates_offroad_crashes_and_overlap_proxy():
    assert _collision_only_flags([True, True, False], [True, False, False]) == [False, True, False]
    assert _collision_proxy_flags([{"min_gap": -0.1}, {"min_gap": 2.0}, {}]) == [True, False, False]


def test_checkpoint_artifacts_are_atomic_and_hashed(tmp_path):
    checkpoint = tmp_path / "best.pt"
    summary: dict[str, object] = {"metric_capability_passed": True}
    payload = {"policy_state_dict": {"weight": torch.arange(3)}}

    digest = save_checkpoint_artifacts(checkpoint, payload, summary)

    assert checkpoint.is_file()
    assert not (tmp_path / ".best.pt.tmp").exists()
    assert digest == sha256_file(checkpoint)
    assert summary["checkpoint_saved"] is True
    assert summary["checkpoint_sha256"] == digest
    assert (tmp_path / "best.pt.sha256").read_text().split() == [digest, "best.pt"]


def test_validation_history_is_appended_one_epoch_at_a_time(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"epoch": 1.0, "validation_skill": 0.1, "is_best_so_far": True})
    append_jsonl(path, {"epoch": 2.0, "validation_skill": 0.2, "is_best_so_far": True})

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1.0, 2.0]
    assert [row["validation_skill"] for row in rows] == [0.1, 0.2]


def test_training_artifact_completion_does_not_depend_on_rollout_quality(tmp_path):
    for name in ("best.pt", "best.pt.sha256", "split_manifest.json"):
        (tmp_path / name).write_text("artifact\n")
    summary = {
        "checkpoint_saved": True,
        "checkpoint_sha256": "abc",
        "initial_validation_mse": 0.4,
        "validation_mse": 0.1,
        "validation_mae": 0.2,
        "validation_skill": 0.75,
        "validation_prediction_std_ratio": [0.8, 0.2],
        "validation_prediction_target_correlation": [0.9, 0.3],
        "rollout_capability_passed": False,
        "capability_passed": False,
    }

    assert training_artifact_is_complete(summary, tmp_path)


def test_shared_paper_validation_controls_all_vehicles_without_early_termination():
    contract = paper_driver_model_validation_overrides()

    assert contract["validation_vehicle_mode"] == "all"
    assert contract["test_vehicle_mode"] == "all"
    assert contract["evaluation_terminate_when_all_controlled_crashed"] is False
    assert contract["validation_require_exact_horizon"] is False
    assert contract["validation_min_horizon_coverage"] == 0.0
    assert contract["validation_score_crash_metric"] == "vehicle"
    assert contract["evaluation_horizons_seconds"] == "1,5,10,20"
    trainer = (ROOT / "scripts_gail/train_recurrent_bc_policy.py").read_text()
    gail = (ROOT / "scripts_gail/ps_gail/pilot.py").read_text()
    assert "paper_driver_model_validation_overrides()" in trainer
    assert "paper_driver_model_validation_overrides()" in gail


def test_finalizer_requires_all_checkpoints_and_collision_free_test_evaluations(tmp_path):
    policy_root = tmp_path / "policies"
    smoke_root = tmp_path / "activations"
    for domain in ("us", "japanese"):
        for layers in (2, 3):
            for seed in (0, 1, 2):
                relative = Path(domain) / f"recurrent_transformer_{layers}layer" / f"policy_seed_{seed}"
                run_dir = policy_root / relative
                run_dir.mkdir(parents=True)
                summary = {
                    "domain": domain,
                    "transformer_layers": layers,
                    "seed": seed,
                    "metric_capability_passed": True,
                    "rollout_capability_passed": True,
                    "capability_passed": True,
                    "checkpoint_saved": True,
                    "validation_skill": 0.7,
                    "validation_mae": 0.05,
                    "held_out_rollouts": {
                        "bc_eval/crash_episode_fraction": 1.0,
                        "bc_eval/collision_episode_fraction": 0.0,
                        "bc_eval/collision_proxy_episode_fraction": 0.5,
                    },
                    "held_out_evaluation": {
                        "prebuilt_split": "test",
                        "collision_physics_enabled": False,
                        "episodes": 3,
                        "metrics": {
                            "bc_eval/crash_episode_fraction": 1.0,
                            "bc_eval/collision_episode_fraction": 0.0,
                            "bc_eval/collision_proxy_episode_fraction": 0.5,
                        },
                    },
                }
                if domain == "japanese" and layers == 3 and seed == 2:
                    summary["metric_capability_passed"] = False
                    summary["rollout_capability_passed"] = False
                    summary["capability_passed"] = False
                checkpoint = run_dir / "best.pt"
                torch.save(
                    {
                        "policy_state_dict": {"weight": torch.ones(1)},
                        "policy_architecture": {
                            "policy_model": "recurrent_transformer",
                            "transformer_layers": layers,
                        },
                        "config": {"seed": seed},
                    },
                    checkpoint,
                )
                digest = sha256_file(checkpoint)
                summary["checkpoint_saved"] = True
                summary["checkpoint_sha256"] = digest
                (run_dir / "best.pt.sha256").write_text(f"{digest}  best.pt\n")
                for layer in range(layers):
                    manifest = smoke_root / relative / f"residual_layer_{layer}_policy_token" / "manifest.json"
                    manifest.parent.mkdir(parents=True, exist_ok=True)
                    manifest.write_text("{}\n")
                (run_dir / "summary.json").write_text(json.dumps(summary) + "\n")

    result = finalize_study(policy_root, smoke_root, policy_root / "study_manifest.json")

    assert result["model_count"] == 12
    assert result["checkpoint_count"] == 12
    assert result["metric_capability_passed_count"] == 11
    assert result["metric_threshold_not_met_count"] == 1
    assert result["capability_passed_count"] == 11
    assert sum(model["checkpoint"] is not None for model in result["models"]) == 12


def test_submission_uses_one_reporting_serial_bc_job():
    submit = (ROOT / "hpc/slurm/script_full_training/submit_bc_domain_depth_study.bash").read_text()
    runner = (ROOT / "hpc/slurm/script_full_training/run_bc_domain_depth_study.bash").read_text()

    assert submit.count("sbatch --parsable") == 1
    assert "--dependency" not in submit
    assert "#SBATCH --array" not in runner
    assert "scripts_gail.run_bc_domain_depth_matrix" in runner
    assert 'manifest.get("loader_calls") != {"us": 1, "japanese": 1}' in runner
    assert "configs/bc_recovery_recipe.json" in runner
    assert "--checkpoint-archive-root" in runner
    assert 'BC_STUDY_MODEL_LIMIT' in runner
    assert "for domain in us japanese" in runner
    assert "for depth in 2 3" in runner
    assert "for seed in 0 1 2" in runner


def test_aligned_bc_runner_completes_artifacts_and_treats_rollouts_as_descriptive():
    runner = (
        ROOT
        / "hpc/slurm/script_full_training/run_bc_gail_aligned_accel5.bash"
    ).read_text()

    assert "--require-confirmation" not in runner
    assert "--priority-cells-first" in runner
    assert 'payload.get("training_artifact_complete_count", -1)' in runner
    assert 'payload.get("capability_passed_count", -1)' not in runner
    assert '"closed_loop_metrics_role": "descriptive_non_terminal"' in runner
