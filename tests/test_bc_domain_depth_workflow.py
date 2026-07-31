from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.contracts import (
    policy_observation_contract,
    runtime_continuous_action_contract,
)
from scripts_gail.finalize_bc_domain_depth_study import finalize_study, sha256_file
from scripts_gail.train_recurrent_bc_policy import (
    append_jsonl,
    checkpoint_payload,
    collision_physics_evaluation_status,
    evaluation_completion_flags,
    expert_split_provenance,
    final_test_qualification_is_eligible,
    paper_validation_overrides_for_vehicle_mode,
    run_training,
    recurrent_bc_row_sampling_receipt,
    save_checkpoint_artifacts,
    survival_quality_passed,
    training_artifact_is_complete,
    validate_explicit_expert_source_roots,
    validate_recurrent_bc_source_contracts,
    validation_selection_is_eligible,
)
from scripts_gail.train_simple_ps_gail import (
    _collision_only_flags,
    _collision_proxy_flags,
    _unmatched_survival_config,
    _unmatched_survival_env_overrides,
    _unmatched_survival_should_stop,
)
from scripts_gail.ps_gail.envs import make_training_env
from scripts_gail.ps_gail.validation import (
    action_learning_gate,
    closed_loop_policy_quality,
    paper_driver_model_validation_overrides,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_expert_root_manifest(
    root: Path,
    *,
    scene: str,
    split: str,
    episode_names: list[str],
) -> None:
    root.mkdir(parents=True)
    episodes = []
    for index, episode_name in enumerate(episode_names):
        dataset_file = f"episode_{index:04d}_{episode_name}.npz"
        (root / dataset_file).write_bytes(b"fixture")
        episodes.append(
            {
                "episode_name": episode_name,
                "dataset_file": dataset_file,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "prebuilt_split": split,
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )


def test_explicit_source_roots_are_domain_matched_and_episode_disjoint(
    tmp_path,
):
    train = tmp_path / "train"
    validation = tmp_path / "val"
    _write_expert_root_manifest(
        train,
        scene="us-101",
        split="train",
        episode_names=["train_a", "train_b"],
    )
    _write_expert_root_manifest(
        validation,
        scene="us-101",
        split="val",
        episode_names=["val_a"],
    )
    args = SimpleNamespace(
        domain="us",
        scene="us-101",
        expert_data=str(train),
        expert_validation_data=str(validation),
        expert_test_data="",
        test_evaluation_mode="deferred",
    )

    receipt = validate_explicit_expert_source_roots(args)
    assert receipt["status"] == "passed"
    assert receipt["canonical_episode_disjoint"] is True
    assert receipt["test_source_opened"] is False

    args.expert_validation_data = str(train)
    with pytest.raises(ValueError, match="same source"):
        validate_explicit_expert_source_roots(args)

    args.expert_validation_data = str(validation)
    (validation / "manifest.json").write_text(
        json.dumps(
            {
                "scene": "us-101",
                "prebuilt_split": "val",
                "episodes": [
                    {
                        "episode_name": "train_a",
                        "dataset_file": "episode_0000_val_a.npz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical episode identity"):
        validate_explicit_expert_source_roots(args)

    args.scene = "japanese"
    with pytest.raises(ValueError, match="requires scene"):
        validate_explicit_expert_source_roots(args)


def test_explicit_split_provenance_defers_test_without_opening_it(tmp_path):
    train = tmp_path / "train"
    validation = tmp_path / "val"
    test = tmp_path / "test"
    for root in (train, validation, test):
        root.mkdir()
    args = SimpleNamespace(
        expert_data=str(train),
        expert_validation_data=str(validation),
        expert_test_data=str(test),
        test_evaluation_mode="evaluate",
    )

    record = expert_split_provenance(args)

    assert record["method"] == (
        "explicit_collected_train_validation_test_directories"
    )
    assert set(record) == {"method", "train", "validation", "test"}
    args.expert_test_data = ""
    with pytest.raises(ValueError, match="requires --expert-test-data"):
        expert_split_provenance(args)

    args.test_evaluation_mode = "deferred"
    deferred = expert_split_provenance(args)
    assert deferred["method"] == (
        "explicit_collected_train_validation_test_deferred"
    )
    assert deferred["test"] == {"status": "pending_deferred_not_opened"}

    args.expert_test_data = str(test)
    with pytest.raises(ValueError, match="refuses --expert-test-data"):
        expert_split_provenance(args)


def test_collision_reporting_separates_offroad_crashes_and_overlap_proxy():
    assert _collision_only_flags([True, True, False], [True, False, False]) == [True, True, False]
    assert _collision_proxy_flags([{"min_gap": -0.1}, {"min_gap": 2.0}, {}]) == [True, False, False]


def test_unmatched_survival_keeps_collision_and_offroad_independent_at_fixed_horizon():
    assert _unmatched_survival_env_overrides() == {
        "crash_controlled_vehicles_offroad": False,
    }
    source_cfg = PSGAILConfig(terminate_when_all_controlled_crashed=True)
    evaluation_cfg = _unmatched_survival_config(source_cfg)
    assert source_cfg.terminate_when_all_controlled_crashed is True
    assert evaluation_cfg.terminate_when_all_controlled_crashed is False
    inspect.signature(make_training_env).bind(
        evaluation_cfg,
        **_unmatched_survival_env_overrides(),
    )
    assert not _unmatched_survival_should_stop(
        terminated=True,
        truncated=False,
    )
    assert _unmatched_survival_should_stop(
        terminated=False,
        truncated=True,
    )


def test_explicit_validation_contract_is_compared_against_train():
    source = {
        "continuous_action_contract": runtime_continuous_action_contract(),
        "policy_observation_contract": policy_observation_contract(
            lidar_cells=128,
            maximum_range=64.0,
        ),
        "continuous_action_contract_explicit": True,
        "policy_observation_contract_explicit": True,
        "trajectory_id_identity_quality": "scene_and_episode_name",
    }
    metadata = {
        "split_method": "explicit_source_directories_deferred_test",
        "sources": {
            "train": deepcopy(source),
            "validation": deepcopy(source),
        },
    }
    validated = validate_recurrent_bc_source_contracts(
        metadata,
        lidar_cells=128,
        maximum_range=64.0,
        require_explicit=True,
    )
    assert validated["validated_source_splits"] == [
        "train",
        "validation",
    ]
    assert validated["cross_source_contracts_matched"] is True
    assert validated["source_stable_trajectory_identity_passed"] is True

    metadata["sources"]["validation"]["policy_observation_contract"][
        "maximum_range_m"
    ] = 32.0
    with pytest.raises(ValueError, match="validation.*contract is invalid"):
        validate_recurrent_bc_source_contracts(
            metadata,
            lidar_cells=128,
            maximum_range=64.0,
            require_explicit=True,
        )

    metadata["sources"]["validation"] = deepcopy(source)
    metadata["sources"]["validation"][
        "trajectory_id_identity_quality"
    ] = "contains_basename_only_nonqualifying"
    with pytest.raises(ValueError, match="source-stable"):
        validate_recurrent_bc_source_contracts(
            metadata,
            lidar_cells=128,
            maximum_range=64.0,
            require_explicit=True,
        )


def test_row_sampling_receipt_separates_targets_from_actual_overshoot():
    receipt = recurrent_bc_row_sampling_receipt(
        {
            "split_method": "explicit_source_directories_deferred_test",
            "sources": {
                "train": {"num_samples": 305_123},
                "validation": {"num_samples": 101_004},
            },
        },
        requested_train_rows=300_000,
        requested_validation_rows=100_000,
        requested_test_rows=100_000,
    )
    assert receipt["requested_row_targets"]["train"] == 300_000
    assert receipt["actual_loaded_rows"]["train"] == 305_123
    assert "may_overshoot" in receipt["sampling_semantics"]


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


def test_checkpoint_provenance_hashes_closed_loop_action_sources(tmp_path):
    expert = tmp_path / "expert"
    expert.mkdir()
    args = SimpleNamespace(
        checkpoint_purpose="policy",
        policy_head_init_std=0.01,
        split_seed=17,
        expert_data=str(expert),
        expert_validation_data="",
        expert_test_data="",
    )
    cfg = PSGAILConfig(
        policy_model="recurrent_transformer",
        action_mode="continuous",
        continuous_action_dim=2,
        hidden_size=8,
        transformer_layers=2,
        transformer_heads=2,
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
    )
    contract = {
        "continuous_action": {"columns": ["acceleration", "steering"]},
        "policy_observation": {"dimension": 4},
    }
    payload = checkpoint_payload(
        torch.nn.Linear(4, 2),
        cfg,
        args=args,
        obs_dim=4,
        action_dim=2,
        training_summary={
            "checkpoint_selection_rule": "validation_loss",
            "test_evaluation_mode": "deferred",
        },
        training_data_contract=contract,
        split_trajectory_ids={
            "train": ["train"],
            "validation": ["validation"],
            "test": ["test"],
        },
        scenario=("episode", 1),
    )

    source_hashes = payload["provenance"]["source_sha256"]
    assert {
        "recurrent_bc.py",
        "train_recurrent_bc_policy.py",
        "pretrain_continuous_bc_policy.py",
        "train_simple_ps_gail.py",
        "training/evaluation.py",
        "training/policy.py",
        "training/ppo.py",
        "ps_gail/envs.py",
        "models.py",
        "highway_env/envs/common/action.py",
        "highway_env/envs/ngsim_env.py",
    } <= set(source_hashes)
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in source_hashes.values()
    )
    assert payload["bc_stats"]["checkpoint_selection_rule"] == "validation_loss"
    assert payload["bc_stats"]["test_evaluation_mode"] == "deferred"
    assert (
        payload["config"]["policy_observation_standardization_clip"]
        == 0.0
    )
    assert (
        payload["policy_architecture"][
            "policy_observation_standardization_clip"
        ]
        == 0.0
    )


def test_validation_history_is_appended_one_epoch_at_a_time(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"epoch": 1.0, "validation_skill": 0.1, "is_best_so_far": True})
    append_jsonl(path, {"epoch": 2.0, "validation_skill": 0.2, "is_best_so_far": True})

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1.0, 2.0]
    assert [row["validation_skill"] for row in rows] == [0.1, 0.2]


def test_training_artifact_completion_does_not_depend_on_rollout_quality(tmp_path):
    checkpoint = tmp_path / "best.pt"
    summary = {
        "initial_validation_mse": 0.4,
        "validation_mse": 0.1,
        "validation_mae": 0.2,
        "validation_selection_mse": 0.08,
        "validation_skill": 0.75,
        "validation_action_mse": [0.1, 0.06],
        "validation_action_mae": [0.2, 0.1],
        "validation_prediction_std": [0.5, 0.1],
        "validation_target_std": [0.6, 0.2],
        "validation_prediction_std_ratio": [0.8, 0.2],
        "validation_prediction_target_correlation": [0.9, 0.3],
        "validation_prediction_saturation_fraction": [0.0, 0.0],
        "offline_test_evaluated": False,
        "rollout_capability_passed": False,
        "capability_passed": False,
    }
    save_checkpoint_artifacts(
        checkpoint,
        {"policy_state_dict": {"weight": torch.arange(3)}},
        summary,
    )
    (tmp_path / "split_manifest.json").write_text(
        json.dumps(
            {
                "train": ["train:a"],
                "validation": ["validation:b"],
                "test": [],
            }
        )
    )

    assert training_artifact_is_complete(summary, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "short_vector",
        "nonfinite_vector",
        "summary_digest",
        "sidecar_digest",
        "overlapping_split",
        "missing_test_after_evaluation",
    ],
)
def test_training_artifact_completion_rejects_integrity_failures(
    tmp_path,
    mutation,
):
    checkpoint = tmp_path / "best.pt"
    summary = {
        "initial_validation_mse": 0.4,
        "validation_mse": 0.1,
        "validation_mae": 0.2,
        "validation_selection_mse": 0.08,
        "validation_action_mse": [0.1, 0.06],
        "validation_action_mae": [0.2, 0.1],
        "validation_prediction_std": [0.5, 0.1],
        "validation_target_std": [0.6, 0.2],
        "validation_prediction_std_ratio": [0.8, 0.2],
        "validation_prediction_target_correlation": [0.9, 0.3],
        "validation_prediction_saturation_fraction": [0.0, 0.0],
        "offline_test_evaluated": False,
    }
    save_checkpoint_artifacts(
        checkpoint,
        {"policy_state_dict": {"weight": torch.arange(3)}},
        summary,
    )
    manifest = {
        "train": ["train:a"],
        "validation": ["validation:b"],
        "test": [],
    }
    if mutation == "short_vector":
        summary["validation_prediction_std_ratio"] = [0.8]
    elif mutation == "nonfinite_vector":
        summary["validation_action_mse"] = [0.1, float("nan")]
    elif mutation == "summary_digest":
        summary["checkpoint_sha256"] = "0" * 64
    elif mutation == "sidecar_digest":
        (tmp_path / "best.pt.sha256").write_text(
            f"{'0' * 64}  best.pt\n"
        )
    elif mutation == "overlapping_split":
        manifest["validation"] = ["train:a"]
    elif mutation == "missing_test_after_evaluation":
        summary["offline_test_evaluated"] = True
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest))

    assert not training_artifact_is_complete(summary, tmp_path)


def test_split_manifest_uses_source_stable_ids_not_local_file_indices(
    tmp_path,
):
    checkpoint = tmp_path / "best.pt"
    summary = {
        "initial_validation_mse": 0.4,
        "validation_mse": 0.1,
        "validation_mae": 0.2,
        "validation_selection_mse": 0.08,
        "validation_action_mse": [0.1, 0.06],
        "validation_action_mae": [0.2, 0.1],
        "validation_prediction_std": [0.5, 0.1],
        "validation_target_std": [0.6, 0.2],
        "validation_prediction_std_ratio": [0.8, 0.2],
        "validation_prediction_target_correlation": [0.9, 0.3],
        "validation_prediction_saturation_fraction": [0.0, 0.0],
        "offline_test_evaluated": False,
    }
    save_checkpoint_artifacts(
        checkpoint,
        {"policy_state_dict": {"weight": torch.arange(3)}},
        summary,
    )
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train": ["train:us-101/episode-a:7"],
                "validation": ["validation:us-101/episode-b:7"],
                "test": [],
            }
        )
    )
    assert training_artifact_is_complete(summary, tmp_path)

    manifest_path.write_text(
        json.dumps(
            {
                "train": ["train:us-101/episode-a:7"],
                "validation": ["validation:us-101/episode-a:7"],
                "test": [],
            }
        )
    )
    assert not training_artifact_is_complete(summary, tmp_path)

    manifest_path.write_text(
        json.dumps(
            {
                "train": ["train:fallback_basename/train.npz:7"],
                "validation": [
                    "validation:fallback_basename/validation.npz:8"
                ],
                "test": [],
            }
        )
    )
    assert not training_artifact_is_complete(summary, tmp_path)


def test_run_training_refuses_nonempty_output_without_resume(tmp_path):
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    (out_dir / "metrics.jsonl").write_text("interrupted progress\n")
    args = SimpleNamespace(
        test_evaluation_mode="deferred",
        matched_evaluation=False,
        evaluation_split="val",
        checkpoint_purpose="policy",
        epochs=1,
        max_warmup_epochs=5,
        out_dir=str(out_dir),
    )

    with pytest.raises(FileExistsError, match="no resume protocol"):
        run_training(args)
    assert (out_dir / "metrics.jsonl").read_text() == "interrupted progress\n"


def test_deferred_final_collision_status_is_not_reported_as_executed():
    enabled, status = collision_physics_evaluation_status(
        configured=True,
        evaluated=False,
    )
    assert enabled is None
    assert status == "configured_not_evaluated"
    assert collision_physics_evaluation_status(
        configured=True,
        evaluated=True,
    ) == (True, "enabled")


def test_validation_selection_requires_collision_physics_not_collision_absence():
    common = {
        "training_artifact_complete": True,
        "metric_capability_passed": True,
        "validation_rollout_capability_passed": True,
        "validation_contract_passed": True,
    }
    assert not validation_selection_is_eligible(
        **common,
        collision_physics_enabled=False,
    )
    assert validation_selection_is_eligible(
        **common,
        collision_physics_enabled=True,
    )


def test_final_test_qualification_requires_collision_physics():
    common = {
        "test_evaluation_enabled": True,
        "held_out_metric_capability_passed": True,
        "rollout_capability_passed": True,
        "test_contract_passed": True,
    }
    assert not final_test_qualification_is_eligible(
        **common,
        collision_physics_enabled=False,
    )
    assert final_test_qualification_is_eligible(
        **common,
        collision_physics_enabled=True,
    )


def test_survival_quality_gates_collision_and_offroad_separately():
    stats = {
        "bc_eval/mean_episode_length": 200.0,
        # Raw crash includes legacy off-road mutation and must not be gated.
        "bc_eval/crash_episode_fraction": 1.0,
        "bc_eval/collision_episode_fraction": 0.0,
        "bc_eval/offroad_episode_fraction": 0.0,
    }
    assert survival_quality_passed(
        stats,
        min_rollout_steps=100,
        max_collision_fraction=0.05,
        max_offroad_fraction=0.05,
    )
    stats["bc_eval/collision_episode_fraction"] = 0.1
    assert not survival_quality_passed(
        stats,
        min_rollout_steps=100,
        max_collision_fraction=0.05,
        max_offroad_fraction=0.05,
    )


def test_unmatched_completion_never_claims_matched_evaluation():
    flags = evaluation_completion_flags(
        matched_evaluation=False,
        matched_validation_evaluated=False,
        matched_test_evaluated=False,
        validation_survival_evaluated=True,
        test_survival_evaluated=False,
    )

    assert flags == {
        "matched_evaluation_complete": False,
        "validation_survival_evaluation_complete": True,
        "test_survival_evaluation_complete": False,
    }


@pytest.mark.parametrize("mode", ["single", "training_count", "all"])
def test_requested_vehicle_mode_overrides_shared_paper_default(mode):
    contract = paper_validation_overrides_for_vehicle_mode(mode)

    assert contract["validation_vehicle_mode"] == mode
    assert contract["test_vehicle_mode"] == mode
    assert contract["evaluation_horizons_seconds"] == "1,5,10,20"


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


def test_finite_matched_rollout_is_not_automatically_quality_passed():
    metrics = {
        "test/vehicle_crash_rate": 0.82,
        "test/vehicle_offroad_rate": 0.04,
        "test/horizon_coverage_20s": 0.91,
    }

    result = closed_loop_policy_quality(
        metrics,
        prefix="test",
        max_vehicle_crash_rate=0.34,
        max_vehicle_offroad_rate=0.34,
        score_horizon_seconds=20,
    )

    assert result["passed"] is False
    assert result["checks"]["vehicle_crash_rate"] is False
    assert result["checks"]["vehicle_offroad_rate"] is True
    assert result["observed"]["vehicle_crash_rate"] == 0.82


def test_held_out_steering_failure_rejects_offline_action_gate():
    gate = action_learning_gate(
        split="test",
        prediction_std_ratios=[0.88, 0.13],
        prediction_target_correlations=[0.90, -0.01],
        action_indices=[0, 1],
        minimum_std_ratios=[0.25, 0.10],
        minimum_correlations=[0.50, 0.20],
    )

    assert gate["passed"] is False
    assert gate["actions"][0]["passed"] is True
    assert gate["actions"][1]["passed"] is False


def test_new_bc_recipes_require_expert_replay_before_policy_realism():
    for name in (
        "bc_simple_gru_control_v1.json",
        "bc_gail_aligned_accel5_v5.json",
    ):
        recipe = json.loads(
            (ROOT / "configs" / name).read_text(encoding="utf-8")
        )
        replay_gate = recipe["evaluation"]["expert_replay_qualification"]

        assert replay_gate["required_for_policy_realism"] is True
        assert replay_gate["status"].startswith("pending_full_")
        assert replay_gate["maximum_vehicle_crash_rate_gap"] == 0.05
        assert replay_gate["maximum_vehicle_offroad_rate_gap"] == 0.05
        assert recipe["evaluation"]["test_evaluation_mode"] == "deferred"
        assert recipe["evaluation"]["split"] == "val"
        assert recipe["evaluation"]["vehicle_mode"] == "single"
        assert recipe["evaluation"]["validation_episodes"] == 12
        assert recipe["evaluation"]["max_crash_fraction"] == 1.0
        assert recipe["evaluation"]["max_offroad_fraction"] == 1.0
        assert recipe["evaluation"]["engineering_alert_crash_fraction"] == 0.05
        assert recipe["evaluation"]["absolute_outcome_gate_role"] == (
            "engineering_alert_non_publication_gate"
        )
        assert recipe["architecture"]["policy_seeds"] == [0, 1, 2, 3, 4]
        assert recipe["optimization"]["checkpoint_selection_rule"] == (
            "validation_loss"
        )
        assert recipe["optimization"]["early_stopping_min_epochs"] == 40


def test_prospective_repaired_recipes_disable_clip_and_label_row_targets():
    for name in (
        "bc_simple_gru_control_v2.json",
        "bc_gail_aligned_accel5_v6.json",
    ):
        recipe = json.loads(
            (ROOT / "configs" / name).read_text(encoding="utf-8")
        )
        assert (
            recipe["architecture"][
                "policy_observation_standardization_clip"
            ]
            == 0.0
        )
        assert recipe["data"]["row_budget_semantics"] == (
            "trajectory_preserving_targets_may_overshoot"
        )
        assert recipe["data"]["trajectory_identity_requirement"] == (
            "scene_and_episode_name_plus_vehicle_id"
        )


def test_finalizer_requires_collision_enabled_qualified_locked_test(tmp_path):
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
                    "final_test_qualification_passed": True,
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
                        "collision_physics_enabled": True,
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
    assert '"test": 0' in runner
    assert "Explicit sources were not loaded exactly once" in runner
    assert "configs/bc_gail_aligned_accel5_v5.json" in runner
    assert "--checkpoint-archive-root" in runner
    assert 'BC_STUDY_MODEL_LIMIT' in runner
    assert 'manifest.get("recipe_depths", [])' in runner
    assert 'manifest.get("recipe_policy_seeds", [])' in runner
    assert "for depth in 2 3" not in runner
    assert "for seed in 0 1 2" not in runner


def test_aligned_bc_runner_completes_independent_cells_without_ignoring_quality():
    runner = (
        ROOT
        / "hpc/slurm/script_full_training/run_bc_gail_aligned_accel5.bash"
    ).read_text()

    assert "--require-confirmation" not in runner
    assert "--priority-cells-first" in runner
    assert 'payload.get("training_artifact_complete_count", -1)' in runner
    assert 'payload.get("capability_passed_count", -1)' not in runner
    assert (
        '"closed_loop_metrics_role": '
        '"qualification_non_terminal_for_matrix_execution"'
    ) in runner
    assert '"status": "artifact_matrix_completed"' in runner
    assert (
        '"scientific_qualification_status": '
        '"pending_locked_test_and_expert_replay"'
    ) in runner
    assert '"interpretability_qualification_passed": False' in runner
    assert "paper_benchmark_selected" not in runner
    assert "--us-train-expert" in runner
    assert "--us-validation-expert" in runner
    assert "--japanese-train-expert" in runner
    assert "--japanese-validation-expert" in runner
    assert "pending_deferred_not_accepted" in runner


def test_matrix_submit_has_no_historical_promotion_precondition():
    submit = (
        ROOT
        / "hpc/slurm/script_full_training/submit_bc_shared_transformer_matrix.bash"
    ).read_text()

    assert "promotion_passed" not in submit
    assert "US_QUALIFICATION" not in submit
    assert "JAPANESE_QUALIFICATION" not in submit
    assert '"depths": [2]' in submit
    assert '"policy_seeds": [0, 1, 2, 3, 4]' in submit
    assert "tests/test_ps_gail_training_logic.py" in submit
    assert "tests/test_bc_domain_depth_workflow.py" in submit
