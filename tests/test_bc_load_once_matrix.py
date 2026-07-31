from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts_gail.run_bc_domain_depth_matrix as matrix_module
from scripts_gail.train_recurrent_bc_policy import make_config
from scripts_gail.run_bc_domain_depth_matrix import (
    cell_args,
    load_and_prepare_domains,
    matrix_cells,
    priority_cells_first,
    read_locked_recipe,
    recipe_matrix_axes,
    select_validation_candidate_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]


def test_matrix_axes_come_from_locked_depth2_five_seed_recipe():
    recipe = json.loads(
        (ROOT / "configs/bc_gail_aligned_accel5_v5.json").read_text()
    )
    depths, seeds = recipe_matrix_axes(recipe)
    cells = matrix_cells(depths=depths, seeds=seeds)
    assert depths == (2,)
    assert seeds == (0, 1, 2, 3, 4)
    assert len(cells) == 10
    assert len(set(cells)) == 10
    assert {cell[0] for cell in cells} == {"us", "japanese"}
    assert {cell[1] for cell in cells} == {2}
    assert {cell[2] for cell in cells} == {0, 1, 2, 3, 4}
    with pytest.raises(ValueError):
        matrix_cells(depths=depths, seeds=seeds, limit=11)


def test_matrix_requires_explicitly_locked_recipe(tmp_path: Path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "status": "draft"}) + "\n")
    with pytest.raises(ValueError, match="locked supported schema"):
        read_locked_recipe(recipe)


def test_matrix_accepts_current_locked_schema_v2_recipe():
    recipe = read_locked_recipe(
        ROOT / "configs/bc_gail_aligned_accel5_v6.json"
    )
    assert recipe["schema_version"] == 2
    assert recipe["status"] == "locked"


def test_priority_cells_change_order_without_changing_matrix():
    cells = priority_cells_first(depths=(2,), seeds=(0, 1, 2, 3, 4))
    assert cells[:2] == [("us", 2, 0), ("japanese", 2, 0)]
    assert len(cells) == 10
    assert set(cells) == set(
        matrix_cells(depths=(2,), seeds=(0, 1, 2, 3, 4))
    )


def test_simple_gru_recipe_uses_one_depth_and_five_seeds_per_domain():
    cells = matrix_cells(depths=(1,), seeds=(0, 1, 2, 3, 4))
    assert len(cells) == 10
    assert {depth for _domain, depth, _seed in cells} == {1}
    priority = priority_cells_first(
        depths=(1,),
        seeds=(0, 1, 2, 3, 4),
    )
    assert priority[:2] == [("us", 1, 0), ("japanese", 1, 0)]


def test_each_explicit_train_validation_source_is_loaded_once_without_test(
    monkeypatch,
    tmp_path: Path,
):
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(policy_observations=[0, 1])

    def fake_prepare(transitions, **_kwargs):
        return ("prepared", transitions)

    monkeypatch.setattr(matrix_module, "load_expert_transition_data", fake_load)
    monkeypatch.setattr(
        matrix_module,
        "prepare_recurrent_bc_data_from_explicit_splits",
        fake_prepare,
    )
    recipe = {
        "data": {
            "max_expert_samples": 300000,
            "max_validation_samples": 100000,
            "data_seed": 7,
            "split_seed": 8,
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
        },
        "optimization": {"sequence_length": 32},
        "architecture": {"memory_context_length": 32},
        "evaluation": {"test_evaluation_mode": "deferred"},
    }
    transitions, prepared, loader_calls = load_and_prepare_domains(
        {
            "us": {
                "train": tmp_path / "us_train",
                "validation": tmp_path / "us_val",
            },
            "japanese": {
                "train": tmp_path / "jp_train",
                "validation": tmp_path / "jp_val",
            },
        },
        recipe,
    )
    assert loader_calls == {
        "us": {"train": 1, "validation": 1, "test": 0},
        "japanese": {"train": 1, "validation": 1, "test": 0},
    }
    assert len(calls) == 4
    assert [call[1]["max_samples"] for call in calls] == [
        300000,
        100000,
        300000,
        100000,
    ]
    assert [call[1]["seed"] for call in calls] == [7, 8, 7, 8]
    assert set(transitions) == {"us", "japanese"}
    assert all(
        set(transitions[domain]) == {"train", "validation"}
        for domain in transitions
    )
    assert set(prepared) == {"us", "japanese"}


def test_matrix_loader_rejects_test_source_before_any_load(
    monkeypatch,
    tmp_path: Path,
):
    calls: list[str] = []
    monkeypatch.setattr(
        matrix_module,
        "load_expert_transition_data",
        lambda path, **_kwargs: calls.append(path),
    )
    recipe = {
        "data": {"max_expert_samples": 1, "data_seed": 1},
        "optimization": {"sequence_length": 1},
        "architecture": {"memory_context_length": 1},
        "evaluation": {"test_evaluation_mode": "deferred"},
    }
    paths = {
        domain: {
            "train": tmp_path / f"{domain}_train",
            "validation": tmp_path / f"{domain}_val",
        }
        for domain in ("us", "japanese")
    }
    paths["us"]["test"] = tmp_path / "sealed_test"

    with pytest.raises(ValueError, match="never test"):
        load_and_prepare_domains(paths, recipe)

    assert calls == []


def test_paper_metric_seed_selection_uses_validation_not_final_test():
    records = [
        {
            "domain": "us",
            "transformer_layers": 2,
            "seed": 0,
            "relative_path": "seed0",
            "checkpoint": "seed0/best.pt",
            "checkpoint_sha256": "a",
            "training_artifact_complete": True,
            "validation_selection_eligible": False,
            "interpretability_baseline_eligible": False,
            "paper_validation_cost": 1.0,
            "paper_validation_score": -1.0,
        },
        {
            "domain": "us",
            "transformer_layers": 2,
            "seed": 1,
            "relative_path": "seed1",
            "checkpoint": "seed1/best.pt",
            "checkpoint_sha256": "b",
            "training_artifact_complete": True,
            "validation_selection_eligible": True,
            "interpretability_baseline_eligible": False,
            "paper_validation_cost": 2.0,
            "paper_validation_score": -2.0,
        },
        {
            "domain": "us",
            "transformer_layers": 2,
            "seed": 2,
            "relative_path": "seed2",
            "checkpoint": "seed2/best.pt",
            "checkpoint_sha256": "c",
            "training_artifact_complete": True,
            "validation_selection_eligible": True,
            "interpretability_baseline_eligible": True,
            "paper_validation_cost": 3.0,
            "paper_validation_score": -3.0,
        },
    ]

    selections = select_validation_candidate_checkpoints(records)

    assert len(selections) == 1
    assert selections[0]["seed"] == 1
    assert selections[0]["selection_tier"] == (
        "validation_capable_pending_locked_test"
    )
    assert records[1]["validation_candidate_selected"] is True


def test_artifact_fallback_is_not_promoted_as_validation_candidate():
    records = [
        {
            "domain": "us",
            "transformer_layers": 2,
            "seed": 0,
            "relative_path": "seed0",
            "checkpoint": "seed0/best.pt",
            "checkpoint_sha256": "a",
            "training_artifact_complete": True,
            "validation_selection_eligible": False,
            "paper_validation_cost": 1.0,
            "paper_validation_score": -1.0,
        }
    ]

    selections = select_validation_candidate_checkpoints(records)

    assert selections == []
    assert records[0].get("validation_candidate_selected") is not True
    assert records[0]["development_fallback_selected"] is True
    assert records[0]["development_fallback_reason"] == (
        "no_validation_eligible_candidate"
    )


def test_locked_recipe_values_reach_each_training_cell(tmp_path: Path):
    recipe = {
        "architecture": {
            "hidden_size": 256,
            "transformer_heads": 4,
            "transformer_dropout": 0.1,
            "transformer_norm_first": False,
            "policy_head_init_std": 0.05,
            "memory_tokens": 8,
            "memory_context_length": 32,
        },
        "optimization": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "epochs": 30,
            "early_stopping_patience": 0,
            "early_stopping_min_epochs": 20,
            "early_stopping_min_delta_relative": 0.002,
            "checkpoint_selection_rule": "validation_loss",
            "mirror_augmentation_probability": 0.5,
            "sequence_length": 32,
            "sequences_per_batch": 16,
            "micro_batch_sequences": 4,
            "action_loss_weights": [1.0, 1.0],
            "action_loss_weighting": "inverse_variance",
        },
        "data": {
            "max_expert_samples": 300000,
            "data_seed": 7,
            "split_seed": 8,
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
        },
        "gates": {
            "min_validation_skill": 0.1,
            "max_validation_mae": 0.35,
            "learning_action_index": 0,
            "min_learning_action_std_ratio": 0.25,
            "min_learning_action_correlation": 0.5,
        },
        "evaluation": {
            "test_evaluation_mode": "deferred",
            "episodes": 3,
            "min_rollout_steps": 100,
            "max_crash_fraction": 0.34,
            "max_offroad_fraction": 0.34,
        },
    }
    args = cell_args(
        recipe,
        domain="us",
        depth=3,
        seed=2,
        expert_train_data=tmp_path / "train",
        expert_validation_data=tmp_path / "validation",
        episode_root=tmp_path / "episodes",
        out_dir=tmp_path / "out",
        device="cuda",
    )
    assert args.learning_rate == 0.001
    assert args.weight_decay == 0.0
    assert args.max_grad_norm == 1.0
    assert args.policy_head_init_std == 0.05
    assert args.transformer_dropout == 0.1
    assert args.epochs == 30
    assert args.micro_batch_sequences == 4
    assert args.action_loss_weighting == "inverse_variance"
    assert args.early_stopping_min_epochs == 20
    assert args.early_stopping_min_delta_relative == pytest.approx(0.002)
    assert args.checkpoint_selection_rule == "validation_loss"
    assert args.mirror_augmentation_probability == 0.5
    assert args.test_evaluation_mode == "deferred"
    assert args.expert_data == str(tmp_path / "train")
    assert args.expert_validation_data == str(tmp_path / "validation")
    assert args.expert_test_data == ""
    assert args.evaluation_split == "val"
    assert args.evaluation_vehicle_mode == "single"
    assert args.transformer_layers == 3
    assert args.seed == 2
    cfg = make_config(args)
    assert cfg.validation_vehicle_mode == "single"
    assert cfg.test_vehicle_mode == "single"


def test_closed_loop_failure_does_not_stop_remaining_matrix(
    monkeypatch,
    tmp_path: Path,
):
    policy_root = tmp_path / "policies"
    training_calls: list[tuple[str, int, int]] = []
    fake_transitions = {
        domain: {
            "train": SimpleNamespace(policy_observations=[0, 1]),
            "validation": SimpleNamespace(policy_observations=[0]),
        }
        for domain in ("us", "japanese")
    }
    monkeypatch.setattr(
        matrix_module,
        "parse_args",
        lambda: SimpleNamespace(
            recipe=ROOT / "configs/bc_gail_aligned_accel5_v5.json",
            us_train_expert=tmp_path / "us_train",
            us_validation_expert=tmp_path / "us_val",
            japanese_train_expert=tmp_path / "japanese_train",
            japanese_validation_expert=tmp_path / "japanese_val",
            episode_root=tmp_path / "episodes",
            policy_root=policy_root,
            checkpoint_archive_root=tmp_path / "archive",
            study_id="non_terminal_rollout_test",
            model_limit=2,
            skip_archive=False,
            priority_cells_first=False,
            device="cpu",
        ),
    )
    monkeypatch.setattr(
        matrix_module,
        "load_and_prepare_domains",
        lambda _paths, _recipe: (
            fake_transitions,
            {
                "us": SimpleNamespace(transitions=object()),
                "japanese": SimpleNamespace(transitions=object()),
            },
            {
                "us": {"train": 1, "validation": 1, "test": 0},
                "japanese": {"train": 1, "validation": 1, "test": 0},
            },
        ),
    )

    def fake_run_training(args, **_kwargs):
        training_calls.append(
            (str(args.domain), int(args.transformer_layers), int(args.seed))
        )
        run_dir = Path(args.out_dir)
        run_dir.mkdir(parents=True)
        return {
            "checkpoint": str(run_dir / "best.pt"),
            "checkpoint_sha256": "digest",
            "training_artifact_complete": True,
            "interpretability_baseline_eligible": True,
            "metric_capability_passed": True,
            "paper_validation_cost": 1.0,
            "paper_validation_score": -1.0,
            "closed_loop_evaluation_complete": True,
            "rollout_capability_passed": False,
            "matched_evaluation_passed": False,
            "closed_loop_quality_passed": False,
            "capability_passed": False,
        }

    monkeypatch.setattr(matrix_module, "run_training", fake_run_training)
    monkeypatch.setattr(
        matrix_module,
        "archive_bc_checkpoints",
        lambda sources, **_kwargs: {"checkpoint_count": len(sources)},
    )

    matrix_module.main()

    manifest = json.loads(
        (policy_root / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    assert len(training_calls) == 2
    assert manifest["model_count"] == 2
    assert manifest["training_artifact_complete_count"] == 2
    assert manifest["capability_passed_count"] == 0
    assert manifest["test_source_status"] == "pending_deferred_not_accepted"
    assert manifest["split_method"] == (
        "explicit_independent_train_validation_sources"
    )
