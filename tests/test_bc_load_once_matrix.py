from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts_gail.run_bc_domain_depth_matrix as matrix_module
from scripts_gail.run_bc_domain_depth_matrix import (
    cell_args,
    load_and_prepare_domains,
    matrix_cells,
    priority_cells_first,
    read_locked_recipe,
    select_paper_benchmark_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]


def test_matrix_has_exactly_twelve_unique_controlled_cells():
    cells = matrix_cells()
    assert len(cells) == 12
    assert len(set(cells)) == 12
    assert {cell[0] for cell in cells} == {"us", "japanese"}
    assert {cell[1] for cell in cells} == {2, 3}
    assert {cell[2] for cell in cells} == {0, 1, 2}
    with pytest.raises(ValueError):
        matrix_cells(13)


def test_matrix_requires_explicitly_locked_recipe(tmp_path: Path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema_version": 1, "status": "draft"}) + "\n")
    with pytest.raises(ValueError, match="not a locked"):
        read_locked_recipe(recipe)


def test_priority_cells_change_order_without_changing_matrix():
    cells = priority_cells_first()
    assert cells[:2] == [("us", 3, 0), ("japanese", 3, 0)]
    assert len(cells) == 12
    assert set(cells) == set(matrix_cells())


def test_simple_gru_recipe_uses_one_depth_and_six_seed_cells():
    cells = matrix_cells(6, depths=(1,))
    assert len(cells) == 6
    assert {depth for _domain, depth, _seed in cells} == {1}
    priority = priority_cells_first(6, depths=(1,))
    assert priority[:2] == [("us", 1, 0), ("japanese", 1, 0)]


def test_each_domain_loader_is_called_once(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fake_load(path, **_kwargs):
        calls.append(path)
        return object()

    def fake_prepare(transitions, **_kwargs):
        return ("prepared", transitions)

    monkeypatch.setattr(matrix_module, "load_expert_transition_data", fake_load)
    monkeypatch.setattr(matrix_module, "prepare_recurrent_bc_data", fake_prepare)
    recipe = {
        "data": {
            "max_expert_samples": 300000,
            "data_seed": 7,
            "split_seed": 8,
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
        },
        "optimization": {"sequence_length": 32},
        "architecture": {"memory_context_length": 32},
    }
    transitions, prepared, loader_calls = load_and_prepare_domains(
        {"us": tmp_path / "us", "japanese": tmp_path / "jp"},
        recipe,
    )
    assert loader_calls == {"us": 1, "japanese": 1}
    assert len(calls) == 2
    assert set(transitions) == {"us", "japanese"}
    assert set(prepared) == {"us", "japanese"}


def test_paper_metric_selection_prefers_capable_seed_then_validation_score():
    records = [
        {
            "domain": "us",
            "transformer_layers": 2,
            "seed": 0,
            "relative_path": "seed0",
            "checkpoint": "seed0/best.pt",
            "checkpoint_sha256": "a",
            "training_artifact_complete": True,
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
            "interpretability_baseline_eligible": True,
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
            "interpretability_baseline_eligible": True,
            "paper_validation_cost": 3.0,
            "paper_validation_score": -3.0,
        },
    ]

    selections = select_paper_benchmark_checkpoints(records)

    assert len(selections) == 1
    assert selections[0]["seed"] == 1
    assert selections[0]["selection_tier"] == "offline_capable"


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
        expert_data=tmp_path / "expert",
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
    assert args.transformer_layers == 3
    assert args.seed == 2


def test_closed_loop_failure_does_not_stop_remaining_matrix(
    monkeypatch,
    tmp_path: Path,
):
    policy_root = tmp_path / "policies"
    training_calls: list[tuple[str, int, int]] = []
    fake_transitions = {
        domain: SimpleNamespace(policy_observations=[0, 1])
        for domain in ("us", "japanese")
    }
    monkeypatch.setattr(
        matrix_module,
        "parse_args",
        lambda: SimpleNamespace(
            recipe=ROOT / "configs/bc_gail_aligned_accel5_v4.json",
            us_expert=tmp_path / "us",
            japanese_expert=tmp_path / "japanese",
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
            {"us": object(), "japanese": object()},
            {"us": 1, "japanese": 1},
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
