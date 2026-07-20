from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts_gail.run_bc_domain_depth_matrix as matrix_module
from scripts_gail.run_bc_domain_depth_matrix import (
    cell_args,
    confirmation_first_cells,
    load_and_prepare_domains,
    matrix_cells,
    read_locked_recipe,
)


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


def test_confirmation_cells_run_before_remaining_matrix():
    cells = confirmation_first_cells()
    assert cells[:2] == [("us", 3, 0), ("japanese", 3, 1)]
    assert len(cells) == 12
    assert set(cells) == set(matrix_cells())


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
    assert args.transformer_layers == 3
    assert args.seed == 2
