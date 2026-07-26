from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from scripts_gail.ps_gail.checkpoints import (
    assert_policy_architecture_matches_checkpoint,
    shared_interpretable_transformer_architecture,
)
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.recurrent_bc import (
    _batched_sequence_loss,
    _center_sequence_steps,
    build_sequence_windows,
    prepare_recurrent_bc_data,
    split_trajectory_ids,
    train_recurrent_behavior_clone,
    trajectory_segments,
)
from scripts_gail.ps_gail.training.policy import (
    _make_policy_from_state_dict,
    fit_policy_observation_normalizer,
)
from scripts_gail.train_simple_ps_gail import _policy_action_tuple


def test_temporal_centering_removes_only_per_sequence_offsets():
    base = [
        [torch.tensor([1.0, 3.0]), torch.tensor([2.0, 5.0])],
        [torch.tensor([-2.0, 4.0]), torch.tensor([1.0, 2.0])],
    ]
    offset = [
        [step + torch.tensor([100.0, -20.0]) for step in base[0]],
        [step + torch.tensor([-70.0, 40.0]) for step in base[1]],
    ]
    centered = _center_sequence_steps(base)
    torch.testing.assert_close(_center_sequence_steps(offset), centered)
    torch.testing.assert_close(
        centered,
        torch.tensor(
            [
                [-0.5, -1.0],
                [0.5, 1.0],
                [-1.5, 1.0],
                [1.5, -1.0],
            ]
        ),
    )


def test_simple_recurrent_gru_uses_shared_memory_contract():
    policy = make_actor_critic(
        "recurrent_gru",
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_observation_normalization=True,
        transformer_memory_context_length=3,
    )
    memory = policy.initial_memory(2, device=torch.device("cpu"))
    actions, values, next_memory = policy(
        torch.randn(2, 4),
        memory=memory,
        return_memory=True,
    )
    assert actions.shape == (2, 2)
    assert values.shape == (2,)
    assert next_memory.shape == (2, 1, 8)
    actions.square().mean().backward()
    assert policy.memory_gru.weight_hh.grad is not None


@pytest.mark.parametrize("layers", [2, 3])
def test_dense_temporal_recurrent_transformer_is_depth_explicit_and_trainable(
    layers: int,
):
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=12,
        hidden_size=16,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=layers,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_norm_first=True,
        transformer_observation_normalization=True,
        transformer_observation_tokenization="dense_temporal",
        transformer_memory_tokens=1,
        transformer_memory_context_length=4,
    )
    policy.set_observation_normalizer(
        torch.zeros(12),
        torch.ones(12),
    )
    observations = torch.randn(3, 12)
    current = policy._build_current_tokens(observations)
    actions, values, next_memory = policy(
        observations,
        memory=policy.initial_memory(3),
        return_memory=True,
    )
    assert current.shape == (3, 1, 16)
    assert actions.shape == (3, 2)
    assert values.shape == (3,)
    assert next_memory.shape == (3, 1, 16)
    assert len(policy.encoder.layers) == layers
    actions.square().mean().backward()
    assert policy.dense_observation_proj[0].weight.grad is not None
    assert policy.encoder.layers[-1].self_attn.in_proj_weight.grad is not None


def test_locked_recipe_maps_to_one_shared_bc_gail_actor_contract():
    recipe = {
        "status": "locked",
        "architecture_contract_id": (
            "shared_dense_temporal_recurrent_transformer_v1"
        ),
        "architecture": {
            "policy_model": "recurrent_transformer",
            "depths": [2, 3],
            "hidden_size": 256,
            "transformer_heads": 4,
            "transformer_dropout": 0.0,
            "transformer_norm_first": True,
            "transformer_observation_normalization": True,
            "transformer_observation_tokenization": "dense_temporal",
            "policy_head_init_std": 0.01,
            "memory_tokens": 1,
            "memory_context_length": 32,
        },
    }
    depth2 = shared_interpretable_transformer_architecture(recipe, depth=2)
    depth3 = shared_interpretable_transformer_architecture(recipe, depth=3)
    assert {
        key: value
        for key, value in depth2.items()
        if key != "transformer_layers"
    } == {
        key: value
        for key, value in depth3.items()
        if key != "transformer_layers"
    }
    assert depth2["transformer_layers"] == 2
    assert depth3["transformer_layers"] == 3

    torch.manual_seed(17)
    policy = make_actor_critic(
        depth2["policy_model"],
        obs_dim=12,
        hidden_size=depth2["hidden_size"],
        action_mode=depth2["action_mode"],
        continuous_action_dim=depth2["continuous_action_dim"],
        transformer_layers=depth2["transformer_layers"],
        transformer_heads=depth2["transformer_heads"],
        transformer_dropout=depth2["transformer_dropout"],
        transformer_norm_first=depth2["transformer_norm_first"],
        transformer_observation_normalization=depth2[
            "transformer_observation_normalization"
        ],
        transformer_observation_tokenization=depth2[
            "transformer_observation_tokenization"
        ],
        policy_head_init_std=depth2["policy_head_init_std"],
        transformer_memory_tokens=depth2["transformer_memory_tokens"],
        transformer_memory_context_length=depth2[
            "transformer_memory_context_length"
        ],
    )
    assert float(policy.policy_head.weight.std()) == pytest.approx(
        0.01,
        abs=0.002,
    )


def test_bc_evaluation_action_helper_preserves_full_recurrent_context():
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=322,
        hidden_size=16,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=2,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_norm_first=True,
        transformer_observation_tokenization="dense_temporal",
        transformer_memory_tokens=1,
        transformer_memory_context_length=4,
    ).eval()
    cfg = PSGAILConfig(
        action_mode="continuous",
        continuous_action_dim=2,
        policy_model="recurrent_transformer",
        hidden_size=16,
        transformer_layers=2,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_norm_first=True,
        transformer_observation_tokenization="dense_temporal",
        transformer_memory_tokens=1,
        transformer_memory_context_length=4,
    )
    raw_observations = np.zeros((2, 323), dtype=np.float32)
    initial = policy.initial_memory(2, dtype=torch.float32)
    with torch.no_grad():
        policy_observations = torch.as_tensor(
            np.concatenate(
                [
                    raw_observations[:, :-4],
                    raw_observations[:, -1:],
                    raw_observations[:, -4:-2],
                ],
                axis=1,
            )
        )
        _action, _value, first_step = policy(
            policy_observations,
            memory=initial,
            return_memory=True,
        )
    _actions, first_context = _policy_action_tuple(
        policy,
        raw_observations,
        object(),
        device=torch.device("cpu"),
        deterministic=True,
        cfg=cfg,
        memory=initial,
        return_memory=True,
    )
    assert first_context is not None
    assert first_context.shape == (2, 4, 1, 16)
    torch.testing.assert_close(first_context[:, :-1], initial[:, 1:])
    torch.testing.assert_close(first_context[:, -1], first_step)

    _actions, second_context = _policy_action_tuple(
        policy,
        raw_observations,
        object(),
        device=torch.device("cpu"),
        deterministic=True,
        cfg=cfg,
        memory=first_context,
        return_memory=True,
    )
    assert second_context is not None
    assert second_context.shape == first_context.shape
    torch.testing.assert_close(second_context[:, :-1], first_context[:, 1:])


def test_bc_to_gail_architecture_contract_rejects_tokenization_or_depth_drift():
    cfg = PSGAILConfig(
        action_mode="continuous",
        policy_model="recurrent_transformer",
        hidden_size=256,
        transformer_layers=3,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_norm_first=True,
        transformer_observation_normalization=True,
        transformer_observation_tokenization="dense_temporal",
        transformer_memory_tokens=1,
        transformer_memory_context_length=32,
    )
    checkpoint = {"config": vars(cfg)}
    contract = assert_policy_architecture_matches_checkpoint(cfg, checkpoint)
    assert contract["transformer_layers"] == 3
    assert contract["transformer_observation_tokenization"] == "dense_temporal"

    mismatched = {"config": {**vars(cfg), "transformer_layers": 2}}
    with pytest.raises(RuntimeError, match="transformer_layers"):
        assert_policy_architecture_matches_checkpoint(cfg, mismatched)

    mismatched = {
        "config": {
            **vars(cfg),
            "transformer_observation_tokenization": "semantic",
        }
    }
    with pytest.raises(
        RuntimeError,
        match="transformer_observation_tokenization",
    ):
        assert_policy_architecture_matches_checkpoint(cfg, mismatched)


def test_gail_worker_rebuilds_the_same_dense_temporal_transformer():
    cfg = PSGAILConfig(
        action_mode="continuous",
        continuous_action_dim=2,
        policy_model="recurrent_transformer",
        hidden_size=16,
        transformer_layers=3,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_norm_first=True,
        transformer_observation_normalization=True,
        transformer_observation_tokenization="dense_temporal",
        transformer_memory_tokens=1,
        transformer_memory_context_length=4,
    )
    source = make_actor_critic(
        cfg.policy_model,
        12,
        cfg.hidden_size,
        action_mode=cfg.action_mode,
        continuous_action_dim=cfg.continuous_action_dim,
        transformer_layers=cfg.transformer_layers,
        transformer_heads=cfg.transformer_heads,
        transformer_dropout=cfg.transformer_dropout,
        transformer_norm_first=cfg.transformer_norm_first,
        transformer_observation_normalization=(
            cfg.transformer_observation_normalization
        ),
        transformer_observation_tokenization=(
            cfg.transformer_observation_tokenization
        ),
        transformer_memory_tokens=cfg.transformer_memory_tokens,
        transformer_memory_context_length=cfg.transformer_memory_context_length,
    )
    source.set_observation_normalizer(
        torch.linspace(-1.0, 1.0, 12),
        torch.linspace(0.5, 1.5, 12),
    )
    rebuilt = _make_policy_from_state_dict(
        source.state_dict(),
        cfg,
        policy_obs_dim=12,
        critic_obs_dim=12,
        device=torch.device("cpu"),
    )
    assert rebuilt.observation_tokenization == "dense_temporal"
    assert rebuilt.memory_tokens == 1
    assert len(rebuilt.encoder.layers) == 3
    torch.testing.assert_close(
        rebuilt.observation_normalizer_mean,
        source.observation_normalizer_mean,
    )
    torch.testing.assert_close(
        rebuilt.observation_normalizer_std,
        source.observation_normalizer_std,
    )


def test_scratch_gail_fits_the_shared_observation_normalizer():
    policy = make_actor_critic(
        "recurrent_transformer",
        4,
        8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=2,
        transformer_heads=2,
        transformer_observation_normalization=True,
        transformer_observation_tokenization="dense_temporal",
        transformer_memory_tokens=1,
    )
    observations = np.asarray(
        [
            [1.0, 10.0, -3.0, 0.0],
            [3.0, 14.0, -1.0, 4.0],
        ],
        dtype=np.float32,
    )
    stats = fit_policy_observation_normalizer(policy, observations)
    np.testing.assert_allclose(stats["mean"], [2.0, 12.0, -2.0, 2.0])
    np.testing.assert_allclose(stats["std"], [1.0, 2.0, 1.0, 2.0])
    torch.testing.assert_close(
        policy.observation_normalizer_mean,
        torch.tensor([2.0, 12.0, -2.0, 2.0]),
    )


def synthetic_transitions(seed: int = 0):
    rng = np.random.default_rng(seed)
    observations = []
    actions = []
    trajectory_ids = []
    vehicle_ids = []
    timesteps = []
    dones = []
    for trajectory in range(9):
        previous = np.zeros(2, dtype=np.float32)
        for step in range(4):
            obs = rng.normal(size=4).astype(np.float32)
            action = np.tanh(
                np.asarray(
                    [0.7 * obs[0] - 0.2 * obs[1] + 0.1 * previous[0], -0.6 * obs[2] + 0.2 * previous[1]],
                    dtype=np.float32,
                )
            )
            observations.append(obs)
            actions.append(action)
            trajectory_ids.append(f"episode_{trajectory}:vehicle_{trajectory}")
            vehicle_ids.append(trajectory)
            timesteps.append(step)
            dones.append(step == 7)
            previous = action
    observations_array = np.asarray(observations, dtype=np.float32)
    return types.SimpleNamespace(
        policy_observations=observations_array,
        next_policy_observations=observations_array.copy(),
        actions_continuous_env=np.asarray(actions, dtype=np.float32),
        trajectory_ids=np.asarray(trajectory_ids, dtype=object),
        vehicle_ids=np.asarray(vehicle_ids, dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        dones=np.asarray(dones, dtype=bool),
    )


def test_trajectory_splits_are_disjoint_and_windows_keep_warmup_context():
    transitions = synthetic_transitions()
    split = split_trajectory_ids(
        transitions,
        train_fraction=0.75,
        validation_fraction=0.125,
        seed=17,
    )
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert sum(len(values) for values in split.values()) == 9

    trajectory_id, indices = trajectory_segments(transitions)[0]
    windows = build_sequence_windows(
        transitions,
        [trajectory_id],
        sequence_length=2,
        context_length=2,
    )
    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0].train_indices, indices[:2])
    assert windows[0].context_indices.size == 0
    np.testing.assert_array_equal(windows[1].context_indices, indices[:2])
    np.testing.assert_array_equal(windows[1].train_indices, indices[2:4])

    transitions.timesteps[2] = 5
    split_segments = [segment for name, segment in trajectory_segments(transitions) if name == trajectory_id]
    assert [len(segment) for segment in split_segments] == [2, 1, 1]


def test_recurrent_bc_learns_and_selects_a_capable_checkpoint():
    torch.manual_seed(4)
    transitions = synthetic_transitions(seed=4)
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=2,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
        transformer_use_causal_attention=True,
    )
    recorded_epochs = []
    result = train_recurrent_behavior_clone(
        policy,
        transitions,
        device=torch.device("cpu"),
        seed=4,
        split_seed=9,
        epochs=8,
        learning_rate=1e-2,
        weight_decay=0.0,
        sequence_length=2,
        sequences_per_batch=3,
        micro_batch_sequences=3,
        train_fraction=0.75,
        validation_fraction=0.125,
        early_stopping_patience=0,
        selection_min_validation_skill=0.01,
        epoch_callback=recorded_epochs.append,
    )
    assert result.best_epoch > 0
    assert result.summary["validation_mse"] < result.summary["validation_baseline_mse"]
    assert result.summary["validation_mse"] < result.summary["initial_validation_mse"]
    assert result.summary["relative_validation_improvement"] > 0.0
    assert result.summary["configured_epochs"] == 8
    assert result.summary["completed_epochs"] == 8
    assert result.summary["validation_skill"] > 0.0
    assert len(result.summary["validation_prediction_std"]) == 2
    assert len(result.summary["validation_target_std"]) == 2
    assert len(result.summary["validation_prediction_std_ratio"]) == 2
    assert len(result.summary["validation_prediction_target_correlation"]) == 2
    assert result.summary["validation_prediction_std_ratio"][0] > 0.0
    assert result.summary["test_mse"] < 0.6
    assert recorded_epochs == result.history
    assert len(recorded_epochs) == 8
    assert all("validation_mse" in row for row in recorded_epochs)
    assert all("validation_mae" in row for row in recorded_epochs)
    assert all("validation_skill" in row for row in recorded_epochs)
    assert all("validation_baseline_mse" in row for row in recorded_epochs)
    assert all("validation_prediction_std_ratio" in row for row in recorded_epochs)
    assert all("validation_prediction_target_correlation" in row for row in recorded_epochs)
    assert all("selection_fallback_gate_margin" in row for row in recorded_epochs)
    assert all(row["selection_min_validation_skill"] == 0.01 for row in recorded_epochs)
    assert all("gradient_norm_mean" in row for row in recorded_epochs)
    assert all("gradient_clipped_fraction" in row for row in recorded_epochs)
    assert all(isinstance(row["is_best_so_far"], bool) for row in recorded_epochs)
    assert any(row["is_best_so_far"] for row in recorded_epochs)
    assert result.summary["selection_min_validation_skill"] == 0.01


def test_prepared_data_reuses_exact_splits_windows_and_baseline():
    transitions = synthetic_transitions(seed=8)
    first = prepare_recurrent_bc_data(
        transitions,
        split_seed=19,
        sequence_length=2,
        train_fraction=0.75,
        validation_fraction=0.125,
        context_length=2,
    )
    second = prepare_recurrent_bc_data(
        transitions,
        split_seed=19,
        sequence_length=2,
        train_fraction=0.75,
        validation_fraction=0.125,
        context_length=2,
    )
    assert first.transitions is transitions
    assert first.split_trajectory_ids == second.split_trajectory_ids
    assert first.validation_baseline_mse == second.validation_baseline_mse
    for split in ("train", "validation", "test"):
        assert len(first.split_windows[split]) == len(second.split_windows[split])
        for left, right in zip(first.split_windows[split], second.split_windows[split], strict=True):
            np.testing.assert_array_equal(left.context_indices, right.context_indices)
            np.testing.assert_array_equal(left.train_indices, right.train_indices)


def test_micro_batches_preserve_full_batch_loss_and_gradients():
    torch.manual_seed(12)
    transitions = synthetic_transitions(seed=12)
    prepared = prepare_recurrent_bc_data(
        transitions,
        split_seed=19,
        sequence_length=2,
        train_fraction=0.75,
        validation_fraction=0.125,
        context_length=2,
    )
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=2,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
        transformer_use_causal_attention=True,
    )
    windows = prepared.split_windows["train"][:3]
    full_loss, _mae, full_count = _batched_sequence_loss(
        policy,
        prepared.observations,
        prepared.actions,
        windows,
        device=torch.device("cpu"),
    )
    full_loss.backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None
    }

    policy.zero_grad(set_to_none=True)
    split_loss = torch.zeros(())
    for window in windows:
        loss, _mae, count = _batched_sequence_loss(
            policy,
            prepared.observations,
            prepared.actions,
            [window],
            device=torch.device("cpu"),
        )
        weighted = loss * (float(count) / float(full_count))
        split_loss = split_loss + weighted.detach()
        weighted.backward()

    torch.testing.assert_close(split_loss, full_loss.detach(), rtol=1e-6, atol=1e-7)
    for name, expected in full_gradients.items():
        torch.testing.assert_close(policy.get_parameter(name).grad, expected, rtol=2e-5, atol=2e-6)


def test_inverse_variance_action_loss_weights_are_recorded():
    torch.manual_seed(23)
    transitions = synthetic_transitions(seed=23)
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=2,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
        transformer_use_causal_attention=True,
    )
    result = train_recurrent_behavior_clone(
        policy,
        transitions,
        device=torch.device("cpu"),
        seed=23,
        split_seed=9,
        epochs=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        sequence_length=2,
        sequences_per_batch=3,
        micro_batch_sequences=3,
        train_fraction=0.75,
        validation_fraction=0.125,
        early_stopping_patience=0,
        action_loss_weights=[1.0, 1.0],
        action_loss_weighting="inverse_variance",
    )
    assert result.summary["action_loss_weighting"] == "inverse_variance"
    assert result.summary["action_loss_weights"][1] > result.summary[
        "action_loss_weights"
    ][0]
    assert len(result.summary["training_action_variance"]) == 2
    assert "validation_selection_mse" in result.summary


def test_anti_collapse_objective_adds_finite_moment_losses():
    torch.manual_seed(31)
    transitions = synthetic_transitions(seed=31)
    prepared = prepare_recurrent_bc_data(
        transitions,
        split_seed=19,
        sequence_length=2,
        train_fraction=0.75,
        validation_fraction=0.125,
        context_length=2,
    )
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=2,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
        transformer_use_causal_attention=True,
    )
    diagnostics = {}
    objective, _mae, count = _batched_sequence_loss(
        policy,
        prepared.observations,
        prepared.actions,
        prepared.split_windows["train"][:3],
        device=torch.device("cpu"),
        correlation_loss_weight=0.02,
        variance_loss_weight=0.05,
        minimum_prediction_std_ratios=np.asarray([0.25, 0.10]),
        loss_diagnostics=diagnostics,
    )
    assert count > 0
    assert diagnostics["objective"] >= diagnostics["base_mse"]
    assert diagnostics["correlation_loss"] >= 0.0
    assert diagnostics["variance_loss"] >= 0.0
    assert np.isfinite(diagnostics["prediction_std_ratio"]).all()
    assert np.isfinite(diagnostics["prediction_target_correlation"]).all()
    objective.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )
