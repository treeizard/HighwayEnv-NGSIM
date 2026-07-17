from __future__ import annotations

import types

import numpy as np
import torch

from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.recurrent_bc import (
    build_sequence_windows,
    split_trajectory_ids,
    train_recurrent_behavior_clone,
    trajectory_segments,
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
        epoch_callback=recorded_epochs.append,
    )
    assert result.best_epoch > 0
    assert result.summary["validation_mse"] < result.summary["validation_baseline_mse"]
    assert result.summary["validation_mse"] < result.summary["initial_validation_mse"]
    assert result.summary["relative_validation_improvement"] > 0.0
    assert result.summary["configured_epochs"] == 8
    assert result.summary["completed_epochs"] == 8
    assert result.summary["validation_skill"] > 0.0
    assert result.summary["test_mse"] < 0.6
    assert recorded_epochs == result.history
    assert len(recorded_epochs) == 8
    assert all("validation_mse" in row for row in recorded_epochs)
    assert all("validation_mae" in row for row in recorded_epochs)
    assert all("validation_skill" in row for row in recorded_epochs)
    assert all("validation_baseline_mse" in row for row in recorded_epochs)
    assert all(isinstance(row["is_best_so_far"], bool) for row in recorded_epochs)
    assert any(row["is_best_so_far"] for row in recorded_epochs)
