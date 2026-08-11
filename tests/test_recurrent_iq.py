from __future__ import annotations

import hashlib
import math
import sys
import types

import numpy as np
import pytest
import torch
from policy.contracts.training_config import PSGAILConfig
from policy.evaluation.checkpoints import (
    canonical_tensor_state_sha256,
    policy_architecture_contract,
)
from policy.models.recurrent import make_actor_critic
from policy.methods.bc import build_sequence_windows, split_trajectory_ids
from policy.methods.iq_learn import (
    RecurrentTwinQNetwork,
    update_recurrent_iq,
)
from policy.methods.iq_learn_train import (
    _tensorboard_scalars,
    load_initial_policy_checkpoint,
    make_config,
    parse_args,
    policy_architecture,
    rollout_capability,
    rollout_thresholds,
    sample_policy_replay_action,
)


def test_unconditioned_iq_mode_keeps_behavior_references_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_recurrent_iq_learn.py",
            "--expert-data",
            str(tmp_path / "train"),
            "--expert-validation-data",
            str(tmp_path / "validation"),
            "--out-dir",
            str(tmp_path / "out"),
            "--initial-policy-checkpoint",
            str(tmp_path / "bc.pt"),
            "--transformer-layers",
            "2",
            "--conditioning-mode",
            "none",
            "--initialization-mode",
            "bc",
        ],
    )

    args = parse_args()
    cfg = make_config(args)

    assert not cfg.behavior_conditioning_enabled
    assert cfg.behavior_label_sidecar == ""
    assert cfg.behavior_command_schedule == ""
    assert cfg.behavior_sampling_manifest == ""


def synthetic_transitions(seed: int = 0):
    rng = np.random.default_rng(seed)
    observations = []
    next_observations = []
    actions = []
    trajectory_ids = []
    vehicle_ids = []
    timesteps = []
    dones = []
    for trajectory in range(9):
        trajectory_obs = rng.normal(size=(5, 4)).astype(np.float32)
        for step in range(4):
            obs = trajectory_obs[step]
            next_obs = trajectory_obs[step + 1]
            action = np.tanh(np.asarray([0.6 * obs[0] - 0.2 * obs[1], -0.5 * obs[2]], dtype=np.float32))
            observations.append(obs)
            next_observations.append(next_obs)
            actions.append(action)
            trajectory_ids.append(f"episode_{trajectory}:vehicle_{trajectory}")
            vehicle_ids.append(trajectory)
            timesteps.append(step)
            dones.append(step == 3)
    return types.SimpleNamespace(
        policy_observations=np.asarray(observations, dtype=np.float32),
        next_policy_observations=np.asarray(next_observations, dtype=np.float32),
        actions_continuous_env=np.asarray(actions, dtype=np.float32),
        trajectory_ids=np.asarray(trajectory_ids, dtype=object),
        vehicle_ids=np.asarray(vehicle_ids, dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        dones=np.asarray(dones, dtype=bool),
    )


def make_models():
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
        transformer_use_causal_attention=True,
    )
    q_net = RecurrentTwinQNetwork(
        4,
        2,
        8,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
        memory_tokens=1,
        memory_context_length=2,
        use_causal_attention=True,
    )
    target_q_net = RecurrentTwinQNetwork(
        4,
        2,
        8,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
        memory_tokens=1,
        memory_context_length=2,
        use_causal_attention=True,
    )
    target_q_net.load_state_dict(q_net.state_dict())
    return policy, q_net, target_q_net


def test_recurrent_q_depends_on_memory_context():
    torch.manual_seed(2)
    _policy, q_net, _target = make_models()
    obs = torch.randn(2, 4)
    action = torch.randn(2, 2).tanh()
    empty = q_net.q1.initial_memory(2, device=torch.device("cpu"))
    context = torch.randn_like(empty)

    empty_q = q_net.q1(obs, action, empty)
    context_q = q_net.q1(obs, action, context)

    assert not torch.allclose(empty_q, context_q)


def test_recurrent_iq_update_is_finite_and_changes_actor_and_q():
    torch.manual_seed(3)
    transitions = synthetic_transitions(seed=3)
    split = split_trajectory_ids(transitions, train_fraction=0.75, validation_fraction=0.125, seed=7)
    windows = build_sequence_windows(
        transitions,
        split["train"],
        sequence_length=2,
        context_length=2,
    )[:3]
    policy, q_net, target_q_net = make_models()
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    q_optimizer = torch.optim.Adam(
        [parameter for parameter in q_net.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    actor_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    q_before = {name: value.detach().clone() for name, value in q_net.state_dict().items()}

    stats = update_recurrent_iq(
        policy,
        q_net,
        target_q_net,
        policy_optimizer,
        q_optimizer,
        transitions,
        windows,
        policy_transitions=synthetic_transitions(seed=4),
        policy_windows=build_sequence_windows(
            synthetic_transitions(seed=4),
            split["train"],
            sequence_length=2,
            context_length=2,
        )[:3],
        device=torch.device("cpu"),
        gamma=0.95,
        entropy_temperature=0.01,
        chi2_alpha=0.5,
        target_tau=0.01,
        bc_coef=1.0,
        actor_bc_only=False,
        max_grad_norm=1.0,
        q_l2_coef=1.0e-3,
    )

    assert stats.samples > 0
    assert stats.expert_samples > 0
    assert stats.policy_samples > 0
    assert stats.q1_loss != stats.q2_loss
    assert stats.q_l2_loss > 0.0
    assert all(math.isfinite(value) for value in stats.as_dict().values())
    assert stats.q_abs_max < 100.0
    assert any(not torch.equal(actor_before[name], value) for name, value in policy.state_dict().items())
    assert any(not torch.equal(q_before[name], value) for name, value in q_net.state_dict().items())


def test_q_only_warmup_does_not_change_actor():
    torch.manual_seed(5)
    transitions = synthetic_transitions(seed=5)
    split = split_trajectory_ids(transitions, train_fraction=0.75, validation_fraction=0.125, seed=8)
    windows = build_sequence_windows(
        transitions,
        split["train"],
        sequence_length=2,
        context_length=2,
    )[:2]
    policy, q_net, target_q_net = make_models()
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    q_optimizer = torch.optim.Adam(
        [parameter for parameter in q_net.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    actor_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}

    stats = update_recurrent_iq(
        policy,
        q_net,
        target_q_net,
        policy_optimizer,
        q_optimizer,
        transitions,
        windows,
        device=torch.device("cpu"),
        gamma=0.95,
        entropy_temperature=0.01,
        chi2_alpha=0.5,
        target_tau=0.01,
        bc_coef=1.0,
        actor_bc_only=True,
        update_actor=False,
        max_grad_norm=1.0,
    )

    assert all(torch.equal(actor_before[name], value) for name, value in policy.state_dict().items())
    assert stats.actor_grad_norm == 0.0
    assert stats.actor_rl_grad_norm == 0.0


def test_recurrent_policy_sampling_retains_full_context_and_is_bounded():
    torch.manual_seed(11)
    policy, _q_net, _target = make_models()
    memory = torch.randn_like(policy.initial_memory(2, device=torch.device("cpu")))
    observations = torch.randn(2, 4)

    action_tuple, updated_memory = sample_policy_replay_action(
        policy,
        observations.numpy(),
        memory,
        device=torch.device("cpu"),
        log_std_min=-5.0,
        log_std_max=-1.5,
    )

    actions = np.asarray(action_tuple)
    assert actions.shape == (2, 2)
    assert np.all(np.abs(actions) <= 1.0)
    assert updated_memory.shape == memory.shape
    assert torch.equal(updated_memory[:, :-1], memory[:, 1:])
    assert not torch.equal(updated_memory[:, -1], memory[:, -1])


def test_tensorboard_scalars_flattens_finite_metrics_only():
    scalars = _tensorboard_scalars({
        "phase": "iq_update",
        "update": 3,
        "q_loss": 0.5,
        "validation_action_mse": [0.1, 0.2],
        "nested": {"passed": True, "missing": float("nan")},
    })

    assert scalars == {
        "q_loss": 0.5,
        "validation_action_mse/action_0": 0.1,
        "validation_action_mse/action_1": 0.2,
        "nested/passed": 1.0,
    }


def test_rollout_gate_preserves_absolute_floors_but_allows_matched_bc_baseline():
    args = types.SimpleNamespace(
        min_rollout_steps=100,
        max_crash_fraction=0.34,
        max_offroad_fraction=0.34,
        max_collision_proxy_fraction=0.34,
        max_rollout_mean_length_regression=20.0,
        max_rollout_fraction_regression=0.0,
    )
    baseline = {
        "bc_eval/mean_episode_length": 140.0,
        "bc_eval/crash_episode_fraction": 2.0 / 3.0,
        "bc_eval/offroad_episode_fraction": 2.0 / 3.0,
        "bc_eval/collision_proxy_episode_fraction": 2.0 / 3.0,
    }
    thresholds = rollout_thresholds(args, baseline)
    assert thresholds["min_mean_episode_length"] == 120.0
    assert thresholds["max_crash_episode_fraction"] == pytest.approx(2.0 / 3.0)
    assert rollout_capability(dict(baseline), args, baseline=baseline)
    regressed = dict(baseline)
    regressed["bc_eval/mean_episode_length"] = 119.0
    assert not rollout_capability(regressed, args, baseline=baseline)


def test_initial_policy_checkpoint_requires_exact_recurrent_architecture(tmp_path):
    source_policy, _q_net, _target = make_models()
    cfg = PSGAILConfig(
        action_mode="continuous",
        continuous_action_dim=2,
        policy_model="recurrent_transformer",
        hidden_size=8,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
        transformer_use_causal_attention=True,
    )
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "checkpoint_kind": "behaviour_cloning_best",
            "policy_state_dict": source_policy.state_dict(),
            "policy_architecture": policy_architecture(cfg, obs_dim=4, action_dim=2),
            "config": {"seed": 0, "scene": "us-101"},
            "bc_stats": {
                "validation_skill": 0.8,
                "validation_mae": 0.02,
                "validation_prediction_std_ratio": [0.9, 0.5],
                "validation_prediction_target_correlation": [0.9, 0.5],
            },
            "data_split": {
                "trajectory_ids": {
                    "train": ["episode_0"],
                    "validation": ["episode_1"],
                    "test": ["episode_2"],
                }
            },
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_name("policy.pt.sha256").write_text(f"{digest}  policy.pt\n", encoding="utf-8")
    loaded_policy, _q_net, _target = make_models()

    provenance = load_initial_policy_checkpoint(
        loaded_policy,
        checkpoint,
        obs_dim=4,
        action_dim=2,
        cfg=cfg,
    )

    assert provenance["sha256"]
    assert provenance["split_trajectory_ids"]["validation"] == ["episode_1"]
    assert all(
        torch.equal(source_policy.state_dict()[name], loaded_policy.state_dict()[name])
        for name in source_policy.state_dict()
    )

    incompatible_cfg = PSGAILConfig(**{**vars(cfg), "transformer_memory_context_length": 3})
    with pytest.raises(
        RuntimeError,
        match="architecture contract mismatch.*memory_context_length",
    ):
        load_initial_policy_checkpoint(
            loaded_policy,
            checkpoint,
            obs_dim=4,
            action_dim=2,
            cfg=incompatible_cfg,
        )


def test_initial_policy_checkpoint_accepts_shared_actor_dimension_metadata(tmp_path):
    source_policy, _q_net, _target = make_models()
    cfg = PSGAILConfig(
        seed=0,
        action_mode="continuous",
        continuous_action_dim=2,
        policy_model="recurrent_transformer",
        hidden_size=8,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
        transformer_use_causal_attention=True,
    )
    checkpoint = tmp_path / "shared.pt"
    state = source_policy.state_dict()
    architecture = policy_architecture_contract(cfg)
    assert "obs_dim" not in architecture
    torch.save(
        {
            "checkpoint_kind": "shared_random_actor",
            "policy_state_dict": state,
            "policy_architecture": architecture,
            "policy_observation_dim": 4,
            "actor_state_sha256": canonical_tensor_state_sha256(state),
            "config": vars(cfg),
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_name("shared.pt.sha256").write_text(
        f"{digest}  shared.pt\n",
        encoding="utf-8",
    )
    loaded_policy, _q_net, _target = make_models()

    provenance = load_initial_policy_checkpoint(
        loaded_policy,
        checkpoint,
        obs_dim=4,
        action_dim=2,
        cfg=cfg,
        domain="us",
        seed=0,
        required_checkpoint_kind="shared_random_actor",
    )

    assert provenance["actor_state_sha256"] == canonical_tensor_state_sha256(state)
