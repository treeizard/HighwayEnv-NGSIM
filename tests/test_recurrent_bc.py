from __future__ import annotations

import types

import numpy as np
import pytest
import policy.methods.bc as recurrent_bc_module
import torch
from policy.evaluation.checkpoints import (
    assert_policy_architecture_matches_checkpoint,
    policy_architecture_contract,
    shared_interpretable_transformer_architecture,
)
from policy.contracts.training_config import PSGAILConfig
from policy.models.recurrent import make_actor_critic
from policy.methods.bc import (
    _batched_sequence_loss,
    _center_sequence_steps,
    _mixture_mean_variance,
    build_sequence_windows,
    mirror_normalized_actions,
    mirror_policy_observations,
    prepare_recurrent_bc_data,
    prepare_recurrent_bc_data_from_explicit_splits,
    split_trajectory_ids,
    train_recurrent_behavior_clone,
    trajectory_segments,
)
from policy.methods.gail.policy import (
    _make_policy_from_state_dict,
    fit_policy_observation_normalizer,
)
from policy.methods.gail.train import _policy_action_tuple


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


def test_policy_mirror_augmentation_is_exact_and_involutive():
    observation = np.zeros((2, 322), dtype=np.float32)
    observation[0, 0:2] = [1.0, 2.0]
    observation[0, 254:256] = [3.0, 4.0]
    observation[0, 256:259] = [1.0, 2.0, 3.0]
    observation[0, 316:319] = [4.0, 5.0, 6.0]
    observation[0, 319:322] = [4.5, 15.0, 0.2]
    action = np.asarray([[0.3, -0.4], [-0.2, 0.1]], dtype=np.float32)

    mirrored = mirror_policy_observations(observation)
    mirrored_action = mirror_normalized_actions(action)
    assert mirrored[0, 254:256].tolist() == [1.0, 2.0]
    assert mirrored[0, 0:2].tolist() == [3.0, 4.0]
    assert mirrored[0, 316:319].tolist() == [1.0, 2.0, -3.0]
    assert mirrored[0, 256:259].tolist() == [4.0, 5.0, -6.0]
    np.testing.assert_allclose(
        mirrored[0, 319:322],
        np.asarray([4.5, 15.0, -0.2], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        mirrored_action,
        [[0.3, 0.4], [-0.2, -0.1]],
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        mirror_policy_observations(mirrored),
        observation,
    )
    np.testing.assert_array_equal(
        mirror_normalized_actions(mirrored_action),
        action,
    )


def test_mirror_mixture_moments_match_the_augmented_distribution():
    original = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transformed = np.asarray([[-1.0, 4.0], [-3.0, 2.0]], dtype=np.float32)

    mean, variance = _mixture_mean_variance(
        original,
        transformed,
        transformed_probability=0.5,
    )
    explicit = np.concatenate([original, transformed], axis=0)
    np.testing.assert_allclose(mean, explicit.mean(axis=0), atol=1.0e-12)
    np.testing.assert_allclose(variance, explicit.var(axis=0), atol=1.0e-12)
    assert mean[0] == pytest.approx(0.0)


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


@pytest.mark.parametrize(
    "model_name",
    ["recurrent_gru", "recurrent_transformer"],
)
def test_recurrent_observation_standardization_is_unclipped_by_default(
    model_name: str,
):
    policy = make_actor_critic(
        model_name,
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_observation_normalization=True,
        policy_observation_standardization_clip=0.0,
        transformer_observation_tokenization="dense_temporal",
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
    )
    policy.set_observation_normalizer(torch.zeros(4), torch.ones(4))
    captured: list[torch.Tensor] = []
    input_layer = (
        policy.input_encoder[0]
        if model_name == "recurrent_gru"
        else policy.dense_observation_proj[0]
    )
    hook = input_layer.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    observations = torch.tensor([[6.0, -7.0, 0.5, -0.25]])
    if model_name == "recurrent_gru":
        policy._encode(observations, None)
    else:
        policy._build_current_tokens(observations)
    hook.remove()

    torch.testing.assert_close(captured[0], observations)
    assert policy.policy_observation_standardization_clip == 0.0


@pytest.mark.parametrize(
    "model_name",
    ["recurrent_gru", "recurrent_transformer"],
)
def test_recurrent_observation_clip_requires_explicit_positive_setting(
    model_name: str,
):
    policy = make_actor_critic(
        model_name,
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
        transformer_observation_normalization=True,
        policy_observation_standardization_clip=5.0,
        transformer_observation_tokenization="dense_temporal",
        transformer_memory_tokens=1,
        transformer_memory_context_length=2,
    )
    policy.set_observation_normalizer(torch.zeros(4), torch.ones(4))
    captured: list[torch.Tensor] = []
    input_layer = (
        policy.input_encoder[0]
        if model_name == "recurrent_gru"
        else policy.dense_observation_proj[0]
    )
    hook = input_layer.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    observations = torch.tensor([[6.0, -7.0, 0.5, -0.25]])
    if model_name == "recurrent_gru":
        policy._encode(observations, None)
    else:
        policy._build_current_tokens(observations)
    hook.remove()

    torch.testing.assert_close(
        captured[0],
        torch.tensor([[5.0, -5.0, 0.5, -0.25]]),
    )


def test_recurrent_observation_standardization_rejects_nonfinite_values():
    policy = make_actor_critic(
        "recurrent_gru",
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_observation_normalization=True,
        policy_observation_standardization_clip=0.0,
    )
    policy.set_observation_normalizer(torch.zeros(4), torch.ones(4))
    with pytest.raises(ValueError, match="non-finite"):
        policy._encode(
            torch.tensor([[0.0, float("nan"), 0.0, 0.0]]),
            None,
        )


def test_legacy_checkpoint_without_observation_clip_replays_with_five():
    legacy_checkpoint = {
        "config": {
            "policy_model": "recurrent_transformer",
            "transformer_observation_normalization": True,
        },
        "policy_architecture": {
            "policy_model": "recurrent_transformer",
            "transformer_observation_normalization": True,
        },
    }
    assert (
        policy_architecture_contract(legacy_checkpoint)[
            "policy_observation_standardization_clip"
        ]
        == 5.0
    )
    assert (
        policy_architecture_contract(PSGAILConfig())[
            "policy_observation_standardization_clip"
        ]
        == 0.0
    )


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
            "policy_observation_standardization_clip": 0.0,
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
        policy_observation_standardization_clip=depth2[
            "policy_observation_standardization_clip"
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
    assert policy.policy_observation_standardization_clip == 0.0


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


def test_policy_action_helper_rejects_out_of_range_output_without_clamping():
    class FixedContinuousPolicy(torch.nn.Module):
        action_mode = "continuous"
        supports_recurrent_memory = False

        def __init__(self, output):
            super().__init__()
            self.register_buffer(
                "fixed_output",
                torch.as_tensor(output, dtype=torch.float32),
            )
            self.log_std = torch.nn.Parameter(torch.full((2,), -2.0))

        def forward(self, observations):
            actions = self.fixed_output.expand(len(observations), -1)
            return actions, torch.zeros(
                len(observations),
                dtype=observations.dtype,
                device=observations.device,
            )

    cfg = PSGAILConfig(
        action_mode="continuous",
        continuous_action_dim=2,
    )
    raw_observation = np.zeros((1, 323), dtype=np.float32)
    native = _policy_action_tuple(
        FixedContinuousPolicy([[0.25, -0.75]]),
        raw_observation,
        object(),
        device=torch.device("cpu"),
        deterministic=True,
        cfg=cfg,
    )
    np.testing.assert_array_equal(
        native[0],
        np.asarray([0.25, -0.75], dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="Actions are never clamped"):
        _policy_action_tuple(
            FixedContinuousPolicy([[1.01, 0.0]]),
            raw_observation,
            object(),
            device=torch.device("cpu"),
            deterministic=True,
            cfg=cfg,
        )


def test_stochastic_policy_action_helper_uses_single_tanh_squash():
    class FixedContinuousPolicy(torch.nn.Module):
        action_mode = "continuous"
        supports_recurrent_memory = False

        def __init__(self):
            super().__init__()
            self.register_buffer(
                "fixed_output",
                torch.tensor([[0.8, -0.4]], dtype=torch.float32),
            )
            self.log_std = torch.nn.Parameter(
                torch.log(torch.tensor([0.2, 0.3]))
            )

        def forward(self, observations):
            return (
                self.fixed_output.expand(len(observations), -1),
                torch.zeros(len(observations)),
            )

    policy = FixedContinuousPolicy()
    cfg = PSGAILConfig(
        action_mode="continuous",
        continuous_action_dim=2,
    )
    raw_observation = np.zeros((1, 323), dtype=np.float32)
    torch.manual_seed(123)
    expected = torch.tanh(
        torch.distributions.Normal(
            torch.atanh(policy.fixed_output),
            torch.exp(policy.log_std).expand_as(policy.fixed_output),
        ).sample()
    )
    torch.manual_seed(123)
    observed = _policy_action_tuple(
        policy,
        raw_observation,
        object(),
        device=torch.device("cpu"),
        deterministic=False,
        cfg=cfg,
    )
    np.testing.assert_allclose(
        observed[0],
        expected[0].numpy(),
        rtol=0.0,
        atol=1.0e-7,
    )


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


def test_full_prefix_warmup_reproduces_exact_live_recurrent_suffix():
    torch.manual_seed(91)
    row_count = 8
    observations = np.random.default_rng(91).normal(
        size=(row_count, 4)
    ).astype(np.float32)
    transitions = types.SimpleNamespace(
        policy_observations=observations,
        actions_continuous_env=np.zeros((row_count, 2), dtype=np.float32),
        trajectory_ids=np.asarray(["trajectory"] * row_count, dtype=object),
        timesteps=np.arange(row_count, dtype=np.int64),
        dones=np.asarray([False] * (row_count - 1) + [True]),
    )
    windows = build_sequence_windows(
        transitions,
        ["trajectory"],
        sequence_length=2,
        context_length=2,
        warmup_mode="full_prefix",
    )
    legacy_windows = build_sequence_windows(
        transitions,
        ["trajectory"],
        sequence_length=2,
        context_length=2,
        warmup_mode="bounded_raw_history",
    )
    np.testing.assert_array_equal(
        windows[3].context_indices,
        np.arange(6, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        legacy_windows[3].context_indices,
        np.arange(4, 6, dtype=np.int64),
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
    policy.eval()

    def advance(memory, index):
        prediction, _value, new_memory = policy(
            torch.as_tensor(observations[index : index + 1]),
            memory=memory,
            return_memory=True,
        )
        shifted = torch.cat(
            [memory[:, 1:], new_memory.unsqueeze(1)],
            dim=1,
        )
        return prediction, shifted

    with torch.no_grad():
        live_memory = policy.initial_memory(1)
        live_predictions = []
        for index in range(row_count):
            prediction, live_memory = advance(live_memory, index)
            live_predictions.append(prediction)
        live_predictions = torch.cat(live_predictions, dim=0)

        replay_predictions: dict[int, torch.Tensor] = {}
        for window in windows:
            memory = policy.initial_memory(1)
            for index in window.context_indices:
                _prediction, memory = advance(memory, int(index))
            for index in window.train_indices:
                prediction, memory = advance(memory, int(index))
                replay_predictions[int(index)] = prediction.squeeze(0)
        replay = torch.stack(
            [replay_predictions[index] for index in range(row_count)],
            dim=0,
        )

    torch.testing.assert_close(replay, live_predictions, atol=1e-6, rtol=1e-6)


def test_validation_loss_selection_ignores_impossible_qualification_gates():
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
        action_loss_weighting="inverse_variance",
        selection_min_validation_skill=2.0,
        selection_min_prediction_std_ratios=[100.0, 100.0],
        selection_min_prediction_correlations=[1.0, 1.0],
        evaluate_test=True,
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
    assert all(row["selection_min_validation_skill"] == 2.0 for row in recorded_epochs)
    assert all(not row["selection_eligible"] for row in recorded_epochs)
    assert all("gradient_norm_mean" in row for row in recorded_epochs)
    assert all("gradient_clipped_fraction" in row for row in recorded_epochs)
    assert all(isinstance(row["is_best_so_far"], bool) for row in recorded_epochs)
    assert any(row["is_best_so_far"] for row in recorded_epochs)
    assert result.summary["checkpoint_selection_rule"] == "validation_loss"
    assert result.summary["checkpoint_selection_metric"] == (
        "unweighted_validation_mse"
    )
    assert result.summary["validation_selection_mse"] == pytest.approx(
        result.summary["validation_mse"]
    )
    assert all(
        row["validation_selection_mse"]
        == pytest.approx(row["validation_mse"])
        for row in recorded_epochs
    )
    assert all(
        "validation_weighted_objective_mse" in row
        for row in recorded_epochs
    )
    assert result.summary["best_selection_eligible"] is False
    assert result.best_epoch == int(
        min(
            recorded_epochs,
            key=lambda row: row["validation_selection_mse"],
        )["epoch"]
    )
    assert result.summary["selection_min_validation_skill"] == 2.0
    assert result.summary["selected_history_row"] == next(
        row
        for row in recorded_epochs
        if int(row["epoch"]) == result.best_epoch
    )


def test_recurrent_bc_rejects_nonpositive_epochs_before_data_access():
    policy = types.SimpleNamespace(supports_recurrent_memory=True)
    with pytest.raises(ValueError, match="epochs must be at least one"):
        train_recurrent_behavior_clone(
            policy,
            None,
            device=torch.device("cpu"),
            seed=0,
            split_seed=0,
            epochs=0,
            learning_rate=1e-3,
            weight_decay=0.0,
            sequence_length=2,
            sequences_per_batch=1,
            micro_batch_sequences=1,
        )


def test_minimum_epoch_early_stop_and_deferred_test_are_explicit(monkeypatch):
    torch.manual_seed(44)
    transitions = synthetic_transitions(seed=44)
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
    prepared = prepare_recurrent_bc_data(
        transitions,
        split_seed=9,
        sequence_length=2,
        train_fraction=0.75,
        validation_fraction=0.125,
        context_length=2,
    )
    real_evaluate = recurrent_bc_module.evaluate_recurrent_bc
    evaluated_splits: list[str] = []

    def recording_evaluate(policy, transitions, windows, **kwargs):
        split = next(
            (
                name
                for name, candidate in prepared.split_windows.items()
                if windows is candidate
            ),
            "unknown",
        )
        evaluated_splits.append(split)
        return real_evaluate(policy, transitions, windows, **kwargs)

    monkeypatch.setattr(
        recurrent_bc_module,
        "evaluate_recurrent_bc",
        recording_evaluate,
    )
    result = train_recurrent_behavior_clone(
        policy,
        None,
        device=torch.device("cpu"),
        seed=44,
        split_seed=9,
        epochs=4,
        learning_rate=0.0,
        weight_decay=0.0,
        sequence_length=2,
        sequences_per_batch=3,
        micro_batch_sequences=3,
        early_stopping_patience=1,
        early_stopping_min_epochs=2,
        prepared_data=prepared,
    )

    assert result.best_epoch == 1
    assert result.summary["completed_epochs"] == 2
    assert result.summary["stopping_reason"] == (
        "early_stopping_patience_exhausted"
    )
    assert result.summary["stop_epoch"] == 2
    assert result.summary["early_stopping_min_epochs"] == 2
    assert result.summary["early_stopping_min_delta_relative"] == 0.001
    assert result.summary["selected_history_row"] == result.history[0]
    assert "test" not in evaluated_splits
    assert result.summary["offline_test_evaluated"] is False
    assert result.summary["offline_test_status"] == "pending_deferred"
    assert result.summary["test_mse"] is None
    assert result.summary["test_prediction_std_ratio"] is None


def test_small_strict_loss_improvements_update_checkpoint_not_patience(
    monkeypatch,
):
    torch.manual_seed(45)
    transitions = synthetic_transitions(seed=45)
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
    prepared = prepare_recurrent_bc_data(
        transitions,
        split_seed=9,
        sequence_length=2,
        train_fraction=0.75,
        validation_fraction=0.125,
        context_length=2,
    )
    validation_losses = iter([1.5, 1.0, 0.9996, 0.9992, 0.9992])

    def metrics(loss):
        return {
            "mse": loss,
            "mae": 0.1,
            "samples": 8.0,
            "windows": 2.0,
            "action_mse": [loss, loss],
            "action_mae": [0.1, 0.1],
            "prediction_mean": [0.0, 0.0],
            "prediction_std": [0.2, 0.2],
            "target_mean": [0.0, 0.0],
            "target_std": [0.2, 0.2],
            "prediction_std_ratio": [1.0, 1.0],
            "prediction_target_correlation": [0.5, 0.5],
            "prediction_saturation_fraction": [0.0, 0.0],
        }

    def controlled_evaluate(_policy, _transitions, windows, **_kwargs):
        if windows is prepared.split_windows["validation"]:
            return metrics(next(validation_losses))
        return metrics(0.5)

    monkeypatch.setattr(
        recurrent_bc_module,
        "evaluate_recurrent_bc",
        controlled_evaluate,
    )
    result = train_recurrent_behavior_clone(
        policy,
        None,
        device=torch.device("cpu"),
        seed=45,
        split_seed=9,
        epochs=5,
        learning_rate=0.0,
        weight_decay=0.0,
        sequence_length=2,
        sequences_per_batch=3,
        micro_batch_sequences=3,
        early_stopping_patience=2,
        early_stopping_min_epochs=0,
        early_stopping_min_delta_relative=0.001,
        prepared_data=prepared,
    )

    assert result.summary["completed_epochs"] == 3
    assert result.summary["stale_epochs_at_stop"] == 2
    assert result.summary["stopping_reason"] == (
        "early_stopping_patience_exhausted"
    )
    assert [row["is_best_so_far"] for row in result.history] == [
        True,
        True,
        True,
    ]
    assert [
        row["early_stopping_significant_improvement"]
        for row in result.history
    ] == [True, False, False]
    assert result.best_epoch == 3
    assert result.summary["selected_history_row"] == result.history[2]
    assert result.summary["checkpoint_selection_tie_tolerance"] == 0.0


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


def test_recurrent_bc_rejects_out_of_range_labels_instead_of_clipping():
    transitions = synthetic_transitions(seed=8)
    transitions.actions_continuous_env[0, 1] = 1.01

    with pytest.raises(ValueError, match="Labels are never clipped"):
        prepare_recurrent_bc_data(
            transitions,
            split_seed=19,
            sequence_length=2,
            train_fraction=0.75,
            validation_fraction=0.125,
            context_length=2,
        )


def test_explicit_source_splits_are_preserved_without_internal_resplitting():
    source_splits = {
        "train": synthetic_transitions(seed=1),
        "validation": synthetic_transitions(seed=2),
        "test": synthetic_transitions(seed=3),
    }
    prepared = prepare_recurrent_bc_data_from_explicit_splits(
        source_splits,
        sequence_length=2,
        context_length=2,
    )

    assert prepared.transitions.metadata["split_method"] == "explicit_source_directories"
    assert len(prepared.observations) == sum(
        len(transitions.policy_observations)
        for transitions in source_splits.values()
    )
    assert all(
        trajectory_id.startswith(f"{split}:")
        for split, identifiers in prepared.split_trajectory_ids.items()
        for trajectory_id in identifiers
    )
    index_sets = {
        split: {
            int(index)
            for window in windows
            for index in window.train_indices
        }
        for split, windows in prepared.split_windows.items()
    }
    assert index_sets["train"].isdisjoint(index_sets["validation"])
    assert index_sets["train"].isdisjoint(index_sets["test"])
    assert index_sets["validation"].isdisjoint(index_sets["test"])
    train_indices = np.asarray(sorted(index_sets["train"]), dtype=np.int64)
    validation_indices = np.asarray(
        sorted(index_sets["validation"]),
        dtype=np.int64,
    )
    train_mean = prepared.actions[train_indices].mean(axis=0)
    expected_baseline = np.mean(
        np.square(prepared.actions[validation_indices] - train_mean)
    )
    assert prepared.validation_baseline_mse == pytest.approx(expected_baseline)


def test_explicit_unconditioned_splits_ignore_all_invalid_behavior_sentinels():
    source_splits = {
        "train": synthetic_transitions(seed=21),
        "validation": synthetic_transitions(seed=22),
    }
    for transitions in source_splits.values():
        rows = len(transitions.trajectory_ids)
        transitions.behavior_ids = np.full(rows, -1, dtype=np.int8)
        transitions.next_behavior_ids = np.full(rows, -1, dtype=np.int8)
        transitions.segment_ids = np.full(rows, -1, dtype=np.int64)

    prepared = prepare_recurrent_bc_data_from_explicit_splits(
        source_splits,
        sequence_length=2,
        context_length=2,
    )

    assert prepared.behavior_sampling_manifest is None
    assert prepared.split_windows["train"]


def test_explicit_train_validation_preparation_keeps_test_unopened():
    prepared = prepare_recurrent_bc_data_from_explicit_splits(
        {
            "train": synthetic_transitions(seed=11),
            "validation": synthetic_transitions(seed=12),
        },
        sequence_length=2,
        context_length=2,
    )

    assert prepared.transitions.metadata["split_method"] == (
        "explicit_source_directories_deferred_test"
    )
    assert set(prepared.transitions.metadata["sources"]) == {
        "train",
        "validation",
    }
    assert prepared.split_trajectory_ids["test"] == []
    assert prepared.split_windows["test"] == []


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
