import json
import sys
import types

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    gymnasium = types.ModuleType("gymnasium")

    class _Env:
        pass

    class _Wrapper:
        pass

    class _RecordConstructorArgs:
        pass

    gymnasium.Env = _Env
    gymnasium.Wrapper = _Wrapper

    class _Space:
        def __init__(self, *args, **kwargs):
            self.shape = tuple(kwargs.get("shape", ()))
            self.spaces = tuple(kwargs.get("spaces", ()))

        def contains(self, _value):
            return True

        def sample(self):
            return 0

    class _Tuple:
        def __init__(self, spaces=()):
            self.spaces = tuple(spaces)

    class _Box:
        def __init__(self, shape=()):
            self.shape = tuple(shape)

    class _Discrete:
        def __init__(self, n):
            self.n = int(n)

    class _Dict(dict):
        pass

    gymnasium.spaces = types.SimpleNamespace(
        Space=_Space,
        Tuple=_Tuple,
        Box=_Box,
        Discrete=_Discrete,
        Dict=_Dict,
    )
    gymnasium.make = lambda *args, **kwargs: None
    envs = types.ModuleType("gymnasium.envs")
    registration = types.ModuleType("gymnasium.envs.registration")
    registration.register = lambda *args, **kwargs: None
    registration.registry = {}
    utils = types.ModuleType("gymnasium.utils")
    utils.RecordConstructorArgs = _RecordConstructorArgs
    wrappers = types.ModuleType("gymnasium.wrappers")
    wrappers.RecordVideo = object
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.envs"] = envs
    sys.modules["gymnasium.envs.registration"] = registration
    sys.modules["gymnasium.utils"] = utils
    sys.modules["gymnasium.wrappers"] = wrappers

fake_envs = types.ModuleType("scripts_gail.ps_gail.envs")
fake_envs.controlled_vehicle_snapshot = lambda _env: ([], np.zeros((0, 3), dtype=np.float32))
fake_envs.make_training_env = lambda *args, **kwargs: None
sys.modules.setdefault("scripts_gail.ps_gail.envs", fake_envs)

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.data import (
    ACTION_CONTINUOUS_ENV_COLUMNS,
    ACTION_CONTINUOUS_ENV_KEY,
    ACTION_STEERING_ACCELERATION_COLUMNS,
    ACTION_STEERING_ACCELERATION_KEY,
    build_sequence_windows,
    fit_feature_standardizer,
    load_expert_policy_and_disc_data,
    load_expert_transition_data,
    standardize_features,
    transform_sequence_features,
)
from scripts_gail.ps_gail.models import (
    SequenceTrajectoryDiscriminator,
    TrajectoryDiscriminator,
    make_actor_critic,
)
from scripts_gail.ps_gail.trainer import (
    RolloutBatch,
    policy_distribution_and_values,
    refresh_rollout_rewards,
    sequence_rewards_to_transition_rewards,
    shape_rollout_rewards,
    subsample_rollout_for_training,
    update_discriminator,
    update_policy,
)
from scripts_gail.train_simple_ps_gail import behavior_clone_pretrain
from scripts_gail.train_simple_ps_gail import config_for_round as ps_gail_config_for_round
from scripts_gail.train_simple_airl import config_for_round as airl_config_for_round
from scripts_gail.train_simple_iq_learn import convergence_reached, convergence_score


def _minimal_rollout(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    old_values: np.ndarray,
    returns: np.ndarray,
    advantages: np.ndarray,
    generator_features: np.ndarray | None = None,
    env_penalties: np.ndarray | None = None,
    sequence_features: np.ndarray | None = None,
    sequence_last_indices: np.ndarray | None = None,
    sequence_transition_indices: np.ndarray | None = None,
) -> RolloutBatch:
    n = int(len(observations))
    sequence_features = (
        sequence_features.astype(np.float32)
        if sequence_features is not None
        else np.zeros((0, 1, 2), dtype=np.float32)
    )
    sequence_last_indices = (
        sequence_last_indices.astype(np.int64)
        if sequence_last_indices is not None
        else np.zeros((0,), dtype=np.int64)
    )
    sequence_transition_indices = (
        sequence_transition_indices.astype(np.int64)
        if sequence_transition_indices is not None
        else np.zeros((0, sequence_features.shape[1]), dtype=np.int64)
    )
    return RolloutBatch(
        policy_observations=observations.astype(np.float32),
        next_policy_observations=observations.astype(np.float32),
        actions=actions,
        action_masks=np.ones((n, 5), dtype=bool),
        old_log_probs=old_log_probs.astype(np.float32),
        old_values=old_values.astype(np.float32),
        trajectory_ids=np.arange(n, dtype=np.int32),
        dones=np.ones(n, dtype=bool),
        rewards=np.zeros(n, dtype=np.float32),
        gail_rewards_raw=np.zeros(n, dtype=np.float32),
        gail_rewards_normalized=np.zeros(n, dtype=np.float32),
        env_penalties=(
            env_penalties.astype(np.float32)
            if env_penalties is not None
            else np.zeros(n, dtype=np.float32)
        ),
        returns=returns.astype(np.float32),
        advantages=advantages.astype(np.float32),
        generator_features=(
            generator_features.astype(np.float32)
            if generator_features is not None
            else np.zeros((n, 2), dtype=np.float32)
        ),
        scene_features=np.zeros((0, 2), dtype=np.float32),
        transition_scene_indices=np.full(n, -1, dtype=np.int64),
        sequence_features=sequence_features,
        sequence_last_indices=sequence_last_indices,
        sequence_transition_indices=sequence_transition_indices,
        num_env_steps=n,
        num_agent_steps=n,
    )


def test_rollout_training_subsample_keeps_complete_trajectories_and_remaps_indices():
    n = 12
    rollout = _minimal_rollout(
        observations=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        actions=np.zeros((n, 2), dtype=np.float32),
        old_log_probs=np.zeros(n, dtype=np.float32),
        old_values=np.zeros(n, dtype=np.float32),
        returns=np.zeros(n, dtype=np.float32),
        advantages=np.zeros(n, dtype=np.float32),
        generator_features=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        sequence_features=np.arange(11 * 2, dtype=np.float32).reshape(11, 2),
        sequence_last_indices=np.arange(1, n, dtype=np.int64),
        sequence_transition_indices=np.stack(
            [np.arange(0, n - 1), np.arange(1, n)],
            axis=1,
        ).astype(np.int64),
    )
    rollout.trajectory_ids = np.repeat(np.arange(4, dtype=np.int32), 3)
    rollout.scene_features = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    rollout.transition_scene_indices = np.arange(n, dtype=np.int64)
    cfg = PSGAILConfig(
        rollout_training_subsample=True,
        rollout_training_agent_steps=5,
        gamma=0.95,
    )

    sampled = subsample_rollout_for_training(rollout, cfg, seed=7)

    assert sampled.num_agent_steps == 6
    unique_ids, counts = np.unique(sampled.trajectory_ids, return_counts=True)
    np.testing.assert_array_equal(unique_ids, np.arange(len(unique_ids)))
    np.testing.assert_array_equal(counts, np.full(len(unique_ids), 3))
    assert sampled.policy_observations.shape[0] == sampled.num_agent_steps
    assert sampled.generator_features.shape[0] == sampled.num_agent_steps
    assert sampled.scene_features.shape[0] == sampled.num_agent_steps
    assert np.all(sampled.transition_scene_indices >= 0)
    if sampled.sequence_transition_indices.size:
        assert int(sampled.sequence_transition_indices.max()) < sampled.num_agent_steps
        assert int(sampled.sequence_last_indices.max()) < sampled.num_agent_steps


def _write_action_conditioned_expert_file(path, *, include_actions: bool = True) -> None:
    observations = np.asarray(
        [
            [0.0, 1.0, 2.0, 10.0, 0.10, 2.0, 4.5],
            [0.5, 1.5, 2.5, 11.0, 0.20, 2.0, 4.5],
            [1.0, 2.0, 3.0, 12.0, 0.30, 2.0, 4.5],
        ],
        dtype=np.float32,
    )
    next_observations = observations + 0.25
    trajectory_states = np.asarray(
        [[100.0, 20.0, 10.0], [101.0, 20.5, 10.2], [102.0, 21.0, 10.4]],
        dtype=np.float32,
    )
    arrays = {
        "observations": observations,
        "next_observations": next_observations,
        "trajectory_states": trajectory_states,
        "features": np.concatenate([observations, trajectory_states], axis=1).astype(np.float32),
        "vehicle_ids": np.asarray([7, 7, 7], dtype=np.int64),
        "timesteps": np.asarray([0, 1, 2], dtype=np.int64),
        "dones": np.asarray([False, False, True], dtype=bool),
        "rewards": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema_version": 3,
                    "actions_continuous_env_columns": list(ACTION_CONTINUOUS_ENV_COLUMNS),
                    "actions_steering_acceleration_columns": list(
                        ACTION_STEERING_ACCELERATION_COLUMNS
                    ),
                }
            )
        ),
    }
    if include_actions:
        arrays[ACTION_CONTINUOUS_ENV_KEY] = np.asarray(
            [[0.1, -0.2], [0.0, 0.3], [-0.4, 0.5]],
            dtype=np.float32,
        )
        arrays[ACTION_STEERING_ACCELERATION_KEY] = np.asarray(
            [[-0.157, 0.5], [0.236, 0.0], [0.393, -2.0]],
            dtype=np.float32,
        )
    np.savez_compressed(path, **arrays)


def test_gail_loader_accepts_unified_action_conditioned_dataset(tmp_path):
    path = tmp_path / "expert_unified.npz"
    _write_action_conditioned_expert_file(path)

    policy_obs, features, metadata = load_expert_policy_and_disc_data(str(path))

    assert policy_obs.shape == (3, 6)
    assert features.shape == (3, 9)
    assert metadata["schema_version"] == 3
    assert metadata["actions_continuous_env_columns"] == list(ACTION_CONTINUOUS_ENV_COLUMNS)


def test_action_conditioned_loader_requires_continuous_actions(tmp_path):
    path = tmp_path / "legacy_expert.npz"
    _write_action_conditioned_expert_file(path, include_actions=False)

    with pytest.raises(KeyError, match=ACTION_CONTINUOUS_ENV_KEY):
        load_expert_transition_data(str(path))


def test_action_conditioned_loader_returns_valid_continuous_transition_data(tmp_path):
    path = tmp_path / "expert_unified.npz"
    _write_action_conditioned_expert_file(path)

    data = load_expert_transition_data(str(path), max_samples=2, seed=0)

    assert data.policy_observations.shape == (2, 6)
    assert data.next_policy_observations.shape == (2, 6)
    assert data.actions_continuous_env.shape == (2, 2)
    assert data.actions_steering_acceleration is not None
    assert data.actions_steering_acceleration.shape == (2, 2)
    assert np.all(np.isfinite(data.actions_continuous_env))
    assert data.metadata["continuous_action_dim"] == 2
    assert data.metadata["actions_continuous_env_columns"] == list(ACTION_CONTINUOUS_ENV_COLUMNS)


def test_feature_standardizer_supports_sequence_features_and_clipping():
    features = np.asarray(
        [
            [[1.0, 10.0], [3.0, 14.0]],
            [[5.0, 18.0], [7.0, 22.0]],
        ],
        dtype=np.float32,
    )

    mean, std = fit_feature_standardizer(features)
    normalized = standardize_features(features, mean, std, clip=1.0)

    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert normalized.shape == features.shape
    assert np.max(np.abs(normalized)) <= 1.0


def test_default_trajectory_discriminator_uses_paper_critic_architecture():
    discriminator = TrajectoryDiscriminator(input_dim=10, dropout=0.2)
    linear_layers = [module for module in discriminator.net if isinstance(module, torch.nn.Linear)]
    dropout_layers = [module for module in discriminator.net if isinstance(module, torch.nn.Dropout)]

    assert [layer.out_features for layer in linear_layers] == [128, 128, 64, 1]
    assert [layer.p for layer in dropout_layers] == [0.2, 0.2, 0.2]


def test_transformer_policy_update_uses_rollout_consistent_eval_mode():
    torch.manual_seed(0)
    cfg = PSGAILConfig(
        policy_model="transformer",
        hidden_size=32,
        transformer_heads=4,
        transformer_dropout=0.5,
        batch_size=16,
        ppo_epochs=1,
        entropy_coef=0.0,
        value_coef=0.0,
    )
    policy = make_actor_critic(
        "transformer",
        obs_dim=12,
        hidden_size=32,
        transformer_heads=4,
        transformer_dropout=0.5,
    )
    observations = torch.randn(32, 12)
    policy.eval()
    with torch.no_grad():
        logits, values = policy(observations)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)

    rollout = _minimal_rollout(
        observations=observations.numpy(),
        actions=actions.numpy().astype(np.int64),
        old_log_probs=old_log_probs.numpy(),
        old_values=values.numpy(),
        returns=values.numpy(),
        advantages=np.ones(32, dtype=np.float32),
    )
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0)

    stats = update_policy(policy, optimizer, rollout, cfg, torch.device("cpu"))

    assert policy.training
    assert abs(stats["approx_kl"]) < 1e-7
    assert stats["clip_fraction"] == 0.0
    assert abs(stats["ratio_mean"] - 1.0) < 1e-7


def test_discrete_policy_distribution_respects_action_masks():
    torch.manual_seed(0)
    cfg = PSGAILConfig(action_mode="discrete")
    policy = make_actor_critic("mlp", obs_dim=3, hidden_size=8)
    obs = torch.randn(2, 3)
    masks = torch.tensor(
        [
            [False, True, False, False, False],
            [False, False, True, False, False],
        ],
        dtype=torch.bool,
    )

    dist, _values = policy_distribution_and_values(policy, obs, cfg, masks)

    assert torch.equal(dist.sample(), torch.tensor([1, 2]))
    assert torch.isneginf(dist.logits[0, 0]) or dist.logits[0, 0] < -1.0e8


def test_behavior_clone_pretrain_matches_continuous_actions():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = PSGAILConfig(
        action_mode="continuous",
        bc_pretrain_epochs=20,
        bc_pretrain_batch_size=16,
        bc_pretrain_learning_rate=1e-2,
        bc_pretrain_validation_fraction=0.0,
        hidden_size=32,
    )
    observations = np.random.uniform(-1.0, 1.0, size=(64, 4)).astype(np.float32)
    actions = np.tanh(
        np.stack(
            [
                0.8 * observations[:, 0] - 0.2 * observations[:, 1],
                -0.5 * observations[:, 2] + 0.3 * observations[:, 3],
            ],
            axis=1,
        )
    ).astype(np.float32)
    transitions = types.SimpleNamespace(
        policy_observations=observations,
        actions_continuous_env=actions,
    )
    policy = make_actor_critic(
        "mlp",
        obs_dim=4,
        hidden_size=32,
        action_mode="continuous",
        continuous_action_dim=2,
    )

    before = behavior_clone_pretrain(
        policy,
        transitions,
        cfg,
        torch.device("cpu"),
    )

    assert before["bc/train_mse"] < 0.05
    assert before["bc/train_mae"] < 0.2


def test_iq_learn_convergence_score_rewards_survival_and_penalizes_crashes():
    update_stats = {"bc_loss": 0.25}
    clean_eval = {"eval/mean_length": 120.0, "eval/mean_reward": 5.0, "eval/crashes": 0.0}
    crashed_eval = {"eval/mean_length": 120.0, "eval/mean_reward": 5.0, "eval/crashes": 2.0}

    clean_score = convergence_score(
        update_stats,
        clean_eval,
        crash_penalty=25.0,
        bc_score_weight=1.0,
    )
    crashed_score = convergence_score(
        update_stats,
        crashed_eval,
        crash_penalty=25.0,
        bc_score_weight=1.0,
    )

    assert clean_score == pytest.approx(124.75)
    assert crashed_score == pytest.approx(74.75)


def test_iq_learn_convergence_target_requires_length_bc_and_crash_budget():
    assert convergence_reached(
        {"bc_loss": 0.02},
        {"eval/mean_length": 180.0, "eval/crashes": 0.0},
        target_eval_mean_length=150.0,
        target_bc_loss=0.05,
        max_eval_crashes=0.0,
    )
    assert not convergence_reached(
        {"bc_loss": 0.08},
        {"eval/mean_length": 180.0, "eval/crashes": 0.0},
        target_eval_mean_length=150.0,
        target_bc_loss=0.05,
        max_eval_crashes=0.0,
    )


def test_sequence_feature_local_deltas_remove_cumulative_progress():
    sequence = np.zeros((1, 4, 5), dtype=np.float32)
    sequence[0, :, -3:] = np.asarray(
        [
            [100.0, 2.0, 10.0],
            [103.0, 3.0, 12.0],
            [107.0, 2.5, 11.0],
            [110.0, 2.0, 11.5],
        ],
        dtype=np.float32,
    )

    transformed = transform_sequence_features(sequence, mode="local_deltas")

    np.testing.assert_allclose(transformed[0, 0, -3:], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(transformed[0, 1:, -3:], [[3.0, 1.0, 2.0], [4.0, -0.5, -1.0], [3.0, -0.5, 0.5]])


def test_build_sequence_windows_can_return_transition_indices():
    features = np.arange(10, dtype=np.float32).reshape(5, 2)
    trajectory_ids = np.asarray([1, 1, 2, 2, 2], dtype=np.int64)

    windows, last_indices, window_indices = build_sequence_windows(
        features,
        trajectory_ids,
        sequence_length=2,
        stride=1,
        return_window_indices=True,
    )

    assert windows.shape == (3, 2, 2)
    np.testing.assert_array_equal(last_indices, [1, 3, 4])
    np.testing.assert_array_equal(window_indices, [[0, 1], [2, 3], [3, 4]])


def test_sequence_reward_assignment_last_preserves_existing_sparse_credit():
    rewards = sequence_rewards_to_transition_rewards(
        np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        num_transitions=4,
        sequence_last_indices=np.asarray([1, 2, 2], dtype=np.int64),
        assignment="last",
    )

    np.testing.assert_allclose(rewards, [0.0, 10.0, 25.0, 0.0], rtol=1e-6)


def test_sequence_reward_assignment_mean_densifies_overlapping_windows():
    rewards = sequence_rewards_to_transition_rewards(
        np.asarray([10.0, 20.0], dtype=np.float32),
        num_transitions=4,
        sequence_last_indices=np.asarray([2, 3], dtype=np.int64),
        sequence_transition_indices=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        assignment="mean",
    )

    np.testing.assert_allclose(rewards, [10.0, 15.0, 15.0, 20.0], rtol=1e-6)


def test_refresh_rollout_rewards_can_use_dense_sequence_assignment():
    class FirstTokenDiscriminator(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features[:, 0, 0]

    cfg = PSGAILConfig(
        discriminator_loss="wgan_gp",
        sequence_reward_assignment="mean",
        sequence_reward_coef=1.0,
        wgan_reward_center=False,
        wgan_reward_clip=0.0,
        normalize_gail_reward=False,
        gail_reward_clip=0.0,
        final_reward_clip=0.0,
    )
    rollout = _minimal_rollout(
        observations=np.zeros((4, 1), dtype=np.float32),
        actions=np.zeros((4,), dtype=np.int64),
        old_log_probs=np.zeros((4,), dtype=np.float32),
        old_values=np.zeros((4,), dtype=np.float32),
        returns=np.zeros((4,), dtype=np.float32),
        advantages=np.zeros((4,), dtype=np.float32),
        sequence_features=np.asarray(
            [
                [[10.0], [0.0], [0.0]],
                [[20.0], [0.0], [0.0]],
            ],
            dtype=np.float32,
        ),
        sequence_last_indices=np.asarray([2, 3], dtype=np.int64),
        sequence_transition_indices=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
    )

    refreshed = refresh_rollout_rewards(
        rollout,
        None,
        cfg,
        torch.device("cpu"),
        sequence_discriminator=FirstTokenDiscriminator(),
    )

    np.testing.assert_allclose(refreshed.gail_rewards_raw, [10.0, 15.0, 15.0, 20.0], rtol=1e-6)


def test_refresh_rollout_rewards_applies_discriminator_feature_normalizer():
    class FirstFeatureDiscriminator(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features[:, 0]

    cfg = PSGAILConfig(
        discriminator_loss="bce",
        normalize_gail_reward=False,
        gail_reward_clip=0.0,
        final_reward_clip=0.0,
        discriminator_feature_clip=0.0,
    )
    rollout = _minimal_rollout(
        observations=np.zeros((1, 1), dtype=np.float32),
        actions=np.zeros((1,), dtype=np.int64),
        old_log_probs=np.zeros((1,), dtype=np.float32),
        old_values=np.zeros((1,), dtype=np.float32),
        returns=np.zeros((1,), dtype=np.float32),
        advantages=np.zeros((1,), dtype=np.float32),
        generator_features=np.asarray([[11.0, 0.0]], dtype=np.float32),
    )

    refreshed = refresh_rollout_rewards(
        rollout,
        FirstFeatureDiscriminator(),
        cfg,
        torch.device("cpu"),
        discriminator_normalizer=(
            np.asarray([10.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0], dtype=np.float32),
        ),
    )

    expected = torch.nn.functional.softplus(torch.tensor([1.0])).numpy()
    np.testing.assert_allclose(refreshed.gail_rewards_raw, expected, rtol=1e-6)


def test_wgan_reward_center_and_clip_are_applied_before_env_penalty():
    cfg = PSGAILConfig(
        discriminator_loss="wgan_gp",
        wgan_reward_center=True,
        wgan_reward_clip=2.0,
        normalize_gail_reward=False,
        gail_reward_clip=0.0,
        final_reward_clip=0.0,
    )
    rewards, shaped_gail = shape_rollout_rewards(
        np.asarray([10.0, 12.0, 50.0], dtype=np.float32),
        np.asarray([0.0, -5.0, 0.0], dtype=np.float32),
        cfg,
    )

    np.testing.assert_allclose(shaped_gail, [-2.0, -2.0, 2.0], rtol=1e-6)
    np.testing.assert_allclose(rewards, [-2.0, -7.0, 2.0], rtol=1e-6)


def test_wgan_gp_discriminator_update_returns_critic_metrics():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = PSGAILConfig(
        discriminator_loss="wgan_gp",
        disc_updates_per_round=1,
        disc_batch_size=4,
        wgan_gp_lambda=2.0,
    )
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
        torch.nn.Flatten(0),
    )
    expert = np.ones((8, 2), dtype=np.float32)
    generator = -np.ones((8, 2), dtype=np.float32)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    stats = update_discriminator(discriminator, optimizer, expert, generator, cfg, torch.device("cpu"))

    assert np.isfinite(stats["disc_loss"])
    assert np.isfinite(stats["wgan_loss"])
    assert np.isfinite(stats["gradient_penalty"])
    assert "critic_gap" in stats


def test_wgan_gp_supports_sequence_discriminator_inputs():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = PSGAILConfig(
        discriminator_loss="wgan_gp",
        disc_updates_per_round=1,
        disc_batch_size=4,
        wgan_gp_lambda=2.0,
    )
    discriminator = SequenceTrajectoryDiscriminator(input_dim=3, hidden_size=8)
    expert = np.ones((8, 4, 3), dtype=np.float32)
    generator = -np.ones((8, 4, 3), dtype=np.float32)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    stats = update_discriminator(discriminator, optimizer, expert, generator, cfg, torch.device("cpu"))

    assert np.isfinite(stats["disc_loss"])
    assert np.isfinite(stats["gradient_penalty"])


def test_airl_controlled_vehicle_curriculum_matches_ps_gail_schedule():
    cfg = PSGAILConfig(
        controlled_vehicle_curriculum=True,
        control_all_vehicles=True,
        percentage_controlled_vehicles=1.0,
        initial_controlled_vehicles=0.2,
        final_controlled_vehicles=0.8,
        controlled_vehicle_curriculum_rounds=4,
    )

    fractions = [
        airl_config_for_round(cfg, round_idx).percentage_controlled_vehicles
        for round_idx in range(1, 6)
    ]

    np.testing.assert_allclose(fractions, [0.2, 0.4, 0.6, 0.8, 0.8], rtol=1e-6)
    assert airl_config_for_round(cfg, 1).control_all_vehicles is False


def test_absolute_controlled_vehicle_curriculum_is_shared_by_ps_gail_and_airl():
    cfg = PSGAILConfig(
        total_rounds=6,
        controlled_vehicle_curriculum=True,
        initial_controlled_vehicles=10,
        final_controlled_vehicles=100,
        controlled_vehicle_curriculum_rounds=6,
        initial_rollout_target_agent_steps=10_000,
        final_rollout_target_agent_steps=40_000,
        rollout_target_agent_steps_curriculum_rounds=6,
        initial_gamma=0.95,
        final_gamma=0.99,
        gamma_curriculum_rounds=6,
    )

    round_cfgs = [ps_gail_config_for_round(cfg, round_idx) for round_idx in range(1, 7)]
    airl_round_cfgs = [airl_config_for_round(cfg, round_idx) for round_idx in range(1, 7)]

    np.testing.assert_allclose(
        [item.percentage_controlled_vehicles for item in round_cfgs],
        [10.0, 28.0, 46.0, 64.0, 82.0, 100.0],
    )
    assert [item.percentage_controlled_vehicles for item in airl_round_cfgs] == [
        item.percentage_controlled_vehicles for item in round_cfgs
    ]
    np.testing.assert_allclose(
        [item.gamma for item in round_cfgs],
        [0.95, 0.958, 0.966, 0.974, 0.982, 0.99],
        rtol=1e-6,
    )
    assert [item.rollout_target_agent_steps for item in round_cfgs] == [
        10_000,
        16_000,
        22_000,
        28_000,
        34_000,
        40_000,
    ]
    assert all(item.control_all_vehicles is False for item in round_cfgs)


def test_constant_rollout_target_agent_steps_remains_supported():
    cfg = PSGAILConfig(
        controlled_vehicle_curriculum=True,
        initial_controlled_vehicles=20,
        final_controlled_vehicles=40,
        controlled_vehicle_curriculum_rounds=3,
        rollout_target_agent_steps=12_345,
    )

    ps_gail_counts = [
        ps_gail_config_for_round(cfg, round_idx).percentage_controlled_vehicles
        for round_idx in range(1, 5)
    ]
    airl_counts = [
        airl_config_for_round(cfg, round_idx).percentage_controlled_vehicles
        for round_idx in range(1, 5)
    ]

    assert ps_gail_counts == [20.0, 30.0, 40.0, 40.0]
    assert airl_counts == ps_gail_counts
    assert airl_config_for_round(cfg, 4).rollout_target_agent_steps == 12_345
