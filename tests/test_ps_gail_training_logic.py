import json
import sys
import types
from pathlib import Path

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
from scripts_gail.ps_gail.training import evaluation as eval_mod
from scripts_gail.ps_gail.validation import best_checkpoint_payload
from scripts_gail.ps_gail.validation import scored_validation_metrics
from scripts_gail.ps_gail.validation import validation_cost_and_score
from scripts_gail.ps_gail.models import (
    SequenceTrajectoryDiscriminator,
    TrajectoryDiscriminator,
    make_actor_critic,
)
from scripts_gail.ps_gail.trainer import (
    RolloutBatch,
    central_critic_observation_dim,
    central_critic_observations,
    combine_primary_env_challenge_rewards,
    policy_distribution_and_values,
    policy_distribution_values_memory,
    player_challenge_bonus,
    player_challenge_payoff,
    player_challenge_pressure_from_metric,
    refresh_rollout_rewards,
    sequence_rewards_to_transition_rewards,
    select_hard_discriminator_examples,
    shape_rollout_rewards,
    subsample_rollout_for_training,
    update_discriminator,
    update_policy,
)
from scripts_gail.train_simple_ps_gail import behavior_clone_pretrain
from scripts_gail.train_simple_ps_gail import config_for_round as ps_gail_config_for_round
from scripts_gail.train_simple_ps_gail import parse_args as ps_gail_parse_args
from scripts_gail.train_simple_airl import AIRLReward
from scripts_gail.train_simple_airl import _airl_wgan_gradient_penalty
from scripts_gail.train_simple_airl import _build_recurrent_trajectory_index
from scripts_gail.train_simple_airl import _recurrent_policy_log_probs_for_indices
from scripts_gail.train_simple_airl import append_airl_replay
from scripts_gail.train_simple_airl import concat_airl_replay
from scripts_gail.train_simple_airl import config_for_round as airl_config_for_round
from scripts_gail.train_simple_airl import load_airl_resume_checkpoint
from scripts_gail.train_simple_airl import parse_args as airl_parse_args
from scripts_gail.train_simple_airl import refresh_airl_rewards
from scripts_gail.train_simple_airl import update_reward_model
from scripts_gail.train_simple_iq_learn import convergence_reached, convergence_score


def _minimal_rollout(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    old_values: np.ndarray,
    returns: np.ndarray,
    advantages: np.ndarray,
    next_observations: np.ndarray | None = None,
    critic_observations: np.ndarray | None = None,
    next_critic_observations: np.ndarray | None = None,
    trajectory_ids: np.ndarray | None = None,
    dones: np.ndarray | None = None,
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
        next_policy_observations=(
            next_observations.astype(np.float32)
            if next_observations is not None
            else observations.astype(np.float32)
        ),
        critic_observations=(
            critic_observations.astype(np.float32)
            if critic_observations is not None
            else observations.astype(np.float32)
        ),
        next_critic_observations=(
            next_critic_observations.astype(np.float32)
            if next_critic_observations is not None
            else (
                critic_observations.astype(np.float32)
                if critic_observations is not None
                else observations.astype(np.float32)
            )
        ),
        actions=actions,
        action_masks=np.ones((n, 5), dtype=bool),
        old_log_probs=old_log_probs.astype(np.float32),
        old_values=old_values.astype(np.float32),
        trajectory_ids=(
            trajectory_ids
            if trajectory_ids is not None
            else np.arange(n, dtype=np.int32)
        ),
        dones=(
            dones.astype(bool)
            if dones is not None
            else np.ones(n, dtype=bool)
        ),
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
    assert sampled.critic_observations.shape[0] == sampled.num_agent_steps
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

    assert data.policy_observations.shape == (3, 6)
    assert data.next_policy_observations.shape == (3, 6)
    assert data.actions_continuous_env.shape == (3, 2)
    assert data.actions_steering_acceleration is not None
    assert data.actions_steering_acceleration.shape == (3, 2)
    assert np.all(np.isfinite(data.actions_continuous_env))
    assert data.trajectory_ids.shape == (3,)
    assert data.metadata["trajectory_id_schema"] == "file_index:vehicle_id"
    assert data.metadata["continuous_action_dim"] == 2
    assert data.metadata["actions_continuous_env_columns"] == list(ACTION_CONTINUOUS_ENV_COLUMNS)
    assert data.metadata["sampling"] == "trajectory_preserving_without_replacement"


def test_action_conditioned_loader_caps_by_complete_trajectories(tmp_path):
    path = tmp_path / "expert_multi_vehicle.npz"
    observations = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    arrays = {
        "observations": observations,
        "next_observations": observations + 0.5,
        "trajectory_states": np.stack(
            [
                np.arange(6, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                np.ones(6, dtype=np.float32),
            ],
            axis=1,
        ),
        "vehicle_ids": np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64),
        "timesteps": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "dones": np.asarray([False, False, True, False, False, True], dtype=bool),
        "rewards": np.zeros(6, dtype=np.float32),
        ACTION_CONTINUOUS_ENV_KEY: np.zeros((6, 2), dtype=np.float32),
        ACTION_STEERING_ACCELERATION_KEY: np.zeros((6, 2), dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)

    data = load_expert_transition_data(str(path), max_samples=4, seed=7)

    assert len(data.policy_observations) in {3, 6}
    for trajectory_id in dict.fromkeys(data.trajectory_ids.tolist()):
        steps = data.timesteps[data.trajectory_ids == trajectory_id]
        np.testing.assert_array_equal(steps, [0, 1, 2])
    assert data.metadata["sampling"] == "trajectory_preserving_without_replacement"


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


def test_centralized_critic_uses_separate_observation_path():
    torch.manual_seed(0)
    policy = make_actor_critic(
        "mlp",
        obs_dim=3,
        hidden_size=8,
        centralized_critic=True,
        critic_obs_dim=7,
    )
    obs = torch.randn(5, 3)
    critic_a = torch.randn(5, 7)
    critic_b = critic_a + 0.5

    logits_a, values_a = policy(obs, critic_a)
    logits_b, values_b = policy(obs, critic_b)

    assert logits_a.shape == (5, 5)
    assert values_a.shape == (5,)
    torch.testing.assert_close(logits_a, logits_b)
    assert not torch.allclose(values_a, values_b)


def test_attention_centralized_critic_pools_vehicle_set_without_changing_actor():
    torch.manual_seed(0)
    policy = make_actor_critic(
        "mlp",
        obs_dim=3,
        hidden_size=16,
        centralized_critic=True,
        critic_obs_dim=2 * 5 + 4,
        central_critic_pooling="attention",
        central_critic_max_vehicles=2,
        central_critic_attention_heads=4,
    )
    policy.eval()
    obs = torch.randn(4, 3)
    vehicle_a = torch.tensor([1.0, 2.0, 0.0, 3.0, 0.0])
    vehicle_b = torch.tensor([1.0, -1.0, 1.0, 0.0, 2.0])
    context = torch.tensor([8.0, 0.1, 2.0, 4.5])
    critic_a = torch.cat([vehicle_a, vehicle_b, context]).repeat(4, 1)
    critic_b = torch.cat([vehicle_b, vehicle_a, context]).repeat(4, 1)

    logits_a, values_a = policy(obs, critic_a)
    logits_b, values_b = policy(obs, critic_b)

    torch.testing.assert_close(logits_a, logits_b)
    torch.testing.assert_close(values_a, values_b, atol=1e-6, rtol=1e-6)


def test_update_policy_trains_centralized_critic_values():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = PSGAILConfig(
        centralized_critic=True,
        batch_size=8,
        ppo_epochs=1,
        entropy_coef=0.0,
        value_coef=0.5,
    )
    policy = make_actor_critic(
        "mlp",
        obs_dim=3,
        hidden_size=16,
        centralized_critic=True,
        critic_obs_dim=7,
    )
    observations = torch.randn(16, 3)
    critic_observations = torch.randn(16, 7)
    policy.eval()
    with torch.no_grad():
        logits, values = policy(observations, critic_observations)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)

    rollout = _minimal_rollout(
        observations=observations.numpy(),
        critic_observations=critic_observations.numpy(),
        actions=actions.numpy().astype(np.int64),
        old_log_probs=old_log_probs.numpy(),
        old_values=values.numpy(),
        returns=(values.numpy() + 1.0).astype(np.float32),
        advantages=np.ones(16, dtype=np.float32),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    stats = update_policy(policy, optimizer, rollout, cfg, torch.device("cpu"))

    assert np.isfinite(stats["value_loss"])
    assert stats["value_loss"] > 0.0


def test_central_critic_observations_encode_per_agent_scene_context():
    class Vehicle:
        def __init__(self, x, y, vx, vy, speed, heading, vehicle_id):
            self.position = np.asarray([x, y], dtype=np.float32)
            self.velocity = np.asarray([vx, vy], dtype=np.float32)
            self.speed = speed
            self.heading = heading
            self.WIDTH = 2.0
            self.LENGTH = 4.5
            self.vehicle_ID = vehicle_id

    vehicles = [
        Vehicle(10.0, 0.0, 1.0, 0.0, 10.0, 0.1, 1),
        Vehicle(20.0, 1.0, 2.0, 0.0, 11.0, 0.2, 2),
        Vehicle(30.0, 2.0, 3.0, 0.0, 12.0, 0.3, 3),
    ]
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            controlled_vehicles=vehicles[:2],
            road=types.SimpleNamespace(vehicles=vehicles),
        )
    )
    cfg = PSGAILConfig(
        centralized_critic=True,
        central_critic_max_vehicles=3,
        central_critic_include_local_obs=True,
    )
    local_obs = np.arange(8, dtype=np.float32).reshape(2, 4)

    critic_obs = central_critic_observations(env, cfg, local_obs)

    assert critic_obs.shape == (2, central_critic_observation_dim(4, cfg))
    np.testing.assert_allclose(critic_obs[0, -4:], local_obs[0])
    np.testing.assert_allclose(critic_obs[1, -4:], local_obs[1])
    assert critic_obs[0, 1] == pytest.approx(0.0)
    assert critic_obs[1, 1] == pytest.approx(0.0)
    assert not np.allclose(critic_obs[0], critic_obs[1])


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


def test_wgan_reward_normalization_is_disabled_unless_explicitly_allowed():
    raw = np.asarray([-15.0, -10.0, -20.0], dtype=np.float32)
    penalties = np.asarray([0.0, -2.0, 0.0], dtype=np.float32)
    cfg = PSGAILConfig(
        discriminator_loss="wgan_gp",
        normalize_gail_reward=True,
        allow_wgan_reward_normalization=False,
        gail_reward_clip=5.0,
        final_reward_clip=0.0,
    )

    rewards, shaped_gail = shape_rollout_rewards(raw, penalties, cfg)

    np.testing.assert_allclose(shaped_gail, raw, rtol=1e-6)
    np.testing.assert_allclose(rewards, [-15.0, -12.0, -20.0], rtol=1e-6)


def test_wgan_reward_normalization_can_be_enabled_for_ablation():
    cfg = PSGAILConfig(
        discriminator_loss="wgan_gp",
        normalize_gail_reward=True,
        allow_wgan_reward_normalization=True,
        gail_reward_clip=0.0,
        final_reward_clip=0.0,
    )

    _rewards, shaped_gail = shape_rollout_rewards(
        np.asarray([-15.0, -10.0, -20.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        cfg,
    )

    assert shaped_gail.mean() == pytest.approx(0.0, abs=1e-6)
    assert shaped_gail.std() == pytest.approx(1.0, rel=1e-6)


def test_bce_gail_reward_clip_still_applies_to_logistic_rewards():
    cfg = PSGAILConfig(
        discriminator_loss="bce",
        normalize_gail_reward=False,
        gail_reward_clip=5.0,
        final_reward_clip=0.0,
    )

    _rewards, shaped_gail = shape_rollout_rewards(
        np.asarray([-15.0, -10.0, 20.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        cfg,
    )

    np.testing.assert_allclose(shaped_gail, [-5.0, -5.0, 5.0], rtol=1e-6)


def test_player_challenge_disabled_defaults_leave_rewards_unchanged():
    cfg = PSGAILConfig(final_reward_clip=0.0)
    primary = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    penalties = np.asarray([0.0, -0.25, 0.25], dtype=np.float32)

    rewards, bonuses = combine_primary_env_challenge_rewards(
        primary,
        penalties,
        cfg,
        challenge_payoffs=np.full(3, 100.0, dtype=np.float32),
    )

    assert cfg.enable_hard_example_selection is False
    assert cfg.enable_player_challenge_reward is False
    np.testing.assert_allclose(bonuses, np.zeros(3, dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(rewards, primary + penalties, rtol=1e-6)


def test_player_challenge_pressure_uses_pressure_over_risk():
    cfg = PSGAILConfig(
        challenge_ttc_weight=0.5,
        challenge_gap_weight=0.5,
        challenge_crash_weight=4.0,
        challenge_offroad_weight=2.0,
    )
    metric = {
        "min_ttc": 2.0,
        "ttc_target": 3.0,
        "ttc_floor": 1.0,
        "min_gap": 8.0,
        "gap_target": 12.0,
        "gap_floor": 4.0,
        "crashed": False,
        "offroad": False,
    }

    pressure, _ttc_target, _gap_target = player_challenge_pressure_from_metric(metric, cfg)
    safe_payoff = player_challenge_payoff(pressure, crash_rate_ema=0.0, offroad_rate_ema=0.0, cfg=cfg)
    risky_payoff = player_challenge_payoff(pressure, crash_rate_ema=0.5, offroad_rate_ema=0.5, cfg=cfg)
    unsafe_pressure, _ttc_target, _gap_target = player_challenge_pressure_from_metric(
        {**metric, "min_ttc": 0.25},
        cfg,
    )

    assert pressure == pytest.approx(0.5)
    assert 0.0 < risky_payoff < safe_payoff
    assert unsafe_pressure == 0.0


def test_player_challenge_bonus_is_capped_by_primary_reward_scale():
    cfg = PSGAILConfig(
        enable_player_challenge_reward=True,
        challenge_reward_coef=100.0,
        challenge_reward_clip=100.0,
        challenge_max_primary_reward_fraction=0.10,
        challenge_expert_like_quantile=0.0,
    )
    primary = np.asarray([1.0, 2.0, 0.5], dtype=np.float32)
    payoffs = np.full(3, 10.0, dtype=np.float32)

    bonuses = player_challenge_bonus(payoffs, primary, cfg)

    caps = 0.10 * np.maximum(np.abs(primary), float(np.mean(np.abs(primary))))
    assert np.all(bonuses >= 0.0)
    assert np.all(bonuses <= caps + 1.0e-6)
    np.testing.assert_allclose(bonuses, caps, rtol=1e-6)


def test_player_challenge_targets_follow_scene_idm_presets():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for package_name in (
        "highway_env",
        "highway_env.ngsim_utils",
        "highway_env.ngsim_utils.core",
    ):
        sys.modules.setdefault(package_name, types.ModuleType(package_name))
    constants_name = "highway_env.ngsim_utils.core.constants"
    constants_spec = importlib.util.spec_from_file_location(
        constants_name,
        root / "highway_env" / "ngsim_utils" / "core" / "constants.py",
    )
    constants_module = importlib.util.module_from_spec(constants_spec)
    sys.modules[constants_name] = constants_module
    assert constants_spec.loader is not None
    constants_spec.loader.exec_module(constants_module)
    config_name = "highway_env.ngsim_utils.core.config"
    config_spec = importlib.util.spec_from_file_location(
        config_name,
        root / "highway_env" / "ngsim_utils" / "core" / "config.py",
    )
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules[config_name] = config_module
    assert config_spec.loader is not None
    config_spec.loader.exec_module(config_module)

    targets = {}
    for scene in ("us-101", "japanese"):
        idm_parameters = config_module.resolve_idm_parameters(scene, {})
        cfg = {
            "interaction_ttc_target": 0.0,
            "interaction_ttc_margin": 0.75,
            "interaction_ttc_floor": 0.0,
            "interaction_gap_target": 0.0,
            "interaction_gap_floor": 0.0,
        }
        speed = 12.5
        targets[scene] = config_module.interaction_metric_targets_from_idm(
            idm_parameters,
            cfg,
            speed=speed,
        )
        idm = idm_parameters["idm"]
        np.testing.assert_allclose(
            targets[scene],
            [
                float(idm["time_headway"]) + 0.75,
                max(1.0, 0.5 * float(idm["time_headway"])),
                float(idm["min_gap"]) + speed * float(idm["time_headway"]),
                float(idm["min_gap"]),
            ],
            rtol=1e-6,
        )

    assert targets["us-101"] != targets["japanese"]


def test_hard_selector_feeds_only_selected_partial_subset_to_discriminator():
    class CountingDiscriminator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
            self.train_forward_sizes: list[int] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            flat = x.reshape(x.shape[0], -1)
            if self.training and torch.is_grad_enabled():
                self.train_forward_sizes.append(int(flat.shape[0]))
            return self.linear(flat).reshape(-1)

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = PSGAILConfig(
        enable_hard_example_selection=True,
        hard_example_candidate_samples=40,
        hard_example_selected_fraction=0.25,
        hard_example_uniform_mix=0.0,
        hard_example_min_samples=1,
        discriminator_loss="wgan_gp",
        disc_updates_per_round=1,
        disc_batch_size=128,
        wgan_gp_lambda=0.0,
    )
    discriminator = CountingDiscriminator()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    expert = np.stack([np.linspace(-1.0, 1.0, 80), np.zeros(80)], axis=1).astype(np.float32)
    generator = np.stack([np.linspace(1.0, -1.0, 80), np.ones(80)], axis=1).astype(np.float32)

    stats = update_discriminator(discriminator, optimizer, expert, generator, cfg, torch.device("cpu"))

    assert stats["hard_selector_enabled"] == 1.0
    assert stats["hard_selector_candidate_samples"] == 40.0
    assert stats["hard_selector_selected_samples"] == 10.0
    assert stats["hard_selector_selected_samples"] < stats["hard_selector_candidate_samples"]
    assert discriminator.train_forward_sizes
    assert max(discriminator.train_forward_sizes) == int(stats["hard_selector_selected_samples"])


def test_select_hard_discriminator_examples_returns_partial_candidate_subset():
    class FirstFeatureDiscriminator(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], -1)[:, 0]

    np.random.seed(1)
    cfg = PSGAILConfig(
        enable_hard_example_selection=True,
        hard_example_candidate_samples=50,
        hard_example_selected_fraction=0.20,
        hard_example_uniform_mix=0.0,
        hard_example_min_samples=1,
    )
    expert = np.arange(200, dtype=np.float32).reshape(100, 2)
    generator = -expert.copy()

    selected_expert, selected_generator, stats = select_hard_discriminator_examples(
        FirstFeatureDiscriminator(),
        expert,
        generator,
        cfg,
        torch.device("cpu"),
    )

    assert selected_expert.shape == (10, 2)
    assert selected_generator.shape == (10, 2)
    assert stats["hard_selector_selected_samples"] == 10.0
    assert stats["hard_selector_selected_samples"] < stats["hard_selector_candidate_samples"]
    assert len(selected_expert) < len(expert)


def test_one_round_gail_challenge_smoke_path_refreshes_rewards_and_selector():
    class FirstFeatureDiscriminator(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            flat = x.reshape(x.shape[0], -1)
            return flat[:, 0]

    cfg = PSGAILConfig(
        enable_player_challenge_reward=True,
        enable_hard_example_selection=True,
        challenge_reward_coef=10.0,
        challenge_reward_clip=10.0,
        challenge_max_primary_reward_fraction=0.10,
        challenge_expert_like_quantile=0.0,
        normalize_gail_reward=False,
        gail_reward_clip=0.0,
        final_reward_clip=0.0,
        hard_example_candidate_samples=12,
        hard_example_selected_fraction=0.50,
        hard_example_min_samples=1,
        hard_example_uniform_mix=0.0,
        discriminator_loss="bce",
        disc_updates_per_round=1,
        disc_batch_size=16,
    )
    rollout = _minimal_rollout(
        observations=np.ones((12, 2), dtype=np.float32),
        actions=np.zeros((12, 2), dtype=np.float32),
        old_log_probs=np.zeros(12, dtype=np.float32),
        old_values=np.zeros(12, dtype=np.float32),
        returns=np.zeros(12, dtype=np.float32),
        advantages=np.zeros(12, dtype=np.float32),
        generator_features=np.ones((12, 2), dtype=np.float32),
    )
    rollout.challenge_payoffs = np.full(12, 1.0, dtype=np.float32)

    refreshed = refresh_rollout_rewards(rollout, FirstFeatureDiscriminator(), cfg, torch.device("cpu"))

    assert np.any(refreshed.challenge_bonuses > 0.0)
    assert np.all(
        refreshed.challenge_bonuses
        <= 0.10
        * np.maximum(
            np.abs(refreshed.gail_rewards_normalized),
            float(np.mean(np.abs(refreshed.gail_rewards_normalized))),
        )
        + 1.0e-6
    )
    discriminator = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Flatten(0))
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    stats = update_discriminator(
        discriminator,
        optimizer,
        np.zeros((24, 2), dtype=np.float32),
        refreshed.generator_features,
        cfg,
        torch.device("cpu"),
    )

    assert stats["hard_selector_enabled"] == 1.0
    assert stats["hard_selector_selected_samples"] < stats["hard_selector_candidate_samples"]


def test_airl_wgan_uses_shaped_logits_without_rollout_normalization_or_generic_clip():
    class FixedReward(torch.nn.Module):
        def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return torch.full((obs.shape[0],), 123.0, dtype=torch.float32, device=obs.device)

        def shaped_logits(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            next_obs: torch.Tensor,
            dones: torch.Tensor,
            *,
            gamma: float,
        ) -> torch.Tensor:
            del actions, next_obs, dones, gamma
            return torch.tensor([-15.0, -10.0, -20.0], dtype=torch.float32, device=obs.device)

    cfg = PSGAILConfig(
        discriminator_loss="wgan_gp",
        normalize_gail_reward=True,
        allow_wgan_reward_normalization=False,
        gail_reward_clip=5.0,
        final_reward_clip=0.0,
    )
    rollout = _minimal_rollout(
        observations=np.zeros((3, 2), dtype=np.float32),
        actions=np.zeros((3, 2), dtype=np.float32),
        old_log_probs=np.zeros(3, dtype=np.float32),
        old_values=np.zeros(3, dtype=np.float32),
        returns=np.zeros(3, dtype=np.float32),
        advantages=np.zeros(3, dtype=np.float32),
    )

    refreshed = refresh_airl_rewards(rollout, FixedReward(), cfg, torch.device("cpu"))

    np.testing.assert_allclose(refreshed.gail_rewards_raw, [-15.0, -10.0, -20.0], rtol=1e-6)
    np.testing.assert_allclose(refreshed.gail_rewards_normalized, [-15.0, -10.0, -20.0], rtol=1e-6)
    np.testing.assert_allclose(refreshed.rewards, [-15.0, -10.0, -20.0], rtol=1e-6)


def test_airl_policy_reward_mode_exposes_old_log_prob_feedback_loop():
    class ConstantAIRLReward(torch.nn.Module):
        def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)

        def shaped_logits(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            next_obs: torch.Tensor,
            dones: torch.Tensor,
            *,
            gamma: float,
        ) -> torch.Tensor:
            del actions, next_obs, dones, gamma
            return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)

    rollout = _minimal_rollout(
        observations=np.zeros((3, 2), dtype=np.float32),
        actions=np.zeros((3, 2), dtype=np.float32),
        old_log_probs=np.asarray([0.0, -5.0, -10.0], dtype=np.float32),
        old_values=np.zeros(3, dtype=np.float32),
        returns=np.zeros(3, dtype=np.float32),
        advantages=np.zeros(3, dtype=np.float32),
    )
    base_cfg = dict(
        discriminator_loss="wgan_gp",
        normalize_gail_reward=False,
        gail_reward_clip=0.0,
        final_reward_clip=0.0,
    )

    discriminator_cfg = PSGAILConfig(
        **base_cfg,
        airl_policy_reward_mode="discriminator",
    )
    shaped_cfg = PSGAILConfig(
        **base_cfg,
        airl_policy_reward_mode="shaped",
    )
    default_cfg = PSGAILConfig(**base_cfg)

    discriminator_refreshed = refresh_airl_rewards(
        rollout,
        ConstantAIRLReward(),
        discriminator_cfg,
        torch.device("cpu"),
    )
    shaped_refreshed = refresh_airl_rewards(
        rollout,
        ConstantAIRLReward(),
        shaped_cfg,
        torch.device("cpu"),
    )
    default_refreshed = refresh_airl_rewards(
        rollout,
        ConstantAIRLReward(),
        default_cfg,
        torch.device("cpu"),
    )

    np.testing.assert_allclose(discriminator_refreshed.gail_rewards_raw, [0.0, 5.0, 10.0], rtol=1e-6)
    np.testing.assert_allclose(discriminator_refreshed.rewards, [0.0, 5.0, 10.0], rtol=1e-6)
    np.testing.assert_allclose(shaped_refreshed.gail_rewards_normalized, [0.0, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(shaped_refreshed.rewards, [0.0, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(default_refreshed.rewards, shaped_refreshed.rewards, rtol=1e-6)


def test_airl_resume_rejects_gail_checkpoint_without_reward_state(tmp_path):
    torch.manual_seed(0)
    policy = make_actor_critic(
        "mlp",
        obs_dim=3,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
    )
    reward_model = AIRLReward(obs_dim=3, action_dim=2, hidden_sizes=(8,))
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"policy_state_dict": policy.state_dict(), "round": 1}, checkpoint_path)

    with pytest.raises(RuntimeError, match="missing reward_state_dict"):
        load_airl_resume_checkpoint(
            resume_checkpoint=str(checkpoint_path),
            policy=policy,
            reward_model=reward_model,
            device=torch.device("cpu"),
        )


def test_airl_resume_missing_reward_state_requires_explicit_override(tmp_path):
    torch.manual_seed(0)
    source_policy = make_actor_critic(
        "mlp",
        obs_dim=3,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
    )
    policy = make_actor_critic(
        "mlp",
        obs_dim=3,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
    )
    reward_model = AIRLReward(obs_dim=3, action_dim=2, hidden_sizes=(8,))
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"policy_state_dict": source_policy.state_dict(), "round": 1}, checkpoint_path)

    with pytest.warns(RuntimeWarning, match="no reward_state_dict"):
        checkpoint = load_airl_resume_checkpoint(
            resume_checkpoint=str(checkpoint_path),
            policy=policy,
            reward_model=reward_model,
            device=torch.device("cpu"),
            allow_missing_reward_state=True,
        )

    assert checkpoint["round"] == 1


def test_airl_wgan_gradient_penalty_backprops_through_reward_model_only():
    torch.manual_seed(0)
    reward_model = AIRLReward(obs_dim=3, action_dim=2, hidden_sizes=(8,))
    expert_obs = torch.randn(4, 3)
    expert_actions = torch.randn(4, 2)
    expert_next_obs = torch.randn(4, 3)
    expert_dones = torch.zeros(4)
    gen_obs = torch.randn(4, 3)
    gen_actions = torch.randn(4, 2)
    gen_next_obs = torch.randn(4, 3)
    gen_dones = torch.ones(4)

    gradient_penalty = _airl_wgan_gradient_penalty(
        reward_model,
        expert_obs,
        expert_actions,
        expert_next_obs,
        expert_dones,
        gen_obs,
        gen_actions,
        gen_next_obs,
        gen_dones,
        gamma=0.99,
        gp_lambda=2.0,
    )
    gradient_penalty.backward()

    assert torch.isfinite(gradient_penalty)
    assert any(param.grad is not None for param in reward_model.parameters())


def test_airl_reward_components_match_raw_and_shaped_logits():
    torch.manual_seed(11)
    reward_model = AIRLReward(obs_dim=4, action_dim=2, hidden_sizes=(8,))
    obs = torch.randn(5, 4)
    actions = torch.randn(5, 2)
    next_obs = torch.randn(5, 4)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])

    raw, current_potential, next_potential, shaped = reward_model.components(
        obs,
        actions,
        next_obs,
        dones,
        gamma=0.95,
    )

    torch.testing.assert_close(raw, reward_model(obs, actions))
    torch.testing.assert_close(
        shaped,
        reward_model.shaped_logits(obs, actions, next_obs, dones, gamma=0.95),
    )
    torch.testing.assert_close(current_potential, reward_model.potential(obs).squeeze(-1))
    torch.testing.assert_close(next_potential, reward_model.potential(next_obs).squeeze(-1))


def test_recurrent_airl_log_probs_replay_expert_memory_context():
    torch.manual_seed(3)
    np.random.seed(3)
    cfg = PSGAILConfig(
        action_mode="continuous",
        policy_model="recurrent_transformer",
        continuous_action_dim=2,
        hidden_size=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
    )
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=6,
        hidden_size=16,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
    ).eval()
    observations = np.random.randn(5, 6).astype(np.float32)
    actions = []
    expected_log_probs = []
    memory = policy.initial_memory(1, dtype=torch.float32)
    with torch.no_grad():
        for step in range(len(observations)):
            dist, _values, step_memory = policy_distribution_values_memory(
                policy,
                torch.as_tensor(observations[step : step + 1], dtype=torch.float32),
                cfg,
                None,
                memory=memory,
                return_memory=True,
            )
            action = dist.sample()
            actions.append(action.squeeze(0).detach().cpu().numpy())
            expected_log_probs.append(float(dist.log_prob(action).detach().cpu().item()))
            memory = torch.cat([memory[:, 1:], step_memory.unsqueeze(1)], dim=1)

    reconstructed = _recurrent_policy_log_probs_for_indices(
        policy,
        cfg,
        observations,
        np.asarray(actions, dtype=np.float32),
        np.asarray(["0:7"] * len(observations), dtype=object),
        np.arange(len(observations), dtype=np.int64),
        np.arange(len(observations), dtype=np.int64),
        torch.device("cpu"),
        batch_size=2,
    )

    np.testing.assert_allclose(reconstructed, expected_log_probs, rtol=1e-5, atol=1e-5)


def test_recurrent_airl_log_probs_skip_centralized_critic(monkeypatch):
    torch.manual_seed(5)
    np.random.seed(5)
    cfg = PSGAILConfig(
        action_mode="continuous",
        policy_model="recurrent_transformer",
        continuous_action_dim=2,
        hidden_size=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
        centralized_critic=True,
        central_critic_pooling="attention",
        central_critic_max_vehicles=1,
        central_critic_attention_heads=4,
    )
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=6,
        hidden_size=16,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
        centralized_critic=True,
        critic_obs_dim=central_critic_observation_dim(6, cfg),
        central_critic_pooling="attention",
        central_critic_max_vehicles=1,
        central_critic_attention_heads=4,
    ).eval()

    def fail_if_called(_critic_obs):
        raise AssertionError("AIRL log-prob reconstruction should not run the centralized critic.")

    monkeypatch.setattr(policy.critic_encoder, "forward", fail_if_called)
    observations = np.random.randn(4, 6).astype(np.float32)
    actions = np.tanh(np.random.randn(4, 2)).astype(np.float32)

    reconstructed = _recurrent_policy_log_probs_for_indices(
        policy,
        cfg,
        observations,
        actions,
        np.asarray(["0:7"] * len(observations), dtype=object),
        np.arange(len(observations), dtype=np.int64),
        np.arange(len(observations), dtype=np.int64),
        torch.device("cpu"),
        batch_size=2,
    )

    assert reconstructed.shape == (len(observations),)
    assert np.isfinite(reconstructed).all()


def test_recurrent_airl_log_probs_scan_segments_once(monkeypatch):
    import scripts_gail.train_simple_airl as airl

    torch.manual_seed(7)
    np.random.seed(7)
    cfg = PSGAILConfig(
        action_mode="continuous",
        policy_model="recurrent_transformer",
        continuous_action_dim=2,
        hidden_size=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
    )
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=6,
        hidden_size=16,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
    ).eval()
    observations = np.random.randn(6, 6).astype(np.float32)
    actions = np.tanh(np.random.randn(6, 2)).astype(np.float32)
    seen_rows = 0
    original = airl.policy_distribution_memory

    def counting_policy_distribution_memory(policy_arg, obs_tensor, *args, **kwargs):
        nonlocal seen_rows
        seen_rows += int(obs_tensor.shape[0])
        return original(policy_arg, obs_tensor, *args, **kwargs)

    monkeypatch.setattr(airl, "policy_distribution_memory", counting_policy_distribution_memory)
    reconstructed = _recurrent_policy_log_probs_for_indices(
        policy,
        cfg,
        observations,
        actions,
        np.asarray(["traj"] * len(observations), dtype=object),
        np.arange(len(observations), dtype=np.int64),
        np.asarray([5, 2, 5], dtype=np.int64),
        torch.device("cpu"),
        batch_size=16,
    )

    assert reconstructed.shape == (3,)
    assert reconstructed[0] == pytest.approx(reconstructed[2])
    assert seen_rows == len(observations)


def test_recurrent_airl_log_probs_cached_index_matches_on_demand_with_gaps():
    torch.manual_seed(13)
    np.random.seed(13)
    cfg = PSGAILConfig(
        action_mode="continuous",
        policy_model="recurrent_transformer",
        continuous_action_dim=2,
        hidden_size=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
    )
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=6,
        hidden_size=16,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
    ).eval()
    observations = np.random.randn(8, 6).astype(np.float32)
    actions = np.tanh(np.random.randn(8, 2)).astype(np.float32)
    trajectory_ids = np.asarray(["a", "b", "a", "b", "a", "b", "a", "b"], dtype=object)
    timesteps = np.asarray([0, 0, 1, 1, 3, 2, 4, 4], dtype=np.int64)
    targets = np.asarray([7, 2, 4, 7, 5], dtype=np.int64)
    trajectory_index = _build_recurrent_trajectory_index(trajectory_ids, timesteps)

    on_demand = _recurrent_policy_log_probs_for_indices(
        policy,
        cfg,
        observations,
        actions,
        trajectory_ids,
        timesteps,
        targets,
        torch.device("cpu"),
        batch_size=2,
    )
    cached, stats = _recurrent_policy_log_probs_for_indices(
        policy,
        cfg,
        observations,
        actions,
        trajectory_ids,
        timesteps,
        targets,
        torch.device("cpu"),
        batch_size=2,
        trajectory_index=trajectory_index,
        return_stats=True,
    )

    np.testing.assert_allclose(cached, on_demand, rtol=1e-6, atol=1e-6)
    assert cached[0] == pytest.approx(cached[3])
    assert stats.selected_targets == len(targets)
    assert stats.unique_targets == len(np.unique(targets))
    assert stats.segment_count >= 3
    assert stats.scanned_transitions <= len(observations)


def test_airl_replay_keeps_recurrent_trajectory_context():
    cfg = PSGAILConfig(
        discriminator_replay_rounds=1,
        discriminator_replay_max_samples=0,
    )
    replay_rollout = _minimal_rollout(
        observations=np.arange(12, dtype=np.float32).reshape(6, 2),
        next_observations=np.arange(100, 112, dtype=np.float32).reshape(6, 2),
        actions=np.zeros((6, 2), dtype=np.float32),
        old_log_probs=np.zeros(6, dtype=np.float32),
        old_values=np.zeros(6, dtype=np.float32),
        returns=np.zeros(6, dtype=np.float32),
        advantages=np.zeros(6, dtype=np.float32),
        trajectory_ids=np.asarray([7, 7, 7, 8, 8, 8], dtype=np.int64),
        dones=np.asarray([False, False, True, False, False, True]),
    )
    current_rollout = _minimal_rollout(
        observations=np.arange(20, 26, dtype=np.float32).reshape(3, 2),
        next_observations=np.arange(120, 126, dtype=np.float32).reshape(3, 2),
        actions=np.ones((3, 2), dtype=np.float32),
        old_log_probs=np.full(3, -1.0, dtype=np.float32),
        old_values=np.zeros(3, dtype=np.float32),
        returns=np.zeros(3, dtype=np.float32),
        advantages=np.zeros(3, dtype=np.float32),
        trajectory_ids=np.asarray([9, 9, 9], dtype=np.int64),
        dones=np.asarray([False, False, True]),
    )

    replay = []
    append_airl_replay(replay, replay_rollout, cfg, round_idx=12)
    obs, actions, next_obs, dones, log_probs, trajectory_ids, timesteps = concat_airl_replay(
        current_rollout,
        replay,
        cfg,
        seed=123,
        preserve_recurrent_context=True,
    )

    assert log_probs is not None
    assert obs.shape == (9, 2)
    assert actions.shape == (9, 2)
    assert next_obs.shape == (9, 2)
    assert dones.shape == (9,)
    assert np.isnan(log_probs[:6]).all()
    np.testing.assert_allclose(log_probs[6:], np.full(3, -1.0, dtype=np.float32))
    assert trajectory_ids is not None
    assert timesteps is not None
    assert any(str(item).startswith("round:12:") for item in trajectory_ids)
    assert any(str(item).startswith("current:") for item in trajectory_ids)
    for trajectory_id in dict.fromkeys(trajectory_ids.tolist()):
        steps = timesteps[trajectory_ids == trajectory_id]
        np.testing.assert_array_equal(steps, np.arange(len(steps), dtype=np.int64))


def test_airl_replay_subsamples_complete_recurrent_trajectories():
    cfg = PSGAILConfig(
        discriminator_replay_rounds=1,
        discriminator_replay_max_samples=4,
    )
    replay_rollout = _minimal_rollout(
        observations=np.arange(12, dtype=np.float32).reshape(6, 2),
        actions=np.zeros((6, 2), dtype=np.float32),
        old_log_probs=np.zeros(6, dtype=np.float32),
        old_values=np.zeros(6, dtype=np.float32),
        returns=np.zeros(6, dtype=np.float32),
        advantages=np.zeros(6, dtype=np.float32),
        trajectory_ids=np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64),
    )
    current_rollout = _minimal_rollout(
        observations=np.arange(20, 26, dtype=np.float32).reshape(3, 2),
        actions=np.ones((3, 2), dtype=np.float32),
        old_log_probs=np.full(3, -1.0, dtype=np.float32),
        old_values=np.zeros(3, dtype=np.float32),
        returns=np.zeros(3, dtype=np.float32),
        advantages=np.zeros(3, dtype=np.float32),
        trajectory_ids=np.asarray([3, 3, 3], dtype=np.int64),
    )

    replay = []
    append_airl_replay(replay, replay_rollout, cfg, round_idx=1)
    obs, _actions, _next_obs, _dones, _log_probs, trajectory_ids, timesteps = concat_airl_replay(
        current_rollout,
        replay,
        cfg,
        seed=99,
        preserve_recurrent_context=True,
    )

    assert len(obs) <= 4
    assert trajectory_ids is not None
    assert timesteps is not None
    for trajectory_id in dict.fromkeys(trajectory_ids.tolist()):
        steps = timesteps[trajectory_ids == trajectory_id]
        np.testing.assert_array_equal(steps, np.arange(len(steps), dtype=np.int64))


def test_recurrent_airl_reward_update_requires_log_probs_or_generator_context():
    torch.manual_seed(4)
    np.random.seed(4)
    cfg = PSGAILConfig(
        action_mode="continuous",
        continuous_action_dim=2,
        policy_model="recurrent_transformer",
        hidden_size=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
        discriminator_loss="wgan_gp",
        disc_updates_per_round=1,
        max_grad_norm=0.5,
        gamma=0.99,
        wgan_gp_lambda=2.0,
    )
    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=6,
        hidden_size=16,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=3,
    )
    reward_model = AIRLReward(obs_dim=6, action_dim=2, hidden_sizes=(8,))
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1.0e-3)
    expert_obs = np.random.randn(5, 6).astype(np.float32)
    expert_actions = np.tanh(np.random.randn(5, 2)).astype(np.float32)
    expert_next_obs = np.random.randn(5, 6).astype(np.float32)
    expert_dones = np.zeros(5, dtype=bool)
    gen_obs = np.random.randn(5, 6).astype(np.float32)
    gen_actions = np.tanh(np.random.randn(5, 2)).astype(np.float32)
    gen_next_obs = np.random.randn(5, 6).astype(np.float32)
    gen_dones = np.zeros(5, dtype=bool)
    expert_trajectory_ids = np.asarray(["0:7"] * 5, dtype=object)
    expert_timesteps = np.arange(5, dtype=np.int64)
    generator_trajectory_ids = np.asarray(["round:1:0"] * 5, dtype=object)
    generator_timesteps = np.arange(5, dtype=np.int64)

    with pytest.raises(ValueError, match="generator_log_probs or generator_trajectory_ids"):
        update_reward_model(
            reward_model,
            optimizer,
            policy,
            expert_obs,
            expert_actions,
            expert_next_obs,
            expert_dones,
            gen_obs,
            gen_actions,
            gen_next_obs,
            gen_dones,
            cfg,
            torch.device("cpu"),
            reward_batch_size=2,
            expert_trajectory_ids=expert_trajectory_ids,
            expert_timesteps=expert_timesteps,
        )

    stats = update_reward_model(
        reward_model,
        optimizer,
        policy,
        expert_obs,
        expert_actions,
        expert_next_obs,
        expert_dones,
        gen_obs,
        gen_actions,
        gen_next_obs,
        gen_dones,
        cfg,
        torch.device("cpu"),
        reward_batch_size=2,
        expert_trajectory_ids=expert_trajectory_ids,
        expert_timesteps=expert_timesteps,
        generator_trajectory_ids=generator_trajectory_ids,
        generator_timesteps=generator_timesteps,
    )

    assert np.isfinite(stats["reward_loss"])
    assert all(param.requires_grad for param in policy.parameters())
    assert all(param.grad is None for param in policy.parameters())


def test_airl_wgan_reward_update_with_transformer_policy_keeps_policy_out_of_backward():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = PSGAILConfig(
        action_mode="continuous",
        continuous_action_dim=2,
        discriminator_loss="wgan_gp",
        disc_updates_per_round=1,
        max_grad_norm=0.5,
        gamma=0.99,
        wgan_gp_lambda=2.0,
    )
    policy = make_actor_critic(
        "transformer",
        obs_dim=6,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
    )
    reward_model = AIRLReward(obs_dim=6, action_dim=2, hidden_sizes=(8,))
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1.0e-3)
    expert_obs = np.random.randn(8, 6).astype(np.float32)
    expert_actions = np.tanh(np.random.randn(8, 2)).astype(np.float32)
    expert_next_obs = np.random.randn(8, 6).astype(np.float32)
    expert_dones = np.zeros(8, dtype=bool)
    gen_obs = np.random.randn(8, 6).astype(np.float32)
    gen_actions = np.tanh(np.random.randn(8, 2)).astype(np.float32)
    gen_next_obs = np.random.randn(8, 6).astype(np.float32)
    gen_dones = np.zeros(8, dtype=bool)

    stats = update_reward_model(
        reward_model,
        optimizer,
        policy,
        expert_obs,
        expert_actions,
        expert_next_obs,
        expert_dones,
        gen_obs,
        gen_actions,
        gen_next_obs,
        gen_dones,
        cfg,
        torch.device("cpu"),
        reward_batch_size=4,
    )

    assert np.isfinite(stats["reward_loss"])
    assert np.isfinite(stats["gradient_penalty"])
    assert all(param.requires_grad for param in policy.parameters())
    assert all(param.grad is None for param in policy.parameters())


def test_airl_player_challenge_bonus_uses_same_primary_reward_cap():
    class FixedReward(torch.nn.Module):
        def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)

        def shaped_logits(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            next_obs: torch.Tensor,
            dones: torch.Tensor,
            *,
            gamma: float,
        ) -> torch.Tensor:
            del actions, next_obs, dones, gamma
            return torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32, device=obs.device)

    cfg = PSGAILConfig(
        discriminator_loss="airl_bce",
        enable_player_challenge_reward=True,
        challenge_reward_coef=100.0,
        challenge_reward_clip=100.0,
        challenge_max_primary_reward_fraction=0.10,
        challenge_expert_like_quantile=0.0,
        normalize_gail_reward=False,
        gail_reward_clip=0.0,
        final_reward_clip=0.0,
    )
    rollout = _minimal_rollout(
        observations=np.zeros((3, 2), dtype=np.float32),
        actions=np.zeros((3, 2), dtype=np.float32),
        old_log_probs=np.zeros(3, dtype=np.float32),
        old_values=np.zeros(3, dtype=np.float32),
        returns=np.zeros(3, dtype=np.float32),
        advantages=np.zeros(3, dtype=np.float32),
    )
    rollout.challenge_payoffs = np.full(3, 10.0, dtype=np.float32)

    refreshed = refresh_airl_rewards(rollout, FixedReward(), cfg, torch.device("cpu"))

    primary = np.asarray([1.0, 2.0, 0.5], dtype=np.float32)
    caps = 0.10 * np.maximum(np.abs(primary), float(np.mean(np.abs(primary))))
    np.testing.assert_allclose(refreshed.challenge_bonuses, caps, rtol=1e-6)
    np.testing.assert_allclose(refreshed.rewards, primary + caps, rtol=1e-6)


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
    assert stats["hard_selector_enabled"] == 0.0
    assert stats["hard_selector_selected_samples"] == float(len(generator))


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


def test_training_count_validation_selects_scheduled_vehicle_count_and_clips(monkeypatch):
    valid_ids_by_episode = {
        "ep_a": np.asarray([1, 2, 3], dtype=np.int64),
        "ep_b": np.asarray([10, 11], dtype=np.int64),
    }

    def fake_load_prebuilt_data(*_args, **_kwargs):
        return "", valid_ids_by_episode, {}, ["ep_a", "ep_b"]

    monkeypatch.setattr(eval_mod, "load_prebuilt_data", fake_load_prebuilt_data)
    cfg = PSGAILConfig(percentage_controlled_vehicles=2, seed=123)
    specs = eval_mod._evaluation_training_count_episode_specs(cfg, split="val", episodes=2)
    assert len(specs) == 2
    assert [len(vehicle_ids) for _idx, _episode, vehicle_ids in specs] == [2, 2]
    for _idx, episode, vehicle_ids in specs:
        assert set(vehicle_ids).issubset(set(map(int, valid_ids_by_episode[episode])))

    clipped_cfg = PSGAILConfig(percentage_controlled_vehicles=5, seed=123)
    clipped_specs = eval_mod._evaluation_training_count_episode_specs(
        clipped_cfg,
        split="val",
        episodes=2,
    )
    assert sorted(len(vehicle_ids) for _idx, _episode, vehicle_ids in clipped_specs) == [2, 3]


def test_gail_and_airl_parse_same_validation_strategy_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_simple_ps_gail.py"])
    gail_cfg = ps_gail_parse_args()
    monkeypatch.setattr(sys, "argv", ["train_simple_airl.py"])
    airl_cfg, _reward_batch_size, airl_log_prob_batch_size = airl_parse_args()

    assert gail_cfg.validation_vehicle_mode == "training_count"
    assert airl_cfg.validation_vehicle_mode == "training_count"
    assert gail_cfg.validation_stress_vehicle_mode == "all"
    assert airl_cfg.validation_stress_vehicle_mode == "all"
    assert not gail_cfg.validation_control_all_vehicles
    assert not airl_cfg.validation_control_all_vehicles
    assert airl_log_prob_batch_size == 512
    assert airl_cfg.airl_policy_reward_mode == "shaped"


def test_airl_parse_exposes_separate_log_prob_batch_size(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_simple_airl.py",
            "--reward-batch-size",
            "4096",
            "--airl-log-prob-batch-size",
            "128",
        ],
    )
    _airl_cfg, reward_batch_size, airl_log_prob_batch_size = airl_parse_args()

    assert reward_batch_size == 4096
    assert airl_log_prob_batch_size == 128


def test_paper_slurm_scripts_keep_scene_and_expert_budget_explicit():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "slurum/script_pretrain/train_gail_continuous_gpu_32c_stage1_50veh.bash",
        root / "slurum/script_finetune/train_gail_continuous_gpu_32c_stage2_100veh.bash",
        root / "slurum/script_pretrain/train_airl_continuous_gpu_32c_stage1_50veh.bash",
        root / "slurum/script_finetune/train_airl_continuous_gpu_32c_stage2_100veh.bash",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert 'SCENE="${SCENE:-us-101}"' in text
        assert 'EPISODE_ROOT="${EPISODE_ROOT:-${REPODIR}/highway_env/data/processed_20s}"' in text
        assert 'PREBUILT_SPLIT="${PREBUILT_SPLIT:-train}"' in text
        assert 'MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-0}"' in text
        assert '--scene "${SCENE}"' in text
        assert '--episode-root "${EPISODE_ROOT}"' in text
        assert '--prebuilt-split "${PREBUILT_SPLIT}"' in text
        assert "--scene us-101" not in text
        assert '--max-expert-samples "${MAX_EXPERT_SAMPLES}"' in text


def test_paper_airl_slurm_scripts_pin_shaped_reward_mode():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "slurum/script_pretrain/train_airl_continuous_gpu_32c_stage1_50veh.bash",
        root / "slurum/script_finetune/train_airl_continuous_gpu_32c_stage2_100veh.bash",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert 'AIRL_POLICY_REWARD_MODE="${AIRL_POLICY_REWARD_MODE:-shaped}"' in text
        assert '--airl-policy-reward-mode "${AIRL_POLICY_REWARD_MODE}"' in text
    assert "ALLOW_AIRL_RESUME_WITHOUT_REWARD" in scripts[1].read_text(encoding="utf-8")


def test_paper_airl_slurm_scripts_default_to_gradual_vehicle_schedules():
    root = Path(__file__).resolve().parents[1]
    airl_stage1 = (
        root / "slurum/script_pretrain/train_airl_continuous_gpu_32c_stage1_50veh.bash"
    ).read_text(encoding="utf-8")
    airl_stage2 = (
        root / "slurum/script_finetune/train_airl_continuous_gpu_32c_stage2_100veh.bash"
    ).read_text(encoding="utf-8")

    assert 'CONTROLLED_VEHICLE_SCHEDULE_PROFILE="${CONTROLLED_VEHICLE_SCHEDULE_PROFILE:-gradual}"' in airl_stage1
    assert 'CONTROLLED_VEHICLE_SCHEDULE_PROFILE="${CONTROLLED_VEHICLE_SCHEDULE_PROFILE:-gradual}"' in airl_stage2
    assert 'INITIAL_CONTROLLED_VEHICLES="${INITIAL_CONTROLLED_VEHICLES:-50}"' in airl_stage2
    assert (
        'CONTROLLED_VEHICLE_SCHEDULE_GRADUAL="${CONTROLLED_VEHICLE_SCHEDULE_GRADUAL:-'
        '0:50:50:60;50:100:60:75;100:150:75:90;150:200:90:100}"'
    ) in airl_stage2
    assert 'CONTROLLED_VEHICLE_SCHEDULE_SUDDEN="${CONTROLLED_VEHICLE_SCHEDULE_SUDDEN:-0:200:100:100}"' in airl_stage2


def test_paper_finetune_slurm_scripts_auto_resolve_ckpt_folder():
    root = Path(__file__).resolve().parents[1]
    gail = (
        root / "slurum/script_finetune/train_gail_continuous_gpu_32c_stage2_100veh.bash"
    ).read_text(encoding="utf-8")
    airl = (
        root / "slurum/script_finetune/train_airl_continuous_gpu_32c_stage2_100veh.bash"
    ).read_text(encoding="utf-8")

    assert 'CKPT_ROOT="${CKPT_ROOT:-${REPODIR}/ckpt}"' in gail
    assert 'CKPT_ROOT="${CKPT_ROOT:-${REPODIR}/ckpt}"' in airl
    assert 'CKPT_DATASET="${CKPT_DATASET:-}"' in gail
    assert 'CKPT_DATASET="${CKPT_DATASET:-}"' in airl
    assert '"${CKPT_ROOT}/${CKPT_DATASET}/gail/final_pretrain.pt"' in gail
    assert '"${CKPT_ROOT}/${CKPT_DATASET}/airl/final_pretrain.pt"' in airl
    assert '"${CKPT_ROOT}/${CKPT_DATASET}/gail/best.pt"' not in gail
    assert '"${CKPT_ROOT}/${CKPT_DATASET}/gail/final.pt"' not in gail
    assert '"${CKPT_ROOT}/${CKPT_DATASET}/airl/best.pt"' not in airl
    assert '"${CKPT_ROOT}/${CKPT_DATASET}/airl/final.pt"' not in airl
    assert '$(basename "${RESUME_CHECKPOINT}")" != "final_pretrain.pt"' in gail
    assert '$(basename "${RESUME_CHECKPOINT}")" != "final_pretrain.pt"' in airl
    assert '$(basename "${RESUME_CHECKPOINT}")" != "best.pt"' not in gail
    assert '$(basename "${RESUME_CHECKPOINT}")" != "best.pt"' not in airl
    assert '$(basename "${RESUME_CHECKPOINT}")" != "final.pt"' not in gail
    assert '$(basename "${RESUME_CHECKPOINT}")" != "final.pt"' not in airl


def test_paper_slurm_scripts_use_fixed_low_entropy():
    root = Path(__file__).resolve().parents[1]
    airl_stage1 = (
        root / "slurum/script_pretrain/train_airl_continuous_gpu_32c_stage1_50veh.bash"
    ).read_text(encoding="utf-8")
    airl_stage2 = (
        root / "slurum/script_finetune/train_airl_continuous_gpu_32c_stage2_100veh.bash"
    ).read_text(encoding="utf-8")
    gail_stage1 = (
        root / "slurum/script_pretrain/train_gail_continuous_gpu_32c_stage1_50veh.bash"
    ).read_text(encoding="utf-8")
    gail_stage2 = (
        root / "slurum/script_finetune/train_gail_continuous_gpu_32c_stage2_100veh.bash"
    ).read_text(encoding="utf-8")

    assert 'WARMUP_ENTROPY_COEF="${WARMUP_ENTROPY_COEF:-0.001}"' in airl_stage1
    assert 'ENTROPY_COEF="${ENTROPY_COEF:-0.001}"' in airl_stage1
    assert 'ENTROPY_COEF_SCHEDULE="${ENTROPY_COEF_SCHEDULE:-}"' in airl_stage1
    assert 'ENTROPY_COEF_SCHEDULE="${ENTROPY_COEF_SCHEDULE:-}"' in airl_stage2
    assert 'WARMUP_ENTROPY_COEF="${WARMUP_ENTROPY_COEF:-0.0015}"' in gail_stage1
    assert 'ENTROPY_COEF="${ENTROPY_COEF:-0.0015}"' in gail_stage1
    assert 'ENTROPY_COEF_SCHEDULE="${ENTROPY_COEF_SCHEDULE:-}"' in gail_stage1
    assert 'ENTROPY_COEF_SCHEDULE="${ENTROPY_COEF_SCHEDULE:-}"' in gail_stage2
    for text in (airl_stage1, airl_stage2, gail_stage1, gail_stage2):
        assert "0:100:0.04:0.015" not in text
        assert "0:600:0.0005:0.001" not in text
        assert "0:200:0.0005:0.001" not in text
        assert "0:600:0.0008:0.0015" not in text
        assert "0:200:0.0008:0.0015" not in text


def test_weighted_validation_score_prefers_lower_rmse_and_safety_rates():
    cfg = PSGAILConfig()
    good_metrics = {
        "validation/rmse_position_20s": 1.0,
        "validation/rmse_speed_20s": 0.5,
        "validation/rmse_lane_offset_20s": 0.1,
        "validation/vehicle_crash_rate": 0.0,
        "validation/vehicle_offroad_rate": 0.0,
        "validation/hard_brake_rate": 0.01,
    }
    bad_metrics = {
        "validation/rmse_position_20s": 1.5,
        "validation/rmse_speed_20s": 0.8,
        "validation/rmse_lane_offset_20s": 0.3,
        "validation/vehicle_crash_rate": 0.02,
        "validation/vehicle_offroad_rate": 0.01,
        "validation/hard_brake_rate": 0.05,
    }

    good_cost, good_score, good_components = validation_cost_and_score(good_metrics, cfg)
    bad_cost, bad_score, bad_components = validation_cost_and_score(bad_metrics, cfg)

    assert np.isfinite(good_cost)
    assert np.isfinite(bad_cost)
    assert good_score > bad_score
    assert good_cost < bad_cost
    assert good_components["speed_rmse"] == pytest.approx(0.5)
    assert bad_components["lane_offset_rmse"] == pytest.approx(0.3)


def test_matched_validation_reports_crash_agent_fraction_separately_from_incidence():
    squared = {
        20: {"x": [], "y": [], "position": [], "speed": [], "lane_offset": []}
    }
    final_squared = {"x": [], "y": [], "position": [], "speed": [], "lane_offset": []}

    metrics = eval_mod._matched_eval_metrics(
        prefix="validation",
        attempted_episodes=1,
        evaluated_episodes=1,
        skipped_missing_expert=0,
        skipped_bad_reference=0,
        skipped_empty_rollout=0,
        total_steps=10,
        collision_steps=3,
        offroad_steps=0,
        hard_brake_steps=0,
        episode_lengths=[10],
        squared=squared,
        final_squared=final_squared,
        horizons=[20],
        crashed_vehicle_episodes=1,
        vehicle_episodes=1,
    )

    assert metrics["validation/crash_agent_fraction"] == pytest.approx(0.3)
    assert metrics["validation/collision_duration_rate"] == pytest.approx(0.3)
    assert metrics["validation/vehicle_crash_rate"] == pytest.approx(1.0)
    assert metrics["validation/collision_rate"] == pytest.approx(1.0)


def test_validation_score_uses_crash_agent_fraction_before_vehicle_crash_rate():
    cfg = PSGAILConfig()
    cfg.validation_score_position_weight = 0.0
    cfg.validation_score_speed_weight = 0.0
    cfg.validation_score_lane_offset_weight = 0.0
    cfg.validation_score_crash_weight = 25.0
    cfg.validation_score_offroad_weight = 0.0
    cfg.validation_score_hard_brake_weight = 0.0
    metrics = {
        "validation/rmse_position_20s": 10.0,
        "validation/rmse_speed_20s": 10.0,
        "validation/rmse_lane_offset_20s": 10.0,
        "validation/crash_agent_fraction": 0.04,
        "validation/vehicle_crash_rate": 1.0,
        "validation/vehicle_offroad_rate": 0.0,
        "validation/hard_brake_rate": 0.0,
    }

    scored, cost, score = scored_validation_metrics(metrics, cfg)

    assert cost == pytest.approx(1.0)
    assert score == pytest.approx(-1.0)
    assert scored["validation/score_component_crash_agent_fraction"] == pytest.approx(0.04)
    assert scored["validation/score_component_vehicle_crash_rate"] == pytest.approx(1.0)


def test_best_checkpoint_payload_carries_validation_metadata_and_model_state_keys():
    base_payload = {
        "round": 20,
        "policy_state_dict": {"weight": torch.tensor([1.0])},
        "discriminator_state_dict": {"weight": torch.tensor([2.0])},
        "config": {"run_name": "unit"},
    }
    metrics = {
        "validation/rmse_position_20s": 1.0,
        "validation/rmse_speed_20s": 0.5,
        "validation/rmse_lane_offset_20s": 0.1,
    }
    payload = best_checkpoint_payload(
        base_payload,
        round_idx=20,
        validation_metrics=metrics,
        validation_score=-1.23,
        validation_cost=1.23,
    )

    assert payload["round"] == 20
    assert payload["best_round"] == 20
    assert payload["validation_score"] == pytest.approx(-1.23)
    assert payload["validation_cost"] == pytest.approx(1.23)
    assert payload["validation_metrics"] == metrics
    assert "policy_state_dict" in payload
    assert "discriminator_state_dict" in payload


def test_stage2_scripts_require_final_pretrain_resume_by_default():
    root = Path(__file__).resolve().parents[1]
    gail_script = root / "slurum" / "script_finetune" / "train_gail_continuous_gpu_32c_stage2_100veh.bash"
    airl_script = root / "slurum" / "script_finetune" / "train_airl_continuous_gpu_32c_stage2_100veh.bash"
    for script in (gail_script, airl_script):
        text = script.read_text(encoding="utf-8")
        assert 'ALLOW_NON_BEST_RESUME="${ALLOW_NON_BEST_RESUME:-false}"' in text
        assert '$(basename "${RESUME_CHECKPOINT}")" != "final_pretrain.pt"' in text
        assert '$(basename "${RESUME_CHECKPOINT}")" != "best.pt"' not in text
        assert '$(basename "${RESUME_CHECKPOINT}")" != "final.pt"' not in text
        assert "Set ALLOW_NON_BEST_RESUME=true to override" in text
