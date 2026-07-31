from pathlib import Path

import numpy as np
import torch
from highway_env import research_api
from scripts_gail import policy_api


def test_policy_api_preserves_sensor_and_action_contracts():
    raw = np.arange(2 * 323, dtype=np.float32).reshape(2, 323)
    projected = policy_api.policy_observations_from_flat(raw)

    assert projected.shape == (2, 322)
    np.testing.assert_array_equal(projected[:, -3], raw[:, -1])
    np.testing.assert_array_equal(projected[:, -2], raw[:, -4])
    np.testing.assert_array_equal(projected[:, -1], raw[:, -3])

    contract = policy_api.runtime_continuous_action_contract()
    assert contract["normalized_columns"] == [
        "acceleration_norm",
        "steering_norm",
    ]
    assert contract["scales"] == [5.0, float(np.pi / 4.0)]


def test_policy_api_checkpoint_round_trip(tmp_path: Path):
    policy = policy_api.make_actor_critic(
        "recurrent_transformer",
        obs_dim=322,
        hidden_size=16,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=3,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=2,
        transformer_memory_context_length=4,
    )
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": {
                "policy_model": "recurrent_transformer",
                "action_mode": "continuous",
                "continuous_action_dim": 2,
                "hidden_size": 16,
                "transformer_layers": 3,
                "transformer_heads": 4,
                "transformer_dropout": 0.0,
                "transformer_memory_tokens": 2,
                "transformer_memory_context_length": 4,
            },
        },
        checkpoint,
    )

    loaded = policy_api.load_policy_bundle(checkpoint)

    assert loaded.policy_obs_dim == 322
    assert loaded.critic_obs_dim == 322
    assert loaded.config["transformer_layers"] == 3
    assert loaded.checkpoint_sha256 == policy_api.sha256_file(checkpoint)
    for expected, actual in zip(
        policy.state_dict().values(),
        loaded.policy.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_research_api_reexports_repo_native_environment_contract():
    assert research_api.ENV_ID == "NGSim-US101-v0"
    assert research_api.ACCELERATION_RANGE == (-5.0, 5.0)
    assert research_api.MAX_STEER == np.pi / 4.0
    assert research_api.PUBLIC_RESEARCH_API_VERSION == 2
    assert policy_api.PUBLIC_POLICY_API_VERSION == 2
