from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.contracts import (
    assert_compatible_observation_contracts,
    policy_observation_contract,
    runtime_continuous_action_contract,
    validate_expert_action_contract,
)
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.trainer import policy_distribution_values_memory


def test_runtime_action_and_policy_observation_contracts_are_explicit():
    action = runtime_continuous_action_contract()
    assert action["normalized_columns"] == [
        "acceleration_norm",
        "steering_norm",
    ]
    assert action["physical_columns"] == [
        "steering_rad",
        "acceleration_mps2",
    ]
    assert action["physical_index_for_normalized"] == [1, 0]
    assert action["scales"] == pytest.approx([5.0, np.pi / 4.0])

    observation = policy_observation_contract(
        lidar_cells=128,
        maximum_range=64.0,
    )
    assert observation["raw_observation_dim"] == 323
    assert observation["policy_observation_dim"] == 322
    assert observation["policy_observation_order"] == [
        "lidar_flat",
        "lane_camera_flat",
        "length_m",
        "speed_mps",
        "heading_rad",
    ]
    assert observation["omitted_raw_fields"] == ["width_m"]


def test_action_contract_rejects_legacy_acceleration_scale():
    normalized = np.asarray(
        [[-0.5, -0.25], [0.0, 0.0], [0.75, 0.5]],
        dtype=np.float32,
    )
    physical_accel5 = np.column_stack(
        (
            normalized[:, 1] * (np.pi / 4.0),
            normalized[:, 0] * 5.0,
        )
    )
    inferred = validate_expert_action_contract(
        normalized,
        physical_accel5,
        require_runtime_match=True,
    )
    assert inferred["scales"] == pytest.approx([5.0, np.pi / 4.0])

    physical_accel10 = physical_accel5.copy()
    physical_accel10[:, 1] = normalized[:, 0] * 10.0
    with pytest.raises(ValueError, match="different physical units"):
        validate_expert_action_contract(
            normalized,
            physical_accel10,
            require_runtime_match=True,
        )


def test_observation_contract_rejects_sensor_projection_drift():
    reference = policy_observation_contract(
        lidar_cells=128,
        maximum_range=64.0,
    )
    drifted = policy_observation_contract(
        lidar_cells=64,
        maximum_range=64.0,
    )
    with pytest.raises(ValueError, match="observation contract mismatch"):
        assert_compatible_observation_contracts(reference, drifted)


def test_continuous_gail_distribution_squashes_actor_logits_once():
    policy = make_actor_critic(
        "transformer",
        obs_dim=4,
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=2,
        transformer_dropout=0.0,
    )
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([1.0, -0.75]))
    policy.eval()
    observations = torch.zeros((3, 4), dtype=torch.float32)
    deterministic_actions, _values = policy(observations)

    cfg = PSGAILConfig(
        action_mode="continuous",
        continuous_action_dim=2,
    )
    distribution, _values, _memory = policy_distribution_values_memory(
        policy,
        observations,
        cfg,
    )

    torch.testing.assert_close(
        distribution.normal.loc,
        torch.tensor([[1.0, -0.75]]).expand_as(deterministic_actions),
    )
    torch.testing.assert_close(
        torch.tanh(distribution.normal.loc),
        deterministic_actions,
    )
    assert not torch.allclose(
        torch.tanh(deterministic_actions),
        deterministic_actions,
    )
