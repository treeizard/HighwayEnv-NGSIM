import numpy as np
import pytest

from highway_env.road.route_intent import route_intent_contract
from policy.contracts.imitation import policy_observation_contract
from policy.contracts.observations import policy_observations_from_flat


def test_route_intent_extends_policy_only_and_preserves_raw_observation():
    base = policy_observation_contract(lidar_cells=128, maximum_range=100.0)
    augmented = policy_observation_contract(
        lidar_cells=128,
        maximum_range=100.0,
        route_intent=route_intent_contract(),
    )
    assert augmented["raw_observation_dim"] == base["raw_observation_dim"]
    assert augmented["policy_observation_dim"] == base["policy_observation_dim"] + 16
    assert augmented["schema_version"] == 3
    assert augmented["policy_observation_order"][-1] == (
        "route_intent_policy_projection"
    )

    raw = np.zeros((2, base["raw_observation_dim"]), dtype=np.float32)
    route = np.zeros((2, 16), dtype=np.float32)
    route[:, 0] = 1.0
    projected = policy_observations_from_flat(
        raw,
        route_intent_features=route,
    )
    assert projected.shape == (2, augmented["policy_observation_dim"])
    assert np.array_equal(projected[:, -16:], route)


def test_policy_contract_rejects_action_and_topology_shortcuts():
    action_contract = route_intent_contract()
    action_contract["policy_projection"]["feature_names"][0] = "previous_action"
    with pytest.raises(ValueError, match="forbidden shortcut"):
        policy_observation_contract(
            lidar_cells=128,
            maximum_range=100.0,
            route_intent=action_contract,
        )

    visible_topology = route_intent_contract()
    visible_topology["topology_sidecar"]["policy_visible"] = True
    with pytest.raises(ValueError, match="cannot be policy-visible"):
        policy_observation_contract(
            lidar_cells=128,
            maximum_range=100.0,
            route_intent=visible_topology,
        )


def test_route_features_must_be_finite_normalized_and_row_aligned():
    raw = np.zeros((2, 323), dtype=np.float32)
    with pytest.raises(ValueError, match="same row count"):
        policy_observations_from_flat(
            raw,
            route_intent_features=np.zeros((1, 16), dtype=np.float32),
        )
    invalid = np.zeros((2, 16), dtype=np.float32)
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        policy_observations_from_flat(raw, route_intent_features=invalid)
