from __future__ import annotations

import numpy as np
import pytest
from highway_env.imitation.observation_aligned_reflex import (
    observation_aligned_reflex_actions,
    observation_aligned_reflex_contract,
)
from policy.evaluation.observation_aligned_reflex import episode_seed


def _observation(*, gap_m: float = 64.0, relative_speed: float = 0.0) -> np.ndarray:
    observation = np.zeros(322, dtype=np.float32)
    lidar = observation[:256].reshape(128, 2)
    lidar[:, 0] = 1.0
    lidar[0] = [gap_m / 64.0, relative_speed / 64.0]
    camera = observation[256:319].reshape(21, 3)
    forward = np.linspace(0.05, 0.6, 10, dtype=np.float32)
    camera[:10, 0] = 1.0
    camera[:10, 1] = forward
    camera[:10, 2] = 0.03
    camera[10:20, 0] = 1.0
    camera[10:20, 1] = forward
    camera[10:20, 2] = -0.03
    observation[319:] = [4.5, 15.0, 0.0]
    return observation


def test_contract_is_stateless_current_observation_only():
    contract = observation_aligned_reflex_contract()
    assert contract["policy_observation_dim"] == 322
    assert contract["stateful"] is False
    assert "realized_future_path" in contract["forbidden_reads"]
    assert "expert_action_history" in contract["forbidden_reads"]
    assert len(contract["contract_sha256"]) == 64


def test_actions_are_deterministic_bounded_and_batch_consistent():
    observation = _observation()
    first = observation_aligned_reflex_actions(observation)
    second = observation_aligned_reflex_actions(observation.copy())
    batch = observation_aligned_reflex_actions(np.stack([observation, observation]))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(batch, np.stack([first, first]))
    assert first.shape == (2,)
    assert np.all(np.isfinite(first))
    assert np.all(np.abs(first) <= 1.0)


def test_closer_or_closing_front_vehicle_never_increases_acceleration():
    clear = observation_aligned_reflex_actions(_observation(gap_m=64.0))[0]
    close = observation_aligned_reflex_actions(_observation(gap_m=10.0))[0]
    closing = observation_aligned_reflex_actions(
        _observation(gap_m=10.0, relative_speed=-5.0)
    )[0]
    assert close <= clear
    assert closing <= close


def test_nearby_forward_beam_cannot_hide_a_lead_vehicle():
    clear = _observation(gap_m=64.0)
    adjacent = clear.copy()
    adjacent_lidar = adjacent[:256].reshape(128, 2)
    adjacent_lidar[3] = [10.0 / 64.0, -5.0 / 64.0]

    assert observation_aligned_reflex_actions(adjacent)[0] < (
        observation_aligned_reflex_actions(clear)[0]
    )


def test_lane_centre_shift_has_expected_steering_sign():
    left = _observation()
    right = _observation()
    left[258:319:3] += 0.02
    right[258:319:3] -= 0.02
    assert observation_aligned_reflex_actions(left)[1] > 0.0
    assert observation_aligned_reflex_actions(right)[1] < 0.0


def test_invalid_shape_and_nonfinite_inputs_fail_closed():
    with pytest.raises(ValueError, match=r"\[N,322\]"):
        observation_aligned_reflex_actions(np.zeros(321, dtype=np.float32))
    invalid = _observation()
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        observation_aligned_reflex_actions(invalid)


def test_reflex_evaluation_uses_the_frozen_paired_protocol_seed():
    assert episode_seed(
        evaluation_scenario_seed=20260716,
        seed_offset=10_000,
        episode_idx=2,
    ) == 20270718
