import gymnasium as gym
import numpy as np
import pytest

import highway_env
from highway_env.ngsim_utils.core.config import normalize_action_mode
from highway_env.ngsim_utils.core.constants import ACCELERATION_RANGE


gym.register_envs(highway_env)


@pytest.mark.parametrize(
    "action_config",
    [
        {"type": "ContinuousAction"},
        {"type": "DiscreteAction"},
        {"type": "DiscreteMetaAction"},
    ],
)
def test_action_type(action_config):
    env = gym.make("highway-v0", config={"action": action_config})
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(obs)
    env.close()


def test_zero_centered_continuous_acceleration():
    env = gym.make(
        "highway-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": [-10.0, 10.0],
                "zero_centered_acceleration": True,
            }
        },
    )
    env.reset()

    assert env.unwrapped.action_type.get_action(np.asarray([-1.0, 0.0], dtype=np.float32))[
        "acceleration"
    ] == pytest.approx(-10.0)
    assert env.unwrapped.action_type.get_action(np.asarray([0.0, 0.0], dtype=np.float32))[
        "acceleration"
    ] == pytest.approx(0.0)
    assert env.unwrapped.action_type.get_action(np.asarray([1.0, 0.0], dtype=np.float32))[
        "acceleration"
    ] == pytest.approx(10.0)

    env.close()


def test_continuous_action_accepts_numpy_range_configs():
    env = gym.make(
        "highway-v0",
        config={
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": np.asarray([-10.0, 10.0], dtype=np.float32),
                "steering_range": np.asarray([-0.5, 0.5], dtype=np.float32),
                "speed_range": np.asarray([0.0, 30.0], dtype=np.float32),
                "zero_centered_acceleration": True,
            }
        },
    )
    env.reset()

    mapped = env.unwrapped.action_type.get_action(np.asarray([-1.0, 1.0], dtype=np.float32))

    assert mapped["acceleration"] == pytest.approx(-10.0)
    assert mapped["steering"] == pytest.approx(0.5)
    assert env.unwrapped.action_type.controlled_vehicle.MIN_SPEED == pytest.approx(0.0)
    assert env.unwrapped.action_type.controlled_vehicle.MAX_SPEED == pytest.approx(30.0)

    env.close()


def test_ngsim_continuous_action_configs_use_shared_acceleration_range():
    direct_cfg = {"action": {"type": "ContinuousAction"}}
    assert normalize_action_mode(direct_cfg, {"action_mode": "continuous"}) == "continuous"
    assert direct_cfg["action"]["acceleration_range"] == list(ACCELERATION_RANGE)
    assert direct_cfg["action"]["zero_centered_acceleration"] is True

    multi_cfg = {
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "ContinuousAction"},
        }
    }
    assert normalize_action_mode(multi_cfg, multi_cfg) == "continuous"
    assert multi_cfg["action"]["action_config"]["acceleration_range"] == list(ACCELERATION_RANGE)
    assert multi_cfg["action"]["action_config"]["zero_centered_acceleration"] is True
