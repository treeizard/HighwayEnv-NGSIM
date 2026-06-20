import gymnasium as gym
import numpy as np
import pytest

import highway_env


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
