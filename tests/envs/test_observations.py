import gymnasium as gym
import numpy as np
import pytest

import highway_env
from highway_env.envs.common.observations.lidar import LidarObservation
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


gym.register_envs(highway_env)


@pytest.mark.parametrize(
    "observation_config",
    [
        {"type": "LidarObservation"},
    ],
)
def test_observation_type(observation_config):
    env = gym.make("parking-v0", config={"observation": observation_config})
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(obs)
    env.close()


def test_multi_origin_lidar_road_edges_match_single_origin_reference():
    env = gym.make("parking-v0", config={"observation": {"type": "LidarObservation", "cells": 32}})
    env.reset(seed=7)
    lidar = env.unwrapped.observation_type
    origin = np.asarray(env.unwrapped.vehicle.position, dtype=float)
    origins = np.stack([origin, origin + np.asarray([0.25, 0.0])], axis=0)

    actual = lidar._distance_to_road_edges_many(
        origins,
        lidar._directions,
        lidar.maximum_range,
        lidar.coarse_step,
        lidar.refine_iters,
    )
    expected = np.stack(
        [
            lidar._distance_to_road_edges_batch(
                value,
                lidar._directions,
                lidar.maximum_range,
                lidar.coarse_step,
                lidar.refine_iters,
            )
            for value in origins
        ],
        axis=0,
    )

    np.testing.assert_array_equal(actual, expected)
    env.close()


def test_separated_lidar_preserves_positive_dynamic_distance_initialization():
    class _Env:
        pass

    env = _Env()
    env.road = Road(
        RoadNetwork.straight_road_network(lanes=1, length=200.0)
    )
    ego = Vehicle(env.road, [40.0, 0.0], heading=0.0, speed=10.0)
    lead = Vehicle(env.road, [50.0, 0.0], heading=0.0, speed=8.0)
    env.road.vehicles.extend([ego, lead])
    env.vehicle = ego
    lidar = LidarObservation(
        env,
        cells=64,
        maximum_range=64.0,
        normalize=True,
        ego_centric=True,
        separate_road_edge_return=True,
    )

    observation = lidar.observe()

    dynamic = observation[:, LidarObservation.DYNAMIC_PRESENCE] >= 0.5
    assert np.any(dynamic)
    assert set(np.flatnonzero(dynamic)).intersection({0, 63})
    assert np.all(
        observation[dynamic, LidarObservation.DYNAMIC_DISTANCE] > 0.0
    )
    assert np.all(
        observation[dynamic, LidarObservation.DYNAMIC_DISTANCE] < 1.0
    )
    assert np.all(
        observation[~dynamic, LidarObservation.DYNAMIC_DISTANCE] == 1.0
    )


if __name__ == "__main__":
    pytest.main([__file__])
