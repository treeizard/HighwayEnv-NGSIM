import gymnasium as gym
import numpy as np
import pytest

import highway_env
from highway_env.envs.common.observations.camera import _lane_state_from_camera
from highway_env.envs.common.observations.lidar import LidarObservation
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


gym.register_envs(highway_env)


def _lane_state_scalar_reference(
    camera_observation: np.ndarray, *, maximum_range: float
) -> tuple[float, float, float]:
    """Pre-fast-path lane fit retained as the exact test oracle."""
    values = np.asarray(camera_observation, dtype=float)
    scale = float(maximum_range)
    points = values[:, 1:3] * scale
    valid = (
        (values[:, 0] >= 0.5)
        & np.all(np.isfinite(points), axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= min(scale, 12.0))
    )

    def fit_side(intercept_sign: float) -> tuple[float, float] | None:
        side_points = points[valid]
        if len(side_points) < 3:
            return None
        best: tuple[tuple[int, float, float], np.ndarray] | None = None
        for first in range(len(side_points) - 1):
            for second in range(first + 1, len(side_points)):
                dx = float(side_points[second, 0] - side_points[first, 0])
                if abs(dx) < 0.75:
                    continue
                slope = float(
                    (side_points[second, 1] - side_points[first, 1]) / dx
                )
                if abs(slope) > 1.0:
                    continue
                intercept = float(
                    side_points[first, 1] - slope * side_points[first, 0]
                )
                if intercept * intercept_sign <= 0.0:
                    continue
                residuals = np.abs(
                    side_points[:, 1]
                    - (slope * side_points[:, 0] + intercept)
                )
                inliers = residuals <= 0.15
                count = int(inliers.sum())
                if count < 3:
                    continue
                score = (
                    count,
                    -abs(intercept),
                    -float(np.median(residuals[inliers])),
                )
                if best is None or score > best[0]:
                    best = score, inliers
        if best is None:
            return None
        inlier_points = side_points[best[1]]
        x = inlier_points[:, 0]
        y = inlier_points[:, 1]
        design = np.column_stack([x, np.ones_like(x)])
        weights = 1.0 / (1.0 + x / 15.0)
        coefficients, *_ = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]),
            y * np.sqrt(weights),
            rcond=None,
        )
        slope, intercept = (float(value) for value in coefficients)
        if (
            not np.isfinite(slope)
            or not np.isfinite(intercept)
            or intercept * intercept_sign <= 0.0
        ):
            return None
        return float(np.clip(slope, -1.0, 1.0)), float(
            np.clip(intercept, -scale, scale)
        )

    left = fit_side(1.0)
    right = fit_side(-1.0)
    available = [fit for fit in (left, right) if fit is not None]
    if not available:
        return 0.0, 0.0, 0.0
    lane_slope = float(np.mean([fit[0] for fit in available]))
    lane_heading_error = -float(np.arctan(lane_slope))
    if left is None or right is None:
        return 0.0, lane_heading_error, 0.0
    lane_center = 0.5 * (left[1] + right[1])
    lane_width = abs(left[1] - right[1])
    if not 2.4 <= lane_width <= 5.5:
        return 0.0, lane_heading_error, 0.0
    return -float(lane_center), lane_heading_error, float(lane_width)


def test_vectorized_lane_state_matches_scalar_reference_exactly() -> None:
    rng = np.random.default_rng(20260818)
    cases = []
    for _ in range(256):
        camera = np.zeros((21, 3), dtype=np.float32)
        count = int(rng.integers(0, 22))
        camera[:count, 0] = 1.0
        camera[:count, 1] = rng.uniform(-0.1, 0.25, count)
        camera[:count, 2] = rng.uniform(-0.08, 0.08, count)
        cases.append(camera)
    for slope in np.linspace(-0.2, 0.2, 5):
        camera = np.zeros((21, 3), dtype=np.float32)
        x = np.linspace(1.0, 12.0, 10)
        camera[:10] = np.column_stack(
            [np.ones(10), x / 64.0, (slope * x + 1.8) / 64.0]
        )
        camera[10:20] = np.column_stack(
            [np.ones(10), x / 64.0, (slope * x - 1.8) / 64.0]
        )
        cases.append(camera)

    for camera in cases:
        assert _lane_state_from_camera(
            camera, maximum_range=64.0
        ) == _lane_state_scalar_reference(camera, maximum_range=64.0)


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
