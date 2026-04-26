from __future__ import annotations

from typing import Any

import numpy as np


EGO_STATE_DIM = 4
POLICY_STATE_DIM = 3


def flatten_observation_value(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [flatten_observation_value(obs[key]) for key in sorted(obs)]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if isinstance(obs, (tuple, list)):
        parts = [flatten_observation_value(item) for item in obs]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def flatten_lidar_lane_state_observation(obs: Any) -> np.ndarray | None:
    if not isinstance(obs, tuple) or len(obs) != 3:
        return None
    lidar, lane, ego_state = obs
    lidar_arr = np.asarray(lidar, dtype=np.float32)
    lane_arr = np.asarray(lane, dtype=np.float32)
    ego_arr = np.asarray(ego_state, dtype=np.float32).reshape(-1)
    if lidar_arr.ndim != 2 or lane_arr.ndim != 2 or ego_arr.size < EGO_STATE_DIM:
        return None
    return np.concatenate(
        [lidar_arr.reshape(-1), lane_arr.reshape(-1), ego_arr[:EGO_STATE_DIM]],
        axis=0,
    ).astype(np.float32, copy=False)


def flatten_agent_observations(obs: Any) -> np.ndarray:
    if isinstance(obs, tuple) and obs and isinstance(obs[0], (tuple, list, dict)):
        fast_rows: list[np.ndarray] = []
        for item in obs:
            row = flatten_lidar_lane_state_observation(item)
            if row is None:
                return np.stack([flatten_observation_value(agent_obs) for agent_obs in obs], axis=0)
            fast_rows.append(row)
        return np.stack(fast_rows, axis=0)
    if isinstance(obs, tuple):
        row = flatten_lidar_lane_state_observation(obs)
        return (row if row is not None else flatten_observation_value(obs)).reshape(1, -1)
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim >= 2:
        return arr.reshape(arr.shape[0], -1)
    return arr.reshape(1, -1)


def policy_observations_from_flat(flat_observations: np.ndarray) -> np.ndarray:
    """Return lidar + lane + [length, velocity, heading].

    The repo's LidarCameraObservations appends ego state as
    [speed, heading, width, length]. The requested policy state is
    [length, velocity, heading], so width is intentionally omitted.
    """
    obs = np.asarray(flat_observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] < EGO_STATE_DIM:
        raise ValueError(f"Expected observations with ego state tail, got {obs.shape}.")
    sensor = obs[:, :-EGO_STATE_DIM]
    speed = obs[:, -4:-3]
    heading = obs[:, -3:-2]
    length = obs[:, -1:]
    return np.concatenate([sensor, length, speed, heading], axis=1).astype(np.float32, copy=False)
