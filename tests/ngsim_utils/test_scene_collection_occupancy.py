from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from highway_env.envs.ngsim_env import NGSimEnv
from scripts_gail import build_ps_traj_expert_discrete as collector


def _active_traj(length: int, start: int = 0, end: int | None = None) -> np.ndarray:
    traj = np.zeros((length, 4), dtype=float)
    end = length if end is None else int(end)
    for idx in range(int(start), min(int(end), int(length))):
        traj[idx] = [float(idx + 1), 1.0, 1.0, 1.0]
    return traj


def _collection_args(**overrides):
    defaults = {
        "scene": "us-101",
        "episode_root": "highway_env/data/processed_20s",
        "prebuilt_split": "train",
        "controlled_min_occupancy": 0.8,
        "max_steps_per_episode": 200,
        "max_episode_steps": 300,
        "percentage_controlled_vehicles": 0.5,
        "control_all_vehicles": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _occupancy_env(**config_overrides) -> NGSimEnv:
    env = NGSimEnv.__new__(NGSimEnv)
    config = {
        "controlled_vehicle_min_occupancy": 0.8,
        "max_episode_steps": 300,
        "scene_collection_min_occupancy_steps": None,
    }
    config.update(config_overrides)
    env.config = config
    return env


def _ego_for_traj(traj: np.ndarray, start: int = 0, end: int | None = None):
    return SimpleNamespace(
        scene_collection_full_traj=traj,
        scene_collection_start_index=int(start),
        scene_collection_end_index=int(len(traj) - 1 if end is None else end),
    )


def test_scene_collection_occupancy_uses_available_episode_horizon():
    env = _occupancy_env(max_episode_steps=300)
    ego = _ego_for_traj(_active_traj(200))

    assert env._scene_collection_spawn_active_occupancy(ego) == pytest.approx(1.0)
    assert env._scene_collection_spawn_has_min_occupancy(ego)


def test_scene_collection_occupancy_counts_late_starts_against_collection_window():
    env = _occupancy_env(max_episode_steps=200)
    passing_ego = _ego_for_traj(_active_traj(200, start=40), start=40)
    failing_ego = _ego_for_traj(_active_traj(200, start=41), start=41)

    assert env._scene_collection_spawn_active_occupancy(passing_ego) == pytest.approx(0.8)
    assert env._scene_collection_spawn_has_min_occupancy(passing_ego)
    assert env._scene_collection_spawn_active_occupancy(failing_ego) == pytest.approx(0.795)
    assert not env._scene_collection_spawn_has_min_occupancy(failing_ego)


def test_available_collection_scenarios_filters_without_probe_env(monkeypatch):
    valid_ids = {"episode-a": np.asarray([1, 2], dtype=np.int64)}
    traj_all = {
        "episode-a": {
            1: {"trajectory": _active_traj(200, start=40)},
            2: {"trajectory": _active_traj(200, start=41)},
        }
    }

    def fake_load_prebuilt_data(*_args, **_kwargs):
        return "prebuilt", valid_ids, traj_all, ["episode-a"]

    def fail_if_probe_env_is_built(_args):
        raise AssertionError("scenario discovery should not construct a probe env")

    monkeypatch.setattr(collector, "load_prebuilt_data", fake_load_prebuilt_data)
    monkeypatch.setattr(collector, "make_expert_scene_env", fail_if_probe_env_is_built)

    scenarios = collector.available_collection_scenarios(_collection_args())

    assert scenarios == [{"episode_name": "episode-a", "ego_ids": [1]}]
