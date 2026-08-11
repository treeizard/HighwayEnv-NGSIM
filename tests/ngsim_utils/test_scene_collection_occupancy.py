from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from highway_env.envs.ngsim_env import NGSimEnv
from policy.data import collect_expert as collector


def _active_traj(length: int, start: int = 0, end: int | None = None) -> np.ndarray:
    traj = np.zeros((length, 4), dtype=float)
    end = length if end is None else int(end)
    for idx in range(int(start), min(int(end), int(length))):
        traj[idx] = [float(idx + 1), 1.0, 1.0, 1.0]
    return traj


def _collection_args(**overrides):
    defaults = {
        "scene": "us-101",
        "episode_root": "data/highway_env/processed_20s",
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


def test_poststep_activation_refreshes_the_observation_before_return():
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = {"scene_dataset_collection_mode": True}
    env.control_mode = "continuous"
    env.steps = 7
    state = {"active": False, "observed_after_sync": False}

    def sync(*, step_index: int) -> None:
        assert step_index == 7
        state["active"] = True

    class Observation:
        @staticmethod
        def observe() -> np.ndarray:
            state["observed_after_sync"] = bool(state["active"])
            return np.asarray([1.0 if state["active"] else -1.0])

    def info(obs, _action):
        return {"observation_echo": np.asarray(obs).copy()}

    env._sync_scene_collection_controlled_vehicles = sync
    env.observation_type = Observation()
    env._info = info

    refreshed, refreshed_info = env._refresh_scene_collection_observation_after_sync(
        np.asarray([-1.0]),
        np.zeros(2, dtype=np.float32),
    )

    assert state["observed_after_sync"]
    np.testing.assert_array_equal(refreshed, [1.0])
    np.testing.assert_array_equal(
        refreshed_info["observation_echo"],
        refreshed,
    )


def test_teleport_scene_replay_uses_only_past_recorded_motion_heading():
    class Lane:
        @staticmethod
        def local_coordinates(_position):
            return 0.0, 0.0

        @staticmethod
        def heading_at(_longitudinal):
            return 0.0

    class Network:
        @staticmethod
        def get_lane(_lane_index):
            return Lane()

    env = NGSimEnv.__new__(NGSimEnv)
    env.scene = "japanese"
    env.config = {
        "simulation_frequency": 10,
        "source_preserving_trajectory_state": True,
    }
    env.road = SimpleNamespace(network=Network())
    ego = SimpleNamespace(
        heading=0.0,
        LENGTH=4.5,
        WIDTH=1.8,
        scene_collection_real_length=4.5,
        scene_collection_real_width=1.8,
    )
    row = np.asarray([10.0, 2.0, 12.0, 1.0])
    previous_row = np.asarray([9.0, 1.8, 12.0, 1.0])
    next_row = np.asarray([11.0, 2.2, 12.0, 1.0])

    env.control_mode = "teleport"
    env._set_scene_collection_vehicle_from_row(
        ego,
        row,
        previous_row=previous_row,
        next_row=next_row,
    )
    assert ego.heading == pytest.approx(np.arctan2(0.2, 1.0))

    # Metamorphic causal check: changing every future coordinate cannot alter
    # the actor-visible pose at the current row.
    future_perturbed = np.asarray([-400.0, 900.0, 12.0, 1.0])
    env._set_scene_collection_vehicle_from_row(
        ego,
        row,
        previous_row=previous_row,
        next_row=future_perturbed,
    )
    assert ego.heading == pytest.approx(np.arctan2(0.2, 1.0))

    env.control_mode = "continuous"
    env._set_scene_collection_vehicle_from_row(
        ego,
        row,
        previous_row=previous_row,
        next_row=next_row,
    )
    assert ego.heading == pytest.approx(0.0)


def test_source_preserving_scene_replay_rejects_negative_speed_without_source_mutation():
    env = NGSimEnv.__new__(NGSimEnv)
    env.scene = "japanese"
    env.control_mode = "teleport"
    env.config = {
        "simulation_frequency": 10,
        "source_preserving_trajectory_state": True,
    }

    class Lane:
        @staticmethod
        def local_coordinates(_position):
            return 0.0, 0.0

        @staticmethod
        def heading_at(_longitudinal):
            return 0.0

    class Network:
        @staticmethod
        def get_lane(_lane_index):
            return Lane()

    env.road = SimpleNamespace(network=Network())
    ego = SimpleNamespace(
        heading=0.0,
        LENGTH=4.5,
        WIDTH=1.8,
        scene_collection_real_length=4.5,
        scene_collection_real_width=1.8,
    )

    row = np.asarray([10.0, 2.0, -0.1, 1.0])
    source_copy = row.copy()
    with pytest.raises(ValueError, match="negative recorded speed"):
        env._set_scene_collection_vehicle_from_row(ego, row)
    np.testing.assert_array_equal(row, source_copy)


def test_delayed_scene_activation_resets_tracker_to_source_relative_offset():
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = {"disable_scene_collection_spawn_safety": True}
    env.control_mode = "continuous"
    reset_offsets: list[int] = []
    tracker = SimpleNamespace(reset=lambda *, k0=0: reset_offsets.append(int(k0)))
    env._expert_state_by_vehicle_id = {7: {"tracker": tracker}}
    ego = SimpleNamespace(
        vehicle_ID=7,
        scene_collection_is_active=False,
        scene_collection_start_index=3,
        scene_collection_full_traj=_active_traj(12, start=3),
    )

    def activate_from_row(
        _ego,
        _row,
        *,
        previous_row=None,
        next_row=None,
        provider_observation_flag=None,
    ):
        assert previous_row is not None
        assert next_row is not None
        assert provider_observation_flag is None
        _ego.scene_collection_is_active = True

    env._set_scene_collection_vehicle_from_row = activate_from_row

    env._activate_scene_collection_vehicle(ego, step_index=5)
    assert reset_offsets == [2]
    assert env._expert_state_by_vehicle_id[7]["activation_tracker_offset"] == 2

    env._activate_scene_collection_vehicle(ego, step_index=6)
    assert reset_offsets == [2]


def test_external_scene_controller_configures_active_span_without_privileged_tracker():
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = {"scene_collection_external_controller": True}
    env.control_mode = "continuous"
    env._expert_state_by_vehicle_id = {}
    env._deactivate_scene_collection_vehicle = lambda _ego: setattr(
        _ego, "scene_collection_is_active", False
    )
    ego = SimpleNamespace(vehicle_ID=7, LENGTH=4.5, WIDTH=1.8)
    traj = _active_traj(12, start=3, end=10)

    env._configure_scene_collection_vehicle(
        ego=ego,
        ego_rec={},
        ego_traj_full=traj,
    )

    assert ego.scene_collection_start_index == 3
    assert ego.scene_collection_end_index == 9
    np.testing.assert_array_equal(ego.scene_collection_spawn_position, [4.0, 1.0])
    assert ego.scene_collection_spawn_speed == pytest.approx(1.0)


def test_external_scene_controller_activation_does_not_require_privileged_tracker():
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = {
        "disable_scene_collection_spawn_safety": True,
        "scene_collection_external_controller": True,
    }
    env.control_mode = "continuous"
    env._expert_state_by_vehicle_id = {}
    ego = SimpleNamespace(
        vehicle_ID=7,
        scene_collection_is_active=False,
        scene_collection_start_index=3,
        scene_collection_provider_observation_mask=None,
        scene_collection_full_traj=_active_traj(12, start=3),
    )
    activations: list[int] = []

    def activate_from_row(
        _ego,
        _row,
        *,
        previous_row=None,
        next_row=None,
        provider_observation_flag=None,
    ):
        assert previous_row is not None
        assert next_row is not None
        assert provider_observation_flag is None
        activations.append(1)
        _ego.scene_collection_is_active = True

    env._set_scene_collection_vehicle_from_row = activate_from_row

    env._activate_scene_collection_vehicle(ego, step_index=5)

    assert activations == [1]
    assert ego.scene_collection_is_active


def test_scene_collection_conflicts_are_checked_at_actual_activation_time():
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = {"disable_scene_collection_spawn_safety": False}
    env.control_mode = "continuous"
    tracker_resets: list[int] = []
    env._expert_state_by_vehicle_id = {7: {"tracker": SimpleNamespace(reset=lambda *, k0=0: tracker_resets.append(k0))}}
    ego = SimpleNamespace(
        vehicle_ID=7,
        scene_collection_is_active=False,
        scene_collection_start_index=3,
        scene_collection_full_traj=_active_traj(12, start=3),
    )
    deactivated: list[int] = []
    env._scene_collection_row_has_conflict = lambda _ego, _row, *, previous_row=None, next_row=None: True
    env._deactivate_scene_collection_vehicle = lambda _ego: deactivated.append(int(_ego.vehicle_ID))

    env._activate_scene_collection_vehicle(ego, step_index=5)

    assert deactivated == [7]
    assert tracker_resets == []
