from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from highway_env.envs.ngsim_env import (
    VEHICLE_LIFECYCLE_CONTRACT_ID,
    VEHICLE_LIFECYCLE_EVENT_TYPES,
    NGSimEnv,
)
from highway_env.ngsim_utils.vehicles.replay import NGSIMVehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Obstacle


def _lifecycle_env(
    *,
    enabled: bool = True,
    disable_background_spawn_safety: bool = False,
    road_length: float = 100.0,
) -> NGSimEnv:
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = {
        "record_vehicle_lifecycle": enabled,
        "disable_background_replay_spawn_safety": disable_background_spawn_safety,
        "scene_dataset_collection_mode": False,
        "complete_controlled_vehicles_at_road_end": True,
        "crash_controlled_vehicles_offroad": True,
        "expert_test_mode": False,
    }
    env.scene = "unit-test"
    env.episode_name = "episode-a"
    env.steps = 0
    env._site_manifest = None
    env.controlled_vehicles = []
    env._reset_vehicle_lifecycle()
    env.road = Road(
        RoadNetwork.straight_road_network(lanes=1, length=road_length),
        np_random=np.random.RandomState(0),
    )
    env.road.vehicle_lifecycle_event_recorder = env._record_vehicle_lifecycle_event if enabled else None
    env.road.disable_background_replay_spawn_safety = disable_background_spawn_safety
    return env


def _replay_vehicle(
    road: Road,
    trajectory: np.ndarray,
    *,
    vehicle_id: int,
    allow_idm: bool = False,
) -> NGSIMVehicle:
    vehicle = NGSIMVehicle.create(
        road=road,
        vehicle_ID=vehicle_id,
        position=np.asarray(trajectory[0, :2], dtype=float),
        v_length=5.0,
        v_width=2.0,
        ngsim_traj=np.asarray(trajectory, dtype=float),
        scene="unit-test",
        heading=0.0,
        speed=float(trajectory[0, 2]),
        allow_idm=allow_idm,
    )
    vehicle.on_state_update()
    return vehicle


def test_lifecycle_contract_is_opt_in_stable_and_json_safe():
    assert NGSimEnv.default_config()["record_vehicle_lifecycle"] is False
    assert NGSimEnv.default_config()["disable_background_replay_spawn_safety"] is False
    assert NGSimEnv.default_config()["runtime_eligibility_mode"] is None

    env = _lifecycle_env(enabled=False)
    env._record_vehicle_lifecycle_event(
        "trajectory_exhaustion",
        vehicle_id=4,
        actor_role="background",
    )

    summary = env.vehicle_lifecycle_summary()
    assert summary["contract_id"] == VEHICLE_LIFECYCLE_CONTRACT_ID
    assert summary["event_cardinality"] == "once_per_vehicle_role_event_type"
    assert summary["enabled"] is False
    assert summary["data_binding_id"] is None
    assert summary["runtime_policy"]["record_vehicle_lifecycle"] is False
    assert summary["runtime_policy"]["disable_background_replay_spawn_safety"] is False
    assert summary["event_count"] == 0
    assert tuple(summary["event_counts"]) == VEHICLE_LIFECYCLE_EVENT_TYPES
    assert all(count == 0 for count in summary["event_counts"].values())
    json.dumps(summary, allow_nan=False)

    disabled_replay = _replay_vehicle(
        env.road,
        np.asarray([[10.0, 0.0, 8.0, 1.0]]),
        vehicle_id=3,
    )
    env.road.vehicles.append(disabled_replay)
    disabled_replay.step(0.1)
    assert disabled_replay.remove_from_road is True
    assert not hasattr(disabled_replay, "lifecycle_removal_reason")

    enabled_env = _lifecycle_env()
    for vehicle_id in (4, 4, 5):
        enabled_env._record_vehicle_lifecycle_event(
            "background_initial_spawn_conflict",
            vehicle_id=vehicle_id,
            actor_role="background",
        )
    enabled_summary = enabled_env.vehicle_lifecycle_summary()
    assert enabled_summary["event_count"] == 2
    assert [event["sequence"] for event in enabled_summary["events"]] == [0, 1]
    compact_summary = enabled_env.vehicle_lifecycle_summary(include_events=False)
    assert compact_summary["event_count"] == 2
    assert compact_summary["events_included"] is False
    assert compact_summary["events"] == []


def test_create_road_attaches_opt_in_lifecycle_and_background_spawn_policy():
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = NGSimEnv.default_config()
    env.config["record_vehicle_lifecycle"] = True
    env.config["disable_background_replay_spawn_safety"] = True
    env.scene = "us-101"
    env._road_geometry_manifest = None
    env._prebuilt_dir = "."
    env._vehicle_lifecycle_enabled = True
    env.np_random = np.random.RandomState(0)

    env._create_road()

    assert callable(env.road.vehicle_lifecycle_event_recorder)
    assert env.road.disable_background_replay_spawn_safety is True


def test_step_info_exposes_compact_summary_only_when_enabled():
    def add_controlled_vehicle(env: NGSimEnv, vehicle_id: int) -> SimpleNamespace:
        vehicle = SimpleNamespace(
            vehicle_ID=vehicle_id,
            speed=8.0,
            crashed=False,
            completed=False,
            on_road=True,
        )
        env.controlled_vehicles = [vehicle]
        env.road.vehicles = [vehicle]
        env.ego_ids = [vehicle_id]
        return vehicle

    env = _lifecycle_env()
    controlled = add_controlled_vehicle(env, 14)
    env._record_vehicle_lifecycle_event(
        "source_before_active",
        vehicle=controlled,
        actor_role="controlled",
        source_index=0,
    )
    info = env._info(np.zeros(1), action=None)
    assert info["vehicle_lifecycle_summary"]["event_count"] == 1
    assert info["vehicle_lifecycle_summary"]["events_included"] is False
    assert info["vehicle_lifecycle_summary"]["events"] == []

    disabled_env = _lifecycle_env(enabled=False)
    add_controlled_vehicle(disabled_env, 15)
    assert "vehicle_lifecycle_summary" not in disabled_env._info(
        np.zeros(1),
        action=None,
    )


def test_controlled_source_inactivity_and_conflict_suppression_are_distinct():
    env = _lifecycle_env()
    env.config["scene_dataset_collection_mode"] = True
    env.control_mode = "teleport"
    trajectory = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 8.0, 1.0],
            [11.0, 0.0, 8.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    ego = SimpleNamespace(
        vehicle_ID=7,
        scene_collection_full_traj=trajectory,
        scene_collection_start_index=1,
        scene_collection_end_index=2,
        scene_collection_is_active=False,
        scene_collection_padding_position=np.asarray([-1.0e6, -1.0e6]),
        scene_collection_real_length=5.0,
        scene_collection_real_width=2.0,
        heading=0.0,
        visible=False,
        position=np.zeros(2),
        speed=0.0,
        target_speed=0.0,
        LENGTH=0.0,
        WIDTH=0.0,
    )
    env.controlled_vehicles = [ego]
    env._scene_collection_row_has_conflict = lambda *_args, **_kwargs: True

    env.steps = 0
    env._sync_scene_collection_controlled_vehicles(step_index=0)
    env.steps = 1
    env._sync_scene_collection_controlled_vehicles(step_index=1)
    env.steps = 3
    env._sync_scene_collection_controlled_vehicles(step_index=3)

    summary = env.vehicle_lifecycle_summary()
    assert summary["event_counts"]["source_before_active"] == 1
    assert summary["event_counts"]["source_after_end"] == 1
    assert summary["event_counts"]["controlled_spawn_conflict_deactivation"] == 1
    assert summary["actor_role_event_counts"]["controlled"]["controlled_spawn_conflict_deactivation"] == 1
    assert [event["event_type"] for event in summary["events"]] == [
        "source_before_active",
        "controlled_spawn_conflict_deactivation",
        "source_after_end",
    ]


def test_episode_store_initial_spawn_conflict_and_opt_in_bypass():
    trajectory_set = {
        "ego": {},
        9: {
            "trajectory": np.asarray([[20.0, 0.0, 10.0, 1.0], [21.0, 0.0, 10.0, 1.0]]),
            "length": 5.0,
            "width": 2.0,
        },
    }

    env = _lifecycle_env()
    env.trajectory_set = trajectory_set
    env._ego_start_indices = {}
    env._episode_timestep_s = 0.1
    env.road.objects.append(Obstacle(env.road, position=[20.0, 0.0]))
    env._spawn_si_surrounding_vehicles(max_surrounding=1)
    assert env.road.vehicles == []
    assert env.vehicle_lifecycle_summary()["event_counts"]["background_initial_spawn_conflict"] == 1

    source_faithful_env = _lifecycle_env(disable_background_spawn_safety=True)
    source_faithful_env.trajectory_set = trajectory_set
    source_faithful_env._ego_start_indices = {}
    source_faithful_env._episode_timestep_s = 0.1
    source_faithful_env.road.objects.append(Obstacle(source_faithful_env.road, position=[20.0, 0.0]))
    source_faithful_env._spawn_si_surrounding_vehicles(max_surrounding=1)
    assert len(source_faithful_env.road.vehicles) == 1
    source_faithful_summary = source_faithful_env.vehicle_lifecycle_summary()
    assert source_faithful_summary["runtime_policy"]["disable_background_replay_spawn_safety"] is True
    assert source_faithful_summary["event_counts"]["background_initial_spawn_conflict"] == 0


def test_background_delayed_spawn_conflict_and_opt_in_bypass():
    trajectory = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 10.0, 1.0],
            [21.0, 0.0, 10.0, 1.0],
        ]
    )

    env = _lifecycle_env()
    env.road.objects.append(Obstacle(env.road, position=[20.0, 0.0]))
    replay = _replay_vehicle(env.road, trajectory, vehicle_id=12)
    env.road.vehicles.append(replay)
    replay.step(0.1)
    replay.step(0.1)
    summary = env.vehicle_lifecycle_summary()
    assert summary["event_counts"]["source_before_active"] == 1
    assert summary["event_counts"]["background_delayed_spawn_conflict"] == 1
    assert replay.appear is False

    source_faithful_env = _lifecycle_env(disable_background_spawn_safety=True)
    source_faithful_env.road.objects.append(Obstacle(source_faithful_env.road, position=[20.0, 0.0]))
    source_faithful = _replay_vehicle(
        source_faithful_env.road,
        trajectory,
        vehicle_id=12,
    )
    source_faithful_env.road.vehicles.append(source_faithful)
    source_faithful.step(0.1)
    source_faithful.step(0.1)
    assert source_faithful.appear is True
    np.testing.assert_allclose(source_faithful.position, [20.0, 0.0])
    assert source_faithful_env.vehicle_lifecycle_summary()["event_counts"]["background_delayed_spawn_conflict"] == 0


def test_sample_anchor_v2_controlled_replay_restores_later_active_span():
    trajectory = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 8.0, 1.0],
            [11.0, 0.0, 8.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 8.0, 1.0],
            [21.0, 0.0, 8.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    legacy_env = _lifecycle_env()
    legacy_env.config["scene_dataset_collection_mode"] = True
    legacy_env.control_mode = "teleport"
    legacy_ego = SimpleNamespace(vehicle_ID=50, LENGTH=5.0, WIDTH=2.0)
    legacy_env._configure_scene_collection_vehicle(
        ego=legacy_ego,
        ego_rec={},
        ego_traj_full=trajectory,
    )
    assert legacy_ego.scene_collection_start_index == 1
    assert legacy_ego.scene_collection_end_index == 2

    env = _lifecycle_env()
    env.config.update(
        {
            "scene_dataset_collection_mode": True,
            "disable_scene_collection_spawn_safety": True,
            "runtime_eligibility_mode": "sample_anchor_v2",
        }
    )
    env.control_mode = "teleport"
    ego = SimpleNamespace(vehicle_ID=50, LENGTH=5.0, WIDTH=2.0)
    env._configure_scene_collection_vehicle(
        ego=ego,
        ego_rec={},
        ego_traj_full=trajectory,
    )
    assert ego.scene_collection_start_index == 1
    assert ego.scene_collection_end_index == 6

    def set_from_source_row(
        replay_ego,
        row,
        **_kwargs,
    ) -> None:
        replay_ego.position = np.asarray(row[:2], dtype=float)
        replay_ego.speed = float(row[2])
        replay_ego.target_speed = float(row[2])
        replay_ego.LENGTH = replay_ego.scene_collection_real_length
        replay_ego.WIDTH = replay_ego.scene_collection_real_width
        replay_ego.visible = True
        replay_ego.scene_collection_is_active = True

    env._set_scene_collection_vehicle_from_row = set_from_source_row
    env.controlled_vehicles = [ego]
    env._sync_scene_collection_controlled_vehicles(step_index=1)
    assert ego.scene_collection_is_active is True
    np.testing.assert_allclose(ego.position, [10.0, 0.0])

    env._sync_scene_collection_controlled_vehicles(step_index=3)
    assert ego.scene_collection_is_active is False

    env._sync_scene_collection_controlled_vehicles(step_index=5)
    assert ego.scene_collection_is_active is True
    np.testing.assert_allclose(ego.position, [20.0, 0.0])
    assert env.vehicle_lifecycle_summary()["event_counts"]["source_after_end"] == 0


def test_sample_anchor_v2_scene_collection_preserves_episode_zero_origin():
    records = {
        7: {
            "trajectory": np.asarray(
                [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 8.0, 1.0]]
            )
        },
        8: {
            "trajectory": np.asarray(
                [[0.0, 0.0, 0.0, 0.0], [12.0, 0.0, 8.0, 1.0]]
            )
        },
    }
    env = _lifecycle_env()
    env.ego_ids = [7, 8]
    env.config["scene_dataset_collection_mode"] = True
    env.config["runtime_eligibility_mode"] = "sample_anchor_v2"

    assert env._resolve_shared_ego_start_index(records) == 0

    env.config["runtime_eligibility_mode"] = None
    assert env._resolve_shared_ego_start_index(records) == 1


def test_source_faithful_background_replay_restores_later_active_span():
    trajectory = np.asarray(
        [
            [10.0, 0.0, 8.0, 1.0],
            [11.0, 0.0, 8.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 8.0, 1.0],
            [21.0, 0.0, 8.0, 1.0],
        ]
    )

    legacy_env = _lifecycle_env()
    legacy_replay = _replay_vehicle(
        legacy_env.road,
        trajectory,
        vehicle_id=51,
    )
    legacy_env.road.vehicles.append(legacy_replay)
    for _ in range(3):
        legacy_replay.step(0.1)
    assert legacy_replay.remove_from_road is True
    assert legacy_env.vehicle_lifecycle_summary()["event_counts"]["source_after_end"] == 1

    env = _lifecycle_env(disable_background_spawn_safety=True)
    replay = _replay_vehicle(env.road, trajectory, vehicle_id=51)
    env.road.vehicles.append(replay)
    for _ in range(5):
        replay.step(0.1)
    assert replay.remove_from_road is False
    assert replay.appear is True
    np.testing.assert_allclose(replay.position, [20.0, 0.0])
    assert env.vehicle_lifecycle_summary()["event_counts"]["source_after_end"] == 0


def test_replay_end_states_terminal_removal_and_pruning_are_distinct():
    env = _lifecycle_env()

    source_after = _replay_vehicle(
        env.road,
        np.asarray(
            [
                [10.0, 0.0, 10.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        vehicle_id=20,
    )
    exhausted = _replay_vehicle(
        env.road,
        np.asarray([[30.0, 0.0, 10.0, 1.0]]),
        vehicle_id=21,
    )
    terminal = _replay_vehicle(
        env.road,
        np.asarray(
            [
                [98.0, 0.0, 10.0, 1.0],
                [99.0, 0.0, 10.0, 1.0],
                [100.0, 0.0, 10.0, 1.0],
            ]
        ),
        vehicle_id=22,
    )
    env.road.vehicles.extend([source_after, exhausted, terminal])

    source_after.step(0.1)
    source_after.step(0.1)
    exhausted.step(0.1)
    terminal.step(0.1)
    env._prune_removed_vehicles()

    summary = env.vehicle_lifecycle_summary()
    assert summary["event_counts"]["source_after_end"] == 1
    assert summary["event_counts"]["trajectory_exhaustion"] == 1
    assert summary["event_counts"]["terminal_lane_removal"] == 1
    assert summary["event_counts"]["pruning"] == 3
    pruning_reasons = {
        event["details"]["removal_reason"] for event in summary["events"] if event["event_type"] == "pruning"
    }
    assert pruning_reasons == {
        "source_after_end",
        "trajectory_exhaustion",
        "terminal_lane_removal",
    }


def test_source_faithful_background_replay_keeps_logged_terminal_lane_rows():
    trajectory = np.asarray(
        [[98.0, 0.0, 10.0, 1.0], [99.0, 0.0, 10.0, 1.0], [100.0, 0.0, 10.0, 1.0]]
    )
    default_env = _lifecycle_env()
    default_replay = _replay_vehicle(default_env.road, trajectory, vehicle_id=23)
    default_env.road.vehicles.append(default_replay)
    default_replay.step(0.1)
    assert default_replay.remove_from_road is True

    source_env = _lifecycle_env()
    source_env.road.source_faithful_background_replay = True
    source_replay = _replay_vehicle(source_env.road, trajectory, vehicle_id=23)
    source_env.road.vehicles.append(source_replay)
    source_replay.step(0.1)
    assert source_replay.remove_from_road is False
    assert source_replay.appear is True
    np.testing.assert_allclose(source_replay.position, [98.0, 0.0])
    assert source_env.vehicle_lifecycle_summary()["event_counts"][
        "terminal_lane_removal"
    ] == 0


def test_source_faithful_background_sync_activates_exact_reached_row():
    trajectory = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 8.0, 1.0],
            [21.0, 0.0, 9.0, 1.0],
        ]
    )
    env = _lifecycle_env(disable_background_spawn_safety=True)
    env.config["source_faithful_background_replay"] = True
    env.road.source_faithful_background_replay = True
    replay = _replay_vehicle(env.road, trajectory, vehicle_id=52)
    env.road.vehicles.append(replay)

    env._sync_source_faithful_background_vehicles(step_index=2)

    assert replay.sim_steps == 2
    assert replay.appear is True
    assert replay.visible is True
    np.testing.assert_allclose(replay.position, [20.0, 0.0])
    assert replay.speed == 8.0
    assert replay.LENGTH == replay.real_length
    assert replay.WIDTH == replay.real_width


def test_source_faithful_background_sync_preserves_irreversible_idm_handover():
    trajectory = np.asarray(
        [
            [10.0, 0.0, 8.0, 1.0],
            [11.0, 0.0, 8.0, 1.0],
            [12.0, 0.0, 8.0, 1.0],
        ]
    )
    env = _lifecycle_env(disable_background_spawn_safety=True)
    env.config["source_faithful_background_replay"] = True
    env.road.source_faithful_background_replay = True
    replay = _replay_vehicle(env.road, trajectory, vehicle_id=53, allow_idm=True)
    replay.overtaken = True
    replay.position = np.asarray([42.0, 1.0])
    env.road.vehicles.append(replay)

    env._sync_source_faithful_background_vehicles(step_index=2)

    np.testing.assert_allclose(replay.position, [42.0, 1.0])
    assert replay.overtaken is True


def test_controlled_completion_offroad_and_collision_crashes_are_distinct():
    completion_env = _lifecycle_env()
    completed = SimpleNamespace(
        vehicle_ID=30,
        crashed=False,
        completed=False,
        lane_index=("0", "1", 0),
        target_lane_index=("0", "1", 0),
        collidable=True,
        check_collisions=True,
        remove_from_road=False,
    )
    completion_env.controlled_vehicles = [completed]
    completion_env.road.vehicles = [completed]
    completion_env._vehicle_reached_terminal_road_end = lambda _vehicle: True
    completion_env._complete_road_end_controlled_vehicles()
    completion_env._prune_removed_vehicles()
    completion_summary = completion_env.vehicle_lifecycle_summary()
    assert completion_summary["event_counts"]["road_end_completion"] == 1
    assert completion_summary["event_counts"]["pruning"] == 1

    offroad_env = _lifecycle_env()
    offroad = SimpleNamespace(
        vehicle_ID=31,
        crashed=False,
        completed=False,
        on_road=False,
        lane_index=("0", "1", 0),
    )
    offroad_env.controlled_vehicles = [offroad]
    offroad_env.road.vehicles = [offroad]
    offroad_env._crash_offroad_controlled_vehicles()
    assert offroad_env.vehicle_lifecycle_summary()["event_counts"]["offroad_crash"] == 1

    collision_env = _lifecycle_env()
    collided = SimpleNamespace(
        vehicle_ID=32,
        crashed=True,
        first_collision_partner_vehicle_id=44,
        first_collision_partner_type="NGSIMVehicle",
        first_collision_partner_provenance="physics_current_intersection",
    )
    collision_env.controlled_vehicles = [collided]
    collision_env.road.vehicles = [collided]
    collision_env._record_new_collision_crashes({id(collided): False})
    collision_summary = collision_env.vehicle_lifecycle_summary()
    assert collision_summary["event_counts"]["collision_crash"] == 1
    assert collision_summary["events"][0]["details"]["partner_vehicle_id"] == 44


def test_idm_handover_is_recorded_once_with_reason():
    env = _lifecycle_env()
    rear = _replay_vehicle(
        env.road,
        np.asarray(
            [
                [0.1, 0.0, 10.0, 1.0],
                [1.1, 0.0, 10.0, 1.0],
                [2.1, 0.0, 10.0, 1.0],
            ]
        ),
        vehicle_id=40,
        allow_idm=True,
    )
    front = _replay_vehicle(
        env.road,
        np.asarray(
            [
                [7.0, 0.0, 0.0, 1.0],
                [7.0, 0.0, 0.0, 1.0],
                [7.0, 0.0, 0.0, 1.0],
            ]
        ),
        vehicle_id=41,
        allow_idm=True,
    )
    front.overtaken = True
    env.road.vehicles.extend([rear, front])

    rear.step(0.1)
    rear.step(0.1)

    handovers = [event for event in env.vehicle_lifecycle_summary()["events"] if event["event_type"] == "idm_handover"]
    assert len(handovers) == 1
    assert handovers[0]["details"]["reason"]
