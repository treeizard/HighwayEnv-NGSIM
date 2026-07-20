from __future__ import annotations

import numpy as np

from highway_env.envs.ngsim_env import NGSimEnv
from highway_env.ngsim_utils.vehicles.replay import NGSIMVehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle


def _token(entity) -> str:
    return str(getattr(entity, "test_token"))


def _query_road() -> tuple[Road, Vehicle]:
    road = Road(
        RoadNetwork.straight_road_network(lanes=2, length=2000.0),
        use_query_fast_path=True,
    )
    ego = Vehicle(road, [500.0, 0.0], speed=0.0)
    ego.test_token = "ego"
    road.vehicles.append(ego)
    for index in range(70):
        lane_y = 0.0 if index % 2 == 0 else 4.0
        vehicle = Vehicle(road, [20.0 * index, lane_y], speed=0.0)
        vehicle.test_token = f"v{index}"
        road.vehicles.append(vehicle)
    for index, x_position in enumerate((480.0, 520.0, 1500.0)):
        obstacle = Obstacle(road, [x_position, 0.0])
        obstacle.test_token = f"o{index}"
        road.objects.append(obstacle)
    return road, ego


def test_spatial_queries_match_legacy_results_and_tie_order() -> None:
    road, ego = _query_road()
    query_variants = [
        {"distance": 80.0, "sort": False, "vehicles_only": False},
        {"distance": 80.0, "sort": True, "vehicles_only": False},
        {
            "distance": 80.0,
            "count": 4,
            "see_behind": False,
            "sort": True,
            "vehicles_only": True,
        },
    ]

    for kwargs in query_variants:
        road.use_query_fast_path = False
        legacy = [_token(item) for item in road.close_objects_to(ego, **kwargs)]
        road.use_query_fast_path = True
        optimized = [_token(item) for item in road.close_objects_to(ego, **kwargs)]
        assert optimized == legacy

    road.use_query_fast_path = False
    legacy_front, legacy_rear = road.neighbour_vehicles(ego)
    road.use_query_fast_path = True
    fast_front, fast_rear = road.neighbour_vehicles(ego)
    assert (_token(fast_front), _token(fast_rear)) == (
        _token(legacy_front),
        _token(legacy_rear),
    )


def _collision_road(*, use_broadphase: bool) -> Road:
    road = Road(
        RoadNetwork.straight_road_network(lanes=3, length=5000.0),
        use_collision_broadphase=use_broadphase,
        collision_cell_size=8.0,
        collision_broadphase_min_entities=0,
    )
    rng = np.random.default_rng(1234)
    for index in range(48):
        # Dense clusters exercise simultaneous collisions and original pair order;
        # sparse offsets ensure the broadphase also rejects many pairs.
        cluster = index // 6
        x_position = 35.0 * cluster + 3.5 * (index % 6)
        y_position = 4.0 * (index % 3)
        vehicle = Vehicle(
            road,
            [x_position, y_position],
            heading=float(rng.uniform(-0.08, 0.08)),
            speed=float(rng.uniform(0.0, 35.0)),
        )
        vehicle.action = {
            "steering": float(rng.uniform(-0.02, 0.02)),
            "acceleration": float(rng.uniform(-1.0, 1.0)),
        }
        road.vehicles.append(vehicle)
    road.objects.extend(
        [
            Obstacle(road, [18.0, 0.0]),
            Obstacle(road, [88.0, 4.0]),
            Obstacle(road, [1000.0, 0.0]),
        ]
    )
    return road


def _collision_snapshot(road: Road) -> list[tuple]:
    return [
        (
            np.asarray(entity.position).copy(),
            float(entity.speed),
            bool(entity.crashed),
            bool(entity.hit),
            np.asarray(entity.impact).copy(),
        )
        for entity in list(road.vehicles) + list(road.objects)
    ]


def test_collision_broadphase_matches_legacy_outcomes() -> None:
    legacy = _collision_road(use_broadphase=False)
    optimized = _collision_road(use_broadphase=True)

    legacy.step(0.1)
    optimized.step(0.1)

    for legacy_state, optimized_state in zip(
        _collision_snapshot(legacy), _collision_snapshot(optimized), strict=True
    ):
        np.testing.assert_array_equal(optimized_state[0], legacy_state[0])
        assert optimized_state[1:4] == legacy_state[1:4]
        np.testing.assert_array_equal(optimized_state[4], legacy_state[4])


def test_collision_broadphase_never_misses_exact_swept_or_object_pair() -> None:
    road = Road(
        RoadNetwork.straight_road_network(lanes=2, length=5000.0),
        use_collision_broadphase=True,
        collision_cell_size=7.0,
        collision_broadphase_min_entities=0,
    )
    rng = np.random.default_rng(20260720)
    for _ in range(80):
        road.vehicles.append(
            Vehicle(
                road,
                [float(rng.uniform(0.0, 1000.0)), float(rng.choice([0.0, 4.0]))],
                heading=float(rng.uniform(-np.pi, np.pi)),
                speed=float(rng.uniform(0.0, 90.0)),
            )
        )
    # Force a high-speed future contact and a vehicle-object contact.
    road.vehicles[0].position = np.array([0.0, 0.0])
    road.vehicles[0].heading = 0.0
    road.vehicles[0].speed = 90.0
    road.vehicles[1].position = np.array([13.0, 0.0])
    road.vehicles[1].heading = 0.0
    road.vehicles[1].speed = 0.0
    road.objects.extend([Obstacle(road, [9.0, 0.0]), Obstacle(road, [4000.0, 0.0])])

    dt = 0.1
    candidates = road._collision_candidate_indices(dt)
    assert candidates is not None
    vehicle_candidates, object_candidates = candidates
    candidate_vehicle_pairs = {
        (first, second)
        for first, seconds in enumerate(vehicle_candidates)
        for second in seconds
    }
    candidate_object_pairs = {
        (first, second)
        for first, seconds in enumerate(object_candidates)
        for second in seconds
    }

    for first, vehicle in enumerate(road.vehicles):
        for second in range(first + 1, len(road.vehicles)):
            intersecting, will_intersect, _ = vehicle._is_colliding(
                road.vehicles[second], dt
            )
            if intersecting or will_intersect:
                assert (first, second) in candidate_vehicle_pairs
        for object_index, road_object in enumerate(road.objects):
            intersecting, will_intersect, _ = vehicle._is_colliding(road_object, dt)
            if intersecting or will_intersect:
                assert (first, object_index) in candidate_object_pairs

    assert (0, 1) in candidate_vehicle_pairs
    assert (0, 0) in candidate_object_pairs


def _replay_vehicle(road: Road) -> NGSIMVehicle:
    trajectory = np.asarray(
        [
            [10.0, 0.0, 10.0, 1.0],
            [11.0, 0.0, 10.0, 1.0],
            [12.0, 0.0, 10.0, 1.0],
        ],
        dtype=float,
    )
    vehicle = NGSIMVehicle.create(
        road=road,
        vehicle_ID=1,
        position=trajectory[0, :2],
        v_length=5.0,
        v_width=2.0,
        ngsim_traj=trajectory,
        scene="unit-test",
        heading=0.0,
        speed=10.0,
        allow_idm=False,
    )
    road.vehicles.append(vehicle)
    return vehicle


def test_replay_diagnostics_can_be_disabled_without_changing_dynamics() -> None:
    diagnostic_road = Road(
        RoadNetwork.straight_road_network(lanes=1),
        record_replay_diagnostics=True,
    )
    headless_road = Road(
        RoadNetwork.straight_road_network(lanes=1),
        record_replay_diagnostics=False,
    )
    diagnostic_vehicle = _replay_vehicle(diagnostic_road)
    headless_vehicle = _replay_vehicle(headless_road)

    diagnostic_vehicle.step(0.1)
    headless_vehicle.step(0.1)

    np.testing.assert_array_equal(headless_vehicle.position, diagnostic_vehicle.position)
    assert headless_vehicle.speed == diagnostic_vehicle.speed
    assert headless_vehicle.sim_steps == diagnostic_vehicle.sim_steps
    assert diagnostic_vehicle.traj.shape[0] == 2
    assert headless_vehicle.traj.shape[0] == 1
    assert len(diagnostic_vehicle.speed_history) == 1
    assert len(headless_vehicle.speed_history) == 0
    assert len(headless_vehicle.heading_history) == 0
    assert len(headless_vehicle.crash_history) == 0
    assert len(headless_vehicle.overtaken_history) == 0


def test_ngsim_environment_wires_fast_path_modes_to_road() -> None:
    env = object.__new__(NGSimEnv)
    env.scene = "us-101"
    env.np_random = np.random.default_rng(1)
    env.config = NGSimEnv.default_config()
    env.config.update(
        {
            "road_query_mode": "spatial",
            "collision_check_mode": "broadphase",
            "road_query_cell_size": 20.0,
            "collision_broadphase_cell_size": 9.0,
            "collision_broadphase_min_entities": 12,
            "record_replay_diagnostics": False,
        }
    )

    env._create_road()

    assert env.road.use_query_fast_path is True
    assert env.road.use_collision_broadphase is True
    assert env.road.query_cell_size == 20.0
    assert env.road.collision_cell_size == 9.0
    assert env.road.collision_broadphase_min_entities == 12
    assert env.road.record_replay_diagnostics is False


def test_ngsim_fast_path_defaults_remain_legacy_compatible() -> None:
    config = NGSimEnv.default_config()
    assert config["road_query_mode"] == "legacy"
    assert config["collision_check_mode"] == "legacy"
    assert config["record_replay_diagnostics"] is True
