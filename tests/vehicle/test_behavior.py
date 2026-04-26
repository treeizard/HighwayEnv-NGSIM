import pytest

from highway_env.ngsim_utils.road.gen_road import create_ngsim_101_road
from highway_env.ngsim_utils.road.lane_mapping import target_lane_index_from_lane_id
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


FPS = 15


def test_stop_before_obstacle():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    vehicle = IDMVehicle(road=road, position=[0, 0], speed=20, heading=0)
    obstacle = Obstacle(road=road, position=[80, 0])
    road.vehicles.append(vehicle)
    road.objects.append(obstacle)
    for _ in range(10 * FPS):
        road.act()
        road.step(dt=1 / FPS)
    assert not vehicle.crashed
    assert vehicle.position[0] < obstacle.position[0]
    assert vehicle.position[1] == pytest.approx(0)
    assert vehicle.speed == pytest.approx(0, abs=1)
    assert vehicle.heading == pytest.approx(0)


def test_idm_profile_presets():
    road = Road(RoadNetwork.straight_road_network(lanes=1))

    us_vehicle = IDMVehicle(road=road, position=[0, 0], speed=15, params_profile="us-101")
    jp_vehicle = IDMVehicle(road=road, position=[0, 0], speed=15, params_profile="japanese")

    assert us_vehicle.params.time_headway < jp_vehicle.params.time_headway
    assert us_vehicle.params.politeness < jp_vehicle.params.politeness
    assert us_vehicle.params.lane_change_delay < jp_vehicle.params.lane_change_delay


def test_ballistic_stop_prevents_negative_speed():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    vehicle = IDMVehicle(road=road, position=[0, 0], speed=10, heading=0)
    obstacle = Obstacle(road=road, position=[12, 0])
    road.vehicles.append(vehicle)
    road.objects.append(obstacle)

    road.act()
    road.step(dt=1.0)

    assert vehicle.speed == pytest.approx(0.0)
    assert vehicle.position[0] == pytest.approx(5.0, abs=1e-6)
    assert vehicle.position[0] < obstacle.position[0]


def test_standstill_vehicle_does_not_reverse():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    vehicle = IDMVehicle(road=road, position=[0, 0], speed=0, heading=0)
    road.vehicles.append(vehicle)

    vehicle.action = {"steering": 0.0, "acceleration": -5.0}
    vehicle.step(dt=1.0)

    assert vehicle.speed == pytest.approx(0.0)
    assert vehicle.position[0] == pytest.approx(0.0)


def test_safe_guard_does_not_freeze_stopped_vehicle_when_gap_is_clear():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    leader = IDMVehicle(road=road, position=[20, 0], speed=8, heading=0)
    follower = IDMVehicle(road=road, position=[0, 0], speed=0, heading=0)
    road.vehicles.extend([follower, leader])

    acc = follower.acceleration(follower, front_vehicle=leader)

    assert acc > 0.0


def test_stopped_vehicle_reaccelerates_when_leader_pulls_away():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    leader = IDMVehicle(road=road, position=[20, 0], speed=8, heading=0)
    follower = IDMVehicle(road=road, position=[0, 0], speed=0, heading=0)
    road.vehicles.extend([follower, leader])

    road.act()
    road.step(dt=1.0)

    assert follower.speed > 0.0


def test_pending_lane_change_is_abandoned_at_low_speed():
    road = Road(RoadNetwork.straight_road_network(lanes=2))
    vehicle = IDMVehicle(road=road, position=[10, 0], speed=0.5, heading=0)
    road.vehicles.append(vehicle)
    vehicle.on_state_update()
    assert vehicle.lane_index == ("0", "1", 0)

    vehicle.target_lane_index = ("0", "1", 1)
    vehicle.change_lane_policy()

    assert vehicle.target_lane_index == vehicle.lane_index


def test_mobil_rejects_target_lane_front_obstacle():
    road = Road(RoadNetwork.straight_road_network(lanes=2))
    vehicle = IDMVehicle(road=road, position=[10, 0], speed=12, heading=0)
    obstacle = Obstacle(road=road, position=[16, 4])
    road.vehicles.append(vehicle)
    road.objects.append(obstacle)
    vehicle.on_state_update()

    assert not vehicle.mobil(("0", "1", 1))


def test_mobil_rejects_target_lane_front_vehicle_requiring_hard_brake():
    road = Road(RoadNetwork.straight_road_network(lanes=2))
    vehicle = IDMVehicle(road=road, position=[10, 0], speed=12, heading=0)
    stopped_front = IDMVehicle(road=road, position=[16, 4], speed=0, heading=0)
    road.vehicles.extend([vehicle, stopped_front])
    vehicle.on_state_update()
    stopped_front.on_state_update()

    assert not vehicle.mobil(("0", "1", 1))


def test_ngsim_lane_mapping_uses_canonical_positive_lane_ids():
    net = create_ngsim_101_road()

    assert target_lane_index_from_lane_id(net, "us-101", 200.0, 6) == ("s2", "s3", 5)
    assert target_lane_index_from_lane_id(net, "us-101", 200.0, 7) == (
        "merge_in",
        "s2",
        0,
    )
    assert target_lane_index_from_lane_id(net, "us-101", 500.0, 8) == (
        "s3",
        "merge_out",
        0,
    )


def test_low_speed_steering_control_does_not_saturate_to_max_lock():
    road = Road(RoadNetwork.straight_road_network(lanes=2))
    vehicle = ControlledVehicle(road=road, position=[10, 0], speed=0.0, heading=0.0)
    road.vehicles.append(vehicle)
    vehicle.on_state_update()

    steering = vehicle.steering_control(("0", "1", 1))

    assert abs(steering) < vehicle.MAX_STEERING_ANGLE
