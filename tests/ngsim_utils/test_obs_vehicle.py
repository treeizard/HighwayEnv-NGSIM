import numpy as np
import pytest

from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Obstacle


def _make_vehicle(
    road: Road,
    *,
    vehicle_id: int,
    x: float,
    speed: float,
    lane_id: int = 1,
) -> NGSIMVehicle:
    dt = 0.1
    traj = np.array(
        [
            [x, 0.0, speed, lane_id],
            [x + speed * dt, 0.0, speed, lane_id],
            [x + 2.0 * speed * dt, 0.0, speed, lane_id],
        ],
        dtype=float,
    )
    vehicle = NGSIMVehicle.create(
        road=road,
        vehicle_ID=vehicle_id,
        position=traj[0][:2],
        v_length=5.0,
        v_width=2.0,
        ngsim_traj=traj,
        scene="unit-test",
        heading=0.0,
        speed=speed,
        allow_idm=True,
    )
    vehicle.on_state_update()
    return vehicle


def test_front_gap_logic_ignores_plain_replay_front_vehicle():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    rear = _make_vehicle(road, vehicle_id=1, x=0.0, speed=10.0)
    front = _make_vehicle(road, vehicle_id=2, x=10.0, speed=8.0)
    road.vehicles.extend([rear, front])

    (
        _front_vehicle,
        gap,
        desired_gap,
        relevant_front,
        handover_needed,
        _reason,
    ) = rear._front_gap_logic()

    assert relevant_front is False
    assert handover_needed is False
    assert gap == pytest.approx(100.0)
    assert desired_gap == pytest.approx(50.0)


def test_front_gap_logic_respects_overtaken_front_vehicle():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    rear = _make_vehicle(road, vehicle_id=1, x=0.0, speed=10.0)
    front = _make_vehicle(road, vehicle_id=2, x=10.0, speed=2.0)
    front.overtaken = True
    road.vehicles.extend([rear, front])

    (
        front_vehicle,
        gap,
        desired_gap,
        relevant_front,
        handover_needed,
        _reason,
    ) = rear._front_gap_logic()

    assert front_vehicle is front
    assert relevant_front is True
    assert handover_needed is True
    assert gap == pytest.approx(rear.lane_distance_to(front))
    assert desired_gap == pytest.approx(rear.desired_gap(rear, front))


def test_front_gap_logic_respects_crashed_front_vehicle():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    rear = _make_vehicle(road, vehicle_id=1, x=0.0, speed=10.0)
    front = _make_vehicle(road, vehicle_id=2, x=10.0, speed=0.0)
    front.crashed = True
    road.vehicles.extend([rear, front])

    (
        front_vehicle,
        gap,
        desired_gap,
        relevant_front,
        handover_needed,
        _reason,
    ) = rear._front_gap_logic()

    assert front_vehicle is front
    assert relevant_front is True
    assert handover_needed is True
    assert gap == pytest.approx(rear.lane_distance_to(front))
    assert desired_gap == pytest.approx(rear.desired_gap(rear, front))


def test_replay_vehicle_hands_over_when_overtaken_front_vehicle_is_too_close():
    road = Road(RoadNetwork.straight_road_network(lanes=1))
    rear = _make_vehicle(road, vehicle_id=1, x=0.0, speed=10.0)
    front = _make_vehicle(road, vehicle_id=2, x=7.0, speed=0.0)
    front.overtaken = True
    road.vehicles.extend([rear, front])

    rear.step(0.1)

    assert rear.overtaken is True


def test_handover_replay_target_lane_is_rejected_when_unsafe():
    road = Road(RoadNetwork.straight_road_network(lanes=2))
    vehicle = _make_vehicle(
        road,
        vehicle_id=1,
        x=10.0,
        speed=12.0,
        lane_id=1,
    )
    obstacle = Obstacle(road=road, position=[16.0, 4.0])
    road.vehicles.append(vehicle)
    road.objects.append(obstacle)
    vehicle.on_state_update()

    assert not vehicle._handover_target_lane_is_safe(("0", "1", 1))


def test_stale_same_segment_target_lane_is_cancelled_when_unsafe():
    road = Road(RoadNetwork.straight_road_network(lanes=2))
    vehicle = _make_vehicle(
        road,
        vehicle_id=1,
        x=10.0,
        speed=12.0,
        lane_id=1,
    )
    obstacle = Obstacle(road=road, position=[16.0, 4.0])
    road.vehicles.append(vehicle)
    road.objects.append(obstacle)
    vehicle.lane_index = ("0", "1", 0)
    vehicle.lane = road.network.get_lane(vehicle.lane_index)
    vehicle.target_lane_index = ("0", "1", 1)

    vehicle._reset_target_lane_if_unsafe()

    assert vehicle.target_lane_index == vehicle.lane_index
