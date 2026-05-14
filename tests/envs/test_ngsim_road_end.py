import numpy as np

from highway_env.envs.ngsim_env import NGSimEnv
from highway_env.ngsim_utils.road.gen_road import (
    create_japanese_road,
    create_ngsim_101_road,
)
from highway_env.ngsim_utils.vehicles.ego import EgoVehicle
from highway_env.road.road import Road


def _make_ngsim_env_with_road(scene: str) -> NGSimEnv:
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = NGSimEnv.default_config()
    env.config["scene"] = scene
    env.scene = scene
    env.net = create_japanese_road() if scene == "japanese" else create_ngsim_101_road()
    env.road = Road(network=env.net, np_random=np.random.RandomState(0))
    env.controlled_vehicles = []
    env.vehicle = None
    return env


def _add_vehicle_at_front_bumper_end(env: NGSimEnv, lane_index):
    lane = env.road.network.get_lane(lane_index)
    vehicle = EgoVehicle(
        road=env.road,
        position=lane.position(lane.length - EgoVehicle.LENGTH / 2.0, 0.0),
        heading=lane.heading_at(lane.length),
        speed=10.0,
        control_mode="continuous",
    )
    vehicle.target_lane_index = lane_index
    vehicle.lane_index = lane_index
    vehicle.lane = lane
    env.road.vehicles.append(vehicle)
    env.controlled_vehicles.append(vehicle)
    env.vehicle = vehicle
    return vehicle


def test_us101_controlled_vehicle_completes_at_terminal_road_end():
    env = _make_ngsim_env_with_road("us-101")
    vehicle = _add_vehicle_at_front_bumper_end(env, ("s3", "s4", 0))

    env._complete_road_end_controlled_vehicles()
    env._crash_offroad_controlled_vehicles()

    assert vehicle.completed is True
    assert vehicle.reached_road_end is True
    assert vehicle.remove_from_road is True
    assert vehicle.crashed is False
    assert env._is_terminated() is True


def test_us101_controlled_vehicle_completes_at_merge_exit_end():
    env = _make_ngsim_env_with_road("us-101")
    vehicle = _add_vehicle_at_front_bumper_end(env, ("s3", "merge_out", 0))

    env._complete_road_end_controlled_vehicles()
    env._crash_offroad_controlled_vehicles()

    assert vehicle.completed is True
    assert vehicle.reached_road_end is True
    assert vehicle.remove_from_road is True
    assert vehicle.crashed is False
    assert env._is_terminated() is True


def test_japanese_controlled_vehicle_completes_at_terminal_road_end():
    env = _make_ngsim_env_with_road("japanese")
    vehicle = _add_vehicle_at_front_bumper_end(env, ("c", "d", 0))

    env._complete_road_end_controlled_vehicles()
    env._crash_offroad_controlled_vehicles()

    assert vehicle.completed is True
    assert vehicle.reached_road_end is True
    assert vehicle.remove_from_road is True
    assert vehicle.crashed is False
    assert env._is_terminated() is True


def test_lateral_departure_near_road_end_still_crashes_as_offroad():
    env = _make_ngsim_env_with_road("japanese")
    lane_index = ("c", "d", 0)
    vehicle = _add_vehicle_at_front_bumper_end(env, lane_index)
    lane = env.road.network.get_lane(lane_index)
    vehicle.position = lane.position(
        lane.length - vehicle.LENGTH / 2.0,
        -3.0 * lane.width,
    )

    env._complete_road_end_controlled_vehicles()
    env._crash_offroad_controlled_vehicles()

    assert getattr(vehicle, "completed", False) is False
    assert vehicle.crashed is True
