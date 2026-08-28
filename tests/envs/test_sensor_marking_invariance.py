from types import SimpleNamespace

import numpy as np
from highway_env.envs.common.observations.camera import LaneCameraObservation
from highway_env.envs.common.observations.lidar import LidarObservation
from highway_env.road.lane import LineType, PolyLaneFixedWidth
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

LANE_LENGTH_M = 100.5
LANE_WIDTH_M = 4.0


def _marking_profile(lane_index: int) -> list[dict[str, object]]:
    outer = LineType.CONTINUOUS_LINE
    inner = LineType.STRIPED
    first = [outer, inner] if lane_index == 0 else [inner, outer]
    second = [outer, LineType.NONE] if lane_index == 0 else [LineType.NONE, outer]
    return [
        {"start_s_m": 0.0, "end_s_m": 40.25, "line_types": first},
        {
            "start_s_m": 40.25,
            "end_s_m": LANE_LENGTH_M,
            "line_types": second,
        },
    ]


def _sensor_env(*, marking_variant: str) -> SimpleNamespace:
    if marking_variant not in {"unmarked", "line_types", "profile"}:
        raise ValueError(f"Unknown marking variant: {marking_variant!r}")
    network = RoadNetwork()
    for lane_index, center_y in enumerate((-2.0, 2.0)):
        if marking_variant == "unmarked":
            line_types = [LineType.NONE, LineType.NONE]
            marking_profile = None
        else:
            line_types = (
                [LineType.CONTINUOUS_LINE, LineType.STRIPED]
                if lane_index == 0
                else [LineType.STRIPED, LineType.CONTINUOUS_LINE]
            )
            marking_profile = (
                _marking_profile(lane_index)
                if marking_variant == "profile"
                else None
            )
        network.add_lane(
            "start",
            "end",
            PolyLaneFixedWidth(
                [(0.0, center_y), (LANE_LENGTH_M, center_y)],
                width=LANE_WIDTH_M,
                line_types=line_types,
                marking_profile=marking_profile,
            ),
        )

    road = Road(network)
    ego = Vehicle(road, [25.0, -2.0], heading=0.0, speed=10.0)
    lead = Vehicle(road, [38.0, -2.0], heading=0.0, speed=7.0)
    road.vehicles.extend([ego, lead])
    return SimpleNamespace(road=road, vehicle=ego)


def _lidar_observation(env: SimpleNamespace) -> np.ndarray:
    return LidarObservation(
        env,
        cells=64,
        maximum_range=40.0,
        normalize=False,
        ego_centric=True,
        separate_road_edge_return=True,
        coarse_step=0.25,
        refine_iters=12,
    ).observe()


def test_lidar_road_edge_and_dynamic_returns_ignore_visual_markings() -> None:
    observations = [
        _lidar_observation(_sensor_env(marking_variant=variant))
        for variant in ("unmarked", "line_types", "profile")
    ]

    np.testing.assert_array_equal(observations[1], observations[0])
    np.testing.assert_array_equal(observations[2], observations[1])
    assert np.any(observations[2][:, LidarObservation.DYNAMIC_PRESENCE] == 1.0)
    assert np.any(observations[2][:, LidarObservation.DYNAMIC_DISTANCE] < 40.0)
    assert np.any(observations[2][:, LidarObservation.ROAD_EDGE_DISTANCE] < 40.0)


def test_lane_camera_ignores_line_types_and_marking_profile() -> None:
    observations = []
    for variant in ("unmarked", "line_types", "profile"):
        env = _sensor_env(marking_variant=variant)
        observations.append(
            LaneCameraObservation(
                env,
                cells=31,
                maximum_range=40.0,
                field_of_view=np.pi / 2.0,
                normalize=False,
                longitudinal_resolution=0.5,
            ).observe()
        )

    np.testing.assert_array_equal(observations[1], observations[0])
    np.testing.assert_array_equal(observations[2], observations[1])
    assert np.any(observations[2][:, LaneCameraObservation.PRESENCE] == 1.0)


def test_lidar_passes_through_shared_lane_boundary_and_stops_at_outer_edge() -> None:
    env = _sensor_env(marking_variant="profile")
    lidar = LidarObservation(
        env,
        cells=2,
        maximum_range=10.0,
        normalize=False,
        coarse_step=0.25,
        refine_iters=12,
    )
    origin = np.asarray([50.0, -2.0], dtype=float)
    directions = np.asarray([[0.0, 1.0], [0.0, -1.0]], dtype=float)

    distances = lidar._distance_to_road_edges_batch(
        origin,
        directions,
        max_range=10.0,
        coarse_step=0.25,
        refine_iters=12,
    )

    assert lidar._on_road_at(np.asarray([50.0, 0.0]))
    np.testing.assert_allclose(distances, [6.0, 2.0], rtol=0.0, atol=1.0e-3)
    assert distances[0] > distances[1]
