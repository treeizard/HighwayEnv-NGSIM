"""Build and query NGSIM road topology mappings."""

# Modified by: Yide Tao (yide.tao@monash.edu)
# Reference: @article{huang2021driving,
#   title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
#   author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
#   journal={IEEE Transactions on Intelligent Transportation Systems},
#   year={2021},
#   publisher={IEEE}
# }
# @misc{highway-env,
#   author = {Leurent, Edouard},
#   title = {An Environment for Autonomous Driving Decision-Making},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/eleurent/highway-env}},
# }


import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from highway_env.road.lane import LineType, PolyLaneFixedWidth, StraightLane, SineLane
from highway_env.road.road import RoadNetwork
from highway_env.ngsim_utils.core.constants import (
    US101_LANE_WIDTH_M,
    US101_MAINLINE_LENGTH_M,
    US101_MERGE_IN_START_M,
    US101_MERGE_OUT_END_M,
    US101_SECTION_ENDS_M,
)

def create_ngsim_101_road():
    net = RoadNetwork()
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

    length = US101_MAINLINE_LENGTH_M
    width = US101_LANE_WIDTH_M
    ends = US101_SECTION_ENDS_M

    # first section (5 lanes)
    line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
    for lane in range(5):
        origin = [ends[0], lane * width]
        end = [ends[1], lane * width]
        net.add_lane("s1", "s2", StraightLane(origin, end, width=width, line_types=line_types[lane]))

    # merge_in (forbidden)
    net.add_lane("merge_in", "s2", StraightLane([US101_MERGE_IN_START_M, 5.5*width], [ends[1], 5*width], width=width, line_types=[c, c], forbidden=True))

    # second section (6 lanes)
    line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
    for lane in range(6):
        origin = [ends[1], lane * width]
        end = [ends[2], lane * width]
        net.add_lane("s2", "s3", StraightLane(origin, end, width=width, line_types=line_types[lane]))

    # third section (5 lanes)
    line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
    for lane in range(5):
        origin = [ends[2], lane * width]
        end = [ends[3], lane * width]
        net.add_lane("s3", "s4", StraightLane(origin, end, width=width, line_types=line_types[lane]))

    # merge_out (forbidden)
    net.add_lane("s3", "merge_out", StraightLane([ends[2], 5*width], [US101_MERGE_OUT_END_M, 7*width], width=width, line_types=[c, c], forbidden=True))
    
    return net

JAPANESE_SOURCE_PREPROCESSING_CONTRACT = (
    "source_bound_shared_rigid_se2_raw_state_no_smoothing_no_clipping_v3"
)
JAPANESE_SOURCE_ROAD_CONTRACT = "source_derived_polyline_road_train_only_v2"
JAPANESE_LEGACY_SOURCE_ROAD_CONTRACTS = {
    "source_derived_polyline_road_train_only_v1",
}


def _load_japanese_geometry(
    geometry: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(geometry, Mapping):
        payload = dict(geometry)
    else:
        path = Path(geometry).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") not in {
        JAPANESE_SOURCE_ROAD_CONTRACT,
        *JAPANESE_LEGACY_SOURCE_ROAD_CONTRACTS,
    }:
        raise ValueError(
            "Japanese geometry manifest does not declare the source-derived "
            f"train-only contract: {payload.get('contract')!r}."
        )
    if payload.get("fit_split") != "train" or payload.get("test_rows_used") is not False:
        raise ValueError("Japanese road geometry must be fitted on train rows only.")
    lane_centerlines = payload.get("lane_centerlines")
    if not isinstance(lane_centerlines, dict) or set(lane_centerlines) != {"1", "2", "3"}:
        raise ValueError("Japanese road geometry requires lane centerlines 1, 2, and 3.")
    for lane_id, points in lane_centerlines.items():
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            raise ValueError(f"Japanese lane {lane_id} has invalid polyline points.")
        if not np.all(np.isfinite(values)) or np.any(np.diff(values[:, 0]) <= 0.0):
            raise ValueError(f"Japanese lane {lane_id} points must be finite and x-monotonic.")
    return payload


def _polyline_slice(points: list[list[float]], start_x: float, end_x: float) -> list[tuple[float, float]]:
    values = np.asarray(points, dtype=float)
    x = values[:, 0]
    if not float(start_x) < float(end_x):
        raise ValueError("Japanese road segment bounds must increase.")
    interior = values[(x > float(start_x)) & (x < float(end_x))]
    start = np.asarray(
        [float(start_x), np.interp(float(start_x), x, values[:, 1])],
        dtype=float,
    )
    end = np.asarray(
        [float(end_x), np.interp(float(end_x), x, values[:, 1])],
        dtype=float,
    )
    segment = np.vstack((start, interior, end))
    return [tuple(map(float, row)) for row in segment]


def _create_source_derived_japanese_road(payload: Mapping[str, Any]) -> RoadNetwork:
    net = RoadNetwork()
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
    width = float(payload["lane_width_m"])
    merge_start = float(payload["merge_start_x_m"])
    merge_end = float(payload["merge_end_x_m"])
    x_start = float(payload["x_start_m"])
    x_end = float(payload["x_end_m"])
    if not (0.0 < width < 10.0 and x_start < merge_start < merge_end < x_end):
        raise ValueError("Japanese source-derived road bounds or width are invalid.")

    lane_points = dict(payload["lane_centerlines"])
    main_specs = (
        ("2", [c, s]),
        ("1", [n, c]),
    )
    for start_node, end_node, lower, upper in (
        ("a", "b", x_start, merge_start),
        ("b", "c", merge_start, merge_end),
        ("c", "d", merge_end, x_end),
    ):
        for lane_id, line_types in main_specs:
            net.add_lane(
                start_node,
                end_node,
                PolyLaneFixedWidth(
                    _polyline_slice(lane_points[lane_id], lower, upper),
                    width=width,
                    line_types=line_types,
                ),
            )

    merge_points = np.asarray(lane_points["3"], dtype=float)
    merge_x_start = float(merge_points[0, 0])
    if merge_x_start >= merge_start:
        merge_x_start = float(np.nextafter(merge_start, -np.inf))
    net.add_lane(
        "j",
        "b",
        PolyLaneFixedWidth(
            _polyline_slice(lane_points["3"], merge_x_start, merge_start),
            width=width,
            line_types=[c, c],
            forbidden=True,
        ),
    )
    net.add_lane(
        "b",
        "c",
        PolyLaneFixedWidth(
            _polyline_slice(lane_points["3"], merge_start, merge_end),
            width=width,
            line_types=[n, c],
            forbidden=True,
        ),
    )
    net.japanese_geometry_contract = str(payload["contract"])
    net.japanese_merge_start_x_m = merge_start
    net.japanese_merge_end_x_m = merge_end
    net.japanese_x_start_m = x_start
    net.japanese_x_end_m = x_end
    return net


def create_japanese_road(
    geometry: str | Path | Mapping[str, Any] | None = None,
) -> RoadNetwork:
    """
    Japanese road layout matched to the processed dataset convention:

    - lane_id 2: right/main lane
    - lane_id 1: left/main lane
    - lane_id 3: left-side merge lane

    The dataset lateral positions are approximately centered around:
      lane 2 -> y ~= -1.9
      lane 1 -> y ~= +1.9
      lane 3 -> y ~= +5.6
    """
    if geometry is not None:
        return _create_source_derived_japanese_road(_load_japanese_geometry(geometry))

    net = RoadNetwork()

    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
    width = 3.75
    x_merge_start = 150.0
    x_merge_end = 315.0
    x_end = 800.0

    y_right = -0.5 * width
    y_left = 0.5 * width
    y_merge = 1.5 * width

    # Two-lane mainline before the merge segment.
    net.add_lane(
        "a",
        "b",
        StraightLane([0.0, y_right], [x_merge_start, y_right], width=width, line_types=[c, s]),
    )
    net.add_lane(
        "a",
        "b",
        StraightLane([0.0, y_left], [x_merge_start, y_left], width=width, line_types=[n, c]),
    )

    # Three-lane section while the left merge lane exists.
    net.add_lane(
        "b",
        "c",
        StraightLane([x_merge_start, y_right], [x_merge_end, y_right], width=width, line_types=[c, s]),
    )
    net.add_lane(
        "b",
        "c",
        StraightLane([x_merge_start, y_left], [x_merge_end, y_left], width=width, line_types=[n, s]),
    )
    net.add_lane(
        "b",
        "c",
        SineLane(
            [x_merge_start, 0.8 * (y_merge + y_left)],
            [x_merge_end, 0.8 * (y_merge + y_left)],
            amplitude=0.8 * (y_merge - y_left),
            pulsation=np.pi / (x_merge_end - x_merge_start),
            phase=np.pi / 2.0,
            width=width,
            line_types=[n, c],
            forbidden=True,
        ),
    )

    # Two-lane mainline after the merge.
    net.add_lane(
        "c",
        "d",
        StraightLane([x_merge_end, y_right], [x_end, y_right], width=width, line_types=[c, s]),
    )
    net.add_lane(
        "c",
        "d",
        StraightLane([x_merge_end, y_left], [x_end, y_left], width=width, line_types=[n, c]),
    )

    # Left-side merge approach feeding the temporary merge lane.
    net.add_lane(
        "j",
        "b",
        StraightLane([100.0, y_merge+2.4], [x_merge_start, y_merge+2.4], width=width+2.5, line_types=[c, c], forbidden=True),
    )

    net.japanese_geometry_contract = "legacy_hand_built_straight_road_v1"
    net.japanese_merge_start_x_m = x_merge_start
    net.japanese_merge_end_x_m = x_merge_end
    net.japanese_x_start_m = 0.0
    net.japanese_x_end_m = x_end

    return net


def clamp_location_ngsim(x_pos, lane0, net, warning=False):
    """
    Docstring for clamp_location_ngsim
    
    :param x_pos: position of the ego vehicle
    :param lane0: position of the ego vehicle with respect to lane
    :param net: the general RoadNetwork() class from highway env
    :param warning: show warning
    """
    width = US101_LANE_WIDTH_M
    ends = US101_SECTION_ENDS_M

    x_m = float(x_pos)
    if x_m < ends[1]:
        main_edge = ("s1", "s2")
    elif x_m < ends[2]:
        main_edge = ("s2", "s3")
    else:
        main_edge = ("s3", "s4")

    lane_index = int(lane0)
    lanes_on_edge = net.graph[main_edge[0]][main_edge[1]]
    n_lanes = len(lanes_on_edge)

    if lane_index < 0 or lane_index >= n_lanes:
        if warning:
            print(
                f"[NGSimEnv] WARNING: lane_index {lane_index} out of range for "
                f"edge {main_edge} (n_lanes={n_lanes}); clamping."
            )
        lane_index = int(np.clip(lane_index, 0, n_lanes - 1))

    lane_idx_tuple = (main_edge[0], main_edge[1], lane_index)
    ego_lane = net.get_lane(lane_idx_tuple)
    return lane_idx_tuple, ego_lane
