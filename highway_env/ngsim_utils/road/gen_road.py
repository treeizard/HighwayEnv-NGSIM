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


import numpy as np
from highway_env.road.lane import LineType, StraightLane, SineLane
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

def create_japanese_road() -> None:
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


'''
def clamp_location_ngsim(x_pos, lane0, net, warning = False):
    """
    Docstring for clamp_location_ngsim
    
    :param x_pos: position of the ego vehicle
    :param lane0: position of the ego vehicle with respect to lane
    :param net: the general RoadNetwork() class from highway env
    :param warning: show warning
    """
    width = US101_LANE_WIDTH_M
    ends = US101_SECTION_ENDS_M

    x_m =  float(x_pos)
    if x_m < ends[1]:
        main_edge = ("s1", "s2")   # first 5-lane section
    elif x_m < ends[2]:
        main_edge = ("s2", "s3")   # 6-lane section (lane 5 lives here)
    else:
        main_edge = ("s3", "s4")   # last 5-lane section
    
    lane_index = int(lane0)
    lanes_on_edge = net.graph[main_edge[0]][main_edge[1]]
    n_lanes = len(lanes_on_edge)
    if lane_index < 0 or lane_index >= n_lanes:
        if warning == True:
            print(
                f"[NGSimEnv] WARNING: lane_index {lane_index} out of range for "
                f"edge {main_edge} (n_lanes={n_lanes}); clamping."
            )
        
        lane_index = int(np.clip(lane_index, 0, n_lanes - 1))
    
    ego_lane = net.get_lane((*main_edge, lane_index))
    return ego_lane
'''
