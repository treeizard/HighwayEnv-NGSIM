import numpy as np
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import RoadNetwork

def create_ngsim_101_road():
    net = RoadNetwork()
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

    length = 2150 / 3.281  # m
    width = 12 / 3.281     # m
    ends = [0, 560/3.281, (698+578+150)/3.281, length]

    # first section (5 lanes)
    line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
    for lane in range(5):
        origin = [ends[0], lane * width]
        end = [ends[1], lane * width]
        net.add_lane("s1", "s2", StraightLane(origin, end, width=width, line_types=line_types[lane]))

    # merge_in (forbidden)
    net.add_lane("merge_in", "s2", StraightLane([480/3.281, 5.5*width], [ends[1], 5*width], width=width, line_types=[c, c], forbidden=True))

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
    net.add_lane("s3", "merge_out", StraightLane([ends[2], 5*width], [1550/3.281, 7*width], width=width, line_types=[c, c], forbidden=True))
    
    return net

def create_japanese_road() -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [100, 50, 50, 600]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        width = 3.75
        y = [0, width]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[s, s], [n, c]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        # Example lane width
        lane_width = width  # typically 3.75

        # Main-road target lane center
        target_y = -lane_width

        # Start the merge lane closer to the main road
        merge_start_y = -6.0   # adjust this value as needed

        # 1) Straight ramp approach
        ljk = StraightLane(
            [0, merge_start_y],
            [ends[0], merge_start_y],
            line_types=[c, c],
            forbidden=True
        )

        # 2) Smooth merge toward the target lane
        # Reference line is the midpoint between start_y and target_y
        mid_y = 0.5 * (merge_start_y + target_y)
        amplitude = 0.5 * (merge_start_y - target_y)

        lkb = SineLane(
            [ends[0], mid_y],                 # start of sine reference line
            [sum(ends[:2]), mid_y],           # end of sine reference line
            amplitude,                        # chosen so actual path goes start_y -> target_y
            2 * np.pi / (2 * ends[1]),        # = pi / ends[1]
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )

        # 3) Straight lane after merge, aligned with the main-road lane
        lbc = StraightLane(
            [sum(ends[:2]), target_y],
            [sum(ends[:3]), target_y],
            line_types=[c, n],
            forbidden=True,
        )

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        return net


def clamp_location_ngsim(x_pos, lane0, net, warning=False):
    """
    Docstring for clamp_location_ngsim
    
    :param x_pos: position of the ego vehicle
    :param lane0: position of the ego vehicle with respect to lane
    :param net: the general RoadNetwork() class from highway env
    :param warning: show warning
    """
    length = 2150 / 3.281  # m
    width = 12 / 3.281     # m
    ends = [0, 560/3.281, (698+578+150)/3.281, length] # m

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
    length = 2150 / 3.281  # m
    width = 12 / 3.281     # m
    ends = [0, 560/3.281, (698+578+150)/3.281, length] # m

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