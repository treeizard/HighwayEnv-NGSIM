from highway_env.road.lane import LineType, StraightLane
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