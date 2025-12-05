
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import RoadNetwork


def create_japanese_road():
    net = RoadNetwork()
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

    length = 2150 / 3.281  # m
    width = 3.25    # m
    ends = [0 , 252.23, ]

    line_types = [[c,n],[s,n],[s,n],[s,n],[s,c]]

    for lane in range(4):
        origin = [ends[0], lane*width]
        end = [ends[1], lane*width]
        net.add_lane("s1","s2",StraightLane(origin, end, width=width, line_types=line_types[lane]))
    
    net.addlane('merg_in',"s2", StraightLane([]))