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
from highway_env.ngsim_utils.core.constants import FEET_PER_METER, US101_SECTION_ENDS_M
from highway_env.ngsim_utils.data.trajectory_gen import trajectory_row_is_active

# -------------------------------------------------------------------------
# ROAD / LANE HELPERS
# -------------------------------------------------------------------------
def us101_edge_from_x(x: float) -> tuple[str, str]:
    ends = US101_SECTION_ENDS_M
    x_m = float(x)
    if x_m < ends[1]:
        return ("s1", "s2")
    if x_m < ends[2]:
        return ("s2", "s3")
    return ("s3", "s4")


def i80_edge_from_x(x: float) -> tuple[str, str]:
    x_m = float(x)
    if x_m <= 600 / FEET_PER_METER:
        return ("s1", "s2")
    if x_m <= 700 / FEET_PER_METER:
        return ("s2", "s3")
    if x_m <= 900 / FEET_PER_METER:
        return ("s3", "s4")
    return ("s4", "s5")

def edge_from_x(net, x: float) -> tuple[str, str]:
    """
    Infer the active mainline edge for a longitudinal position x from the road graph.

    This works for both the US-101 graph (s1->s2->s3->s4) and the Japanese
    graph (a->b->c->d), and avoids assuming specific node names.
    """
    x_m = float(x)
    candidates = []

    for src, dsts in net.graph.items():
        for dst, lanes in dsts.items():
            if not lanes:
                continue

            lane0 = lanes[0]
            start_x = float(min(lane0.start[0], lane0.end[0]))
            end_x = float(max(lane0.start[0], lane0.end[0]))

            # Prefer edges with multiple lanes, which represent the main carriageway.
            score = len(lanes)
            if start_x <= x_m <= end_x:
                return (src, dst)

            dist = min(abs(x_m - start_x), abs(x_m - end_x))
            candidates.append((dist, -score, start_x, src, dst))

    if not candidates:
        raise KeyError("Road network graph does not contain any lane edges.")

    _, _, _, src, dst = min(candidates)
    return (src, dst)

def clamp_lane_id_for_x(net,
                        x: float, lane_id: int) -> int:
    edge = edge_from_x(net, x)
    n_lanes = len(net.graph[edge[0]][edge[1]])
    return int(np.clip(int(lane_id), 0, n_lanes - 1))


def _last_lane_id(net, edge: tuple[str, str]) -> int:
    return len(net.graph[edge[0]][edge[1]]) - 1


def target_lane_index_from_lane_id(
    net,
    scene: str,
    x: float,
    lane_id: int,
) -> tuple[str, str, int] | None:
    """
    Map a recorded dataset lane id to a highway-env LaneIndex on the active road graph.

    Returns None when the lane id does not map to a drivable lane in the current scene.
    """
    lane_id = int(lane_id)
    x = float(x)
    scene = str(scene)

    if scene == "us-101":
        if lane_id <= 5:
            edge = us101_edge_from_x(x)
            return (edge[0], edge[1], lane_id - 1)
        if lane_id == 6:
            edge = ("s2", "s3")
            return (edge[0], edge[1], _last_lane_id(net, edge))
        if lane_id == 7:
            edge = ("merge_in", "s2")
            return (edge[0], edge[1], _last_lane_id(net, edge))
        if lane_id == 8:
            edge = ("s3", "merge_out")
            return (edge[0], edge[1], _last_lane_id(net, edge))
        return None

    if scene == "i-80":
        if lane_id <= 6:
            edge = i80_edge_from_x(x)
            return (edge[0], edge[1], lane_id - 1)
        if lane_id == 7:
            edge = ("s1", "s2")
            return (edge[0], edge[1], _last_lane_id(net, edge))
        return None

    if scene == "japanese":
        # Dataset convention:
        #   lane_id 2 -> right mainline lane
        #   lane_id 1 -> left mainline lane
        #   lane_id 3 -> left merge lane
        if lane_id == 2:
            if x < 150.0:
                return ("a", "b", 0)
            if x < 260.0:
                return ("b", "c", 0)
            return ("c", "d", 0)
        if lane_id == 1:
            if x < 150.0:
                return ("a", "b", 1)
            if x < 260.0:
                return ("b", "c", 1)
            return ("c", "d", 1)
        if lane_id == 3:
            if x < 150.0:
                return ("j", "b", 0)
            if x < 260.0:
                return ("b", "c", 2)
            # After the merge lane disappears, fold back onto the left mainline lane.
            return ("c", "d", 1)
        return None

    return None


def heading_from_trajectory_row(
    net,
    scene: str,
    row: np.ndarray,
    *,
    next_row: np.ndarray | None = None,
    fallback_heading: float = 0.0,
) -> float:
    """Infer heading from a trajectory row using lane geometry, then motion delta."""
    row_arr = np.asarray(row, dtype=float)
    x, y, _speed, lane_id = row_arr[:4]
    mapped_lane_index = target_lane_index_from_lane_id(
        net,
        scene,
        float(x),
        int(lane_id),
    )
    if mapped_lane_index is not None:
        lane = net.get_lane(mapped_lane_index)
        local_s, _local_r = lane.local_coordinates(np.array([x, y], dtype=float))
        return float(lane.heading_at(local_s))
    if next_row is not None and trajectory_row_is_active(next_row):
        next_arr = np.asarray(next_row, dtype=float)
        dx = float(next_arr[0] - x)
        dy = float(next_arr[1] - y)
        if np.hypot(dx, dy) > 1e-3:
            return float(np.arctan2(dy, dx))
    return float(fallback_heading)
