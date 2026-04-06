import numpy as np
from highway_env.ngsim_utils.constants import FEET_PER_METER, US101_SECTION_ENDS_M
from highway_env.ngsim_utils.trajectory_gen import process_raw_trajectory
from highway_env.ngsim_utils.trajectory_to_action import traj_to_expert_actions

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
            return ("s2", "s3", -1)
        if lane_id == 7:
            return ("merge_in", "s2", -1)
        if lane_id == 8:
            return ("s3", "merge_out", -1)
        return None

    if scene == "i-80":
        if lane_id <= 6:
            edge = i80_edge_from_x(x)
            return (edge[0], edge[1], lane_id - 1)
        if lane_id == 7:
            return ("s1", "s2", -1)
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


# -------------------------------------------------------------------------
# EXPERT HELPERS (consistent with create_ngsim_101_road)
# -------------------------------------------------------------------------
def load_ego_trajectory(ego_rec, scene = 'us-101'):
    """Process and return the ego trajectory data."""
    ego_traj_full = process_raw_trajectory(ego_rec["trajectory"], scene)  # [T,4]: x, y, v, lane_id
    return ego_traj_full

def get_ego_dimensions(ego_rec, f2m_conv, scene):
    """Return the ego vehicle's length and width in meters."""
    if scene == "us-101":
        ego_len = ego_rec["length"] / f2m_conv
        ego_wid = ego_rec["width"] / f2m_conv
    else:
        ego_len = ego_rec["length"]
        ego_wid = ego_rec["width"]
        
    return ego_len, ego_wid

# highway_env/ngsim_utils/helper_ngsim.py

def setup_expert_tracker(net, ego_traj_full, ego_len, config):
    """
    Setup the expert trajectory tracker. 
    MODIFIED: We do NOT clamp lanes here anymore. We pass raw lanes to the tracker.
    """
    expert = traj_to_expert_actions(ego_traj_full, dt=1.0 / config["policy_frequency"], L_forward=ego_len)
    start_idx = int(expert["start_idx"])
    end_idx = int(expert["end_idx"])

    if end_idx <= start_idx:
        raise RuntimeError(f"Invalid expert start/end idx: {start_idx}, {end_idx}")

    ref_slice = ego_traj_full[start_idx: end_idx + 1]
    ref_xy_pol = ref_slice[:, :2]
    ref_v_pol = ref_slice[:, 2]
    
    # Just grab the raw lane IDs. Do not clamp yet.
    lane_pol = ref_slice[:, 3].astype(int)

    # Return raw lane_pol
    return ref_xy_pol, ref_v_pol, lane_pol, start_idx
