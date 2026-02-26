import numpy as np
from highway_env.ngsim_utils.trajectory_gen import process_raw_trajectory
from highway_env.ngsim_utils.trajectory_to_action import traj_to_expert_actions

# -------------------------------------------------------------------------
# NGSIM SECTION HELPERS (consistent with create_ngsim_101_road)
# -------------------------------------------------------------------------
def main_edge_from_x(x: float) -> tuple[str, str]:
        length = 2150 / 3.281
        ends = [0.0, 560 / 3.281, (698 + 578 + 150) / 3.281, length]
        x_m = float(x)
        if x_m < ends[1]:
            return ("s1", "s2")  # 5 lanes
        elif x_m < ends[2]:
            return ("s2", "s3")  # 6 lanes
        else:
            return ("s3", "s4")  # 5 lanes

def clamp_lane_id_for_x(net, 
                        x: float, lane_id: int) -> int:
    edge = main_edge_from_x(x)
    n_lanes = len(net.graph[edge[0]][edge[1]])
    return int(np.clip(int(lane_id), 0, n_lanes - 1))


# -------------------------------------------------------------------------
# EXPERT HELPERS (consistent with create_ngsim_101_road)
# -------------------------------------------------------------------------
def load_ego_trajectory(ego_rec):
    """Process and return the ego trajectory data."""
    ego_traj_full = process_raw_trajectory(ego_rec["trajectory"])  # [T,4]: x, y, v, lane_id
    return ego_traj_full

def get_ego_dimensions(ego_rec, f2m_conv):
    """Return the ego vehicle's length and width in meters."""
    ego_len = ego_rec["length"] / f2m_conv
    ego_wid = ego_rec["width"] / f2m_conv
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
