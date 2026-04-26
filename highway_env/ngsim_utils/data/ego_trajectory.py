from __future__ import annotations

import numpy as np

from highway_env.ngsim_utils.data.trajectory_gen import process_raw_trajectory
from highway_env.ngsim_utils.expert.trajectory_to_action import traj_to_expert_actions


def load_ego_trajectory(ego_rec, scene: str = "us-101"):
    """Process and return the ego trajectory data."""
    return process_raw_trajectory(ego_rec["trajectory"], scene)


def get_ego_dimensions(ego_rec, feet_per_meter: float, scene: str) -> tuple[float, float]:
    """Return the ego vehicle length and width in meters."""
    if scene == "us-101":
        return ego_rec["length"] / feet_per_meter, ego_rec["width"] / feet_per_meter
    return ego_rec["length"], ego_rec["width"]


def setup_expert_tracker(_net, ego_traj_full: np.ndarray, ego_len: float, config: dict):
    """
    Build the expert reference arrays used by the pure-pursuit tracker.

    Lane ids remain in dataset coordinates here; mapping to highway-env lane
    indices is deferred to the components that need road topology.
    """
    expert = traj_to_expert_actions(
        ego_traj_full,
        dt=1.0 / config["policy_frequency"],
        L_forward=ego_len,
    )
    start_idx = int(expert["start_idx"])
    end_idx = int(expert["end_idx"])

    if end_idx <= start_idx:
        raise RuntimeError(f"Invalid expert start/end idx: {start_idx}, {end_idx}")

    ref_slice = ego_traj_full[start_idx : end_idx + 1]
    ref_xy_pol = ref_slice[:, :2]
    ref_v_pol = ref_slice[:, 2]
    lane_pol = ref_slice[:, 3].astype(int)
    return ref_xy_pol, ref_v_pol, lane_pol, start_idx
