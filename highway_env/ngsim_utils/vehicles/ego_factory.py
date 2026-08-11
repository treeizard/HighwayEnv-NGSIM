"""Replay and control NGSIM vehicles inside highway-env scenes."""

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


from __future__ import annotations

import numpy as np

from highway_env.ngsim_utils.road.lane_mapping import (
    heading_from_trajectory_row,
    target_lane_index_from_lane_id,
)
from highway_env.ngsim_utils.vehicles.ego import EgoVehicle


def estimate_initial_heading(ego_traj: np.ndarray) -> float:
    dx0 = ego_traj[1, 0] - ego_traj[0, 0]
    dy0 = ego_traj[1, 1] - ego_traj[0, 1]
    disp = np.hypot(dx0, dy0)
    return float(np.arctan2(dy0, dx0)) if disp >= 0.1 else 0.0


def target_speeds_for_trajectory(
    ego_traj: np.ndarray,
    *,
    control_mode: str,
    action_cfg: dict,
) -> np.ndarray | None:
    target_speeds_cfg = action_cfg.get("target_speeds", None)
    if control_mode != "discrete":
        return np.array(target_speeds_cfg, dtype=float) if target_speeds_cfg is not None else None

    if target_speeds_cfg is None:
        base = np.array(EgoVehicle.DEFAULT_TARGET_SPEEDS, dtype=float)
    else:
        base = np.array(target_speeds_cfg, dtype=float)

    if base.ndim != 1 or base.size < 2 or not np.all(np.isfinite(base)):
        return np.array(EgoVehicle.DEFAULT_TARGET_SPEEDS, dtype=float)

    diffs = np.diff(base)
    positive_diffs = diffs[diffs > 1e-6]
    step = float(np.median(positive_diffs)) if positive_diffs.size else 2.0
    step = max(0.5, step)

    valid_speeds = ego_traj[:, 2]
    valid_speeds = valid_speeds[np.isfinite(valid_speeds) & (valid_speeds >= 0.0)]
    if valid_speeds.size == 0:
        return base

    max_speed = max(float(base[-1]), float(np.max(valid_speeds)) + step)
    count = int(np.ceil(max_speed / step)) + 1
    return np.arange(count, dtype=float) * step


def build_ego_vehicle(
    *,
    road,
    scene: str,
    ego_traj: np.ndarray,
    ego_len: float,
    ego_wid: float,
    control_mode: str,
    action_cfg: dict,
) -> EgoVehicle:
    x0, y0, ego_speed, lane0 = ego_traj[0]
    ego_xy = np.array([x0, y0], dtype=float)
    heading_raw = heading_from_trajectory_row(
        road.network,
        scene,
        np.asarray(ego_traj[0], dtype=float),
        fallback_heading=0.0,
        prefer_motion=False,
    )
    target_speeds = target_speeds_for_trajectory(
        ego_traj,
        control_mode=control_mode,
        action_cfg=action_cfg,
    )

    ego = EgoVehicle(
        road=road,
        position=ego_xy,
        speed=ego_speed,
        heading=heading_raw,
        control_mode=control_mode,
        target_speeds=target_speeds,
        lateral_offset_step=float(
            action_cfg.get(
                "lateral_offset_step",
                EgoVehicle.DEFAULT_LATERAL_OFFSET_STEP,
            )
        ),
        lateral_offset_max=float(
            action_cfg.get(
                "lateral_offset_max",
                EgoVehicle.DEFAULT_LATERAL_OFFSET_MAX,
            )
        ),
    )
    ego.set_ego_dimension(width=ego_wid, length=ego_len)

    mapped_lane_index = target_lane_index_from_lane_id(road.network, scene, x0, int(lane0))
    if mapped_lane_index is not None:
        ego.target_lane_index = mapped_lane_index
        ego.lane_index = mapped_lane_index
        ego.lane = road.network.get_lane(mapped_lane_index)
        # Preserve the recorded pose.  Moving a source vehicle onto the lane
        # would make its observation disagree with its action label; corpus
        # qualification separately rejects poses that do not overlap their
        # mapped lane.
        # Keep reliable filtered motion heading at the controlled-policy
        # boundary. ``heading_from_trajectory_row`` already uses this lane
        # tangent as the low-speed/unreliable-motion fallback, matching the
        # direct yaw-rate label contract used during expert collection.

    return ego
