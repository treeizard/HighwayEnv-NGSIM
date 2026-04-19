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
from scipy import signal
import matplotlib.pyplot as plt
from highway_env.data.ngsim import *
from typing import Any, Dict
from highway_env.ngsim_utils.constants import FEET_PER_METER
"""
def trajectory_smoothing(trajectory):
    trajectory = np.array(trajectory)
    x = trajectory[:,0]
    y = trajectory[:,1]
    speed = trajectory[:,2]
    lane = trajectory[:,3]

    window_length = 21 if len(x[np.nonzero(x)]) >= 21 else len(x[np.nonzero(x)]) if len(x[np.nonzero(x)]) % 2 !=0 else len(x[np.nonzero(x)])-1
    x[np.nonzero(x)] = signal.savgol_filter(x[np.nonzero(x)], window_length=window_length, polyorder=3) # window size used for filtering, order of fitted polynomial
    y[np.nonzero(y)] = signal.savgol_filter(y[np.nonzero(y)], window_length=window_length, polyorder=3)
    speed[np.nonzero(speed)] = signal.savgol_filter(speed[np.nonzero(speed)], window_length=window_length, polyorder=3)
   
    return [[float(x), float(y), float(s), int(l)] for x, y, s, l in zip(x, y, speed, lane)]
"""
def trajectory_smoothing(trajectory):
    """
    trajectory: array-like of shape (T, 4) with columns [x, y, speed, lane]
    Returns: list of [x, y, speed, lane] with smoothed x, y, speed
    """
    trajectory = np.array(trajectory)
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    speed = trajectory[:, 2]
    lane = trajectory[:, 3]  # lane kept as-is

    def _smooth_1d(arr):
        idx = np.nonzero(arr)[0]
        if len(idx) == 0:
            return arr
        n = len(idx)
        # Start with window capped at 21 but not larger than n
        window_length = 21 if n >= 21 else n
        # Make it odd
        if window_length % 2 == 0:
            window_length -= 1
        # Need window_length > polyorder
        polyorder = 3
        if window_length <= polyorder:
            # Not enough points to fit a cubic; skip smoothing
            return arr
        arr[idx] = signal.savgol_filter(
            arr[idx], window_length=window_length, polyorder=polyorder
        )
        return arr

    x = _smooth_1d(x)
    y = _smooth_1d(y)
    speed = _smooth_1d(speed)

    return [
        [float(xx), float(yy), float(s), int(l)]
        for xx, yy, s, l in zip(x, y, speed, lane)
    ]


def build_trajectory(scene, period, vehicle_ID):
    ng = ngsim_data(scene)
    ng.load('highway_env/data/processed/'+scene)
    records = ng.vr_dict
    vehicles = ng.veh_dict
    snapshots = ng.snap_dict
    surroundings = []
    record_trajectory = {'ego':{'length':0, 'width':0, 'trajectory':[]}}
    
    for veh_ID, v in vehicles.items():
        v.build_trajectory()

    ego_trajectories = vehicles[vehicle_ID].trajectory
    selected_trajectory = ego_trajectories[period]

    D = 200 if scene == 'us-101' else 20

    ego = []
    nearby_IDs = []
    for position in selected_trajectory:
        record_trajectory['ego']['length'] = position.len
        record_trajectory['ego']['width'] = position.wid
        ego.append([position.x, position.y, position.spd, position.lane_ID])
        records = snapshots[position.unixtime].vr_list
        other = []
        for record in records:
            if record.veh_ID != vehicle_ID:
                other.append([record.veh_ID, record.len, record.wid, record.x, record.y, record.spd, record.lane_ID])
                d = abs(position.y - record.y)
                if d <= D:            
                    nearby_IDs.append(record.veh_ID)
        surroundings.append(other)
        
    record_trajectory['ego']['trajectory'] = ego

    for v_ID in set(nearby_IDs):
        record_trajectory[v_ID] = {'length':0, 'width':0, 'trajectory':[]}
    
    # fill in data
    for timestep_record in surroundings:
        scene_IDs = []
        for vehicle_record in timestep_record:
            v_ID = vehicle_record[0]
            v_length = vehicle_record[1]
            v_width = vehicle_record[2]
            v_x = vehicle_record[3]
            v_y = vehicle_record[4]
            v_s = vehicle_record[5]
            v_laneID = vehicle_record[6]
            if v_ID in set(nearby_IDs):
                scene_IDs.append(v_ID)
                record_trajectory[v_ID]['length'] = v_length
                record_trajectory[v_ID]['width'] = v_width
                record_trajectory[v_ID]['trajectory'].append([v_x, v_y, v_s, v_laneID])
        for v_ID in set(nearby_IDs):
            if v_ID not in scene_IDs:
                record_trajectory[v_ID]['trajectory'].append([0, 0, 0, 0])
    
    # trajectory smoothing
    for key in record_trajectory.keys():
        orginal_trajectory = record_trajectory[key]['trajectory']
        smoothed_trajectory = trajectory_smoothing(orginal_trajectory)
        record_trajectory[key]['trajectory'] = smoothed_trajectory

    return record_trajectory


def build_trajectory_from_chunk(scene, vehicle_ID, episode_dir):
    """
    Load only one processed episode folder instead of the full us-101 dataset.

    episode_dir example:
        "highway_env/data/processed_20s/us-101/t1118846663000"
    """
    ng = ngsim_data(scene)
    ng.load(episode_dir)          # <-- loads only that small episode
    records    = ng.vr_dict
    vehicles   = ng.veh_dict
    snapshots  = ng.snap_dict
    surroundings = []
    record_trajectory = {'ego': {'length': 0, 'width': 0, 'trajectory': []}}

    # Only build ego trajectory, NOT all vehicles
    ego_vehicle = vehicles[vehicle_ID]
    ego_vehicle.build_trajectory()
    ego_trajectories = ego_vehicle.trajectory
    #print('trajectory:',ego_trajectories)
    # With a fixed-length chunk you'll almost always have a single period = 0
    selected_trajectory = ego_trajectories[0]

    D = 200 if scene == 'us-101' else 20

    ego = []
    nearby_IDs = []
    for position in selected_trajectory:
        record_trajectory['ego']['length'] = position.len
        record_trajectory['ego']['width']  = position.wid
        ego.append([position.x, position.y, position.spd, position.lane_ID])

        records = snapshots[position.unixtime].vr_list
        other = []
        for record in records:
            if record.veh_ID != vehicle_ID:
                other.append([record.veh_ID, record.len, record.wid,
                              record.x, record.y, record.spd, record.lane_ID])
                d = abs(position.y - record.y)
                if d <= D:
                    nearby_IDs.append(record.veh_ID)
        surroundings.append(other)

    record_trajectory['ego']['trajectory'] = ego

    # allocate neighbor containers
    for v_ID in set(nearby_IDs):
        record_trajectory[v_ID] = {'length': 0, 'width': 0, 'trajectory': []}

    # fill neighbors per time step
    for timestep_record in surroundings:
        scene_IDs = []
        for vehicle_record in timestep_record:
            v_ID, v_length, v_width, v_x, v_y, v_s, v_laneID = vehicle_record
            if v_ID in record_trajectory:
                scene_IDs.append(v_ID)
                record_trajectory[v_ID]['length'] = v_length
                record_trajectory[v_ID]['width']  = v_width
                record_trajectory[v_ID]['trajectory'].append(
                    [v_x, v_y, v_s, v_laneID]
                )
        for v_ID in record_trajectory.keys():
            if v_ID == 'ego':
                continue
            if v_ID not in scene_IDs:
                record_trajectory[v_ID]['trajectory'].append([0, 0, 0, 0])

    # smoothing
    for key in record_trajectory.keys():
        orginal_trajectory = record_trajectory[key]['trajectory']
        smoothed_trajectory = trajectory_smoothing(orginal_trajectory)
        record_trajectory[key]['trajectory'] = smoothed_trajectory

    return record_trajectory



def build_all_trajectories_for_scene(
    scene: str,
    episodes_root: str,
    train_val_div: str,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Preload and preprocess trajectories for all processed intervals (episodes)
    of a given scene.

    Args:
        scene:
            Name of the NGSIM scene, e.g. "us-101".
        episodes_root:
            Path to the directory that contains the processed episode folders
            for this scene. Typical layout:

                episodes_root/
                  us-101/
                    t1118846663000/
                    t1118846673000/
                    ...

            In that case you would pass episodes_root=".../processed_20s"
            and scene="us-101", and this function will look under
            episodes_root/scene for folders starting with "t".

    Returns:
        A nested dict:

        {
            "t1118846663000": {
                veh_ID_1: {
                    "length": float,
                    "width": float,
                    "trajectory": np.ndarray [T, 4]  # [x, y, speed, lane_ID]
                },
                veh_ID_2: { ... },
                ...
            },
            "t1118846673000": {
                ...
            },
            ...
        }

        All trajectories within a given episode are aligned on the same
        time grid (same length T). Missing times are padded with [0,0,0,0].
    """
    scene_root = os.path.join(episodes_root, scene, train_val_div)

    # Find all 10-second episode folders (same rule as _ensure_episode_list)
    episode_names = sorted(
        d for d in os.listdir(scene_root)
        if d.startswith("t") and os.path.isdir(os.path.join(scene_root, d))
    )
    if not episode_names:
        raise RuntimeError(f"No 10-second episodes found in {scene_root}")

    all_episodes: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for ep_name in episode_names:
        ep_dir = os.path.join(scene_root, ep_name)
        print(f"[build_all_trajectories_for_scene] Loading episode: {ep_dir}")

        ng = ngsim_data(scene)
        ng.load(ep_dir)

        snapshots = ng.snap_dict  # unixtime -> snapshot with vr_list
        time_keys = sorted(snapshots.keys())
        if not time_keys:
            print(f"[build_all_trajectories_for_scene] WARNING: empty episode {ep_dir}")
            all_episodes[ep_name] = {}
            continue

        # veh_ID -> {"length": float, "width": float, "trajectory": list[list[4]]}
        record_trajectory: Dict[int, Dict[str, Any]] = {}

        for step_idx, unixtime in enumerate(time_keys):
            snap = snapshots[unixtime]
            present_ids = set()

            # 1) Update/add vehicles present in this snapshot
            for vr in snap.vr_list:
                v_ID = vr.veh_ID
                present_ids.add(v_ID)

                if v_ID not in record_trajectory:
                    # New vehicle: pre-fill zeros for previous steps
                    record_trajectory[v_ID] = {
                        "length": vr.len,
                        "width": vr.wid,
                        "trajectory": [],
                    }
                    if step_idx > 0:
                        record_trajectory[v_ID]["trajectory"].extend(
                            [[0.0, 0.0, 0.0, 0] for _ in range(step_idx)]
                        )

                rec = record_trajectory[v_ID]
                rec["length"] = vr.len
                rec["width"] = vr.wid
                rec["trajectory"].append(
                    [vr.x, vr.y, vr.spd, vr.lane_ID]
                )

            # 2) For vehicles we've already seen but that are not
            #    present in this snapshot: append a zero row to keep
            #    trajectories aligned in time.
            for v_ID, rec in record_trajectory.items():
                if v_ID not in present_ids:
                    rec["trajectory"].append([0.0, 0.0, 0.0, 0])

        # 3) Smoothing per vehicle and convert to np.ndarray
        for v_ID, rec in record_trajectory.items():
            original_traj = rec["trajectory"]
            smoothed = trajectory_smoothing(original_traj)
            rec["trajectory"] = np.asarray(smoothed, dtype=float)

        all_episodes[ep_name] = record_trajectory

    return all_episodes



def process_raw_trajectory(trajectory, scene):
    if scene == "us-101":
        trajectory = np.array(trajectory)
        for i in range(trajectory.shape[0]):
            '''
            if np.all(trajectory[i] == 0):
                trajectory[i][0] = 0
                trajectory[i][1] = 0
                trajectory[i][2] = 0
            
            else:
            '''
            x = trajectory[i][0] - 6
            y = trajectory[i][1]
            speed = trajectory[i][2]
            trajectory[i][0] = y / FEET_PER_METER
            trajectory[i][1] = x / FEET_PER_METER
            trajectory[i][2] = speed / FEET_PER_METER
    elif scene == "japanese":
        trajectory = np.array(trajectory)
        for i in range(trajectory.shape[0]):
            '''
            if np.all(trajectory[i] == 0):
                trajectory[i][0] = 0
                trajectory[i][1] = 0
                trajectory[i][2] = 0
            
            else:
            '''
            x = trajectory[i][0]
            y = trajectory[i][1]
            speed = trajectory[i][2]

            trajectory[i][0] = x
            trajectory[i][1] = y
            # Japanese trajectory coordinates are already in meters, while speed is stored in km/h.
            # Convert only the speed channel so the processed trajectory is consistently in SI units.
            trajectory[i][2] = speed / 3.6
            #print(x, y, speed)
    else:
        raise UserWarning("The Scene is not Recognised Please Try Again")

    return trajectory

def trajectory_row_is_active(row):
    """
    Return whether a processed trajectory row represents an active vehicle.

    Some processed padding rows preserve a lateral sentinel value in y, so y
    alone is not enough to mark a vehicle as present.
    """
    x, y, spd, lane = np.asarray(row, dtype=float)[:4]
    invalid_sentinel = (
        np.isclose(x, 0.0)
        and np.isclose(y, -1.82871076)
        and np.isclose(spd, 0.0)
        and np.isclose(lane, 0.0)
    )
    inactive_padding = (
        np.isclose(x, 0.0)
        and np.isclose(spd, 0.0)
        and np.isclose(lane, 0.0)
    )
    return not (invalid_sentinel or inactive_padding)


def trajectory_rows_are_continuous(
    prev_row,
    next_row,
    *,
    data_dt: float = 0.1,
    speed_scale: float = 3.0,
    margin_m: float = 4.0,
    min_jump_threshold_m: float = 8.0,
) -> bool:
    """
    Return whether two active trajectory rows look physically continuous.

    A pair is considered discontinuous when the measured position jump is far
    larger than what the recorded speeds could plausibly cover in one sample.
    """
    prev_arr = np.asarray(prev_row, dtype=float)
    next_arr = np.asarray(next_row, dtype=float)
    if not trajectory_row_is_active(prev_arr) or not trajectory_row_is_active(next_arr):
        return False

    prev_xy = prev_arr[:2]
    next_xy = next_arr[:2]
    jump_m = float(np.linalg.norm(next_xy - prev_xy))
    prev_speed = max(float(prev_arr[2]), 0.0)
    next_speed = max(float(next_arr[2]), 0.0)
    speed_bound_m = max(prev_speed, next_speed) * float(data_dt)
    allowed_jump_m = max(float(min_jump_threshold_m), speed_scale * speed_bound_m + float(margin_m))
    return jump_m <= allowed_jump_m


def longest_continuous_active_span(
    traj,
    *,
    data_dt: float = 0.1,
    speed_scale: float = 3.0,
    margin_m: float = 4.0,
    min_jump_threshold_m: float = 8.0,
) -> int:
    """
    Return the longest active-and-continuous span length in a trajectory.

    A span breaks on either padded/inactive rows or on implausibly large jumps
    between consecutive active positions.
    """
    traj_arr = np.asarray(traj, dtype=float)
    if traj_arr.ndim != 2 or traj_arr.shape[0] == 0:
        return 0

    best = 0
    current = 0
    prev_active_row = None

    for row in traj_arr:
        if not trajectory_row_is_active(row):
            current = 0
            prev_active_row = None
            continue

        if prev_active_row is None:
            current = 1
        elif trajectory_rows_are_continuous(
            prev_active_row,
            row,
            data_dt=data_dt,
            speed_scale=speed_scale,
            margin_m=margin_m,
            min_jump_threshold_m=min_jump_threshold_m,
        ):
            current += 1
        else:
            current = 1

        best = max(best, current)
        prev_active_row = np.asarray(row, dtype=float)

    return int(best)


def longest_continuous_active_span_bounds(
    traj,
    *,
    data_dt: float = 0.1,
    speed_scale: float = 3.0,
    margin_m: float = 4.0,
    min_jump_threshold_m: float = 8.0,
) -> tuple[int | None, int | None, int]:
    """
    Return ``(start_idx, end_idx, length)`` for the longest continuous active span.
    """
    traj_arr = np.asarray(traj, dtype=float)
    if traj_arr.ndim != 2 or traj_arr.shape[0] == 0:
        return None, None, 0

    best_start = None
    best_end = None
    best_len = 0

    current_start = None
    current_len = 0
    prev_active_row = None

    for idx, row in enumerate(traj_arr):
        if not trajectory_row_is_active(row):
            current_start = None
            current_len = 0
            prev_active_row = None
            continue

        if prev_active_row is None:
            current_start = idx
            current_len = 1
        elif trajectory_rows_are_continuous(
            prev_active_row,
            row,
            data_dt=data_dt,
            speed_scale=speed_scale,
            margin_m=margin_m,
            min_jump_threshold_m=min_jump_threshold_m,
        ):
            current_len += 1
        else:
            current_start = idx
            current_len = 1

        if current_len > best_len:
            best_len = current_len
            best_start = current_start
            best_end = idx

        prev_active_row = np.asarray(row, dtype=float)

    return best_start, best_end, int(best_len)


def trajectory_has_min_continuous_occupancy(
    traj,
    *,
    min_presence_ratio: float = 0.8,
    data_dt: float = 0.1,
    speed_scale: float = 3.0,
    margin_m: float = 4.0,
    min_jump_threshold_m: float = 8.0,
) -> bool:
    """
    Return whether one continuous active span occupies enough of the scene.

    This is stricter than counting all active rows: a vehicle must stay present
    without gaps or teleporting jumps for at least ``min_presence_ratio`` of the
    full episode/window length.
    """
    traj_arr = np.asarray(traj, dtype=float)
    if traj_arr.ndim != 2 or traj_arr.shape[0] == 0:
        return False

    _start_idx, _end_idx, longest_span = longest_continuous_active_span_bounds(
        traj_arr,
        data_dt=data_dt,
        speed_scale=speed_scale,
        margin_m=margin_m,
        min_jump_threshold_m=min_jump_threshold_m,
    )
    return float(longest_span) / float(traj_arr.shape[0]) >= float(min_presence_ratio)


def first_valid_index(traj):
    """Return index of first active trajectory entry, or None if all padding."""
    for i, row in enumerate(np.asarray(traj, dtype=float)):
        if trajectory_row_is_active(row):
            return i
    return None


def common_first_valid_index(trajectories):
    """
    Return the first episode index at which every trajectory has valid data.

    The environment uses this shared reset index when spawning multiple
    controlled vehicles so all egos are present from step 0.
    """
    start_indices = []
    for traj in trajectories:
        start_idx = first_valid_index(traj)
        if start_idx is None:
            return None
        start_indices.append(int(start_idx))
    if not start_indices:
        return None
    return max(start_indices)

if __name__ == "__main__":
    scene  = "us-101"
    period = 0

    # Build once
    trajectory_set = build_trajectory(scene, period)

    # Choose a subset to keep the figure readable
    veh_ids = list(trajectory_set.keys())
    if len(veh_ids) == 0:
        raise RuntimeError("No trajectories found. Check your data path and period index.")
    sample_size = min(150, len(veh_ids))
    sel = np.random.choice(veh_ids, size=sample_size, replace=False)

    plt.figure(figsize=(10, 4))
    for vid in sel:
        traj_ft = trajectory_set[vid]["trajectory"]        # [x_ft, y_ft, spd, lane]
        traj_m  = process_raw_trajectory(traj_ft)          # [s_m, r_m, spd_mps, lane]
        arr = np.asarray(traj_m)
        s = arr[:, 0]   # longitudinal (meters)
        r = arr[:, 1]   # lateral (meters)
        plt.plot(s, r, linewidth=1, alpha=0.45)

    # Axes/labels
    plt.gca().set_aspect('auto', 'datalim')
    plt.xlabel("Longitudinal position [m]", fontsize=14)
    plt.ylabel("Lateral position [m]", fontsize=14)

    # Optional: crop to a typical US-101 clip span
    plt.xlim(0, 660)
    plt.ylim(0, 25)

    # NOTE: If your lane index increases downward in your rendering,
    # you can flip lateral with: plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
