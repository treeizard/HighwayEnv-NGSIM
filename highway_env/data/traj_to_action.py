# Written by: Huy Nguyen （hngu0143@student.monash.edu）
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

import pandas as pd
from scipy import signal
import numpy as np
import argparse
import os

# Definition: https://data.transportation.gov/api/views/8ect-6jqj/files/ddb2c29d-2ef4-4b67-94ea-b55169229bd9?download=true&filename=1-%20US%20101%20Metadata%20Documentation.pdf
colidxmap = {
    "id": 0,
    "veh_ID": 1,
    "time": 2,
    "x": 3,
    "y": 4,
    "lat": 5,
    "lon": 6,
    "length": 7,
    "width": 8,
    "class": 9,
    "speed": 10,
    "accel": 11,
    "lane_id": 12,
    "pred_veh_id": 13,
    "follow_veh_id": 14,
    "shead": 15,
    "thead": 16,
}

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angle to [-pi, pi].
    Works elementwise on numpy arrays.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


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

    # Only smooth non-zero entries to avoid artifacts at padding / missing data
    nonzero_x = np.nonzero(x)[0]
    if len(nonzero_x) > 0:
        n = len(nonzero_x)
        window_length = 21 if n >= 21 else n
        # Savitzky-Golay requires odd window length >= polyorder+2
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 5:  # sanity check so polyorder=3 is valid
            x[nonzero_x] = signal.savgol_filter(
                x[nonzero_x], window_length=window_length, polyorder=3
            )

    nonzero_y = np.nonzero(y)[0]
    if len(nonzero_y) > 0:
        n = len(nonzero_y)
        window_length = 21 if n >= 21 else n
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 5:
            y[nonzero_y] = signal.savgol_filter(
                y[nonzero_y], window_length=window_length, polyorder=3
            )

    nonzero_speed = np.nonzero(speed)[0]
    if len(nonzero_speed) > 0:
        n = len(nonzero_speed)
        window_length = 21 if n >= 21 else n
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 5:
            speed[nonzero_speed] = signal.savgol_filter(
                speed[nonzero_speed], window_length=window_length, polyorder=3
            )

    return [
        [float(xx), float(yy), float(s), int(l)]
        for xx, yy, s, l in zip(x, y, speed, lane)
    ]


def traj_discrete_action(dir_path: str,
                vel_threshold: float = 0.3) -> None:
    # Define the input and output path directory based on the timestamp directory.
    input_path = os.path.join(dir_path, "vehicle_record_file.csv")
    output_path = os.path.join(dir_path, "processed_dma_data.csv")
    
    # Read the DMA data
    data = pd.read_csv(input_path)
    # Rename columns based on colidxmap
    data.columns = [list(colidxmap.keys())[i] for i in range(len(data.columns))]

    # Perform basic cleaning: keep columns we need
    relevant_columns = ["veh_ID", "time", "x", "y", "speed", "lane_id"]
    data = data[relevant_columns]

    # Sort data by vehicle ID and time
    data = data.sort_values(by=["veh_ID", "time"]).reset_index(drop=True)

    # ---- 1) Smooth trajectories per vehicle ----
    smoothed_chunks = []
    for veh_id, veh_df in data.groupby("veh_ID", sort=False):
        traj = veh_df[["x", "y", "speed", "lane_id"]].to_numpy()
        smoothed_traj = np.array(trajectory_smoothing(traj))

        veh_df_smoothed = veh_df.copy()
        veh_df_smoothed["x"] = smoothed_traj[:, 0]
        veh_df_smoothed["y"] = smoothed_traj[:, 1]
        veh_df_smoothed["speed"] = smoothed_traj[:, 2]
        # lane_id remains unchanged
        smoothed_chunks.append(veh_df_smoothed)

    data = pd.concat(smoothed_chunks, axis=0)
    data = data.sort_values(by=["veh_ID", "time"]).reset_index(drop=True)

    # ---- 2) Generate meta actions on smoothed trajectories (per vehicle) ----
    actions: list[str] = []

    for veh_id, veh_df in data.groupby("veh_ID", sort=False):
        lane = veh_df["lane_id"].to_numpy()
        v = veh_df["speed"].to_numpy()

        n = len(veh_df)
        for i in range(n - 1):
            lane_now = lane[i]
            lane_next = lane[i + 1]
            v_now = v[i]
            v_next = v[i + 1]

            # Lane changes override speed actions
            if (1 <= lane_now <= 5) and (1 <= lane_next <= 5) and (lane_next > lane_now):
                actions.append("LANE_RIGHT")
            elif (1 <= lane_now <= 5) and (1 <= lane_next <= 5) and (lane_next < lane_now):
                actions.append("LANE_LEFT")
            else:
                dv = v_next - v_now
                if dv > vel_threshold:
                    actions.append("FASTER")   # accelerate
                elif dv < -vel_threshold:
                    actions.append("SLOWER")   # decelerate
                else:
                    actions.append("IDLE")     # maintain

        # Last timestep for this vehicle: no look-ahead, default IDLE
        actions.append("IDLE")

    # Sanity check: 1 action per row
    assert len(actions) == len(data), (
        f"Length of actions ({len(actions)}) "
        f"must match number of rows in data ({len(data)})"
    )

    # Final data
    dma_data = data[["veh_ID", "time"]].copy()
    dma_data["action"] = actions

    # Optional: keep smoothed positions and speed
    # dma_data["x"] = data["x"]
    # dma_data["y"] = data["y"]
    # dma_data["speed"] = data["speed"]
    # dma_data["lane_id"] = data["lane_id"]

    # Save the cleaned data to the output path
    dma_data.to_csv(output_path, index=False)
    return None


def traj_cont_action(
    dir_path: str,
    wheelbase: float = 5.0,    # approximate Vehicle.LENGTH from highway-env
    time_scale: float = 1.0,   # multiply 'time' column by this to get seconds (1.0 if already in s, 1e-3 if ms)
) -> None:
    """
    Derive continuous acceleration and steering from smoothed NGSIM trajectories.

    Input:
        - dir_path/vehicle_record_file.csv  (raw NGSIM-style trajectories)

    Output:
        - dir_path/processed_cont_actions.csv
          with columns: veh_ID, time, accel, steering
    """
    input_path = os.path.join(dir_path, "vehicle_record_file.csv")
    output_path = os.path.join(dir_path, "processed_cont_actions.csv")

    # ---- 0) Load and basic preprocessing (same as discrete version) ----
    data = pd.read_csv(input_path)
    data.columns = [list(colidxmap.keys())[i] for i in range(len(data.columns))]

    # Keep only what we need
    relevant_columns = ["veh_ID", "time", "x", "y", "speed", "lane_id"]
    data = data[relevant_columns]

    # Sort by vehicle and time
    data = data.sort_values(by=["veh_ID", "time"]).reset_index(drop=True)

    # ---- 1) Smooth trajectories per vehicle (reuse your function) ----
    smoothed_chunks = []
    for veh_id, veh_df in data.groupby("veh_ID", sort=False):
        traj = veh_df[["x", "y", "speed", "lane_id"]].to_numpy()
        smoothed_traj = np.array(trajectory_smoothing(traj))

        veh_df_smoothed = veh_df.copy()
        veh_df_smoothed["x"] = smoothed_traj[:, 0]
        veh_df_smoothed["y"] = smoothed_traj[:, 1]
        veh_df_smoothed["speed"] = smoothed_traj[:, 2]
        smoothed_chunks.append(veh_df_smoothed)

    data = pd.concat(smoothed_chunks, axis=0)
    data = data.sort_values(by=["veh_ID", "time"]).reset_index(drop=True)

    # ---- 2) Derive continuous actions per vehicle ----
    accel_list = []
    steering_list = []

    eps_speed = 1e-3

    for veh_id, veh_df in data.groupby("veh_ID", sort=False):
        x = veh_df["x"].to_numpy()
        y = veh_df["y"].to_numpy()
        v = veh_df["speed"].to_numpy()
        t = veh_df["time"].to_numpy() * time_scale  # convert to seconds if needed

        n = len(veh_df)
        if n < 2:
            # Degenerate case: single frame for a vehicle
            accel = np.zeros(n)
            steering = np.zeros(n)
        else:
            # --- 2.1. Time deltas ---
            dt = np.diff(t)
            # Avoid zeros in dt (just in case of duplicated timestamps)
            dt[dt <= 0] = np.min(dt[dt > 0]) if np.any(dt > 0) else 1.0

            # --- 2.2. Heading from displacement ---
            dx = np.diff(x)
            dy = np.diff(y)
            heading = np.empty(n)
            heading[:-1] = np.arctan2(dy, dx)
            heading[-1] = heading[-2]  # replicate last valid heading

            # --- 2.3. Yaw rate ---
            dtheta = np.diff(heading)
            dtheta = wrap_to_pi(dtheta)
            yaw_rate = np.empty(n)
            yaw_rate[:-1] = dtheta / dt
            yaw_rate[-1] = 0.0  # no info for last step

            # --- 2.4. Acceleration from speed ---
            dv = np.diff(v)
            accel = np.empty(n)
            accel[:-1] = dv / dt
            accel[-1] = 0.0  # or accel[-1] = accel[-2]

            # --- 2.5. Steering from bicycle model inversion ---
            steering = np.empty(n)
            v_safe = np.maximum(v, eps_speed)
            steering[:-1] = np.arctan(wheelbase * yaw_rate[:-1] / v_safe[:-1])
            steering[-1] = steering[-2] if n > 1 else 0.0

        accel_list.append(accel)
        steering_list.append(steering)

    # Concatenate back into full-length arrays aligned with `data`
    accel_all = np.concatenate(accel_list, axis=0)
    steering_all = np.concatenate(steering_list, axis=0)

    # ---- 3) Build output DataFrame ----
    cont_data = data[["veh_ID", "time"]].copy()
    cont_data["accel"] = accel_all
    cont_data["steering"] = steering_all

    # Optionally keep smoothed x, y, speed, lane_id for debugging/visualization
    # cont_data["x"] = data["x"]
    # cont_data["y"] = data["y"]
    # cont_data["speed"] = data["speed"]
    # cont_data["lane_id"] = data["lane_id"]

    cont_data.to_csv(output_path, index=False)