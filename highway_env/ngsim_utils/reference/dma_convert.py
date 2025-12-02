import pandas as pd
from scipy import signal
import numpy as np
import argparse
import os

# Written by: Huy Nguyen
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


vel_threshold = 0.3  # feet/s: threshold for speed change to consider as action

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    help="the path to the DMA csv file",
    default="./highway_env/data/processed/us-101/vehicle_record_file.csv",
)
parser.add_argument(
    "--output_path",
    help="the path to save the processed DMA csv file",
    default="./highway_env/data/processed/us-101/processed_dma_data.csv",
)
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

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

# Read the DMA data
data = pd.read_csv(input_path)
# Rename columns based on colidxmap
data.columns = [list(colidxmap.keys())[i] for i in range(len(data.columns))]

# Perform basic cleaning: keep columns we need, including x,y so we can smooth them
relevant_columns = ["veh_ID", "time", "x", "y", "speed", "lane_id"]
data = data[relevant_columns]

# sort data by vehicle ID and time
data = data.sort_values(by=["veh_ID", "time"])

# ---- 1) Smooth trajectories per vehicle ----
smoothed_chunks = []
for veh_id, veh_df in data.groupby("veh_ID", sort=False):
    traj = veh_df[["x", "y", "speed", "lane_id"]].to_numpy()
    smoothed_traj = np.array(trajectory_smoothing(traj))

    veh_df_smoothed = veh_df.copy()
    veh_df_smoothed["x"] = smoothed_traj[:, 0]
    veh_df_smoothed["y"] = smoothed_traj[:, 1]
    veh_df_smoothed["speed"] = smoothed_traj[:, 2]
    # lane_id remains unchanged (column already there)
    smoothed_chunks.append(veh_df_smoothed)

data = pd.concat(smoothed_chunks, axis=0)
data = data.sort_values(by=["veh_ID", "time"]).reset_index(drop=True)

# ---- 2) Generate meta actions on smoothed trajectories (per vehicle) ----
actions = []

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
        if (lane_next > lane_now) and (1 <= lane_next <= 5) and (1 <= lane_now <= 5):
            actions.append("LANE_RIGHT")
        elif (lane_next < lane_now) and (1 <= lane_next <= 5) and (1 <= lane_now <= 5):
            actions.append("LANE_LEFT")
        else:
            dv = v_next - v_now
            if dv > vel_threshold:
                actions.append("FASTER")  # accelerate
            elif dv < -vel_threshold:
                actions.append("SLOWER")  # decelerate
            else:
                actions.append("IDLE")  # maintain

    # Last timestep for this vehicle: no look-ahead, default IDLE
    actions.append("IDLE")

# Sanity check
assert len(actions) == len(data), "Length of actions must match number of rows in data"

# Final data
dma_data = data[["veh_ID", "time"]].copy()
dma_data["action"] = actions  # last action per vehicle is IDLE

# (Optional) if you want to also keep smoothed x,y,speed, uncomment:
# dma_data["x"] = data["x"]
# dma_data["y"] = data["y"]
# dma_data["speed"] = data["speed"]
# dma_data["lane_id"] = data["lane_id"]

# Save the cleaned data to the output path
dma_data.to_csv(output_path, index=False)
print(f"Processed DMA data saved to {output_path}")
