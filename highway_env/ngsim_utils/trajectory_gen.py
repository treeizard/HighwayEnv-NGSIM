'''
Modified by: Yide Tao (yide.tao@monash.edu)
Source: https://github.com/MCZhi/Driving-IRL-NGSIM/tree/main/NGSIM_env
Reference: @article{huang2021driving,
  title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
  author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from highway_env.data.ngsim import *

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
# 

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

    D = 50 if scene == 'us-101' else 20

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

def process_raw_trajectory(trajectory):
    trajectory = np.array(trajectory)
    for i in range(trajectory.shape[0]):
        x = trajectory[i][0] - 6
        y = trajectory[i][1]
        speed = trajectory[i][2]
        trajectory[i][0] = y / 3.281
        trajectory[i][1] = x / 3.281
        trajectory[i][2] = speed / 3.281

    return trajectory

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