
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# $env:PYTHONPATH = "C:\Users\teeds\OneDrive\Attachments\Desktop\projects\research\HIGH\HighwayEnv-NGSIM"
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



def build_trajecotry(scene, period, vehicle_ID):
    ng = ngsim_data(scene)
    ng.load('processed_10s\\'+scene)
    records = ng.vr_dict
    vehicles = ng.veh_dict
    snapshots = ng.snap_dict

    surroundings = []
    record_trajectory = {'ego':{'length':0, 'width':0, 'trajectory':[]}}
    
    print("PRINT VEHICLES INFORMATION FROM veh_dict")
    for veh_ID, v in vehicles.items():
        print(v)
        v.build_trajectory()
    

    ego_trajectories = vehicles[vehicle_ID].trajectory
    selected_trajectory = ego_trajectories[period]

    D = 50 if scene == 'us-101' else 20

    ego = []
    nearby_IDs = []
    excluded = []

    for position in selected_trajectory:
        print("POS")
        print(position)
        record_trajectory['ego']['length'] = position.len
        record_trajectory['ego']['width'] = position.wid
        ego.append([position.x, position.y, position.spd, position.lane_ID])
        records = snapshots[position.unixtime].vr_list
        other = []
        for record in records:
            if record.veh_ID != vehicle_ID:
                other.append([record.veh_ID, record.len, record.wid, record.x, record.y, record.spd, record.lane_ID])
                d = abs(position.y - record.y)
                print(record.veh_ID)
                print(d)
                if d <= D:            
                    nearby_IDs.append(record.veh_ID)
                else:
                    excluded.append((record.veh_ID, d))
        surroundings.append(other)
        
    print("----- timestep -----")
    print("Excluded {< = D}:", excluded)
    print()

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
        # store original first
        record_trajectory[key]['original_trajectory'] = [x.copy() for x in orginal_trajectory]  
        # then smooth
        smoothed_trajectory = trajectory_smoothing(orginal_trajectory)
        record_trajectory[key]['trajectory'] = smoothed_trajectory

    print('ALL VEHICLEs')

    print(record_trajectory.keys())
    return record_trajectory

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    scene = 'us-101'
    trajectory_set = build_trajecotry(scene, 0, 2)

    print(trajectory_set)

    # Extract original and smoothed trajectories
    ego_original = np.array(trajectory_set['ego']['original_trajectory'])
    ego_smooth = np.array(trajectory_set['ego']['trajectory'])

    print(np.unique(ego_smooth[:][3]))
        
    # Threshold for speed changes to consider longitudinal actions
    vel_threshold = 0.3  # feet/s, can adjust if needed

    actions = []

    for i in range(len(ego_smooth)-1):
        lane_now = ego_smooth[i][3]
        lane_next = ego_smooth[i+1][3]
        speed_now = ego_smooth[i][2]
        speed_next = ego_smooth[i+1][2]
        
        # Determine lateral action first
        if lane_next > lane_now:
            actions.append("LANE_RIGHT")
        elif lane_next < lane_now:
            actions.append("LANE_LEFT")
        else:
            # If no lane change, check longitudinal action
            dv = speed_next - speed_now
            if dv > vel_threshold:
                actions.append("FASTER")
            elif dv < -vel_threshold:
                actions.append("SLOWER")
            else:
                actions.append("IDLE")

    # Add IDLE for last timestep
    actions.append("IDLE")

    print(actions)
    # Prepare surrounding vehicles
    vehicles_traj = {}
    for v_ID, data in trajectory_set.items():
        traj = np.array(data['trajectory'])
        mask = ~np.all(traj[:, :2] == 0.0, axis=1)
        traj_filtered = traj.copy()
        traj_filtered[~mask, :] = np.nan  # hide missing
        vehicles_traj[v_ID] = traj_filtered

    # Determine total timesteps
    timesteps = ego_smooth.shape[0]

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(ego_original[:,1]/3.281, ego_original[:,0]/3.281, label='Ego Original', linestyle='--', color='blue')
    ax.plot(ego_smooth[:,1]/3.281, ego_smooth[:,0]/3.281, label='Ego Smoothed', linestyle='-', color='red')

    lines = {}  # for flashing points
    points = {}
    for v_ID, traj in vehicles_traj.items():
        if v_ID == 'ego':
            points[v_ID], = ax.plot([], [], 'bo', markersize=8)  # ego flashing point
        else:
            ax.plot(traj[:,1]/3.281, traj[:,0]/3.281, '--', alpha=0.5, label=f'Veh {v_ID}')
            points[v_ID], = ax.plot([], [], 'ro', markersize=5)  # moving point

    ax.invert_yaxis()
    ax.set_xlabel('Longitudinal position [m]')
    ax.set_ylabel('Lateral position [m]')
    ax.legend()

    def init():
        for pt in points.values():
            pt.set_data([], [])
        return points.values()

    def update(frame):
        for v_ID, traj in vehicles_traj.items():
            x = traj[frame,1]/3.281
            y = traj[frame,0]/3.281
            points[v_ID].set_data([x], [y])
        return points.values()

    ani = FuncAnimation(fig, update, frames=range(0, timesteps, 5),  # skip frames to make ~0.5s interval
                        init_func=init, blit=True, interval=500, repeat=True)

    plt.show()
    
    # plt.plot(ego_original[:,1]/3.281, ego_original[:,0]/3.281, label='Original', linestyle='--', color='blue')
    # plt.plot(ego_smooth[:,1]/3.281, ego_smooth[:,0]/3.281, label='Smoothed', linestyle='-', color='red')

    # # Plot surrounding vehicles
    # for v_ID, data in trajectory_set.items():
    #     if v_ID == 'ego':
    #         continue
    #     traj = np.array(data['trajectory'])
    #     # Filter out 0.0,0.0 placeholders
    #     mask = ~np.all(traj[:, :2] == 0.0, axis=1)
    #     traj_filtered = traj[mask]
    #     if len(traj_filtered) > 0:
    #         plt.plot(traj_filtered[:,1]/3.281, traj_filtered[:,0]/3.281, '--', label=f'Veh {v_ID}')

    # plt.gca().set_aspect('auto', 'datalim')
    # plt.gca().invert_yaxis()
    # plt.xlabel('Longitudinal position [m]', fontsize=20)
    # plt.ylabel('Lateral position [m]', fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.legend(fontsize=12)

    # plt.tight_layout()
    # plt.show()