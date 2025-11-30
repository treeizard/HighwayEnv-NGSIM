import pandas as pd
import argparse
import os

# TODO: Define action constants
vel_threshold = 0.3  # feet/s: threshold for speed change to consider as action

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", help="the path to the DMA csv file", default='./highway_env/data/processed/us-101/vehicle_record_file.csv')
parser.add_argument("--output_path", help="the path to save the processed DMA csv file", default='./highway_env/data/processed/us-101/processed_dma_data.csv')
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

# Definition: https://data.transportation.gov/api/views/8ect-6jqj/files/ddb2c29d-2ef4-4b67-94ea-b55169229bd9?download=true&filename=1-%20US%20101%20Metadata%20Documentation.pdf 
colidxmap = {
    'id': 0,
    'veh_ID': 1,
    'time': 2,
    'x': 3,
    'y': 4,
    'lat': 5,
    'lon': 6,
    'length': 7,
    'width': 8,
    'class': 9,    
    'speed': 10,
    'accel': 11,
    'lane_id': 12,
    'pred_veh_id': 13,
    'follow_veh_id': 14,
    'shead': 15,
    'thead': 16,
}


# Read the DMA data
data = pd.read_csv(input_path)
# Rename columns based on colidxmap
data.columns = [list(colidxmap.keys())[i] for i in range(len(data.columns))]

# Perform basic cleaning: remove non-relevant columns 
relevant_columns = ['veh_ID', 'time', 'speed', 'lane_id']
data = data[relevant_columns]

# sort data by vehicle ID and time
data = data.sort_values(by=['veh_ID', 'time'])
# add action column based on lane changes and speed changes
actions = []

for i in range(len(data) - 1):
    # assuming it takes the controller 0.1s to turn the action into effect
    # assuming when turning lane, lateral actions can be anything

    lane_now = data["lane_id"].iloc[i]
    lane_next = data["lane_id"].iloc[i+1]
    v_now = data["speed"].iloc[i]
    v_next = data["speed"].iloc[i+1]

    # Lane changes override speed actions
    if (lane_next > lane_now) and (1 <= lane_next <= 5) and (1 <= lane_now <= 5):
        actions.append("LANE_RIGHT")   
    elif (lane_next < lane_now) and (1 <= lane_next <= 5) and (1 <= lane_now <= 5):
        actions.append("LANE_LEFT")   
    else:
        dv = v_next - v_now
        if dv > vel_threshold:
            actions.append("FASTER")   # FASTER
        elif dv < -vel_threshold:
            actions.append("SLOWER")   # SLOWER
        else:
            actions.append("IDLE")   # IDLE
actions = actions + ["IDLE"]  # last action is IDLE


# Final data
dma_data = data[['veh_ID', 'time']]
dma_data['action'] = actions  # last action is IDLE
# Save the cleaned data to the output path
dma_data.to_csv(output_path, index=False)   
print(f"Processed DMA data saved to {output_path}")
