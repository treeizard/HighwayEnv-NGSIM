import os
import argparse
import sys

# --- Add parent directory to path so Python can find highway_env ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from highway_env.data.traj_to_action import traj_cont_action


scene = 'us-101/'
timestamp = 't1118846989700'
save_path = 'highway_env/data/processed_10s/'+scene+timestamp

traj_cont_action(save_path)