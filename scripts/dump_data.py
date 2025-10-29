'''
Github Link: https://github.com/MCZhi/Driving-IRL-NGSIM/tree/main
@article{huang2021driving,
  title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
  author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
'''
import os
import argparse
import sys

# --- Add parent directory to path so Python can find highway_env ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from highway_env.data.ngsim import *
# -------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("path", help="the path to the NGSIM csv file")
parser.add_argument("--scene", help="location", default='us-101')
args = parser.parse_args()

path = args.path
scene = args.scene
reader = ngsim_data(scene)
reader.read_from_csv(path)
reader.clean()

save_path = 'highway_env/data/processed/'+scene
if not os.path.exists(save_path):
    os.makedirs(save_path)
reader.dump(folder=save_path)