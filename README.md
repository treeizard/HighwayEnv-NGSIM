# Highway Env with NGSIM Integration Guide
## 1. Reference Documentation
- Original Highway_Env: https://highway-env.farama.org/index.html
- Stable Baseline 3: https://stable-baselines3.readthedocs.io/en/master/
- Drive-IRL Sim: https://github.com/MCZhi/Driving-IRL-NGSIM/blob/main/NGSIM_env/envs/ngsim_env.py
## 2. Installation Guide
### 2.1. Install All Packages
1. Create a conda environment:
```
conda create -n highway_ngsim python=3.11
```
2. Activate conda environment:
```
conda activate highway_ngsim
```
3. Install **Pytorch with CUDA** following the guidelines outlined on the official documentation, then install stable-baseline 3 for RL. 
```
pip install git+https://github.com/DLR-RM/stable-baselines3
```
Sometimes the packages: `typeguard` and `pyyaml` will be missing. 
4. Install highway-env:
```
pip install highway-env
```
5. Install gymnasium, imitation and tensorboard:
```
pip install gymnasium
```
```
pip install gymnasium[other]
```
```
pip install imitation
```
```
pip install tensorboard
```
### 2.2. Validate Installation
Within the `./HighwayEnv-NGSIM` directory level, run: 
```
python3 scripts/sb3_highway_dqn.py
```
You should see rendering of the highway env environment and video playing. 

### 2.3. Data set up
1. Download the raw NGSIM data from the [link](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data), the download process may take some time. 
2. Place the csv trajectory file inside the `raw_data` folder. If you do not change the name of the raw Data, you can just run:
```
python dump_data.py raw_data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv 
```
in development:
```
python dump_data_time.py raw_data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv
```
