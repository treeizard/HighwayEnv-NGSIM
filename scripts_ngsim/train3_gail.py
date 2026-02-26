import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation, RecordVideo

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imitation_logger

# Ensure highway_env is accessible
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from highway_env.envs.ngsim_env import NGSimEnv
except ImportError:
    print("Warning: NGSimEnv not found.")

# Configuration
SEED = 42
EXPERT_DATA_PATH = "expert_data/ngsim_expert_continuous.npz" # Changed naming
LOG_DIR = "logs/gail_continuous"
VIDEO_DIR = "logs/videos"

# ----------------------------------------------------------------
# 1. Data Loading (Updated for Continuous)
# ----------------------------------------------------------------
def load_expert_transitions(path: str) -> Transitions:
    print(f"Loading expert data from {path}...")
    data = np.load(path)
    obs = data["obs"].astype(np.float32)
    # Continuous actions should be float32
    acts = data["acts"].astype(np.float32) 
    next_obs = data["next_obs"].astype(np.float32)
    dones = data["dones"].astype(bool)
    
    infos = np.array([{} for _ in range(len(obs))], dtype=object)
    return Transitions(obs=obs, acts=acts, next_obs=next_obs, dones=dones, infos=infos)

# ----------------------------------------------------------------
# 2. Environment Factory (Updated for Continuous)
# ----------------------------------------------------------------
def make_env(render_mode=None):
    config = {
        "scene": "us-101",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        # --- CHANGED TO CONTINUOUS ---
        "action": {
            "type": "ContinuousAction", 
            "acceleration_range": [-5, 5],
            "steering_range": [-np.pi/4, np.pi/4]
        },
        "simulation_frequency": 100,
        "policy_frequency": 100,
        "max_episode_steps": 300, # Shortened for your 10-step request
        "expert_test_mode": False, 
        "max_surrounding": 500000,
    }
    env = NGSimEnv(config=config, render_mode=render_mode)
    env = FlattenObservation(env)
    return env

# ----------------------------------------------------------------
# 3. Main Logic
# ----------------------------------------------------------------
def main():
    # Setup for 10-step quick run
    n_envs = 8
    venv = make_vec_env(lambda: make_env(), n_envs=n_envs, seed=SEED)
    
    # Load continuous demos
    if not os.path.exists(EXPERT_DATA_PATH):
        print(f"Expert data missing at {EXPERT_DATA_PATH}")
        return
    demos = load_expert_transitions(EXPERT_DATA_PATH)
    
    # Learner
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        n_steps=128,
        batch_size=256,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=3e-4,
        gamma=0.99,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR, # <--- 1. Agent Tensorboard
    )

    # GAIL Setup
    reward_net = BasicRewardNet(venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm)
    gail_trainer = GAIL(
        demonstrations=demos,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=100_000,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # Train for a tiny bit or just skip to video if you just want to see it run
    print("Running 10 steps of training/testing...")
    gail_trainer.train(total_timesteps=20_000) 

    # 4. Save Video (10 Steps)
    save_video_of_model(learner, length=300)

def save_video_of_model(model, length=300):
    print(f"Recording {length} steps of continuous action...")
    eval_env = make_env(render_mode="rgb_array") 
    
    video_env = RecordVideo(
        eval_env, 
        video_folder=VIDEO_DIR, 
        episode_trigger=lambda x: True,
        name_prefix="gail_continuous_run"
    )

    obs, info = video_env.reset()
    for _ in range(length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = video_env.step(action)
        if terminated or truncated:
            break
            
    video_env.close()
    print(f"Video saved to {VIDEO_DIR}")

if __name__ == "__main__":
    main()