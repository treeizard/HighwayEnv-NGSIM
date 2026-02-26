import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation, RecordVideo

# Ensure highway-env is importable
try:
    from highway_env.envs.ngsim_env import NGSimEnv
except ImportError:
    print("Error: highway-env not installed.")
    sys.exit(1)

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
MODEL_PATH = "logs/gail_discrete_simple/gail_discrete_simple_model"
VIDEO_DIR = "logs/videos"
VIDEO_LENGTH = 1000  # Max steps per episode
N_VIDEOS = 5         # <--- Number of videos to generate


# ----------------------------------------------------------------
# Environment Factory
# ----------------------------------------------------------------
def make_env(render_mode=None, max_episode_steps=300):
    config = {
        "scene": "us-101",
        "episode_root": "highway_env/data/processed_10s",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "action": {"type": "DiscreteSteerMetaAction"},
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "max_episode_steps": max_episode_steps, 
        "expert_test_mode": False, 
        "max_surrounding": 500000,
    }
    
    env = NGSimEnv(config=config, render_mode=render_mode)
    env = FlattenObservation(env)
    return env


# ----------------------------------------------------------------
# Inference / Recording Loop
# ----------------------------------------------------------------
def main():
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Error: Model not found at {MODEL_PATH}.zip")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    print(f"Recording {N_VIDEOS} videos (Max length: {VIDEO_LENGTH})...")
    
    # 1. Create environment
    env = make_env(render_mode="rgb_array", max_episode_steps=VIDEO_LENGTH)
    
    # 2. Configure RecordVideo
    # episode_trigger=lambda x: x < N_VIDEOS ensures we record episodes 0, 1, 2, 3, 4
    env = RecordVideo(
        env, 
        video_folder=VIDEO_DIR, 
        episode_trigger=lambda x: x < N_VIDEOS,  # <--- CHANGED: Records first N episodes
        name_prefix="inference_run"
    )

    # 3. Run the Simulation Loop
    for video_idx in range(N_VIDEOS):
        print(f"--- Recording Video {video_idx + 1}/{N_VIDEOS} ---")
        
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            step += 1
            if terminated or truncated or step >= VIDEO_LENGTH:
                done = True
                print(f"Episode {video_idx + 1} finished at step {step}")
            
    env.close()
    print(f"Done! All {N_VIDEOS} videos saved to {VIDEO_DIR}")

if __name__ == "__main__":
    main()