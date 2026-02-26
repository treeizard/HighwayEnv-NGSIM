import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gymnasium.wrappers import FlattenObservation, RecordVideo

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imitation_logger  # <--- Added for GAIL Tensorboard

# Import your custom environment registration
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ensure this import works in your specific setup
try:
    from highway_env.envs.ngsim_env import NGSimEnv
except ImportError:
    # Fallback/Mock for syntax checking if highway_env isn't installed in the current context
    print("Warning: NGSimEnv not found. Ensure highway-env is installed.")
    NGSimEnv = None

# Configuration
SEED = 42
EXPERT_DATA_PATH = "expert_data/ngsim_expert_discrete.npz"
LOG_DIR = "logs/gail_discrete_simple"
VIDEO_DIR = "logs/videos" # <--- Added directory for videos


# ----------------------------------------------------------------
# 1. Data Loading
# ----------------------------------------------------------------
def load_expert_transitions(path: str) -> Transitions:
    print(f"Loading expert data from {path}...")
    data = np.load(path)
    obs = data["obs"].astype(np.float32)
    acts = data["acts"] 
    next_obs = data["next_obs"].astype(np.float32)
    dones = data["dones"].astype(bool)
    
    acts = acts.astype(int)
    if acts.ndim == 2 and acts.shape[1] == 1:
        acts = acts.ravel()
    
    infos = np.array([{} for _ in range(len(obs))], dtype=object)
    
    t = Transitions(obs=obs, acts=acts, next_obs=next_obs, dones=dones, infos=infos)
    print(f"Loaded {len(obs)} transitions.")
    return t


# ----------------------------------------------------------------
# 2. Environment Factory
# ----------------------------------------------------------------
def make_env(render_mode=None):
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
        "max_episode_steps": 300,
        "expert_test_mode": False, 
        "max_surrounding": 500000,
    }
    # Pass render_mode if recording video
    env = NGSimEnv(config=config, render_mode=render_mode)
    env = FlattenObservation(env)
    return env


# ----------------------------------------------------------------
# 3. Main Training Loop
# ----------------------------------------------------------------
def main():
    # A. Setup Environment
    n_envs = 8
    # Using lambda to pass arguments to make_env if needed, mostly for structure
    venv = make_vec_env(lambda: make_env(), n_envs=n_envs, seed=SEED, vec_env_cls=SubprocVecEnv)
    
    print("Env Observation Space:", venv.observation_space)
    print("Env Action Space:", venv.action_space)

    # B. Load Expert Data
    if not os.path.exists(EXPERT_DATA_PATH):
        # Create dummy data if file missing (FOR DEBUGGING ONLY)
        print(f"WARNING: Expert data not found at {EXPERT_DATA_PATH}.")
        return

    demos = load_expert_transitions(EXPERT_DATA_PATH)
    
    # C. Setup Learner (Generator)
    # The 'tensorboard_log' here captures the PPO Agent's stats (Policy Loss, Value Loss)
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

    # D. Setup Discriminator (Reward Net)
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # E. Setup GAIL Logger
    # <--- 2. Discriminator Tensorboard Setup
    # This configures the imitation library to write to TensorBoard + Standard Out
    gail_logger = imitation_logger.configure(
        folder=os.path.join(LOG_DIR, "gail_metrics"), 
        format_strs=["tensorboard", "stdout"]
    )

    # F. Setup GAIL
    gail_trainer = GAIL(
        demonstrations=demos,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=100_000,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        custom_logger=gail_logger, # <--- Pass the logger here
    )

    # G. Train
    total_timesteps = 500_000
    print(f"Starting GAIL Training for {total_timesteps} timesteps...")
    
    try:
        gail_trainer.train(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model...")

    # H. Save Policy
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    save_path = os.path.join(LOG_DIR, "gail_discrete_simple_model")
    learner.save(save_path)
    print(f"Model saved to {save_path}")

    # Close training envs to free memory for video recording
    venv.close()
    
    # ----------------------------------------------------------------
    # 4. Save Video of Trained Policy
    # ----------------------------------------------------------------
    save_video_of_model(learner, length=300)


def save_video_of_model(model, length=300):
    """
    Runs a single episode with the trained model and saves the video.
    """
    print("Recording video of trained policy...")
    
    # 1. Create a single environment specifically for recording
    # render_mode="rgb_array" is required for RecordVideo
    eval_env = make_env(render_mode="rgb_array") 
    
    # 2. Wrap it with RecordVideo
    # episode_trigger=lambda x: x == 0 ensures we record the very first episode run
    video_env = RecordVideo(
        eval_env, 
        video_folder=VIDEO_DIR, 
        episode_trigger=lambda x: x == 0,
        name_prefix="gail_agent_run"
    )

    # 3. Run the loop
    obs, info = video_env.reset()
    for _ in range(length):
        # Deterministic=True is usually better for evaluation/video
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = video_env.step(action)
        
        if terminated or truncated:
            break
            
    video_env.close()
    print(f"Video saved to {VIDEO_DIR}")


if __name__ == "__main__":
    main()