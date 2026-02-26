import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import FlattenObservation

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

# Import your custom environment registration
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from highway_env.envs.ngsim_env import NGSimEnv

# Configuration
SEED = 42
EXPERT_DATA_PATH = "expert_data/ngsim_expert_discrete.npz"
VAE_PATH = "lidar_vae.pth"
LOG_DIR = "logs/gail_discrete_manifold"
LATENT_DIM = 16  # Must match your trained VAE

# ----------------------------------------------------------------
# 1. Define the VAE Architecture (Must match training script)
# ----------------------------------------------------------------
class LidarVAE(nn.Module):
    def __init__(self, input_dim=128, latent_dim=16):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, 64)
        self.enc2 = nn.Linear(64, 32)
        self.mu = nn.Linear(32, latent_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.mu(h) 

# ----------------------------------------------------------------
# 2. Define the Manifold Reward Net (Discriminator)
# ----------------------------------------------------------------
class ManifoldRewardNet(BasicRewardNet):
    def __init__(
        self, 
        observation_space, 
        action_space, 
        vae_path, 
        latent_dim=16,
        **kwargs
    ):
        # Trick: Tell parent class the input is latent_dim size, not obs_dim size
        dummy_obs_space = Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)
        
        super().__init__(
            observation_space=dummy_obs_space, 
            action_space=action_space, 
            **kwargs
        )

        # Initialize VAE and load weights
        self.vae = LidarVAE(input_dim=observation_space.shape[0], latent_dim=latent_dim)
        
        if os.path.exists(vae_path):
            state_dict = torch.load(vae_path)
            # Filter state dict in case you saved the whole model vs just encoder
            # If your VAE save had 'encoder.' prefix, strip it here if needed.
            self.vae.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded VAE weights from {vae_path}")
        else:
            print(f"⚠️  WARNING: VAE path {vae_path} not found. Using random initialization.")

        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

    def forward(self, state, action, next_state, done):
        # Project raw LIDAR 'state' onto Manifold 'latent_state'
        with torch.no_grad():
            latent_state = self.vae.encode(state)
        
        # Pass latent state to the standard Discriminator MLP
        return super().forward(latent_state, action, next_state, done)

# ----------------------------------------------------------------
# 3. Data Loading
# ----------------------------------------------------------------
def load_expert_transitions(path: str) -> Transitions:
    print(f"Loading expert data from {path}...")
    data = np.load(path)
    obs = data["obs"].astype(np.float32)
    acts = data["acts"].astype(np.float32) # Should be shape [N, 1] or [N]
    next_obs = data["next_obs"].astype(np.float32)
    dones = data["dones"].astype(bool)
    
    # Imitation lib expects discrete actions to be integers of shape (N,) or (N,1)
    # Ensure it's not float
    acts = acts.astype(int)
    
    infos = np.array([{} for _ in range(len(obs))], dtype=object)
    
    t = Transitions(obs=obs, acts=acts, next_obs=next_obs, dones=dones, infos=infos)
    print(f"Loaded {len(obs)} transitions. Obs shape: {obs.shape}, Acts shape: {acts.shape}")
    return t

# ----------------------------------------------------------------
# 4. Environment Factory
# ----------------------------------------------------------------
def make_env():
    config = {
        "scene": "us-101",
        "episode_root": "highway_env/data/processed_10s",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        # CRITICAL: Use Discrete Action Space
        "action": {"type": "DiscreteSteerMetaAction"},
        
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "max_episode_steps": 300,
        "expert_test_mode": False, # We are training, not replaying
        "max_surrounding": 50,
    }
    env = NGSimEnv(config=config)
    env = FlattenObservation(env)
    return env

# ----------------------------------------------------------------
# 5. Main Training Loop
# ----------------------------------------------------------------
def main():
    # A. Setup Environment
    # Use SubprocVecEnv for parallel data collection (speeds up GAIL significantly)
    venv = make_vec_env(make_env, n_envs=8, seed=SEED, vec_env_cls=SubprocVecEnv)
    
    print("Env Observation Space:", venv.observation_space)
    print("Env Action Space:", venv.action_space)

    # B. Load Expert Data
    demos = load_expert_transitions(EXPERT_DATA_PATH)
    
    # C. Setup Learner (Generator) - Simple MLP
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        n_steps=128,        # Increased for discrete stability
        batch_size=256,
        n_epochs=10,
        ent_coef=0.01,      # Slight entropy to prevent early collapse to "IDLE"
        learning_rate=3e-4,
        gamma=0.99,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # D. Setup Discriminator - Manifold VAE + MLP
    reward_net = ManifoldRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        vae_path=VAE_PATH,
        latent_dim=LATENT_DIM,
        # normalization is handled internally by GAIL wrapper usually, 
        # but explicit layers can be added here if needed.
    )

    # E. Setup GAIL
    gail_trainer = GAIL(
        demonstrations=demos,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=100_000,
        n_disc_updates_per_round=8, # Train discriminator harder than generator
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # F. Train
    print("Starting GAIL Training...")
    gail_trainer.train(total_timesteps=500_000)
    
    # G. Save Policy
    save_path = os.path.join(LOG_DIR, "gail_discrete_model")
    learner.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()