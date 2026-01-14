import os
import gymnasium as gym
import highway_env  # registers envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

# --------------------------
# Config
# --------------------------
HIGHWAY_CONFIG = {
    "observation": {
        "type": "LidarObservation",
        "cells": 128,
        "maximum_range": 64,
        "normalize": True,
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,  # accel/brake
        "lateral": True,       # steering
    },
    "policy_frequency": 10,
    "simulation_frequency": 10,
    "duration": 40,

    # Scenario knobs
    "vehicles_count": 50,
    "controlled_vehicles": 1,
}

def make_env(env_id: str, env_config: dict, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env.configure(env_config)
        env.reset(seed=seed)
        # Records episode reward/length for TensorBoard ("rollout/ep_rew_mean", etc.)
        env = Monitor(env)
        return env
    return _init

def train_highway(run_name: str = "ppo_lidar_continuous_highway", total_timesteps: int = 2_000_000):
    log_dir = os.path.join("tb_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_env("highway-v0", HIGHWAY_CONFIG, seed=0)])
    env = VecMonitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,

        # PPO defaults that are reasonable for continuous control
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=run_name,
        progress_bar=True,
    )

    model.save(os.path.join(log_dir, "model.zip"))
    env.close()
    return log_dir

if __name__ == "__main__":
    log_dir = train_highway()
    print("TensorBoard logs written to:", log_dir)
    print("Run: tensorboard --logdir tb_logs")
