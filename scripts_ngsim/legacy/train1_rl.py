#!/usr/bin/env python3
"""
PPO training on highway-env:
- 5 lanes
- variable vehicles_count per episode
- shorter vehicle length (applied after reset each episode)
- no rendering
- short debug run saved as base_model_ppo.zip

Run:
  python train_highway.py
  python train_highway.py --total-timesteps 2000000 --run-name ppo_5lanes_vartraffic --save-name model

TensorBoard:
  tensorboard --logdir tb_logs
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # registers envs  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


# --------------------------
# Base Config: 5 lanes + your obs/action
# --------------------------
HIGHWAY_CONFIG = {
    "lanes_count": 5,

    "observation": {
        "type": "LidarObservation",
        "cells": 128,
        "maximum_range": 64,
        "normalize": True,
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
    },

    "policy_frequency": 10,
    "simulation_frequency": 10,
    "duration": 40,

    # overwritten per episode
    "vehicles_count": 800,
    "controlled_vehicles": 1,
    "vehicles_density": 3,
    "ego_spacing": 0.5,

    # off-road terminal 
    "offroad_terminal": True,
}


# --------------------------
# PPO-safe per-episode randomization:
# - vehicles_count sampled at reset (before spawn)
# - after reset, force vehicle geometry to be shorter
# --------------------------
class EpisodeDomainRandomization(gym.Wrapper):
    def __init__(
        self,
        env,
        min_vehicles: int = 20,
        max_vehicles: int = 80,
        vehicle_length_m: float = 3.5,   # shorter than default ~5m
        vehicle_width_m: float = 1.8,
        apply_to_all_vehicles: bool = True,
    ):
        super().__init__(env)
        assert 0 <= min_vehicles <= max_vehicles
        self.min_vehicles = int(min_vehicles)
        self.max_vehicles = int(max_vehicles)
        self.vehicle_length_m = float(vehicle_length_m)
        self.vehicle_width_m = float(vehicle_width_m)
        self.apply_to_all_vehicles = bool(apply_to_all_vehicles)

    def _apply_size(self):
        """Force size after vehicles exist (post-reset)."""
        uw = self.env.unwrapped

        # Controlled ego vehicle is usually uw.vehicle; other vehicles in uw.road.vehicles
        vehicles = []
        try:
            if self.apply_to_all_vehicles and getattr(uw, "road", None) is not None:
                vehicles.extend(list(getattr(uw.road, "vehicles", [])))
        except Exception:
            pass

        # Always ensure controlled vehicle is included
        ego = getattr(uw, "vehicle", None)
        if ego is not None:
            vehicles.append(ego)

        # De-dup
        seen = set()
        uniq = []
        for v in vehicles:
            if id(v) not in seen:
                uniq.append(v)
                seen.add(id(v))

        for v in uniq:
            # highway-env commonly uses LENGTH/WIDTH attributes
            if hasattr(v, "LENGTH"):
                v.LENGTH = self.vehicle_length_m
            if hasattr(v, "WIDTH"):
                v.WIDTH = self.vehicle_width_m

            # Some forks also store instance fields
            if hasattr(v, "length"):
                v.length = self.vehicle_length_m
            if hasattr(v, "width"):
                v.width = self.vehicle_width_m

    def reset(self, **kwargs):
        # 1) sample density BEFORE reset to affect spawn
        n = np.random.randint(self.min_vehicles, self.max_vehicles + 1)
        self.env.unwrapped.config["vehicles_count"] = int(n)

        # 2) reset spawns vehicles
        obs, info = self.env.reset(**kwargs)

        # 3) enforce smaller geometry AFTER spawn
        self._apply_size()

        # debug signals
        info["vehicles_count"] = int(n)
        info["vehicle_length_m"] = float(self.vehicle_length_m)
        info["vehicle_width_m"] = float(self.vehicle_width_m)
        return obs, info


def make_env(
    env_id: str,
    env_config: dict,
    seed: int,
    min_vehicles: int,
    max_vehicles: int,
    vehicle_length_m: float,
    vehicle_width_m: float,
):
    def _init():
        env = gym.make(env_id)  # no render_mode
        env.configure(env_config)

        env = EpisodeDomainRandomization(
            env,
            min_vehicles=min_vehicles,
            max_vehicles=max_vehicles,
            vehicle_length_m=vehicle_length_m,
            vehicle_width_m=vehicle_width_m,
            apply_to_all_vehicles=True,
        )

        # seed once
        env.reset(seed=seed)

        # TB logging
        env = Monitor(env)
        return env

    return _init


def train(
    run_name: str,
    save_name: str,
    total_timesteps: int,
    seed: int,
    min_vehicles: int,
    max_vehicles: int,
    vehicle_length_m: float,
    vehicle_width_m: float,
):
    log_dir = os.path.join("tb_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([
        make_env(
            "highway-v0",
            HIGHWAY_CONFIG,
            seed=seed,
            min_vehicles=min_vehicles,
            max_vehicles=max_vehicles,
            vehicle_length_m=vehicle_length_m,
            vehicle_width_m=vehicle_width_m,
        )
    ])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
        seed=seed,
    )

    model.learn(
        total_timesteps=int(total_timesteps),
        tb_log_name=run_name,
        progress_bar=True,
    )

    model_path = os.path.join(log_dir, f"{save_name}.zip")
    model.save(model_path)
    env.close()
    return log_dir, model_path


def parse_args():
    p = argparse.ArgumentParser()
    # Defaults: short debug run producing base_model_ppo.zip
    p.add_argument("--run-name", type=str, default="base_model_ppo")
    p.add_argument("--save-name", type=str, default="base_model_ppo")
    p.add_argument("--total-timesteps", type=int, default=800_000)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-vehicles", type=int, default=20)
    p.add_argument("--max-vehicles", type=int, default=100)

    # shorter vehicle geometry
    p.add_argument("--vehicle-length-m", type=float, default=3.5)
    p.add_argument("--vehicle-width-m", type=float, default=1.8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log_dir, model_path = train(
        run_name=args.run_name,
        save_name=args.save_name,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        min_vehicles=args.min_vehicles,
        max_vehicles=args.max_vehicles,
        vehicle_length_m=args.vehicle_length_m,
        vehicle_width_m=args.vehicle_width_m,
    )

    print("TensorBoard logs written to:", log_dir)
    print("Model saved to:", model_path)
    print("Run: tensorboard --logdir tb_logs")
