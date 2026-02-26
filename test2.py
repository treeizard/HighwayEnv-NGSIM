#!/usr/bin/env python3
"""
Record a trained PPO policy (highway-env) to an MP4 video.

Example:
  python record_highway_video.py \
    --model tb_logs/base_model_ppo/base_model_ppo.zip \
    --env-id highway-v0 \
    --outdir videos \
    --episodes 3 \
    --seed 0

Notes:
- Uses Gymnasium RecordVideo with render_mode="rgb_array".
- Keeps config consistent with training (5 lanes, lidar, continuous actions, etc.).
- If your model was trained with a different config, align the config here accordingly.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # registers envs  # noqa: F401

from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo


# Match your training config (edit if needed)
EVAL_CONFIG = {
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
    "vehicles_count": 50,         # fixed for evaluation video; change as desired
    "controlled_vehicles": 1,
    "vehicles_density": 1,
    "ego_spacing": 1,
    "offroad_terminal": True,

}


class FixedVehicleSizeWrapper(gym.Wrapper):
    """Force vehicle geometry after reset (useful if you trained with shorter vehicles)."""

    def __init__(self, env, vehicle_length_m: float = 3.5, vehicle_width_m: float = 1.8, apply_to_all: bool = True):
        super().__init__(env)
        self.vehicle_length_m = float(vehicle_length_m)
        self.vehicle_width_m = float(vehicle_width_m)
        self.apply_to_all = bool(apply_to_all)

    def _apply_size(self):
        uw = self.env.unwrapped
        vehicles = []

        if self.apply_to_all and getattr(uw, "road", None) is not None:
            vehicles.extend(list(getattr(uw.road, "vehicles", [])))

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
            if hasattr(v, "LENGTH"):
                v.LENGTH = self.vehicle_length_m
            if hasattr(v, "WIDTH"):
                v.WIDTH = self.vehicle_width_m
            if hasattr(v, "length"):
                v.length = self.vehicle_length_m
            if hasattr(v, "width"):
                v.width = self.vehicle_width_m

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_size()
        return obs, info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip model file")
    p.add_argument("--env-id", type=str, default="highway-v0")
    p.add_argument("--outdir", type=str, default="videos")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    # Evaluation traffic and geometry (optional overrides)
    p.add_argument("--vehicles-count", type=int, default=50)
    p.add_argument("--lanes-count", type=int, default=5)
    p.add_argument("--duration", type=int, default=40)

    p.add_argument("--vehicle-length-m", type=float, default=3.5)
    p.add_argument("--vehicle-width-m", type=float, default=1.8)
    p.add_argument("--no-size-wrapper", action="store_true", help="Disable forced vehicle size.")

    # Video capture control
    p.add_argument("--fps", type=int, default=10, help="Video FPS (should match policy_frequency for clean playback).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    model = PPO.load(args.model)

    # Build eval config (keep consistent with training)
    cfg = dict(EVAL_CONFIG)
    cfg["vehicles_count"] = int(args.vehicles_count)
    cfg["lanes_count"] = int(args.lanes_count)
    cfg["duration"] = int(args.duration)

    # Create env with rgb_array rendering for video
    env = gym.make(args.env_id, render_mode="rgb_array")
    env.configure(cfg)

    # Force shorter vehicles if you trained that way (recommended for visual consistency)
    if not args.no_size_wrapper:
        env = FixedVehicleSizeWrapper(
            env,
            vehicle_length_m=args.vehicle_length_m,
            vehicle_width_m=args.vehicle_width_m,
            apply_to_all=True,
        )

    # Wrap video recorder
    # Record every episode
    env = RecordVideo(
        env,
        video_folder=args.outdir,
        episode_trigger=lambda ep: True,
        name_prefix="highway_ppo",
        disable_logger=True,
    )

    # Seed
    obs, info = env.reset(seed=args.seed)

    for ep in range(args.episodes):
        done = truncated = False
        obs, info = env.reset(seed=args.seed + ep)

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        print(f"Episode {ep+1}/{args.episodes} finished. done={done}, truncated={truncated}")

    env.close()
    print(f"Videos saved to: {os.path.abspath(args.outdir)}")
    print("Tip: If playback is too fast/slow, ensure policy_frequency==fps (and your viewer respects FPS).")


if __name__ == "__main__":
    main()
