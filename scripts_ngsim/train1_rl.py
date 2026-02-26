#!/usr/bin/env python3
"""
PPO training on highway-env with optional live rendering OR periodic video recording.

Features:
- 5 lanes
- LidarObservation (128 cells)
- ContinuousAction (longitudinal + lateral)
- Fixed vehicle geometry applied after reset (ego + others)
  - IMPORTANT: recompute observation after geometry change so Lidar is consistent at t=0
- Optional live render callback (human window)
- Optional RecordVideo wrapper (rgb_array)

Examples:
  # Fast headless training
  python train_highway_render.py --total-timesteps 800000

  # Live render during training (slow)
  python train_highway_render.py --render --render-freq 1 --total-timesteps 200000

  # Record one episode every 5 episodes (recommended for debugging)
  python train_highway_render.py --record-video --video-every-episodes 5 --video-length 400

TensorBoard:
  tensorboard --logdir tb_logs
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import highway_env  # registers envs  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from gymnasium.wrappers import RecordVideo


# --------------------------
# Base Config: 5 lanes + lidar + continuous actions
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
    "vehicles_count": 50,
    "controlled_vehicles": 1,
    "vehicles_density": 3,
    "ego_spacing": 0.5,
    "offroad_terminal": True,
}


# --------------------------
# Wrapper: fixed vehicle geometry + recompute obs after reset
# --------------------------
class FixedVehicleSizeWrapper(gym.Wrapper):
    def __init__(self, env, vehicle_length_m=3.5, vehicle_width_m=1.8, apply_to_all=True):
        super().__init__(env)
        self.vehicle_length_m = float(vehicle_length_m)
        self.vehicle_width_m = float(vehicle_width_m)
        self.apply_to_all = bool(apply_to_all)

    def _apply_size(self) -> None:
        uw = self.env.unwrapped
        vehicles = []
        if self.apply_to_all and getattr(uw, "road", None) is not None:
            vehicles.extend(list(getattr(uw.road, "vehicles", [])))
        ego = getattr(uw, "vehicle", None)
        if ego is not None:
            vehicles.append(ego)

        seen, uniq = set(), []
        for v in vehicles:
            if id(v) not in seen:
                uniq.append(v); seen.add(id(v))

        for v in uniq:
            if hasattr(v, "LENGTH"): v.LENGTH = self.vehicle_length_m
            if hasattr(v, "WIDTH"):  v.WIDTH  = self.vehicle_width_m
            if hasattr(v, "length"): v.length = self.vehicle_length_m
            if hasattr(v, "width"):  v.width  = self.vehicle_width_m

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_size()
        # NO manual obs recompute (match your eval path)
        #info = dict(info)
        #info["vehicle_length_m"] = self.vehicle_length_m
        #info["vehicle_width_m"] = self.vehicle_width_m
        return obs, info


# --------------------------
# Callbacks
# --------------------------
class VecRenderCallback(BaseCallback):
    def __init__(self, render_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = max(1, int(render_freq))

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq != 0:
            return True

        try:
            env0 = self.training_env.envs[0]  # Monitor(RecordVideo(FixedWrapper(base)))
            # unwrap down to the actual base env that owns the viewer
            while hasattr(env0, "env"):
                env0 = env0.env
            env0.render()
        except Exception:
            pass
        return True



class VideoRecorderCallback(BaseCallback):
    """
    Lightweight callback: lets RecordVideo handle triggering; we just keep training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True


# --------------------------
# Environment factory
# --------------------------
def make_env(
    env_id: str,
    env_config: dict,
    seed: int,
    vehicle_length_m: float,
    vehicle_width_m: float,
    render: bool,
    record_video: bool,
    video_dir: str,
    video_every_episodes: int,
    video_length: int,
):
    def _init():
        # Decide render_mode
        # - For live rendering: "human"
        # - For video recording: "rgb_array"
        # - Otherwise: None (fastest)
        render_mode = None
        if render:
            render_mode = "human"
        elif record_video:
            render_mode = "rgb_array"

        env = gym.make(env_id, render_mode=render_mode)
        env.configure(env_config)

        # Apply fixed vehicle size wrapper BEFORE Monitor/RecordVideo is fine,
        # but we must ensure obs is recomputed after reset.
        env = FixedVehicleSizeWrapper(
            env,
            vehicle_length_m=vehicle_length_m,
            vehicle_width_m=vehicle_width_m,
            apply_to_all=True,
        )

        # Optional: video recording wrapper
        if record_video:
            os.makedirs(video_dir, exist_ok=True)

            def _trigger(episode_id: int) -> bool:
                # record episode 0, N, 2N, ...
                return (episode_id % max(1, int(video_every_episodes))) == 0

            env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=_trigger,
                name_prefix="ppo_train",
                video_length=int(video_length),
                disable_logger=True,
            )

        # Monitor outermost: best practice for SB3 episode stats
        env = Monitor(env)

        # Seed AFTER all wrappers, so counters/recording/monitor are consistent
        env.reset(seed=seed)
        return env

    return _init


# --------------------------
# Training
# --------------------------
def train(args):
    log_dir = os.path.join("tb_logs", args.run_name)
    os.makedirs(log_dir, exist_ok=True)

    video_dir = os.path.join(log_dir, "videos")

    env = DummyVecEnv(
        [
            make_env(
                env_id=args.env_id,
                env_config=HIGHWAY_CONFIG,
                seed=args.seed,
                vehicle_length_m=args.vehicle_length_m,
                vehicle_width_m=args.vehicle_width_m,
                render=args.render,
                record_video=args.record_video,
                video_dir=video_dir,
                video_every_episodes=args.video_every_episodes,
                video_length=args.video_length,
            )
        ]
    )
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        seed=args.seed,
    )

    callbacks = []
    if args.render:
        callbacks.append(VecRenderCallback(render_freq=args.render_freq))
    if args.record_video:
        callbacks.append(VideoRecorderCallback())

    callback = CallbackList(callbacks) if callbacks else None

    model.learn(
        total_timesteps=int(args.total_timesteps),
        tb_log_name=args.run_name,
        progress_bar=True,
        callback=callback,
    )

    model_path = os.path.join(log_dir, f"{args.save_name}.zip")
    model.save(model_path)

    env.close()

    print("TensorBoard logs written to:", log_dir)
    print("Model saved to:", model_path)
    if args.record_video:
        print("Training videos saved to:", video_dir)
    print("Run: tensorboard --logdir tb_logs")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--env-id", type=str, default="highway-v0")

    p.add_argument("--run-name", type=str, default="base_model_ppo")
    p.add_argument("--save-name", type=str, default="base_model_ppo")
    p.add_argument("--total-timesteps", type=int, default=800_000)

    p.add_argument("--seed", type=int, default=0)

    # Fixed geometry
    p.add_argument("--vehicle-length-m", type=float, default=3.5)
    p.add_argument("--vehicle-width-m", type=float, default=1.8)

    # PPO hparams
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)

    # Live rendering
    p.add_argument("--render", action="store_true", help="Enable live rendering during training (slow).")
    p.add_argument(
        "--render-freq",
        type=int,
        default=1,
        help="Render every N training steps (higher=faster training).",
    )

    # Video recording during training
    p.add_argument("--record-video", action="store_true", help="Record training episodes to MP4 (headless).")
    p.add_argument(
        "--video-every-episodes",
        type=int,
        default=10,
        help="Record one episode every N episodes (0th, Nth, 2Nth...).",
    )
    p.add_argument(
        "--video-length",
        type=int,
        default=400,
        help="Max frames per recorded video episode.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
