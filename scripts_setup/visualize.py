#!/usr/bin/env python3
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from PIL import Image


# ---------------------------------------------------------------------
# Make project importable
# ---------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

register(
    id="NGSim-US101-v0",
    entry_point="highway_env.envs.ngsim_env:NGSimEnv",
)


def save_highres_snapshot(
    out_path: str = "plots/japanese_snapshot.png",
    width: int = 5000,
    height: int = 700,
    scaling: float = 7.0,
    episode_name: str | None = None,
    ego_vehicle_ID: int | None = None,
    seed: int = 42,
    road_only: bool = False,
):
    """
    Save a high-resolution PNG snapshot of the Japanese scene.

    Args:
        out_path: output image path
        width: render width in pixels
        height: render height in pixels
        scaling: renderer zoom factor
        episode_name: optional fixed episode name
        ego_vehicle_ID: optional fixed ego vehicle id
        seed: random seed
        road_only: if True, render road layout only; otherwise render full scene
    """

    config = {
        "scene": "japanese",
        "seed": seed,
        "show_trajectories": True,
        "screen_width": width,
        "screen_height": height,
        "scaling": scaling,
        "simulation_period": {"episode_name": episode_name} if episode_name is not None else None,
        "ego_vehicle_ID": ego_vehicle_ID,
        "action_mode": "discrete",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "policy_frequency": 10,
        "simulation_frequency": 10,
        "max_episode_steps": 300,
        "centering_position": [0.65, 0.5],
    }

    env = gym.make(
        "NGSim-US101-v0",
        config=config,
        render_mode="rgb_array",
    )

    try:
        base_env = env.unwrapped

        # Make sure rendering config is set on the underlying env
        base_env.config["screen_width"] = width
        base_env.config["screen_height"] = height
        base_env.config["scaling"] = scaling

        if road_only:
            # Build road only, without resetting into an episode
            base_env._create_road()

            # Some renderers expect this attribute to exist
            if not hasattr(base_env, "vehicle"):
                base_env.vehicle = None

            frame = base_env.render()
        else:
            # Full scene: road + ego + surrounding vehicles
            reset_out = env.reset(seed=seed)
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, info = reset_out
            else:
                obs = reset_out
                info = {}

            frame = env.render()

        if frame is None:
            raise RuntimeError(
                "render() returned None. Check that render_mode='rgb_array' is supported."
            )

        frame = np.asarray(frame)

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        image = Image.fromarray(frame)
        image.save(out_path)

        print(f"Saved snapshot to: {out_path}")
        print(f"Frame shape: {frame.shape}")
        print(f"Road only: {road_only}")

    finally:
        env.close()


if __name__ == "__main__":
    # -------------------------------------------------------------
    # Example 1: Japanese road only
    # -------------------------------------------------------------
    save_highres_snapshot(
        out_path="plots/japanese_road_only.png",
        width=5000,
        height=700,
        scaling=7.0,
        seed=42,
        road_only=True,
    )

    