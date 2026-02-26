#!/usr/bin/env python3
import os, sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def run_one_episode(env, max_steps=300):
    taken_labels = []
    taken_idxs = []

    obs, info = env.reset()

    for t in range(max_steps):
        # action ignored by expert_test_mode, but must be valid
        a = 0
        obs, r, terminated, truncated, info = env.step(a)

        if isinstance(info, dict):
            if "expert_action_discrete" in info:
                taken_labels.append(str(info["expert_action_discrete"]))
            if "expert_action_discrete_idx" in info:
                taken_idxs.append(int(info["expert_action_discrete_idx"]))

        if terminated or truncated:
            break

    # Optional: metrics per episode
    metrics = None
    try:
        metrics = env.unwrapped.expert_replay_metrics()
    except Exception:
        pass

    return taken_labels, taken_idxs, metrics


def main():
    out_dir = os.path.abspath("./videos_discrete_test")
    os.makedirs(out_dir, exist_ok=True)

    DISCRETE_ACTION_TYPE_NAME = "DiscreteSteerMetaAction"

    base_cfg = {
        "scene": "us-101",
        "observation": {"type": "LidarObservation", "cells": 128, "maximum_range": 64, "normalize": True},
        "action": {"type": DISCRETE_ACTION_TYPE_NAME, "target_speeds": list(np.arange(0.0, 35.0 + 1e-6, 2.0))},
        "show_trajectories": True,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 400,
        "screen_height": 150,
        "scaling": 2.0,
        "offscreen_rendering": True,
        "ego_vehicle_ID": None,
        "episode_root": "highway_env/data/processed_10s",
        "replay_period": None,          # random episode each reset
        "max_surrounding": 20000,         
        "expert_test_mode": True,
        "expert_action_mode": "discrete",
        "expert_speed_deadband_mps": 0.5,
        "expert_steer_deadband_rad": 0.05,
        "expert_one_action_per_step": True,
        "expert_prefer_speed": False,
    }

    env = gym.make("NGSim-US101-v0", render_mode="rgb_array", config=base_cfg)

    env = RecordVideo(
        env,
        video_folder=out_dir,
        episode_trigger=lambda ep_idx: True,     # record every episode
        name_prefix="ngsim_expert_discrete",
    )

    N_EPISODES = 20
    MAX_STEPS = 300

    for ep in range(N_EPISODES):
        labels, idxs, metrics = run_one_episode(env, max_steps=MAX_STEPS)
        unwrapped = env.unwrapped
        print(f"\n=== Episode {ep} ===")
        print(f"episode_name={getattr(unwrapped, 'episode_name', None)} ego_id={getattr(unwrapped, 'ego_id', None)}")
        if metrics is not None:
            print("metrics:", metrics)

        # quick histogram
        if labels:
            labs, cnt = np.unique(np.array(labels, dtype=str), return_counts=True)
            print("label counts:", dict(zip(labs.tolist(), cnt.tolist())))

    env.close()
    print(f"\nSaved videos to: {out_dir}")


if __name__ == "__main__":
    main()
