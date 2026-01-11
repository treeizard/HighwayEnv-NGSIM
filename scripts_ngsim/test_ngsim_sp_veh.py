# scripts/make_videos_multi.py
import os, sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register

# Make sure project root is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Register NGSimEnv
register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def main():
    out_dir = os.path.abspath("./videos")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------- Debug Config ----------------------
    # Force a specific episode + ego for debugging
    TARGET_EPISODE = "t1118849619700"
    TARGET_EGO_ID = 1859

    base_cfg = {
        "scene": "us-101",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalise": True,
        },
        "action": {"type": "ContinuousAction"},
        "show_trajectories": True,

        # Frequencies / rendering
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 400,
        "screen_height": 150,
        "scaling": 2.0,
        "offscreen_rendering": True,

        # --- KEY DEBUG OVERRIDES ---
        # Use this particular ego from this particular episode
        "ego_vehicle_ID": TARGET_EGO_ID,
        "simulation_period": {
            # NGSimEnv._load_trajectory will use this to fix the episode
            "episode_name": TARGET_EPISODE,
            # Optional: if you later want to start at a specific frame, add:
            # "ego_start_index": 150,
        },

        # Replay chunk root directory
        "episode_root": "highway_env/data/processed_10s",

        # We let NGSimEnv handle replay selection; here we fix it via simulation_period
        "replay_period": None,

        # Surrounding vehicles
        "max_surrounding": 20000,

        # Optional: offset after start (still applied after ego_start_index logic)
        "reset_step_offset": 1,

        # Use closed-loop expert controller that overrides actions
        "expert_test_mode": True,
    }

    # --------------------- Generate 1 Debug Replay ---------------------
    NUM_REPLAYS = 1

    for i in range(NUM_REPLAYS):
        print(f"\n=== Generating debug video #{i+1}/{NUM_REPLAYS} ===")
        print(f"Episode: {TARGET_EPISODE}, Ego ID: {TARGET_EGO_ID}")

        # Build the environment, passing config DIRECTLY into __init__
        env = gym.make(
            "NGSim-US101-v0",
            render_mode="rgb_array",
            config=base_cfg,
        )

        # Video recording
        env = RecordVideo(
            env,
            video_folder=out_dir,
            episode_trigger=lambda ep_idx: True,
            name_prefix=f"ngsim_{TARGET_EPISODE}_ego{TARGET_EGO_ID}",
        )

        # Reset (NGSimEnv will now use that specific episode + ego)
        obs, info = env.reset(seed=0)

        # In expert_test_mode, the env ignores the action you pass and uses tracker output,
        # so we can just send zeros (or anything).
        STEPS = 450
        for t in range(STEPS):
            # Dummy action: will be overridden internally by expert tracker
            a = env.action_space.sample()  # or np.zeros_like(env.action_space.sample())
            obs, r, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                print(f"Terminated at step {t}")
                break

        env.close()
        print(f"✓ Saved video {i+1} to {out_dir}")

    print(f"\nAll {NUM_REPLAYS} videos written to: {out_dir}")


if __name__ == "__main__":
    main()
