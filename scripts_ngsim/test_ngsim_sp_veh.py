# scripts/make_videos_multi.py
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register

# Make sure project root is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Register NGSimEnv
register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def plot_trajectory_comparison(env_unwrapped, episode_name, ego_id, out_dir):
    """
    Extracts trajectory data from the unwrapped environment and plots
    Expert (Reference) vs Actual (Simulated) paths.
    """
    # 1. Retrieve data from the env
    # These attributes must exist in NGSimEnv when expert_test_mode=True
    if not hasattr(env_unwrapped, "_expert_ref_xy_pol") or not hasattr(env_unwrapped, "_replay_xy_pol"):
        print("Warning: Trajectory data not found in environment. Skipping plot.")
        return

    ref_traj = np.array(env_unwrapped._expert_ref_xy_pol)
    sim_traj = np.array(env_unwrapped._replay_xy_pol)

    # 2. Setup Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Expert (Ground Truth)
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], 
             color='black', linestyle='--', linewidth=2, alpha=0.5, 
             label='Expert Reference')
    
    # Plot Simulation (Actual)
    plt.plot(sim_traj[:, 0], sim_traj[:, 1], 
             color='blue', linewidth=1.5, alpha=0.8, 
             label='Simulated Trajectory')

    # 3. Add Start/End markers 
    if len(sim_traj) > 0:
        plt.scatter(sim_traj[0, 0], sim_traj[0, 1], color='green', marker='o', label='Start')
        plt.scatter(sim_traj[-1, 0], sim_traj[-1, 1], color='red', marker='x', label='End')

    # 4. Formatting
    plt.title(f"Trajectory Tracking: Episode {episode_name} | Ego {ego_id}")
    plt.xlabel("Longitudinal Position (x) [m]")
    plt.ylabel("Lateral Position (y) [m]")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')  # Crucial to see actual lane change geometry

    # 5. Save
    save_path = os.path.join(out_dir, f"traj_plot_{episode_name}_ego{ego_id}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved trajectory plot to: {save_path}")


def main():
    out_dir = os.path.abspath("./videos")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------- Debug Config ----------------------
    TARGET_EPISODE = "t1118849169700"
    TARGET_EGO_ID = 785

    base_cfg = {
        "scene": "us-101",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "action": {"type": "DiscreteSteerMetaAction"},
        "show_trajectories": True,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 400,
        "screen_height": 150,
        "scaling": 2.0,
        "offscreen_rendering": True,
        "ego_vehicle_ID": TARGET_EGO_ID,
        "simulation_period": {"episode_name": TARGET_EPISODE},
        "episode_root": "highway_env/data/processed_10s",
        "replay_period": None,
        "max_surrounding": 20000,
        "reset_step_offset": 1,
        "expert_test_mode": True,
        "expert_action_mode": "discrete"
    }

    # --------------------- Generate 1 Debug Replay ---------------------
    NUM_REPLAYS = 1

    for i in range(NUM_REPLAYS):
        print(f"\n=== Generating debug video #{i+1}/{NUM_REPLAYS} ===")
        print(f"Episode: {TARGET_EPISODE}, Ego ID: {TARGET_EGO_ID}")

        env = gym.make(
            "NGSim-US101-v0",
            render_mode="rgb_array",
            config=base_cfg,
        )

        env = RecordVideo(
            env,
            video_folder=out_dir,
            episode_trigger=lambda ep_idx: True,
            name_prefix=f"ngsim_{TARGET_EPISODE}_ego{TARGET_EGO_ID}",
        )

        obs, info = env.reset(seed=0)

        STEPS = 450
        for t in range(STEPS):
            # In expert_test_mode, actual action input is ignored
            a = 0 
            obs, r, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                print(f"Terminated at step {t}")
                break

        # --- NEW: Generate Plot before closing ---
        # We access the internal environment via env.unwrapped
        plot_trajectory_comparison(env.unwrapped, TARGET_EPISODE, TARGET_EGO_ID, out_dir)

        env.close()
        print(f"✓ Saved video {i+1} to {out_dir}")

    print(f"\nAll {NUM_REPLAYS} videos and plots written to: {out_dir}")


if __name__ == "__main__":
    main()