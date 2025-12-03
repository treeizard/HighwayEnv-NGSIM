# scripts/make_videos_multi.py
import os, sys, random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register

# Make sure project root is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Register NGSimEnv
register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def main():
    out_dir = os.path.abspath("./videos")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------- Base Config ----------------------
    base_cfg = {
        "scene": "us-101",
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction"},
        "show_trajectories": False,

        # Rendering
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "screen_width": 400,
        "screen_height": 150,
        "scaling": 2.0,
        "offscreen_rendering": True,

        # Ego vehicle override (None → auto-random by new NGSimEnv)
        "ego_vehicle_ID": None,

        # Replay chunk root directory
        "episode_root": "highway_env/data/processed_10s",

        # Random episode selection each reset
        "replay_period": None,

        # How many vehicles to spawn
        "max_surrounding": 60,

        # Optional: start the replay slightly after beginning
        "reset_step_offset": 1,
    }

    # --------------------- Generate 5 Random Replays ---------------------
    NUM_REPLAYS = 5

    for i in range(NUM_REPLAYS):
        print(f"\n=== Generating video #{i+1}/{NUM_REPLAYS} ===")

        # Build the environment
        env = gym.make("NGSim-US101-v0", render_mode="rgb_array")

        # Apply base config (random chunk + random ego handled inside env)
        env.unwrapped.configure(dict(base_cfg))

        # Activate video recording
        env = RecordVideo(
            env,
            video_folder=out_dir,
            episode_trigger=lambda ep_idx: True,
            name_prefix=f"ngsim_random_{i}"
        )

        # Reset (this triggers random chunk + random ego)
        obs, info = env.reset(seed=i)

        # Roll out using expert policy until it ends or safety cap
        STEPS = 2000  # big safety cap, truncation will usually fire earlier

        for k in range(STEPS):
            # Expert continuous action [steering, accel]
            a = env.unwrapped.expert_action_at()  # uses env.unwrapped.steps
            a_rand = env.action_space.sample()

            print('actions: ', a, a_rand)
            obs, r, terminated, truncated, info = env.step(a)

            if terminated or truncated:
                print(
                    f"Episode ended at policy step {env.unwrapped.steps} "
                    f"(terminated={terminated}, truncated={truncated})"
                )
                break


        env.close()
        print(f"✓ Saved video {i+1} to {out_dir}")

    print(f"\n🎉 All {NUM_REPLAYS} videos written to: {out_dir}")


if __name__ == "__main__":
    main()
