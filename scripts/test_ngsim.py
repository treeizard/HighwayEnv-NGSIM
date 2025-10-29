# scripts/make_videos_multi.py
import os, sys, gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register

# Make sure the project root is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# If you've already registered elsewhere, you can remove this.
register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")

def main():
    out_dir = os.path.abspath("./videos")
    os.makedirs(out_dir, exist_ok=True)

    # ------------ Base env config (matches the new NGSimEnv) ------------
    base_cfg = {
        # Core env
        "scene": "us-101",
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction"},
        "show_trajectories": False,

        # Frequencies / rendering
        "simulation_frequency": 15,   # 15 Hz physics
        "policy_frequency": 5,        # 5 Hz agent decisions
        "screen_width": 400, "screen_height": 150, "scaling": 2.0,
        "offscreen_rendering": True,  # use rgb_array for video capture

        # Ego spawn (you can tweak these)
        "ego_speed": 30.0,            # m/s
        "ego_lane_index": 1,          # on ("s1","s2")
        "ego_longitudinal_m": 30.0,

        # Replay controls in the new env:
        # - replay_period: which precomputed period (int) to use from build_trajectory()
        # - reset_step_offset: which frame to start from in each replay sequence
        "replay_period": 0,
        "reset_step_offset": 1,

        # Surrounding spawn limits
        "spawn_radius_m": 120.0,
        "max_surrounding": 60,
    }

    # ------------ Choose periods & offsets for multiple videos ------------
    # Tip: periods are dataset-dependent; 0..4 usually exist for NGSIM cuts.
    # Offsets are "frame indices" into each processed trajectory.
    PERIODS = [0, 1, 2, 3, 4]        # try several replay windows
    OFFSETS = [50] # start a bit into the slice to avoid empty steps

    # Pair them cyclically to produce N videos
    NUM_VIDEOS = min(len(PERIODS) * len(OFFSETS), 12)  # cap for a short run
    pairs = []
    pi = oi = 0
    for _ in range(NUM_VIDEOS):
        pairs.append((PERIODS[pi], OFFSETS[oi]))
        oi = (oi + 1) % len(OFFSETS)
        if oi == 0:
            pi = (pi + 1) % len(PERIODS)

    # ------------ Roll out & record episodes ------------
    for ep, (period, offset) in enumerate(pairs):
        # Build a fresh env every episode to apply new config
        env = gym.make("NGSim-US101-v0", render_mode="rgb_array")
        cfg = dict(base_cfg)
        cfg["replay_period"] = int(period)
        cfg["reset_step_offset"] = int(offset)
        env.unwrapped.configure(cfg)

        env = RecordVideo(
            env,
            video_folder=out_dir,
            episode_trigger=lambda ep_idx: True,
            name_prefix=f"ngsim_period{period}_off{offset}"
        )

        obs, info = env.reset(seed=ep)
        # ~30s at 15 Hz; adjust as needed
        STEPS_PER_VIDEO = 450

        for _ in range(STEPS_PER_VIDEO):
            # Random policy just to generate footage; replace with your agent’s action
            a = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break

        env.close()
        print(f"✓ wrote video for period={period}, offset={offset}")

    print(f"✅ Done. Videos are in: {out_dir}")

if __name__ == "__main__":
    main()
