# scripts_ngsim/test_ngsim_expert_videos_both_same_episode_ego.py
import os, sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register

# Make sure project root is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Register NGSimEnv
register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def _print_action_mapping(env):
    print("Action space:", env.action_space)
    try:
        unwrapped = env.unwrapped
        at = getattr(unwrapped, "action_type", None)
        if at is not None and hasattr(at, "actions"):
            print("Action mapping (index -> label):", at.actions)
        if at is not None and hasattr(at, "actions_indexes"):
            print("Action mapping (label -> index):", at.actions_indexes)
        print("ActionType class:", type(at).__name__ if at is not None else None)
    except Exception as e:
        print("Could not introspect action mappings:", repr(e))


def _assert_same_episode_and_ego(env, target_episode: str, target_ego_id: int):
    """
    Best-effort validation: relies on NGSimEnv exposing episode_name and ego_id on unwrapped.
    """
    unwrapped = env.unwrapped
    ep = getattr(unwrapped, "episode_name", None)
    ego = getattr(unwrapped, "ego_id", None)
    print(f"[DEBUG] episode_name={ep!r}, ego_id={ego!r}")

    if ep is not None and ep != target_episode:
        raise RuntimeError(f"Episode mismatch: expected {target_episode!r}, got {ep!r}")
    if ego is not None and int(ego) != int(target_ego_id):
        raise RuntimeError(f"Ego ID mismatch: expected {target_ego_id}, got {ego}")


def _run_one(env, steps: int, mode_name: str, trajectory_save_path: str = None):
    """
    Run one rollout and optionally save the full position history to a .npz file.
    """
    obs, info = env.reset(seed=0)

    taken_labels = []
    taken_idxs = []

    print(f"\n[{mode_name}] Starting simulation loop...")

    for t in range(int(steps)):
        if isinstance(env.action_space, gym.spaces.Discrete):
            a = 0
        else:
            a = np.array([0.0, 0.0], dtype=np.float32)

        # Step the environment
        obs, r, terminated, truncated, info = env.step(a)

        # --- OPTIONAL: Live Tracking Print (every 10 steps) ---
        # Useful to see "different positions at different times" in the console
        if t % 10 == 0:
            veh_pos = env.unwrapped.vehicle.position
            print(f"  Step {t}: Pos=({veh_pos[0]:.2f}, {veh_pos[1]:.2f})")

        # Capture Discrete Actions for Debugging
        if isinstance(info, dict):
            if "expert_action_discrete" in info:
                taken_labels.append(str(info["expert_action_discrete"]))
            if "expert_action_discrete_idx" in info:
                taken_idxs.append(int(info["expert_action_discrete_idx"]))
            if (
                "applied_action" in info
                and isinstance(info["applied_action"], (int, np.integer))
                and isinstance(env.action_space, gym.spaces.Discrete)
            ):
                taken_idxs.append(int(info["applied_action"]))

        if terminated or truncated:
            print(f"[{mode_name}] Episode ended at step {t} (terminated={terminated}, truncated={truncated})")
            break

    # --- Summary Counts ---
    if taken_labels:
        labels, counts = np.unique(np.asarray(taken_labels, dtype=str), return_counts=True)
        print(f"\n[{mode_name}] Discrete label counts:")
        for lab, c in sorted(zip(labels.tolist(), counts.tolist()), key=lambda x: -x[1]):
            print(f"  {lab:>12s}: {c}")

    if taken_idxs:
        uniq, cnt = np.unique(np.asarray(taken_idxs, dtype=int), return_counts=True)
        print(f"\n[{mode_name}] Discrete index counts:")
        for u, c in zip(uniq.tolist(), cnt.tolist()):
            print(f"  idx {u:>2d}: {c}")

    # --- Metrics ---
    try:
        metrics = env.unwrapped.expert_replay_metrics()
        print(f"\n[{mode_name}] Expert replay metrics:", metrics)
    except Exception as e:
        print(f"\n[{mode_name}] expert_replay_metrics() unavailable/failed:", repr(e))

    # --- NEW: Save Trajectory Data for Analysis ---
    if trajectory_save_path:
        try:
            unwrapped = env.unwrapped
            data_to_save = {}

            # 1. Simulated Trajectory (The Ego Vehicle)
            if hasattr(unwrapped, "_replay_xy_pol"):
                # Convert to numpy for easy saving
                sim_traj = np.array(unwrapped._replay_xy_pol)
                data_to_save["sim_traj"] = sim_traj
                print(f"[{mode_name}] Captured Simulated Trajectory: {sim_traj.shape} points")

            # 2. Reference Trajectory (The Expert Ghost)
            if hasattr(unwrapped, "_expert_ref_xy_pol"):
                ref_traj = np.array(unwrapped._expert_ref_xy_pol)
                data_to_save["ref_traj"] = ref_traj
                print(f"[{mode_name}] Captured Reference Trajectory: {ref_traj.shape} points")

            # Save to .npz
            np.savez(trajectory_save_path, **data_to_save)
            print(f"[{mode_name}] >> FULL TRAJECTORY SAVED TO: {trajectory_save_path}")
            print(f"[{mode_name}] You can load this with: data = np.load('{trajectory_save_path}'); sim=data['sim_traj']; ref=data['ref_traj']")

        except Exception as e:
            print(f"[{mode_name}] Failed to save trajectory data: {e}")


def main():
    # ---------------------- Target episode + ego ----------------------
    TARGET_EPISODE = "t1118849169700"
    TARGET_EGO_ID = 785
    
    # Output directory
    out_dir = os.path.abspath("./videos_discrete_test")
    os.makedirs(out_dir, exist_ok=True)

    DISCRETE_ACTION_TYPE_NAME = "DiscreteSteerMetaAction"
    MAX_SURROUNDING = 40000000

    # ---------------------- Common Config ----------------------
    common_cfg = {
        "scene": "us-101",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "show_trajectories": True,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 400,
        "screen_height": 150,
        "scaling": 2.0,
        "offscreen_rendering": True,
        "episode_root": "highway_env/data/processed_10s",
        "replay_period": None,
        "ego_vehicle_ID": TARGET_EGO_ID,
        "simulation_period": {"episode_name": TARGET_EPISODE},
        "max_surrounding": MAX_SURROUNDING,
        "expert_test_mode": True,
        "expert_speed_deadband_mps": 0.5,
        "expert_steer_deadband_rad": 0.05,
        "expert_one_action_per_step": True,
        "expert_prefer_speed": False,
    }

    # ---------------------------
    # 1) Continuous expert run
    # ---------------------------
    cfg_cont = dict(common_cfg)
    cfg_cont.update({
        "action": {"type": "ContinuousAction"},
        "expert_action_mode": "continuous",
    })

    env_cont = gym.make("NGSim-US101-v0", render_mode="rgb_array", config=cfg_cont)
    env_cont = RecordVideo(
        env_cont,
        video_folder=out_dir,
        episode_trigger=lambda ep_idx: True,
        name_prefix=f"ngsim_{TARGET_EPISODE}_ego{TARGET_EGO_ID}_expert_continuous",
    )

    print("\n=== Continuous expert run ===")
    _print_action_mapping(env_cont)
    obs, info = env_cont.reset(seed=0)
    _assert_same_episode_and_ego(env_cont, TARGET_EPISODE, TARGET_EGO_ID)
    
    # Save trajectory to .npz
    traj_path_cont = os.path.join(out_dir, f"traj_{TARGET_EPISODE}_ego{TARGET_EGO_ID}_continuous.npz")
    _run_one(env_cont, steps=450, mode_name="continuous", trajectory_save_path=traj_path_cont)
    env_cont.close()

    # ---------------------------
    # 2) Discrete expert run
    # ---------------------------
    cfg_disc = dict(common_cfg)
    cfg_disc.update({
        "action": {
            "type": DISCRETE_ACTION_TYPE_NAME,
            "target_speeds": list(np.arange(0.0, 35.0 + 1e-6, 2.0)),
        },
        "expert_action_mode": "discrete",
    })

    env_disc = gym.make("NGSim-US101-v0", render_mode="rgb_array", config=cfg_disc)
    env_disc = RecordVideo(
        env_disc,
        video_folder=out_dir,
        episode_trigger=lambda ep_idx: True,
        name_prefix=f"ngsim_{TARGET_EPISODE}_ego{TARGET_EGO_ID}_expert_discrete",
    )

    print("\n=== Discrete expert run ===")
    _print_action_mapping(env_disc)
    obs, info = env_disc.reset(seed=0)
    _assert_same_episode_and_ego(env_disc, TARGET_EPISODE, TARGET_EGO_ID)

    # Save trajectory to .npz
    traj_path_disc = os.path.join(out_dir, f"traj_{TARGET_EPISODE}_ego{TARGET_EGO_ID}_discrete.npz")
    _run_one(env_disc, steps=450, mode_name="discrete", trajectory_save_path=traj_path_disc)
    env_disc.close()

    print(f"\nSaved videos and trajectory data to: {out_dir}")


if __name__ == "__main__":
    main()