import os
import sys
import numpy as np

# Ensure highway_env is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from highway_env.ngsim_utils.trajectory_gen import (
    build_all_trajectories_for_scene,
)


def main():
    episode_root = "highway_env/data/processed_10s"
    scene = "us-101"
    train_val_div = "train"

    # Build all trajectories
    traj_all_by_episode = build_all_trajectories_for_scene(
        scene=scene,
        episodes_root=episode_root,
        train_val_div=train_val_div,
    )

    valid_ids_by_episode = {}
    episodes = sorted(traj_all_by_episode.keys())

    # Threshold: must be present (non-zero) for at least 60% of the episode
    presence_frac_threshold = 0.6

    for ep_name, veh_dict in traj_all_by_episode.items():
        valid_ids = []
        for vid, meta in veh_dict.items():
            traj = meta["trajectory"]

            # Skip trivial trajectories
            if traj.shape[0] < 2:
                continue

            total_steps = traj.shape[0]

            # Non-zero presence mask over (x, y, v) columns
            nonzero = np.any(traj[:, :3] != 0.0, axis=1)
            nonzero_count = int(nonzero.sum())

            # Require at least 60% of the episode to be "active" (non-zero)
            if nonzero_count / float(total_steps) >= presence_frac_threshold:
                valid_ids.append(vid)

        valid_ids_by_episode[ep_name] = valid_ids

    # Build output path
    out_dir = os.path.join(episode_root, scene, f"prebuilt")
    os.makedirs(out_dir, exist_ok=True)

    out_path_id = os.path.join(out_dir,  f"veh_ids_{train_val_div}.npy")
    out_path_traj = os.path.join(out_dir, f"trajectory_{train_val_div}.npy")
    # Save dictionary for fast loading
    np.save(out_path_id, valid_ids_by_episode)
    print(f"Saved {len(episodes)} episodes to {out_path_id}")
    np.save(out_path_traj, traj_all_by_episode)
    print(f"Saved {len(episodes)} episodes to {out_path_traj}")

if __name__ == "__main__":
    main()
