import os
import sys
import numpy as np

# Ensure highway_env is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from highway_env.ngsim_utils.trajectory_gen import (
    build_all_trajectories_for_scene,
    trajectory_has_min_continuous_occupancy,
)


def build_prebuilt_split(
    episode_root: str,
    scene: str,
    train_val_div: str,
    presence_frac_threshold: float = 0.8,
    car_only: bool = True,
    car_length_min: float = 10.0,
    car_length_max: float = 22.0,
    car_width_min: float = 4.0,
    car_width_max: float = 8.0,
):
    # Build all trajectories
    traj_all_by_episode = build_all_trajectories_for_scene(
        scene=scene,
        episodes_root=episode_root,
        train_val_div=train_val_div,
    )

    valid_ids_by_episode = {}
    episodes = sorted(traj_all_by_episode.keys())

    for ep_name, veh_dict in traj_all_by_episode.items():
        valid_ids = []
        for vid, meta in veh_dict.items():
            traj = meta["trajectory"]
            length = float(meta.get("length", 0.0))
            width = float(meta.get("width", 0.0))

            if car_only and not (
                car_length_min <= length <= car_length_max
                and car_width_min <= width <= car_width_max
            ):
                continue

            # Skip trivial trajectories
            if traj.shape[0] < 2:
                continue

            if trajectory_has_min_continuous_occupancy(
                traj,
                min_presence_ratio=presence_frac_threshold,
            ):
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


def main():
    episode_root = "highway_env/data/processed_20s"
    scene = "us-101"

    for train_val_div in ("train", "val", "test"):
        print(f"Building prebuilt data for split: {train_val_div}")
        build_prebuilt_split(
            episode_root=episode_root,
            scene=scene,
            train_val_div=train_val_div,
        )


if __name__ == "__main__":
    main()
