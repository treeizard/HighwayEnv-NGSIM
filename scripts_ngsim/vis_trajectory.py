import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root so `highway_env` is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from highway_env.ngsim_utils.trajectory_gen import (
    build_all_trajectories_for_scene,
    process_raw_trajectory,
)


def plot_ngsim_episode(
    scene: str = "us-101",
    episodes_root: str = "highway_env/data/processed_10s",
    episode_name: str | None = None,
) -> None:
    # Load all episodes
    all_episodes = build_all_trajectories_for_scene(scene, episodes_root)
    if not all_episodes:
        raise RuntimeError(f"No episodes found under {episodes_root}/{scene}")

    # Choose episode: given or first one
    ep_names = sorted(all_episodes.keys())
    if episode_name is None:
        episode_name = ep_names[0]
    elif episode_name not in all_episodes:
        raise ValueError(
            f"Requested episode '{episode_name}' not found. "
            f"Available episodes include: {ep_names[:5]}"
        )

    episode_trajs = all_episodes[episode_name]
    if not episode_trajs:
        raise RuntimeError(f"Episode '{episode_name}' has no trajectories.")

    plt.figure(figsize=(10, 4))

    for vid, rec in episode_trajs.items():
        traj_ft = rec["trajectory"]              # [x_ft, y_ft, spd, lane]
        traj_m = process_raw_trajectory(traj_ft) # [s_m, r_m, spd_mps, lane]
        arr = np.asarray(traj_m, dtype=float)

        if arr.ndim != 2 or arr.shape[0] == 0:
            continue

        # Drop pure padding rows (exact zeros)
        mask = ~np.all(arr[:, :3] == 0.0, axis=1)
        arr = arr[mask]
        if arr.shape[0] == 0:
            continue

        s = arr[:, 0]  # longitudinal (m)
        r = arr[:, 1]  # lateral (m)

        plt.plot(s, r, linewidth=0.7, alpha=0.4)

    plt.xlabel("Longitudinal position [m]")
    plt.ylabel("Lateral position [m]")
    plt.title(f"NGSIM {scene} trajectories — episode {episode_name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example: plot the first episode
    plot_ngsim_episode(
        scene="us-101",
        episodes_root="highway_env/data/processed_10s",
        episode_name=None,  # or e.g. "t1118846663000"
    )
