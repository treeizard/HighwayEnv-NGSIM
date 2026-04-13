#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import load_expert_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a saved expert trajectory dataset.")
    parser.add_argument("dataset", type=str, help="Path to the .npz dataset.")
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample transitions to print from the first episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = os.path.abspath(args.dataset)
    data = load_expert_dataset(path)
    metadata = data["metadata"]
    dataset_mode = str(metadata.get("dataset_mode", "per_vehicle"))

    lengths = np.asarray([len(np.asarray(obs)) for obs in data["observations"]], dtype=np.int32)
    total_transitions = int(lengths.sum())
    first_episode_idx = 0
    first_obs = np.asarray(data["observations"][first_episode_idx], dtype=np.float32)
    if metadata["action_mode"] == "discrete":
        first_actions = np.asarray(data["actions"][first_episode_idx], dtype=np.int64)
    else:
        first_actions = np.asarray(data["actions"][first_episode_idx], dtype=np.float32)

    print(f"dataset={path}")
    print(f"schema_version={metadata['schema_version']}")
    print(f"scene={metadata['scene']} action_mode={metadata['action_mode']} dataset_mode={dataset_mode}")
    print(f"dataset_episodes={len(data['episode_id'])}")
    print(f"total_transitions={total_transitions}")
    print(
        "trajectory_length_stats="
        f"min:{int(lengths.min())} mean:{float(lengths.mean()):.2f} median:{float(np.median(lengths)):.2f} max:{int(lengths.max())}"
    )
    if dataset_mode == "scene":
        print(f"agents_per_scene_first_episode={first_obs.shape[1]}")
        print(f"observation_shape={first_obs.shape[2:]}")
        print(f"action_shape={first_actions.shape[2:] if first_actions.ndim > 2 else ()}")
    else:
        print(f"observation_shape={first_obs.shape[1:]}")
        print(f"action_shape={first_actions.shape[1:] if first_actions.ndim > 1 else ()}")
    print(f"action_dtype={first_actions.dtype}")
    print(f"first_episode_name={data['episode_name'][0]}")
    print(f"first_scenario_id={data['scenario_id'][0]}")
    if dataset_mode == "scene":
        print(f"first_agent_ids={np.asarray(data['agent_ids'][0], dtype=np.int32).tolist()[:12]}")
    else:
        print(f"first_ego_id={int(data['ego_id'][0])}")

    sample_count = min(args.samples, len(first_obs))
    print("")
    print(f"Sample transitions from episode 0 ({sample_count} rows)")
    for idx in range(sample_count):
        action = np.asarray(data["actions"][0][idx])
        if dataset_mode == "scene":
            agent_ids = np.asarray(data["agent_ids"][0], dtype=np.int32)
            alive_mask = np.asarray(data["alive_mask"][0][idx], dtype=bool)
            preview = action[: min(5, len(action))].tolist()
            print(
                f"t={int(data['timesteps'][0][idx])} "
                f"done={bool(data['dones'][0][idx])} "
                f"reward={float(data['rewards'][0][idx]):.3f} "
                f"agents={len(agent_ids)} alive={int(alive_mask.sum())} "
                f"action_preview={preview} "
                f"obs_shape={tuple(np.asarray(data['observations'][0][idx]).shape)}"
            )
        else:
            action_repr = action.tolist() if action.ndim > 0 else int(action)
            print(
                f"t={int(data['timesteps'][0][idx])} "
                f"done={bool(data['dones'][0][idx])} "
                f"reward={float(data['rewards'][0][idx]):.3f} "
                f"action={action_repr} "
                f"obs_shape={tuple(np.asarray(data['observations'][0][idx]).shape)}"
            )

    print("")
    print("Replay command for the first episode")
    if dataset_mode == "scene":
        agent_ids = " ".join(str(v) for v in np.asarray(data["agent_ids"][0], dtype=np.int32))
        print(
            "python scripts_test/test_ngsim_discrete.py "
            f"--scene {metadata['scene']} "
            f"--episode-root {metadata['episode_root']} "
            f"--prebuilt-split {metadata.get('prebuilt_split', 'train')} "
            f"--episode-name {data['episode_name'][0]} "
            f"--controlled-vehicles {len(np.asarray(data['agent_ids'][0]))} "
            f"--ego-ids {agent_ids} "
            "--episodes 1 --max-steps 50"
        )
    else:
        print(
            "python scripts_test/test_ngsim_discrete.py "
            f"--scene {metadata['scene']} "
            f"--episode-root {metadata['episode_root']} "
            f"--prebuilt-split {metadata.get('prebuilt_split', 'train')} "
            f"--episode-name {data['episode_name'][0]} "
            f"--ego-ids {int(data['ego_id'][0])} "
            "--episodes 1 --max-steps 50"
        )


if __name__ == "__main__":
    main()
