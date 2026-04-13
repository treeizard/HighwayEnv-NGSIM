#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import build_expert_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an expert trajectory dataset from the repo's real NGSIM replay pipeline."
    )
    parser.add_argument(
        "--episode-root",
        type=str,
        default="highway_env/data/processed_20s",
        help="Processed trajectory root containing <scene>/prebuilt/*.npy.",
    )
    parser.add_argument(
        "--prebuilt-split",
        choices=["train", "val"],
        default="train",
        help="Which prebuilt episode split to replay from.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="us-101",
        help="Scene name available under the processed trajectory root.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="train",
        help="Metadata label for the source split.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Action convention to export.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="expert_data/ngsim_expert_dataset_discrete.npz",
        help="Output dataset path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=32,
        help="Number of replay scenarios to sample.",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="Optional fixed episode name for targeted replay or benchmarking.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=None,
        help="Optional cap on collected steps per scenario.",
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=1,
        help="Number of expert-controlled vehicles to replay per scenario.",
    )
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        help="Control every valid vehicle in each traffic segment together.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["per_vehicle", "scene"],
        default="per_vehicle",
        help="Save either one episode per vehicle or one episode per full scene.",
    )
    parser.add_argument(
        "--max-surrounding",
        default="all",
        help="Surrounding vehicles to spawn. Use 'all' for full replay context.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for scenario shuffling and env resets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_expert_dataset(
        scene=args.scene,
        action_mode=args.action_mode,
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        output_path=args.out,
        num_episodes=args.episodes,
        fixed_episode_name=args.episode_name,
        max_horizon=args.max_horizon,
        controlled_vehicles=args.controlled_vehicles,
        control_all_vehicles=args.control_all_vehicles,
        dataset_mode=args.dataset_mode,
        max_surrounding=args.max_surrounding,
        source_split=args.source_split,
        seed=args.seed,
    )

    metadata = result["metadata"]
    print(f"Saved expert dataset to: {result['output_path']}")
    print(f"dataset_episodes={metadata['num_dataset_episodes']}")
    print(
        f"scene={metadata['scene']} action_mode={metadata['action_mode']} "
        f"dataset_mode={metadata.get('dataset_mode', 'per_vehicle')}"
    )
    print(f"observation_shape={tuple(metadata['observation_shape'])}")
    print(f"action_shape={tuple(metadata['action_shape'])}")


if __name__ == "__main__":
    main()
