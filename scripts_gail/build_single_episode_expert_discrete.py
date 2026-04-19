#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import build_expert_dataset


def _resolve_episode_name(episode_root: str, scene: str, prebuilt_split: str) -> str:
    split_dir = Path(episode_root) / scene / prebuilt_split
    if not split_dir.exists():
        raise FileNotFoundError(f"Episode split directory not found: {split_dir}")

    episode_names = sorted(
        path.name
        for path in split_dir.iterdir()
        if path.is_dir() and path.name.startswith("t")
    )
    if not episode_names:
        raise RuntimeError(f"No episode folders found under {split_dir}")
    return episode_names[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a discrete expert-action dataset from a single fixed 20s NGSIM "
            "episode in the selected split."
        )
    )
    parser.add_argument(
        "--episode-root",
        type=str,
        default="highway_env/data/processed_20s",
        help="Processed trajectory root containing <scene>/<split>/t... and <scene>/prebuilt/*.npy.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="us-101",
        help="Scene name under the processed trajectory root.",
    )
    parser.add_argument(
        "--prebuilt-split",
        choices=["train", "val", "test"],
        default="train",
        help="Which prebuilt split to sample from.",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help=(
            "Optional fixed episode folder name such as t1118846663000. "
            "If omitted, the earliest episode in the split is used."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="expert_data/ngsim_single_train_episode_discrete.npz",
        help="Output dataset path.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=None,
        help="Optional cap on collected steps for the chosen 20s episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for scenario selection and env reset.",
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=128,
        help="Number of lidar cells used for observations.",
    )
    parser.add_argument(
        "--maximum-range",
        type=float,
        default=64.0,
        help="Lidar maximum range in meters.",
    )
    parser.add_argument(
        "--max-surrounding",
        default="all",
        help="How many surrounding vehicles to spawn. Use 'all' for full replay context.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["per_vehicle", "scene"],
        default="scene",
        help=(
            "Save either one per-vehicle replay episode or one full-scene replay with "
            "all viable prebuilt vehicle ids for the selected time interval."
        ),
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=1,
        help=(
            "Number of controlled vehicles when dataset_mode=per_vehicle. "
            "Ignored when --dataset-mode scene is used with --control-all-vehicles."
        ),
    )
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        default=True,
        help=(
            "When enabled with --dataset-mode scene, replay all viable vehicle ids "
            "for the chosen time interval together."
        ),
    )
    parser.add_argument(
        "--no-control-all-vehicles",
        dest="control_all_vehicles",
        action="store_false",
        help="Disable all-vehicle scene replay and use only --controlled-vehicles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_name = args.episode_name or _resolve_episode_name(
        episode_root=args.episode_root,
        scene=args.scene,
        prebuilt_split=args.prebuilt_split,
    )

    observation_config = {
        "type": "LidarCameraObservations",
        "lidar": {
            "cells": int(args.cells),
            "maximum_range": float(args.maximum_range),
            "normalize": True,
        },
        "camera": {
            "cells": 21,
            "maximum_range": float(args.maximum_range),
            "field_of_view": np.pi / 2,
            "normalize": True,
        },
    }

    dataset_mode = str(args.dataset_mode)
    control_all_vehicles = bool(args.control_all_vehicles)
    controlled_vehicles = int(args.controlled_vehicles)
    if dataset_mode == "scene" and not control_all_vehicles and controlled_vehicles <= 1:
        controlled_vehicles = 2

    build_kwargs = dict(
        scene=args.scene,
        action_mode="discrete",
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        output_path=args.out,
        num_episodes=1,
        fixed_episode_name=episode_name,
        max_horizon=args.max_horizon,
        controlled_vehicles=controlled_vehicles,
        control_all_vehicles=control_all_vehicles,
        dataset_mode=dataset_mode,
        max_surrounding=args.max_surrounding,
        source_split=args.prebuilt_split,
        seed=args.seed,
        observation_config=observation_config,
    )

    result = build_expert_dataset(**build_kwargs)
    metadata = result["metadata"]
    print(f"Saved discrete expert dataset to: {result['output_path']}")
    print(f"episode_name={episode_name}")
    print(f"source_split={metadata['source_split']}")
    print(f"dataset_mode={metadata['dataset_mode']}")
    print(f"dataset_episodes={metadata['num_dataset_episodes']}")
    print(
        "controlled_vehicles_per_scenario="
        f"{metadata['controlled_vehicles_per_scenario']} "
        f"control_all_vehicles={metadata['control_all_vehicles']}"
    )
    print(f"observation_shape={tuple(metadata['observation_shape'])}")
    print(f"action_shape={tuple(metadata['action_shape'])}")


if __name__ == "__main__":
    main()
