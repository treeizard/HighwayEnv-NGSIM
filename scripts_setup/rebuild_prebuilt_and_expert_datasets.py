#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from highway_env.imitation.expert_dataset import (  # noqa: E402
    build_expert_dataset,
    default_observation_config,
)
from scripts_setup.build_data_time import build_prebuilt_split  # noqa: E402


DEFAULT_US_RAW = (
    "raw_data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv"
)
DEFAULT_JP_RAW = "raw_data/morinomiya_filtered_800.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild US and Japanese processed splits, prebuilt replay caches, and "
            "expert datasets using 1/3 train-val-test splits by default."
        )
    )
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--expert-out-root", default="expert_data/rebuilt")
    parser.add_argument("--us-raw-csv", default=DEFAULT_US_RAW)
    parser.add_argument("--japanese-input-npy", default=DEFAULT_JP_RAW)
    parser.add_argument("--us-scene", default="us-101")
    parser.add_argument("--japanese-scene", default="japanese")
    parser.add_argument("--window-sec", type=float, default=20.0)
    parser.add_argument("--val-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--test-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--dataset-mode", choices=["scene", "per_vehicle"], default="scene")
    parser.add_argument("--controlled-vehicles", type=int, default=1)
    parser.add_argument("--control-all-vehicles", action="store_true", default=True)
    parser.add_argument(
        "--no-control-all-vehicles",
        dest="control_all_vehicles",
        action="store_false",
    )
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--maximum-range", type=float, default=64.0)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--presence-ratio-threshold", type=float, default=0.8)
    parser.add_argument("--skip-us-dump", action="store_true")
    parser.add_argument("--skip-us-prebuilt", action="store_true")
    parser.add_argument("--skip-japanese-prebuilt", action="store_true")
    parser.add_argument("--skip-expert", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing scene output folders under --episode-root and --expert-out-root first.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Removed directory: {path}")
    elif path.exists():
        path.unlink()
        print(f"Removed file: {path}")


def clean_outputs(args: argparse.Namespace) -> None:
    episode_root = (PROJECT_ROOT / args.episode_root).resolve()
    expert_root = (PROJECT_ROOT / args.expert_out_root).resolve()
    remove_path(episode_root / args.us_scene)
    remove_path(episode_root / args.japanese_scene)
    remove_path(expert_root)


def rebuild_us_processed_and_prebuilt(args: argparse.Namespace) -> None:
    run_command(
        [
            sys.executable,
            "scripts_setup/dump_data_time_ngsim.py",
            args.us_raw_csv,
            "--scene",
            args.us_scene,
            "--out_root",
            args.episode_root,
            "--episode_len_sec",
            str(args.window_sec),
            "--stride_sec",
            str(args.window_sec),
            "--val_ratio",
            str(args.val_ratio),
            "--test_ratio",
            str(args.test_ratio),
        ]
    )

    if args.skip_us_prebuilt:
        return

    for split in ("train", "val", "test"):
        print(f"Building US prebuilt split: {split}")
        build_prebuilt_split(
            episode_root=args.episode_root,
            scene=args.us_scene,
            train_val_div=split,
            presence_frac_threshold=args.presence_ratio_threshold,
        )


def rebuild_japanese_prebuilt(args: argparse.Namespace) -> None:
    run_command(
        [
            sys.executable,
            "scripts_setup/build_prebuilt_japanese.py",
            "--input_npy",
            args.japanese_input_npy,
            "--episode_root",
            args.episode_root,
            "--scene",
            args.japanese_scene,
            "--window_sec",
            str(int(round(args.window_sec))),
            "--val_ratio",
            str(args.val_ratio),
            "--test_ratio",
            str(args.test_ratio),
            "--presence_ratio_threshold",
            str(args.presence_ratio_threshold),
        ]
    )


def count_valid_scenarios(episode_root: str, scene: str, split: str) -> int:
    path = PROJECT_ROOT / episode_root / scene / "prebuilt" / f"veh_ids_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing prebuilt vehicle-id file: {path}")
    data = np.load(path, allow_pickle=True).item()
    return sum(1 for ego_ids in data.values() if len(ego_ids) > 0)


def build_scene_expert_dataset(args: argparse.Namespace, scene: str, split: str) -> None:
    num_episodes = count_valid_scenarios(args.episode_root, scene, split)
    if num_episodes <= 0:
        print(f"Skipping expert dataset for scene={scene} split={split}: no valid scenarios.")
        return

    out_root = (PROJECT_ROOT / args.expert_out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{scene}_{split}_{args.dataset_mode}_{args.action_mode}.npz"

    observation_config = default_observation_config(
        cells=args.cells,
        maximum_range=args.maximum_range,
    )
    controlled_vehicles = int(args.controlled_vehicles)
    if args.dataset_mode == "scene" and not args.control_all_vehicles and controlled_vehicles <= 1:
        controlled_vehicles = 2

    result = build_expert_dataset(
        scene=scene,
        action_mode=args.action_mode,
        episode_root=args.episode_root,
        prebuilt_split=split,
        output_path=str(out_path),
        num_episodes=num_episodes,
        controlled_vehicles=controlled_vehicles,
        control_all_vehicles=bool(args.control_all_vehicles),
        dataset_mode=args.dataset_mode,
        max_surrounding=args.max_surrounding,
        source_split=split,
        seed=int(args.seed),
        observation_config=observation_config,
    )
    print(
        f"Saved expert dataset: {result['output_path']} "
        f"(scene={scene}, split={split}, episodes={num_episodes})"
    )


def main() -> None:
    args = parse_args()
    if args.clean:
        clean_outputs(args)

    if not args.skip_us_dump:
        rebuild_us_processed_and_prebuilt(args)

    if not args.skip_japanese_prebuilt:
        rebuild_japanese_prebuilt(args)

    if args.skip_expert:
        return

    for scene in (args.us_scene, args.japanese_scene):
        for split in ("train", "val", "test"):
            build_scene_expert_dataset(args, scene=scene, split=split)


if __name__ == "__main__":
    main()
