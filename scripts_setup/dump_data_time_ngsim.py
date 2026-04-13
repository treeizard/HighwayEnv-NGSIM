# Modified by: Yide Tao (yide.tao@monash.edu), Huy Nguyen, Ted Sing Lo
# Reference: @article{huang2021driving,
#   title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
#   author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
#   journal={IEEE Transactions on Intelligent Transportation Systems},
#   year={2021},
#   publisher={IEEE}
# }
# @misc{highway-env,
#   author = {Leurent, Edouard},
#   title = {An Environment for Autonomous Driving Decision-Making},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/eleurent/highway-env}},
# }
# dump_data_time.py
#
# Read an NGSIM CSV (e.g. us-101) using ngsim_data, then dump the data
# into 20-second (or configurable) time windows. This file is constructed to improve the data processing efficiency later on
#
# For example, with default settings:
#    highway_env/data/processed_20s/us-101/
#        train/
#            t1118846663000/
#                vehicle_record_file.csv
#                vehicle_file.csv
#                snapshot_file.csv
#            ...
#        val/
#            t1118846673000/
#                ...
#
# Each subfolder contains the same three files and formats as ngsim_data.dump(),
# but restricted to the snapshots and vehicles that appear in that time window.

import os
import argparse
from collections import defaultdict
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from highway_env.data.ngsim import ngsim_data
#from highway_env.data.traj_to_action import traj_cont_action


def build_episodes(
    reader: ngsim_data,
    episode_len_ms: int,
    stride_ms: int,
):
    """
    Build a list of episodes, each described by:
        (episode_start_time, [snap_times_in_episode])

    Non-overlapping by default if stride_ms == episode_len_ms.
    """
    # Make sure we have ordered snapshot times
    if not reader.snap_ordered_list:
        reader.snap_ordered_list = sorted(reader.snap_dict.keys())

    times = reader.snap_ordered_list
    if len(times) == 0:
        return []

    episodes = []
    i = 0
    n = len(times)

    while i < n:
        t_start = times[i]
        t_end = t_start + episode_len_ms

        # Collect all snapshot times in [t_start, t_end)
        snap_times = []
        j = i
        while j < n and times[j] < t_end:
            snap_times.append(times[j])
            j += 1

        if len(snap_times) > 0:
            episodes.append((t_start, snap_times))

        # Move to next episode start
        # If stride == episode length, this is non-overlapping
        # Move forward until we hit the stride boundary
        next_start_time = t_start + stride_ms
        # Find the smallest index with times[idx] >= next_start_time
        while i < n and times[i] < next_start_time:
            i += 1

    return episodes


def dump_episode(
    reader: ngsim_data,
    snap_times,
    out_dir: str,
    *,
    car_only: bool = False,
    car_length_min: float = 10.0,
    car_length_max: float = 22.0,
    car_width_min: float = 4.0,
    car_width_max: float = 8.0,
):
    """
    Dump one episode (subset of time) in the same format as ngsim_data.dump(),
    but restricted to:
        - vehicle_record_file.csv : only vehicle_records whose unixtime is in snap_times
        - snapshot_file.csv       : only snapshots for snap_times
        - vehicle_file.csv        : only vehicles that appear in snap_times,
                                    and only vr IDs from those times
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) Collect all vehicle_records (vr) in this episode
    episode_vr_ids = []
    episode_vr_by_id = {}

    for t in snap_times:
        snap = reader.snap_dict[t]
        for vr in snap.vr_list:
            if car_only:
                vr_length = float(getattr(vr, "len", 0.0))
                vr_width = float(getattr(vr, "wid", 0.0))
                if not (
                    car_length_min <= vr_length <= car_length_max
                    and car_width_min <= vr_width <= car_width_max
                ):
                    continue
            if vr.ID not in episode_vr_by_id:
                episode_vr_by_id[vr.ID] = vr
                episode_vr_ids.append(vr.ID)

    # Sort vr IDs for deterministic order
    episode_vr_ids.sort()

    # 2) Build vehicle -> [vr] mapping restricted to this episode
    veh_to_vrs = defaultdict(list)
    for vr_id in episode_vr_ids:
        vr = episode_vr_by_id[vr_id]
        veh_to_vrs[vr.veh_ID].append(vr)

    # Sort each vehicle's vrs in time
    for veh_id in veh_to_vrs:
        veh_to_vrs[veh_id].sort(key=lambda v: v.unixtime)

    # 3) Write vehicle_record_file.csv (same format as vehicle_record.to_string())
    vr_path = os.path.join(out_dir, "vehicle_record_file.csv")
    with open(vr_path, "w", encoding="utf-8") as f_vr:
        for vr_id in episode_vr_ids:
            vr = episode_vr_by_id[vr_id]
            f_vr.write(vr.to_string() + "\n")

    # 4) Write vehicle_file.csv (veh_ID, vr_ID, vr_ID, ...)
    v_path = os.path.join(out_dir, "vehicle_file.csv")
    with open(v_path, "w", encoding="utf-8") as f_v:
        for veh_id in sorted(veh_to_vrs.keys()):
            vr_ids = [vr.ID for vr in veh_to_vrs[veh_id]]
            line = ",".join([str(veh_id)] + [str(vr_id) for vr_id in vr_ids])
            f_v.write(line + "\n")

    # 5) Write snapshot_file.csv (unixtime, vr_ID, vr_ID, ...)
    ss_path = os.path.join(out_dir, "snapshot_file.csv")
    with open(ss_path, "w", encoding="utf-8") as f_ss:
        for t in sorted(snap_times):
            snap = reader.snap_dict[t]
            # Only include vr IDs that are in this episode
            vr_ids = [vr.ID for vr in snap.vr_list if vr.ID in episode_vr_by_id]
            if len(vr_ids) == 0:
                continue
            line = ",".join([str(t)] + [str(vr_id) for vr_id in vr_ids])
            f_ss.write(line + "\n")


def split_episodes_by_time(
    episodes,
    val_ratio: float,
    test_ratio: float,
):
    """
    Split episode windows into consecutive train/val/test segments ordered by time.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val_ratio must be in [0, 1).")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("--test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val_ratio + --test_ratio must be < 1.0.")

    ordered = sorted(episodes, key=lambda item: item[0])
    n_total = len(ordered)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = min(max(n_test, 0), n_total)
    n_val = min(max(n_val, 0), n_total - n_test)
    n_train = n_total - n_val - n_test

    train_episodes = ordered[:n_train]
    val_episodes = ordered[n_train : n_train + n_val]
    test_episodes = ordered[n_train + n_val :]
    return train_episodes, val_episodes, test_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Dump NGSIM data into 20-second (or custom) time windows."
    )
    parser.add_argument(
        "path",
        help="Path to the raw NGSIM CSV file (e.g. us-101).",
    )
    parser.add_argument(
        "--scene",
        help="Location / scene name (used by ngsim_data and output path).",
        default="us-101",
    )
    parser.add_argument(
        "--out_root",
        help="Root folder to store time-windowed processed data.",
        default="highway_env/data/processed_20s",
    )
    parser.add_argument(
        "--episode_len_sec",
        type=float,
        default=20.0,
        help="Episode length in seconds (default: 20.0).",
    )
    parser.add_argument(
        "--stride_sec",
        type=float,
        default=20.0,
        help="Stride between episode starts in seconds (default: 20.0 = non-overlapping).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of episodes assigned to validation from the middle time segment.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of episodes assigned to test from the latest time segment.",
    )
    parser.add_argument(
        "--car_only",
        action="store_true",
        help="Keep only car-sized vehicles when writing episode folders.",
    )
    parser.add_argument(
        "--car_length_min",
        type=float,
        default=10.0,
        help="Minimum vehicle length in feet for the US/NGSIM car filter.",
    )
    parser.add_argument(
        "--car_length_max",
        type=float,
        default=22.0,
        help="Maximum vehicle length in feet for the US/NGSIM car filter.",
    )
    parser.add_argument(
        "--car_width_min",
        type=float,
        default=4.0,
        help="Minimum vehicle width in feet for the US/NGSIM car filter.",
    )
    parser.add_argument(
        "--car_width_max",
        type=float,
        default=8.0,
        help="Maximum vehicle width in feet for the US/NGSIM car filter.",
    )

    args = parser.parse_args()

    path = args.path
    scene = args.scene
    out_root = args.out_root

    episode_len_ms = int(args.episode_len_sec * 1000)
    stride_ms = int(args.stride_sec * 1000)

    # 1) Load raw NGSIM and build internal structures
    reader = ngsim_data(scene)
    reader.read_from_csv(path)
    reader.clean()

    # 2) Build episodes
    episodes = build_episodes(reader, episode_len_ms, stride_ms)
    print(
        f"Found {len(episodes)} episodes "
        f"of length {args.episode_len_sec} s (stride {args.stride_sec} s)."
    )

    # 2.5) Chronological split into train / val / test
    train_episodes, val_episodes, test_episodes = split_episodes_by_time(
        episodes=episodes,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    base_dir = os.path.join(out_root, scene)
    os.makedirs(base_dir, exist_ok=True)

    # 3) Dump each episode into split subfolders
    for split_name, split_episodes in [
        ("train", train_episodes),
        ("val", val_episodes),
        ("test", test_episodes),
    ]:
        split_dir = os.path.join(base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for idx, (t_start, snap_times) in enumerate(split_episodes):
            # Folder name: t<start_unixtime>, e.g. t1118846663000
            ep_dir = os.path.join(split_dir, f"t{t_start}")
            print(
                f"[{split_name}] [{idx+1}/{len(split_episodes)}] "
                f"Dumping episode starting at {t_start} "
                f"with {len(snap_times)} snapshots -> {ep_dir}"
            )
            dump_episode(
                reader,
                snap_times,
                ep_dir,
                car_only=args.car_only,
                car_length_min=args.car_length_min,
                car_length_max=args.car_length_max,
                car_width_min=args.car_width_min,
                car_width_max=args.car_width_max,
            )
            #traj_cont_action(ep_dir)

    print(
        f"Chronological split summary: "
        f"{len(train_episodes)} train, {len(val_episodes)} val, {len(test_episodes)} test"
    )
    print("Done.")


if __name__ == "__main__":
    main()
