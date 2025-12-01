"""
dump_data_time.py

Read an NGSIM CSV (e.g. us-101) using ngsim_data, then dump the data
into 10-second (or configurable) time windows. This file is constructed to improve the data processing efficiency later on

For example, with default settings:

    highway_env/data/processed_10s/us-101/
        t1118846663000/
            vehicle_record_file.csv
            vehicle_file.csv
            snapshot_file.csv
        t1118846673000/
            ...

Each subfolder contains the same three files and formats as ngsim_data.dump(),
but restricted to the snapshots and vehicles that appear in that time window.
"""

import os
import argparse
from collections import defaultdict

from highway_env.data.ngsim import ngsim_data


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


def dump_episode(reader: ngsim_data, snap_times, out_dir: str):
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


def main():
    parser = argparse.ArgumentParser(
        description="Dump NGSIM data into 10-second (or custom) time windows."
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
        default="/processed_10s",
    )
    parser.add_argument(
        "--episode_len_sec",
        type=float,
        default=10.0,
        help="Episode length in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--stride_sec",
        type=float,
        default=10.0,
        help="Stride between episode starts in seconds (default: 10.0 = non-overlapping).",
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

    # 2) Build 10-second episodes
    episodes = build_episodes(reader, episode_len_ms, stride_ms)
    print(f"Found {len(episodes)} episodes "
          f"of length {args.episode_len_sec} s (stride {args.stride_sec} s).")

    # 3) Dump each episode
    base_dir = os.path.join(out_root, scene)
    os.makedirs(base_dir, exist_ok=True)

    for idx, (t_start, snap_times) in enumerate(episodes):
        # Folder name: t<start_unixtime>, e.g. t1118846663000
        ep_dir = os.path.join(base_dir, f"t{t_start}")
        print(f"[{idx+1}/{len(episodes)}] Dumping episode starting at {t_start} "
              f"with {len(snap_times)} snapshots -> {ep_dir}")
        dump_episode(reader, snap_times, ep_dir)

    print("Done.")


if __name__ == "__main__":
    main()
