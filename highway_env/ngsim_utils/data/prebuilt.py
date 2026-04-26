# Modified by: Yide Tao (yide.tao@monash.edu)
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


from __future__ import annotations

import os
from typing import Any

import numpy as np

from highway_env.ngsim_utils.data.trajectory_gen import trajectory_has_min_continuous_occupancy


PrebuiltCache = dict[
    tuple[str, str, str, float],
    tuple[dict[str, np.ndarray], dict[str, dict[Any, Any]], list[str]],
]


def refine_valid_ids_by_episode(
    raw_valid_ids: dict[str, np.ndarray],
    traj_all: dict[str, dict[Any, Any]],
    *,
    min_occupancy: float,
) -> dict[str, np.ndarray]:
    refined: dict[str, np.ndarray] = {}
    for episode_name, veh_dict in traj_all.items():
        candidate_ids = raw_valid_ids.get(episode_name, veh_dict.keys())
        filtered_ids = []
        for veh_id in candidate_ids:
            meta = veh_dict.get(int(veh_id))
            if meta is None:
                continue
            traj = np.asarray(meta.get("trajectory", []), dtype=float)
            if trajectory_has_min_continuous_occupancy(
                traj,
                min_presence_ratio=min_occupancy,
            ):
                filtered_ids.append(int(veh_id))
        refined[episode_name] = np.asarray(filtered_ids, dtype=np.int64)
    return refined


def load_prebuilt_data(
    episode_root: str,
    scene: str,
    prebuilt_split: str,
    *,
    min_occupancy: float,
    cache: PrebuiltCache,
) -> tuple[
    str,
    dict[str, np.ndarray],
    dict[str, dict[Any, Any]],
    list[str],
]:
    episode_root_abs = os.path.abspath(episode_root)
    split = str(prebuilt_split)
    cache_key = (episode_root_abs, scene, split, float(min_occupancy))
    prebuilt_dir = os.path.join(episode_root_abs, scene, "prebuilt")

    cached = cache.get(cache_key)
    if cached is None:
        veh_ids_path = os.path.join(prebuilt_dir, f"veh_ids_{split}.npy")
        traj_path = os.path.join(prebuilt_dir, f"trajectory_{split}.npy")
        if not os.path.exists(veh_ids_path):
            raise FileNotFoundError(
                f"Missing prebuilt vehicle id file for split={split!r}: {veh_ids_path}"
            )
        if not os.path.exists(traj_path):
            raise FileNotFoundError(
                f"Missing prebuilt trajectory file for split={split!r}: {traj_path}"
            )
        raw_valid_ids = np.load(veh_ids_path, allow_pickle=True).item()
        traj_all = np.load(traj_path, allow_pickle=True).item()
        valid_ids = refine_valid_ids_by_episode(
            raw_valid_ids,
            traj_all,
            min_occupancy=min_occupancy,
        )
        episodes = sorted(traj_all.keys())
        cached = (valid_ids, traj_all, episodes)
        cache[cache_key] = cached

    valid_ids_by_episode, traj_all_by_episode, episodes = cached
    return prebuilt_dir, valid_ids_by_episode, traj_all_by_episode, episodes
