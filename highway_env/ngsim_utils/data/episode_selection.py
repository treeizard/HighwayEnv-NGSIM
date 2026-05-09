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

from numbers import Real
from typing import Any

import numpy as np


def select_episode_name(
    sim_period: Any,
    traj_all_by_episode: dict[str, dict[Any, Any]],
    episodes: list[str],
    np_random,
) -> str:
    if isinstance(sim_period, dict) and "episode_name" in sim_period:
        episode_name = sim_period["episode_name"]
        if episode_name not in traj_all_by_episode:
            raise ValueError(f"Episode {episode_name} not found.")
        return str(episode_name)
    return str(np_random.choice(episodes))


def select_ego_ids(
    valid_ids: np.ndarray,
    explicit_ego_ids: int | list[int] | tuple[int, ...] | np.ndarray | None,
    *,
    percentage_controlled_vehicles: int | float | str,
    np_random,
    episode_name: str,
    control_all_vehicles: bool = False,
    clip_to_available: bool = False,
) -> list[int]:
    resolved_num_vehicles = resolve_num_ego_vehicles(
        percentage_controlled_vehicles,
        len(valid_ids),
    )

    if explicit_ego_ids is None:
        if resolved_num_vehicles > len(valid_ids):
            if not clip_to_available:
                raise ValueError(
                    f"Requested {resolved_num_vehicles} ego vehicles, but only "
                    f"{len(valid_ids)} valid ids are available"
                )
            resolved_num_vehicles = len(valid_ids)
        if control_all_vehicles:
            ego_ids = [int(eid) for eid in valid_ids]
            if not ego_ids:
                raise ValueError(
                    "control_all_vehicles selected no viable ego ids for the current "
                    "episode."
                )
            return ego_ids
        return list(
            map(
                int,
                np_random.choice(
                    valid_ids,
                    size=resolved_num_vehicles,
                    replace=False,
                ),
            )
        )

    if np.isscalar(explicit_ego_ids):
        selected_ego_ids = [int(explicit_ego_ids)]
    else:
        selected_ego_ids = [int(eid) for eid in explicit_ego_ids]

    invalid_ids = [eid for eid in selected_ego_ids if eid not in valid_ids]
    if invalid_ids:
        raise ValueError(f"Ego IDs {invalid_ids} not in episode {episode_name}")

    if not control_all_vehicles and len(selected_ego_ids) != resolved_num_vehicles:
        raise ValueError(
            f"Expected {resolved_num_vehicles} explicit ego ids, got {len(selected_ego_ids)}"
        )

    return [int(eid) for eid in selected_ego_ids]


def resolve_num_ego_vehicles(
    percentage_controlled_vehicles: int | float | str,
    valid_id_count: int,
) -> int:
    """
    Resolve an ego-vehicle request to an absolute count.

    ``percentage_controlled_vehicles`` accepts fractional values such as ``0.1``
    and percentage strings such as ``"10%"``. Integer values are accepted for
    compatibility and treated as absolute counts.
    """
    valid_id_count = int(valid_id_count)
    if valid_id_count < 1:
        raise ValueError("No valid ego ids are available.")

    if isinstance(percentage_controlled_vehicles, str):
        text = percentage_controlled_vehicles.strip()
        if text.endswith("%"):
            percentage = float(text[:-1]) / 100.0
            return _percentage_to_count(
                percentage,
                valid_id_count,
                original=percentage_controlled_vehicles,
            )

        numeric = float(text)
        if not numeric.is_integer() or 0.0 < numeric < 1.0:
            return _percentage_to_count(
                numeric,
                valid_id_count,
                original=percentage_controlled_vehicles,
            )
        percentage_controlled_vehicles = int(numeric)

    if (
        isinstance(percentage_controlled_vehicles, Real)
        and not float(percentage_controlled_vehicles).is_integer()
    ):
        return _percentage_to_count(
            float(percentage_controlled_vehicles),
            valid_id_count,
            original=percentage_controlled_vehicles,
        )

    count = int(percentage_controlled_vehicles)
    if count < 1:
        raise ValueError(
            "percentage_controlled_vehicles must be >= 1 or a positive "
            f"percentage, got {percentage_controlled_vehicles}"
        )
    return count


def _percentage_to_count(
    percentage: float,
    valid_id_count: int,
    *,
    original: int | float | str,
) -> int:
    if not 0.0 < float(percentage) <= 1.0:
        raise ValueError(
            f"num_vehicles percentage must be in (0, 100%], got {original!r}"
        )
    return max(1, int(np.ceil(float(percentage) * int(valid_id_count))))


def build_trajectory_set(
    traj_all_by_episode: dict[str, dict[Any, Any]],
    episode_name: str,
    ego_ids: list[int],
) -> dict[Any, Any]:
    traj_all = traj_all_by_episode[episode_name]
    ego_id_set = set(ego_ids)
    return {
        "ego": {eid: traj_all[eid] for eid in ego_ids},
        **{vid: meta for vid, meta in traj_all.items() if vid not in ego_id_set},
    }
