import numpy as np

from highway_env.ngsim_utils.data.episode_selection import (
    resolve_num_ego_vehicles,
    select_ego_ids,
)


def test_resolve_num_ego_vehicles_accepts_percentage_string():
    assert resolve_num_ego_vehicles("10%", valid_id_count=100) == 10
    assert resolve_num_ego_vehicles("10%", valid_id_count=12) == 2


def test_select_ego_ids_accepts_percentage_request():
    valid_ids = np.arange(100)
    rng = np.random.default_rng(0)

    selected_ids = select_ego_ids(
        valid_ids,
        explicit_ego_ids=None,
        percentage_controlled_vehicles="10%",
        np_random=rng,
        episode_name="episode",
    )

    assert len(selected_ids) == 10
    assert len(set(selected_ids)) == 10
    assert all(vehicle_id in valid_ids for vehicle_id in selected_ids)


def test_select_ego_ids_honors_explicit_ids_when_control_all_is_enabled():
    valid_ids = np.asarray([1, 2, 3, 4], dtype=np.int64)
    rng = np.random.default_rng(0)

    selected_ids = select_ego_ids(
        valid_ids,
        explicit_ego_ids=[2, 4],
        percentage_controlled_vehicles=0.25,
        np_random=rng,
        episode_name="episode",
        control_all_vehicles=True,
    )

    assert selected_ids == [2, 4]
