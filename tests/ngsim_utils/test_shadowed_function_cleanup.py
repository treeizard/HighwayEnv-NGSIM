from __future__ import annotations

import importlib
import inspect

import numpy as np

from highway_env.ngsim_utils.core.constants import US101_SECTION_ENDS_M


def test_trajectory_smoothing_public_import_and_behavior_are_unchanged() -> None:
    module = importlib.import_module(
        "highway_env.ngsim_utils.data.trajectory_gen"
    )
    smoothing = module.trajectory_smoothing
    assert smoothing.__module__ == module.__name__

    trajectory = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 10.0, 5.0, 1.0],
            [4.0, 11.0, 4.0, 1.0],
            [9.0, 12.0, 3.0, 2.0],
            [16.0, 13.0, 2.0, 2.0],
            [25.0, 14.0, 1.0, 3.0],
            [36.0, 15.0, 6.0, 3.0],
        ],
        dtype=float,
    )
    expected = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 10.0, 5.0, 1.0],
            [4.0, 11.0, 4.0, 1.0],
            [9.0, 12.0, 3.0, 2.0],
            [16.0, 13.0, 1.485714285714, 2.0],
            [25.0, 14.0, 1.342857142857, 3.0],
            [36.0, 15.0, 5.914285714286, 3.0],
        ],
        dtype=float,
    )

    actual = np.asarray(smoothing(trajectory), dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert inspect.getsource(module).count("def trajectory_smoothing(") == 1


def test_clamp_location_public_import_and_return_contract_are_unchanged(
    capsys,
) -> None:
    module = importlib.import_module("highway_env.ngsim_utils.road.gen_road")
    clamp_location = module.clamp_location_ngsim
    assert clamp_location.__module__ == module.__name__
    network = module.create_ngsim_101_road()

    cases = [
        (0.0, 0, ("s1", "s2", 0)),
        (US101_SECTION_ENDS_M[1] + 0.1, 5, ("s2", "s3", 5)),
        (US101_SECTION_ENDS_M[2] + 0.1, 99, ("s3", "s4", 4)),
        (-1.0, -2, ("s1", "s2", 0)),
    ]
    for x_position, lane, expected_index in cases:
        lane_index, lane_object = clamp_location(
            x_position,
            lane,
            network,
            warning=lane in {-2, 99},
        )
        assert lane_index == expected_index
        assert lane_object is network.get_lane(expected_index)

    warnings = capsys.readouterr().out
    assert warnings.count("clamping") == 2
    assert inspect.getsource(module).count("def clamp_location_ngsim(") == 1
