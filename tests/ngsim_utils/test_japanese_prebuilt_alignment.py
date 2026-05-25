from __future__ import annotations

import numpy as np
import pandas as pd

from scripts_setup.build_prebuilt_japanese import (
    align_lanes_to_japanese_road,
    build_episode_dicts,
    clip_japanese_lateral_to_road,
    parse_datetime_jst,
    parse_morinomiya_clock_datetime,
    target_japanese_lane_center_y,
)
from highway_env.ngsim_utils.road.gen_road import create_japanese_road


def test_parse_datetime_jst_handles_mixed_timezone_offsets():
    parsed = parse_datetime_jst(
        [
            "2020-01-01T10:00:00.500000+0900",
            "2020-01-01T12:00:00.500000+1100",
        ]
    )

    assert str(parsed.dtype) == "datetime64[ns, Asia/Tokyo]"
    assert parsed.dt.strftime("%Y-%m-%d %H:%M:%S.%f%z").tolist() == [
        "2020-01-01 10:00:00.500000+0900",
        "2020-01-01 10:00:00.500000+0900",
    ]


def test_parse_morinomiya_clock_datetime_handles_hhmmssmmm_values():
    parsed = parse_morinomiya_clock_datetime([94000500, 100000600, 121500600])

    assert str(parsed.dtype) == "datetime64[ns, Asia/Tokyo]"
    assert parsed.dt.strftime("%Y-%m-%d %H:%M:%S.%f%z").tolist() == [
        "2020-01-01 09:40:00.500000+0900",
        "2020-01-01 10:00:00.600000+0900",
        "2020-01-01 12:15:00.600000+0900",
    ]


def test_japanese_prebuilt_trajectories_are_padded_to_shared_time_grid():
    base_time = pd.Timestamp("2020-01-01 09:00:00", tz="Asia/Tokyo")
    rows = []

    for frame in range(10):
        rows.append(
            {
                "vehicle_id": 1,
                "vehicle_length": 4.5,
                "vehicle_width": 1.7,
                "x_smooth": 10.0 + frame,
                "y_smooth": 1.0,
                "v_smooth": 5.0,
                "traffic_lane": 1,
                "datetime_jst": base_time + pd.to_timedelta(0.1 * frame, unit="s"),
            }
        )

    for frame in range(3, 6):
        rows.append(
            {
                "vehicle_id": 2,
                "vehicle_length": 3.5,
                "vehicle_width": 1.7,
                "x_smooth": 20.0 + frame,
                "y_smooth": 5.0,
                "v_smooth": 3.0,
                "traffic_lane": 3,
                "datetime_jst": base_time + pd.to_timedelta(0.1 * frame, unit="s"),
            }
        )

    veh_ids_by_episode, trajectories_by_episode = build_episode_dicts(
        pd.DataFrame(rows),
        window_sec=1,
        presence_ratio_threshold=0.8,
    )

    episode_name = next(iter(trajectories_by_episode))
    episode = trajectories_by_episode[episode_name]

    assert episode[np.int64(1)]["trajectory"].shape == (10, 4)
    assert episode[np.int64(2)]["trajectory"].shape == (10, 4)
    np.testing.assert_allclose(episode[np.int64(2)]["trajectory"][:3], np.zeros((3, 4)))
    np.testing.assert_allclose(
        episode[np.int64(2)]["trajectory"][3],
        np.array([23.0, 5.0, 3.0, 3.0]),
    )
    np.testing.assert_allclose(episode[np.int64(2)]["trajectory"][6:], np.zeros((4, 4)))
    assert veh_ids_by_episode[episode_name] == [np.int64(1)]


def test_lane_center_alignment_preserves_residuals_and_clips_offroad_centers():
    rows = []
    net = create_japanese_road()
    for x in np.arange(0.0, 800.0, 20.0):
        target_y = target_japanese_lane_center_y(net, x, 2)
        rows.append(
            {
                "x_curved": x,
                "y_curved": target_y + 0.75 + (0.10 if int(x / 20.0) % 2 else -0.10),
                "traffic_lane": 2,
            }
        )
    rows.append({"x_curved": 120.0, "y_curved": 20.0, "traffic_lane": 2})

    aligned, profile, summary = align_lanes_to_japanese_road(
        pd.DataFrame(rows),
        bin_size_m=80.0,
        sample_step=1,
        max_abs_lateral_m=1.65,
    )

    assert not profile.empty
    assert summary["aligned_rows"] == len(rows)
    assert summary["clipped_rows"] == 1

    residuals = []
    for row in aligned.iloc[:-1].itertuples(index=False):
        target_y = target_japanese_lane_center_y(net, row.x_curved, row.traffic_lane)
        residuals.append(row.y_curved - target_y)

    assert abs(float(np.median(residuals))) < 0.05
    clipped_row = aligned.iloc[-1]
    clipped_target = target_japanese_lane_center_y(
        net,
        clipped_row["x_curved"],
        clipped_row["traffic_lane"],
    )
    assert abs(float(clipped_row["y_curved"] - clipped_target)) <= 1.65 + 1e-6


def test_post_smoothing_lateral_clip_keeps_centers_inside_lane():
    net = create_japanese_road()
    target_y = target_japanese_lane_center_y(net, 500.0, 1)
    rows = pd.DataFrame(
        [
            {"x_smooth": 500.0, "y_smooth": target_y + 2.4, "traffic_lane": 1},
            {"x_smooth": 500.0, "y_smooth": target_y - 0.2, "traffic_lane": 1},
        ]
    )

    clipped, summary = clip_japanese_lateral_to_road(rows, max_abs_lateral_m=1.65)

    assert summary == {"checked_rows": 2, "clipped_rows": 1}
    np.testing.assert_allclose(clipped.loc[0, "y_smooth"], target_y + 1.65)
    np.testing.assert_allclose(clipped.loc[1, "y_smooth"], target_y - 0.2)
