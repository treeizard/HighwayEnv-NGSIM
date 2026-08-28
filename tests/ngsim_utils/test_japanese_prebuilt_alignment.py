from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
from highway_env.ngsim_utils.road.gen_road import create_japanese_road
from highway_env.ngsim_utils.road.lane_mapping import edge_from_x
from highway_env.ngsim_utils.road.manifest_road import RoadGeometryV3
from scripts_setup import prepare_morinomiya_from_csv
from scripts_setup.build_prebuilt_japanese import (
    add_causal_source_headings,
    align_lanes_to_japanese_road,
    audit_registered_kilopost,
    build_episode_dicts,
    clip_japanese_lateral_to_road,
    enforce_split_isolation,
    fit_source_derived_japanese_road,
    parse_datetime_jst,
    parse_morinomiya_clock_datetime,
    quarantine_near_coincident_trajectories,
    rigid_register_shared_coordinates,
    source_preserving_trajectory_rows,
    split_episode_keys_by_source,
    target_japanese_lane_center_y,
)


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


def test_kilopost_diagnostic_uses_only_explicit_train_and_validation_masks():
    rows = pd.DataFrame(
        {
            "source_file": ["L003_F001"] * 300,
            "kilopost": np.arange(300, dtype=float) + 1800.0,
            "x_registered": np.arange(300, dtype=float),
        }
    )
    train = np.zeros(300, dtype=bool)
    validation = np.zeros(300, dtype=bool)
    train[:100] = True
    validation[100:200] = True

    receipt = audit_registered_kilopost(
        rows, fit_mask=train, validation_mask=validation
    )

    assert receipt["diagnostic_only"] is True
    assert receipt["qualification_gate"] is False
    assert receipt["train"]["row_count"] == 100
    assert receipt["validation"]["row_count"] == 100
    assert receipt["test_rows_aggregated"] is False
    assert "all_retained_rows" not in receipt


def test_causal_heading_is_future_invariant_and_survives_episode_partition():
    base = pd.Timestamp("2020-01-01 09:00:00", tz="Asia/Tokyo")
    rows = pd.DataFrame(
        {
            "source_file": ["L003_F001"] * 205,
            "vehicle_id": [1] * 205,
            "datetime_jst": [
                base + pd.to_timedelta(frame * 0.1, unit="s")
                for frame in range(205)
            ],
            "x_smooth": np.arange(205, dtype=float) * 0.2,
            "y_smooth": np.zeros(205),
            "v_smooth": np.full(205, 7.2),
            "traffic_lane": np.ones(205, dtype=int),
            "detected_flag": np.ones(205, dtype=int),
        }
    )
    road = {"lane_centerlines": {"1": [[0.0, 0.0], [100.0, 0.0]]}}

    original = add_causal_source_headings(rows, road_geometry=road)
    mutated_rows = rows.copy()
    mutated_rows.loc[204, ["x_smooth", "y_smooth"]] = [10.0, 50.0]
    mutated = add_causal_source_headings(mutated_rows, road_geometry=road)

    np.testing.assert_array_equal(
        original["heading_rad"].to_numpy()[:204],
        mutated["heading_rad"].to_numpy()[:204],
    )
    assert original["heading_derivation"].iloc[0] == 2
    assert original["heading_derivation"].iloc[200] == 1
    assert original["heading_valid"].all()


def test_full_road_preparer_atomically_writes_source_partitioned_parquet(
    tmp_path, monkeypatch
):
    source_dir = (
        tmp_path
        / "raw"
        / "L003_F001_ALL"
        / "L003_F001_TRAJECTORY"
    )
    source_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            [1, 90000000, 1, 36.0, 1, 135.527945, 34.681580, 1819.7, 4.5, 1],
            [1, 90000100, 1, 36.0, 1, 135.537945, 34.681580, 2819.7, 4.5, 0],
        ]
    )
    rows.to_csv(source_dir / "L003_F001_trajectory.csv", index=False, header=False)
    output_npy = tmp_path / "canonical.npy"
    summary_json = tmp_path / "summary.json"
    parquet_root = tmp_path / "canonical_parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_morinomiya_from_csv.py",
            "--raw-root",
            str(tmp_path / "raw"),
            "--output-npy",
            str(output_npy),
            "--summary-json",
            str(summary_json),
            "--availability-csv",
            str(tmp_path / "availability.csv"),
            "--parquet-root",
            str(parquet_root),
            "--full-road",
        ],
    )

    prepare_morinomiya_from_csv.main()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    parquet = summary["canonical_parquet"]
    assert summary["full_road"] is True
    assert summary["rows_after_duplicate_drop"] == 2
    assert parquet["row_count"] == 2
    assert len(parquet["files"]) == 1
    assert (parquet_root / parquet["files"][0]["relative_path"]).is_file()
    assert not parquet_root.with_name("canonical_parquet.partial").exists()


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
                "source_file": "L003_F001",
                "vehicle_length": 4.5,
                "vehicle_width": 1.7,
                "x_smooth": 10.0 + frame,
                "y_smooth": 1.0,
                "v_smooth": 5.0,
                "traffic_lane": 1,
                "kilopost": 1800.0 + frame,
                "detected_flag": 1,
                "datetime_jst": base_time + pd.to_timedelta(0.1 * frame, unit="s"),
            }
        )

    for frame in range(3, 6):
        rows.append(
            {
                "vehicle_id": 2,
                "source_file": "L003_F001",
                "vehicle_length": 3.5,
                "vehicle_width": 1.7,
                "x_smooth": 20.0 + frame,
                "y_smooth": 5.0,
                "v_smooth": 3.0,
                "traffic_lane": 3,
                "kilopost": 1810.0 + frame,
                "detected_flag": 0,
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
    assert episode_name.startswith("L003_F001__t")
    assert episode[np.int64(1)]["source_file"] == "L003_F001"


def test_japanese_episode_builder_never_merges_sources_at_the_same_clock_time():
    base_time = pd.Timestamp("2020-01-01 10:00:00", tz="Asia/Tokyo")
    rows = []
    for source_index, source in enumerate(("L003_F001", "L003_F002")):
        for frame in range(10):
            rows.append(
                {
                    "source_file": source,
                    "vehicle_id": source_index * 10_000_000 + 1,
                    "vehicle_length": 4.5,
                    "vehicle_width": 1.7,
                    "x_smooth": 10.0 + frame,
                    "y_smooth": 1.0,
                    "v_smooth": 5.0,
                    "traffic_lane": 1,
                    "kilopost": 1800.0 + frame,
                    "detected_flag": 1,
                    "datetime_jst": base_time + pd.to_timedelta(0.1 * frame, unit="s"),
                }
            )

    valid_ids, trajectories = build_episode_dicts(
        pd.DataFrame(rows),
        window_sec=1,
        presence_ratio_threshold=0.8,
    )

    assert sorted(trajectories) == [
        "L003_F001__t1577840400000",
        "L003_F002__t1577840400000",
    ]
    assert [len(trajectories[key]) for key in sorted(trajectories)] == [1, 1]
    assert [len(valid_ids[key]) for key in sorted(valid_ids)] == [1, 1]


def test_japanese_episode_split_is_chronological_within_every_source():
    keys = [
        f"{source}__t{timestamp}"
        for source in ("L003_F001", "L003_F002")
        for timestamp in range(1000, 7000, 1000)
    ]

    train, val, test = split_episode_keys_by_source(
        keys,
        val_ratio=1 / 3,
        test_ratio=1 / 3,
    )

    for source in ("L003_F001", "L003_F002"):
        assert [key for key in train if key.startswith(source)] == [
            f"{source}__t1000",
            f"{source}__t2000",
        ]
        assert [key for key in val if key.startswith(source)] == [
            f"{source}__t3000",
            f"{source}__t4000",
        ]
        assert [key for key in test if key.startswith(source)] == [
            f"{source}__t5000",
            f"{source}__t6000",
        ]


def test_split_isolation_guards_boundaries_and_removes_complete_cross_split_identity():
    keys = [f"L003_F001__t{timestamp}" for timestamp in range(1000, 7000, 1000)]
    train, val, test = split_episode_keys_by_source(
        keys, val_ratio=1 / 3, test_ratio=1 / 3
    )
    trajectories = {
        key: {
            np.int64(vehicle_id): {
                "provider_observation_mask": np.ones(10, dtype=np.int8)
            }
            for vehicle_id in ({1, int(key.rsplit("t", 1)[1])})
        }
        for key in keys
    }
    valid_ids = {key: list(values) for key, values in trajectories.items()}

    guarded, filtered_ids, filtered_trajectories, receipt = enforce_split_isolation(
        split_keys={"train": train, "val": val, "test": test},
        veh_ids_by_episode=valid_ids,
        trajectories_by_episode=trajectories,
        window_sec=1,
        boundary_guard_sec=1,
    )

    assert guarded == {
        "train": ["L003_F001__t1000", "L003_F001__t2000"],
        "val": ["L003_F001__t4000"],
        "test": ["L003_F001__t6000"],
    }
    assert receipt["guard_removed_episode_keys"] == {
        "val": ["L003_F001__t3000"],
        "test": ["L003_F001__t5000"],
    }
    assert receipt["quarantined_cross_split_identity_count"] == 1
    assert all(1 not in filtered_trajectories[key] for key in filtered_trajectories)
    assert all(1 not in filtered_ids[key] for key in filtered_ids)


def test_near_coincident_raw_tracks_are_quarantined_before_smoothing():
    rows = []
    for timestamp in (100000000, 100000100):
        rows.extend(
            [
                {
                    "source_file": "L003_F001",
                    "vehicle_id": 1,
                    "datetime": timestamp,
                    "x_m": 10.0,
                    "y_m": 1.0,
                },
                {
                    "source_file": "L003_F001",
                    "vehicle_id": 2,
                    "datetime": timestamp,
                    "x_m": 10.4,
                    "y_m": 1.1,
                },
                {
                    "source_file": "L003_F001",
                    "vehicle_id": 3,
                    "datetime": timestamp,
                    "x_m": 12.0,
                    "y_m": 1.0,
                },
            ]
        )

    retained, report = quarantine_near_coincident_trajectories(
        pd.DataFrame(rows),
        minimum_center_separation_m=1.0,
    )

    assert retained["vehicle_id"].unique().tolist() == [3]
    assert report["pair_count"] == 1
    assert report["quarantined_identity_count"] == 2
    assert report["removed_row_count"] == 4
    assert report["pairs"][0]["event_count"] == 2
    assert report["pairs"][0]["minimum_center_distance_m"] < 0.5


def test_near_coincident_quarantine_identity_is_scoped_to_source_session():
    rows = pd.DataFrame(
        [
            {"source_file": "L003_F001", "vehicle_id": 1, "datetime": 1, "x_m": 0.0, "y_m": 0.0},
            {"source_file": "L003_F001", "vehicle_id": 2, "datetime": 1, "x_m": 0.2, "y_m": 0.0},
            {"source_file": "L003_F002", "vehicle_id": 1, "datetime": 1, "x_m": 10.0, "y_m": 0.0},
        ]
    )

    retained, report = quarantine_near_coincident_trajectories(
        rows,
        minimum_center_separation_m=1.0,
    )

    assert retained[["source_file", "vehicle_id"]].to_records(index=False).tolist() == [
        ("L003_F002", 1)
    ]
    assert report["identity_key_format"] == "<source_file>,<vehicle_id>"
    assert report["quarantined_identity_count"] == 2


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


def test_source_preserving_registration_changes_only_the_coordinate_frame():
    rows = []
    base = pd.Timestamp("2020-01-01 09:00:00", tz="Asia/Tokyo")
    for lane_id, lateral, x_values in (
        (1, 1.875, np.linspace(0.0, 800.0, 400)),
        (2, -1.875, np.linspace(0.0, 800.0, 400)),
        (3, 5.625, np.linspace(120.0, 320.0, 200)),
    ):
        for index, x in enumerate(x_values):
            rows.append(
                {
                    "source_file": "L003_F001",
                    "traffic_lane": lane_id,
                    "x_m": x,
                    "y_m": lateral + 0.008 * x + 0.8 * np.sin(x / 180.0),
                    "velocity": 36.0,
                    "datetime_jst": base + pd.to_timedelta(index * 0.1, unit="s"),
                }
            )
    frame = pd.DataFrame(rows).reset_index(drop=True)
    fit_mask = np.ones(len(frame), dtype=bool)

    registered, receipt = rigid_register_shared_coordinates(frame, fit_mask=fit_mask)
    output, rejections = source_preserving_trajectory_rows(registered)

    assert receipt["transform_type"] == "rigid_se2"
    assert receipt["transform_scope"] == "one_shared_transform_all_sources"
    assert receipt["preserves_inter_source_geometry"] is True
    assert receipt["lane_specific_transform"] is False
    assert receipt["nonlinear_warp"] is False
    assert rejections["rejected_rows"] == 0
    np.testing.assert_array_equal(output["v_smooth"], output["velocity"])
    np.testing.assert_array_equal(output["x_smooth"], output["x_registered"])
    np.testing.assert_array_equal(output["y_smooth"], output["y_registered"])
    before = np.linalg.norm(frame.loc[0, ["x_m", "y_m"]].to_numpy(dtype=float) - frame.loc[300, ["x_m", "y_m"]].to_numpy(dtype=float))
    after = np.linalg.norm(output.loc[0, ["x_registered", "y_registered"]].to_numpy(dtype=float) - output.loc[300, ["x_registered", "y_registered"]].to_numpy(dtype=float))
    assert after == pytest.approx(before)


def test_shared_registration_does_not_shift_recordings_into_distinct_road_frames():
    rows = []
    for source in ("L003_F001", "L003_F002"):
        for x in np.linspace(0.0, 800.0, 400):
            rows.extend(
                [
                    {"source_file": source, "traffic_lane": 1, "x_m": x, "y_m": 2.0},
                    {"source_file": source, "traffic_lane": 2, "x_m": x, "y_m": -2.0},
                ]
            )
    frame = pd.DataFrame(rows)
    registered, receipt = rigid_register_shared_coordinates(
        frame, fit_mask=np.ones(len(frame), dtype=bool)
    )
    first = registered[registered["source_file"] == "L003_F001"]
    second = registered[registered["source_file"] == "L003_F002"]

    np.testing.assert_allclose(
        first[["x_registered", "y_registered"]].to_numpy(),
        second[["x_registered", "y_registered"]].to_numpy(),
    )
    assert receipt["fit_rows_by_source"] == {"L003_F001": 800, "L003_F002": 800}


def test_source_derived_japanese_road_is_curved_and_train_only():
    rows = []
    for lane_id, lateral, x_values in (
        (1, 1.875, np.linspace(0.0, 800.0, 400)),
        (2, -1.875, np.linspace(0.0, 800.0, 400)),
        (3, 5.625, np.linspace(120.0, 320.0, 200)),
    ):
        for x in x_values:
            rows.append(
                {
                    "source_file": "L003_F001",
                    "traffic_lane": lane_id,
                    "x_registered": x,
                    "y_registered": lateral + 1.2 * np.sin(x / 180.0),
                }
            )
    frame = pd.DataFrame(rows)
    geometry = fit_source_derived_japanese_road(
        frame,
        fit_mask=np.ones(len(frame), dtype=bool),
        bin_size_m=10.0,
    )
    net = create_japanese_road(geometry)
    manifest_geometry = RoadGeometryV3.from_source(geometry)
    lane = net.get_lane(("c", "d", 0))

    assert geometry["fit_split"] == "train"
    assert geometry["test_rows_used"] is False
    assert geometry["trajectory_coordinates_modified_to_fit_road"] is False
    fit_receipt = geometry["geometry_fit_receipt"]
    assert fit_receipt["fit_split"] == "train"
    assert fit_receipt["validation_or_test_rows_used"] is False
    assert fit_receipt["source_position_invariance"]["passed"] is True
    assert fit_receipt["lane_spacing"]["maximum_absolute_error_m"] < 1.0e-12
    assert fit_receipt["lane_spacing"]["envelope_overlap_max_m"] < 1.0e-12
    assert fit_receipt["lane3_merge"]["approach_gap_nonincreasing"] is True
    assert fit_receipt["lane3_merge"]["junction_position_error_m"] < 1.0e-12
    assert geometry["source_document_topology"]["digitized_detector_kiloposts_m"] == [
        1800.0,
        2500.0,
        2900.0,
        3500.0,
        4000.0,
        4500.0,
    ]
    assert len(
        geometry["source_document_topology"]["digitized_section_drawing_bands"]
    ) == 6
    assert manifest_geometry.site_id == "morinomiya"
    assert manifest_geometry.test_rows_used is False
    assert edge_from_x(net, 500.0) == ("c", "d")
    headings = np.asarray(
        [lane.heading_at(value) for value in np.linspace(5.0, lane.length - 5.0, 20)],
        dtype=float,
    )
    assert float(np.ptp(headings)) > 1.0e-3


def test_morinomiya_refit_is_source_balanced_parallel_and_source_preserving():
    rows = []
    widths = {
        "L003_F001": 3.2,
        "L003_F002": 3.4,
        "L003_F003": 4.6,
    }
    for source_index, (source, width) in enumerate(widths.items()):
        source_bias = 0.03 * source_index
        for lane_id, sign in ((1, 1.0), (2, -1.0)):
            for x in np.linspace(0.0, 800.0, 400):
                spine = 1.5 * np.sin(x / 220.0)
                lane_noise = sign * 0.08 * np.sin(x / 17.0 + source_index)
                rows.append(
                    {
                        "source_file": source,
                        "traffic_lane": lane_id,
                        "x_registered": x,
                        "y_registered": spine + source_bias + sign * width / 2.0 + lane_noise,
                    }
                )
        if source != "L003_F003":
            for x in np.linspace(120.0, 320.0, 200):
                lane1_y = 1.5 * np.sin(x / 220.0) + source_bias + width / 2.0
                rows.append(
                    {
                        "source_file": source,
                        "traffic_lane": 3,
                        "x_registered": x,
                        "y_registered": lane1_y + 6.0 - 0.005 * (x - 120.0),
                    }
                )
    frame = pd.DataFrame(rows)
    original_xy = frame[["x_registered", "y_registered"]].to_numpy(copy=True)

    geometry = fit_source_derived_japanese_road(
        frame,
        fit_mask=np.ones(len(frame), dtype=bool),
        bin_size_m=10.0,
    )

    np.testing.assert_array_equal(
        frame[["x_registered", "y_registered"]].to_numpy(), original_xy
    )
    assert geometry["lane_width_m"] == pytest.approx(3.4, abs=0.02)
    estimator = geometry["lane_width_estimator"]
    assert estimator["contract"] == "equal_source_median_of_binned_mainline_medians_v1"
    assert estimator["source_count"] == 3
    lane1 = np.asarray(geometry["lane_centerlines"]["1"])
    lane2 = np.asarray(geometry["lane_centerlines"]["2"])
    np.testing.assert_allclose(
        np.linalg.norm(lane1 - lane2, axis=1),
        geometry["lane_width_m"],
        atol=1.0e-12,
    )
    assert np.all(np.diff(lane1[:, 0]) > 0.0)
    assert np.all(np.diff(lane2[:, 0]) > 0.0)
    receipt = geometry["geometry_fit_receipt"]
    assert receipt["mainline_spine"]["selected_knot_spacing_m"] in (25.0, 50.0, 100.0)
    assert receipt["smoothness"]["mainline_max_heading_change_per_sample_deg"] <= 2.0
    assert receipt["lane3_merge"]["junction_tangent_error_rad"] == 0.0
