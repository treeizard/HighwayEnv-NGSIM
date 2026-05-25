from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("gymnasium")

from scripts_env_test.diagnose_japanese_replay_overlaps import (  # noqa: E402
    ScanConfig,
    active_vehicle,
    compare_lane3_boundary_rows,
    inactive_render_risk,
    inspect_pair_geometry_window,
    run_env_overlap_scan,
)


def test_inactive_zero_footprint_vehicle_is_excluded_from_overlap_and_render_risk():
    vehicle = SimpleNamespace(
        remove_from_road=False,
        appear=False,
        visible=False,
        LENGTH=0.0,
        WIDTH=0.0,
        position=np.array([0.0, 0.0], dtype=float),
    )

    assert not active_vehicle(vehicle)
    assert not inactive_render_risk(vehicle)


def test_japanese_lane3_boundary_comparison_matches_road_315_boundary():
    rows = {row["x"]: row for row in compare_lane3_boundary_rows()}

    assert rows[260.0]["current"] == "('b', 'c', 2)"
    assert rows[260.0]["current"] == rows[260.0]["road_315_boundary"]
    assert rows[314.9]["road_315_boundary"] == "('b', 'c', 2)"
    assert rows[314.9]["current"] == rows[314.9]["road_315_boundary"]
    assert rows[315.0]["road_315_boundary"] == "('c', 'd', 1)"


def test_known_japanese_seed_3165_3167_overlap_state_is_explicit():
    config = ScanConfig(
        scene="japanese",
        episode_root="highway_env/data/processed_20s",
        prebuilt_split="train",
        episode_name="t1577840400000",
        ego_vehicle_id=2586,
        max_steps=31,
        max_surrounding="all",
        allow_idm=True,
        vehicle_ids=(3165, 3167),
    )

    result = run_env_overlap_scan(config)
    hits = result["overlaps"]

    if hits:
        first = hits[0]
        assert first["vehicle_id_a"] == 3165
        assert first["vehicle_id_b"] == 3167
        assert 28 <= first["step"] <= 30
        assert first["classification"] == "raw-data"
        assert first["polygon_intersection"] is True
        assert first["overtaken_a"] is False
        assert first["overtaken_b"] is False
    else:
        rows = inspect_pair_geometry_window(config, (3165, 3167), 28, 29)
        current_28 = next(
            row
            for row in rows
            if row["step"] == 28 and row["variant"] == "current_mapped"
        )
        assert current_28["polygon_intersection"] is False


def test_pair_geometry_window_compares_current_motion_and_lane3_boundary_variants():
    config = ScanConfig(
        scene="japanese",
        episode_root="highway_env/data/processed_20s",
        prebuilt_split="train",
        episode_name="t1577840400000",
        ego_vehicle_id=2586,
        max_steps=31,
        max_surrounding="all",
        allow_idm=True,
    )

    rows = inspect_pair_geometry_window(config, (3165, 3167), 28, 29)
    variants = {(row["step"], row["variant"]) for row in rows}

    assert (28, "current_mapped") in variants
    assert (28, "motion_derived") in variants
    assert (28, "lane3_315_boundary") in variants
    current_28 = next(row for row in rows if row["step"] == 28 and row["variant"] == "current_mapped")
    assert isinstance(current_28["polygon_intersection"], bool)
