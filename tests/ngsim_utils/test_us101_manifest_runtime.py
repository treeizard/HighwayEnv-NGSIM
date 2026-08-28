from __future__ import annotations

import numpy as np
import pytest
from highway_env.ngsim_utils.road.gen_road import create_ngsim_101_road
from highway_env.ngsim_utils.road.lane_mapping import (
    target_lane_index_from_position_and_lane_id,
)
from highway_env.ngsim_utils.road.manifest_road import (
    RoadGeometryV3,
    build_road_network,
)

from policy.data.us101_trajectory import us101_road_geometry_payload


def _edge_lane_counts(network) -> dict[tuple[str, str], int]:
    return {
        (start, end): len(lanes) for start, destinations in network.graph.items() for end, lanes in destinations.items()
    }


def test_us101_manifest_road_has_exact_legacy_graph_geometry_and_style() -> None:
    legacy = create_ngsim_101_road()
    geometry = RoadGeometryV3.from_source(us101_road_geometry_payload())
    manifest = build_road_network(geometry)

    assert (
        _edge_lane_counts(manifest)
        == _edge_lane_counts(legacy)
        == {
            ("s1", "s2"): 5,
            ("merge_in", "s2"): 1,
            ("s2", "s3"): 6,
            ("s3", "s4"): 5,
            ("s3", "merge_out"): 1,
        }
    )
    for start, destinations in legacy.graph.items():
        for end, legacy_lanes in destinations.items():
            manifest_lanes = manifest.graph[start][end]
            assert len(manifest_lanes) == len(legacy_lanes)
            for legacy_lane, manifest_lane in zip(legacy_lanes, manifest_lanes, strict=True):
                np.testing.assert_allclose(
                    manifest_lane.position(0.0, 0.0),
                    legacy_lane.position(0.0, 0.0),
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    manifest_lane.position(manifest_lane.length, 0.0),
                    legacy_lane.position(legacy_lane.length, 0.0),
                    rtol=0.0,
                    atol=1e-9,
                )
                assert manifest_lane.width == pytest.approx(legacy_lane.width)
                assert manifest_lane.forbidden is legacy_lane.forbidden
                assert manifest_lane.line_types == legacy_lane.line_types
                assert manifest_lane.speed_limit == legacy_lane.speed_limit
                assert manifest_lane.priority == legacy_lane.priority


def test_us101_manifest_mapping_uses_pose_and_keeps_uncertain_raw_lanes() -> None:
    network = build_road_network(RoadGeometryV3.from_source(us101_road_geometry_payload()))

    main_1_up = network.manifest_lane_index_by_id["main_1_s1_s2"]
    main_1_mid = network.manifest_lane_index_by_id["main_1_s2_s3"]
    ramp_in = network.manifest_lane_index_by_id["merge_in_7"]
    main_6_mid = network.manifest_lane_index_by_id["main_6_s2_s3"]
    ramp_out = network.manifest_lane_index_by_id["merge_out_8"]

    assert target_lane_index_from_position_and_lane_id(network, "us-101", np.asarray([50.0, 0.0]), 1) == main_1_up
    assert target_lane_index_from_position_and_lane_id(network, "us-101", np.asarray([250.0, 0.0]), 1) == main_1_mid
    assert target_lane_index_from_position_and_lane_id(network, "us-101", np.asarray([150.0, 20.0]), 7) == ramp_in
    assert target_lane_index_from_position_and_lane_id(network, "us-101", np.asarray([250.0, 18.0]), 7) == main_6_mid
    assert target_lane_index_from_position_and_lane_id(network, "us-101", np.asarray([450.0, 21.0]), 8) == ramp_out


def test_manifest_road_rejects_invalid_explicit_line_style() -> None:
    payload = us101_road_geometry_payload()
    payload["edges"][0]["lanes"][0]["line_types"] = [999, 0]

    with pytest.raises(ValueError, match="line_types"):
        RoadGeometryV3.from_source(payload)
