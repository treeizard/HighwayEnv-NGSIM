import numpy as np
import pytest

from highway_env.ngsim_utils.road.gen_road import (
    create_japanese_road,
    create_ngsim_101_road,
)
from highway_env.road.lane import StraightLane
from highway_env.road.road import RoadNetwork
from highway_env.road.route_intent import (
    RouteIntentError,
    TopologyAwareRouteIntentProvider,
    route_intent_contract,
)


def _compute(network, current, route, position, *, heading=0.0, domain="test"):
    return TopologyAwareRouteIntentProvider().compute(
        network,
        ego_position=np.asarray(position, dtype=float),
        ego_heading=heading,
        current_lane_index=current,
        route=route,
        planner_source_id="unit_test_planner",
        planner_source_kind="scenario_route_command",
        plan_revision="frozen-v1",
        analysis_domain=domain,
    )


def _straight_network(*, angle=0.0, translation=(0.0, 0.0)):
    network = RoadNetwork()
    direction = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([-np.sin(angle), np.cos(angle)])
    translation = np.asarray(translation, dtype=float)
    for lane_id in range(2):
        lateral = lane_id * 4.0 * left
        network.add_lane(
            "a",
            "b",
            StraightLane(
                translation + lateral,
                translation + 100.0 * direction + lateral,
            ),
        )
        network.add_lane(
            "b",
            "c",
            StraightLane(
                translation + 100.0 * direction + lateral,
                translation + 220.0 * direction + lateral,
            ),
        )
    return network


def test_shared_schema_is_finite_and_keeps_topology_out_of_policy():
    provider = TopologyAwareRouteIntentProvider()
    contract = route_intent_contract()
    forbidden = ("action", "steer", "accel", "domain", "topology", "graph")
    assert contract["policy_projection"]["feature_dim"] == 16
    assert contract["controller_projection"]["feature_dim"] == 35
    assert contract["topology_sidecar"]["policy_visible"] is False
    assert contract["policy_projection"]["includes_current_tracking_error"] is False
    assert contract["policy_projection"]["includes_previous_action"] is False
    assert not any(
        token in field.lower()
        for field in provider.policy_feature_names
        for token in forbidden
    )

    cases = [
        (
            "us",
            create_ngsim_101_road(),
            ("s1", "s2", 0),
            [("s1", "s2", 0), ("s2", "s3", 0), ("s3", "s4", 0)],
            [10.0, 0.0],
        ),
        (
            "japanese",
            create_japanese_road(),
            ("a", "b", 0),
            [("a", "b", 0), ("b", "c", 0), ("c", "d", 0)],
            [10.0, -1.875],
        ),
    ]
    for domain, network, current, route, position in cases:
        frame = _compute(network, current, route, position, domain=domain)
        assert frame.policy_features.shape == (16,)
        assert frame.controller_features.shape == (35,)
        assert np.all(np.isfinite(frame.policy_features))
        assert np.max(np.abs(frame.policy_features)) <= 1.0
        assert frame.topology["policy_visible"] is False
        assert frame.topology["analysis_domain"] == domain
        assert frame.provenance["realized_future_path_read"] is False
        assert frame.provenance["expert_action_history_read"] is False


def test_policy_intent_is_global_pose_and_tracking_error_invariant():
    angle = 0.7
    translation = np.array([123.0, -45.0])
    rotated = _straight_network(angle=angle, translation=translation)
    canonical = _straight_network()
    route = [("a", "b", 0), ("b", "c", 0)]
    canonical_lane = canonical.get_lane(("a", "b", 0))
    rotated_lane = rotated.get_lane(("a", "b", 0))

    base = _compute(
        canonical,
        ("a", "b", 0),
        route,
        canonical_lane.position(10.0, 0.0),
    )
    transformed = _compute(
        rotated,
        ("a", "b", 0),
        route,
        rotated_lane.position(10.0, 0.0),
        heading=angle,
    )
    assert np.allclose(base.policy_features, transformed.policy_features, atol=1e-6)
    assert np.allclose(
        base.controller_features,
        transformed.controller_features,
        atol=1e-6,
    )
    assert (
        base.topology["structural_topology_fingerprint"]
        == transformed.topology["structural_topology_fingerprint"]
    )
    assert (
        base.topology["exact_graph_sha256"]
        != transformed.topology["exact_graph_sha256"]
    )

    displaced = _compute(
        canonical,
        ("a", "b", 0),
        route,
        canonical_lane.position(10.0, 1.25),
        heading=0.2,
    )
    assert np.allclose(base.policy_features, displaced.policy_features, atol=1e-6)
    assert not np.allclose(base.controller_features, displaced.controller_features)


def test_missing_disconnected_and_teacher_routes_fail_closed():
    network = _straight_network()
    kwargs = {
        "network": network,
        "ego_position": np.array([10.0, 0.0]),
        "ego_heading": 0.0,
        "current_lane_index": ("a", "b", 0),
        "planner_source_id": "test",
        "planner_source_kind": "scenario_route_command",
        "plan_revision": "1",
    }
    provider = TopologyAwareRouteIntentProvider()
    with pytest.raises(RouteIntentError, match="explicit online-planner route"):
        provider.compute(route=[], **kwargs)
    with pytest.raises(RouteIntentError, match="disconnected"):
        provider.compute(route=[("a", "b", 0), ("x", "y", 0)], **kwargs)
    kwargs["planner_source_kind"] = "realized_future_trajectory"
    with pytest.raises(RouteIntentError, match="forbidden"):
        provider.compute(route=[("a", "b", 0), ("b", "c", 0)], **kwargs)
    kwargs["planner_source_kind"] = "unregistered_planner_adapter"
    with pytest.raises(RouteIntentError, match="frozen online source kinds"):
        provider.compute(route=[("a", "b", 0), ("b", "c", 0)], **kwargs)


def test_merge_split_metadata_is_analysis_only():
    network = RoadNetwork()
    network.add_lane("a", "b", StraightLane([0, 0], [50, 0]))
    network.add_lane("b", "c", StraightLane([50, 0], [120, 0]))
    network.add_lane("b", "d", StraightLane([50, 0], [100, 50]))
    frame = _compute(
        network,
        ("a", "b", 0),
        [("a", "b", 0), ("b", "d", 0)],
        [10.0, 0.0],
    )
    decision = next(event for event in frame.topology["events"] if event["is_decision"])
    assert "split_or_branch" in decision["reasons"]
    assert "heading_change" in decision["reasons"]
    assert frame.policy_features[1] == 1.0
