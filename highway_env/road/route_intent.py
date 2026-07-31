"""Planner-facing, topology-aware route intent for policy and controller use.

The provider deliberately separates three views of one explicit planner route:

* ``controller_features`` are ego-relative metric corridor samples suitable for
  a downstream trajectory/controller module;
* ``policy_features`` are a smaller lane-anchor-relative intent projection that
  omits immediate tracking error and actuator/action history; and
* ``topology`` is an analysis-only sidecar for stratifying merges, splits and
  graph differences without giving a policy a domain/topology identifier.

No route is inferred from a realised demonstration suffix.  Callers must pass
an explicit route produced by an online planner or a prospectively prescribed
scenario command.  Missing or disconnected routes fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from highway_env.road.road import LaneIndex, RoadNetwork


ROUTE_INTENT_SCHEMA_VERSION = 1
ROUTE_INTENT_PROVIDER_ID = "topology_aware_route_intent_v1"
FORBIDDEN_PLANNER_SOURCE_KINDS = frozenset(
    {
        "expert_action_history",
        "realized_future_path",
        "realized_future_trajectory",
        "teacher_tracker",
        "teacher_tracker_state",
    }
)
PERMITTED_PLANNER_SOURCE_KINDS = frozenset(
    {
        "external_online_planner",
        "planner_graph_route",
        "scenario_route_command",
    }
)


class RouteIntentError(ValueError):
    """Raised when a planner route cannot satisfy the deployable contract."""


@dataclass(frozen=True)
class RouteIntentConfig:
    """Frozen numerical and feature-layout contract for route intent."""

    controller_horizons_m: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)
    policy_horizons_m: tuple[float, ...] = (20.0, 40.0, 80.0)
    maximum_speed_mps: float = 40.0
    maximum_lane_width_m: float = 10.0
    decision_distance_cap_m: float = 200.0
    decision_lateral_cap_m: float = 20.0
    decision_heading_threshold_rad: float = 0.08
    decision_lateral_threshold_m: float = 0.5

    def __post_init__(self) -> None:
        controller_horizons = tuple(float(v) for v in self.controller_horizons_m)
        policy_horizons = tuple(float(v) for v in self.policy_horizons_m)
        for name, values in (
            ("controller_horizons_m", controller_horizons),
            ("policy_horizons_m", policy_horizons),
        ):
            if not values or any(not np.isfinite(v) or v <= 0.0 for v in values):
                raise RouteIntentError(f"{name} must contain positive finite values.")
            if any(right <= left for left, right in zip(values, values[1:])):
                raise RouteIntentError(f"{name} must be strictly increasing.")
        for name in (
            "maximum_speed_mps",
            "maximum_lane_width_m",
            "decision_distance_cap_m",
            "decision_lateral_cap_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise RouteIntentError(f"{name} must be positive and finite.")
        if (
            not np.isfinite(self.decision_heading_threshold_rad)
            or self.decision_heading_threshold_rad < 0.0
        ):
            raise RouteIntentError(
                "decision_heading_threshold_rad must be finite and non-negative."
            )
        if (
            not np.isfinite(self.decision_lateral_threshold_m)
            or self.decision_lateral_threshold_m < 0.0
        ):
            raise RouteIntentError(
                "decision_lateral_threshold_m must be finite and non-negative."
            )
        object.__setattr__(self, "controller_horizons_m", controller_horizons)
        object.__setattr__(self, "policy_horizons_m", policy_horizons)


@dataclass(frozen=True)
class RouteSample:
    horizon_m: float
    valid: bool
    lane_index: LaneIndex | None
    position: np.ndarray
    heading: float
    speed_limit_mps: float
    lane_width_m: float


@dataclass(frozen=True)
class RouteIntentFrame:
    """One immutable route-intent result."""

    policy_features: np.ndarray
    controller_features: np.ndarray
    policy_feature_names: tuple[str, ...]
    controller_feature_names: tuple[str, ...]
    resolved_route: tuple[LaneIndex, ...]
    topology: dict[str, Any]
    provenance: dict[str, Any]

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_INTENT_SCHEMA_VERSION,
            "provider_id": ROUTE_INTENT_PROVIDER_ID,
            "policy_feature_names": list(self.policy_feature_names),
            "policy_features": self.policy_features.astype(float).tolist(),
            "controller_feature_names": list(self.controller_feature_names),
            "controller_features": self.controller_features.astype(float).tolist(),
            "resolved_route": [list(index) for index in self.resolved_route],
            "topology": self.topology,
            "provenance": self.provenance,
        }


def _wrap_to_pi(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _frame_coordinates(
    point: np.ndarray,
    *,
    origin: np.ndarray,
    heading: float,
) -> tuple[float, float]:
    delta = np.asarray(point, dtype=float).reshape(2) - np.asarray(
        origin, dtype=float
    ).reshape(2)
    cosine = float(np.cos(heading))
    sine = float(np.sin(heading))
    forward = cosine * float(delta[0]) + sine * float(delta[1])
    left = -sine * float(delta[0]) + cosine * float(delta[1])
    return forward, left


def _lane_speed_limit(lane: Any) -> float:
    value = getattr(lane, "speed_limit", 0.0)
    if value is None:
        return 0.0
    value = float(value)
    return value if np.isfinite(value) and value >= 0.0 else 0.0


def _incoming_degree(network: RoadNetwork, node: str) -> int:
    return int(
        sum(1 for destinations in network.graph.values() if node in destinations)
    )


def _lane_count(network: RoadNetwork, index: LaneIndex) -> int:
    return int(len(network.graph[index[0]][index[1]]))


def _route_digest(
    route: Sequence[LaneIndex],
    *,
    planner_source_id: str,
    plan_revision: str,
) -> str:
    payload = {
        "planner_source_id": str(planner_source_id),
        "plan_revision": str(plan_revision),
        "route": [list(index) for index in route],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def topology_fingerprint(network: RoadNetwork) -> str:
    """Return a node-name and global-pose invariant structural fingerprint."""

    incoming = {
        node: _incoming_degree(network, node)
        for node in {
            *network.graph,
            *(destination for values in network.graph.values() for destination in values),
        }
    }
    edge_rows: list[dict[str, Any]] = []
    for source, destinations in network.graph.items():
        for destination, lanes in destinations.items():
            lane_rows: list[dict[str, Any]] = []
            for lane in lanes:
                length = float(getattr(lane, "length", 0.0))
                start_heading = float(lane.heading_at(0.0))
                end_heading = float(lane.heading_at(max(0.0, length)))
                lane_rows.append(
                    {
                        "class": lane.__class__.__name__,
                        "length_mm": int(round(length * 1000.0)),
                        "heading_change_urad": int(
                            round(_wrap_to_pi(end_heading - start_heading) * 1.0e6)
                        ),
                        "forbidden": bool(getattr(lane, "forbidden", False)),
                    }
                )
            edge_rows.append(
                {
                    "source_in_degree": int(incoming.get(source, 0)),
                    "source_out_degree": int(len(network.graph.get(source, {}))),
                    "destination_in_degree": int(incoming.get(destination, 0)),
                    "destination_out_degree": int(
                        len(network.graph.get(destination, {}))
                    ),
                    "lanes": sorted(
                        lane_rows,
                        key=lambda row: json.dumps(row, sort_keys=True),
                    ),
                }
            )
    serialized = json.dumps(
        sorted(edge_rows, key=lambda row: json.dumps(row, sort_keys=True)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def exact_graph_sha256(network: RoadNetwork) -> str:
    serialized = json.dumps(
        network.to_config(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class TopologyAwareRouteIntentProvider:
    """Compute controller and policy projections from an explicit lane route."""

    def __init__(self, config: RouteIntentConfig | None = None) -> None:
        self.config = config or RouteIntentConfig()

    @property
    def policy_feature_names(self) -> tuple[str, ...]:
        names = [
            "route_available",
            "decision_present",
            "decision_distance_norm",
            "decision_lateral_offset_norm",
        ]
        for horizon in self.config.policy_horizons_m:
            suffix = f"{int(round(horizon))}m"
            names.extend(
                [
                    f"route_valid_{suffix}",
                    f"route_lateral_geometry_norm_{suffix}",
                    f"route_sin_heading_change_{suffix}",
                    f"route_cos_heading_change_{suffix}",
                ]
            )
        return tuple(names)

    @property
    def controller_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for horizon in self.config.controller_horizons_m:
            suffix = f"{int(round(horizon))}m"
            names.extend(
                [
                    f"corridor_valid_{suffix}",
                    f"corridor_forward_norm_{suffix}",
                    f"corridor_left_norm_{suffix}",
                    f"corridor_sin_heading_error_{suffix}",
                    f"corridor_cos_heading_error_{suffix}",
                    f"corridor_speed_limit_norm_{suffix}",
                    f"corridor_lane_width_norm_{suffix}",
                ]
            )
        return tuple(names)

    def contract(self) -> dict[str, Any]:
        policy_low = [0.0, 0.0, 0.0, -1.0]
        policy_high = [1.0, 1.0, 1.0, 1.0]
        for _horizon in self.config.policy_horizons_m:
            policy_low.extend([0.0, -1.0, -1.0, -1.0])
            policy_high.extend([1.0, 1.0, 1.0, 1.0])
        controller_low: list[float] = []
        controller_high: list[float] = []
        for _horizon in self.config.controller_horizons_m:
            controller_low.extend([0.0, -2.0, -2.0, -1.0, -1.0, 0.0, 0.0])
            controller_high.extend([1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        return {
            "schema_version": ROUTE_INTENT_SCHEMA_VERSION,
            "provider_id": ROUTE_INTENT_PROVIDER_ID,
            "requires_explicit_online_planner_route": True,
            "missing_route_behavior": "fail_closed",
            "policy_projection": {
                "frame": "current_route_lane_anchor",
                "feature_names": list(self.policy_feature_names),
                "feature_dim": len(self.policy_feature_names),
                "low": policy_low,
                "high": policy_high,
                "includes_current_tracking_error": False,
                "includes_previous_action": False,
                "includes_actuator_state": False,
                "includes_topology_identity": False,
            },
            "controller_projection": {
                "frame": "ego",
                "feature_names": list(self.controller_feature_names),
                "feature_dim": len(self.controller_feature_names),
                "low": controller_low,
                "high": controller_high,
            },
            "topology_sidecar": {
                "policy_visible": False,
                "uses": [
                    "domain_matched_stratification",
                    "merge_split_controls",
                    "topology_decoding_and_suppression",
                    "provenance",
                ],
            },
            "forbidden_sources": sorted(FORBIDDEN_PLANNER_SOURCE_KINDS),
            "permitted_planner_source_kinds": sorted(
                PERMITTED_PLANNER_SOURCE_KINDS
            ),
            "teacher_state_read": False,
            "expert_action_history_read": False,
            "realized_future_path_read": False,
        }

    def _resolve_route(
        self,
        network: RoadNetwork,
        *,
        current_lane_index: LaneIndex,
        route: Sequence[tuple[str, str, int | None]],
    ) -> tuple[LaneIndex, ...]:
        if not route:
            raise RouteIntentError(
                "An explicit online-planner route is required; no topology fallback "
                "or random continuation is permitted."
            )
        current = (
            str(current_lane_index[0]),
            str(current_lane_index[1]),
            int(current_lane_index[2]),
        )
        try:
            network.get_lane(current)
        except (KeyError, IndexError, TypeError) as exc:
            raise RouteIntentError(
                f"Current lane index {current!r} is absent from the road network."
            ) from exc

        requested = [
            (str(source), str(destination), None if lane_id is None else int(lane_id))
            for source, destination, lane_id in route
        ]
        if requested[0][:2] == current[:2]:
            if requested[0][2] not in {None, current[2]}:
                raise RouteIntentError(
                    "The first route lane must be the vehicle's current lane; "
                    f"got current={current!r}, route_head={requested[0]!r}."
                )
            requested[0] = current
        elif requested[0][0] == current[1]:
            requested.insert(0, current)
        else:
            raise RouteIntentError(
                "The planner route must begin on the current road or its immediate "
                f"successor; current={current!r}, route_head={requested[0]!r}."
            )

        resolved: list[LaneIndex] = [current]
        for source, destination, requested_lane_id in requested[1:]:
            previous = resolved[-1]
            if previous[1] != source:
                raise RouteIntentError(
                    "Planner route is disconnected at "
                    f"{previous!r} -> {(source, destination, requested_lane_id)!r}."
                )
            lanes = network.graph.get(source, {}).get(destination)
            if not lanes:
                raise RouteIntentError(
                    f"Planner route edge {(source, destination)!r} is absent."
                )
            if requested_lane_id is None:
                previous_lane = network.get_lane(previous)
                previous_end = previous_lane.position(previous_lane.length, 0.0)
                previous_heading = previous_lane.heading_at(previous_lane.length)
                candidates = [
                    lane_id
                    for lane_id, lane in enumerate(lanes)
                    if not bool(getattr(lane, "forbidden", False))
                ]
                if not candidates:
                    candidates = list(range(len(lanes)))

                def continuity_cost(lane_id: int) -> tuple[float, int]:
                    lane = lanes[lane_id]
                    distance = float(
                        np.linalg.norm(lane.position(0.0, 0.0) - previous_end)
                    )
                    heading = abs(
                        _wrap_to_pi(lane.heading_at(0.0) - previous_heading)
                    )
                    return distance + heading, int(lane_id)

                lane_id = min(candidates, key=continuity_cost)
            else:
                lane_id = int(requested_lane_id)
            if lane_id < 0 or lane_id >= len(lanes):
                raise RouteIntentError(
                    f"Planner lane id {lane_id} is invalid for edge "
                    f"{(source, destination)!r} with {len(lanes)} lanes."
                )
            resolved.append((source, destination, lane_id))
        return tuple(resolved)

    def _sample(
        self,
        network: RoadNetwork,
        *,
        resolved_route: Sequence[LaneIndex],
        start_longitudinal_m: float,
        horizon_m: float,
    ) -> RouteSample:
        remaining = float(horizon_m)
        for index, lane_index in enumerate(resolved_route):
            lane = network.get_lane(lane_index)
            start = float(start_longitudinal_m) if index == 0 else 0.0
            available = max(0.0, float(lane.length) - start)
            if remaining <= available + 1.0e-9:
                longitudinal = min(float(lane.length), start + max(0.0, remaining))
                position = np.asarray(lane.position(longitudinal, 0.0), dtype=float)
                heading = float(lane.heading_at(longitudinal))
                return RouteSample(
                    horizon_m=float(horizon_m),
                    valid=True,
                    lane_index=lane_index,
                    position=position,
                    heading=heading,
                    speed_limit_mps=_lane_speed_limit(lane),
                    lane_width_m=float(lane.width_at(longitudinal)),
                )
            remaining -= available
        return RouteSample(
            horizon_m=float(horizon_m),
            valid=False,
            lane_index=None,
            position=np.zeros(2, dtype=float),
            heading=0.0,
            speed_limit_mps=0.0,
            lane_width_m=0.0,
        )

    def _topology_events(
        self,
        network: RoadNetwork,
        *,
        resolved_route: Sequence[LaneIndex],
        start_longitudinal_m: float,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        distance = max(
            0.0,
            float(network.get_lane(resolved_route[0]).length)
            - float(start_longitudinal_m),
        )
        for previous, following in zip(resolved_route, resolved_route[1:]):
            previous_lane = network.get_lane(previous)
            following_lane = network.get_lane(following)
            previous_heading = float(
                previous_lane.heading_at(float(previous_lane.length))
            )
            following_heading = float(following_lane.heading_at(0.0))
            heading_change = _wrap_to_pi(following_heading - previous_heading)
            previous_end = np.asarray(
                previous_lane.position(float(previous_lane.length), 0.0),
                dtype=float,
            )
            following_start = np.asarray(
                following_lane.position(0.0, 0.0),
                dtype=float,
            )
            _forward, lateral = _frame_coordinates(
                following_start,
                origin=previous_end,
                heading=previous_heading,
            )
            outgoing_degree = int(len(network.graph.get(previous[1], {})))
            incoming_degree = _incoming_degree(network, previous[1])
            before_lanes = _lane_count(network, previous)
            after_lanes = _lane_count(network, following)
            lane_delta = int(following[2] - previous[2])
            reasons: list[str] = []
            if outgoing_degree > 1:
                reasons.append("split_or_branch")
            if incoming_degree > 1:
                reasons.append("merge_node")
            if after_lanes > before_lanes:
                reasons.append("lane_gain")
            elif after_lanes < before_lanes:
                reasons.append("lane_drop")
            if lane_delta:
                reasons.append("lane_index_change")
            if abs(heading_change) > self.config.decision_heading_threshold_rad:
                reasons.append("heading_change")
            if abs(lateral) > self.config.decision_lateral_threshold_m:
                reasons.append("lateral_transition")
            events.append(
                {
                    "distance_from_ego_m": float(distance),
                    "from_lane": list(previous),
                    "to_lane": list(following),
                    "incoming_degree": incoming_degree,
                    "outgoing_degree": outgoing_degree,
                    "lane_count_before": before_lanes,
                    "lane_count_after": after_lanes,
                    "lane_index_delta": lane_delta,
                    "heading_change_rad": float(heading_change),
                    "lateral_transition_m": float(lateral),
                    "is_decision": bool(reasons),
                    "reasons": reasons,
                }
            )
            distance += float(following_lane.length)
        return events

    def compute(
        self,
        network: RoadNetwork,
        *,
        ego_position: np.ndarray,
        ego_heading: float,
        current_lane_index: LaneIndex,
        route: Sequence[tuple[str, str, int | None]],
        planner_source_id: str,
        planner_source_kind: str,
        plan_revision: str,
        route_id: str | None = None,
        analysis_domain: str | None = None,
    ) -> RouteIntentFrame:
        """Compute one route-intent frame from an explicit causal route."""

        source_id = str(planner_source_id).strip()
        source_kind = str(planner_source_kind).strip().lower()
        revision = str(plan_revision).strip()
        if not source_id or not source_kind or not revision:
            raise RouteIntentError(
                "planner_source_id, planner_source_kind and plan_revision are required."
            )
        if source_kind in FORBIDDEN_PLANNER_SOURCE_KINDS:
            raise RouteIntentError(
                f"planner_source_kind={source_kind!r} is forbidden by the deployable "
                "route-intent contract."
            )
        if source_kind not in PERMITTED_PLANNER_SOURCE_KINDS:
            raise RouteIntentError(
                f"planner_source_kind={source_kind!r} is not one of the frozen "
                f"online source kinds {sorted(PERMITTED_PLANNER_SOURCE_KINDS)}."
            )
        position = np.asarray(ego_position, dtype=float).reshape(2)
        heading = float(ego_heading)
        if not np.all(np.isfinite(position)) or not np.isfinite(heading):
            raise RouteIntentError("Ego pose must be finite.")

        resolved = self._resolve_route(
            network,
            current_lane_index=current_lane_index,
            route=route,
        )
        current_lane = network.get_lane(resolved[0])
        start_longitudinal, _start_lateral = current_lane.local_coordinates(position)
        start_longitudinal = float(
            np.clip(start_longitudinal, 0.0, float(current_lane.length))
        )
        anchor_position = np.asarray(
            current_lane.position(start_longitudinal, 0.0), dtype=float
        )
        anchor_heading = float(current_lane.heading_at(start_longitudinal))

        controller_values: list[float] = []
        for horizon in self.config.controller_horizons_m:
            sample = self._sample(
                network,
                resolved_route=resolved,
                start_longitudinal_m=start_longitudinal,
                horizon_m=horizon,
            )
            if not sample.valid:
                controller_values.extend([0.0] * 7)
                continue
            forward, left = _frame_coordinates(
                sample.position,
                origin=position,
                heading=heading,
            )
            heading_error = _wrap_to_pi(sample.heading - heading)
            controller_values.extend(
                [
                    1.0,
                    float(np.clip(forward / horizon, -2.0, 2.0)),
                    float(np.clip(left / horizon, -2.0, 2.0)),
                    float(np.sin(heading_error)),
                    float(np.cos(heading_error)),
                    float(
                        np.clip(
                            sample.speed_limit_mps
                            / self.config.maximum_speed_mps,
                            0.0,
                            1.0,
                        )
                    ),
                    float(
                        np.clip(
                            sample.lane_width_m
                            / self.config.maximum_lane_width_m,
                            0.0,
                            1.0,
                        )
                    ),
                ]
            )

        topology_events = self._topology_events(
            network,
            resolved_route=resolved,
            start_longitudinal_m=start_longitudinal,
        )
        first_decision = next(
            (event for event in topology_events if event["is_decision"]),
            None,
        )
        policy_values = [
            1.0,
            1.0 if first_decision is not None else 0.0,
            (
                float(
                    np.clip(
                        first_decision["distance_from_ego_m"]
                        / self.config.decision_distance_cap_m,
                        0.0,
                        1.0,
                    )
                )
                if first_decision is not None
                else 0.0
            ),
            (
                float(
                    np.clip(
                        first_decision["lateral_transition_m"]
                        / self.config.decision_lateral_cap_m,
                        -1.0,
                        1.0,
                    )
                )
                if first_decision is not None
                else 0.0
            ),
        ]
        for horizon in self.config.policy_horizons_m:
            sample = self._sample(
                network,
                resolved_route=resolved,
                start_longitudinal_m=start_longitudinal,
                horizon_m=horizon,
            )
            if not sample.valid:
                policy_values.extend([0.0] * 4)
                continue
            _forward, left = _frame_coordinates(
                sample.position,
                origin=anchor_position,
                heading=anchor_heading,
            )
            heading_change = _wrap_to_pi(sample.heading - anchor_heading)
            policy_values.extend(
                [
                    1.0,
                    float(np.clip(left / horizon, -1.0, 1.0)),
                    float(np.sin(heading_change)),
                    float(np.cos(heading_change)),
                ]
            )

        policy_array = np.asarray(policy_values, dtype=np.float32)
        controller_array = np.asarray(controller_values, dtype=np.float32)
        if (
            policy_array.shape != (len(self.policy_feature_names),)
            or controller_array.shape != (len(self.controller_feature_names),)
            or not np.all(np.isfinite(policy_array))
            or not np.all(np.isfinite(controller_array))
        ):
            raise RouteIntentError("Route-intent feature construction was invalid.")

        route_sha256 = _route_digest(
            resolved,
            planner_source_id=source_id,
            plan_revision=revision,
        )
        topology = {
            "policy_visible": False,
            "analysis_domain": (
                None if analysis_domain is None else str(analysis_domain)
            ),
            "exact_graph_sha256": exact_graph_sha256(network),
            "structural_topology_fingerprint": topology_fingerprint(network),
            "current_lane_index": list(resolved[0]),
            "resolved_route_lane_count": len(resolved),
            "events": topology_events,
        }
        provenance = {
            "schema_version": ROUTE_INTENT_SCHEMA_VERSION,
            "provider_id": ROUTE_INTENT_PROVIDER_ID,
            "planner_source_id": source_id,
            "planner_source_kind": source_kind,
            "plan_revision": revision,
            "route_id": str(route_id) if route_id is not None else route_sha256[:16],
            "resolved_route_sha256": route_sha256,
            "teacher_state_read": False,
            "expert_action_history_read": False,
            "realized_future_path_read": False,
            "policy_projection_uses_ego_tracking_error": False,
            "topology_sidecar_policy_visible": False,
        }
        return RouteIntentFrame(
            policy_features=policy_array,
            controller_features=controller_array,
            policy_feature_names=self.policy_feature_names,
            controller_feature_names=self.controller_feature_names,
            resolved_route=resolved,
            topology=topology,
            provenance=provenance,
        )


def route_intent_contract(
    config: RouteIntentConfig | None = None,
) -> dict[str, Any]:
    """Return the serializable provider contract used by policy manifests."""

    return TopologyAwareRouteIntentProvider(config).contract()
