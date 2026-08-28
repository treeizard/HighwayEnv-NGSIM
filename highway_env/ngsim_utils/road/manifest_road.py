"""Validate and build generic roads from the RoadGeometryV3 contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from highway_env.ngsim_utils.core.constants import (
    MORINOMIYA_MANIFEST_ENVIRONMENT_IDS,
)
from highway_env.road.lane import (
    LaneMarkingInterval,
    LineType,
    PolyLaneFixedWidth,
    normalize_marking_profile,
)
from highway_env.road.road import RoadNetwork

ROAD_GEOMETRY_V3 = "road_geometry_v3"


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is not permitted.")


def _load_json_object(
    source: str | os.PathLike[str] | Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        payload = dict(source)
        path = None
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_nonfinite_json,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"RoadGeometryV3 is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("RoadGeometryV3 must contain a JSON object.")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("RoadGeometryV3 must contain finite JSON values.") from error
    return payload, path


def _nonempty_string(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires a non-empty string field {key!r}.")
    return value.strip()


def _optional_nonempty_string(payload: Mapping[str, Any], key: str, *, label: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    return _nonempty_string(payload, key, label=label)


@dataclass(frozen=True)
class RoadNodeV3:
    node_id: str
    payload: Mapping[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class RoadLaneV3:
    lane_id: str
    polyline_xy_m: tuple[tuple[float, float], ...]
    width_m: float
    raw_lane_ids: tuple[str | int, ...]
    successors: tuple[str, ...]
    forbidden: bool
    line_types: tuple[int, int]
    speed_limit_mps: float
    priority: int
    payload: Mapping[str, Any] = field(repr=False, compare=False)
    marking_profile: tuple[LaneMarkingInterval, ...] | None = None


@dataclass(frozen=True)
class RoadEdgeV3:
    edge_id: str
    from_node: str
    to_node: str
    lanes: tuple[RoadLaneV3, ...]
    payload: Mapping[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class RoadGeometryV3:
    """Validated, extra-field-tolerant train-fitted road geometry."""

    site_id: str
    environment_id: str | None
    coordinate_frame: Mapping[str, Any]
    nodes: tuple[RoadNodeV3, ...]
    edges: tuple[RoadEdgeV3, ...]
    payload: Mapping[str, Any] = field(repr=False, compare=False)
    path: Path | None = field(default=None, repr=False, compare=False)
    schema_version: int = field(default=3, init=False)
    contract_id: str = field(default=ROAD_GEOMETRY_V3, init=False)
    fit_split: str = field(default="train", init=False)
    test_rows_used: bool = field(default=False, init=False)

    @classmethod
    def from_source(
        cls,
        source: str | os.PathLike[str] | Mapping[str, Any],
    ) -> "RoadGeometryV3":
        payload, path = _load_json_object(source)
        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 3:
            raise ValueError("RoadGeometryV3 requires schema_version=3.")
        if payload.get("contract_id") != ROAD_GEOMETRY_V3:
            raise ValueError("RoadGeometryV3 requires contract_id='road_geometry_v3'.")
        if payload.get("fit_split") != "train":
            raise ValueError("RoadGeometryV3 requires fit_split='train'.")
        if payload.get("test_rows_used") is not False:
            raise ValueError("RoadGeometryV3 requires test_rows_used=false.")
        coordinate_frame = payload.get("coordinate_frame")
        if not isinstance(coordinate_frame, Mapping):
            raise ValueError("RoadGeometryV3 requires object coordinate_frame.")
        raw_nodes = payload.get("nodes")
        raw_edges = payload.get("edges")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError("RoadGeometryV3 requires a non-empty nodes list.")
        if not isinstance(raw_edges, list) or not raw_edges:
            raise ValueError("RoadGeometryV3 requires a non-empty edges list.")
        site_id = _nonempty_string(payload, "site_id", label="RoadGeometryV3")
        environment_id = _optional_nonempty_string(payload, "environment_id", label="RoadGeometryV3")
        if environment_id in MORINOMIYA_MANIFEST_ENVIRONMENT_IDS and site_id != "morinomiya":
            raise ValueError("RoadGeometryV3 Morinomiya environment_id requires site_id='morinomiya'.")
        if environment_id == "i-80" and site_id != "i-80":
            raise ValueError("RoadGeometryV3 environment_id='i-80' requires site_id='i-80'.")
        if environment_id == "us-101" and site_id != "us-101":
            raise ValueError(
                "RoadGeometryV3 environment_id='us-101' requires site_id='us-101'."
            )
        if (
            site_id == "morinomiya"
            and environment_id is not None
            and (environment_id not in MORINOMIYA_MANIFEST_ENVIRONMENT_IDS)
        ):
            raise ValueError(f"RoadGeometryV3 has unsupported Morinomiya environment_id {environment_id!r}.")

        nodes: list[RoadNodeV3] = []
        node_ids: set[str] = set()
        for index, raw_node in enumerate(raw_nodes):
            label = f"RoadGeometryV3 node {index}"
            if isinstance(raw_node, str):
                node_id = raw_node.strip()
                node_payload: Mapping[str, Any] = {"node_id": raw_node}
            elif isinstance(raw_node, Mapping):
                node_payload = dict(raw_node)
                node_id = _nonempty_string(node_payload, "node_id", label=label)
            else:
                raise ValueError(f"{label} must be a node-id string or object.")
            if not node_id:
                raise ValueError(f"{label} requires a non-empty node id.")
            if node_id in node_ids:
                raise ValueError(f"RoadGeometryV3 contains duplicate node_id {node_id!r}.")
            node_ids.add(node_id)
            nodes.append(RoadNodeV3(node_id=node_id, payload=node_payload))

        edges: list[RoadEdgeV3] = []
        edge_ids: set[str] = set()
        lane_ids: set[str] = set()
        for edge_index, raw_edge in enumerate(raw_edges):
            label = f"RoadGeometryV3 edge {edge_index}"
            if not isinstance(raw_edge, Mapping):
                raise ValueError(f"{label} must be an object.")
            edge_payload = dict(raw_edge)
            edge_id = _nonempty_string(edge_payload, "edge_id", label=label)
            from_node = _nonempty_string(edge_payload, "from_node", label=label)
            to_node = _nonempty_string(edge_payload, "to_node", label=label)
            if edge_id in edge_ids:
                raise ValueError(f"RoadGeometryV3 contains duplicate edge_id {edge_id!r}.")
            if from_node not in node_ids or to_node not in node_ids:
                raise ValueError(f"{label} references undeclared nodes {from_node!r}->{to_node!r}.")
            raw_lanes = edge_payload.get("lanes")
            if not isinstance(raw_lanes, list) or not raw_lanes:
                raise ValueError(f"{label} requires a non-empty lanes list.")
            lanes: list[RoadLaneV3] = []
            for lane_index, raw_lane in enumerate(raw_lanes):
                lane_label = f"{label} lane {lane_index}"
                if not isinstance(raw_lane, Mapping):
                    raise ValueError(f"{lane_label} must be an object.")
                lane_payload = dict(raw_lane)
                lane_id = _nonempty_string(lane_payload, "lane_id", label=lane_label)
                if lane_id in lane_ids:
                    raise ValueError(f"RoadGeometryV3 contains duplicate lane_id {lane_id!r}.")
                points = cls._validate_polyline(lane_payload.get("polyline_xy_m"), label=lane_label)
                width = lane_payload.get("width_m")
                if isinstance(width, bool) or not isinstance(width, (int, float)):
                    raise ValueError(f"{lane_label} width_m must be numeric.")
                width_m = float(width)
                if not math.isfinite(width_m) or width_m <= 0.0:
                    raise ValueError(f"{lane_label} width_m must be finite and positive.")
                raw_lane_ids = lane_payload.get("raw_lane_ids")
                if not isinstance(raw_lane_ids, list):
                    raise ValueError(f"{lane_label} raw_lane_ids must be a list.")
                normalized_raw_ids: list[str | int] = []
                for raw_lane_id in raw_lane_ids:
                    if isinstance(raw_lane_id, bool) or not isinstance(raw_lane_id, (str, int)):
                        raise ValueError(f"{lane_label} raw_lane_ids values must be strings or integers.")
                    if isinstance(raw_lane_id, str) and not raw_lane_id.strip():
                        raise ValueError(f"{lane_label} raw_lane_ids may not contain empty strings.")
                    normalized_raw_ids.append(raw_lane_id)
                raw_successors = lane_payload.get("successors")
                if not isinstance(raw_successors, list):
                    raise ValueError(f"{lane_label} successors must be a list.")
                successors: list[str] = []
                for successor in raw_successors:
                    if not isinstance(successor, str) or not successor.strip():
                        raise ValueError(f"{lane_label} successor references must be non-empty strings.")
                    successors.append(successor.strip())
                forbidden = lane_payload.get("forbidden")
                if not isinstance(forbidden, bool):
                    raise ValueError(f"{lane_label} forbidden must be boolean.")
                raw_line_types = lane_payload.get("line_types", [LineType.STRIPED, LineType.STRIPED])
                if (
                    not isinstance(raw_line_types, list)
                    or len(raw_line_types) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value
                        not in {
                            LineType.NONE,
                            LineType.STRIPED,
                            LineType.CONTINUOUS,
                            LineType.CONTINUOUS_LINE,
                        }
                        for value in raw_line_types
                    )
                ):
                    raise ValueError(f"{lane_label} line_types must contain two valid LineType integers.")
                point_deltas = np.diff(np.asarray(points, dtype=np.float64), axis=0)
                # Match LinearSpline2D's sequential accumulation exactly so a
                # validated full partition remains exact when the runtime lane
                # normalizes it a second time.
                lane_length_m = float(
                    np.cumsum(
                        np.sqrt(point_deltas[:, 0] ** 2 + point_deltas[:, 1] ** 2)
                    )[-1]
                )
                if "marking_profile" in lane_payload and lane_payload["marking_profile"] is None:
                    raise ValueError(f"{lane_label} marking_profile must be a non-empty ordered list of intervals.")
                marking_profile = normalize_marking_profile(
                    lane_payload.get("marking_profile"),
                    lane_length=lane_length_m,
                    label=f"{lane_label} marking_profile",
                )
                raw_speed_limit = lane_payload.get("speed_limit_mps", 20.0)
                if isinstance(raw_speed_limit, bool) or not isinstance(raw_speed_limit, (int, float)):
                    raise ValueError(f"{lane_label} speed_limit_mps must be numeric.")
                speed_limit_mps = float(raw_speed_limit)
                if not math.isfinite(speed_limit_mps) or speed_limit_mps <= 0.0:
                    raise ValueError(f"{lane_label} speed_limit_mps must be finite and positive.")
                raw_priority = lane_payload.get("priority", 0)
                if isinstance(raw_priority, bool) or not isinstance(raw_priority, int):
                    raise ValueError(f"{lane_label} priority must be an integer.")
                lanes.append(
                    RoadLaneV3(
                        lane_id=lane_id,
                        polyline_xy_m=points,
                        width_m=width_m,
                        raw_lane_ids=tuple(normalized_raw_ids),
                        successors=tuple(successors),
                        forbidden=forbidden,
                        line_types=(int(raw_line_types[0]), int(raw_line_types[1])),
                        marking_profile=marking_profile,
                        speed_limit_mps=speed_limit_mps,
                        priority=int(raw_priority),
                        payload=lane_payload,
                    )
                )
                lane_ids.add(lane_id)
            edges.append(
                RoadEdgeV3(
                    edge_id=edge_id,
                    from_node=from_node,
                    to_node=to_node,
                    lanes=tuple(lanes),
                    payload=edge_payload,
                )
            )
            edge_ids.add(edge_id)

        lane_edge_by_id = {lane.lane_id: edge for edge in edges for lane in edge.lanes}
        for edge in edges:
            for lane in edge.lanes:
                missing = sorted(set(lane.successors).difference(lane_ids))
                if missing:
                    raise ValueError(f"RoadGeometryV3 lane {lane.lane_id!r} has undeclared successors {missing!r}.")
                disconnected = sorted(
                    successor for successor in lane.successors if lane_edge_by_id[successor].from_node != edge.to_node
                )
                if disconnected:
                    raise ValueError(
                        f"RoadGeometryV3 lane {lane.lane_id!r} has successor lanes "
                        f"that do not start at node {edge.to_node!r}: {disconnected!r}."
                    )

        return cls(
            site_id=site_id,
            environment_id=environment_id,
            coordinate_frame=dict(coordinate_frame),
            nodes=tuple(nodes),
            edges=tuple(edges),
            payload=payload,
            path=path,
        )

    @staticmethod
    def _validate_polyline(
        raw_points: Any,
        *,
        label: str,
    ) -> tuple[tuple[float, float], ...]:
        try:
            values = np.asarray(raw_points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} polyline_xy_m must be numeric.") from error
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            raise ValueError(f"{label} polyline_xy_m must have shape [N>=2, 2].")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} polyline_xy_m must contain finite coordinates.")
        segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
        if np.any(segment_lengths <= 0.0):
            raise ValueError(f"{label} polyline_xy_m may not contain consecutive duplicate points.")
        return tuple((float(point[0]), float(point[1])) for point in values)

    @property
    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            self.payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_dataset(self, dataset: Any) -> None:
        if str(getattr(dataset, "site_id", "")) != self.site_id:
            raise ValueError("DatasetManifestV2 and RoadGeometryV3 site_id values do not match.")
        dataset_frame = getattr(dataset, "coordinate_frame", None)
        if dataset_frame != self.coordinate_frame:
            raise ValueError("DatasetManifestV2 and RoadGeometryV3 coordinate_frame values do not match.")
        dataset_environment_id = getattr(dataset, "environment_id", None)
        if dataset_environment_id != self.environment_id:
            raise ValueError("DatasetManifestV2 and RoadGeometryV3 environment_id values do not match.")


def build_road_network(geometry: RoadGeometryV3) -> RoadNetwork:
    """Build a HighwayEnv RoadNetwork while preserving manifest lane order."""

    net = RoadNetwork()
    lane_index_by_id: dict[str, tuple[str, str, int]] = {}
    lane_successors: dict[str, tuple[str, ...]] = {}
    raw_lane_indexes: dict[str | int, list[tuple[str, str, int]]] = {}
    for edge in geometry.edges:
        existing_lanes = len(net.graph.get(edge.from_node, {}).get(edge.to_node, ()))
        for lane_offset, lane in enumerate(edge.lanes):
            lane_number = existing_lanes + lane_offset
            lane_index = (edge.from_node, edge.to_node, lane_number)
            # LinearSpline2D duplicates its terminal sample when the total
            # polyline length is an exact integer number of metres. The two
            # duplicate poses have zero-length tangents. Remove only those
            # derived cache entries; the source polyline/interpolator remains
            # byte-for-byte geometrically unchanged.
            with np.errstate(divide="ignore", invalid="ignore"):
                highway_lane = PolyLaneFixedWidth(
                    list(lane.polyline_xy_m),
                    width=lane.width_m,
                    line_types=list(lane.line_types),
                    marking_profile=lane.marking_profile,
                    forbidden=lane.forbidden,
                    speed_limit=lane.speed_limit_mps,
                    priority=lane.priority,
                )
            finite_pose = np.asarray(
                [
                    np.all(np.isfinite(pose.normal)) and float(getattr(pose, "length", 0.0)) > 0.0
                    for pose in highway_lane.curve.poses
                ],
                dtype=bool,
            )
            if not np.all(finite_pose):
                highway_lane.curve.s_samples = highway_lane.curve.s_samples[finite_pose]
                highway_lane.curve.poses = [pose for pose, keep in zip(highway_lane.curve.poses, finite_pose) if keep]
                if not highway_lane.curve.poses:
                    raise ValueError(f"RoadGeometryV3 lane {lane.lane_id!r} has no valid tangent.")
            net.add_lane(
                edge.from_node,
                edge.to_node,
                highway_lane,
            )
            lane_index_by_id[lane.lane_id] = lane_index
            lane_successors[lane.lane_id] = lane.successors
            for raw_lane_id in lane.raw_lane_ids:
                raw_lane_indexes.setdefault(raw_lane_id, []).append(lane_index)

    net.road_geometry_contract = ROAD_GEOMETRY_V3
    net.road_geometry_sha256 = geometry.canonical_sha256
    net.site_id = geometry.site_id
    net.environment_id = geometry.environment_id
    net.coordinate_frame = dict(geometry.coordinate_frame)
    net.manifest_lane_index_by_id = lane_index_by_id
    net.manifest_lane_successors = lane_successors
    net.manifest_lane_id_by_index = {lane_index: lane_id for lane_id, lane_index in lane_index_by_id.items()}
    net.manifest_lane_successor_indexes = {
        lane_index_by_id[lane_id]: tuple(lane_index_by_id[successor_id] for successor_id in successors)
        for lane_id, successors in lane_successors.items()
    }
    net.manifest_raw_lane_indexes = {raw_lane_id: tuple(indexes) for raw_lane_id, indexes in raw_lane_indexes.items()}
    return net
