"""Build and query NGSIM road topology mappings."""

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
import numpy as np

from highway_env.ngsim_utils.core.constants import (
    FEET_PER_METER,
    KINEMATIC_HEADING_MIN_DISPLACEMENT_M,
    KINEMATIC_HEADING_MIN_SPEED_MPS,
    US101_MERGE_OUT_END_M,
    US101_SECTION_ENDS_M,
)
from highway_env.ngsim_utils.data.trajectory_gen import trajectory_row_is_active


# -------------------------------------------------------------------------
# ROAD / LANE HELPERS
# -------------------------------------------------------------------------
def us101_edge_from_x(x: float) -> tuple[str, str]:
    ends = US101_SECTION_ENDS_M
    x_m = float(x)
    if x_m < ends[1]:
        return ("s1", "s2")
    if x_m < ends[2]:
        return ("s2", "s3")
    return ("s3", "s4")


def i80_edge_from_x(x: float) -> tuple[str, str]:
    x_m = float(x)
    if x_m <= 600 / FEET_PER_METER:
        return ("s1", "s2")
    if x_m <= 700 / FEET_PER_METER:
        return ("s2", "s3")
    if x_m <= 900 / FEET_PER_METER:
        return ("s3", "s4")
    return ("s4", "s5")

def edge_from_x(net, x: float) -> tuple[str, str]:
    """
    Infer the active mainline edge for a longitudinal position x from the road graph.

    This works for both the US-101 graph (s1->s2->s3->s4) and the Japanese
    graph (a->b->c->d), and avoids assuming specific node names.
    """
    x_m = float(x)
    candidates = []
    containing = []

    for src, dsts in net.graph.items():
        for dst, lanes in dsts.items():
            if not lanes:
                continue

            lane0 = lanes[0]
            if hasattr(lane0, "start") and hasattr(lane0, "end"):
                endpoint_x = [float(lane0.start[0]), float(lane0.end[0])]
            else:
                # Curved HighwayEnv lanes (for example PolyLaneFixedWidth) do
                # not expose StraightLane's start/end attributes.  Query the
                # common AbstractLane geometry interface instead.
                endpoint_x = [
                    float(lane0.position(0.0, 0.0)[0]),
                    float(lane0.position(float(lane0.length), 0.0)[0]),
                ]
            start_x = float(min(endpoint_x))
            end_x = float(max(endpoint_x))

            # Prefer edges with multiple lanes, which represent the main carriageway.
            score = len(lanes)
            if start_x <= x_m <= end_x:
                containing.append((-score, start_x, src, dst))
                continue

            dist = min(abs(x_m - start_x), abs(x_m - end_x))
            candidates.append((dist, -score, start_x, src, dst))

    if containing:
        _, _, src, dst = min(containing)
        return (src, dst)
    if not candidates:
        raise KeyError("Road network graph does not contain any lane edges.")

    _, _, _, src, dst = min(candidates)
    return (src, dst)

def clamp_lane_id_for_x(net,
                        x: float, lane_id: int) -> int:
    edge = edge_from_x(net, x)
    n_lanes = len(net.graph[edge[0]][edge[1]])
    return int(np.clip(int(lane_id), 0, n_lanes - 1))


def _last_lane_id(net, edge: tuple[str, str]) -> int:
    return len(net.graph[edge[0]][edge[1]]) - 1


def target_lane_index_from_lane_id(
    net,
    scene: str,
    x: float,
    lane_id: int,
) -> tuple[str, str, int] | None:
    """
    Map a recorded dataset lane id to a highway-env LaneIndex on the active road graph.

    Returns None when the lane id does not map to a drivable lane in the current scene.
    """
    lane_id = int(lane_id)
    x = float(x)
    scene = str(scene)

    if scene == "us-101":
        if lane_id <= 5:
            edge = us101_edge_from_x(x)
            return (edge[0], edge[1], lane_id - 1)
        if lane_id == 6:
            edge = ("s2", "s3")
            return (edge[0], edge[1], _last_lane_id(net, edge))
        if lane_id == 7:
            # Lane 7 is the merge approach only before the first section
            # boundary.  The recorded identifier can persist briefly after
            # the ramp has joined the six-lane mainline; bind those rows to
            # the spatially valid outer mainline lane.
            edge = ("merge_in", "s2") if x < US101_SECTION_ENDS_M[1] else us101_edge_from_x(x)
            return (edge[0], edge[1], _last_lane_id(net, edge))
        if lane_id == 8:
            # Likewise, lane 8 can appear just before the modeled exit ramp.
            # It is the outer mainline lane until the ramp starts, and is not
            # representable after the modeled ramp ends.
            if x < US101_SECTION_ENDS_M[2]:
                edge = us101_edge_from_x(x)
            elif x <= US101_MERGE_OUT_END_M:
                edge = ("s3", "merge_out")
            else:
                return None
            return (edge[0], edge[1], _last_lane_id(net, edge))
        return None

    if scene == "i-80":
        if lane_id <= 6:
            edge = i80_edge_from_x(x)
            return (edge[0], edge[1], lane_id - 1)
        if lane_id == 7:
            edge = ("s1", "s2")
            return (edge[0], edge[1], _last_lane_id(net, edge))
        return None

    if scene == "japanese":
        # Dataset convention:
        #   lane_id 2 -> right mainline lane
        #   lane_id 1 -> left mainline lane
        #   lane_id 3 -> left merge lane
        x_merge_start = float(getattr(net, "japanese_merge_start_x_m", 150.0))
        x_merge_end = float(getattr(net, "japanese_merge_end_x_m", 315.0))
        if lane_id == 2:
            if x < x_merge_start:
                return ("a", "b", 0)
            if x < x_merge_end:
                return ("b", "c", 0)
            return ("c", "d", 0)
        if lane_id == 1:
            if x < x_merge_start:
                return ("a", "b", 1)
            if x < x_merge_end:
                return ("b", "c", 1)
            return ("c", "d", 1)
        if lane_id == 3:
            if x < x_merge_start:
                return ("j", "b", 0)
            if x < x_merge_end:
                return ("b", "c", 2)
            # After the merge lane disappears, fold back onto the left mainline lane.
            return ("c", "d", 1)
        return None

    return None


def resolve_target_lane_index_from_row(
    net,
    scene: str,
    row: np.ndarray,
    *,
    vehicle_width_m: float = 0.0,
    provider_observation_flag: int | None = None,
    measurement_tolerance_m: float = 0.2,
) -> tuple[tuple[str, str, int] | None, bool]:
    """Resolve a current-row lane without changing source trajectory state.

    Morinomiya provider-interpolated rows can retain a stale categorical lane
    identifier even when the current recorded pose has already moved to an
    adjacent lane.  For those rows only, replace a declared lane that does not
    overlap the current vehicle footprint with the closest overlapping lane.
    This uses no future row.  Image-detected rows and every non-Japanese scene
    retain the dataset lane-id mapping unchanged.
    """
    values = np.asarray(row, dtype=float)
    declared = target_lane_index_from_lane_id(
        net,
        scene,
        float(values[0]),
        int(values[3]),
    )
    if (
        str(scene) != "japanese"
        or provider_observation_flag != 0
        or declared is None
    ):
        return declared, False

    position = values[:2]
    overlap_margin = max(0.0, 0.5 * float(vehicle_width_m)) + max(
        0.0,
        float(measurement_tolerance_m),
    )

    declared_lane = net.get_lane(declared)
    declared_s, declared_r = declared_lane.local_coordinates(position)
    if declared_lane.on_lane(
        position,
        declared_s,
        declared_r,
        margin=overlap_margin,
    ):
        return declared, False

    overlapping: list[tuple[float, tuple[str, str, int]]] = []
    seen: set[tuple[str, str, int]] = set()
    for candidate_lane_id in (1, 2, 3):
        candidate = target_lane_index_from_lane_id(
            net,
            scene,
            float(values[0]),
            candidate_lane_id,
        )
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        lane = net.get_lane(candidate)
        longitudinal, lateral = lane.local_coordinates(position)
        if lane.on_lane(
            position,
            longitudinal,
            lateral,
            margin=overlap_margin,
        ):
            overlapping.append((abs(float(lateral)), candidate))

    if not overlapping:
        return declared, False
    resolved = min(overlapping, key=lambda item: (item[0], item[1]))[1]
    return resolved, resolved != declared


def heading_from_trajectory_row(
    net,
    scene: str,
    row: np.ndarray,
    *,
    previous_row: np.ndarray | None = None,
    next_row: np.ndarray | None = None,
    fallback_heading: float = 0.0,
    prefer_motion: bool = False,
    lane_index_override: tuple[str, str, int] | None = None,
) -> float:
    """Infer a causal heading from lane geometry or past recorded motion.

    Actor-visible state at time ``t`` must not read ``t+1``.  Reliable motion
    heading therefore uses ``previous_row -> row``.  ``next_row`` remains only
    for source compatibility with older callers and is deliberately ignored.
    ``prefer_motion`` is reserved for teleport replay; other callers retain
    the established lane-first behavior.
    """
    row_arr = np.asarray(row, dtype=float)
    x, y, _speed, lane_id = row_arr[:4]
    motion_heading = None
    if previous_row is not None and trajectory_row_is_active(previous_row):
        previous_arr = np.asarray(previous_row, dtype=float)
        dx = float(x - previous_arr[0])
        dy = float(y - previous_arr[1])
        reliable_motion = (
            float(row_arr[2]) >= KINEMATIC_HEADING_MIN_SPEED_MPS
            and np.hypot(dx, dy) >= KINEMATIC_HEADING_MIN_DISPLACEMENT_M
        )
        if reliable_motion:
            motion_heading = float(np.arctan2(dy, dx))
    del next_row
    if bool(prefer_motion) and motion_heading is not None:
        return motion_heading
    mapped_lane_index = lane_index_override
    if mapped_lane_index is None:
        mapped_lane_index = target_lane_index_from_lane_id(
            net,
            scene,
            float(x),
            int(lane_id),
        )
    if mapped_lane_index is not None:
        lane = net.get_lane(mapped_lane_index)
        local_s, _local_r = lane.local_coordinates(np.array([x, y], dtype=float))
        return float(lane.heading_at(local_s))
    if motion_heading is not None:
        return motion_heading
    return float(fallback_heading)
