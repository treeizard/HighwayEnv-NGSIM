"""Path-free behavior commands and frozen 10 Hz maneuver annotation.

The policy-visible contract is deliberately only a four-way categorical
command.  Lane identities, map topology, lead-vehicle identities, future
positions, and annotation state are labeler/scheduler inputs and never actor
features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

import numpy as np

BEHAVIOR_INTENT_SCHEMA_VERSION = 1
BEHAVIOR_INTENT_DIM = 4
MANEUVER_FAMILY_SCHEMA_VERSION = 1
MANEUVER_FAMILY_DIM = 3
INVALID_BEHAVIOR_ID = -1


class BehaviorIntent(IntEnum):
    """Stable behavior-command IDs used by data, policies, and interventions."""

    KEEP_LANE = 0
    CHANGE_LEFT = 1
    CHANGE_RIGHT = 2
    OVERTAKE = 3


class ManeuverFamilyIntent(IntEnum):
    """Coarse observational behavior families for versioned recovery trials.

    This projection pools only the two lateral directions.  It is derived from
    immutable raw-trajectory annotations and must not be presented as a
    prospective route command unless a separate current-state provider is
    qualified.
    """

    KEEP_LANE = 0
    LANE_CHANGE = 1
    OVERTAKE = 2


def behavior_intent_contract() -> dict[str, Any]:
    """Return the complete path-free actor-input contract."""
    return {
        "schema_version": BEHAVIOR_INTENT_SCHEMA_VERSION,
        "contract_id": "path_free_behavior_intent_one_hot_v1",
        "feature_dim": BEHAVIOR_INTENT_DIM,
        "encoding": "one_hot",
        "classes": [
            {"id": int(intent), "name": intent.name}
            for intent in BehaviorIntent
        ],
        "policy_visible": True,
        "unstandardized": True,
        "overtake_direction_inferred_from_current_traffic_and_lane_observations": True,
        "forbidden_policy_inputs": [
            "route",
            "future_coordinates",
            "future_state",
            "target_vehicle_id",
            "lane_id",
            "topology_id",
            "domain_id",
            "controller_state",
        ],
    }


def behavior_intent_one_hot(
    behavior_ids: np.ndarray | Sequence[int] | int,
) -> np.ndarray:
    """Validate behavior IDs and return finite float32 one-hot rows."""
    ids = np.asarray(behavior_ids)
    scalar = ids.ndim == 0
    ids = ids.reshape(-1)
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
            raise ValueError("Behavior IDs must be finite integers.")
        ids = ids.astype(np.int64)
    else:
        ids = ids.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.any(ids >= BEHAVIOR_INTENT_DIM):
        raise ValueError(
            "Behavior IDs must be in "
            f"[0, {BEHAVIOR_INTENT_DIM - 1}], got {ids.tolist()}."
        )
    features = np.eye(BEHAVIOR_INTENT_DIM, dtype=np.float32)[ids]
    return features[0] if scalar else features


def behavior_to_maneuver_family_ids(
    behavior_ids: np.ndarray | Sequence[int] | int,
) -> np.ndarray:
    """Project four-way annotations to KEEP/LANE_CHANGE/OVERTAKE IDs."""

    ids = np.asarray(behavior_ids)
    scalar = ids.ndim == 0
    ids = ids.reshape(-1)
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
            raise ValueError("Behavior IDs must be finite integers.")
        ids = ids.astype(np.int64)
    else:
        ids = ids.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.any(ids >= BEHAVIOR_INTENT_DIM):
        raise ValueError(
            "Behavior IDs must be in "
            f"[0, {BEHAVIOR_INTENT_DIM - 1}], got {ids.tolist()}."
        )
    mapping = np.asarray(
        [
            int(ManeuverFamilyIntent.KEEP_LANE),
            int(ManeuverFamilyIntent.LANE_CHANGE),
            int(ManeuverFamilyIntent.LANE_CHANGE),
            int(ManeuverFamilyIntent.OVERTAKE),
        ],
        dtype=np.int8,
    )
    projected = mapping[ids]
    return projected[0] if scalar else projected


def maneuver_family_one_hot(
    behavior_ids: np.ndarray | Sequence[int] | int,
) -> np.ndarray:
    """Encode the versioned three-family projection from four-way labels."""

    projected = np.asarray(behavior_to_maneuver_family_ids(behavior_ids))
    scalar = projected.ndim == 0
    projected = projected.reshape(-1).astype(np.int64, copy=False)
    features = np.eye(MANEUVER_FAMILY_DIM, dtype=np.float32)[projected]
    return features[0] if scalar else features


def maneuver_family_contract() -> dict[str, Any]:
    """Return the explicit claim boundary for the pooled recovery ontology."""

    return {
        "schema_version": MANEUVER_FAMILY_SCHEMA_VERSION,
        "contract_id": "observational_maneuver_family_projection_v1",
        "feature_dim": MANEUVER_FAMILY_DIM,
        "encoding": "one_hot",
        "classes": [
            {"id": int(intent), "name": intent.name}
            for intent in ManeuverFamilyIntent
        ],
        "source_behavior_contract": behavior_intent_contract()["contract_id"],
        "source_to_family": {
            BehaviorIntent.KEEP_LANE.name: ManeuverFamilyIntent.KEEP_LANE.name,
            BehaviorIntent.CHANGE_LEFT.name: ManeuverFamilyIntent.LANE_CHANGE.name,
            BehaviorIntent.CHANGE_RIGHT.name: ManeuverFamilyIntent.LANE_CHANGE.name,
            BehaviorIntent.OVERTAKE.name: ManeuverFamilyIntent.OVERTAKE.name,
        },
        "prospective_command_qualified": False,
        "claim_boundary": (
            "Retrospective raw-trajectory maneuver family for support and "
            "observability trials; not a deployable route-intent claim."
        ),
    }


@dataclass(frozen=True)
class BehaviorLabelerConfig:
    """Frozen 10 Hz behavior-label definition."""

    sample_hz: int = 10
    stable_lane_seconds: float = 0.5
    lane_change_pre_seconds: float = 2.0
    lane_change_post_stabilization_seconds: float = 1.0
    overtake_min_initial_gap_m: float = 5.0
    overtake_max_initial_gap_m: float = 60.0
    overtake_min_clearance_m: float = 3.0
    overtake_max_seconds: float = 10.0
    lane_ordinal_increases_to_left: bool = True

    def __post_init__(self) -> None:
        if int(self.sample_hz) != 10:
            raise ValueError("Behavior label schema v1 is frozen at 10 Hz.")
        positive = (
            self.stable_lane_seconds,
            self.lane_change_pre_seconds,
            self.lane_change_post_stabilization_seconds,
            self.overtake_min_initial_gap_m,
            self.overtake_max_initial_gap_m,
            self.overtake_min_clearance_m,
            self.overtake_max_seconds,
        )
        if not all(np.isfinite(value) and float(value) > 0.0 for value in positive):
            raise ValueError("Behavior labeler thresholds must be finite and positive.")
        if self.overtake_min_initial_gap_m >= self.overtake_max_initial_gap_m:
            raise ValueError("Overtake initial-gap bounds are reversed.")

    @property
    def stable_lane_steps(self) -> int:
        return int(round(self.stable_lane_seconds * self.sample_hz))

    @property
    def lane_change_pre_steps(self) -> int:
        return int(round(self.lane_change_pre_seconds * self.sample_hz))

    @property
    def lane_change_post_steps(self) -> int:
        return int(
            round(self.lane_change_post_stabilization_seconds * self.sample_hz)
        )

    @property
    def overtake_max_steps(self) -> int:
        return int(round(self.overtake_max_seconds * self.sample_hz))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BEHAVIOR_INTENT_SCHEMA_VERSION,
            **vars(self),
            "stable_lane_steps": self.stable_lane_steps,
            "lane_change_pre_steps": self.lane_change_pre_steps,
            "lane_change_post_steps": self.lane_change_post_steps,
            "overtake_max_steps": self.overtake_max_steps,
        }


@dataclass(frozen=True)
class BehaviorLabels:
    behavior_ids: np.ndarray
    next_behavior_ids: np.ndarray
    segment_ids: np.ndarray
    label_valid: np.ndarray
    invalid_reasons: np.ndarray


def _as_1d(name: str, values: Any, *, length: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got {array.shape}.")
    if length is not None and len(array) != int(length):
        raise ValueError(f"{name} length {len(array)} does not match {length}.")
    return array


def _contiguous(indices: np.ndarray, timesteps: np.ndarray) -> bool:
    return bool(
        len(indices) > 0
        and (
            len(indices) == 1
            or np.all(np.diff(timesteps[indices].astype(np.int64)) == 1)
        )
    )


def _lane_stable(
    indices: np.ndarray,
    lane_groups: np.ndarray,
    lane_ordinals: np.ndarray,
    lane_valid: np.ndarray,
    timesteps: np.ndarray,
) -> bool:
    return bool(
        _contiguous(indices, timesteps)
        and np.all(lane_valid[indices])
        and len(set(lane_groups[indices].tolist())) == 1
        and len(set(lane_ordinals[indices].astype(np.int64).tolist())) == 1
    )


def _assign_or_invalidate(
    behavior_ids: np.ndarray,
    label_valid: np.ndarray,
    invalid_reasons: np.ndarray,
    indices: np.ndarray,
    behavior: BehaviorIntent,
    *,
    precedence: bool = False,
) -> None:
    if not len(indices):
        return
    protected_invalid = (~label_valid[indices]) & (
        invalid_reasons[indices] != "inactive_or_missing_lane"
    )
    conflicts = label_valid[indices] & (behavior_ids[indices] != int(behavior))
    if bool(np.any(conflicts)) and not precedence:
        conflicting_indices = indices[conflicts]
        behavior_ids[conflicting_indices] = INVALID_BEHAVIOR_ID
        label_valid[conflicting_indices] = False
        invalid_reasons[conflicting_indices] = "conflicting_overlap"
    if not precedence:
        indices = indices[~(conflicts | protected_invalid)]
    behavior_ids[indices] = int(behavior)
    label_valid[indices] = True
    invalid_reasons[indices] = ""


def label_behavior_rows(
    *,
    vehicle_ids: np.ndarray | Sequence[int],
    timesteps: np.ndarray | Sequence[int],
    lane_group_ids: np.ndarray | Sequence[Hashable],
    lane_ordinals: np.ndarray | Sequence[int],
    longitudinal_positions_m: np.ndarray | Sequence[float],
    lane_valid: np.ndarray | Sequence[bool],
    active: np.ndarray | Sequence[bool] | None = None,
    config: BehaviorLabelerConfig | None = None,
) -> BehaviorLabels:
    """Label aligned scene rows using only annotation-side lane/traffic state.

    Rows may be in any order.  Lane ordinal is interpreted within a lane group;
    ordinal zero is valid whenever ``lane_valid`` is true.  Longitudinal
    positions are lane-local, so the result is invariant to global coordinate
    translation and rotation.
    """
    cfg = config or BehaviorLabelerConfig()
    vehicle_ids_arr = _as_1d("vehicle_ids", vehicle_ids)
    n = len(vehicle_ids_arr)
    timesteps_arr = _as_1d("timesteps", timesteps, length=n).astype(
        np.int64, copy=False
    )
    lane_groups_arr = _as_1d("lane_group_ids", lane_group_ids, length=n)
    lane_ordinals_arr = _as_1d(
        "lane_ordinals", lane_ordinals, length=n
    ).astype(np.int64, copy=False)
    longitudinal_arr = _as_1d(
        "longitudinal_positions_m", longitudinal_positions_m, length=n
    ).astype(np.float64, copy=False)
    lane_valid_arr = _as_1d("lane_valid", lane_valid, length=n).astype(
        bool, copy=False
    )
    active_arr = (
        np.ones(n, dtype=bool)
        if active is None
        else _as_1d("active", active, length=n).astype(bool, copy=False)
    )
    if len(np.unique(np.column_stack((vehicle_ids_arr, timesteps_arr)), axis=0)) != n:
        raise ValueError("Behavior label rows contain duplicate vehicle/timestep keys.")
    finite_position = np.isfinite(longitudinal_arr)
    usable = active_arr & lane_valid_arr & finite_position

    behavior_ids = np.full(n, INVALID_BEHAVIOR_ID, dtype=np.int8)
    label_valid_out = np.zeros(n, dtype=bool)
    invalid_reasons = np.full(n, "inactive_or_missing_lane", dtype="<U48")

    rows_by_vehicle: dict[int, np.ndarray] = {}
    row_for_key: dict[tuple[int, int], int] = {}
    for vehicle_id in np.unique(vehicle_ids_arr):
        indices = np.flatnonzero(vehicle_ids_arr == vehicle_id)
        order = np.argsort(timesteps_arr[indices], kind="stable")
        rows = indices[order]
        rows_by_vehicle[int(vehicle_id)] = rows
        for row in rows:
            row_for_key[(int(vehicle_id), int(timesteps_arr[row]))] = int(row)

    stable = cfg.stable_lane_steps
    for vehicle_id, rows in rows_by_vehicle.items():
        for position in range(1, len(rows)):
            previous_row = int(rows[position - 1])
            current_row = int(rows[position])
            if timesteps_arr[current_row] != timesteps_arr[previous_row] + 1:
                continue
            previous_lane = (
                lane_groups_arr[previous_row],
                int(lane_ordinals_arr[previous_row]),
            )
            current_lane = (
                lane_groups_arr[current_row],
                int(lane_ordinals_arr[current_row]),
            )
            if previous_lane == current_lane:
                continue
            before = rows[max(0, position - stable) : position]
            after = rows[position : min(len(rows), position + stable)]
            local_start = max(0, position - cfg.lane_change_pre_steps)
            local_stop = min(
                len(rows),
                position + stable + cfg.lane_change_post_steps,
            )
            window = rows[local_start:local_stop]
            adjacent = bool(
                lane_groups_arr[previous_row] == lane_groups_arr[current_row]
                and abs(
                    int(lane_ordinals_arr[current_row])
                    - int(lane_ordinals_arr[previous_row])
                )
                == 1
            )
            stable_boundary = (
                len(before) == stable
                and len(after) == stable
                and _lane_stable(
                    before,
                    lane_groups_arr,
                    lane_ordinals_arr,
                    lane_valid_arr & active_arr,
                    timesteps_arr,
                )
                and _lane_stable(
                    after,
                    lane_groups_arr,
                    lane_ordinals_arr,
                    lane_valid_arr & active_arr,
                    timesteps_arr,
                )
            )
            if not adjacent or not stable_boundary:
                behavior_ids[window] = INVALID_BEHAVIOR_ID
                label_valid_out[window] = False
                invalid_reasons[window] = (
                    "non_adjacent_lane_jump"
                    if not adjacent
                    else "unstable_lane_transition"
                )
                continue
            delta = (
                int(lane_ordinals_arr[current_row])
                - int(lane_ordinals_arr[previous_row])
            )
            goes_left = delta > 0 if cfg.lane_ordinal_increases_to_left else delta < 0
            behavior = (
                BehaviorIntent.CHANGE_LEFT
                if goes_left
                else BehaviorIntent.CHANGE_RIGHT
            )
            _assign_or_invalidate(
                behavior_ids,
                label_valid_out,
                invalid_reasons,
                window,
                behavior,
            )

    # Detect passes from a stable same-lane lead.  Each target must be present
    # without gaps until longitudinal order reverses with frozen clearance.
    for ego_id, ego_rows in rows_by_vehicle.items():
        occupied_by_timestep = {
            int(timesteps_arr[row]): int(row) for row in ego_rows if usable[row]
        }
        consumed_until = -1
        for ego_row in ego_rows:
            start_t = int(timesteps_arr[ego_row])
            if start_t <= consumed_until or not usable[ego_row]:
                continue
            group = lane_groups_arr[ego_row]
            ordinal = int(lane_ordinals_arr[ego_row])
            ego_s = float(longitudinal_arr[ego_row])
            lead_candidates: list[tuple[float, int, int]] = []
            for other_id, other_rows in rows_by_vehicle.items():
                if other_id == ego_id:
                    continue
                other_row = row_for_key.get((other_id, start_t))
                if other_row is None or not usable[other_row]:
                    continue
                if (
                    lane_groups_arr[other_row] != group
                    or int(lane_ordinals_arr[other_row]) != ordinal
                ):
                    continue
                gap = float(longitudinal_arr[other_row]) - ego_s
                if (
                    cfg.overtake_min_initial_gap_m
                    <= gap
                    <= cfg.overtake_max_initial_gap_m
                ):
                    stable_timesteps = range(start_t, start_t + stable)
                    ego_stable_rows = np.asarray(
                        [
                            row_for_key.get((ego_id, timestep), -1)
                            for timestep in stable_timesteps
                        ],
                        dtype=np.int64,
                    )
                    target_stable_rows = np.asarray(
                        [
                            row_for_key.get((other_id, timestep), -1)
                            for timestep in stable_timesteps
                        ],
                        dtype=np.int64,
                    )
                    if np.any(ego_stable_rows < 0) or np.any(
                        target_stable_rows < 0
                    ):
                        continue
                    if not (
                        _lane_stable(
                            ego_stable_rows,
                            lane_groups_arr,
                            lane_ordinals_arr,
                            lane_valid_arr & active_arr,
                            timesteps_arr,
                        )
                        and _lane_stable(
                            target_stable_rows,
                            lane_groups_arr,
                            lane_ordinals_arr,
                            lane_valid_arr & active_arr,
                            timesteps_arr,
                        )
                        and np.all(
                            lane_groups_arr[target_stable_rows]
                            == lane_groups_arr[ego_stable_rows]
                        )
                        and np.all(
                            lane_ordinals_arr[target_stable_rows]
                            == lane_ordinals_arr[ego_stable_rows]
                        )
                    ):
                        continue
                    stable_gaps = (
                        longitudinal_arr[target_stable_rows]
                        - longitudinal_arr[ego_stable_rows]
                    )
                    if not np.all(
                        (stable_gaps >= cfg.overtake_min_initial_gap_m)
                        & (stable_gaps <= cfg.overtake_max_initial_gap_m)
                    ):
                        continue
                    lead_candidates.append((gap, other_id, other_row))
            if not lead_candidates:
                continue
            _gap, target_id, _target_start_row = min(lead_candidates)
            observed_rows: list[int] = []
            left_original_lane = False
            completion_t: int | None = None
            missing_target = False
            for offset in range(cfg.overtake_max_steps + 1):
                timestep = start_t + offset
                current_ego_row = occupied_by_timestep.get(timestep)
                current_target_row = row_for_key.get((target_id, timestep))
                if current_ego_row is None or current_target_row is None:
                    missing_target = offset > 0
                    break
                if not usable[current_ego_row] or not active_arr[current_target_row]:
                    missing_target = offset > 0
                    break
                observed_rows.append(current_ego_row)
                left_original_lane = bool(
                    left_original_lane
                    or lane_groups_arr[current_ego_row] != group
                    or int(lane_ordinals_arr[current_ego_row]) != ordinal
                )
                clearance = (
                    float(longitudinal_arr[current_ego_row])
                    - float(longitudinal_arr[current_target_row])
                )
                if left_original_lane and clearance >= cfg.overtake_min_clearance_m:
                    completion_t = timestep
                    break
            if completion_t is None:
                if left_original_lane and observed_rows:
                    invalid = np.asarray(observed_rows, dtype=np.int64)
                    behavior_ids[invalid] = INVALID_BEHAVIOR_ID
                    label_valid_out[invalid] = False
                    invalid_reasons[invalid] = (
                        "missing_overtake_target"
                        if missing_target
                        else "incomplete_overtake"
                    )
                continue
            overtake_rows = np.asarray(observed_rows, dtype=np.int64)
            _assign_or_invalidate(
                behavior_ids,
                label_valid_out,
                invalid_reasons,
                overtake_rows,
                BehaviorIntent.OVERTAKE,
                precedence=True,
            )
            consumed_until = int(completion_t)

    # KEEP_LANE is assigned only to contiguous stable-lane runs outside all
    # maneuver/invalid windows.
    for rows in rows_by_vehicle.values():
        run_start = 0
        while run_start < len(rows):
            first = int(rows[run_start])
            if not usable[first]:
                run_start += 1
                continue
            run_stop = run_start + 1
            while run_stop < len(rows):
                previous = int(rows[run_stop - 1])
                current = int(rows[run_stop])
                if (
                    not usable[current]
                    or timesteps_arr[current] != timesteps_arr[previous] + 1
                    or lane_groups_arr[current] != lane_groups_arr[first]
                    or lane_ordinals_arr[current] != lane_ordinals_arr[first]
                ):
                    break
                run_stop += 1
            run = rows[run_start:run_stop]
            if len(run) >= stable:
                unassigned = run[invalid_reasons[run] == "inactive_or_missing_lane"]
                _assign_or_invalidate(
                    behavior_ids,
                    label_valid_out,
                    invalid_reasons,
                    unassigned,
                    BehaviorIntent.KEEP_LANE,
                )
            run_start = max(run_stop, run_start + 1)

    # ``label_valid`` is transition-level: both the current command and the
    # command appended to the next observation must be known.  Exclude the
    # outgoing transition immediately before an invalid/discontinuous row
    # instead of inventing a next command.  The annotation snapshot lets that
    # row still provide a truthful next command for its predecessor.
    annotation_valid = label_valid_out.copy()
    annotation_behavior_ids = behavior_ids.copy()
    next_behavior_ids = np.full(n, INVALID_BEHAVIOR_ID, dtype=np.int8)
    for rows in rows_by_vehicle.values():
        for position, row_value in enumerate(rows):
            row = int(row_value)
            if not annotation_valid[row]:
                continue
            if position == len(rows) - 1:
                next_behavior_ids[row] = annotation_behavior_ids[row]
                continue
            next_row = int(rows[position + 1])
            if (
                timesteps_arr[next_row] == timesteps_arr[row] + 1
                and annotation_valid[next_row]
            ):
                next_behavior_ids[row] = annotation_behavior_ids[next_row]
                continue
            label_valid_out[row] = False
            invalid_reasons[row] = "invalid_or_missing_next_behavior"

    segment_ids = np.full(n, -1, dtype=np.int64)
    next_segment_id = 0
    for rows in rows_by_vehicle.values():
        previous_row: int | None = None
        current_segment = -1
        for row_value in rows:
            row = int(row_value)
            starts_segment = bool(
                label_valid_out[row]
                and (
                    previous_row is None
                    or not label_valid_out[previous_row]
                    or timesteps_arr[row] != timesteps_arr[previous_row] + 1
                    or behavior_ids[row] != behavior_ids[previous_row]
                )
            )
            if starts_segment:
                current_segment = next_segment_id
                next_segment_id += 1
            if label_valid_out[row]:
                segment_ids[row] = current_segment
            previous_row = row

    return BehaviorLabels(
        behavior_ids=behavior_ids,
        next_behavior_ids=next_behavior_ids,
        segment_ids=segment_ids,
        label_valid=label_valid_out,
        invalid_reasons=invalid_reasons,
    )


@dataclass(frozen=True)
class BehaviorCommandSchedulerConfig:
    """Frozen prospective command schedule; it never emits actions."""

    schema_version: int = 1
    schedule_id: str = "feasible_uniform_behavior_commands_v1"
    keep_lane_horizon_steps: int = 20
    lane_change_horizon_steps: int = 40
    overtake_horizon_steps: int = 100
    overtake_min_lead_gap_m: float = 5.0
    overtake_max_lead_gap_m: float = 60.0

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError("Behavior command scheduler schema_version must be 1.")
        if not str(self.schedule_id):
            raise ValueError("Behavior command scheduler requires schedule_id.")
        if min(
            int(self.keep_lane_horizon_steps),
            int(self.lane_change_horizon_steps),
            int(self.overtake_horizon_steps),
        ) < 1:
            raise ValueError("Behavior command horizons must be positive.")
        if not (
            0.0
            < float(self.overtake_min_lead_gap_m)
            < float(self.overtake_max_lead_gap_m)
        ):
            raise ValueError("Behavior scheduler overtake-gap bounds are invalid.")

    @classmethod
    def from_json(
        cls, path: str | Path
    ) -> "BehaviorCommandSchedulerConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Behavior command schedule must be a JSON object.")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class BehaviorFeasibility:
    keep_lane: bool
    change_left: bool
    change_right: bool
    overtake: bool

    def feasible(self, intent: BehaviorIntent) -> bool:
        return {
            BehaviorIntent.KEEP_LANE: self.keep_lane,
            BehaviorIntent.CHANGE_LEFT: self.change_left,
            BehaviorIntent.CHANGE_RIGHT: self.change_right,
            BehaviorIntent.OVERTAKE: self.overtake,
        }[intent]


@dataclass
class _HeldCommand:
    intent: BehaviorIntent
    remaining_steps: int


class ProspectiveBehaviorCommandScheduler:
    """Select feasible commands from current state/map and hold their horizon."""

    def __init__(
        self,
        config: BehaviorCommandSchedulerConfig,
        *,
        seed: int,
    ) -> None:
        self.config = config
        self._rng = np.random.default_rng(int(seed))
        self._held: dict[Hashable, _HeldCommand] = {}
        self.attempt_counts = np.zeros(BEHAVIOR_INTENT_DIM, dtype=np.int64)
        self.infeasible_attempt_counts = np.zeros(
            BEHAVIOR_INTENT_DIM, dtype=np.int64
        )

    def _horizon(self, intent: BehaviorIntent) -> int:
        if intent == BehaviorIntent.KEEP_LANE:
            return int(self.config.keep_lane_horizon_steps)
        if intent in (BehaviorIntent.CHANGE_LEFT, BehaviorIntent.CHANGE_RIGHT):
            return int(self.config.lane_change_horizon_steps)
        return int(self.config.overtake_horizon_steps)

    def command_for(
        self,
        key: Hashable,
        feasibility: BehaviorFeasibility,
    ) -> BehaviorIntent:
        held = self._held.get(key)
        if held is not None and held.remaining_steps > 0:
            return held.intent
        candidates = list(BehaviorIntent)
        order = self._rng.permutation(len(candidates))
        selected = BehaviorIntent.KEEP_LANE
        for index in order:
            attempted = candidates[int(index)]
            self.attempt_counts[int(attempted)] += 1
            if feasibility.feasible(attempted):
                selected = attempted
                break
            self.infeasible_attempt_counts[int(attempted)] += 1
        self._held[key] = _HeldCommand(
            intent=selected,
            remaining_steps=self._horizon(selected),
        )
        return selected

    def advance(self, keys: Iterable[Hashable]) -> None:
        for key in keys:
            held = self._held.get(key)
            if held is not None:
                held.remaining_steps = max(0, int(held.remaining_steps) - 1)

    def clear_except(self, keys: Iterable[Hashable]) -> None:
        retained = set(keys)
        self._held = {
            key: value for key, value in self._held.items() if key in retained
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "schedule": self.config.to_dict(),
            "attempt_counts": self.attempt_counts.tolist(),
            "infeasible_attempt_counts": self.infeasible_attempt_counts.tolist(),
            "outcome_dependent_relabeling": False,
            "action_teacher": False,
        }


def current_vehicle_behavior_feasibility(
    road: Any,
    vehicle: Any,
    *,
    config: BehaviorCommandSchedulerConfig,
) -> BehaviorFeasibility:
    """Compute command feasibility from current road state only."""
    network = getattr(road, "network", None)
    position = getattr(vehicle, "position", None)
    if network is None or position is None:
        return BehaviorFeasibility(True, False, False, False)
    lane_index = getattr(vehicle, "lane_index", None)
    if not (
        isinstance(lane_index, tuple)
        and len(lane_index) == 3
        and lane_index[2] is not None
    ):
        lane_index = network.get_closest_lane_index(
            np.asarray(position, dtype=float),
            float(getattr(vehicle, "heading", 0.0)),
        )
    side_lanes = set(network.side_lanes(lane_index))
    left_index = (lane_index[0], lane_index[1], int(lane_index[2]) + 1)
    right_index = (lane_index[0], lane_index[1], int(lane_index[2]) - 1)
    change_left = left_index in side_lanes
    change_right = right_index in side_lanes
    front = None
    if hasattr(road, "neighbour_vehicles"):
        front, _rear = road.neighbour_vehicles(vehicle, lane_index)
    overtake = False
    if front is not None and (change_left or change_right):
        lane = network.get_lane(lane_index)
        ego_s = float(lane.local_coordinates(np.asarray(position, dtype=float))[0])
        front_s = float(
            lane.local_coordinates(np.asarray(front.position, dtype=float))[0]
        )
        gap = front_s - ego_s
        overtake = bool(
            config.overtake_min_lead_gap_m
            <= gap
            <= config.overtake_max_lead_gap_m
        )
    return BehaviorFeasibility(True, change_left, change_right, overtake)


__all__ = [
    "BEHAVIOR_INTENT_DIM",
    "BEHAVIOR_INTENT_SCHEMA_VERSION",
    "MANEUVER_FAMILY_DIM",
    "MANEUVER_FAMILY_SCHEMA_VERSION",
    "BehaviorCommandSchedulerConfig",
    "BehaviorFeasibility",
    "BehaviorIntent",
    "BehaviorLabelerConfig",
    "BehaviorLabels",
    "INVALID_BEHAVIOR_ID",
    "ManeuverFamilyIntent",
    "ProspectiveBehaviorCommandScheduler",
    "behavior_intent_contract",
    "behavior_intent_one_hot",
    "behavior_to_maneuver_family_ids",
    "current_vehicle_behavior_feasibility",
    "label_behavior_rows",
    "maneuver_family_contract",
    "maneuver_family_one_hot",
]
