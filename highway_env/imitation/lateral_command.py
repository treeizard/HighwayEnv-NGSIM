"""Prospective lateral commands and a current-map synthetic teacher.

The actor-visible contract is deliberately small: a three-way command chosen
before the policy observation is constructed.  Current lane geometry and the
held target lane are scheduler/teacher-private state.  Interaction labels are
non-exclusive analysis metadata and never actor features.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np

from highway_env import utils
from highway_env.imitation.observation_aligned_reflex import (
    ObservationAlignedReflexConfig,
    observation_aligned_reflex_actions,
)
from highway_env.ngsim_utils.core.constants import MAX_STEER

LATERAL_COMMAND_SCHEMA_VERSION = 2
LATERAL_COMMAND_DIM = 3
LATERAL_COMMAND_CONTRACT_ID = "prospective_lateral_command_one_hot_v2"
LATERAL_COMMAND_TEACHER_ID = "current_map_lateral_command_teacher_v1"


class LateralCommand(IntEnum):
    """Stable actor-visible command IDs."""

    KEEP = 0
    CHANGE_LEFT = 1
    CHANGE_RIGHT = 2


class OvertakePhase(IntEnum):
    """Analysis-only phase of a non-exclusive passing interaction."""

    NONE = 0
    APPROACH = 1
    ALONGSIDE = 2
    CLEARED = 3


@dataclass(frozen=True)
class InteractionOverlay:
    """Non-exclusive interaction metadata; never append this to observations."""

    following: bool = False
    overtaking: bool = False
    overtake_phase: OvertakePhase = OvertakePhase.NONE

    def __post_init__(self) -> None:
        phase = OvertakePhase(int(self.overtake_phase))
        if not self.overtaking and phase != OvertakePhase.NONE:
            raise ValueError("A non-NONE overtake phase requires overtaking=True.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "following": bool(self.following),
            "overtaking": bool(self.overtaking),
            "overtake_phase": OvertakePhase(int(self.overtake_phase)).name.lower(),
            "policy_visible": False,
        }


def lateral_command_contract() -> dict[str, Any]:
    """Return the complete actor-input and non-leakage contract."""

    return {
        "schema_version": LATERAL_COMMAND_SCHEMA_VERSION,
        "contract_id": LATERAL_COMMAND_CONTRACT_ID,
        "feature_dim": LATERAL_COMMAND_DIM,
        "encoding": "one_hot",
        "classes": [{"id": int(command), "name": command.name} for command in LateralCommand],
        "selected_before_policy_observation": True,
        "held_target_lane_is_teacher_private": True,
        "interaction_overlays_policy_visible": False,
        "forbidden_policy_inputs": [
            "domain_id",
            "future_coordinates",
            "future_state",
            "lane_id",
            "route_or_destination",
            "target_lane_id",
            "target_vehicle_id",
            "teacher_controller_state",
            "topology_id",
        ],
    }


def lateral_command_one_hot(
    command_ids: np.ndarray | Sequence[int] | int,
) -> np.ndarray:
    """Validate command IDs and return exact float32 one-hot rows."""

    ids = np.asarray(command_ids)
    scalar = ids.ndim == 0
    ids = ids.reshape(-1)
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
            raise ValueError("Lateral command IDs must be finite integers.")
        ids = ids.astype(np.int64)
    else:
        ids = ids.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.any(ids >= LATERAL_COMMAND_DIM):
        raise ValueError(f"Lateral command IDs must be in [0, {LATERAL_COMMAND_DIM - 1}].")
    rows = np.eye(LATERAL_COMMAND_DIM, dtype=np.float32)[ids]
    return rows[0] if scalar else rows


@dataclass(frozen=True)
class CommandInstanceV2:
    """Immutable identity of one pre-observation command intervention."""

    instance_id: str
    domain: str
    split: str
    episode_id: str
    vehicle_id: str
    reset_seed: int
    onset_step: int
    pre_command_state_sha256: str
    command: LateralCommand
    horizon_steps: int = 50
    stable_completion_steps: int = 5
    target_lane_private: tuple[Any, Any, int] | None = None
    schema_version: int = LATERAL_COMMAND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != LATERAL_COMMAND_SCHEMA_VERSION:
            raise ValueError("CommandInstanceV2 requires schema_version=2.")
        if self.domain not in {"us", "japanese"}:
            raise ValueError(f"Unsupported command-instance domain: {self.domain}")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported command-instance split: {self.split}")
        if not self.instance_id or not self.episode_id or not self.vehicle_id:
            raise ValueError("Command instance identity fields must be non-empty.")
        if len(self.pre_command_state_sha256) != 64:
            raise ValueError("pre_command_state_sha256 must be a SHA-256 digest.")
        if int(self.onset_step) < 0 or int(self.horizon_steps) < 1:
            raise ValueError("Command onset/horizon is invalid.")
        if not 1 <= int(self.stable_completion_steps) <= int(self.horizon_steps):
            raise ValueError("Stable completion must fit within the horizon.")
        command = LateralCommand(int(self.command))
        if command != LateralCommand.KEEP and self.target_lane_private is None:
            raise ValueError("Lane-change instances require a private target lane.")

    def to_dict(self, *, include_teacher_private: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = LateralCommand(int(self.command)).name
        payload["selected_before_policy_observation"] = True
        payload["test_split_opened"] = self.split == "test"
        if not include_teacher_private:
            payload.pop("target_lane_private", None)
        return payload


@dataclass(frozen=True)
class LateralCommandSchedulerConfig:
    """Frozen prospective schedule.  It selects commands, never actions."""

    schema_version: int = LATERAL_COMMAND_SCHEMA_VERSION
    schedule_id: str = "feasible_prospective_lateral_commands_v2"
    horizon_steps: int = 50
    stable_completion_steps: int = 5
    minimum_lateral_offset_m: float = 0.25
    minimum_front_gap_m: float = 12.0
    minimum_rear_gap_m: float = 15.0
    minimum_front_ttc_s: float = 2.0
    minimum_rear_ttc_s: float = 3.0

    def __post_init__(self) -> None:
        if int(self.schema_version) != LATERAL_COMMAND_SCHEMA_VERSION:
            raise ValueError("Lateral command scheduler requires schema_version=2.")
        if not self.schedule_id or int(self.horizon_steps) < 1:
            raise ValueError("Lateral command schedule is invalid.")
        if not 1 <= int(self.stable_completion_steps) <= int(self.horizon_steps):
            raise ValueError("Stable completion steps must fit within the horizon.")
        safety_values = np.asarray(
            [
                self.minimum_lateral_offset_m,
                self.minimum_front_gap_m,
                self.minimum_rear_gap_m,
                self.minimum_front_ttc_s,
                self.minimum_rear_ttc_s,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(safety_values)) or np.any(safety_values <= 0.0):
            raise ValueError("Lateral opportunity thresholds must be positive and finite.")

    @classmethod
    def from_json(cls, path: str) -> "LateralCommandSchedulerConfig":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("Lateral command schedule must be a JSON object.")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LateralFeasibility:
    """Feasible commands and their private current-map targets."""

    keep_lane: tuple[Any, Any, int] | None
    change_left: tuple[Any, Any, int] | None
    change_right: tuple[Any, Any, int] | None
    keep_reason: str = "safe"
    change_left_reason: str = "unreported"
    change_right_reason: str = "unreported"
    change_left_diagnostic: Mapping[str, Any] | None = None
    change_right_diagnostic: Mapping[str, Any] | None = None

    def target(self, command: LateralCommand) -> tuple[Any, Any, int] | None:
        return {
            LateralCommand.KEEP: self.keep_lane,
            LateralCommand.CHANGE_LEFT: self.change_left,
            LateralCommand.CHANGE_RIGHT: self.change_right,
        }[LateralCommand(int(command))]

    def feasible(self, command: LateralCommand) -> bool:
        return self.target(command) is not None

    def reason(self, command: LateralCommand) -> str:
        """Return a teacher-private present-time feasibility diagnosis."""

        return {
            LateralCommand.KEEP: self.keep_reason,
            LateralCommand.CHANGE_LEFT: self.change_left_reason,
            LateralCommand.CHANGE_RIGHT: self.change_right_reason,
        }[LateralCommand(int(command))]

    def diagnostic(self, command: LateralCommand) -> dict[str, Any]:
        """Return a copy of teacher-private present-time opportunity metrics."""

        command = LateralCommand(int(command))
        if command == LateralCommand.KEEP:
            return {"reason": self.keep_reason}
        payload = (
            self.change_left_diagnostic
            if command == LateralCommand.CHANGE_LEFT
            else self.change_right_diagnostic
        )
        return dict(payload or {"reason": self.reason(command)})

    @property
    def commands(self) -> tuple[LateralCommand, ...]:
        return tuple(command for command in LateralCommand if self.feasible(command))


def _current_lane_index(road: Any, vehicle: Any) -> tuple[Any, Any, int] | None:
    network = getattr(road, "network", None)
    position = getattr(vehicle, "position", None)
    if network is None or position is None:
        return None
    lane_index = getattr(vehicle, "lane_index", None)
    if not (isinstance(lane_index, tuple) and len(lane_index) == 3 and lane_index[2] is not None):
        lane_index = network.get_closest_lane_index(
            np.asarray(position, dtype=float),
            float(getattr(vehicle, "heading", 0.0)),
        )
    if not isinstance(lane_index, tuple) or len(lane_index) != 3:
        return None
    return (lane_index[0], lane_index[1], int(lane_index[2]))


def _target_lane_opportunity_diagnostic(
    road: Any,
    vehicle: Any,
    lane_index: tuple[Any, Any, int],
    config: LateralCommandSchedulerConfig,
) -> dict[str, Any]:
    """Measure present-time target-lane opportunity without future reads."""

    neighbour_vehicles = getattr(road, "neighbour_vehicles", None)
    if not callable(neighbour_vehicles):
        return {"reason": "neighbour_query_unavailable"}
    lane = road.network.get_lane(lane_index)
    ego_s = float(lane.local_coordinates(np.asarray(vehicle.position, dtype=float))[0])
    ego_speed = float(getattr(vehicle, "speed", 0.0))
    ego_length = float(getattr(vehicle, "LENGTH", 5.0))
    front, rear = neighbour_vehicles(vehicle, lane_index)
    payload: dict[str, Any] = {
        "reason": "safe",
        "violations": [],
        "front_gap_m": None,
        "front_ttc_s": None,
        "rear_gap_m": None,
        "rear_ttc_s": None,
    }
    if front is not None:
        front_s = float(lane.local_coordinates(np.asarray(front.position, dtype=float))[0])
        front_gap = front_s - ego_s - 0.5 * (ego_length + float(getattr(front, "LENGTH", 5.0)))
        front_closing = ego_speed - float(getattr(front, "speed", 0.0))
        front_ttc = front_gap / front_closing if front_gap > 0.0 and front_closing > 0.0 else np.inf
        payload["front_gap_m"] = float(front_gap)
        payload["front_ttc_s"] = float(front_ttc) if np.isfinite(front_ttc) else None
        if front_gap < float(config.minimum_front_gap_m):
            payload["violations"].append("front_gap")
        if front_ttc < float(config.minimum_front_ttc_s):
            payload["violations"].append("front_ttc")
    if rear is not None:
        rear_s = float(lane.local_coordinates(np.asarray(rear.position, dtype=float))[0])
        rear_gap = ego_s - rear_s - 0.5 * (ego_length + float(getattr(rear, "LENGTH", 5.0)))
        rear_closing = float(getattr(rear, "speed", 0.0)) - ego_speed
        rear_ttc = rear_gap / rear_closing if rear_gap > 0.0 and rear_closing > 0.0 else np.inf
        payload["rear_gap_m"] = float(rear_gap)
        payload["rear_ttc_s"] = float(rear_ttc) if np.isfinite(rear_ttc) else None
        if rear_gap < float(config.minimum_rear_gap_m):
            payload["violations"].append("rear_gap")
        if rear_ttc < float(config.minimum_rear_ttc_s):
            payload["violations"].append("rear_ttc")
    if payload["violations"]:
        payload["reason"] = payload["violations"][0]
    return payload


def current_vehicle_lateral_feasibility(
    road: Any,
    vehicle: Any,
    *,
    config: LateralCommandSchedulerConfig | None = None,
) -> LateralFeasibility:
    """Resolve left/right geometrically in the ego frame at the current time."""

    cfg = config or LateralCommandSchedulerConfig()
    network = getattr(road, "network", None)
    lane_index = _current_lane_index(road, vehicle)
    position = np.asarray(getattr(vehicle, "position", ()), dtype=float)
    if network is None or lane_index is None or position.shape != (2,):
        return LateralFeasibility(
            None,
            None,
            None,
            keep_reason="no_current_lane",
            change_left_reason="no_current_lane",
            change_right_reason="no_current_lane",
            change_left_diagnostic={"reason": "no_current_lane"},
            change_right_diagnostic={"reason": "no_current_lane"},
        )
    heading = float(getattr(vehicle, "heading", 0.0))
    ego_left = np.asarray([-np.sin(heading), np.cos(heading)], dtype=float)
    left: tuple[Any, Any, int] | None = None
    right: tuple[Any, Any, int] | None = None
    left_offset = 0.0
    right_offset = 0.0
    left_reason = "no_adjacent_lane"
    right_reason = "no_adjacent_lane"
    left_diagnostic: dict[str, Any] = {"reason": left_reason}
    right_diagnostic: dict[str, Any] = {"reason": right_reason}
    for candidate in network.side_lanes(lane_index):
        candidate = (candidate[0], candidate[1], int(candidate[2]))
        if candidate == lane_index:
            continue
        lane = network.get_lane(candidate)
        longitudinal, _ = lane.local_coordinates(position)
        centre = np.asarray(lane.position(longitudinal, 0.0), dtype=float)
        offset = float(np.dot(centre - position, ego_left))
        if offset >= float(cfg.minimum_lateral_offset_m):
            direction = LateralCommand.CHANGE_LEFT
        elif offset <= -float(cfg.minimum_lateral_offset_m):
            direction = LateralCommand.CHANGE_RIGHT
        else:
            continue
        diagnostic = (
            {"reason": "unreachable"}
            if not lane.is_reachable_from(position)
            else _target_lane_opportunity_diagnostic(road, vehicle, candidate, cfg)
        )
        diagnostic = {**diagnostic, "lateral_offset_m": float(offset)}
        reason = str(diagnostic["reason"])
        if direction == LateralCommand.CHANGE_LEFT and offset > left_offset:
            left_offset = offset
            left_reason = reason
            left_diagnostic = diagnostic
            left = candidate if reason == "safe" else None
        elif direction == LateralCommand.CHANGE_RIGHT and offset < right_offset:
            right_offset = offset
            right_reason = reason
            right_diagnostic = diagnostic
            right = candidate if reason == "safe" else None
    return LateralFeasibility(
        lane_index,
        left,
        right,
        keep_reason="safe",
        change_left_reason=left_reason,
        change_right_reason=right_reason,
        change_left_diagnostic=left_diagnostic,
        change_right_diagnostic=right_diagnostic,
    )


@dataclass
class _HeldCommand:
    command: LateralCommand
    target_lane: tuple[Any, Any, int]
    remaining_steps: int


class ProspectiveLateralCommandScheduler:
    """Choose a feasible command before observation and hold its target."""

    def __init__(self, config: LateralCommandSchedulerConfig, *, seed: int) -> None:
        self.config = config
        self._rng = np.random.default_rng(int(seed))
        self._held: dict[Hashable, _HeldCommand] = {}
        self.attempt_counts = np.zeros(LATERAL_COMMAND_DIM, dtype=np.int64)
        self.infeasible_attempt_counts = np.zeros(LATERAL_COMMAND_DIM, dtype=np.int64)
        self.safety_revocation_counts = np.zeros(LATERAL_COMMAND_DIM, dtype=np.int64)

    def assignment_for(
        self,
        key: Hashable,
        feasibility: LateralFeasibility,
        *,
        requested: LateralCommand | None = None,
    ) -> _HeldCommand:
        held = self._held.get(key)
        if held is not None and held.remaining_steps > 0:
            if held.command == LateralCommand.KEEP:
                return held
            current_safe_targets = {
                feasibility.keep_lane,
                feasibility.change_left,
                feasibility.change_right,
            }
            current_safe_targets.discard(None)
            if held.target_lane in current_safe_targets:
                return held
            self.safety_revocation_counts[int(held.command)] += 1
            held.remaining_steps = 0
        candidates = (
            [LateralCommand(int(requested))]
            if requested is not None
            else [LateralCommand(int(index)) for index in self._rng.permutation(LATERAL_COMMAND_DIM)]
        )
        selected: LateralCommand | None = None
        target: tuple[Any, Any, int] | None = None
        for command in candidates:
            self.attempt_counts[int(command)] += 1
            target = feasibility.target(command)
            if target is not None:
                selected = command
                break
            self.infeasible_attempt_counts[int(command)] += 1
        if selected is None or target is None:
            if feasibility.keep_lane is None:
                raise RuntimeError("No current lane is available for a safe KEEP fallback.")
            selected, target = LateralCommand.KEEP, feasibility.keep_lane
        held = _HeldCommand(
            command=selected,
            target_lane=target,
            remaining_steps=int(self.config.horizon_steps),
        )
        self._held[key] = held
        return held

    def advance(self, keys: Iterable[Hashable]) -> None:
        for key in keys:
            held = self._held.get(key)
            if held is not None:
                held.remaining_steps = max(0, int(held.remaining_steps) - 1)

    def clear_except(self, keys: Iterable[Hashable]) -> None:
        retained = set(keys)
        self._held = {key: value for key, value in self._held.items() if key in retained}

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": LATERAL_COMMAND_SCHEMA_VERSION,
            "schedule": self.config.to_dict(),
            "attempt_counts": self.attempt_counts.tolist(),
            "infeasible_attempt_counts": self.infeasible_attempt_counts.tolist(),
            "safety_revocation_counts": self.safety_revocation_counts.tolist(),
            "selected_before_policy_observation": True,
            "outcome_dependent_relabeling": False,
            "action_teacher": False,
        }


def command_state_sha256(payload: Any) -> str:
    """Hash a caller-provided deterministic pre-command state description."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LateralCommandTeacherConfig:
    """Current-time target-lane steering constants."""

    tau_pursuit_s: float = 0.5
    heading_time_constant_s: float = 0.2
    lateral_time_constant_s: float = 0.6
    low_speed_steering_limit_rad: float = 0.2
    high_speed_steering_limit_rad: float = np.pi / 3
    high_speed_threshold_mps: float = 5.0

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Lateral teacher constants must be positive and finite.")


def lateral_command_teacher_contract(
    config: LateralCommandTeacherConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LateralCommandTeacherConfig()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": LATERAL_COMMAND_TEACHER_ID,
        "claim_boundary": "prospective_current_map_synthetic_lateral_controller",
        "policy_observation_dim": 325,
        "sensor_observation_dim": 322,
        "command_dim": LATERAL_COMMAND_DIM,
        "action_order": ["acceleration_norm", "steering_norm"],
        "physical_action_scales": [5.0, float(MAX_STEER)],
        "current_time_private_inputs": ["road_lane_geometry", "held_target_lane"],
        "forbidden_reads": [
            "domain_identity",
            "expert_action_history",
            "realized_future_path",
            "route_or_destination",
            "teacher_tracker_state",
        ],
        "config": asdict(cfg),
    }
    payload["contract_sha256"] = command_state_sha256(payload)
    return payload


def _current_map_steering(
    road: Any,
    vehicle: Any,
    target_lane: tuple[Any, Any, int],
    config: LateralCommandTeacherConfig,
) -> float:
    lane = road.network.get_lane(target_lane)
    position = np.asarray(vehicle.position, dtype=float)
    speed = float(vehicle.speed)
    heading = float(vehicle.heading)
    length = max(float(getattr(vehicle, "LENGTH", 5.0)), 1.0)
    longitudinal, lateral = lane.local_coordinates(position)
    future_heading = float(lane.heading_at(longitudinal + speed * config.tau_pursuit_s))
    control_speed = max(abs(speed), 1.0)
    lateral_speed_command = -(1.0 / config.lateral_time_constant_s) * float(lateral)
    heading_command = np.arcsin(np.clip(lateral_speed_command / control_speed, -1.0, 1.0))
    heading_reference = future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
    heading_rate = (1.0 / config.heading_time_constant_s) * utils.wrap_to_pi(heading_reference - heading)
    slip_angle = np.arcsin(np.clip(length / (2.0 * control_speed) * heading_rate, -1.0, 1.0))
    steering = float(np.arctan(2.0 * np.tan(slip_angle)))
    limit = float(
        np.interp(
            control_speed,
            [1.0, config.high_speed_threshold_mps],
            [config.low_speed_steering_limit_rad, config.high_speed_steering_limit_rad],
        )
    )
    return float(np.clip(steering, -limit, limit))


def lateral_command_teacher_action(
    policy_observation: np.ndarray,
    *,
    road: Any,
    vehicle: Any,
    command: LateralCommand,
    target_lane: tuple[Any, Any, int] | None = None,
    scheduler_config: LateralCommandSchedulerConfig | None = None,
    teacher_config: LateralCommandTeacherConfig | None = None,
    reflex_config: ObservationAlignedReflexConfig | None = None,
) -> np.ndarray:
    """Return one normalized ``[acceleration, steering]`` teacher action."""

    observation = np.asarray(policy_observation, dtype=np.float32).reshape(-1)
    if observation.shape not in {(322,), (325,)}:
        raise ValueError(f"Lateral teacher requires 322 or 325 fields, got {observation.shape}.")
    sensors = observation[:322]
    command = LateralCommand(int(command))
    if observation.shape == (325,):
        expected = lateral_command_one_hot(command)
        if not np.array_equal(observation[322:], expected):
            raise ValueError("Observation command tail does not match the requested command.")
    if target_lane is None:
        feasibility = current_vehicle_lateral_feasibility(
            road,
            vehicle,
            config=scheduler_config,
        )
        target_lane = feasibility.target(command)
    if target_lane is None:
        raise ValueError(f"Requested command {command.name} is infeasible at the current state.")
    acceleration = float(observation_aligned_reflex_actions(sensors, config=reflex_config)[0])
    steering_rad = _current_map_steering(
        road,
        vehicle,
        target_lane,
        teacher_config or LateralCommandTeacherConfig(),
    )
    action = np.asarray(
        [acceleration, np.clip(steering_rad / float(MAX_STEER), -1.0, 1.0)],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(action)):
        raise RuntimeError("Lateral command teacher produced a non-finite action.")
    return action


__all__ = [
    "LATERAL_COMMAND_CONTRACT_ID",
    "LATERAL_COMMAND_DIM",
    "LATERAL_COMMAND_SCHEMA_VERSION",
    "LATERAL_COMMAND_TEACHER_ID",
    "CommandInstanceV2",
    "InteractionOverlay",
    "LateralCommand",
    "LateralCommandSchedulerConfig",
    "LateralCommandTeacherConfig",
    "LateralFeasibility",
    "OvertakePhase",
    "ProspectiveLateralCommandScheduler",
    "command_state_sha256",
    "current_vehicle_lateral_feasibility",
    "lateral_command_contract",
    "lateral_command_one_hot",
    "lateral_command_teacher_action",
    "lateral_command_teacher_contract",
]
