"""Observation-aligned synthetic expert for bounded policy qualification.

This module intentionally defines a narrow controller, not a human driver.
Every output is a deterministic function of the deployed 322-field policy
observation at the current pre-action instant.  In particular, it does not
read a realised trajectory suffix, route, expert action history, tracker
state, domain identity, or simulator object.

The controller has two transparent parts:

* a lane-boundary reflex estimates a short-horizon centre point from the
  forward lane camera and applies a stateless pure-pursuit mapping; and
* a car-following reflex combines a fixed free-flow speed with the forward
  lidar gap and relative speed.

It is a prospective observation-aligned target for engineering the policy
and interpretation pipeline.  It cannot support human-intent, human-driver,
policy-realism, culture, or route-choice claims.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from highway_env.ngsim_utils.core.constants import MAX_STEER

OBSERVATION_ALIGNED_REFLEX_ID = "observation_aligned_reflex_v2"


@dataclass(frozen=True)
class ObservationAlignedReflexConfig:
    """Frozen physical constants for the synthetic reflex target."""

    maximum_range_m: float = 64.0
    camera_lookahead_m: float = 12.8
    camera_fit_min_forward_m: float = 1.92
    camera_fit_max_forward_m: float = 41.6
    desired_speed_mps: float = 20.0
    speed_gain_per_s: float = 0.8
    standstill_gap_m: float = 5.0
    time_headway_s: float = 1.2
    gap_response_time_s: float = 0.5
    closing_speed_gain: float = 0.5
    forward_lidar_half_width_cells: int = 4
    acceleration_scale_mps2: float = 5.0

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Observation-aligned reflex constants must be positive and finite.")
        if self.camera_fit_min_forward_m >= self.camera_fit_max_forward_m:
            raise ValueError("Camera fit bounds must be strictly increasing.")
        if self.camera_lookahead_m > self.camera_fit_max_forward_m:
            raise ValueError("Camera lookahead must lie inside the fitted range.")
        if not 1 <= int(self.forward_lidar_half_width_cells) < 64:
            raise ValueError("Forward lidar half-width must be in [1, 63].")


def observation_aligned_reflex_contract(
    config: ObservationAlignedReflexConfig | None = None,
) -> dict[str, Any]:
    """Return the immutable input/target contract and its canonical digest."""

    cfg = config or ObservationAlignedReflexConfig()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": OBSERVATION_ALIGNED_REFLEX_ID,
        "claim_boundary": "sensor_conditioned_synthetic_reflex_controller",
        "policy_observation_dim": 322,
        "policy_observation_contract": "shared_lidar_camera_policy_projection_v2_bounded_sensors",
        "action_order": ["acceleration_norm", "steering_norm"],
        "physical_action_scales": [float(cfg.acceleration_scale_mps2), float(MAX_STEER)],
        "causal_inputs": [
            "current_lidar",
            "current_lane_camera",
            "current_vehicle_length",
            "current_speed",
        ],
        "forbidden_reads": [
            "domain_identity",
            "expert_action_history",
            "realized_future_path",
            "route_or_destination",
            "simulator_object",
            "teacher_tracker_state",
        ],
        "stateful": False,
        "config": asdict(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _fit_camera_boundary(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one y=a+b*x boundary per row with closed-form least squares."""

    weights = np.asarray(mask, dtype=np.float64)
    count = weights.sum(axis=1)
    sum_x = (weights * x).sum(axis=1)
    sum_y = (weights * y).sum(axis=1)
    sum_xx = (weights * x * x).sum(axis=1)
    sum_xy = (weights * x * y).sum(axis=1)
    denominator = count * sum_xx - np.square(sum_x)
    slope = np.divide(
        count * sum_xy - sum_x * sum_y,
        denominator,
        out=np.zeros_like(count),
        where=np.abs(denominator) > 1.0e-8,
    )
    intercept = np.divide(
        sum_y - slope * sum_x,
        count,
        out=np.zeros_like(count),
        where=count > 0.0,
    )
    return intercept, slope, count >= 2.0


def observation_aligned_reflex_actions(
    policy_observations: np.ndarray,
    *,
    config: ObservationAlignedReflexConfig | None = None,
) -> np.ndarray:
    """Compute normalized ``[acceleration, steering]`` from 322 fields.

    The function accepts either one observation or a batch and returns a
    float32 batch.  Invalid or incomplete lane-camera fits fail safely to zero
    steering; all outputs are finite and clipped to the runtime action range.
    """

    cfg = config or ObservationAlignedReflexConfig()
    observations = np.asarray(policy_observations, dtype=np.float32)
    squeeze = observations.ndim == 1
    if squeeze:
        observations = observations.reshape(1, -1)
    if observations.ndim != 2 or observations.shape[1] != 322:
        raise ValueError(
            "Observation-aligned reflex requires [N,322] policy observations, "
            f"got {observations.shape}."
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("Observation-aligned reflex input contains non-finite values.")

    rows = len(observations)
    lidar = observations[:, :256].reshape(rows, 128, 2)
    camera = observations[:, 256:319].reshape(rows, 21, 3)
    vehicle_length = np.maximum(observations[:, 319].astype(np.float64), 1.0)
    speed = np.maximum(observations[:, 320].astype(np.float64), 0.0)

    x = camera[:, :, 1].astype(np.float64)
    y = camera[:, :, 2].astype(np.float64)
    fit_min = float(cfg.camera_fit_min_forward_m / cfg.maximum_range_m)
    fit_max = float(cfg.camera_fit_max_forward_m / cfg.maximum_range_m)
    visible = (camera[:, :, 0] > 0.5) & (x >= fit_min) & (x <= fit_max)
    left_intercept, left_slope, left_valid = _fit_camera_boundary(
        x, y, visible & (y > 0.0)
    )
    right_intercept, right_slope, right_valid = _fit_camera_boundary(
        x, y, visible & (y < 0.0)
    )
    lookahead_normalized = float(cfg.camera_lookahead_m / cfg.maximum_range_m)
    centre_left = left_intercept + left_slope * lookahead_normalized
    centre_right = right_intercept + right_slope * lookahead_normalized
    centre_lateral_normalized = 0.5 * (centre_left + centre_right)
    centre_lateral_normalized = np.where(
        left_valid & right_valid,
        centre_lateral_normalized,
        0.0,
    )
    target_left_m = centre_lateral_normalized * float(cfg.maximum_range_m)
    target_forward_m = np.full(rows, float(cfg.camera_lookahead_m), dtype=np.float64)
    lookahead_distance = np.hypot(target_forward_m, target_left_m)
    bearing = np.arctan2(target_left_m, target_forward_m)
    curvature = 2.0 * np.sin(bearing) / np.maximum(lookahead_distance, 1.0e-3)
    steering_rad = np.arctan(vehicle_length * curvature)
    steering_normalized = np.clip(steering_rad / float(MAX_STEER), -1.0, 1.0)

    half_width = int(cfg.forward_lidar_half_width_cells)
    forward_indices = np.concatenate(
        [
            np.arange(0, half_width + 1, dtype=np.int64),
            np.arange(128 - half_width, 128, dtype=np.int64),
        ]
    )
    forward_lidar = lidar[:, forward_indices, :].astype(np.float64)
    nearest_forward_index = np.argmin(forward_lidar[:, :, 0], axis=1)
    row_indices = np.arange(rows, dtype=np.int64)
    forward_gap_m = np.clip(
        forward_lidar[row_indices, nearest_forward_index, 0]
        * float(cfg.maximum_range_m),
        0.0,
        float(cfg.maximum_range_m),
    )
    forward_relative_speed = (
        forward_lidar[row_indices, nearest_forward_index, 1]
        * float(cfg.maximum_range_m)
    )
    desired_gap = float(cfg.standstill_gap_m) + float(cfg.time_headway_s) * speed
    free_acceleration = float(cfg.speed_gain_per_s) * (
        float(cfg.desired_speed_mps) - speed
    )
    response_denominator = np.maximum(
        1.0,
        float(cfg.gap_response_time_s) * np.maximum(speed, 1.0),
    )
    safe_acceleration = (forward_gap_m - desired_gap) / response_denominator
    safe_acceleration -= float(cfg.closing_speed_gain) * np.maximum(
        -forward_relative_speed,
        0.0,
    )
    acceleration_mps2 = np.minimum(free_acceleration, safe_acceleration)
    acceleration_normalized = np.clip(
        acceleration_mps2 / float(cfg.acceleration_scale_mps2), -1.0, 1.0
    )

    actions = np.stack(
        [acceleration_normalized, steering_normalized], axis=1
    ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Observation-aligned reflex produced non-finite actions.")
    return actions[0] if squeeze else actions


__all__ = [
    "OBSERVATION_ALIGNED_REFLEX_ID",
    "ObservationAlignedReflexConfig",
    "observation_aligned_reflex_actions",
    "observation_aligned_reflex_contract",
]
