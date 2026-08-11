"""Causal observation-aligned controller for the 329-field route-free sensor.

The controller is deliberately small and transparent.  It consumes exactly
the current ``route_free_ego_lidar_v2`` policy observation and emits normalized
``[acceleration, steering]`` commands.  It never reads a trajectory suffix,
route, simulator lane object, vehicle identity, previous action, tracker state,
domain identity, or topology identity.

This is a synthetic sensor-conditioned expert for behavior-cloning and
recovery-data experiments.  It is not a human-driver model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from highway_env.ngsim_utils.core.constants import MAX_STEER

ROUTE_FREE_OBSERVATION_ALIGNED_REFLEX_ID = (
    "route_free_observation_aligned_reflex_v7"
)
ROUTE_FREE_OBSERVATION_DIM = 329
ROUTE_FREE_LIDAR_CELLS = 64
ROUTE_FREE_LIDAR_FEATURES = 4
ROUTE_FREE_LANE_CELLS = 21
ROUTE_FREE_LANE_FEATURES = 3


@dataclass(frozen=True)
class RouteFreeReflexConfig:
    """Frozen physical constants for the route-free synthetic expert."""

    maximum_range_m: float = 64.0
    lidar_cells: int = ROUTE_FREE_LIDAR_CELLS
    camera_lookahead_m: float = 12.8
    camera_fit_min_forward_m: float = 1.92
    camera_fit_max_forward_m: float = 41.6
    desired_speed_mps: float = 20.0
    speed_gain_per_s: float = 0.8
    maximum_free_acceleration_mps2: float = 5.0
    standstill_gap_m: float = 5.0
    time_headway_s: float = 1.2
    gap_response_time_s: float = 0.5
    closing_speed_gain: float = 0.5
    non_reversing_brake_horizon_s: float = 0.1
    forward_collision_corridor_half_width_m: float = 2.0
    road_edge_longitudinal_guard: bool = False
    acceleration_scale_mps2: float = 5.0

    def __post_init__(self) -> None:
        numeric = np.asarray(
            [
                self.maximum_range_m,
                self.lidar_cells,
                self.camera_lookahead_m,
                self.camera_fit_min_forward_m,
                self.camera_fit_max_forward_m,
                self.desired_speed_mps,
                self.speed_gain_per_s,
                self.maximum_free_acceleration_mps2,
                self.standstill_gap_m,
                self.time_headway_s,
                self.gap_response_time_s,
                self.closing_speed_gain,
                self.non_reversing_brake_horizon_s,
                self.forward_collision_corridor_half_width_m,
                self.acceleration_scale_mps2,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(numeric)) or np.any(numeric <= 0.0):
            raise ValueError("Route-free reflex constants must be positive and finite.")
        if int(self.lidar_cells) != ROUTE_FREE_LIDAR_CELLS:
            raise ValueError("The v7 route-free reflex is frozen to 64 LiDAR beams.")
        if self.camera_fit_min_forward_m >= self.camera_fit_max_forward_m:
            raise ValueError("Camera fit bounds must be strictly increasing.")
        if not self.camera_fit_min_forward_m <= self.camera_lookahead_m <= self.camera_fit_max_forward_m:
            raise ValueError("Camera lookahead must lie inside the fit interval.")
        if not isinstance(self.road_edge_longitudinal_guard, bool):
            raise ValueError("Road-edge longitudinal guard must be boolean.")


def route_free_reflex_contract(
    config: RouteFreeReflexConfig | None = None,
) -> dict[str, Any]:
    """Return the immutable causal input/action contract and its digest."""

    cfg = config or RouteFreeReflexConfig()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": ROUTE_FREE_OBSERVATION_ALIGNED_REFLEX_ID,
        "claim_boundary": "sensor_conditioned_synthetic_reflex_controller",
        "policy_observation_contract": "route_free_ego_lidar_v2",
        "policy_observation_dim": ROUTE_FREE_OBSERVATION_DIM,
        "action_order": ["acceleration_norm", "steering_norm"],
        "physical_action_scales": [
            float(cfg.acceleration_scale_mps2),
            float(MAX_STEER),
        ],
        "causal_inputs": [
            "current_ego_fixed_separated_lidar",
            "current_lane_camera",
            "current_vehicle_dimensions_and_kinematics",
            "current_sensor_derived_lane_state",
        ],
        "forbidden_reads": [
            "domain_identity",
            "expert_action_history",
            "realized_future_path",
            "route_or_destination",
            "simulator_lane_or_road_object",
            "teacher_tracker_state",
            "topology_identity",
            "vehicle_identity",
        ],
        "stateful": False,
        "steering_source": (
            "robust_current_camera_derived_lane_state_with_direct_camera_fallback"
        ),
        "dynamic_gap_geometry": (
            "ray_distance_minus_exact_ego_rectangle_support"
        ),
        "non_reversing_guard": (
            "acceleration_lower_bounded_by_signed_speed_over_brake_horizon"
        ),
        "output_saturation": "explicit_normalized_physical_action_bounds",
        "config": asdict(cfg),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _fit_camera_boundary(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one ``y = intercept + slope*x`` boundary per observation row."""

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


def route_free_reflex_actions(
    observations: np.ndarray,
    *,
    config: RouteFreeReflexConfig | None = None,
) -> np.ndarray:
    """Compute normalized ``[acceleration, steering]`` from 329 causal fields."""

    cfg = config or RouteFreeReflexConfig()
    values = np.asarray(observations, dtype=np.float32)
    squeeze = values.ndim == 1
    if squeeze:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != ROUTE_FREE_OBSERVATION_DIM:
        raise ValueError(
            "Route-free reflex requires [N,329] observations, "
            f"got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Route-free reflex input contains non-finite values.")

    rows = len(values)
    lidar_stop = ROUTE_FREE_LIDAR_CELLS * ROUTE_FREE_LIDAR_FEATURES
    lane_stop = lidar_stop + ROUTE_FREE_LANE_CELLS * ROUTE_FREE_LANE_FEATURES
    lidar = values[:, :lidar_stop].reshape(
        rows,
        ROUTE_FREE_LIDAR_CELLS,
        ROUTE_FREE_LIDAR_FEATURES,
    )
    camera = values[:, lidar_stop:lane_stop].reshape(
        rows,
        ROUTE_FREE_LANE_CELLS,
        ROUTE_FREE_LANE_FEATURES,
    )
    ego = values[:, lane_stop:]
    vehicle_length = np.maximum(ego[:, 0].astype(np.float64), 1.0)
    vehicle_width = np.maximum(ego[:, 1].astype(np.float64), 0.5)
    signed_speed = ego[:, 2].astype(np.float64)
    speed = np.maximum(signed_speed, 0.0)

    # Fit only current camera returns.  Both boundaries are required for a
    # lateral centre estimate; otherwise steering fails safely to zero.
    x = camera[:, :, 1].astype(np.float64)
    y = camera[:, :, 2].astype(np.float64)
    fit_min = float(cfg.camera_fit_min_forward_m / cfg.maximum_range_m)
    fit_max = float(cfg.camera_fit_max_forward_m / cfg.maximum_range_m)
    visible = (camera[:, :, 0] > 0.5) & (x >= fit_min) & (x <= fit_max)
    left_intercept, left_slope, left_valid = _fit_camera_boundary(
        x,
        y,
        visible & (y > 0.0),
    )
    right_intercept, right_slope, right_valid = _fit_camera_boundary(
        x,
        y,
        visible & (y < 0.0),
    )
    lookahead_norm = float(cfg.camera_lookahead_m / cfg.maximum_range_m)
    centre_left = left_intercept + left_slope * lookahead_norm
    centre_right = right_intercept + right_slope * lookahead_norm
    camera_valid = left_valid & right_valid
    direct_camera_centre_lateral_m = np.where(
        camera_valid,
        0.5 * (centre_left + centre_right) * float(cfg.maximum_range_m),
        0.0,
    )
    # The final ten policy-visible fields already contain a robust, causal fit
    # of the same current lane-camera returns.  Use that explicit lane offset
    # and heading error for steering; the direct two-boundary fit above remains
    # a deterministic fallback when the robust fit has no valid lane width.
    lane_offset_m = ego[:, 6].astype(np.float64)
    lane_heading_error = np.arctan2(
        ego[:, 7].astype(np.float64),
        ego[:, 8].astype(np.float64),
    )
    lane_width_m = ego[:, 9].astype(np.float64)
    sensor_lane_valid = (
        (lane_width_m >= 2.4)
        & (lane_width_m <= 5.5)
        & np.isfinite(lane_heading_error)
        & (np.abs(lane_heading_error) <= np.pi / 3.0)
    )
    sensor_lane_centre_lateral_m = -lane_offset_m - np.tan(
        lane_heading_error
    ) * float(cfg.camera_lookahead_m)
    centre_lateral_m = np.where(
        sensor_lane_valid,
        sensor_lane_centre_lateral_m,
        direct_camera_centre_lateral_m,
    )
    bearing = np.arctan2(
        centre_lateral_m,
        np.full(rows, float(cfg.camera_lookahead_m), dtype=np.float64),
    )
    lookahead_distance = np.hypot(float(cfg.camera_lookahead_m), centre_lateral_m)
    curvature = 2.0 * np.sin(bearing) / np.maximum(lookahead_distance, 1.0e-3)
    steering_rad = np.arctan(vehicle_length * curvature)
    steering_normalized = np.clip(
        steering_rad / float(MAX_STEER),
        -1.0,
        1.0,
    )

    # Dynamic presence is separate from road-edge range in v2, so empty and
    # road-boundary returns cannot be mistaken for a lead vehicle.  Select a
    # vehicle-scale lateral corridor in the ego frame instead of an arbitrary
    # fixed beam cone: a wide cone spuriously brakes for distant adjacent-lane
    # traffic, while a narrow cone misses close hazards on curved roads.
    forward = lidar.astype(np.float64)
    dynamic_present = route_free_forward_collision_mask(forward, config=cfg)
    candidate_distance = np.where(dynamic_present, forward[:, :, 1], np.inf)
    angles = (
        np.arange(ROUTE_FREE_LIDAR_CELLS, dtype=np.float64) + 0.5
    ) * (2.0 * np.pi / float(ROUTE_FREE_LIDAR_CELLS))
    road_lateral_m = (
        forward[:, :, 3]
        * float(cfg.maximum_range_m)
        * np.sin(angles).reshape(1, -1)
    )
    road_forward = (
        (np.cos(angles).reshape(1, -1) > 0.0)
        & (
            np.abs(road_lateral_m)
            <= float(cfg.forward_collision_corridor_half_width_m)
        )
    )
    if cfg.road_edge_longitudinal_guard:
        # Preserve the v2 semantic separation while using the current visible
        # road-edge range as a causal upper bound on safe forward travel.  A
        # road edge never supplies a fabricated relative velocity.
        effective_distance = np.where(
            road_forward,
            np.minimum(candidate_distance, forward[:, :, 3]),
            np.inf,
        )
    else:
        effective_distance = candidate_distance
    nearest_index = np.argmin(effective_distance, axis=1)
    row_index = np.arange(rows, dtype=np.int64)
    has_lead = np.any(dynamic_present, axis=1)
    has_forward_constraint = (
        np.ones(rows, dtype=bool)
        if cfg.road_edge_longitudinal_guard
        else has_lead
    )
    selected_angles = angles[nearest_index]
    longitudinal_support_m = np.divide(
        0.5 * vehicle_length,
        np.abs(np.cos(selected_angles)),
        out=np.full(rows, np.inf, dtype=np.float64),
        where=np.abs(np.cos(selected_angles)) > 1.0e-9,
    )
    lateral_support_m = np.divide(
        0.5 * vehicle_width,
        np.abs(np.sin(selected_angles)),
        out=np.full(rows, np.inf, dtype=np.float64),
        where=np.abs(np.sin(selected_angles)) > 1.0e-9,
    )
    ego_ray_support_m = np.minimum(
        longitudinal_support_m,
        lateral_support_m,
    )
    origin_distance_m = (
        effective_distance[row_index, nearest_index]
        * float(cfg.maximum_range_m)
    )
    forward_gap_m = np.where(
        has_forward_constraint,
        np.maximum(origin_distance_m - ego_ray_support_m, 0.0),
        float(cfg.maximum_range_m),
    )
    selected_dynamic = (
        dynamic_present[row_index, nearest_index]
        & (
            candidate_distance[row_index, nearest_index]
            <= forward[row_index, nearest_index, 3]
        )
        if cfg.road_edge_longitudinal_guard
        else has_lead
    )
    forward_relative_speed = np.where(
        selected_dynamic,
        forward[row_index, nearest_index, 2] * float(cfg.maximum_range_m),
        0.0,
    )
    desired_gap = float(cfg.standstill_gap_m) + float(cfg.time_headway_s) * speed
    free_acceleration = np.minimum(
        float(cfg.speed_gain_per_s) * (float(cfg.desired_speed_mps) - speed),
        float(cfg.maximum_free_acceleration_mps2),
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
    # Highway control must not turn braking into reverse propulsion.  Bound
    # the commanded deceleration so the current signed speed cannot cross
    # below zero over one frozen 10 Hz policy interval.  This guard reads only
    # the actor-visible current speed and leaves collision impulses outside the
    # controller's claim boundary.
    non_reversing_lower_bound = -np.maximum(signed_speed, 0.0) / float(
        cfg.non_reversing_brake_horizon_s
    )
    acceleration_mps2 = np.maximum(
        acceleration_mps2,
        non_reversing_lower_bound,
    )
    acceleration_normalized = np.clip(
        acceleration_mps2 / float(cfg.acceleration_scale_mps2),
        -1.0,
        1.0,
    )

    actions = np.stack(
        [acceleration_normalized, steering_normalized],
        axis=1,
    ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Route-free reflex produced non-finite actions.")
    return actions[0] if squeeze else actions


def route_free_forward_collision_mask(
    lidar: np.ndarray,
    *,
    config: RouteFreeReflexConfig | None = None,
) -> np.ndarray:
    """Return dynamic beams intersecting the causal ego-frame safety corridor."""

    cfg = config or RouteFreeReflexConfig()
    values = np.asarray(lidar)
    if values.ndim == 2:
        values = values.reshape(1, *values.shape)
    if values.ndim != 3 or values.shape[1:] != (
        ROUTE_FREE_LIDAR_CELLS,
        ROUTE_FREE_LIDAR_FEATURES,
    ):
        raise ValueError(
            "Route-free forward mask requires [N,64,4] LiDAR, "
            f"got {values.shape}."
        )
    angles = (
        np.arange(ROUTE_FREE_LIDAR_CELLS, dtype=np.float64) + 0.5
    ) * (2.0 * np.pi / float(ROUTE_FREE_LIDAR_CELLS))
    distance_m = values[:, :, 1].astype(np.float64) * float(
        cfg.maximum_range_m
    )
    longitudinal_m = distance_m * np.cos(angles).reshape(1, -1)
    lateral_m = distance_m * np.sin(angles).reshape(1, -1)
    return (
        (values[:, :, 0] >= 0.5)
        & (longitudinal_m > 0.0)
        & (
            np.abs(lateral_m)
            <= float(cfg.forward_collision_corridor_half_width_m)
        )
    )


__all__ = [
    "ROUTE_FREE_OBSERVATION_ALIGNED_REFLEX_ID",
    "RouteFreeReflexConfig",
    "route_free_forward_collision_mask",
    "route_free_reflex_actions",
    "route_free_reflex_contract",
]
