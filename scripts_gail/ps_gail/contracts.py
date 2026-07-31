"""Explicit sensor and continuous-action contracts for imitation learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from highway_env.ngsim_utils.core.constants import (
    ACCELERATION_RANGE,
    MAX_STEER,
)


NORMALIZED_ACTION_COLUMNS = ("acceleration_norm", "steering_norm")
PHYSICAL_ACTION_COLUMNS = ("steering_rad", "acceleration_mps2")
PHYSICAL_INDEX_FOR_NORMALIZED = (1, 0)
LANE_CAMERA_CELLS = 21
LIDAR_FEATURE_DIM = 2
LANE_CAMERA_FEATURE_DIM = 3
RAW_EGO_COLUMNS = ("speed_mps", "heading_rad", "width_m", "length_m")
POLICY_EGO_COLUMNS = ("length_m", "speed_mps", "heading_rad")
NORMALIZED_OBSERVATION_BOUND_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class ContinuousActionContract:
    """Affine mapping from normalized policy actions to physical controls."""

    normalized_columns: tuple[str, str]
    physical_columns: tuple[str, str]
    physical_index_for_normalized: tuple[int, int]
    scales: tuple[float, float]
    offsets: tuple[float, float]
    maximum_absolute_residual: float
    acceleration_range_mps2: tuple[float, float]
    steering_range_rad: tuple[float, float]
    schema_version: int = 1
    source: str = "inferred_from_aligned_expert_arrays"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            "normalized_columns": list(self.normalized_columns),
            "physical_columns": list(self.physical_columns),
            "physical_index_for_normalized": list(
                self.physical_index_for_normalized
            ),
            "scales": list(self.scales),
            "offsets": list(self.offsets),
            "maximum_absolute_residual": float(
                self.maximum_absolute_residual
            ),
            "acceleration_range_mps2": list(self.acceleration_range_mps2),
            "steering_range_rad": list(self.steering_range_rad),
            "source": str(self.source),
        }


def runtime_continuous_action_contract() -> dict[str, object]:
    """Return the exact action semantics imported by the current process."""
    low, high = (float(value) for value in ACCELERATION_RANGE)
    return ContinuousActionContract(
        normalized_columns=NORMALIZED_ACTION_COLUMNS,
        physical_columns=PHYSICAL_ACTION_COLUMNS,
        physical_index_for_normalized=PHYSICAL_INDEX_FOR_NORMALIZED,
        scales=(high, float(MAX_STEER)),
        offsets=(0.0, 0.0),
        maximum_absolute_residual=0.0,
        acceleration_range_mps2=(low, high),
        steering_range_rad=(-float(MAX_STEER), float(MAX_STEER)),
        source="runtime_shared_constants",
    ).to_dict()


def infer_continuous_action_contract(
    normalized_actions: np.ndarray,
    physical_actions: np.ndarray,
    *,
    residual_tolerance: float = 5.0e-4,
    minimum_spread: float = 1.0e-6,
) -> ContinuousActionContract:
    """Infer the physical meaning of aligned normalized and physical actions."""
    normalized = np.asarray(normalized_actions, dtype=np.float64)
    physical = np.asarray(physical_actions, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 2:
        raise ValueError(
            "Normalized expert actions must be [N, 2] with columns "
            f"{NORMALIZED_ACTION_COLUMNS}; got {normalized.shape}."
        )
    if physical.ndim != 2 or physical.shape != normalized.shape:
        raise ValueError(
            "Physical expert actions must align with normalized actions and "
            f"use columns {PHYSICAL_ACTION_COLUMNS}; got {physical.shape}."
        )
    if len(normalized) < 2:
        raise ValueError("At least two aligned action rows are required.")

    scales: list[float] = []
    offsets: list[float] = []
    maximum_residual = 0.0
    for normalized_index, physical_index in enumerate(
        PHYSICAL_INDEX_FOR_NORMALIZED
    ):
        x = normalized[:, normalized_index]
        y = physical[:, physical_index]
        if float(np.ptp(x)) <= float(minimum_spread):
            # A constant all-zero steering column is valid but cannot identify a
            # scale. The collection-level audit must include a varying file.
            expected_scale = (
                float(ACCELERATION_RANGE[1])
                if normalized_index == 0
                else float(MAX_STEER)
            )
            expected_offset = float(np.mean(y))
            prediction = expected_scale * x + expected_offset
            residual = float(np.max(np.abs(prediction - y)))
            if residual > float(residual_tolerance):
                raise ValueError(
                    f"Constant action column {NORMALIZED_ACTION_COLUMNS[normalized_index]!r} "
                    f"has residual {residual:.8g}."
                )
            scale, offset = expected_scale, expected_offset
        else:
            design = np.column_stack((x, np.ones_like(x)))
            scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
            prediction = scale * x + offset
            residual = float(np.max(np.abs(prediction - y)))
        if (
            not np.isfinite(scale)
            or not np.isfinite(offset)
            or float(scale) <= 0.0
            or residual > float(residual_tolerance)
        ):
            raise ValueError(
                "Expert normalized/physical action arrays do not form one "
                f"finite affine contract for {NORMALIZED_ACTION_COLUMNS[normalized_index]!r}: "
                f"scale={scale}, offset={offset}, max_residual={residual}."
            )
        scales.append(float(scale))
        offsets.append(float(offset))
        maximum_residual = max(maximum_residual, residual)

    return ContinuousActionContract(
        normalized_columns=NORMALIZED_ACTION_COLUMNS,
        physical_columns=PHYSICAL_ACTION_COLUMNS,
        physical_index_for_normalized=PHYSICAL_INDEX_FOR_NORMALIZED,
        scales=(scales[0], scales[1]),
        offsets=(offsets[0], offsets[1]),
        maximum_absolute_residual=maximum_residual,
        acceleration_range_mps2=(-scales[0], scales[0]),
        steering_range_rad=(-scales[1], scales[1]),
    )


def assert_compatible_action_contracts(
    reference: dict[str, Any] | ContinuousActionContract,
    candidate: dict[str, Any] | ContinuousActionContract,
    *,
    tolerance: float = 5.0e-4,
) -> None:
    """Reject contracts that assign different physical meanings to actions."""
    reference_payload = (
        reference.to_dict()
        if isinstance(reference, ContinuousActionContract)
        else reference
    )
    candidate_payload = (
        candidate.to_dict()
        if isinstance(candidate, ContinuousActionContract)
        else candidate
    )
    for key in (
        "normalized_columns",
        "physical_columns",
        "physical_index_for_normalized",
    ):
        if tuple(reference_payload[key]) != tuple(candidate_payload[key]):
            raise ValueError(
                f"Continuous action contract column mismatch for {key}: "
                f"{reference_payload[key]} != {candidate_payload[key]}."
            )
    if not np.allclose(
        reference_payload["scales"],
        candidate_payload["scales"],
        atol=float(tolerance),
        rtol=0.0,
    ):
        raise ValueError(
            "Continuous action scale mismatch: "
            f"{reference_payload['scales']} != {candidate_payload['scales']}."
        )
    if not np.allclose(
        reference_payload["offsets"],
        candidate_payload["offsets"],
        atol=float(tolerance),
        rtol=0.0,
    ):
        raise ValueError(
            "Continuous action offset mismatch: "
            f"{reference_payload['offsets']} != {candidate_payload['offsets']}."
        )


def validate_expert_action_contract(
    normalized_actions: np.ndarray,
    physical_actions: np.ndarray,
    *,
    recorded_contract: dict[str, Any] | None = None,
    require_runtime_match: bool = True,
) -> dict[str, object]:
    """Validate aligned arrays, recorded metadata, and the runtime decoder."""
    inferred = infer_continuous_action_contract(
        normalized_actions,
        physical_actions,
    )
    inferred_payload = inferred.to_dict()
    if recorded_contract is not None:
        assert_compatible_action_contracts(
            inferred_payload,
            recorded_contract,
        )
    if require_runtime_match:
        runtime = runtime_continuous_action_contract()
        try:
            assert_compatible_action_contracts(runtime, inferred_payload)
        except ValueError as exc:
            raise ValueError(
                "Expert actions and the simulator runtime use different physical "
                "units. Set NGSIM_ACCELERATION_LIMIT_MPS2 to the collection's "
                f"recorded scale before loading it. runtime={runtime['scales']} "
                f"expert={inferred_payload['scales']}."
            ) from exc
    return inferred_payload


def policy_observation_contract(
    *,
    lidar_cells: int,
    maximum_range: float,
    lane_camera_cells: int = LANE_CAMERA_CELLS,
    route_intent: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Describe the raw sensor tuple and the exact actor input projection."""
    lidar_cells = int(lidar_cells)
    lane_camera_cells = int(lane_camera_cells)
    lidar_flat_dim = lidar_cells * LIDAR_FEATURE_DIM
    lane_flat_dim = lane_camera_cells * LANE_CAMERA_FEATURE_DIM
    raw_dim = lidar_flat_dim + lane_flat_dim + len(RAW_EGO_COLUMNS)
    policy_dim = lidar_flat_dim + lane_flat_dim + len(POLICY_EGO_COLUMNS)
    policy_order = [
        "lidar_flat",
        "lane_camera_flat",
        *POLICY_EGO_COLUMNS,
    ]
    route_intent_payload = None
    if route_intent is not None:
        route_intent_payload = dict(route_intent)
        if int(route_intent_payload.get("schema_version", -1)) != 1:
            raise ValueError("Route-intent contract schema_version must be 1.")
        projection = dict(route_intent_payload.get("policy_projection") or {})
        fields = list(projection.get("feature_names") or [])
        feature_dim = int(projection.get("feature_dim", -1))
        low = list(projection.get("low") or [])
        high = list(projection.get("high") or [])
        if (
            not route_intent_payload.get("provider_id")
            or feature_dim < 1
            or len(fields) != feature_dim
            or len(set(fields)) != feature_dim
            or len(low) != feature_dim
            or len(high) != feature_dim
        ):
            raise ValueError("Route-intent policy projection is incomplete.")
        if projection.get("includes_previous_action") is not False:
            raise ValueError("Route-intent policy projection cannot include actions.")
        if projection.get("includes_topology_identity") is not False:
            raise ValueError(
                "Route-intent topology identity must remain analysis-only."
            )
        if (
            dict(route_intent_payload.get("topology_sidecar") or {}).get(
                "policy_visible"
            )
            is not False
        ):
            raise ValueError("Route-intent topology sidecar cannot be policy-visible.")
        if any(
            token in field.lower()
            for field in fields
            for token in ("action", "steer", "accel", "domain", "topology", "graph_hash")
        ):
            raise ValueError(
                "Route-intent policy fields contain a forbidden shortcut field."
            )
        policy_dim += feature_dim
        policy_order.append("route_intent_policy_projection")
    return {
        "schema_version": 3 if route_intent_payload is not None else 2,
        "raw_observation_components": [
            {
                "name": "lidar",
                "shape": [lidar_cells, LIDAR_FEATURE_DIM],
                "columns": ["distance_norm", "relative_speed_norm"],
                "normalized": True,
                "low": [0.0, -1.0],
                "high": [1.0, 1.0],
            },
            {
                "name": "lane_camera",
                "shape": [lane_camera_cells, LANE_CAMERA_FEATURE_DIM],
                "columns": ["presence", "relative_x_norm", "relative_y_norm"],
                "normalized": True,
                "low": [0.0, -1.0, -1.0],
                "high": [1.0, 1.0, 1.0],
            },
            {
                "name": "ego_state",
                "shape": [len(RAW_EGO_COLUMNS)],
                "columns": list(RAW_EGO_COLUMNS),
                "normalized": False,
            },
        ],
        "raw_observation_dim": raw_dim,
        "policy_observation_order": policy_order,
        "policy_observation_dim": policy_dim,
        "omitted_raw_fields": ["width_m"],
        "maximum_range_m": float(maximum_range),
        "lidar_cells": lidar_cells,
        "lane_camera_cells": lane_camera_cells,
        "source": "shared_lidar_camera_policy_projection_v2_bounded_sensors",
        "route_intent_contract": route_intent_payload,
    }


def validate_declared_raw_observation_space(
    observations: np.ndarray,
    contract: dict[str, Any],
    *,
    context: str = "raw observations",
    bound_tolerance: float = NORMALIZED_OBSERVATION_BOUND_TOLERANCE,
) -> dict[str, Any]:
    """Validate stored raw sensors against their declared observation space.

    This deliberately checks only fields whose bounds are part of the sensor
    contract. In particular, it does not invent dataset-specific limits for
    ego speed or vehicle dimensions.
    """
    values = np.asarray(observations)
    if values.ndim != 2 or not len(values):
        raise ValueError(
            f"{context} must be a non-empty rank-2 raw observation array; "
            f"got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context} contains non-finite raw observations.")

    raw_dim = int(contract.get("raw_observation_dim", -1))
    if values.shape[1] != raw_dim:
        raise ValueError(
            f"{context} raw observation dimension {values.shape[1]} does not "
            f"match the declared dimension {raw_dim}."
        )

    components = contract.get("raw_observation_components")
    if not isinstance(components, list):
        raise ValueError(
            f"{context} has no declared raw_observation_components."
        )
    expected_layout = {
        "lidar": {
            "columns": ("distance_norm", "relative_speed_norm"),
        },
        "lane_camera": {
            "columns": ("presence", "relative_x_norm", "relative_y_norm"),
        },
        "ego_state": {
            "columns": RAW_EGO_COLUMNS,
        },
    }
    component_names = [str(item.get("name", "")) for item in components]
    if component_names != list(expected_layout):
        raise ValueError(
            f"{context} declares unsupported raw component order "
            f"{component_names}; expected {list(expected_layout)}."
        )

    tolerance = float(bound_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"bound_tolerance must be finite and non-negative, got {tolerance}."
        )

    offset = 0
    field_receipts: dict[str, dict[str, float]] = {}
    component_receipts: dict[str, dict[str, Any]] = {}
    for component in components:
        name = str(component["name"])
        specification = expected_layout[name]
        shape = component.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(int(size) <= 0 for size in shape)
        ):
            raise ValueError(
                f"{context} component {name!r} has invalid shape {shape!r}."
            )
        shape_tuple = tuple(int(size) for size in shape)
        columns = tuple(str(value) for value in component.get("columns", []))
        if columns != specification["columns"]:
            raise ValueError(
                f"{context} component {name!r} columns {columns!r} do not "
                f"match the supported declaration {specification['columns']!r}."
            )
        if shape_tuple[-1] != len(columns):
            raise ValueError(
                f"{context} component {name!r} shape {shape_tuple} does not "
                f"match its {len(columns)} columns."
            )
        width = int(np.prod(shape_tuple, dtype=np.int64))
        stop = offset + width
        if stop > values.shape[1]:
            raise ValueError(
                f"{context} component {name!r} exceeds the raw observation "
                f"dimension at columns [{offset}, {stop})."
            )
        component_values = values[:, offset:stop].reshape(
            len(values),
            -1,
            len(columns),
        )
        bounds: tuple[tuple[float, float], ...] | None = None
        if name in {"lidar", "lane_camera"}:
            low = component.get("low")
            high = component.get("high")
            if (
                not isinstance(low, list)
                or not isinstance(high, list)
                or len(low) != len(columns)
                or len(high) != len(columns)
            ):
                raise ValueError(
                    f"{context} component {name!r} must declare one low/high "
                    f"bound per column; got low={low!r}, high={high!r}."
                )
            bounds = tuple(
                (float(lower), float(upper))
                for lower, upper in zip(low, high)
            )
            if any(
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower > upper
                for lower, upper in bounds
            ):
                raise ValueError(
                    f"{context} component {name!r} declares invalid bounds "
                    f"low={low!r}, high={high!r}."
                )
        if bounds is not None and component.get("normalized") is not True:
            raise ValueError(
                f"{context} component {name!r} must declare normalized=true "
                "for the supported normalized sensor fields."
            )
        component_fields: dict[str, Any] = {}
        for field_index, field_name in enumerate(columns):
            field_values = component_values[:, :, field_index]
            minimum = float(np.min(field_values))
            maximum = float(np.max(field_values))
            field_receipt: dict[str, Any] = {
                "minimum": minimum,
                "maximum": maximum,
            }
            if bounds is not None:
                lower, upper = bounds[field_index]
                violation_mask = (field_values < lower - tolerance) | (
                    field_values > upper + tolerance
                )
                violation_count = int(np.count_nonzero(violation_mask))
                field_receipt.update(
                    {
                        "declared_lower": float(lower),
                        "declared_upper": float(upper),
                        "violation_count": violation_count,
                    }
                )
                if violation_count:
                    first_row, first_cell = np.argwhere(violation_mask)[0]
                    first_value = float(field_values[first_row, first_cell])
                    raise ValueError(
                        f"{context} violates declared {name}.{field_name} "
                        f"bounds [{lower}, {upper}]: count={violation_count}, "
                        f"minimum={minimum:.9g}, maximum={maximum:.9g}, "
                        f"first_row={int(first_row)}, "
                        f"first_cell={int(first_cell)}, "
                        f"first_value={first_value:.9g}."
                    )
            component_fields[field_name] = field_receipt
            field_receipts[f"{name}.{field_name}"] = field_receipt
        component_receipts[name] = {
            "shape": list(shape_tuple),
            "columns": list(columns),
            "fields": component_fields,
        }
        offset = stop

    if offset != values.shape[1]:
        raise ValueError(
            f"{context} declared components cover {offset} columns, but the "
            f"raw observations have {values.shape[1]}."
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "context": str(context),
        "row_count": int(len(values)),
        "raw_observation_dim": int(values.shape[1]),
        "bound_tolerance": tolerance,
        "components": component_receipts,
        "fields": field_receipts,
    }


def assert_compatible_observation_contracts(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    """Reject sensor contracts that change the actor input semantics."""
    keys: Iterable[str] = (
        "schema_version",
        "raw_observation_components",
        "raw_observation_dim",
        "policy_observation_dim",
        "policy_observation_order",
        "omitted_raw_fields",
        "maximum_range_m",
        "lidar_cells",
        "lane_camera_cells",
        "route_intent_contract",
    )
    for key in keys:
        reference_value = reference.get(key)
        candidate_value = candidate.get(key)
        if isinstance(reference_value, float) or isinstance(candidate_value, float):
            equal = bool(
                np.isclose(
                    float(reference_value),
                    float(candidate_value),
                    atol=1.0e-8,
                    rtol=0.0,
                )
            )
        else:
            equal = reference_value == candidate_value
        if not equal:
            raise ValueError(
                f"Policy observation contract mismatch for {key}: "
                f"{reference_value!r} != {candidate_value!r}."
            )


def validate_training_data_contracts(
    metadata: dict[str, Any],
    *,
    lidar_cells: int,
    maximum_range: float,
    require_explicit: bool,
) -> dict[str, object]:
    """Validate the expert collection against this policy/simulator process."""
    action_contract = metadata.get("continuous_action_contract")
    if not isinstance(action_contract, dict):
        raise ValueError(
            "Action-conditioned training data has no inferred continuous-action contract."
        )
    assert_compatible_action_contracts(
        runtime_continuous_action_contract(),
        action_contract,
    )

    expected_observation = policy_observation_contract(
        lidar_cells=int(lidar_cells),
        maximum_range=float(maximum_range),
    )
    recorded_observation = metadata.get("policy_observation_contract")
    if isinstance(recorded_observation, dict):
        assert_compatible_observation_contracts(
            expected_observation,
            recorded_observation,
        )
    elif bool(require_explicit):
        raise ValueError(
            "Expert data has no explicit policy-observation contract. Recollect "
            "it with the contract-aware collector before a comparison run."
        )

    if bool(require_explicit):
        if metadata.get("continuous_action_contract_explicit") is not True:
            raise ValueError(
                "Not every expert file records an explicit continuous-action contract."
            )
        if metadata.get("policy_observation_contract_explicit") is not True:
            raise ValueError(
                "Not every expert file records an explicit policy-observation contract."
            )
    return {
        "schema_version": 1,
        "continuous_action": action_contract,
        "policy_observation": (
            recorded_observation
            if isinstance(recorded_observation, dict)
            else expected_observation
        ),
        "explicit_contracts_required": bool(require_explicit),
        "explicit_contracts_passed": bool(
            metadata.get("continuous_action_contract_explicit") is True
            and metadata.get("policy_observation_contract_explicit") is True
        ),
    }
