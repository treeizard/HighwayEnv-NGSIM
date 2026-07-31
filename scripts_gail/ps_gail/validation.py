"""Validation scoring and best-checkpoint helpers for imitation trainers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PSGAILConfig


PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK = "shared_bc_gail_paper_metrics_v1"
POLICY_REALISM_QUALIFICATION_FRAMEWORK = "closed_loop_policy_quality_v1"


def paper_driver_model_validation_overrides() -> dict[str, Any]:
    """Shared Bhattacharyya et al.-aligned BC/GAIL validation contract."""
    return {
        "validation_vehicle_mode": "all",
        "test_vehicle_mode": "all",
        "evaluation_horizons_seconds": "1,5,10,20",
        "evaluation_terminate_when_all_controlled_crashed": False,
        "validation_score_horizon_seconds": 20,
        # Paper-style RMSE curves use the vehicles with expert reference
        # available at each horizon. Coverage is reported, not thresholded.
        "validation_require_exact_horizon": False,
        "validation_min_horizon_coverage": 0.0,
        "validation_score_crash_metric": "vehicle",
        "validation_score_position_weight": 1.0,
        "validation_score_speed_weight": 0.5,
        "validation_score_lane_offset_weight": 2.0,
        "validation_score_crash_weight": 25.0,
        "validation_score_offroad_weight": 25.0,
        "validation_score_hard_brake_weight": 2.0,
    }


def _finite_metric(
    metrics: dict[str, float],
    keys: tuple[str, ...],
) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            return value
    return float("nan")


def closed_loop_policy_quality(
    metrics: dict[str, float],
    *,
    prefix: str,
    max_vehicle_crash_rate: float,
    max_vehicle_offroad_rate: float,
    score_horizon_seconds: int = 20,
    min_horizon_coverage: float = 0.0,
) -> dict[str, Any]:
    """Apply explicit safety/coverage gates to one matched rollout split.

    A completed finite rollout is evidence that the evaluator ran, not that the
    policy drove acceptably.  Keep that distinction machine-readable so a
    catastrophic policy cannot be labelled as passing merely because its
    metrics are finite.
    """

    thresholds = {
        "max_vehicle_crash_rate": float(max_vehicle_crash_rate),
        "max_vehicle_offroad_rate": float(max_vehicle_offroad_rate),
        "min_horizon_coverage": float(min_horizon_coverage),
    }
    if not 0.0 <= thresholds["max_vehicle_crash_rate"] <= 1.0:
        raise ValueError("max_vehicle_crash_rate must be in [0, 1].")
    if not 0.0 <= thresholds["max_vehicle_offroad_rate"] <= 1.0:
        raise ValueError("max_vehicle_offroad_rate must be in [0, 1].")
    if not 0.0 <= thresholds["min_horizon_coverage"] <= 1.0:
        raise ValueError("min_horizon_coverage must be in [0, 1].")

    horizon = int(score_horizon_seconds)
    observed = {
        "vehicle_crash_rate": _finite_metric(
            metrics,
            (
                f"{prefix}/vehicle_crash_rate",
                f"{prefix}/collision_rate",
            ),
        ),
        "vehicle_offroad_rate": _finite_metric(
            metrics,
            (
                f"{prefix}/vehicle_offroad_rate",
                f"{prefix}/offroad_duration_rate",
            ),
        ),
        "horizon_coverage": _finite_metric(
            metrics,
            (
                f"{prefix}/rollout_horizon_coverage_{horizon}s",
                f"{prefix}/horizon_coverage_{horizon}s",
            ),
        ),
        "reference_horizon_coverage": _finite_metric(
            metrics,
            (
                f"{prefix}/reference_horizon_coverage_{horizon}s",
                f"{prefix}/horizon_coverage_{horizon}s",
            ),
        ),
    }
    checks = {
        "finite_vehicle_crash_rate": bool(
            np.isfinite(observed["vehicle_crash_rate"])
        ),
        "finite_vehicle_offroad_rate": bool(
            np.isfinite(observed["vehicle_offroad_rate"])
        ),
        "finite_horizon_coverage": bool(
            np.isfinite(observed["horizon_coverage"])
        ),
        "vehicle_crash_rate": bool(
            np.isfinite(observed["vehicle_crash_rate"])
            and observed["vehicle_crash_rate"]
            <= thresholds["max_vehicle_crash_rate"]
        ),
        "vehicle_offroad_rate": bool(
            np.isfinite(observed["vehicle_offroad_rate"])
            and observed["vehicle_offroad_rate"]
            <= thresholds["max_vehicle_offroad_rate"]
        ),
        "horizon_coverage": bool(
            np.isfinite(observed["horizon_coverage"])
            and observed["horizon_coverage"]
            >= thresholds["min_horizon_coverage"]
        ),
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "framework": POLICY_REALISM_QUALIFICATION_FRAMEWORK,
        "split": str(prefix),
        "score_horizon_seconds": horizon,
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "failed_checks": failed_checks,
        "passed": not failed_checks,
    }


def action_learning_gate(
    *,
    split: str,
    prediction_std_ratios: list[float] | tuple[float, ...],
    prediction_target_correlations: list[float] | tuple[float, ...],
    action_indices: list[int],
    minimum_std_ratios: list[float],
    minimum_correlations: list[float],
) -> dict[str, Any]:
    """Return a fail-closed per-action imitation gate for one offline split."""

    if not (
        len(action_indices)
        == len(minimum_std_ratios)
        == len(minimum_correlations)
    ):
        raise ValueError(
            "Learning action indices and per-action gate thresholds must have "
            "the same length."
        )
    action_names = ("acceleration_norm", "steering_norm")
    gates: list[dict[str, object]] = []
    for action_index, minimum_std_ratio, minimum_correlation in zip(
        action_indices,
        minimum_std_ratios,
        minimum_correlations,
    ):
        if not 0 <= int(action_index) < len(prediction_std_ratios):
            raise ValueError(
                f"learning_action_index={action_index} is outside the action "
                f"dimension [0, {len(prediction_std_ratios)})."
            )
        std_ratio = float(prediction_std_ratios[int(action_index)])
        correlation = float(prediction_target_correlations[int(action_index)])
        passed = bool(
            np.isfinite(std_ratio)
            and np.isfinite(correlation)
            and std_ratio >= float(minimum_std_ratio)
            and correlation >= float(minimum_correlation)
        )
        gates.append(
            {
                "action_index": int(action_index),
                "action_name": action_names[int(action_index)],
                "prediction_std_ratio": std_ratio,
                "minimum_prediction_std_ratio": float(minimum_std_ratio),
                "prediction_target_correlation": correlation,
                "minimum_prediction_target_correlation": float(
                    minimum_correlation
                ),
                "passed": passed,
            }
        )
    return {
        "split": str(split),
        "aggregation": "all_required_actions",
        "actions": gates,
        "passed": all(bool(gate["passed"]) for gate in gates),
        # Compatibility fields for older report readers. The actions list is
        # authoritative and all listed dimensions must pass.
        "action_index": int(gates[0]["action_index"]),
        "prediction_std_ratio": float(gates[0]["prediction_std_ratio"]),
        "minimum_prediction_std_ratio": float(
            gates[0]["minimum_prediction_std_ratio"]
        ),
        "prediction_target_correlation": float(
            gates[0]["prediction_target_correlation"]
        ),
        "minimum_prediction_target_correlation": float(
            gates[0]["minimum_prediction_target_correlation"]
        ),
    }


def validation_cost_and_score(
    metrics: dict[str, float],
    cfg: PSGAILConfig,
    *,
    prefix: str = "validation",
) -> tuple[float, float, dict[str, float]]:
    """Return weighted validation cost, score, and component values."""
    horizon = int(getattr(cfg, "validation_score_horizon_seconds", 20))
    strict_horizon = bool(getattr(cfg, "validation_require_exact_horizon", False))
    crash_metric = str(getattr(cfg, "validation_score_crash_metric", "duration")).strip().lower()
    if crash_metric not in {"duration", "vehicle"}:
        raise ValueError(
            "validation_score_crash_metric must be 'duration' or 'vehicle', "
            f"got {crash_metric!r}."
        )
    position_keys = (f"{prefix}/rmse_position_{horizon}s",)
    speed_keys = (f"{prefix}/rmse_speed_{horizon}s",)
    lane_keys = (f"{prefix}/rmse_lane_offset_{horizon}s",)
    if not strict_horizon:
        position_keys += (f"{prefix}/rmse_position_final",)
        speed_keys += (f"{prefix}/rmse_speed_final",)
        lane_keys += (f"{prefix}/rmse_lane_offset_final",)
    crash_keys = (
        (
            f"{prefix}/vehicle_crash_rate",
            f"{prefix}/collision_rate",
            f"{prefix}/collision_duration_rate",
        )
        if crash_metric == "vehicle"
        else (
            f"{prefix}/collision_duration_rate",
            f"{prefix}/crash_agent_fraction",
            f"{prefix}/vehicle_crash_rate",
            f"{prefix}/collision_rate",
        )
    )
    components = {
        "position_rmse": _finite_metric(metrics, position_keys),
        "speed_rmse": _finite_metric(metrics, speed_keys),
        "lane_offset_rmse": _finite_metric(metrics, lane_keys),
        "crash_rate": _finite_metric(metrics, crash_keys),
        "vehicle_offroad_rate": _finite_metric(
            metrics,
            (
                f"{prefix}/vehicle_offroad_rate",
                f"{prefix}/offroad_duration_rate",
            ),
        ),
        "hard_brake_rate": _finite_metric(metrics, (f"{prefix}/hard_brake_rate",)),
        "horizon_coverage": _finite_metric(
            metrics,
            (
                f"{prefix}/reference_horizon_coverage_{horizon}s",
                f"{prefix}/horizon_coverage_{horizon}s",
            ),
        ),
        "rollout_horizon_coverage": _finite_metric(
            metrics,
            (
                f"{prefix}/rollout_horizon_coverage_{horizon}s",
                f"{prefix}/horizon_coverage_{horizon}s",
            ),
        ),
    }
    required_components = {
        key: value
        for key, value in components.items()
        if key not in {"horizon_coverage", "rollout_horizon_coverage"}
    }
    if not all(np.isfinite(value) for value in required_components.values()):
        return float("inf"), float("-inf"), components
    minimum_coverage = max(0.0, float(getattr(cfg, "validation_min_horizon_coverage", 0.0)))
    if strict_horizon:
        coverage = float(components["horizon_coverage"])
        if not np.isfinite(coverage) or coverage < minimum_coverage:
            return float("inf"), float("-inf"), components

    cost = (
        float(getattr(cfg, "validation_score_position_weight", 1.0)) * components["position_rmse"]
        + float(getattr(cfg, "validation_score_speed_weight", 0.5)) * components["speed_rmse"]
        + float(getattr(cfg, "validation_score_lane_offset_weight", 2.0)) * components["lane_offset_rmse"]
        + float(getattr(cfg, "validation_score_crash_weight", 25.0)) * components["crash_rate"]
        + float(getattr(cfg, "validation_score_offroad_weight", 25.0)) * components["vehicle_offroad_rate"]
        + float(getattr(cfg, "validation_score_hard_brake_weight", 2.0)) * components["hard_brake_rate"]
    )
    return float(cost), float(-cost), components


def scored_validation_metrics(
    metrics: dict[str, float],
    cfg: PSGAILConfig,
    *,
    prefix: str = "validation",
) -> tuple[dict[str, float], float, float]:
    """Return metrics augmented with validation score/cost entries."""
    cost, score, components = validation_cost_and_score(metrics, cfg, prefix=prefix)
    scored = dict(metrics)
    scored[f"{prefix}/cost"] = float(cost)
    scored[f"{prefix}/score"] = float(score)
    for name, value in components.items():
        scored[f"{prefix}/score_component_{name}"] = float(value)
    scored[f"{prefix}/score_component_vehicle_crash_rate"] = _finite_metric(
        scored,
        (
            f"{prefix}/vehicle_crash_rate",
            f"{prefix}/collision_rate",
        ),
    )
    return scored, cost, score


def best_checkpoint_payload(
    base_payload: dict[str, Any],
    *,
    round_idx: int,
    validation_metrics: dict[str, float],
    validation_score: float,
    validation_cost: float,
) -> dict[str, Any]:
    """Attach best-validation metadata to a normal trainer checkpoint payload."""
    payload = dict(base_payload)
    method = str(payload.get("method") or "").strip().lower()
    if method:
        from .checkpoints import set_checkpoint_kind

        payload = set_checkpoint_kind(payload, f"{method}_best")
    payload.update(
        {
            "best_round": int(round_idx),
            "validation_metrics": dict(validation_metrics),
            "validation_score": float(validation_score),
            "validation_cost": float(validation_cost),
        }
    )
    return payload


def matched_validation_summary(prefix: str, label: str, metrics: dict[str, float]) -> str:
    """Compact human-readable matched-validation summary."""
    return (
        f"[{prefix} {label}] "
        f"episodes={metrics.get(f'{prefix}/episodes', 0):.0f} "
        f"vehicles={metrics.get(f'{prefix}/vehicles', metrics.get(f'{prefix}/episodes', 0)):.0f} "
        f"vehicle_episodes={metrics.get(f'{prefix}/vehicle_episodes', metrics.get(f'{prefix}/episodes', 0)):.0f} "
        f"rmse_pos_20s={metrics.get(f'{prefix}/rmse_position_20s', float('nan')):.4f} "
        f"rmse_speed_20s={metrics.get(f'{prefix}/rmse_speed_20s', float('nan')):.4f} "
        f"rmse_lane_20s={metrics.get(f'{prefix}/rmse_lane_offset_20s', float('nan')):.4f} "
        f"crash_agent={metrics.get(f'{prefix}/crash_agent_fraction', metrics.get(f'{prefix}/collision_duration_rate', 0.0)):.4f} "
        f"vehicle_crash={metrics.get(f'{prefix}/vehicle_crash_rate', metrics.get(f'{prefix}/collision_rate', 0.0)):.4f} "
        f"offroad={metrics.get(f'{prefix}/vehicle_offroad_rate', metrics.get(f'{prefix}/offroad_duration_rate', 0.0)):.4f} "
        f"hard_brake={metrics.get(f'{prefix}/hard_brake_rate', 0.0):.4f} "
        f"score={metrics.get(f'{prefix}/score', float('nan')):.4f}"
    )
