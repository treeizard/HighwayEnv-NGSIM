"""Validation scoring and best-checkpoint helpers for imitation trainers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PSGAILConfig


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


def validation_cost_and_score(
    metrics: dict[str, float],
    cfg: PSGAILConfig,
    *,
    prefix: str = "validation",
) -> tuple[float, float, dict[str, float]]:
    """Return weighted validation cost, score, and component values."""
    horizon = int(getattr(cfg, "validation_score_horizon_seconds", 20))
    components = {
        "position_rmse": _finite_metric(
            metrics,
            (
                f"{prefix}/rmse_position_{horizon}s",
                f"{prefix}/rmse_position_final",
            ),
        ),
        "speed_rmse": _finite_metric(
            metrics,
            (
                f"{prefix}/rmse_speed_{horizon}s",
                f"{prefix}/rmse_speed_final",
            ),
        ),
        "lane_offset_rmse": _finite_metric(
            metrics,
            (
                f"{prefix}/rmse_lane_offset_{horizon}s",
                f"{prefix}/rmse_lane_offset_final",
            ),
        ),
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
        "hard_brake_rate": _finite_metric(metrics, (f"{prefix}/hard_brake_rate",)),
    }
    if not all(np.isfinite(value) for value in components.values()):
        return float("inf"), float("-inf"), components

    cost = (
        float(getattr(cfg, "validation_score_position_weight", 1.0)) * components["position_rmse"]
        + float(getattr(cfg, "validation_score_speed_weight", 0.5)) * components["speed_rmse"]
        + float(getattr(cfg, "validation_score_lane_offset_weight", 2.0)) * components["lane_offset_rmse"]
        + float(getattr(cfg, "validation_score_crash_weight", 25.0)) * components["vehicle_crash_rate"]
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
        f"collision={metrics.get(f'{prefix}/vehicle_crash_rate', metrics.get(f'{prefix}/collision_rate', 0.0)):.4f} "
        f"offroad={metrics.get(f'{prefix}/vehicle_offroad_rate', metrics.get(f'{prefix}/offroad_duration_rate', 0.0)):.4f} "
        f"hard_brake={metrics.get(f'{prefix}/hard_brake_rate', 0.0):.4f} "
        f"score={metrics.get(f'{prefix}/score', float('nan')):.4f}"
    )
