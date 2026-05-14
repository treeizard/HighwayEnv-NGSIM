from __future__ import annotations

from dataclasses import replace

from .config import PSGAILConfig


def _positive_value(value: float | int, *, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _linear_schedule(
    *,
    start: float,
    end: float,
    round_idx: int,
    rounds: int,
) -> float:
    rounds = max(1, int(rounds))
    if rounds == 1:
        return float(end)
    progress = min(1.0, max(0.0, (int(round_idx) - 1) / float(rounds - 1)))
    return float(start + (end - start) * progress)


def _scheduled_controlled_vehicles(cfg: PSGAILConfig, round_idx: int) -> float:
    start = _positive_value(
        cfg.initial_controlled_vehicles,
        name="initial_controlled_vehicles",
    )
    end = _positive_value(
        cfg.final_controlled_vehicles,
        name="final_controlled_vehicles",
    )
    return _linear_schedule(
        start=start,
        end=end,
        round_idx=int(round_idx),
        rounds=int(cfg.controlled_vehicle_curriculum_rounds),
    )


def _scheduled_rollout_target_agent_steps(cfg: PSGAILConfig, round_idx: int) -> int:
    start = int(cfg.initial_rollout_target_agent_steps)
    end = int(cfg.final_rollout_target_agent_steps)
    if start <= 0 and end <= 0:
        return int(cfg.rollout_target_agent_steps)
    if start <= 0:
        start = (
            int(cfg.rollout_target_agent_steps)
            if int(cfg.rollout_target_agent_steps) > 0
            else end
        )
    if end <= 0:
        end = start
    return max(
        1,
        int(
            round(
                _linear_schedule(
                    start=float(start),
                    end=float(end),
                    round_idx=int(round_idx),
                    rounds=int(cfg.rollout_target_agent_steps_curriculum_rounds),
                )
            )
        ),
    )


def _scheduled_gamma(cfg: PSGAILConfig, round_idx: int) -> float:
    start = (
        float(cfg.initial_gamma)
        if float(cfg.initial_gamma) > 0.0
        else float(cfg.gamma)
    )
    end = float(cfg.final_gamma) if float(cfg.final_gamma) > 0.0 else start
    if int(cfg.gamma_curriculum_rounds) <= 0 and float(cfg.final_gamma) <= 0.0:
        return float(cfg.gamma)
    return _linear_schedule(
        start=start,
        end=end,
        round_idx=int(round_idx),
        rounds=int(cfg.gamma_curriculum_rounds),
    )


def config_for_round(cfg: PSGAILConfig, round_idx: int) -> PSGAILConfig:
    percentage_controlled_vehicles = float(cfg.percentage_controlled_vehicles)
    if bool(cfg.controlled_vehicle_curriculum):
        percentage_controlled_vehicles = _scheduled_controlled_vehicles(cfg, round_idx)

    return replace(
        cfg,
        control_all_vehicles=(
            False
            if bool(cfg.controlled_vehicle_curriculum)
            else bool(cfg.control_all_vehicles)
        ),
        percentage_controlled_vehicles=percentage_controlled_vehicles,
        rollout_target_agent_steps=_scheduled_rollout_target_agent_steps(cfg, round_idx),
        gamma=_scheduled_gamma(cfg, round_idx),
    )
