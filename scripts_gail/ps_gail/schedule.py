from __future__ import annotations

from dataclasses import replace

from .config import PSGAILConfig


def _clamp_fraction(value: float) -> float:
    return float(min(1.0, max(1e-6, float(value))))


def _phase1_agent_count(cfg: PSGAILConfig, round_idx: int) -> int:
    interval = max(1, int(getattr(cfg, "paper_agent_increment_interval", 200)))
    increments = (max(1, int(round_idx)) - 1) // interval
    return max(
        1,
        int(getattr(cfg, "paper_initial_agent_count", 10))
        + increments * int(getattr(cfg, "paper_agent_increment", 10)),
    )


def _phase2_agent_count(cfg: PSGAILConfig, phase2_round_idx: int) -> int:
    final_count = max(1, int(getattr(cfg, "paper_phase2_agent_count", 100)))
    ramp_rounds = max(0, int(getattr(cfg, "paper_phase2_agent_ramp_rounds", 0)))
    initial_count = int(getattr(cfg, "paper_phase2_initial_agent_count", 0))
    if ramp_rounds <= 0 or initial_count <= 0:
        return final_count
    if ramp_rounds == 1:
        return final_count

    start = max(1, initial_count)
    progress = min(1.0, max(0.0, (int(phase2_round_idx) - 1) / float(ramp_rounds - 1)))
    count = round(start + (final_count - start) * progress)
    return max(1, int(count))


def config_for_round(cfg: PSGAILConfig, round_idx: int) -> PSGAILConfig:
    if bool(getattr(cfg, "paper_style_training", False)):
        phase1_rounds = max(1, int(getattr(cfg, "paper_phase1_rounds", 1000)))
        if int(round_idx) <= phase1_rounds:
            return replace(
                cfg,
                control_all_vehicles=False,
                percentage_controlled_vehicles=float(_phase1_agent_count(cfg, round_idx)),
                gamma=float(getattr(cfg, "paper_phase1_gamma", 0.95)),
                rollout_target_agent_steps=max(
                    1,
                    int(getattr(cfg, "paper_phase1_agent_steps", 10_000)),
                ),
            )

        phase2_round_idx = int(round_idx) - phase1_rounds
        return replace(
            cfg,
            control_all_vehicles=False,
            percentage_controlled_vehicles=float(_phase2_agent_count(cfg, phase2_round_idx)),
            gamma=float(getattr(cfg, "paper_phase2_gamma", 0.99)),
            rollout_target_agent_steps=max(
                1,
                int(getattr(cfg, "paper_phase2_agent_steps", 40_000)),
            ),
        )

    if not bool(cfg.controlled_vehicle_curriculum):
        return cfg

    curriculum_rounds = max(1, int(cfg.controlled_vehicle_curriculum_rounds))
    progress = (
        1.0
        if curriculum_rounds == 1
        else min(1.0, max(0.0, (int(round_idx) - 1) / float(curriculum_rounds - 1)))
    )
    initial = _clamp_fraction(cfg.initial_controlled_vehicle_fraction)
    final = _clamp_fraction(cfg.final_controlled_vehicle_fraction)
    fraction = initial + (final - initial) * progress
    return replace(
        cfg,
        control_all_vehicles=False,
        percentage_controlled_vehicles=_clamp_fraction(fraction),
    )
