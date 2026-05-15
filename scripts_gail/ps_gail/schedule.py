from __future__ import annotations

from dataclasses import replace

import numpy as np

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


def _integer_count_schedule(
    *,
    start: float,
    end: float,
    round_idx: int,
    rounds: int,
) -> float:
    scheduled = _linear_schedule(
        start=float(start),
        end=float(end),
        round_idx=int(round_idx),
        rounds=int(rounds),
    )
    if float(end) >= float(start):
        return float(int(min(float(end), np.floor(scheduled + 1.0e-9))))
    return float(int(max(float(end), np.ceil(scheduled - 1.0e-9))))


def _integer_count_step_schedule(
    *,
    start: float,
    end: float,
    round_idx: int,
    increment_rounds: int,
) -> float:
    increment_rounds = max(1, int(increment_rounds))
    start_count = int(round(float(start)))
    end_count = int(round(float(end)))
    direction = 1 if end_count >= start_count else -1
    increments_elapsed = (max(1, int(round_idx)) - 1) // increment_rounds
    scheduled_count = start_count + direction * int(increments_elapsed)
    if direction > 0:
        scheduled_count = min(end_count, scheduled_count)
    else:
        scheduled_count = max(end_count, scheduled_count)
    return float(scheduled_count)


def _parse_controlled_vehicle_schedule(
    schedule: str,
) -> list[tuple[int, int, float, float]]:
    segments: list[tuple[int, int, float, float]] = []
    text = str(schedule or "").strip()
    if not text:
        return segments
    for raw_segment in text.split(";"):
        segment = raw_segment.strip()
        if not segment:
            continue
        parts = [part.strip() for part in segment.replace(",", ":").split(":")]
        if len(parts) != 4:
            raise ValueError(
                "controlled_vehicle_schedule segments must be "
                "start_round:end_round:start_vehicles:end_vehicles, got "
                f"{segment!r}."
            )
        start_round = int(float(parts[0]))
        end_round = int(float(parts[1]))
        start_vehicles = _positive_value(
            float(parts[2]),
            name="controlled_vehicle_schedule start_vehicles",
        )
        end_vehicles = _positive_value(
            float(parts[3]),
            name="controlled_vehicle_schedule end_vehicles",
        )
        if end_round < start_round:
            raise ValueError(
                f"controlled_vehicle_schedule end round must be >= start round, got {segment!r}."
            )
        segments.append((start_round, end_round, start_vehicles, end_vehicles))
    return segments


def _scheduled_controlled_vehicle_piecewise(cfg: PSGAILConfig, round_idx: int) -> float | None:
    segments = _parse_controlled_vehicle_schedule(
        str(getattr(cfg, "controlled_vehicle_schedule", "") or "")
    )
    if not segments:
        return None
    round_idx = int(round_idx)
    selected = None
    for segment in segments:
        start_round, end_round, start_vehicles, end_vehicles = segment
        effective_start = max(1, int(start_round))
        effective_end = max(effective_start, int(end_round))
        if effective_start <= round_idx <= effective_end:
            selected = (effective_start, effective_end, start_vehicles, end_vehicles)
    if selected is None:
        if round_idx < max(1, segments[0][0]):
            return float(segments[0][2])
        return float(segments[-1][3])
    effective_start, effective_end, start_vehicles, end_vehicles = selected
    if start_vehicles >= 1.0 and end_vehicles >= 1.0:
        return _integer_count_schedule(
            start=start_vehicles,
            end=end_vehicles,
            round_idx=round_idx - effective_start + 1,
            rounds=effective_end - effective_start + 1,
        )
    return _linear_schedule(
        start=start_vehicles,
        end=end_vehicles,
        round_idx=round_idx - effective_start + 1,
        rounds=effective_end - effective_start + 1,
    )


def _scheduled_controlled_vehicles(cfg: PSGAILConfig, round_idx: int) -> float:
    piecewise = _scheduled_controlled_vehicle_piecewise(cfg, round_idx)
    if piecewise is not None:
        return piecewise
    start = _positive_value(
        cfg.initial_controlled_vehicles,
        name="initial_controlled_vehicles",
    )
    end = _positive_value(
        cfg.final_controlled_vehicles,
        name="final_controlled_vehicles",
    )
    if start >= 1.0 and end >= 1.0:
        increment_rounds = int(getattr(cfg, "controlled_vehicle_increment_rounds", 0))
        if increment_rounds > 0:
            return _integer_count_step_schedule(
                start=start,
                end=end,
                round_idx=int(round_idx),
                increment_rounds=increment_rounds,
            )
        return _integer_count_schedule(
            start=start,
            end=end,
            round_idx=int(round_idx),
            rounds=int(cfg.controlled_vehicle_curriculum_rounds),
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


def _scheduled_policy_bc_coef(cfg: PSGAILConfig, round_idx: int) -> float:
    start = max(0.0, float(getattr(cfg, "policy_bc_regularization_coef", 0.0)))
    end = max(0.0, float(getattr(cfg, "policy_bc_regularization_final_coef", 0.0)))
    rounds = int(getattr(cfg, "policy_bc_regularization_decay_rounds", 0))
    if start <= 0.0 or rounds <= 0:
        return start
    return _linear_schedule(start=start, end=end, round_idx=int(round_idx), rounds=rounds)


def _scheduled_warmup_value(
    *,
    start: float,
    end: float,
    round_idx: int,
    warmup_rounds: int,
) -> float:
    warmup_rounds = int(warmup_rounds)
    if warmup_rounds <= 0 or int(round_idx) > warmup_rounds:
        return float(end)
    return _linear_schedule(
        start=float(start),
        end=float(end),
        round_idx=int(round_idx),
        rounds=warmup_rounds,
    )


def config_for_round(cfg: PSGAILConfig, round_idx: int) -> PSGAILConfig:
    percentage_controlled_vehicles = float(cfg.percentage_controlled_vehicles)
    if bool(cfg.controlled_vehicle_curriculum):
        percentage_controlled_vehicles = _scheduled_controlled_vehicles(cfg, round_idx)

    warmup_rounds = int(getattr(cfg, "warmup_rounds", 0))
    learning_rate = float(cfg.learning_rate)
    warmup_learning_rate = float(getattr(cfg, "warmup_learning_rate", 0.0))
    if warmup_learning_rate > 0.0:
        learning_rate = _scheduled_warmup_value(
            start=warmup_learning_rate,
            end=float(cfg.learning_rate),
            round_idx=int(round_idx),
            warmup_rounds=warmup_rounds,
        )
    disc_learning_rate = float(cfg.disc_learning_rate)
    warmup_disc_learning_rate = float(getattr(cfg, "warmup_disc_learning_rate", 0.0))
    if warmup_disc_learning_rate > 0.0:
        disc_learning_rate = _scheduled_warmup_value(
            start=warmup_disc_learning_rate,
            end=float(cfg.disc_learning_rate),
            round_idx=int(round_idx),
            warmup_rounds=warmup_rounds,
        )
    entropy_coef = float(cfg.entropy_coef)
    warmup_entropy_coef = float(getattr(cfg, "warmup_entropy_coef", -1.0))
    if warmup_entropy_coef >= 0.0:
        entropy_coef = _scheduled_warmup_value(
            start=warmup_entropy_coef,
            end=float(cfg.entropy_coef),
            round_idx=int(round_idx),
            warmup_rounds=warmup_rounds,
        )
    clip_range = float(cfg.clip_range)
    warmup_clip_range = float(getattr(cfg, "warmup_clip_range", 0.0))
    if warmup_clip_range > 0.0:
        clip_range = _scheduled_warmup_value(
            start=warmup_clip_range,
            end=float(cfg.clip_range),
            round_idx=int(round_idx),
            warmup_rounds=warmup_rounds,
        )
    gail_reward_clip = float(cfg.gail_reward_clip)
    warmup_gail_reward_clip = float(getattr(cfg, "warmup_gail_reward_clip", 0.0))
    if warmup_gail_reward_clip > 0.0:
        gail_reward_clip = _scheduled_warmup_value(
            start=warmup_gail_reward_clip,
            end=float(cfg.gail_reward_clip),
            round_idx=int(round_idx),
            warmup_rounds=warmup_rounds,
        )
    final_reward_clip = float(cfg.final_reward_clip)
    warmup_final_reward_clip = float(getattr(cfg, "warmup_final_reward_clip", 0.0))
    if warmup_final_reward_clip > 0.0:
        final_reward_clip = _scheduled_warmup_value(
            start=warmup_final_reward_clip,
            end=float(cfg.final_reward_clip),
            round_idx=int(round_idx),
            warmup_rounds=warmup_rounds,
        )
    disc_updates_per_round = int(cfg.disc_updates_per_round)
    warmup_disc_updates = int(getattr(cfg, "warmup_disc_updates_per_round", 0))
    if warmup_disc_updates > 0:
        disc_updates_per_round = max(
            1,
            int(
                round(
                    _scheduled_warmup_value(
                        start=float(warmup_disc_updates),
                        end=float(cfg.disc_updates_per_round),
                        round_idx=int(round_idx),
                        warmup_rounds=warmup_rounds,
                    )
                )
            ),
        )

    return replace(
        cfg,
        control_all_vehicles=(
            False
            if bool(cfg.controlled_vehicle_curriculum)
            else bool(cfg.control_all_vehicles)
        ),
        percentage_controlled_vehicles=percentage_controlled_vehicles,
        learning_rate=learning_rate,
        disc_learning_rate=disc_learning_rate,
        entropy_coef=entropy_coef,
        clip_range=clip_range,
        disc_updates_per_round=disc_updates_per_round,
        gail_reward_clip=gail_reward_clip,
        final_reward_clip=final_reward_clip,
        policy_bc_regularization_coef=_scheduled_policy_bc_coef(cfg, round_idx),
        rollout_target_agent_steps=_scheduled_rollout_target_agent_steps(cfg, round_idx),
        gamma=_scheduled_gamma(cfg, round_idx),
    )
