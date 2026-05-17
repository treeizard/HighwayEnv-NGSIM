from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from math import ceil, floor

from .config import PSGAILConfig

ScheduleSegment = tuple[int, int, float, float]


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
        return float(int(min(float(end), floor(scheduled + 1.0e-9))))
    return float(int(max(float(end), ceil(scheduled - 1.0e-9))))


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


@lru_cache(maxsize=128)
def _parse_value_schedule(schedule: str, name: str) -> tuple[ScheduleSegment, ...]:
    text = str(schedule or "").strip()
    if not text:
        return ()
    segments: list[ScheduleSegment] = []
    for raw_segment in text.split(";"):
        segment = raw_segment.strip()
        if not segment:
            continue
        parts = [part.strip() for part in segment.replace(",", ":").split(":")]
        if len(parts) != 4:
            raise ValueError(
                f"{name} segments must be start_round:end_round:start_value:end_value, "
                f"got {segment!r}."
            )
        start_round = int(float(parts[0]))
        end_round = int(float(parts[1]))
        if end_round < start_round:
            raise ValueError(f"{name} end round must be >= start round, got {segment!r}.")
        segments.append((start_round, end_round, float(parts[2]), float(parts[3])))
    return tuple(segments)


@lru_cache(maxsize=128)
def _parse_controlled_vehicle_schedule(schedule: str) -> tuple[ScheduleSegment, ...]:
    segments = _parse_value_schedule(schedule, "controlled_vehicle_schedule")
    for _start_round, _end_round, start_vehicles, end_vehicles in segments:
        _positive_value(start_vehicles, name="controlled_vehicle_schedule start_vehicles")
        _positive_value(end_vehicles, name="controlled_vehicle_schedule end_vehicles")
    return segments


def _effective_round_bounds(segment: ScheduleSegment) -> tuple[int, int]:
    start_round, end_round, _start_value, _end_value = segment
    effective_start = max(1, int(start_round))
    effective_end = max(effective_start, int(end_round))
    return effective_start, effective_end


def _select_segment(
    segments: tuple[ScheduleSegment, ...],
    round_idx: int,
) -> ScheduleSegment | None:
    selected: ScheduleSegment | None = None
    for segment in segments:
        effective_start, effective_end = _effective_round_bounds(segment)
        if effective_start <= int(round_idx) <= effective_end:
            selected = segment
    return selected


def _scheduled_piecewise_value(
    schedule: str,
    round_idx: int,
    *,
    name: str,
) -> float | None:
    segments = _parse_value_schedule(str(schedule or ""), name)
    if not segments:
        return None
    round_idx = int(round_idx)
    selected = _select_segment(segments, round_idx)
    if selected is None:
        first_start, _first_end, first_value, _first_end_value = segments[0]
        _last_start, _last_end, _last_value, last_end_value = segments[-1]
        return float(first_value if round_idx < max(1, first_start) else last_end_value)

    effective_start, effective_end = _effective_round_bounds(selected)
    _start_round, _end_round, start_value, end_value = selected
    return _linear_schedule(
        start=start_value,
        end=end_value,
        round_idx=round_idx - effective_start + 1,
        rounds=effective_end - effective_start + 1,
    )


def _scheduled_controlled_vehicle_piecewise(
    cfg: PSGAILConfig,
    round_idx: int,
) -> float | None:
    segments = _parse_controlled_vehicle_schedule(
        str(getattr(cfg, "controlled_vehicle_schedule", "") or "")
    )
    if not segments:
        return None
    round_idx = int(round_idx)
    selected = _select_segment(segments, round_idx)
    if selected is None:
        first_start, _first_end, first_value, _first_end_value = segments[0]
        _last_start, _last_end, _last_value, last_end_value = segments[-1]
        return float(first_value if round_idx < max(1, first_start) else last_end_value)

    effective_start, effective_end = _effective_round_bounds(selected)
    _start_round, _end_round, start_vehicles, end_vehicles = selected
    local_round = round_idx - effective_start + 1
    local_rounds = effective_end - effective_start + 1
    if start_vehicles >= 1.0 and end_vehicles >= 1.0:
        return _integer_count_schedule(
            start=start_vehicles,
            end=end_vehicles,
            round_idx=local_round,
            rounds=local_rounds,
        )
    return _linear_schedule(
        start=start_vehicles,
        end=end_vehicles,
        round_idx=local_round,
        rounds=local_rounds,
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
    piecewise = _scheduled_piecewise_value(
        str(getattr(cfg, "rollout_target_agent_steps_schedule", "") or ""),
        int(round_idx),
        name="rollout_target_agent_steps_schedule",
    )
    if piecewise is not None:
        return max(1, int(round(piecewise)))

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


def _scheduled_max_episode_steps(cfg: PSGAILConfig, round_idx: int) -> int:
    piecewise = _scheduled_piecewise_value(
        str(getattr(cfg, "max_episode_steps_schedule", "") or ""),
        int(round_idx),
        name="max_episode_steps_schedule",
    )
    if piecewise is not None:
        return max(1, int(round(piecewise)))
    return int(cfg.max_episode_steps)


def _scheduled_gamma(cfg: PSGAILConfig, round_idx: int) -> float:
    piecewise = _scheduled_piecewise_value(
        str(getattr(cfg, "gamma_schedule", "") or ""),
        int(round_idx),
        name="gamma_schedule",
    )
    if piecewise is not None:
        return float(piecewise)

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


def _scheduled_float_or_default(
    cfg: PSGAILConfig,
    round_idx: int,
    *,
    schedule_attr: str,
    name: str,
    default: float,
) -> float:
    piecewise = _scheduled_piecewise_value(
        str(getattr(cfg, schedule_attr, "") or ""),
        int(round_idx),
        name=name,
    )
    return float(default if piecewise is None else piecewise)


def _scheduled_int_or_default(
    cfg: PSGAILConfig,
    round_idx: int,
    *,
    schedule_attr: str,
    name: str,
    default: int,
    minimum: int = 1,
) -> int:
    piecewise = _scheduled_piecewise_value(
        str(getattr(cfg, schedule_attr, "") or ""),
        int(round_idx),
        name=name,
    )
    if piecewise is None:
        return int(default)
    return max(int(minimum), int(round(piecewise)))


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


def _vehicle_increase_warmup_round_idx(
    cfg: PSGAILConfig,
    round_idx: int,
    current_controlled_vehicles: float,
) -> int | None:
    local_warmup_rounds = int(getattr(cfg, "vehicle_increase_warmup_rounds", 0))
    if local_warmup_rounds <= 0:
        return None
    round_idx = max(1, int(round_idx))
    if round_idx <= local_warmup_rounds:
        return round_idx

    previous_controlled_vehicles = float(_scheduled_controlled_vehicles(cfg, round_idx - 1))
    if current_controlled_vehicles > previous_controlled_vehicles + 1.0e-9:
        return 1

    earliest = max(2, round_idx - local_warmup_rounds + 1)
    for candidate_round in range(round_idx - 1, earliest - 1, -1):
        current = float(_scheduled_controlled_vehicles(cfg, candidate_round))
        previous = float(_scheduled_controlled_vehicles(cfg, candidate_round - 1))
        if current > previous + 1.0e-9:
            return round_idx - candidate_round + 1
    return None


def _warmup_context(
    cfg: PSGAILConfig,
    round_idx: int,
    current_controlled_vehicles: float,
) -> tuple[int, int]:
    local_round_idx = _vehicle_increase_warmup_round_idx(
        cfg,
        int(round_idx),
        current_controlled_vehicles,
    )
    if local_round_idx is not None:
        return int(getattr(cfg, "vehicle_increase_warmup_rounds", 0)), int(local_round_idx)
    return int(getattr(cfg, "warmup_rounds", 0)), int(round_idx)


def _apply_warmup(
    *,
    start: float,
    end: float,
    warmup_round_idx: int,
    warmup_rounds: int,
    enabled: bool,
) -> float:
    if not enabled:
        return float(end)
    return _scheduled_warmup_value(
        start=float(start),
        end=float(end),
        round_idx=int(warmup_round_idx),
        warmup_rounds=int(warmup_rounds),
    )


def config_for_round(cfg: PSGAILConfig, round_idx: int) -> PSGAILConfig:
    percentage_controlled_vehicles = float(cfg.percentage_controlled_vehicles)
    if bool(cfg.controlled_vehicle_curriculum):
        percentage_controlled_vehicles = _scheduled_controlled_vehicles(cfg, round_idx)

    warmup_rounds, warmup_round_idx = _warmup_context(
        cfg,
        int(round_idx),
        percentage_controlled_vehicles,
    )
    learning_rate = _apply_warmup(
        start=float(getattr(cfg, "warmup_learning_rate", 0.0)),
        end=float(cfg.learning_rate),
        warmup_round_idx=warmup_round_idx,
        warmup_rounds=warmup_rounds,
        enabled=float(getattr(cfg, "warmup_learning_rate", 0.0)) > 0.0,
    )
    disc_learning_rate = _apply_warmup(
        start=float(getattr(cfg, "warmup_disc_learning_rate", 0.0)),
        end=float(cfg.disc_learning_rate),
        warmup_round_idx=warmup_round_idx,
        warmup_rounds=warmup_rounds,
        enabled=float(getattr(cfg, "warmup_disc_learning_rate", 0.0)) > 0.0,
    )
    entropy_coef = _apply_warmup(
        start=float(getattr(cfg, "warmup_entropy_coef", -1.0)),
        end=float(cfg.entropy_coef),
        warmup_round_idx=warmup_round_idx,
        warmup_rounds=warmup_rounds,
        enabled=float(getattr(cfg, "warmup_entropy_coef", -1.0)) >= 0.0,
    )
    clip_range = _apply_warmup(
        start=float(getattr(cfg, "warmup_clip_range", 0.0)),
        end=float(cfg.clip_range),
        warmup_round_idx=warmup_round_idx,
        warmup_rounds=warmup_rounds,
        enabled=float(getattr(cfg, "warmup_clip_range", 0.0)) > 0.0,
    )
    gail_reward_clip = _apply_warmup(
        start=float(getattr(cfg, "warmup_gail_reward_clip", 0.0)),
        end=float(cfg.gail_reward_clip),
        warmup_round_idx=warmup_round_idx,
        warmup_rounds=warmup_rounds,
        enabled=float(getattr(cfg, "warmup_gail_reward_clip", 0.0)) > 0.0,
    )
    final_reward_clip = _apply_warmup(
        start=float(getattr(cfg, "warmup_final_reward_clip", 0.0)),
        end=float(cfg.final_reward_clip),
        warmup_round_idx=warmup_round_idx,
        warmup_rounds=warmup_rounds,
        enabled=float(getattr(cfg, "warmup_final_reward_clip", 0.0)) > 0.0,
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
                        round_idx=warmup_round_idx,
                        warmup_rounds=warmup_rounds,
                    )
                )
            ),
        )

    learning_rate = _scheduled_float_or_default(
        cfg,
        round_idx,
        schedule_attr="learning_rate_schedule",
        name="learning_rate_schedule",
        default=learning_rate,
    )
    disc_learning_rate = _scheduled_float_or_default(
        cfg,
        round_idx,
        schedule_attr="disc_learning_rate_schedule",
        name="disc_learning_rate_schedule",
        default=disc_learning_rate,
    )
    entropy_coef = _scheduled_float_or_default(
        cfg,
        round_idx,
        schedule_attr="entropy_coef_schedule",
        name="entropy_coef_schedule",
        default=entropy_coef,
    )
    clip_range = _scheduled_float_or_default(
        cfg,
        round_idx,
        schedule_attr="clip_range_schedule",
        name="clip_range_schedule",
        default=clip_range,
    )
    disc_updates_per_round = _scheduled_int_or_default(
        cfg,
        round_idx,
        schedule_attr="disc_updates_per_round_schedule",
        name="disc_updates_per_round_schedule",
        default=disc_updates_per_round,
        minimum=1,
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
        max_episode_steps=_scheduled_max_episode_steps(cfg, round_idx),
        policy_bc_regularization_coef=_scheduled_policy_bc_coef(cfg, round_idx),
        rollout_target_agent_steps=_scheduled_rollout_target_agent_steps(cfg, round_idx),
        gamma=_scheduled_gamma(cfg, round_idx),
    )
