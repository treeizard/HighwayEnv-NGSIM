"""Matched-trajectory evaluation utilities for PS-GAIL policies."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import OrderedDict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env
from highway_env.ngsim_utils.core.constants import MAX_STEER, denormalize_acceleration
from highway_env.ngsim_utils.data.episode_selection import resolve_num_ego_vehicles
from highway_env.ngsim_utils.data.prebuilt import load_prebuilt_data

from ..config import PSGAILConfig
from ..envs import configure_native_continuous_action_passthrough
from ..data import (
    SCENE_FEATURE_DIM_PER_VEHICLE,
    build_sequence_windows,
    discriminator_features,
    normalize_trajectory_frame,
    scene_snapshot_features,
    standardize_features,
    transform_sequence_features,
)
from ..models import NUM_DISCRETE_META_ACTIONS, make_actor_critic
from ..observations import flatten_agent_observations, policy_observations_from_flat

from .policy import (
    _actions_to_env_tuple,
    _is_continuous,
    _make_policy_from_state_dict,
    _masked_discrete_logits,
    central_critic_observation_dim,
    central_critic_observations,
    centralized_critic_enabled,
    discrete_action_masks_from_env,
    policy_action_dim,
    recurrent_policy_enabled,
    _shift_recurrent_memory,
)

_EVAL_POLICY_CACHE = {}
_EVAL_ENV_CACHE = OrderedDict()
_EVAL_PREBUILT_CACHE = {}
_EVAL_ENV_CACHE_HITS = 0
_EVAL_ENV_CACHE_MISSES = 0
EpisodeSpec = tuple[int, str, tuple[int, ...] | None]

@contextmanager
def evaluation_thread_context(cfg: PSGAILConfig):
    """Apply the configured native thread budget while evaluation is running."""
    requested_workers = max(1, int(getattr(cfg, "evaluation_num_workers", 1)))
    if requested_workers != 1:
        print(
            "[evaluation] evaluation_num_workers="
            f"{requested_workers} requested; serial evaluation path is active for this call.",
            flush=True,
        )
    threads = max(1, int(getattr(cfg, "evaluation_worker_threads", 2)))
    env_names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    old_env = {name: os.environ.get(name) for name in env_names}
    old_torch_threads = torch.get_num_threads()
    try:
        for name in env_names:
            os.environ[name] = str(threads)
        torch.set_num_threads(threads)
        yield
    finally:
        torch.set_num_threads(old_torch_threads)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

def _parse_evaluation_horizons(cfg: PSGAILConfig) -> list[int]:
    values: list[int] = []
    for raw in str(getattr(cfg, "evaluation_horizons_seconds", "1,5,10,20")).replace(";", ",").split(","):
        text = raw.strip()
        if not text:
            continue
        horizon = int(round(float(text)))
        if horizon > 0 and horizon not in values:
            values.append(horizon)
    return values or [1, 5, 10, 20]


def _evaluation_should_stop(
    cfg: PSGAILConfig,
    *,
    terminated: bool,
    truncated: bool,
) -> bool:
    """Apply the fixed-horizon policy used by matched evaluation.

    The strict pilot sets ``evaluation_terminate_when_all_controlled_crashed``
    to false so that a crash is recorded without erasing the requested
    post-crash horizon. Gym still reports ``terminated`` after that crash; in
    fixed-horizon mode evaluation therefore continues until the time-limit
    truncation. Truncation is always respected.
    """

    return bool(truncated) or (
        bool(terminated)
        and bool(getattr(cfg, "evaluation_terminate_when_all_controlled_crashed", True))
    )


def _evaluation_protocol_seed(cfg: PSGAILConfig) -> int:
    """Return the policy-seed-independent RNG seed for paired evaluation."""

    return int(getattr(cfg, "evaluation_scenario_seed", 20260716))


def _evaluation_scenarios(
    cfg: PSGAILConfig,
    *,
    split: str,
    episodes: int,
) -> list[tuple[str, int]]:
    if int(episodes) <= 0:
        return []
    _prebuilt_dir, valid_ids_by_episode, _traj_all_by_episode, episode_names = load_prebuilt_data(
        cfg.episode_root,
        cfg.scene,
        str(split),
        min_occupancy=0.8,
        cache=_EVAL_PREBUILT_CACHE,
    )
    scenarios: list[tuple[str, int]] = []
    for episode_name in sorted(str(name) for name in episode_names):
        for vehicle_id in sorted(int(value) for value in valid_ids_by_episode.get(episode_name, [])):
            scenarios.append((episode_name, vehicle_id))
    if not scenarios:
        raise RuntimeError(f"No evaluation scenarios found for split={split!r}.")
    rng = np.random.default_rng(
        _evaluation_protocol_seed(cfg)
        + (17_003 if str(split) == "val" else 31_337)
    )
    order = rng.permutation(len(scenarios))
    return [scenarios[int(idx)] for idx in order[: min(int(episodes), len(scenarios))]]

def _evaluation_episode_names(
    cfg: PSGAILConfig,
    *,
    split: str,
    episodes: int,
) -> list[str]:
    if int(episodes) <= 0:
        return []
    _prebuilt_dir, _valid_ids_by_episode, _traj_all_by_episode, episode_names = load_prebuilt_data(
        cfg.episode_root,
        cfg.scene,
        str(split),
        min_occupancy=0.8,
        cache=_EVAL_PREBUILT_CACHE,
    )
    names = sorted(str(name) for name in episode_names)
    if not names:
        raise RuntimeError(f"No evaluation episodes found for split={split!r}.")
    rng = np.random.default_rng(
        _evaluation_protocol_seed(cfg)
        + (19_019 if str(split) == "val" else 37_037)
    )
    order = rng.permutation(len(names))
    return [names[int(idx)] for idx in order[: min(int(episodes), len(names))]]

def _normalize_evaluation_vehicle_mode(cfg: PSGAILConfig, *, prefix: str) -> str:
    mode = str(getattr(cfg, f"{prefix}_vehicle_mode", "") or "").strip().lower()
    if not mode:
        mode = "all" if bool(getattr(cfg, f"{prefix}_control_all_vehicles", False)) else "single"
    if bool(getattr(cfg, f"{prefix}_control_all_vehicles", False)):
        mode = "all"
    if mode not in {"single", "training_count", "all"}:
        raise ValueError(
            f"{prefix}_vehicle_mode must be one of single, training_count, or all; got {mode!r}."
        )
    return mode

def _evaluation_training_count_episode_specs(
    cfg: PSGAILConfig,
    *,
    split: str,
    episodes: int,
) -> list[EpisodeSpec]:
    if int(episodes) <= 0:
        return []
    _prebuilt_dir, valid_ids_by_episode, _traj_all_by_episode, episode_names = load_prebuilt_data(
        cfg.episode_root,
        cfg.scene,
        str(split),
        min_occupancy=0.8,
        cache=_EVAL_PREBUILT_CACHE,
    )
    names = sorted(str(name) for name in episode_names)
    if not names:
        raise RuntimeError(f"No evaluation episodes found for split={split!r}.")
    rng = np.random.default_rng(
        _evaluation_protocol_seed(cfg)
        + (23_417 if str(split) == "val" else 41_713)
    )
    order = rng.permutation(len(names))
    specs: list[EpisodeSpec] = []
    for local_idx, name_idx in enumerate(order[: min(int(episodes), len(names))]):
        episode_name = names[int(name_idx)]
        valid_ids = np.asarray(valid_ids_by_episode.get(episode_name, []), dtype=np.int64)
        if valid_ids.size == 0:
            continue
        requested = resolve_num_ego_vehicles(
            getattr(cfg, "percentage_controlled_vehicles", 1.0),
            int(valid_ids.size),
        )
        selected_count = min(int(requested), int(valid_ids.size))
        selected = rng.choice(valid_ids, size=selected_count, replace=False)
        specs.append(
            (
                int(local_idx),
                str(episode_name),
                tuple(int(value) for value in selected),
            )
        )
    if not specs:
        raise RuntimeError(f"No training-count evaluation episodes found for split={split!r}.")
    return specs


def _apply_simulator_runtime_options(
    env_cfg: dict[str, object],
    cfg: PSGAILConfig,
) -> None:
    env_cfg["road_query_mode"] = str(getattr(cfg, "road_query_mode", "legacy"))
    env_cfg["collision_check_mode"] = str(getattr(cfg, "collision_check_mode", "legacy"))
    env_cfg["record_replay_diagnostics"] = bool(
        getattr(cfg, "record_replay_diagnostics", True)
    )
    env_cfg["sensor_road_edge_mode"] = str(
        getattr(cfg, "sensor_road_edge_mode", "per_vehicle")
    )
    env_cfg["reuse_pre_reset_spaces"] = bool(
        getattr(cfg, "reuse_pre_reset_spaces", False)
    )

def _make_matched_eval_env(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
    vehicle_id: int,
) -> gym.Env:
    from ..envs import observation_config

    register_ngsim_env()
    env_cfg = build_env_config(
        scene=cfg.scene,
        action_mode=str(cfg.action_mode),
        episode_root=cfg.episode_root,
        prebuilt_split=str(split),
        percentage_controlled_vehicles=1.0,
        control_all_vehicles=False,
        max_surrounding=cfg.max_surrounding,
        observation_config=observation_config(cfg),
        simulation_frequency=cfg.simulation_frequency,
        policy_frequency=cfg.policy_frequency,
        max_episode_steps=cfg.max_episode_steps,
        seed=None,
        simulation_period={"episode_name": str(episode_name)},
        ego_vehicle_id=[int(vehicle_id)],
        scene_dataset_collection_mode=False,
        allow_idm=cfg.allow_idm,
    )
    # The matched-trajectory evaluator needs the expert reference arrays that
    # NGSimEnv prepares during expert-test reset, but policy actions must still
    # control the vehicle during evaluation. The eval loop flips this back off
    # immediately after reset and before the first env.step(...).
    env_cfg["expert_test_mode"] = True
    env_cfg["truncate_to_trajectory_length"] = False
    env_cfg["complete_controlled_vehicles_at_road_end"] = False
    env_cfg["disable_controlled_vehicle_collisions"] = False
    env_cfg["terminate_when_all_controlled_crashed"] = bool(
        getattr(cfg, "evaluation_terminate_when_all_controlled_crashed", True)
    )
    env_cfg["allow_idm"] = bool(cfg.allow_idm)
    # Keep physical collision and off-road outcomes separate. Off-road is
    # reported independently and must not mutate the vehicle's crash flag.
    env_cfg["crash_controlled_vehicles_offroad"] = False
    configure_native_continuous_action_passthrough(
        env_cfg,
        action_mode=str(cfg.action_mode),
    )
    _apply_simulator_runtime_options(env_cfg, cfg)
    return gym.make(ENV_ID, config=env_cfg)

def _make_matched_eval_all_vehicle_env(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
) -> gym.Env:
    from ..envs import observation_config

    register_ngsim_env()
    env_cfg = build_env_config(
        scene=cfg.scene,
        action_mode=str(cfg.action_mode),
        episode_root=cfg.episode_root,
        prebuilt_split=str(split),
        percentage_controlled_vehicles=1.0,
        control_all_vehicles=True,
        max_surrounding=cfg.max_surrounding,
        observation_config=observation_config(cfg),
        simulation_frequency=cfg.simulation_frequency,
        policy_frequency=cfg.policy_frequency,
        max_episode_steps=cfg.max_episode_steps,
        seed=None,
        simulation_period={"episode_name": str(episode_name)},
        scene_dataset_collection_mode=False,
        allow_idm=cfg.allow_idm,
    )
    env_cfg["expert_test_mode"] = True
    env_cfg["truncate_to_trajectory_length"] = False
    env_cfg["complete_controlled_vehicles_at_road_end"] = False
    env_cfg["disable_controlled_vehicle_collisions"] = False
    env_cfg["terminate_when_all_controlled_crashed"] = bool(
        getattr(cfg, "evaluation_terminate_when_all_controlled_crashed", True)
    )
    env_cfg["allow_idm"] = bool(cfg.allow_idm)
    env_cfg["crash_controlled_vehicles_offroad"] = False
    configure_native_continuous_action_passthrough(
        env_cfg,
        action_mode=str(cfg.action_mode),
    )
    _apply_simulator_runtime_options(env_cfg, cfg)
    return gym.make(ENV_ID, config=env_cfg)

def _make_matched_eval_selected_vehicle_env(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
    vehicle_ids: tuple[int, ...],
) -> gym.Env:
    from ..envs import observation_config

    selected_ids = tuple(int(vehicle_id) for vehicle_id in vehicle_ids)
    if not selected_ids:
        raise ValueError("Selected-vehicle evaluation requires at least one vehicle id.")
    register_ngsim_env()
    env_cfg = build_env_config(
        scene=cfg.scene,
        action_mode=str(cfg.action_mode),
        episode_root=cfg.episode_root,
        prebuilt_split=str(split),
        percentage_controlled_vehicles=float(len(selected_ids)),
        control_all_vehicles=False,
        max_surrounding=cfg.max_surrounding,
        observation_config=observation_config(cfg),
        simulation_frequency=cfg.simulation_frequency,
        policy_frequency=cfg.policy_frequency,
        max_episode_steps=cfg.max_episode_steps,
        seed=None,
        simulation_period={"episode_name": str(episode_name)},
        ego_vehicle_id=list(selected_ids),
        scene_dataset_collection_mode=False,
        allow_idm=cfg.allow_idm,
    )
    env_cfg["expert_test_mode"] = True
    env_cfg["truncate_to_trajectory_length"] = False
    env_cfg["complete_controlled_vehicles_at_road_end"] = False
    env_cfg["disable_controlled_vehicle_collisions"] = False
    env_cfg["terminate_when_all_controlled_crashed"] = bool(
        getattr(cfg, "evaluation_terminate_when_all_controlled_crashed", True)
    )
    env_cfg["allow_idm"] = bool(cfg.allow_idm)
    env_cfg["crash_controlled_vehicles_offroad"] = False
    configure_native_continuous_action_passthrough(
        env_cfg,
        action_mode=str(cfg.action_mode),
    )
    _apply_simulator_runtime_options(env_cfg, cfg)
    return gym.make(ENV_ID, config=env_cfg)

def _validated_deterministic_continuous_actions(
    policy_out: torch.Tensor,
    cfg: PSGAILConfig,
) -> tuple[object, ...]:
    """Convert native tanh outputs to env actions without clamping them."""
    detached = policy_out.detach()
    if not bool(torch.isfinite(detached).all()):
        raise RuntimeError(
            "Deterministic policy evaluation produced non-finite actions."
        )
    outside = (detached < -1.0) | (detached > 1.0)
    if bool(outside.any()):
        first = tuple(
            int(value)
            for value in torch.nonzero(outside, as_tuple=False)[0].tolist()
        )
        raise RuntimeError(
            "Deterministic policy evaluation violated the normalized [-1, 1] "
            f"action contract at index {first}: {float(detached[first])}. "
            "Actions are never clamped."
        )
    actions_np = detached.cpu().numpy().astype(np.float32, copy=False).reshape(
        -1,
        int(cfg.continuous_action_dim),
    )
    return tuple(action.copy() for action in actions_np)


def _deterministic_policy_action_tuple(
    policy: nn.Module,
    env: gym.Env,
    obs: object,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    memory: torch.Tensor | None = None,
    return_memory: bool = False,
) -> tuple[object, ...] | tuple[tuple[object, ...], torch.Tensor | None]:
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    critic_obs_agents = central_critic_observations(env, cfg, obs_agents)
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
        critic_obs_tensor = torch.as_tensor(critic_obs_agents, dtype=torch.float32, device=device)
        if recurrent_policy_enabled(policy):
            policy_out, _values, new_memory = policy(
                obs_tensor,
                critic_obs_tensor,
                memory=memory,
                return_memory=True,
            )
            if new_memory is not None:
                new_memory = (
                    new_memory.unsqueeze(1)
                    if memory is None
                    else _shift_recurrent_memory(memory, new_memory)
                )
        else:
            policy_out, _values = policy(obs_tensor, critic_obs_tensor)
            new_memory = None
        if _is_continuous(cfg):
            actions = _validated_deterministic_continuous_actions(
                policy_out,
                cfg,
            )
            return (actions, new_memory) if return_memory else actions
        masks = discrete_action_masks_from_env(
            env,
            num_agents=len(obs_agents),
            num_actions=policy_action_dim(policy),
            enabled=bool(getattr(cfg, "enable_action_masking", True)),
        )
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
        logits = _masked_discrete_logits(policy_out, mask_tensor)
        actions = _actions_to_env_tuple(torch.argmax(logits, dim=-1), cfg)
        return (actions, new_memory) if return_memory else actions

def _lane_offset_for_position(env: gym.Env, position: np.ndarray, vehicle: object | None = None) -> float:
    position = np.asarray(position, dtype=np.float32).reshape(-1)[:2]
    lane = getattr(vehicle, "lane", None) if vehicle is not None else None
    if lane is None:
        road = getattr(env.unwrapped, "road", None)
        network = getattr(road, "network", None)
        if network is not None:
            try:
                lane = network.get_lane(network.get_closest_lane_index(position))
            except Exception:
                lane = None
    if lane is None:
        return float("nan")
    try:
        _longitudinal, lateral = lane.local_coordinates(position)
        return float(lateral)
    except Exception:
        return float("nan")

def _first_controlled_vehicle(env: gym.Env) -> object | None:
    controlled = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
    return controlled[0] if controlled else None


def _validated_normalized_acceleration(value: object, *, context: str) -> float:
    """Validate one normalized acceleration without silently saturating it."""
    acceleration = float(value)
    if not np.isfinite(acceleration):
        raise RuntimeError(f"{context} contains a non-finite acceleration.")
    if acceleration < -1.0 or acceleration > 1.0:
        raise RuntimeError(
            f"{context} violates the normalized [-1, 1] acceleration contract: "
            f"{acceleration}. Acceleration is never clipped during evaluation."
        )
    return acceleration


def _physical_accel_from_action(action_tuple: tuple[object, ...], cfg: PSGAILConfig) -> float:
    if not _is_continuous(cfg) or not action_tuple:
        return float("nan")
    action = np.asarray(action_tuple[0], dtype=np.float32).reshape(-1)
    if action.size < 1:
        return float("nan")
    acceleration = _validated_normalized_acceleration(
        action[0],
        context="Matched evaluation action",
    )
    return denormalize_acceleration(acceleration)

def _physical_accels_from_actions(action_tuple: tuple[object, ...], cfg: PSGAILConfig) -> np.ndarray:
    if not _is_continuous(cfg) or not action_tuple:
        return np.zeros((0,), dtype=np.float32)
    accels = []
    for index, action in enumerate(action_tuple):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < 1:
            continue
        acceleration = _validated_normalized_acceleration(
            action_arr[0],
            context=f"Matched evaluation action {index}",
        )
        accels.append(denormalize_acceleration(acceleration))
    return np.asarray(accels, dtype=np.float32)


def _audit_native_action_execution(
    env: gym.Env,
    info: dict[str, object],
    action_tuple: tuple[object, ...],
    pre_step_vehicles: list[tuple[object, bool, float]],
    cfg: PSGAILConfig,
) -> dict[str, float]:
    """Audit the native command path without treating physics as policy edits."""

    if not _is_continuous(cfg):
        return {}
    echoed = info.get("applied_actions")
    if not isinstance(echoed, (tuple, list)):
        raise RuntimeError(
            "Matched evaluation did not receive an applied_actions receipt."
        )
    if len(echoed) != len(action_tuple):
        raise RuntimeError(
            "Matched evaluation action echo count changed: "
            f"{len(echoed)} != {len(action_tuple)}."
        )
    if len(pre_step_vehicles) != len(action_tuple):
        raise RuntimeError(
            "Matched evaluation controlled-vehicle/action count changed: "
            f"{len(pre_step_vehicles)} != {len(action_tuple)}."
        )

    result = {
        "normalized_echo_count": 0.0,
        "normalized_echo_exact_count": 0.0,
        "post_step_action_count": 0.0,
        "post_step_action_exact_count": 0.0,
        "crash_physics_override_count": 0.0,
        "speed_bound_override_count": 0.0,
        "unexpected_override_count": 0.0,
        "maximum_post_step_action_abs_difference": 0.0,
    }
    for index, (requested_value, echoed_value, vehicle_state) in enumerate(
        zip(action_tuple, echoed, pre_step_vehicles, strict=True)
    ):
        requested = np.asarray(requested_value, dtype=np.float32).reshape(-1)
        echo = np.asarray(echoed_value)
        result["normalized_echo_count"] += 1.0
        if (
            echo.dtype == np.dtype(np.float32)
            and echo.shape == requested.shape
            and np.array_equal(echo, requested)
        ):
            result["normalized_echo_exact_count"] += 1.0
        else:
            raise RuntimeError(
                "The normalized policy action did not pass unchanged through "
                f"env.step for controlled action {index}."
            )
        if requested.shape != (2,):
            raise RuntimeError(
                "Native continuous evaluation requires exactly "
                "[acceleration, steering]."
            )
        expected = np.asarray(
            [
                denormalize_acceleration(
                    _validated_normalized_acceleration(
                        requested[0],
                        context=f"Matched evaluation action {index}",
                    )
                ),
                float(requested[1]) * float(MAX_STEER),
            ],
            dtype=np.float64,
        )
        vehicle, crashed_before_step, speed_before_step = vehicle_state
        action_state = getattr(vehicle, "action", None)
        if not isinstance(action_state, dict):
            raise RuntimeError(
                f"Controlled vehicle {index} exposes no physical action state."
            )
        realized = np.asarray(
            [
                float(action_state.get("acceleration", np.nan)),
                float(action_state.get("steering", np.nan)),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(realized)):
            raise RuntimeError(
                f"Controlled vehicle {index} has a non-finite action state."
            )
        difference = float(np.max(np.abs(realized - expected)))
        result["post_step_action_count"] += 1.0
        result["maximum_post_step_action_abs_difference"] = max(
            result["maximum_post_step_action_abs_difference"],
            difference,
        )
        if np.allclose(realized, expected, rtol=1.0e-7, atol=1.0e-7):
            result["post_step_action_exact_count"] += 1.0
            continue
        if crashed_before_step:
            result["crash_physics_override_count"] += 1.0
            continue
        minimum_speed = float(getattr(vehicle, "MIN_SPEED", -np.inf))
        maximum_speed = float(getattr(vehicle, "MAX_SPEED", np.inf))
        if (
            float(speed_before_step) < minimum_speed
            or float(speed_before_step) > maximum_speed
        ):
            result["speed_bound_override_count"] += 1.0
            continue
        result["unexpected_override_count"] += 1.0
        raise RuntimeError(
            "A controlled vehicle action changed without a documented native "
            "crash/speed-bound override: "
            f"index={index}, requested_physical={expected.tolist()}, "
            f"realized={realized.tolist()}."
        )
    return result


def _accumulate_action_execution_receipt(
    destination: dict[str, float],
    receipt: dict[str, float],
) -> None:
    for key, value in receipt.items():
        if key == "maximum_post_step_action_abs_difference":
            destination[key] = max(float(destination.get(key, 0.0)), float(value))
        else:
            destination[key] = float(destination.get(key, 0.0)) + float(value)


def _record_deterministic_continuous_actions(
    action_values: list[list[float]],
    action_tuple: tuple[object, ...],
) -> None:
    """Accumulate finite normalized actions for deterministic-policy audits."""
    for action in action_tuple:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        for dim, values in enumerate(action_values):
            if dim < action_arr.size and np.isfinite(action_arr[dim]):
                values.append(float(action_arr[dim]))

def _matched_eval_metrics(
    *,
    prefix: str,
    attempted_episodes: int,
    evaluated_episodes: int,
    skipped_missing_expert: int,
    skipped_bad_reference: int,
    skipped_empty_rollout: int,
    total_steps: int,
    collision_steps: int,
    offroad_steps: int,
    hard_brake_steps: int,
    episode_lengths: list[int],
    squared: dict[int, dict[str, list[float]]],
    final_squared: dict[str, list[float]],
    horizons: list[int],
    terminated_episodes: int = 0,
    truncated_episodes: int = 0,
    crashed_vehicle_episodes: int | None = None,
    offroad_vehicle_episodes: int | None = None,
    vehicles: int | None = None,
    vehicle_episodes: int | None = None,
    controlled_vehicle_counts: list[int] | None = None,
    requested_controlled_vehicle_counts: list[int] | None = None,
    vehicle_ids: set[int] | None = None,
    action_values: list[list[float]] | None = None,
    rollout_covered_vehicle_counts: dict[int, int] | None = None,
    background_idm_handover_vehicle_episodes: int = 0,
    action_execution_receipt: dict[str, float] | None = None,
    include_raw: bool = False,
) -> dict[str, float]:
    collision_duration_rate = float(collision_steps / total_steps) if total_steps else float("nan")
    offroad_duration_rate = float(offroad_steps / total_steps) if total_steps else float("nan")
    crash_agent_fraction = collision_duration_rate
    offroad_agent_fraction = offroad_duration_rate
    vehicle_denominator = (
        int(vehicle_episodes)
        if vehicle_episodes is not None
        else int(evaluated_episodes)
    )
    vehicle_crash_rate = (
        float(crashed_vehicle_episodes / vehicle_denominator)
        if crashed_vehicle_episodes is not None and vehicle_denominator > 0
        else float("nan")
    )
    vehicle_offroad_rate = (
        float(offroad_vehicle_episodes / vehicle_denominator)
        if offroad_vehicle_episodes is not None and vehicle_denominator > 0
        else float("nan")
    )
    metrics: dict[str, float] = {
        f"{prefix}/attempted_episodes": float(attempted_episodes),
        f"{prefix}/episodes": float(evaluated_episodes),
        f"{prefix}/skipped_missing_expert": float(skipped_missing_expert),
        f"{prefix}/skipped_bad_reference": float(skipped_bad_reference),
        f"{prefix}/skipped_empty_rollout": float(skipped_empty_rollout),
        f"{prefix}/evaluated_steps": float(total_steps),
        f"{prefix}/controlled_vehicle_agent_steps": float(total_steps),
        f"{prefix}/crash_agent_fraction": crash_agent_fraction,
        f"{prefix}/collision_duration_rate": collision_duration_rate,
        f"{prefix}/collision_agent_step_rate": collision_duration_rate,
        f"{prefix}/collision_rate": vehicle_crash_rate if np.isfinite(vehicle_crash_rate) else collision_duration_rate,
        f"{prefix}/vehicle_crash_rate": vehicle_crash_rate,
        f"{prefix}/crashed_controlled_vehicle_episodes": float(
            crashed_vehicle_episodes or 0
        ),
        f"{prefix}/offroad_agent_fraction": offroad_agent_fraction,
        f"{prefix}/offroad_duration_rate": offroad_duration_rate,
        f"{prefix}/vehicle_offroad_rate": vehicle_offroad_rate,
        f"{prefix}/hard_brake_rate": float(hard_brake_steps / total_steps) if total_steps else float("nan"),
        f"{prefix}/hard_brake_agent_step_rate": (
            float(hard_brake_steps / total_steps)
            if total_steps
            else float("nan")
        ),
        f"{prefix}/mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        f"{prefix}/terminated_episodes": float(terminated_episodes),
        f"{prefix}/truncated_episodes": float(truncated_episodes),
        f"{prefix}/background_idm_handover_count": float(
            background_idm_handover_vehicle_episodes
        ),
        f"{prefix}/mean_background_idm_handovers_per_episode": (
            float(background_idm_handover_vehicle_episodes)
            / float(evaluated_episodes)
            if evaluated_episodes
            else float("nan")
        ),
    }
    if vehicles is not None:
        metrics[f"{prefix}/vehicles"] = float(vehicles)
    if vehicle_episodes is not None:
        metrics[f"{prefix}/vehicle_episodes"] = float(vehicle_episodes)
        metrics[f"{prefix}/controlled_vehicle_rate_denominator"] = float(
            vehicle_episodes
        )
    if controlled_vehicle_counts is not None:
        metrics[f"{prefix}/mean_controlled_vehicles_per_episode"] = (
            float(np.mean(controlled_vehicle_counts)) if controlled_vehicle_counts else 0.0
        )
    if requested_controlled_vehicle_counts is not None and requested_controlled_vehicle_counts:
        metrics[f"{prefix}/mean_requested_controlled_vehicles_per_episode"] = float(
            np.mean(requested_controlled_vehicle_counts)
        )
    execution = action_execution_receipt or {}
    echo_count = float(execution.get("normalized_echo_count", 0.0))
    post_step_count = float(execution.get("post_step_action_count", 0.0))
    metrics[f"{prefix}/normalized_action_echo_exact_rate"] = (
        float(execution.get("normalized_echo_exact_count", 0.0)) / echo_count
        if echo_count
        else float("nan")
    )
    metrics[f"{prefix}/post_step_action_state_exact_rate"] = (
        float(execution.get("post_step_action_exact_count", 0.0))
        / post_step_count
        if post_step_count
        else float("nan")
    )
    metrics[f"{prefix}/crash_physics_action_override_count"] = float(
        execution.get("crash_physics_override_count", 0.0)
    )
    metrics[f"{prefix}/speed_bound_action_override_count"] = float(
        execution.get("speed_bound_override_count", 0.0)
    )
    metrics[f"{prefix}/unexpected_action_override_count"] = float(
        execution.get("unexpected_override_count", 0.0)
    )
    metrics[f"{prefix}/maximum_post_step_action_abs_difference"] = float(
        execution.get("maximum_post_step_action_abs_difference", 0.0)
    )
    for dim, values in enumerate(action_values or []):
        array = np.asarray(values, dtype=np.float64)
        metrics[f"{prefix}/deterministic_action_{dim}_mean"] = (
            float(np.mean(array)) if array.size else float("nan")
        )
        metrics[f"{prefix}/deterministic_action_{dim}_std"] = (
            float(np.std(array)) if array.size else float("nan")
        )
        metrics[f"{prefix}/deterministic_action_{dim}_count"] = float(array.size)
        if dim == 0:
            metrics[f"{prefix}/acceleration_action_mean"] = metrics[
                f"{prefix}/deterministic_action_{dim}_mean"
            ]
            metrics[f"{prefix}/acceleration_action_std"] = metrics[
                f"{prefix}/deterministic_action_{dim}_std"
            ]
        elif dim == 1:
            metrics[f"{prefix}/steering_action_mean"] = metrics[
                f"{prefix}/deterministic_action_{dim}_mean"
            ]
            metrics[f"{prefix}/steering_action_std"] = metrics[
                f"{prefix}/deterministic_action_{dim}_std"
            ]
    for horizon in horizons:
        for name, values in squared[horizon].items():
            metric_name = f"{prefix}/rmse_{name}_{horizon}s"
            metrics[metric_name] = float(np.sqrt(np.mean(values))) if values else float("nan")
        position_count = len(squared[horizon].get("position", ()))
        reference_coverage = (
            float(position_count / vehicle_denominator)
            if vehicle_denominator > 0
            else float("nan")
        )
        rollout_count = (
            None
            if rollout_covered_vehicle_counts is None
            else int(rollout_covered_vehicle_counts.get(horizon, 0))
        )
        rollout_coverage = (
            float(rollout_count / vehicle_denominator)
            if rollout_count is not None and vehicle_denominator > 0
            else float("nan")
        )
        # Backward-compatible ``horizon_coverage`` is the coverage of the
        # paired policy/reference RMSE sample.  Keep it, but expose its exact
        # meaning and the independent rollout-completion coverage so a short
        # reference cannot be mistaken for early policy termination.
        metrics[f"{prefix}/horizon_coverage_{horizon}s"] = reference_coverage
        metrics[
            f"{prefix}/reference_horizon_coverage_{horizon}s"
        ] = reference_coverage
        metrics[
            f"{prefix}/rollout_horizon_coverage_{horizon}s"
        ] = rollout_coverage
    for name, values in final_squared.items():
        metric_name = f"{prefix}/rmse_{name}_final"
        metrics[metric_name] = float(np.sqrt(np.mean(values))) if values else float("nan")
    if include_raw:
        metrics[f"__raw/{prefix}/collision_steps"] = float(collision_steps)
        metrics[f"__raw/{prefix}/offroad_steps"] = float(offroad_steps)
        metrics[f"__raw/{prefix}/hard_brake_steps"] = float(hard_brake_steps)
        metrics[f"__raw/{prefix}/crashed_vehicle_episodes"] = float(crashed_vehicle_episodes or 0)
        metrics[f"__raw/{prefix}/offroad_vehicle_episodes"] = float(offroad_vehicle_episodes or 0)
        metrics[
            f"__raw/{prefix}/background_idm_handover_vehicle_episodes"
        ] = float(background_idm_handover_vehicle_episodes)
        metrics[f"__raw/{prefix}/episode_lengths"] = tuple(int(value) for value in episode_lengths)
        metrics[f"__raw/{prefix}/controlled_vehicle_counts"] = tuple(
            int(value) for value in (controlled_vehicle_counts or [])
        )
        metrics[f"__raw/{prefix}/requested_controlled_vehicle_counts"] = tuple(
            int(value) for value in (requested_controlled_vehicle_counts or [])
        )
        metrics[f"__raw/{prefix}/vehicle_ids"] = tuple(
            sorted(int(value) for value in (vehicle_ids or set()))
        )
        for dim, values in enumerate(action_values or []):
            array = np.asarray(values, dtype=np.float64)
            metrics[f"__raw/{prefix}/action_sum_{dim}"] = float(array.sum())
            metrics[f"__raw/{prefix}/action_sumsq_{dim}"] = float(np.square(array).sum())
            metrics[f"__raw/{prefix}/action_count_{dim}"] = float(array.size)
        for name, value in execution.items():
            metrics[f"__raw/{prefix}/action_execution_{name}"] = float(value)
        for horizon in horizons:
            metrics[
                f"__raw/{prefix}/rollout_covered_vehicle_count_{horizon}s"
            ] = float(
                (rollout_covered_vehicle_counts or {}).get(horizon, 0)
            )
            for name, values in squared[horizon].items():
                arr = np.asarray(values, dtype=np.float64)
                metrics[f"__raw/{prefix}/sse_{name}_{horizon}s"] = float(arr.sum()) if arr.size else 0.0
                metrics[f"__raw/{prefix}/count_{name}_{horizon}s"] = float(arr.size)
        for name, values in final_squared.items():
            arr = np.asarray(values, dtype=np.float64)
            metrics[f"__raw/{prefix}/sse_{name}_final"] = float(arr.sum()) if arr.size else 0.0
            metrics[f"__raw/{prefix}/count_{name}_final"] = float(arr.size)
    return metrics

def _strip_internal_matched_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if not key.startswith("__raw/") and isinstance(value, (int, float, np.floating))
    }

def _combine_matched_eval_metric_dicts(
    parts: list[dict[str, object]],
    *,
    prefix: str,
    horizons: list[int],
) -> dict[str, float]:
    if not parts:
        return {}
    sum_keys = (
        "attempted_episodes",
        "episodes",
        "skipped_missing_expert",
        "skipped_bad_reference",
        "skipped_empty_rollout",
        "evaluated_steps",
        "terminated_episodes",
        "truncated_episodes",
        "vehicle_episodes",
    )
    metrics: dict[str, float] = {}
    for name in sum_keys:
        metrics[f"{prefix}/{name}"] = float(sum(float(part.get(f"{prefix}/{name}", 0.0)) for part in parts))
    collision_steps = float(sum(float(part.get(f"__raw/{prefix}/collision_steps", 0.0)) for part in parts))
    offroad_steps = float(sum(float(part.get(f"__raw/{prefix}/offroad_steps", 0.0)) for part in parts))
    hard_brake_steps = float(sum(float(part.get(f"__raw/{prefix}/hard_brake_steps", 0.0)) for part in parts))
    crashed_vehicle_episodes = float(
        sum(float(part.get(f"__raw/{prefix}/crashed_vehicle_episodes", 0.0)) for part in parts)
    )
    offroad_vehicle_episodes = float(
        sum(float(part.get(f"__raw/{prefix}/offroad_vehicle_episodes", 0.0)) for part in parts)
    )
    background_idm_handover_vehicle_episodes = float(
        sum(
            float(
                part.get(
                    f"__raw/{prefix}/background_idm_handover_vehicle_episodes",
                    0.0,
                )
            )
            for part in parts
        )
    )
    total_steps = float(metrics.get(f"{prefix}/evaluated_steps", 0.0))
    vehicle_episodes = float(metrics.get(f"{prefix}/vehicle_episodes", metrics.get(f"{prefix}/episodes", 0.0)))
    metrics[f"{prefix}/collision_duration_rate"] = collision_steps / total_steps if total_steps else float("nan")
    metrics[f"{prefix}/collision_agent_step_rate"] = metrics[
        f"{prefix}/collision_duration_rate"
    ]
    metrics[f"{prefix}/crash_agent_fraction"] = metrics[f"{prefix}/collision_duration_rate"]
    metrics[f"{prefix}/offroad_duration_rate"] = offroad_steps / total_steps if total_steps else float("nan")
    metrics[f"{prefix}/offroad_agent_fraction"] = metrics[f"{prefix}/offroad_duration_rate"]
    metrics[f"{prefix}/hard_brake_rate"] = hard_brake_steps / total_steps if total_steps else float("nan")
    metrics[f"{prefix}/hard_brake_agent_step_rate"] = metrics[
        f"{prefix}/hard_brake_rate"
    ]
    metrics[f"{prefix}/vehicle_crash_rate"] = (
        crashed_vehicle_episodes / vehicle_episodes if vehicle_episodes else float("nan")
    )
    metrics[f"{prefix}/vehicle_offroad_rate"] = (
        offroad_vehicle_episodes / vehicle_episodes if vehicle_episodes else float("nan")
    )
    metrics[f"{prefix}/collision_rate"] = (
        metrics[f"{prefix}/vehicle_crash_rate"]
        if np.isfinite(metrics[f"{prefix}/vehicle_crash_rate"])
        else metrics[f"{prefix}/collision_duration_rate"]
    )
    metrics[f"{prefix}/controlled_vehicle_agent_steps"] = total_steps
    metrics[f"{prefix}/controlled_vehicle_rate_denominator"] = vehicle_episodes
    metrics[f"{prefix}/crashed_controlled_vehicle_episodes"] = (
        crashed_vehicle_episodes
    )
    metrics[f"{prefix}/background_idm_handover_count"] = (
        background_idm_handover_vehicle_episodes
    )
    metrics[f"{prefix}/mean_background_idm_handovers_per_episode"] = (
        background_idm_handover_vehicle_episodes
        / float(metrics.get(f"{prefix}/episodes", 0.0))
        if metrics.get(f"{prefix}/episodes", 0.0)
        else float("nan")
    )
    for raw_name, metric_name in (
        ("policy_load_seconds", "eval_policy_load_seconds"),
        ("policy_forward_seconds", "eval_policy_forward_seconds"),
        ("env_step_seconds", "eval_env_step_seconds"),
        ("env_reset_seconds", "eval_env_reset_seconds"),
        ("env_cache_hits", "eval_env_cache_hits"),
        ("env_cache_misses", "eval_env_cache_misses"),
    ):
        metrics[f"{prefix}/{metric_name}"] = float(
            sum(float(part.get(f"__raw/{prefix}/{raw_name}", 0.0)) for part in parts)
        )
    episode_lengths: list[int] = []
    controlled_counts: list[int] = []
    requested_controlled_counts: list[int] = []
    vehicle_ids: set[int] = set()
    for part in parts:
        episode_lengths.extend(int(value) for value in part.get(f"__raw/{prefix}/episode_lengths", ()))
        controlled_counts.extend(int(value) for value in part.get(f"__raw/{prefix}/controlled_vehicle_counts", ()))
        requested_controlled_counts.extend(
            int(value) for value in part.get(f"__raw/{prefix}/requested_controlled_vehicle_counts", ())
        )
        vehicle_ids.update(int(value) for value in part.get(f"__raw/{prefix}/vehicle_ids", ()))
    metrics[f"{prefix}/mean_episode_length"] = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    if controlled_counts:
        metrics[f"{prefix}/mean_controlled_vehicles_per_episode"] = float(np.mean(controlled_counts))
    if requested_controlled_counts:
        metrics[f"{prefix}/mean_requested_controlled_vehicles_per_episode"] = float(
            np.mean(requested_controlled_counts)
        )
    metrics[f"{prefix}/vehicles"] = (
        float(len(vehicle_ids))
        if controlled_counts and vehicle_ids
        else metrics.get(f"{prefix}/episodes", 0.0)
    )
    execution_names = (
        "normalized_echo_count",
        "normalized_echo_exact_count",
        "post_step_action_count",
        "post_step_action_exact_count",
        "crash_physics_override_count",
        "speed_bound_override_count",
        "unexpected_override_count",
    )
    execution = {
        name: float(
            sum(
                float(
                    part.get(
                        f"__raw/{prefix}/action_execution_{name}",
                        0.0,
                    )
                )
                for part in parts
            )
        )
        for name in execution_names
    }
    execution["maximum_post_step_action_abs_difference"] = max(
        (
            float(
                part.get(
                    f"__raw/{prefix}/action_execution_"
                    "maximum_post_step_action_abs_difference",
                    0.0,
                )
            )
            for part in parts
        ),
        default=0.0,
    )
    echo_count = execution["normalized_echo_count"]
    post_step_count = execution["post_step_action_count"]
    metrics[f"{prefix}/normalized_action_echo_exact_rate"] = (
        execution["normalized_echo_exact_count"] / echo_count
        if echo_count
        else float("nan")
    )
    metrics[f"{prefix}/post_step_action_state_exact_rate"] = (
        execution["post_step_action_exact_count"] / post_step_count
        if post_step_count
        else float("nan")
    )
    metrics[f"{prefix}/crash_physics_action_override_count"] = execution[
        "crash_physics_override_count"
    ]
    metrics[f"{prefix}/speed_bound_action_override_count"] = execution[
        "speed_bound_override_count"
    ]
    metrics[f"{prefix}/unexpected_action_override_count"] = execution[
        "unexpected_override_count"
    ]
    metrics[f"{prefix}/maximum_post_step_action_abs_difference"] = execution[
        "maximum_post_step_action_abs_difference"
    ]
    action_dims = sorted(
        {
            int(key.rsplit("_", 1)[1])
            for part in parts
            for key in part
            if key.startswith(f"__raw/{prefix}/action_count_")
        }
    )
    for dim in action_dims:
        count = float(
            sum(float(part.get(f"__raw/{prefix}/action_count_{dim}", 0.0)) for part in parts)
        )
        total = float(
            sum(float(part.get(f"__raw/{prefix}/action_sum_{dim}", 0.0)) for part in parts)
        )
        total_squared = float(
            sum(float(part.get(f"__raw/{prefix}/action_sumsq_{dim}", 0.0)) for part in parts)
        )
        mean = total / count if count else float("nan")
        variance = max(0.0, total_squared / count - mean * mean) if count else float("nan")
        metrics[f"{prefix}/deterministic_action_{dim}_mean"] = mean
        metrics[f"{prefix}/deterministic_action_{dim}_std"] = (
            float(np.sqrt(variance)) if count else float("nan")
        )
        metrics[f"{prefix}/deterministic_action_{dim}_count"] = count
        semantic_name = "acceleration" if dim == 0 else ("steering" if dim == 1 else "")
        if semantic_name:
            metrics[f"{prefix}/{semantic_name}_action_mean"] = mean
            metrics[f"{prefix}/{semantic_name}_action_std"] = metrics[
                f"{prefix}/deterministic_action_{dim}_std"
            ]
    names = ("x", "y", "position", "speed", "lane_offset")
    for horizon in horizons:
        position_count = 0.0
        for name in names:
            sse = float(sum(float(part.get(f"__raw/{prefix}/sse_{name}_{horizon}s", 0.0)) for part in parts))
            count = float(sum(float(part.get(f"__raw/{prefix}/count_{name}_{horizon}s", 0.0)) for part in parts))
            metrics[f"{prefix}/rmse_{name}_{horizon}s"] = float(np.sqrt(sse / count)) if count else float("nan")
            if name == "position":
                position_count = count
        reference_coverage = (
            float(position_count / vehicle_episodes)
            if vehicle_episodes > 0.0
            else float("nan")
        )
        rollout_count = float(
            sum(
                float(
                    part.get(
                        f"__raw/{prefix}/rollout_covered_vehicle_count_{horizon}s",
                        0.0,
                    )
                )
                for part in parts
            )
        )
        rollout_coverage = (
            float(rollout_count / vehicle_episodes)
            if vehicle_episodes > 0.0
            else float("nan")
        )
        metrics[f"{prefix}/horizon_coverage_{horizon}s"] = reference_coverage
        metrics[
            f"{prefix}/reference_horizon_coverage_{horizon}s"
        ] = reference_coverage
        metrics[
            f"{prefix}/rollout_horizon_coverage_{horizon}s"
        ] = rollout_coverage
    for name in names:
        sse = float(sum(float(part.get(f"__raw/{prefix}/sse_{name}_final", 0.0)) for part in parts))
        count = float(sum(float(part.get(f"__raw/{prefix}/count_{name}_final", 0.0)) for part in parts))
        metrics[f"{prefix}/rmse_{name}_final"] = float(np.sqrt(sse / count)) if count else float("nan")
    return metrics

def _chunk_evenly(items: list[object], chunks: int) -> list[list[object]]:
    chunks = max(1, int(chunks))
    if not items:
        return []
    active = min(chunks, len(items))
    return [
        items[start::active]
        for start in range(active)
        if items[start::active]
    ]

def _policy_input_dim(policy: nn.Module) -> int:
    obs_dim = getattr(policy, "obs_dim", None)
    if obs_dim is not None:
        return int(obs_dim)
    encoder = getattr(policy, "encoder", None)
    first = encoder[0] if isinstance(encoder, nn.Sequential) and len(encoder) else None
    in_features = getattr(first, "in_features", None)
    if in_features is None:
        raise RuntimeError("Cannot infer policy observation dimension for parallel evaluation.")
    return int(in_features)

def _cpu_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}

def _configure_evaluation_worker_threads(cfg: PSGAILConfig) -> None:
    threads = max(1, int(getattr(cfg, "evaluation_worker_threads", 2)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)

def _eval_policy_cache_key(
    cfg: PSGAILConfig,
    policy_obs_dim: int,
    critic_obs_dim: int,
) -> tuple[object, ...]:
    return (
        str(cfg.policy_model),
        int(policy_obs_dim),
        int(critic_obs_dim),
        int(cfg.hidden_size),
        str(cfg.action_mode),
        int(cfg.continuous_action_dim),
        int(cfg.transformer_layers),
        int(cfg.transformer_heads),
        float(cfg.transformer_dropout),
        bool(getattr(cfg, "transformer_temporal_module", False)),
        int(getattr(cfg, "transformer_temporal_kernel_size", 5)),
        int(getattr(cfg, "transformer_temporal_layers", 1)),
        int(getattr(cfg, "transformer_memory_tokens", 8)),
        int(getattr(cfg, "transformer_memory_context_length", 32)),
        bool(getattr(cfg, "transformer_use_causal_attention", True)),
        bool(centralized_critic_enabled(cfg)),
        str(getattr(cfg, "central_critic_pooling", "flat")),
        int(getattr(cfg, "central_critic_max_vehicles", 64)),
        int(getattr(cfg, "central_critic_attention_heads", 4)),
    )

def _cached_eval_policy(
    cfg: PSGAILConfig,
    policy_state_dict: dict[str, torch.Tensor],
    policy_obs_dim: int,
    critic_obs_dim: int,
) -> nn.Module:
    key = _eval_policy_cache_key(cfg, policy_obs_dim, critic_obs_dim)
    policy = _EVAL_POLICY_CACHE.get(key)
    if policy is None:
        policy = _make_policy_from_state_dict(
            policy_state_dict,
            cfg,
            int(policy_obs_dim),
            int(critic_obs_dim),
            torch.device("cpu"),
        )
        _EVAL_POLICY_CACHE[key] = policy
    else:
        policy.load_state_dict(policy_state_dict)
        policy.eval()
    return policy

def _matched_eval_env_cache_key(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
    vehicle_id: int | None,
    vehicle_ids: tuple[int, ...] | None = None,
    all_vehicle: bool,
) -> tuple[object, ...]:
    selected_ids = None if vehicle_ids is None else tuple(int(value) for value in vehicle_ids)
    return (
        "matched_all" if bool(all_vehicle) else ("matched_multi" if selected_ids is not None else "matched_single"),
        str(cfg.scene),
        os.path.abspath(str(cfg.episode_root)),
        str(split),
        str(episode_name),
        selected_ids if selected_ids is not None else (None if vehicle_id is None else int(vehicle_id)),
        str(cfg.action_mode),
        str(cfg.max_surrounding),
        int(cfg.cells),
        float(cfg.maximum_range),
        int(cfg.simulation_frequency),
        int(cfg.policy_frequency),
        int(cfg.max_episode_steps),
        bool(cfg.enable_collision),
        bool(getattr(cfg, "evaluation_terminate_when_all_controlled_crashed", True)),
        bool(cfg.allow_idm),
        str(getattr(cfg, "road_query_mode", "legacy")),
        str(getattr(cfg, "collision_check_mode", "legacy")),
        bool(getattr(cfg, "record_replay_diagnostics", True)),
        str(getattr(cfg, "sensor_road_edge_mode", "per_vehicle")),
        bool(getattr(cfg, "reuse_pre_reset_spaces", False)),
    )

def _get_matched_eval_env(
    cfg: PSGAILConfig,
    *,
    split: str,
    episode_name: str,
    vehicle_id: int | None = None,
    vehicle_ids: tuple[int, ...] | None = None,
    all_vehicle: bool,
) -> tuple[gym.Env, bool]:
    global _EVAL_ENV_CACHE_HITS, _EVAL_ENV_CACHE_MISSES
    cache_enabled = bool(getattr(cfg, "evaluation_cache_envs", True))
    selected_ids = None if vehicle_ids is None else tuple(int(value) for value in vehicle_ids)
    if not cache_enabled:
        env = (
            _make_matched_eval_all_vehicle_env(cfg, split=split, episode_name=episode_name)
            if all_vehicle
            else (
                _make_matched_eval_selected_vehicle_env(
                    cfg,
                    split=split,
                    episode_name=episode_name,
                    vehicle_ids=selected_ids,
                )
                if selected_ids is not None
                else _make_matched_eval_env(
                    cfg,
                    split=split,
                    episode_name=episode_name,
                    vehicle_id=int(vehicle_id),
                )
            )
        )
        _EVAL_ENV_CACHE_MISSES += 1
        return env, False
    key = _matched_eval_env_cache_key(
        cfg,
        split=split,
        episode_name=episode_name,
        vehicle_id=vehicle_id,
        vehicle_ids=selected_ids,
        all_vehicle=all_vehicle,
    )
    env = _EVAL_ENV_CACHE.get(key)
    if env is not None:
        _EVAL_ENV_CACHE.move_to_end(key)
        _EVAL_ENV_CACHE_HITS += 1
        env.unwrapped.config["expert_test_mode"] = True
        return env, True
    env = (
        _make_matched_eval_all_vehicle_env(cfg, split=split, episode_name=episode_name)
        if all_vehicle
        else (
            _make_matched_eval_selected_vehicle_env(
                cfg,
                split=split,
                episode_name=episode_name,
                vehicle_ids=selected_ids,
            )
            if selected_ids is not None
            else _make_matched_eval_env(
                cfg,
                split=split,
                episode_name=episode_name,
                vehicle_id=int(vehicle_id),
            )
        )
    )
    _EVAL_ENV_CACHE[key] = env
    _EVAL_ENV_CACHE_MISSES += 1
    max_cached = max(0, int(getattr(cfg, "evaluation_max_cached_envs_per_worker", 4)))
    while max_cached > 0 and len(_EVAL_ENV_CACHE) > max_cached:
        _old_key, old_env = _EVAL_ENV_CACHE.popitem(last=False)
        old_env.close()
    return env, False


def evaluation_worker_cache_stats() -> dict[str, int]:
    return {
        "env_cache_size": len(_EVAL_ENV_CACHE),
        "env_cache_hits": int(_EVAL_ENV_CACHE_HITS),
        "env_cache_misses": int(_EVAL_ENV_CACHE_MISSES),
        "policy_cache_size": len(_EVAL_POLICY_CACHE),
        "prebuilt_split_cache_size": len(_EVAL_PREBUILT_CACHE),
    }


def clear_evaluation_worker_caches() -> None:
    global _EVAL_ENV_CACHE_HITS, _EVAL_ENV_CACHE_MISSES
    for env in _EVAL_ENV_CACHE.values():
        env.close()
    _EVAL_ENV_CACHE.clear()
    _EVAL_POLICY_CACHE.clear()
    _EVAL_PREBUILT_CACHE.clear()
    _EVAL_ENV_CACHE_HITS = 0
    _EVAL_ENV_CACHE_MISSES = 0

def _matched_eval_worker(
    cfg: PSGAILConfig,
    policy_state_dict: dict[str, torch.Tensor],
    policy_obs_dim: int,
    critic_obs_dim: int,
    split: str,
    prefix: str,
    worker_id: int,
    scenarios: list[tuple[int, str, int]] | None,
    episode_names: list[tuple[int, str]] | None,
    episode_specs: list[EpisodeSpec] | None,
) -> dict[str, object]:
    _configure_evaluation_worker_threads(cfg)
    worker_seed = (
        _evaluation_protocol_seed(cfg) + 700_000 + int(worker_id)
    )
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_cfg = replace(cfg, device="cpu", evaluation_num_workers=1)
    hit_start = _EVAL_ENV_CACHE_HITS
    miss_start = _EVAL_ENV_CACHE_MISSES
    started = time.perf_counter()
    policy = _cached_eval_policy(worker_cfg, policy_state_dict, int(policy_obs_dim), int(critic_obs_dim))
    policy_load_seconds = time.perf_counter() - started
    metrics = _evaluate_policy_matched_trajectories_impl(
        policy,
        worker_cfg,
        torch.device("cpu"),
        split=split,
        episodes=len(episode_specs or episode_names or scenarios or ()),
        prefix=prefix,
        scenarios=scenarios,
        episode_names=episode_names,
        episode_specs=episode_specs,
        include_raw=True,
    )
    metrics[f"__raw/{prefix}/policy_load_seconds"] = float(policy_load_seconds)
    metrics[f"__raw/{prefix}/env_cache_hits"] = float(_EVAL_ENV_CACHE_HITS - hit_start)
    metrics[f"__raw/{prefix}/env_cache_misses"] = float(_EVAL_ENV_CACHE_MISSES - miss_start)
    return metrics

def _evaluate_policy_matched_all_vehicle_episodes(
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    split: str,
    episodes: int,
    prefix: str,
    episode_names: list[str] | None = None,
    episode_specs: list[EpisodeSpec] | None = None,
    include_raw: bool = False,
) -> dict[str, float]:
    if episode_specs is not None:
        raw_episode_specs = list(episode_specs)
    else:
        raw_episode_names = (
            list(episode_names)
            if episode_names is not None
            else _evaluation_episode_names(cfg, split=split, episodes=int(episodes))
        )
        raw_episode_specs = [
            (int(item[0]), str(item[1]), None)
            if isinstance(item, tuple) and len(item) == 2
            else (int(local_idx), str(item), None)
            for local_idx, item in enumerate(raw_episode_names)
        ]
    if not raw_episode_specs:
        return {}
    indexed_episode_specs: list[EpisodeSpec] = []
    for local_idx, item in enumerate(raw_episode_specs):
        if isinstance(item, tuple) and len(item) == 3:
            vehicle_ids = None if item[2] is None else tuple(int(value) for value in item[2])
            indexed_episode_specs.append((int(item[0]), str(item[1]), vehicle_ids))
        else:
            episode_name = item
            indexed_episode_specs.append((int(local_idx), str(episode_name), None))
    horizons = _parse_evaluation_horizons(cfg)
    max_steps = max(1, int(max(horizons) * int(cfg.policy_frequency)))
    max_steps = min(max_steps, max(1, int(cfg.max_episode_steps)))
    squared: dict[int, dict[str, list[float]]] = {
        horizon: {"x": [], "y": [], "position": [], "speed": [], "lane_offset": []}
        for horizon in horizons
    }
    rollout_covered_vehicle_counts = {horizon: 0 for horizon in horizons}
    final_squared: dict[str, list[float]] = {
        "x": [], "y": [], "position": [], "speed": [], "lane_offset": []
    }
    collision_steps = 0
    offroad_steps = 0
    hard_brake_steps = 0
    total_agent_steps = 0
    skipped_missing_expert = 0
    skipped_bad_reference = 0
    skipped_empty_rollout = 0
    episode_lengths: list[int] = []
    controlled_vehicle_counts: list[int] = []
    requested_controlled_vehicle_counts: list[int] = []
    evaluated_episodes = 0
    vehicle_episodes = 0
    evaluated_vehicle_ids: set[int] = set()
    terminated_episodes = 0
    truncated_episodes = 0
    crashed_vehicle_episodes = 0
    offroad_vehicle_episodes = 0
    background_handover_keys: set[tuple[int, int]] = set()
    action_values: list[list[float]] = (
        [[] for _ in range(max(0, int(cfg.continuous_action_dim)))]
        if _is_continuous(cfg)
        else []
    )
    action_execution_receipt: dict[str, float] = {}
    eval_policy_seconds = 0.0
    eval_step_seconds = 0.0
    eval_reset_seconds = 0.0
    was_training = policy.training
    policy.eval()
    try:
        for episode_idx, episode_name, selected_vehicle_ids in indexed_episode_specs:
            if selected_vehicle_ids is not None:
                requested_controlled_vehicle_counts.append(int(len(selected_vehicle_ids)))
            env, env_cached = _get_matched_eval_env(
                cfg,
                split=split,
                episode_name=str(episode_name),
                vehicle_ids=selected_vehicle_ids,
                all_vehicle=selected_vehicle_ids is None,
            )
            try:
                reset_started = time.perf_counter()
                obs, _info = env.reset(
                    seed=_evaluation_protocol_seed(cfg)
                    + 100_000
                    + episode_idx
                )
                eval_reset_seconds += time.perf_counter() - reset_started
                env.unwrapped.config["expert_test_mode"] = False
                controlled = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
                controlled_vehicle_counts.append(len(controlled))
                expert_states = getattr(env.unwrapped, "_expert_state_by_vehicle_id", {})
                vehicle_data: dict[int, dict[str, object]] = {}
                for vehicle in controlled:
                    vehicle_id = int(getattr(vehicle, "vehicle_ID", len(vehicle_data)))
                    expert_state = expert_states.get(vehicle_id)
                    if expert_state is None:
                        skipped_missing_expert += 1
                        continue
                    ref_xy = np.asarray(expert_state.get("ref_xy", []), dtype=np.float32)
                    ref_v = np.asarray(expert_state.get("ref_v", []), dtype=np.float32).reshape(-1)
                    if ref_xy.ndim != 2 or ref_xy.shape[1] < 2 or ref_v.size == 0:
                        skipped_bad_reference += 1
                        continue
                    vehicle_data[vehicle_id] = {
                        "vehicle": vehicle,
                        "ref_xy": ref_xy,
                        "ref_v": ref_v,
                        "pred_xy": [],
                        "pred_speed": [],
                        "pred_lane_offset": [],
                        "expert_lane_offset": [],
                    }
                if not vehicle_data:
                    skipped_empty_rollout += 1
                    continue
                eval_memory = (
                    policy.initial_memory(len(controlled), device=device, dtype=torch.float32)
                    if recurrent_policy_enabled(policy)
                    else None
                )
                length = 0
                last_terminated = False
                last_truncated = False
                episode_crashed_vehicle_ids: set[int] = set()
                episode_offroad_vehicle_ids: set[int] = set()
                for _step in range(max_steps):
                    live_controlled = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
                    if not live_controlled:
                        break
                    for vehicle in live_controlled:
                        vehicle_id = int(getattr(vehicle, "vehicle_ID", -1))
                        data = vehicle_data.get(vehicle_id)
                        if data is None:
                            continue
                        position = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=np.float32).reshape(-1)[:2]
                        pred_xy = data["pred_xy"]
                        pred_speed = data["pred_speed"]
                        pred_lane_offset = data["pred_lane_offset"]
                        expert_lane_offset = data["expert_lane_offset"]
                        ref_xy = data["ref_xy"]
                        assert isinstance(pred_xy, list)
                        assert isinstance(pred_speed, list)
                        assert isinstance(pred_lane_offset, list)
                        assert isinstance(expert_lane_offset, list)
                        assert isinstance(ref_xy, np.ndarray)
                        pred_xy.append(position.copy())
                        pred_speed.append(float(getattr(vehicle, "speed", 0.0)))
                        pred_lane_offset.append(_lane_offset_for_position(env, position, vehicle))
                        expert_idx = min(len(pred_xy) - 1, ref_xy.shape[0] - 1)
                        expert_lane_offset.append(_lane_offset_for_position(env, ref_xy[expert_idx, :2]))

                    if recurrent_policy_enabled(policy):
                        policy_started = time.perf_counter()
                        action_tuple, eval_memory = _deterministic_policy_action_tuple(
                            policy,
                            env,
                            obs,
                            cfg,
                            device,
                            memory=eval_memory,
                            return_memory=True,
                        )
                        eval_policy_seconds += time.perf_counter() - policy_started
                    else:
                        policy_started = time.perf_counter()
                        action_tuple = _deterministic_policy_action_tuple(policy, env, obs, cfg, device)
                        eval_policy_seconds += time.perf_counter() - policy_started
                    _record_deterministic_continuous_actions(action_values, action_tuple)
                    accels = _physical_accels_from_actions(action_tuple, cfg)
                    hard_brake_steps += int(np.sum(accels < float(cfg.hard_brake_accel_threshold)))
                    pre_step_vehicles = [
                        (
                            vehicle,
                            bool(getattr(vehicle, "crashed", False)),
                            float(getattr(vehicle, "speed", 0.0)),
                        )
                        for vehicle in live_controlled
                    ]
                    step_started = time.perf_counter()
                    obs, _reward, terminated, truncated, info = env.step(action_tuple)
                    eval_step_seconds += time.perf_counter() - step_started
                    _accumulate_action_execution_receipt(
                        action_execution_receipt,
                        _audit_native_action_execution(
                            env,
                            info,
                            action_tuple,
                            pre_step_vehicles,
                            cfg,
                        ),
                    )
                    crash_flags = list(info.get("controlled_vehicle_crashes", []) or [])
                    offroad_flags = list(info.get("controlled_vehicle_offroad", []) or [])
                    for handover in list(
                        info.get("background_idm_handovers", []) or []
                    ):
                        if isinstance(handover, dict):
                            background_handover_keys.add(
                                (
                                    int(episode_idx),
                                    int(handover.get("vehicle_id", -1)),
                                )
                            )
                    info_vehicle_ids = list(info.get("controlled_vehicle_ids", []) or [])
                    collision_steps += int(sum(bool(flag) for flag in crash_flags))
                    offroad_steps += int(sum(bool(flag) for flag in offroad_flags))
                    for flag_index, flag in enumerate(crash_flags):
                        if bool(flag) and flag_index < len(info_vehicle_ids):
                            episode_crashed_vehicle_ids.add(int(info_vehicle_ids[flag_index]))
                    for flag_index, flag in enumerate(offroad_flags):
                        if bool(flag) and flag_index < len(info_vehicle_ids):
                            episode_offroad_vehicle_ids.add(int(info_vehicle_ids[flag_index]))
                    total_agent_steps += int(max(len(crash_flags), len(offroad_flags), len(live_controlled)))
                    length += 1
                    last_terminated = last_terminated or bool(terminated)
                    last_truncated = bool(truncated)
                    if _evaluation_should_stop(
                        cfg,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    ):
                        break
                if length <= 0:
                    skipped_empty_rollout += 1
                    continue
                evaluated_episodes += 1
                terminated_episodes += int(last_terminated)
                truncated_episodes += int(last_truncated)
                crashed_vehicle_episodes += len(episode_crashed_vehicle_ids & set(vehicle_data.keys()))
                offroad_vehicle_episodes += len(episode_offroad_vehicle_ids & set(vehicle_data.keys()))
                episode_lengths.append(length)
                for vehicle_id, data in vehicle_data.items():
                    pred_xy = data["pred_xy"]
                    pred_speed = data["pred_speed"]
                    pred_lane_offset = data["pred_lane_offset"]
                    expert_lane_offset = data["expert_lane_offset"]
                    ref_xy = data["ref_xy"]
                    ref_v = data["ref_v"]
                    assert isinstance(pred_xy, list)
                    assert isinstance(pred_speed, list)
                    assert isinstance(pred_lane_offset, list)
                    assert isinstance(expert_lane_offset, list)
                    assert isinstance(ref_xy, np.ndarray)
                    assert isinstance(ref_v, np.ndarray)
                    if not pred_xy:
                        skipped_empty_rollout += 1
                        continue
                    vehicle_episodes += 1
                    evaluated_vehicle_ids.add(int(vehicle_id))
                    for horizon in horizons:
                        required_steps = int(
                            horizon * int(cfg.policy_frequency)
                        )
                        if len(pred_xy) >= required_steps:
                            rollout_covered_vehicle_counts[horizon] += 1
                    final_idx = min(len(pred_xy), ref_xy.shape[0], ref_v.size) - 1
                    if final_idx >= 0:
                        dx = float(pred_xy[final_idx][0] - ref_xy[final_idx, 0])
                        dy = float(pred_xy[final_idx][1] - ref_xy[final_idx, 1])
                        ds = float(pred_speed[final_idx] - ref_v[final_idx])
                        dlat = float(pred_lane_offset[final_idx] - expert_lane_offset[final_idx])
                        final_squared["x"].append(dx * dx)
                        final_squared["y"].append(dy * dy)
                        final_squared["position"].append(dx * dx + dy * dy)
                        final_squared["speed"].append(ds * ds)
                        if np.isfinite(dlat):
                            final_squared["lane_offset"].append(dlat * dlat)
                    for horizon in horizons:
                        idx = int(horizon * int(cfg.policy_frequency)) - 1
                        if idx < 0 or idx >= len(pred_xy) or idx >= ref_xy.shape[0] or idx >= ref_v.size:
                            continue
                        dx = float(pred_xy[idx][0] - ref_xy[idx, 0])
                        dy = float(pred_xy[idx][1] - ref_xy[idx, 1])
                        ds = float(pred_speed[idx] - ref_v[idx])
                        dlat = float(pred_lane_offset[idx] - expert_lane_offset[idx])
                        squared[horizon]["x"].append(dx * dx)
                        squared[horizon]["y"].append(dy * dy)
                        squared[horizon]["position"].append(dx * dx + dy * dy)
                        squared[horizon]["speed"].append(ds * ds)
                        if np.isfinite(dlat):
                            squared[horizon]["lane_offset"].append(dlat * dlat)
            finally:
                if not env_cached:
                    env.close()
    finally:
        if was_training:
            policy.train()
    metrics = _matched_eval_metrics(
        prefix=prefix,
        attempted_episodes=len(indexed_episode_specs),
        evaluated_episodes=evaluated_episodes,
        skipped_missing_expert=skipped_missing_expert,
        skipped_bad_reference=skipped_bad_reference,
        skipped_empty_rollout=skipped_empty_rollout,
        total_steps=total_agent_steps,
        collision_steps=collision_steps,
        offroad_steps=offroad_steps,
        hard_brake_steps=hard_brake_steps,
        episode_lengths=episode_lengths,
        squared=squared,
        final_squared=final_squared,
        horizons=horizons,
        terminated_episodes=terminated_episodes,
        truncated_episodes=truncated_episodes,
        crashed_vehicle_episodes=crashed_vehicle_episodes,
        offroad_vehicle_episodes=offroad_vehicle_episodes,
        vehicles=len(evaluated_vehicle_ids),
        vehicle_episodes=vehicle_episodes,
        controlled_vehicle_counts=controlled_vehicle_counts,
        requested_controlled_vehicle_counts=requested_controlled_vehicle_counts,
        vehicle_ids=evaluated_vehicle_ids,
        background_idm_handover_vehicle_episodes=len(
            background_handover_keys
        ),
        action_values=action_values,
        rollout_covered_vehicle_counts=rollout_covered_vehicle_counts,
        include_raw=include_raw,
        action_execution_receipt=action_execution_receipt,
    )
    if include_raw:
        metrics[f"__raw/{prefix}/policy_forward_seconds"] = float(eval_policy_seconds)
        metrics[f"__raw/{prefix}/env_step_seconds"] = float(eval_step_seconds)
        metrics[f"__raw/{prefix}/env_reset_seconds"] = float(eval_reset_seconds)
    return metrics

def evaluate_policy_matched_trajectories(
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    split: str,
    episodes: int,
    prefix: str,
    evaluation_executor: ProcessPoolExecutor | None = None,
) -> dict[str, float]:
    requested_workers = max(1, int(getattr(cfg, "evaluation_num_workers", 1)))
    vehicle_mode = _normalize_evaluation_vehicle_mode(cfg, prefix=prefix)
    all_vehicle_mode = vehicle_mode == "all"
    training_count_mode = vehicle_mode == "training_count"
    if all_vehicle_mode:
        selected_episode_names = _evaluation_episode_names(cfg, split=split, episodes=int(episodes))
        indexed_episode_names = [(idx, name) for idx, name in enumerate(selected_episode_names)]
        indexed_scenarios = None
        indexed_episode_specs = None
        parallel_items: list[object] = indexed_episode_names
    elif training_count_mode:
        indexed_episode_specs = _evaluation_training_count_episode_specs(cfg, split=split, episodes=int(episodes))
        indexed_episode_names = None
        indexed_scenarios = None
        parallel_items = indexed_episode_specs
    else:
        selected_scenarios = _evaluation_scenarios(cfg, split=split, episodes=int(episodes))
        indexed_scenarios = [(idx, name, vehicle_id) for idx, (name, vehicle_id) in enumerate(selected_scenarios)]
        indexed_episode_names = None
        indexed_episode_specs = None
        parallel_items = indexed_scenarios
    active_workers = min(requested_workers, len(parallel_items))
    if active_workers <= 1 and evaluation_executor is None:
        with evaluation_thread_context(cfg):
            return _strip_internal_matched_metrics(_evaluate_policy_matched_trajectories_impl(
                policy,
                cfg,
                device,
                split=split,
                episodes=episodes,
                prefix=prefix,
                scenarios=indexed_scenarios,
                episode_names=indexed_episode_names,
                episode_specs=indexed_episode_specs,
                include_raw=True,
            ))

    policy_obs_dim = _policy_input_dim(policy)
    critic_obs_dim = int(getattr(policy, "critic_obs_dim", central_critic_observation_dim(policy_obs_dim, cfg)))
    policy_state_dict = _cpu_state_dict(policy)
    horizons = _parse_evaluation_horizons(cfg)
    chunks = _chunk_evenly(parallel_items, active_workers)
    print(
        f"[{prefix}] parallel evaluation workers={active_workers} "
        f"worker_threads={max(1, int(getattr(cfg, 'evaluation_worker_threads', 2)))} "
        f"mode={vehicle_mode} items={len(parallel_items)}",
        flush=True,
    )
    executor_context = (
        nullcontext(evaluation_executor)
        if evaluation_executor is not None
        else ProcessPoolExecutor(max_workers=active_workers, mp_context=mp.get_context("spawn"))
    )
    with executor_context as executor:
        futures = []
        for worker_id, chunk in enumerate(chunks):
            worker_scenarios = list(chunk) if vehicle_mode == "single" else None
            worker_episode_names = list(chunk) if all_vehicle_mode else None
            worker_episode_specs = list(chunk) if training_count_mode else None
            futures.append(
                executor.submit(
                    _matched_eval_worker,
                    cfg,
                    policy_state_dict,
                    policy_obs_dim,
                    critic_obs_dim,
                    split,
                    prefix,
                    worker_id,
                    worker_scenarios,
                    worker_episode_names,
                    worker_episode_specs,
                )
            )
        parts = [future.result() for future in futures]
    metrics = _combine_matched_eval_metric_dicts(parts, prefix=prefix, horizons=horizons)
    print(
        f"[{prefix}] eval_timing "
        f"policy_load={metrics.get(f'{prefix}/eval_policy_load_seconds', 0.0):.3f}s "
        f"reset={metrics.get(f'{prefix}/eval_env_reset_seconds', 0.0):.3f}s "
        f"policy={metrics.get(f'{prefix}/eval_policy_forward_seconds', 0.0):.3f}s "
        f"env_step={metrics.get(f'{prefix}/eval_env_step_seconds', 0.0):.3f}s "
        f"cache_hits={metrics.get(f'{prefix}/eval_env_cache_hits', 0.0):.0f} "
        f"cache_misses={metrics.get(f'{prefix}/eval_env_cache_misses', 0.0):.0f}",
        flush=True,
    )
    return metrics


def evaluate_expert_replay_collision_baseline(
    cfg: PSGAILConfig,
    *,
    split: str,
    episodes: int,
    prefix: str = "expert_replay",
) -> dict[str, object]:
    """Replay tracker actions with policy-evaluation collision physics enabled.

    This is a simulator baseline, not an independent road-geometry validation.
    Its purpose is to distinguish collisions already present in same-scenario
    expert replay from collisions introduced by a learned policy.
    """

    episode_names = _evaluation_episode_names(
        cfg,
        split=str(split),
        episodes=int(episodes),
    )
    horizons = _parse_evaluation_horizons(cfg)
    max_steps = min(
        max(1, int(max(horizons) * int(cfg.policy_frequency))),
        max(1, int(cfg.max_episode_steps)),
    )
    evaluated_episodes = 0
    vehicle_episodes = 0
    crashed_vehicle_episodes = 0
    offroad_vehicle_episodes = 0
    collision_agent_steps = 0
    offroad_agent_steps = 0
    controlled_agent_steps = 0
    fully_covered_vehicle_episodes = 0
    episode_records: list[dict[str, object]] = []

    for episode_index, episode_name in enumerate(episode_names):
        env, env_cached = _get_matched_eval_env(
            cfg,
            split=str(split),
            episode_name=str(episode_name),
            all_vehicle=True,
        )
        try:
            _obs, _info = env.reset(
                seed=_evaluation_protocol_seed(cfg)
                + 300_000
                + int(episode_index)
            )
            if not bool(env.unwrapped.config.get("expert_test_mode")):
                raise RuntimeError(
                    "Expert replay baseline lost expert_test_mode after reset."
                )
            controlled = list(
                getattr(env.unwrapped, "controlled_vehicles", ()) or ()
            )
            initial_ids = {
                int(getattr(vehicle, "vehicle_ID", -1))
                for vehicle in controlled
            }
            initial_ids.discard(-1)
            if not initial_ids:
                raise RuntimeError(
                    f"Expert replay spawned no controlled vehicles: {episode_name}"
                )
            episode_crashed_ids: set[int] = set()
            episode_offroad_ids: set[int] = set()
            final_live_ids = set(initial_ids)
            completed_steps = 0
            for _step in range(max_steps):
                live_controlled = list(
                    getattr(env.unwrapped, "controlled_vehicles", ()) or ()
                )
                dummy_action = tuple(
                    np.zeros(
                        (int(cfg.continuous_action_dim),),
                        dtype=np.float32,
                    )
                    for _vehicle in live_controlled
                )
                _obs, _reward, terminated, truncated, info = env.step(
                    dummy_action
                )
                info_vehicle_ids = [
                    int(value)
                    for value in list(
                        info.get("controlled_vehicle_ids", []) or []
                    )
                ]
                crash_flags = list(
                    info.get("controlled_vehicle_crashes", []) or []
                )
                offroad_flags = list(
                    info.get("controlled_vehicle_offroad", []) or []
                )
                for flag_index, flag in enumerate(crash_flags):
                    if bool(flag) and flag_index < len(info_vehicle_ids):
                        episode_crashed_ids.add(info_vehicle_ids[flag_index])
                for flag_index, flag in enumerate(offroad_flags):
                    if bool(flag) and flag_index < len(info_vehicle_ids):
                        episode_offroad_ids.add(info_vehicle_ids[flag_index])
                collision_agent_steps += int(
                    sum(bool(flag) for flag in crash_flags)
                )
                offroad_agent_steps += int(
                    sum(bool(flag) for flag in offroad_flags)
                )
                controlled_agent_steps += int(
                    max(
                        len(info_vehicle_ids),
                        len(crash_flags),
                        len(offroad_flags),
                    )
                )
                final_live_ids = set(info_vehicle_ids)
                completed_steps += 1
                if _evaluation_should_stop(
                    cfg,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                ):
                    break
            evaluated_episodes += 1
            vehicle_episodes += len(initial_ids)
            crashed_vehicle_episodes += len(
                episode_crashed_ids & initial_ids
            )
            offroad_vehicle_episodes += len(
                episode_offroad_ids & initial_ids
            )
            if completed_steps >= max_steps:
                fully_covered_vehicle_episodes += len(
                    final_live_ids & initial_ids
                )
            episode_records.append(
                {
                    "episode_name": str(episode_name),
                    "controlled_vehicle_count": len(initial_ids),
                    "completed_steps": completed_steps,
                    "crashed_vehicle_count": len(
                        episode_crashed_ids & initial_ids
                    ),
                    "offroad_vehicle_count": len(
                        episode_offroad_ids & initial_ids
                    ),
                }
            )
        finally:
            if not env_cached:
                env.close()

    vehicle_crash_rate = (
        float(crashed_vehicle_episodes / vehicle_episodes)
        if vehicle_episodes
        else float("nan")
    )
    vehicle_offroad_rate = (
        float(offroad_vehicle_episodes / vehicle_episodes)
        if vehicle_episodes
        else float("nan")
    )
    horizon_coverage = (
        float(fully_covered_vehicle_episodes / vehicle_episodes)
        if vehicle_episodes
        else float("nan")
    )
    return {
        "framework": "same_scenario_expert_replay_collision_baseline_v1",
        "scope": (
            "simulator/action replay baseline only; not independent surveyed "
            "road-geometry evidence"
        ),
        "split": str(split),
        "vehicle_mode": "all_successfully_spawned",
        "collision_physics_enabled": True,
        "collision_termination_enabled": False,
        "attempted_episodes": len(episode_names),
        "evaluated_episodes": evaluated_episodes,
        "vehicle_episodes": vehicle_episodes,
        "crashed_vehicle_episodes": crashed_vehicle_episodes,
        "offroad_vehicle_episodes": offroad_vehicle_episodes,
        f"{prefix}/vehicle_crash_rate": vehicle_crash_rate,
        f"{prefix}/vehicle_offroad_rate": vehicle_offroad_rate,
        f"{prefix}/collision_agent_step_rate": (
            float(collision_agent_steps / controlled_agent_steps)
            if controlled_agent_steps
            else float("nan")
        ),
        f"{prefix}/offroad_agent_step_rate": (
            float(offroad_agent_steps / controlled_agent_steps)
            if controlled_agent_steps
            else float("nan")
        ),
        f"{prefix}/horizon_coverage_{int(max(horizons))}s": (
            horizon_coverage
        ),
        "episodes": episode_records,
    }


def _first_expert_action_vector(info: dict[str, object]) -> np.ndarray | None:
    """Return the first expert action in normalized [acceleration, steering] order."""

    value = info.get("expert_action_continuous")
    if value is None:
        values = list(info.get("expert_action_continuous_all", []) or [])
        value = values[0] if values else None
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    return vector if vector.size else None


def _first_applied_action_vector(info: dict[str, object]) -> np.ndarray | None:
    """Return the first native action echoed by ``env.step``."""

    values = info.get("applied_actions")
    if values is not None:
        sequence = list(values or [])
        value = sequence[0] if sequence else None
    else:
        value = info.get("applied_action")
        if isinstance(value, tuple):
            value = value[0] if value else None
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    return vector if vector.size else None


def evaluate_expert_replay_matched_single_vehicle_floor(
    cfg: PSGAILConfig,
    *,
    split: str,
    episodes: int,
    prefix: str = "expert_floor",
    scenarios: (
        list[tuple[str, int]]
        | list[tuple[int, str, int]]
        | None
    ) = None,
) -> dict[str, object]:
    """Run tracker actions on the exact single-ego policy-evaluation scenarios.

    Unlike :func:`evaluate_expert_replay_collision_baseline`, this evaluator
    samples ``(episode_name, ego_vehicle_id)`` pairs with
    :func:`_evaluation_scenarios` and uses the policy evaluator's reset seed
    offset.  It therefore defines a pairable simulator feasibility floor.  It
    is still a route-conditioned synthetic tracker baseline and is not a
    human-driver or surveyed-road ground truth.
    """

    raw_scenarios = (
        list(scenarios)
        if scenarios is not None
        else _evaluation_scenarios(
            cfg,
            split=str(split),
            episodes=int(episodes),
        )
    )
    selected_scenarios: list[tuple[int, str, int]] = []
    for local_index, item in enumerate(raw_scenarios):
        if len(item) == 3:
            scenario_index, episode_name, vehicle_id = item
        else:
            episode_name, vehicle_id = item
            scenario_index = local_index
        selected_scenarios.append(
            (
                int(scenario_index),
                str(episode_name),
                int(vehicle_id),
            )
        )
    horizons = _parse_evaluation_horizons(cfg)
    max_steps = min(
        max(1, int(max(horizons) * int(cfg.policy_frequency))),
        max(1, int(cfg.max_episode_steps)),
    )
    evaluated_episodes = 0
    crashed_vehicle_episodes = 0
    offroad_vehicle_episodes = 0
    fully_covered_vehicle_episodes = 0
    collision_steps = 0
    offroad_steps = 0
    action_echo_steps = 0
    action_echo_max_abs_error = 0.0
    action_finite = True
    action_range_valid = True
    episode_records: list[dict[str, object]] = []

    for scenario_index, episode_name, vehicle_id in selected_scenarios:
        env, env_cached = _get_matched_eval_env(
            cfg,
            split=str(split),
            episode_name=str(episode_name),
            vehicle_id=int(vehicle_id),
            all_vehicle=False,
        )
        reset_seed = (
            _evaluation_protocol_seed(cfg)
            + 100_000
            + int(scenario_index)
        )
        try:
            _obs, _info = env.reset(seed=reset_seed)
            if not bool(env.unwrapped.config.get("expert_test_mode")):
                raise RuntimeError(
                    "Matched expert floor lost expert_test_mode after reset."
                )
            controlled = list(
                getattr(env.unwrapped, "controlled_vehicles", ()) or ()
            )
            controlled_ids = [
                int(getattr(vehicle, "vehicle_ID", -1))
                for vehicle in controlled
            ]
            if controlled_ids != [int(vehicle_id)]:
                raise RuntimeError(
                    "Matched expert floor did not spawn exactly the requested "
                    f"single ego: requested={vehicle_id}, spawned={controlled_ids}."
                )

            first_collision_step: int | None = None
            first_offroad_step: int | None = None
            first_collision_partner: dict[str, object] | None = None
            handovers: dict[int, dict[str, object]] = {}
            completed_steps = 0
            encountered_terminated = False
            last_truncated = False
            for step_index in range(max_steps):
                live_controlled = list(
                    getattr(env.unwrapped, "controlled_vehicles", ()) or ()
                )
                if not live_controlled:
                    break
                dummy_action = tuple(
                    np.zeros(
                        (int(cfg.continuous_action_dim),),
                        dtype=np.float32,
                    )
                    for _vehicle in live_controlled
                )
                _obs, _reward, terminated, truncated, info = env.step(
                    dummy_action
                )
                completed_steps += 1
                encountered_terminated = (
                    encountered_terminated or bool(terminated)
                )
                last_truncated = bool(truncated)

                expert_action = _first_expert_action_vector(info)
                applied_action = _first_applied_action_vector(info)
                if expert_action is None or applied_action is None:
                    action_finite = False
                elif expert_action.shape != applied_action.shape:
                    action_finite = False
                else:
                    action_echo_steps += 1
                    action_finite = bool(
                        action_finite
                        and np.isfinite(expert_action).all()
                        and np.isfinite(applied_action).all()
                    )
                    action_range_valid = bool(
                        action_range_valid
                        and np.all(expert_action >= -1.0)
                        and np.all(expert_action <= 1.0)
                    )
                    if np.isfinite(expert_action).all() and np.isfinite(
                        applied_action
                    ).all():
                        action_echo_max_abs_error = max(
                            action_echo_max_abs_error,
                            float(
                                np.max(
                                    np.abs(expert_action - applied_action)
                                )
                            ),
                        )

                crash_flags = [
                    bool(value)
                    for value in list(
                        info.get("controlled_vehicle_crashes", []) or []
                    )
                ]
                offroad_flags = [
                    bool(value)
                    for value in list(
                        info.get("controlled_vehicle_offroad", []) or []
                    )
                ]
                collision_steps += int(any(crash_flags))
                offroad_steps += int(any(offroad_flags))
                if first_collision_step is None and any(crash_flags):
                    first_collision_step = int(step_index)
                    partner_records = list(
                        info.get(
                            "controlled_vehicle_collision_partners",
                            [],
                        )
                        or []
                    )
                    matching = [
                        value
                        for value in partner_records
                        if isinstance(value, dict)
                        and int(value.get("vehicle_id", -1))
                        == int(vehicle_id)
                    ]
                    if matching:
                        first_collision_partner = dict(matching[0])
                if first_offroad_step is None and any(offroad_flags):
                    first_offroad_step = int(step_index)

                for handover in list(
                    info.get("background_idm_handovers", []) or []
                ):
                    if not isinstance(handover, dict):
                        continue
                    handover_vehicle_id = int(
                        handover.get("vehicle_id", -1)
                    )
                    if handover_vehicle_id in handovers:
                        continue
                    handovers[handover_vehicle_id] = {
                        "vehicle_id": handover_vehicle_id,
                        "reported_simulation_step": int(
                            handover.get("handover_step", -1)
                        ),
                        "first_observed_policy_step": int(step_index),
                        "reason": str(
                            handover.get("reason", "unspecified")
                        ),
                    }
                if _evaluation_should_stop(
                    cfg,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                ):
                    break

            evaluated_episodes += 1
            crashed_vehicle_episodes += int(
                first_collision_step is not None
            )
            offroad_vehicle_episodes += int(first_offroad_step is not None)
            fully_covered_vehicle_episodes += int(
                completed_steps >= max_steps
            )
            first_handover_step = min(
                (
                    int(value["first_observed_policy_step"])
                    for value in handovers.values()
                ),
                default=None,
            )
            episode_records.append(
                {
                    "scenario_index": int(scenario_index),
                    "episode_name": str(episode_name),
                    "ego_vehicle_id": int(vehicle_id),
                    "reset_seed": int(reset_seed),
                    "completed_steps": int(completed_steps),
                    "encountered_terminated": bool(
                        encountered_terminated
                    ),
                    "last_truncated": bool(last_truncated),
                    "first_collision_step": first_collision_step,
                    "first_collision_time_seconds": (
                        None
                        if first_collision_step is None
                        else float(
                            (first_collision_step + 1)
                            / int(cfg.policy_frequency)
                        )
                    ),
                    "first_collision_partner": first_collision_partner,
                    "first_offroad_step": first_offroad_step,
                    "first_offroad_time_seconds": (
                        None
                        if first_offroad_step is None
                        else float(
                            (first_offroad_step + 1)
                            / int(cfg.policy_frequency)
                        )
                    ),
                    "first_background_handover_policy_step": (
                        first_handover_step
                    ),
                    "background_handover_precedes_collision": (
                        None
                        if first_collision_step is None
                        else (
                            first_handover_step is not None
                            and first_handover_step
                            <= first_collision_step
                        )
                    ),
                    "background_handovers": sorted(
                        handovers.values(),
                        key=lambda value: (
                            int(value["first_observed_policy_step"]),
                            int(value["vehicle_id"]),
                        ),
                    ),
                }
            )
        finally:
            if not env_cached:
                env.close()

    denominator = float(evaluated_episodes)
    collision_records = [
        record
        for record in episode_records
        if record["first_collision_step"] is not None
    ]
    collision_partner_complete = sum(
        int(
            isinstance(record["first_collision_partner"], dict)
            and record["first_collision_partner"].get("partner_type")
            is not None
            and record["first_collision_partner"].get("provenance")
            in {
                "physics_current_intersection",
                "physics_swept_intersection",
            }
        )
        for record in collision_records
    )
    return {
        "framework": "matched_single_ego_expert_floor_v3",
        "scope": (
            "route-conditioned synthetic tracker feasibility floor under the "
            "single-ego policy evaluator; not human-driver cloning evidence"
        ),
        "split": str(split),
        "vehicle_mode": "single_requested_ego",
        "scenario_selection": "shared_policy_evaluation_scenarios",
        "reset_seed_contract": "evaluation_protocol_seed_plus_100000_plus_scenario_index",
        "collision_physics_enabled": bool(cfg.enable_collision),
        "collision_termination_enabled": bool(
            getattr(
                cfg,
                "evaluation_terminate_when_all_controlled_crashed",
                True,
            )
        ),
        "attempted_episodes": len(selected_scenarios),
        "evaluated_episodes": evaluated_episodes,
        "crashed_vehicle_episodes": crashed_vehicle_episodes,
        "initial_collision_episodes": sum(
            int(record["first_collision_step"] == 0)
            for record in collision_records
        ),
        "collision_partner_complete_episodes": int(
            collision_partner_complete
        ),
        "collision_partner_completeness_rate": (
            float(collision_partner_complete / len(collision_records))
            if collision_records
            else 1.0
        ),
        "collision_after_or_at_background_handover_episodes": sum(
            int(
                record["background_handover_precedes_collision"]
                is True
            )
            for record in collision_records
        ),
        "collision_before_background_handover_or_without_handover_episodes": sum(
            int(
                record["background_handover_precedes_collision"]
                is False
            )
            for record in collision_records
        ),
        "offroad_vehicle_episodes": offroad_vehicle_episodes,
        f"{prefix}/vehicle_crash_rate": (
            float(crashed_vehicle_episodes / denominator)
            if denominator
            else float("nan")
        ),
        f"{prefix}/vehicle_offroad_rate": (
            float(offroad_vehicle_episodes / denominator)
            if denominator
            else float("nan")
        ),
        f"{prefix}/collision_step_rate": (
            float(collision_steps / max(1, sum(
                int(record["completed_steps"])
                for record in episode_records
            )))
            if denominator
            else float("nan")
        ),
        f"{prefix}/offroad_step_rate": (
            float(offroad_steps / max(1, sum(
                int(record["completed_steps"])
                for record in episode_records
            )))
            if denominator
            else float("nan")
        ),
        f"{prefix}/horizon_coverage_{int(max(horizons))}s": (
            float(fully_covered_vehicle_episodes / denominator)
            if denominator
            else float("nan")
        ),
        "action_execution_receipt": {
            "echo_steps": int(action_echo_steps),
            "all_finite": bool(action_finite),
            "normalized_range_valid": bool(action_range_valid),
            "max_abs_expert_applied_error": float(
                action_echo_max_abs_error
            ),
            "exact_echo": bool(
                action_finite
                and action_echo_steps > 0
                and action_echo_max_abs_error == 0.0
            ),
        },
        "episodes": episode_records,
    }


def _evaluate_policy_matched_trajectories_impl(
    policy: nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    split: str,
    episodes: int,
    prefix: str,
    scenarios: list[tuple[str, int]] | None = None,
    episode_names: list[str] | None = None,
    episode_specs: list[EpisodeSpec] | None = None,
    include_raw: bool = False,
    include_cases: bool = False,
) -> dict[str, object]:
    if _normalize_evaluation_vehicle_mode(cfg, prefix=prefix) in {"all", "training_count"}:
        return _evaluate_policy_matched_all_vehicle_episodes(
            policy,
            cfg,
            device,
            split=split,
            episodes=int(episodes),
            prefix=prefix,
            episode_names=episode_names,
            episode_specs=episode_specs,
            include_raw=include_raw,
        )
    raw_scenarios = (
        list(scenarios)
        if scenarios is not None
        else _evaluation_scenarios(cfg, split=split, episodes=int(episodes))
    )
    if not raw_scenarios:
        return {}
    indexed_scenarios: list[tuple[int, str, int]] = []
    for local_idx, item in enumerate(raw_scenarios):
        if isinstance(item, tuple) and len(item) == 3:
            indexed_scenarios.append((int(item[0]), str(item[1]), int(item[2])))
        else:
            episode_name, vehicle_id = item
            indexed_scenarios.append((int(local_idx), str(episode_name), int(vehicle_id)))
    horizons = _parse_evaluation_horizons(cfg)
    max_steps = max(1, int(max(horizons) * int(cfg.policy_frequency)))
    max_steps = min(max_steps, max(1, int(cfg.max_episode_steps)))
    squared: dict[int, dict[str, list[float]]] = {
        horizon: {"x": [], "y": [], "position": [], "speed": [], "lane_offset": []}
        for horizon in horizons
    }
    rollout_covered_vehicle_counts = {horizon: 0 for horizon in horizons}
    final_squared: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "position": [],
        "speed": [],
        "lane_offset": [],
    }
    collision_steps = 0
    offroad_steps = 0
    hard_brake_steps = 0
    total_steps = 0
    skipped_missing_expert = 0
    skipped_bad_reference = 0
    skipped_empty_rollout = 0
    episode_lengths: list[int] = []
    evaluated_episodes = 0
    evaluated_vehicle_ids: set[int] = set()
    terminated_episodes = 0
    truncated_episodes = 0
    crashed_vehicle_episodes = 0
    offroad_vehicle_episodes = 0
    background_handover_keys: set[tuple[int, int]] = set()
    episode_records: list[dict[str, object]] = []
    action_values: list[list[float]] = (
        [[] for _ in range(max(0, int(cfg.continuous_action_dim)))]
        if _is_continuous(cfg)
        else []
    )
    action_execution_receipt: dict[str, float] = {}
    eval_policy_seconds = 0.0
    eval_step_seconds = 0.0
    eval_reset_seconds = 0.0
    was_training = policy.training
    policy.eval()
    try:
        for scenario_idx, episode_name, vehicle_id in indexed_scenarios:
            env, env_cached = _get_matched_eval_env(
                cfg,
                split=split,
                episode_name=episode_name,
                vehicle_id=int(vehicle_id),
                all_vehicle=False,
            )
            try:
                reset_started = time.perf_counter()
                obs, _info = env.reset(
                    seed=_evaluation_protocol_seed(cfg)
                    + 100_000
                    + scenario_idx
                )
                eval_reset_seconds += time.perf_counter() - reset_started
                env.unwrapped.config["expert_test_mode"] = False
                expert_state = getattr(env.unwrapped, "_expert_state_by_vehicle_id", {}).get(int(vehicle_id))
                if expert_state is None:
                    skipped_missing_expert += 1
                    continue
                ref_xy = np.asarray(expert_state.get("ref_xy", []), dtype=np.float32)
                ref_v = np.asarray(expert_state.get("ref_v", []), dtype=np.float32).reshape(-1)
                if ref_xy.ndim != 2 or ref_xy.shape[1] < 2 or ref_v.size == 0:
                    skipped_bad_reference += 1
                    continue
                pred_xy: list[np.ndarray] = []
                pred_speed: list[float] = []
                pred_lane_offset: list[float] = []
                expert_lane_offset: list[float] = []
                length = 0
                episode_crashed = False
                episode_offroad = False
                first_collision_step: int | None = None
                first_offroad_step: int | None = None
                first_background_handover_step: int | None = None
                encountered_terminated = False
                last_truncated = False
                eval_memory = (
                    policy.initial_memory(1, device=device, dtype=torch.float32)
                    if recurrent_policy_enabled(policy)
                    else None
                )
                for _step in range(max_steps):
                    vehicle = _first_controlled_vehicle(env)
                    if vehicle is None:
                        break
                    position = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=np.float32).reshape(-1)[:2]
                    pred_xy.append(position.copy())
                    pred_speed.append(float(getattr(vehicle, "speed", 0.0)))
                    pred_lane_offset.append(_lane_offset_for_position(env, position, vehicle))
                    expert_idx = min(length, ref_xy.shape[0] - 1)
                    expert_lane_offset.append(_lane_offset_for_position(env, ref_xy[expert_idx, :2]))
                    if recurrent_policy_enabled(policy):
                        policy_started = time.perf_counter()
                        action_tuple, eval_memory = _deterministic_policy_action_tuple(
                            policy,
                            env,
                            obs,
                            cfg,
                            device,
                            memory=eval_memory,
                            return_memory=True,
                        )
                        eval_policy_seconds += time.perf_counter() - policy_started
                    else:
                        policy_started = time.perf_counter()
                        action_tuple = _deterministic_policy_action_tuple(policy, env, obs, cfg, device)
                        eval_policy_seconds += time.perf_counter() - policy_started
                    _record_deterministic_continuous_actions(action_values, action_tuple)
                    accel = _physical_accel_from_action(action_tuple, cfg)
                    if np.isfinite(accel) and accel < float(cfg.hard_brake_accel_threshold):
                        hard_brake_steps += 1
                    pre_step_vehicles = [
                        (
                            vehicle,
                            bool(getattr(vehicle, "crashed", False)),
                            float(getattr(vehicle, "speed", 0.0)),
                        )
                    ]
                    step_started = time.perf_counter()
                    obs, _reward, terminated, truncated, info = env.step(action_tuple)
                    eval_step_seconds += time.perf_counter() - step_started
                    _accumulate_action_execution_receipt(
                        action_execution_receipt,
                        _audit_native_action_execution(
                            env,
                            info,
                            action_tuple,
                            pre_step_vehicles,
                            cfg,
                        ),
                    )
                    crash_flags = list(info.get("controlled_vehicle_crashes", []) or [])
                    offroad_flags = list(info.get("controlled_vehicle_offroad", []) or [])
                    for handover in list(
                        info.get("background_idm_handovers", []) or []
                    ):
                        if isinstance(handover, dict):
                            if first_background_handover_step is None:
                                first_background_handover_step = int(_step)
                            background_handover_keys.add(
                                (
                                    int(scenario_idx),
                                    int(handover.get("vehicle_id", -1)),
                                )
                            )
                    episode_crashed = episode_crashed or any(bool(flag) for flag in crash_flags)
                    episode_offroad = episode_offroad or any(bool(flag) for flag in offroad_flags)
                    if first_collision_step is None and any(
                        bool(flag) for flag in crash_flags
                    ):
                        first_collision_step = int(_step)
                    if first_offroad_step is None and any(
                        bool(flag) for flag in offroad_flags
                    ):
                        first_offroad_step = int(_step)
                    collision_steps += int(any(bool(flag) for flag in crash_flags))
                    offroad_steps += int(any(bool(flag) for flag in offroad_flags))
                    total_steps += 1
                    length += 1
                    encountered_terminated = encountered_terminated or bool(terminated)
                    last_truncated = bool(truncated)
                    if _evaluation_should_stop(
                        cfg,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    ):
                        break
                if length <= 0:
                    skipped_empty_rollout += 1
                    continue
                evaluated_episodes += 1
                evaluated_vehicle_ids.add(int(vehicle_id))
                terminated_episodes += int(encountered_terminated)
                truncated_episodes += int(last_truncated)
                crashed_vehicle_episodes += int(episode_crashed)
                offroad_vehicle_episodes += int(episode_offroad)
                episode_lengths.append(length)
                if include_cases:
                    episode_records.append(
                        {
                            "scenario_index": int(scenario_idx),
                            "episode_name": str(episode_name),
                            "ego_vehicle_id": int(vehicle_id),
                            "reset_seed": int(
                                _evaluation_protocol_seed(cfg)
                                + 100_000
                                + scenario_idx
                            ),
                            "completed_steps": int(length),
                            "first_collision_step": first_collision_step,
                            "first_offroad_step": first_offroad_step,
                            "first_background_handover_policy_step": (
                                first_background_handover_step
                            ),
                        }
                    )
                for horizon in horizons:
                    required_steps = int(horizon * int(cfg.policy_frequency))
                    if len(pred_xy) >= required_steps:
                        rollout_covered_vehicle_counts[horizon] += 1
                final_idx = min(len(pred_xy), ref_xy.shape[0], ref_v.size) - 1
                if final_idx >= 0:
                    dx = float(pred_xy[final_idx][0] - ref_xy[final_idx, 0])
                    dy = float(pred_xy[final_idx][1] - ref_xy[final_idx, 1])
                    ds = float(pred_speed[final_idx] - ref_v[final_idx])
                    dlat = float(pred_lane_offset[final_idx] - expert_lane_offset[final_idx])
                    final_squared["x"].append(dx * dx)
                    final_squared["y"].append(dy * dy)
                    final_squared["position"].append(dx * dx + dy * dy)
                    final_squared["speed"].append(ds * ds)
                    if np.isfinite(dlat):
                        final_squared["lane_offset"].append(dlat * dlat)
                for horizon in horizons:
                    idx = int(horizon * int(cfg.policy_frequency)) - 1
                    if idx < 0 or idx >= len(pred_xy) or idx >= ref_xy.shape[0] or idx >= ref_v.size:
                        continue
                    dx = float(pred_xy[idx][0] - ref_xy[idx, 0])
                    dy = float(pred_xy[idx][1] - ref_xy[idx, 1])
                    ds = float(pred_speed[idx] - ref_v[idx])
                    dlat = float(pred_lane_offset[idx] - expert_lane_offset[idx])
                    squared[horizon]["x"].append(dx * dx)
                    squared[horizon]["y"].append(dy * dy)
                    squared[horizon]["position"].append(dx * dx + dy * dy)
                    squared[horizon]["speed"].append(ds * ds)
                    if np.isfinite(dlat):
                        squared[horizon]["lane_offset"].append(dlat * dlat)
            finally:
                if not env_cached:
                    env.close()
    finally:
        if was_training:
            policy.train()

    metrics = _matched_eval_metrics(
        prefix=prefix,
        attempted_episodes=len(indexed_scenarios),
        evaluated_episodes=evaluated_episodes,
        skipped_missing_expert=skipped_missing_expert,
        skipped_bad_reference=skipped_bad_reference,
        skipped_empty_rollout=skipped_empty_rollout,
        total_steps=total_steps,
        collision_steps=collision_steps,
        offroad_steps=offroad_steps,
        hard_brake_steps=hard_brake_steps,
        episode_lengths=episode_lengths,
        squared=squared,
        final_squared=final_squared,
        horizons=horizons,
        terminated_episodes=terminated_episodes,
        truncated_episodes=truncated_episodes,
        crashed_vehicle_episodes=crashed_vehicle_episodes,
        offroad_vehicle_episodes=offroad_vehicle_episodes,
        vehicles=evaluated_episodes,
        vehicle_episodes=evaluated_episodes,
        vehicle_ids=evaluated_vehicle_ids,
        background_idm_handover_vehicle_episodes=len(
            background_handover_keys
        ),
        action_values=action_values,
        rollout_covered_vehicle_counts=rollout_covered_vehicle_counts,
        include_raw=include_raw,
        action_execution_receipt=action_execution_receipt,
    )
    if include_raw:
        metrics[f"__raw/{prefix}/policy_forward_seconds"] = float(eval_policy_seconds)
        metrics[f"__raw/{prefix}/env_step_seconds"] = float(eval_step_seconds)
        metrics[f"__raw/{prefix}/env_reset_seconds"] = float(eval_reset_seconds)
    if include_cases:
        metrics["episodes"] = episode_records
    return metrics

__all__ = [
    'evaluation_thread_context',
    '_parse_evaluation_horizons',
    '_evaluation_scenarios',
    '_evaluation_episode_names',
    '_normalize_evaluation_vehicle_mode',
    '_evaluation_training_count_episode_specs',
    '_apply_simulator_runtime_options',
    '_make_matched_eval_env',
    '_make_matched_eval_all_vehicle_env',
    '_make_matched_eval_selected_vehicle_env',
    '_deterministic_policy_action_tuple',
    '_validated_deterministic_continuous_actions',
    '_validated_normalized_acceleration',
    '_lane_offset_for_position',
    '_first_controlled_vehicle',
    '_physical_accel_from_action',
    '_physical_accels_from_actions',
    '_record_deterministic_continuous_actions',
    '_matched_eval_metrics',
    '_strip_internal_matched_metrics',
    '_combine_matched_eval_metric_dicts',
    '_chunk_evenly',
    '_policy_input_dim',
    '_cpu_state_dict',
    '_configure_evaluation_worker_threads',
    '_eval_policy_cache_key',
    '_cached_eval_policy',
    '_matched_eval_env_cache_key',
    '_get_matched_eval_env',
    'evaluation_worker_cache_stats',
    'clear_evaluation_worker_caches',
    '_matched_eval_worker',
    '_evaluate_policy_matched_all_vehicle_episodes',
    'evaluate_policy_matched_trajectories',
    'evaluate_expert_replay_collision_baseline',
    'evaluate_expert_replay_matched_single_vehicle_floor',
    '_evaluate_policy_matched_trajectories_impl'
]
