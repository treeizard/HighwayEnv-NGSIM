#!/usr/bin/env python3
from __future__ import annotations

"""Build raw-trajectory expert features for parameter-sharing GAIL.

The saved dataset pairs each controlled vehicle's simulator observation with
its current raw NGSIM trajectory state. The policy still learns discrete meta
actions, but the discriminator sees observation + raw motion state instead of
an inferred expert action label.
"""

import argparse
import multiprocessing as mp
import json
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import (  # noqa: E402
    ENV_ID,
    build_env_config,
    register_ngsim_env,
)
from highway_env.ngsim_utils.core.constants import MAX_ACCEL, MAX_STEER  # noqa: E402
from highway_env.ngsim_utils.data.trajectory_gen import (  # noqa: E402
    trajectory_row_is_active,
)
from scripts_gail.ps_gail.data import (  # noqa: E402
    ACTION_CONTINUOUS_ENV_COLUMNS,
    ACTION_CONTINUOUS_ENV_KEY,
    ACTION_STEERING_ACCELERATION_COLUMNS,
    ACTION_STEERING_ACCELERATION_KEY,
    scene_snapshot_features,
)


SCHEMA_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an expert dataset for PS trajectory-GAIL. Each sample contains "
            "the observation state space and the current raw NGSIM trajectory state."
        )
    )
    parser.add_argument("--scene", type=str, default="us-101")
    parser.add_argument("--episode-root", type=str, default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", choices=["train", "val", "test"], default="train")
    parser.add_argument(
        "--out",
        type=str,
        default="expert_data/ngsim_ps_traj_expert_discrete",
        help="Output directory or prefix used to save one dataset file per collected episode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1,
        help="Number of expert replay episodes to collect from.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=200,
        help="Maximum replay steps per expert collection episode.",
    )
    parser.add_argument(
        "--max-samples-per-vehicle",
        type=int,
        default=200,
        help="Cap per-step samples per vehicle. Use 0 for no per-vehicle cap.",
    )
    parser.add_argument(
        "--num-collection-workers",
        type=int,
        default=1,
        help="Number of parallel episode-collection worker processes. Use 1 for serial collection.",
    )
    parser.add_argument(
        "--collection-worker-threads",
        type=int,
        default=2,
        help="Native CPU threads allowed inside each collection worker process.",
    )
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument(
        "--scene-max-vehicles",
        type=int,
        default=64,
        help="Maximum number of road vehicles encoded in each full-scene discriminator snapshot.",
    )
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        default=False,
        help="Collect samples for every viable controlled vehicle in each scene.",
    )
    parser.add_argument(
        "--no-control-all-vehicles",
        dest="control_all_vehicles",
        action="store_false",
        help="Collect only a percentage of valid vehicles.",
    )
    parser.add_argument(
        "--percentage-controlled-vehicles",
        type=float,
        default=0.1,
        help="Fraction of valid ego vehicles to collect when not using all vehicles.",
    )
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--maximum-range", type=float, default=64.0)
    parser.add_argument("--simulation-frequency", type=int, default=10)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--allow-idm",
        action="store_true",
        default=False,
        help="Allow surrounding replay vehicles to hand over to IDM when replay is unsafe/exhausted.",
    )
    parser.add_argument(
        "--no-allow-idm",
        dest="allow_idm",
        action="store_false",
        help="Keep surrounding vehicles on pure replay only; no IDM intervention.",
    )
    parser.add_argument(
        "--expert-control-mode",
        choices=["teleport", "continuous"],
        default="continuous",
        help=(
            "How controlled vehicles are advanced during collection. "
            "'teleport' replays raw rows exactly, while 'continuous' uses the "
            "internal expert controller to produce physically constrained motion."
        ),
    )
    parser.add_argument(
        "--trajectory-state-source",
        choices=["raw", "simulated"],
        default="simulated",
        help=(
            "Which (x, y, v) state to save beside each observation. "
            "'raw' uses the processed NGSIM row, while 'simulated' uses the "
            "current controlled vehicle state."
        ),
    )
    parser.add_argument(
        "--visualize-episode",
        action="store_true",
        help="Render one collection episode while building the expert dataset.",
    )
    parser.add_argument(
        "--visualize-episode-index",
        type=int,
        default=0,
        help="Zero-based collected episode index to render when --visualize-episode is set.",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1200,
        help="Viewer width used by --visualize-episode.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=600,
        help="Viewer height used by --visualize-episode.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=5.5,
        help="Viewer scaling used by --visualize-episode.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the selected visualization episode to disk using RecordVideo.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="expert_data/videos",
        help="Output directory for per-episode videos when --save-video is enabled.",
    )
    parser.add_argument(
        "--video-prefix",
        type=str,
        default="ps_traj_expert",
        help="Filename prefix for per-episode video recordings.",
    )
    parser.add_argument(
        "--progress-update-interval",
        type=int,
        default=10,
        help="How often to refresh tqdm postfix text. Higher values reduce logging overhead.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable per-episode tqdm progress bars. Parallel collection uses one parent progress bar.",
    )
    return parser.parse_args()


def configure_native_threads(num_threads: int) -> None:
    threads = str(max(1, int(num_threads)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads


def observation_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "LidarCameraObservations",
        "lidar": {
            "cells": int(args.cells),
            "maximum_range": float(args.maximum_range),
            "normalize": True,
        },
        "camera": {
            "cells": 21,
            "maximum_range": float(args.maximum_range),
            "field_of_view": np.pi / 2,
            "normalize": True,
        },
    }


def make_expert_scene_env(args: argparse.Namespace, *, render_mode: str | None = None) -> gym.Env:
    cfg = build_env_config(
        scene=args.scene,
        action_mode=str(args.expert_control_mode),
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        percentage_controlled_vehicles=float(args.percentage_controlled_vehicles),
        control_all_vehicles=bool(args.control_all_vehicles),
        max_surrounding=args.max_surrounding,
        observation_config=observation_config_from_args(args),
        simulation_frequency=args.simulation_frequency,
        policy_frequency=args.policy_frequency,
        max_episode_steps=args.max_episode_steps,
        seed=None,
        scene_dataset_collection_mode=True,
        allow_idm=bool(args.allow_idm),
    )
    if cfg.get("observation", {}).get("type") == "MultiAgentObservation":
        shared_obs_cfg = cfg["observation"].get("observation_config", {})
        if shared_obs_cfg.get("type") == "LidarCameraObservations":
            cfg["observation"] = {
                "type": "SharedMultiAgentLidarCameraObservations",
                "lidar": dict(shared_obs_cfg.get("lidar", {})),
                "camera": dict(shared_obs_cfg.get("camera", {})),
            }
    cfg["expert_test_mode"] = str(args.expert_control_mode) != "teleport"
    cfg["disable_controlled_vehicle_collisions"] = True
    cfg["terminate_when_all_controlled_crashed"] = False
    cfg["offscreen_rendering"] = render_mode == "rgb_array"
    cfg["screen_width"] = int(args.screen_width)
    cfg["screen_height"] = int(args.screen_height)
    cfg["scaling"] = float(args.scaling)
    return gym.make(ENV_ID, render_mode=render_mode, config=cfg)


def flatten_observation_value(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [flatten_observation_value(obs[key]) for key in sorted(obs)]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if isinstance(obs, (tuple, list)):
        parts = [flatten_observation_value(item) for item in obs]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def flatten_lidar_camera_agent_observation_fast(obs: Any) -> np.ndarray | None:
    """Fast path for the common (lidar, camera, ego_state) observation tuple."""
    if not isinstance(obs, tuple) or len(obs) != 3:
        return None
    lidar, camera, ego_state = obs
    lidar_arr = np.asarray(lidar, dtype=np.float32)
    camera_arr = np.asarray(camera, dtype=np.float32)
    ego_arr = np.asarray(ego_state, dtype=np.float32)
    if lidar_arr.ndim != 2 or camera_arr.ndim != 2:
        return None
    return np.concatenate(
        (
            lidar_arr.reshape(-1),
            camera_arr.reshape(-1),
            ego_arr.reshape(-1),
        ),
        axis=0,
    ).astype(np.float32, copy=False)


def flatten_agent_observations(obs: Any) -> np.ndarray:
    if isinstance(obs, tuple) and obs and isinstance(obs[0], (tuple, list, dict)):
        fast_rows = []
        use_fast_path = True
        for item in obs:
            fast_row = flatten_lidar_camera_agent_observation_fast(item)
            if fast_row is None:
                use_fast_path = False
                break
            fast_rows.append(fast_row)
        if use_fast_path and fast_rows:
            return np.stack(fast_rows, axis=0)
        return np.stack([flatten_observation_value(item) for item in obs], axis=0)
    if isinstance(obs, tuple):
        return flatten_observation_value(obs).reshape(1, -1)
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim >= 2:
        return arr.reshape(arr.shape[0], -1)
    return arr.reshape(1, -1)


def idle_action_for_env(env: gym.Env, num_agents: int) -> int | tuple[int, ...]:
    idle = 1
    if hasattr(env.action_space, "spaces"):
        return tuple(idle for _ in range(num_agents))
    return idle


def _coerce_continuous_env_action(action: Any) -> np.ndarray:
    """Return normalized env-native [acceleration, steering] action."""
    if action is None:
        raise ValueError("Continuous expert action is missing for an active controlled vehicle.")
    if isinstance(action, dict):
        accel_norm = float(action.get("acceleration", 0.0)) / float(MAX_ACCEL)
        steer_norm = float(action.get("steering", 0.0)) / float(MAX_STEER)
        arr = np.asarray([accel_norm, steer_norm], dtype=np.float32)
    else:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            raise ValueError(
                f"Continuous expert action must contain [acceleration, steering], got shape {arr.shape}."
            )
        arr = arr[:2].astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Continuous expert action contains non-finite values: {arr}.")
    return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)


def continuous_env_action_to_steering_acceleration(action: np.ndarray) -> np.ndarray:
    """Convert normalized [acceleration, steering] to physical [steering, acceleration]."""
    action = _coerce_continuous_env_action(action)
    accel_norm, steer_norm = float(action[0]), float(action[1])
    return np.asarray(
        [steer_norm * float(MAX_STEER), accel_norm * float(MAX_ACCEL)],
        dtype=np.float32,
    )


def continuous_expert_actions_from_info(
    info: dict[str, Any],
    *,
    num_agents: int,
    require: bool,
) -> list[np.ndarray] | None:
    """Extract normalized continuous expert actions from one env step."""
    action_views = None
    if "expert_action_continuous_all" in info:
        action_views = info["expert_action_continuous_all"]
    elif "applied_actions" in info:
        action_views = info["applied_actions"]
    elif int(num_agents) == 1 and "expert_action_continuous" in info:
        action_views = [info["expert_action_continuous"]]
    elif int(num_agents) == 1 and "applied_action" in info:
        action_views = [info["applied_action"]]

    if action_views is None:
        if require:
            raise RuntimeError(
                "Unified expert collection expected continuous expert actions, but the environment "
                "did not expose expert_action_continuous_all/applied_actions. Use "
                "--expert-control-mode continuous."
            )
        return None
    action_list = list(action_views)
    if len(action_list) != int(num_agents):
        raise RuntimeError(
            "Continuous expert action count mismatch: "
            f"actions={len(action_list)} controlled_agents={int(num_agents)}."
        )
    return [_coerce_continuous_env_action(action) for action in action_list]


def validate_transition_array_lengths(arrays: dict[str, np.ndarray]) -> None:
    expected = int(arrays["observations"].shape[0])
    per_transition_keys = {
        "next_observations",
        "trajectory_states",
        "features",
        "vehicle_ids",
        "timesteps",
        "dones",
        "rewards",
        ACTION_CONTINUOUS_ENV_KEY,
        ACTION_STEERING_ACCELERATION_KEY,
    }
    for key in sorted(per_transition_keys.intersection(arrays)):
        actual = int(np.asarray(arrays[key]).shape[0])
        if actual != expected:
            raise ValueError(
                f"Array {key!r} has length {actual}, expected {expected} to match observations."
            )


def raw_trajectory_state_from_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.shape[0] < 3:
        raise ValueError(f"Expected raw trajectory row with at least x, y, speed; got {row.shape}.")
    return row[:3].astype(np.float32, copy=False)


def simulated_trajectory_state_from_vehicle(vehicle: Any) -> np.ndarray:
    position = np.asarray(getattr(vehicle, "position", np.zeros(2, dtype=float)), dtype=np.float32)
    if position.shape[0] < 2:
        raise ValueError(f"Expected vehicle position with at least 2 values, got {position.shape}.")
    speed = np.asarray([float(getattr(vehicle, "speed", 0.0))], dtype=np.float32)
    return np.concatenate([position[:2], speed], axis=0).astype(np.float32, copy=False)


def trajectory_state_from_row_fast(row: np.ndarray) -> np.ndarray:
    row_arr = np.asarray(row, dtype=np.float32)
    if row_arr.ndim != 1 or row_arr.shape[0] < 3:
        return raw_trajectory_state_from_row(row_arr)
    return row_arr[:3]


def reward_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def discriminator_features(observations: np.ndarray, trajectory_states: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    trajectory_states = np.asarray(trajectory_states, dtype=np.float32)
    if observations.ndim != 2 or trajectory_states.ndim != 2:
        raise ValueError("Discriminator observations and trajectory states must be rank-2 arrays.")
    if len(observations) != len(trajectory_states):
        raise ValueError(
            "Observation/trajectory-state count mismatch: "
            f"{len(observations)} != {len(trajectory_states)}"
        )
    return np.concatenate([observations, trajectory_states], axis=1).astype(np.float32, copy=False)


def _sanitize_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token.strip("._-") or "unknown"


def _episode_output_root(out_arg: str) -> str:
    base = os.path.abspath(str(out_arg))
    if base.endswith(".npz"):
        base = os.path.splitext(base)[0]
    return base


def _episode_file_stem(
    *,
    args: argparse.Namespace,
    episode_index: int,
    episode_name: str,
) -> str:
    return (
        f"episode_{episode_index:04d}_"
        f"{_sanitize_path_token(args.scene)}_"
        f"{_sanitize_path_token(args.prebuilt_split)}_"
        f"{_sanitize_path_token(episode_name)}"
    )


def _episode_dataset_path(
    *,
    args: argparse.Namespace,
    episode_index: int,
    episode_name: str,
) -> str:
    root = _episode_output_root(args.out)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, _episode_file_stem(args=args, episode_index=episode_index, episode_name=episode_name) + ".npz")


def _episode_video_path(
    *,
    args: argparse.Namespace,
    episode_index: int,
    episode_name: str,
) -> str:
    return os.path.join(
        os.path.abspath(str(args.video_dir)),
        f"{_sanitize_path_token(args.video_prefix)}_{_episode_file_stem(args=args, episode_index=episode_index, episode_name=episode_name)}.mp4",
    )


def _save_video_frames(path: str, frames: list[np.ndarray], fps: int) -> str | None:
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError:
        warnings.warn(
            "imageio is not installed; continuing without per-episode video export.",
            stacklevel=2,
        )
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with imageio.get_writer(path, fps=max(1, int(fps))) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return path


def collect_expert_episode(
    env: gym.Env,
    args: argparse.Namespace,
    *,
    episode_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str | None]:
    observations: list[np.ndarray] = []
    trajectory_states: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    vehicle_ids: list[int] = []
    timesteps: list[int] = []
    dones: list[bool] = []
    rewards: list[float] = []
    actions_continuous_env: list[np.ndarray] = []
    actions_steering_acceleration: list[np.ndarray] = []
    scene_features: list[np.ndarray] = []
    scene_timesteps: list[int] = []
    frames: list[np.ndarray] = []

    max_per_vehicle = int(args.max_samples_per_vehicle)
    trajectory_state_source = str(args.trajectory_state_source).lower()
    require_continuous_actions = str(args.expert_control_mode).lower() == "continuous"
    progress_update_interval = max(1, int(args.progress_update_interval))
    render_this_episode = (
        bool(args.visualize_episode)
        and episode_index == int(args.visualize_episode_index)
    )
    capture_video = bool(args.save_video)
    current_obs, _reset_info = env.reset(seed=int(args.seed) + episode_index)
    if render_this_episode or capture_video:
        frame = env.render()
        if capture_video and frame is not None:
            frames.append(np.asarray(frame))
    base = env.unwrapped
    per_vehicle_counts: dict[tuple[str, int], int] = {}
    episode_sample_count = 0
    done = False
    progress = tqdm(
        total=int(args.max_steps_per_episode),
        desc=f"episode {episode_index + 1}/{int(args.max_episodes)}",
        unit="step",
        leave=True,
        disable=bool(getattr(args, "disable_progress", False)),
    )
    episode_name = str(getattr(base, "episode_name", episode_index))

    try:
        while not done and int(base.steps) < int(args.max_steps_per_episode):
            step_index = int(base.steps)
            obs_agents = flatten_agent_observations(current_obs)
            vehicles = getattr(base, "controlled_vehicles", ())
            active_controlled_positions = [
                np.asarray(vehicle.position, dtype=np.float32)
                for vehicle in vehicles
                if bool(getattr(vehicle, "scene_collection_is_active", True))
                and getattr(vehicle, "position", None) is not None
            ]
            scene_origin = (
                np.mean(np.stack(active_controlled_positions, axis=0), axis=0)
                if active_controlled_positions
                else None
            )
            road = getattr(base, "road", None)
            scene_features.append(
                scene_snapshot_features(
                    list(getattr(road, "vehicles", ())) if road is not None else [],
                    max_vehicles=int(args.scene_max_vehicles),
                    origin=scene_origin,
                )
            )
            scene_timesteps.append(step_index)
            if len(obs_agents) != len(vehicles):
                raise RuntimeError(
                    "Expert collection observation/vehicle mismatch: "
                    f"obs_agents={len(obs_agents)} vehicles={len(vehicles)}"
                )

            current_states: list[np.ndarray | None] = []
            active_mask: list[bool] = []
            for vehicle in vehicles:
                is_active = bool(getattr(vehicle, "scene_collection_is_active", True))
                active_mask.append(is_active)
                if trajectory_state_source == "simulated" and is_active:
                    current_states.append(simulated_trajectory_state_from_vehicle(vehicle))
                else:
                    current_states.append(None)

            action = idle_action_for_env(env, len(vehicles))
            next_obs, reward, terminated, truncated, info = env.step(action)
            info = info or {}
            step_continuous_actions = continuous_expert_actions_from_info(
                info,
                num_agents=len(vehicles),
                require=require_continuous_actions,
            )
            if render_this_episode or capture_video:
                frame = env.render()
                if capture_video and frame is not None:
                    frames.append(np.asarray(frame))
            next_obs_agents = flatten_agent_observations(next_obs)
            if len(next_obs_agents) != len(vehicles):
                raise RuntimeError(
                    "Expert collection next-observation/vehicle mismatch: "
                    f"next_obs_agents={len(next_obs_agents)} vehicles={len(vehicles)}"
                )
            step_done = bool(terminated or truncated)

            for agent_idx, vehicle in enumerate(vehicles):
                if not active_mask[agent_idx]:
                    continue
                vehicle_id = int(getattr(vehicle, "vehicle_ID", -1))
                key = (episode_name, vehicle_id)
                if max_per_vehicle > 0 and per_vehicle_counts.get(key, 0) >= max_per_vehicle:
                    continue

                traj = getattr(vehicle, "scene_collection_full_traj", ())
                if len(traj) == 0:
                    continue
                if step_index >= len(traj) or not trajectory_row_is_active(traj[step_index]):
                    continue

                observations.append(obs_agents[agent_idx].astype(np.float32, copy=False))
                next_observations.append(next_obs_agents[agent_idx].astype(np.float32, copy=False))
                if trajectory_state_source == "simulated":
                    trajectory_states.append(
                        np.asarray(current_states[agent_idx], dtype=np.float32).copy()
                    )
                else:
                    trajectory_states.append(trajectory_state_from_row_fast(traj[step_index]))
                vehicle_ids.append(vehicle_id)
                timesteps.append(step_index)
                dones.append(step_done)
                rewards.append(reward_scalar(reward))
                if step_continuous_actions is not None:
                    continuous_action = step_continuous_actions[agent_idx]
                    actions_continuous_env.append(continuous_action.copy())
                    actions_steering_acceleration.append(
                        continuous_env_action_to_steering_acceleration(continuous_action)
                    )
                per_vehicle_counts[key] = per_vehicle_counts.get(key, 0) + 1
                episode_sample_count += 1

            progress.update(1)
            if step_index % progress_update_interval == 0:
                active_count = sum(
                    bool(getattr(v, "scene_collection_is_active", True)) for v in vehicles
                )
                progress.set_postfix_str(
                    f"samples={episode_sample_count} active={active_count}"
                )

            if (
                max_per_vehicle > 0
                and episode_sample_count >= max_per_vehicle * max(1, len(vehicles))
            ):
                break

            current_obs = next_obs
            done = step_done
    finally:
        progress.close()

    if not observations:
        raise RuntimeError(f"No expert observation/trajectory samples could be collected for episode {episode_name}.")

    obs_arr = np.stack(observations, axis=0).astype(np.float32, copy=False)
    next_obs_arr = np.stack(next_observations, axis=0).astype(np.float32, copy=False)
    traj_arr = np.stack(trajectory_states, axis=0).astype(np.float32, copy=False)
    feature_arr = discriminator_features(obs_arr, traj_arr)
    arrays = {
        "observations": obs_arr,
        "next_observations": next_obs_arr,
        "trajectory_states": traj_arr,
        "features": feature_arr,
        "vehicle_ids": np.asarray(vehicle_ids, dtype=np.int64),
        "timesteps": np.asarray(timesteps, dtype=np.int64),
        "dones": np.asarray(dones, dtype=bool),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "scene_features": np.stack(scene_features, axis=0).astype(np.float32, copy=False),
        "scene_timesteps": np.asarray(scene_timesteps, dtype=np.int64),
    }
    if actions_continuous_env:
        arrays[ACTION_CONTINUOUS_ENV_KEY] = np.stack(actions_continuous_env, axis=0).astype(
            np.float32,
            copy=False,
        )
        arrays[ACTION_STEERING_ACCELERATION_KEY] = np.stack(
            actions_steering_acceleration,
            axis=0,
        ).astype(np.float32, copy=False)
    validate_transition_array_lengths(arrays)
    metadata = {
        "episode_name": episode_name,
        "num_samples": int(obs_arr.shape[0]),
        "observation_dim": int(obs_arr.shape[1]),
        "next_observation_dim": int(next_obs_arr.shape[1]),
        "trajectory_state_shape": list(traj_arr.shape[1:]),
        "trajectory_state_dim": int(traj_arr.shape[1]),
        "feature_dim": int(feature_arr.shape[1]),
        "scene_feature_dim": int(arrays["scene_features"].shape[1]),
        "scene_max_vehicles": int(args.scene_max_vehicles),
        "continuous_action_dim": int(arrays[ACTION_CONTINUOUS_ENV_KEY].shape[1])
        if ACTION_CONTINUOUS_ENV_KEY in arrays
        else None,
        "actions_continuous_env_columns": list(ACTION_CONTINUOUS_ENV_COLUMNS)
        if ACTION_CONTINUOUS_ENV_KEY in arrays
        else None,
        "actions_steering_acceleration_columns": list(ACTION_STEERING_ACCELERATION_COLUMNS)
        if ACTION_STEERING_ACCELERATION_KEY in arrays
        else None,
        "controlled_vehicle_ids": sorted({int(v) for v in vehicle_ids}),
        "video_requested": bool(args.save_video),
    }
    video_path = None
    if capture_video:
        video_path = _save_video_frames(
            _episode_video_path(args=args, episode_index=episode_index, episode_name=episode_name),
            frames,
            fps=int(args.policy_frequency),
        )
    return arrays, metadata, video_path


def save_expert_dataset(path: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )
    return output_path


def _aggregate_saved_episode_datasets(paths: list[str]) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    arrays_per_key: dict[str, list[np.ndarray]] = {}
    metadata_items: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            required = {"observations", "trajectory_states", "metadata_json"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise KeyError(f"{path} is missing required arrays: {missing}")
            metadata = json.loads(str(data["metadata_json"].item()))
            metadata_items.append(metadata)
            for name in data.files:
                if name == "metadata_json":
                    continue
                arrays_per_key.setdefault(name, []).append(np.asarray(data[name]))

    arrays: dict[str, np.ndarray] = {}
    for name, parts in arrays_per_key.items():
        if name == "features":
            arrays[name] = np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=0)
        elif name in {
            "observations",
            "next_observations",
            "trajectory_states",
            "scene_features",
            ACTION_CONTINUOUS_ENV_KEY,
            ACTION_STEERING_ACCELERATION_KEY,
        }:
            arrays[name] = np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=0)
        elif name == "rewards":
            arrays[name] = np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=0)
        elif name == "dones":
            arrays[name] = np.concatenate([np.asarray(part, dtype=bool) for part in parts], axis=0)
        elif name in {"vehicle_ids", "timesteps", "scene_timesteps"}:
            arrays[name] = np.concatenate([np.asarray(part, dtype=np.int64) for part in parts], axis=0)
        else:
            arrays[name] = np.concatenate(parts, axis=0)

    observations = np.asarray(arrays["observations"], dtype=np.float32)
    trajectory_states = np.asarray(arrays["trajectory_states"], dtype=np.float32)
    features = np.asarray(arrays.get("features", discriminator_features(observations, trajectory_states)), dtype=np.float32)
    arrays["features"] = features
    validate_transition_array_lengths(arrays)
    schema_versions = sorted(
        {
            int(item["schema_version"])
            for item in metadata_items
            if "schema_version" in item
        }
    )

    metadata = {
        "schema_version": schema_versions[0] if len(schema_versions) == 1 else SCHEMA_VERSION,
        "schema_versions": schema_versions,
        "dataset_kind": "ps_traj_observation_per_episode_collection",
        "num_files": len(paths),
        "num_samples": int(observations.shape[0]),
        "observation_dim": int(observations.shape[1]),
        "trajectory_state_shape": list(trajectory_states.shape[1:]),
        "trajectory_state_dim": int(trajectory_states.shape[1]),
        "feature_dim": int(features.shape[1]),
        "scene_feature_dim": int(np.asarray(arrays["scene_features"]).shape[1])
        if "scene_features" in arrays
        else None,
        "continuous_action_dim": int(np.asarray(arrays[ACTION_CONTINUOUS_ENV_KEY]).shape[1])
        if ACTION_CONTINUOUS_ENV_KEY in arrays
        else None,
        "actions_continuous_env_columns": list(ACTION_CONTINUOUS_ENV_COLUMNS)
        if ACTION_CONTINUOUS_ENV_KEY in arrays
        else None,
        "actions_steering_acceleration_columns": list(ACTION_STEERING_ACCELERATION_COLUMNS)
        if ACTION_STEERING_ACCELERATION_KEY in arrays
        else None,
        "episodes": metadata_items,
    }
    return features, metadata, arrays


def load_ps_traj_expert_dataset(path: str) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            file_paths = [
                os.path.join(path, item["dataset_file"])
                for item in manifest.get("episodes", [])
            ]
        else:
            file_paths = sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.endswith(".npz")
            )
        if not file_paths:
            raise FileNotFoundError(f"No per-episode expert datasets were found under {path}.")
        return _aggregate_saved_episode_datasets(file_paths)

    with np.load(path, allow_pickle=True) as data:
        required = {"observations", "trajectory_states", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")

        metadata = json.loads(str(data["metadata_json"].item()))
        observations = np.asarray(data["observations"], dtype=np.float32)
        trajectory_states = np.asarray(data["trajectory_states"], dtype=np.float32)
        if "next_observations" in data.files:
            next_observations = np.asarray(data["next_observations"], dtype=np.float32)
        else:
            next_observations = None
        if "features" in data.files:
            features = np.asarray(data["features"], dtype=np.float32)
        else:
            features = discriminator_features(observations, trajectory_states)

        if observations.ndim != 2:
            raise ValueError(f"Expected observations [N, obs_dim], got {observations.shape}.")
        if next_observations is not None and (
            next_observations.ndim != 2 or next_observations.shape != observations.shape
        ):
            raise ValueError(
                f"Expected next_observations to match observations, got {next_observations.shape} vs {observations.shape}."
            )
        if trajectory_states.ndim != 2 or trajectory_states.shape[1] != 3:
            raise ValueError(f"Expected trajectory_states [N, 3], got {trajectory_states.shape}.")
        if features.ndim != 2 or features.shape[0] != observations.shape[0]:
            raise ValueError(f"Invalid feature shape {features.shape} for observations {observations.shape}.")
        if features.shape[1] != observations.shape[1] + trajectory_states.shape[1]:
            raise ValueError(
                f"Feature dim {features.shape[1]} does not match observation + trajectory-state dims."
            )

        arrays = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
        validate_transition_array_lengths(arrays)
        if ACTION_CONTINUOUS_ENV_KEY in arrays:
            actions_continuous_env = np.asarray(arrays[ACTION_CONTINUOUS_ENV_KEY], dtype=np.float32)
            if actions_continuous_env.ndim != 2 or actions_continuous_env.shape[1] != 2:
                raise ValueError(
                    f"{ACTION_CONTINUOUS_ENV_KEY} must have shape [N, 2] with columns "
                    f"{ACTION_CONTINUOUS_ENV_COLUMNS}, got {actions_continuous_env.shape}."
                )
            if not np.all(np.isfinite(actions_continuous_env)):
                raise ValueError(f"{ACTION_CONTINUOUS_ENV_KEY} contains non-finite values.")
        if ACTION_STEERING_ACCELERATION_KEY in arrays:
            steering_accel = np.asarray(arrays[ACTION_STEERING_ACCELERATION_KEY], dtype=np.float32)
            if steering_accel.ndim != 2 or steering_accel.shape[1] != 2:
                raise ValueError(
                    f"{ACTION_STEERING_ACCELERATION_KEY} must have shape [N, 2] with columns "
                    f"{ACTION_STEERING_ACCELERATION_COLUMNS}, got {steering_accel.shape}."
                )
            if not np.all(np.isfinite(steering_accel)):
                raise ValueError(f"{ACTION_STEERING_ACCELERATION_KEY} contains non-finite values.")
    return features.astype(np.float32, copy=False), metadata, arrays


def render_mode_from_args(args: argparse.Namespace) -> str | None:
    return "rgb_array" if args.save_video else ("human" if args.visualize_episode else None)


def enriched_episode_metadata(
    args: argparse.Namespace,
    *,
    episode_index: int,
    episode_metadata: dict[str, Any],
    video_path: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "ps_traj_observation_episode",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_id": ENV_ID,
        "scene": str(args.scene),
        "action_mode": str(args.expert_control_mode),
        "expert_test_mode": bool(str(args.expert_control_mode) != "teleport"),
        "episode_root": os.path.abspath(args.episode_root),
        "prebuilt_split": str(args.prebuilt_split),
        "max_episodes": int(args.max_episodes),
        "episode_index": int(episode_index),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "max_samples_per_vehicle": int(args.max_samples_per_vehicle),
        "percentage_controlled_vehicles": float(args.percentage_controlled_vehicles),
        "control_all_vehicles": bool(args.control_all_vehicles),
        "max_surrounding": args.max_surrounding,
        "simulation_frequency": int(args.simulation_frequency),
        "policy_frequency": int(args.policy_frequency),
        "trajectory_state_source": str(args.trajectory_state_source),
        "observation_config": observation_config_from_args(args),
        "allow_idm": bool(args.allow_idm),
        "save_video": bool(args.save_video),
        "video_dir": os.path.abspath(str(args.video_dir)) if args.save_video else None,
        "video_path": video_path,
        "collection_worker_threads": int(args.collection_worker_threads),
        **episode_metadata,
    }


def manifest_entry_from_episode(
    *,
    episode_index: int,
    episode_metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_path: str,
    video_path: str | None,
) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "episode_name": str(episode_metadata["episode_name"]),
        "dataset_file": os.path.basename(output_path),
        "video_file": os.path.basename(video_path) if video_path else None,
        "num_samples": int(arrays["observations"].shape[0]),
        "observation_dim": int(arrays["observations"].shape[1]),
        "trajectory_state_dim": int(arrays["trajectory_states"].shape[1]),
        "feature_dim": int(arrays["features"].shape[1]),
        "scene_feature_dim": int(arrays["scene_features"].shape[1]),
        "continuous_action_dim": int(arrays[ACTION_CONTINUOUS_ENV_KEY].shape[1])
        if ACTION_CONTINUOUS_ENV_KEY in arrays
        else None,
    }


def collect_and_save_expert_episode(
    args: argparse.Namespace,
    *,
    episode_index: int,
    render_mode: str | None,
) -> dict[str, Any]:
    expert_env = make_expert_scene_env(args, render_mode=render_mode)
    try:
        arrays, episode_metadata, video_path = collect_expert_episode(
            expert_env,
            args,
            episode_index=episode_index,
        )
    finally:
        expert_env.close()

    episode_metadata = enriched_episode_metadata(
        args,
        episode_index=episode_index,
        episode_metadata=episode_metadata,
        video_path=video_path,
    )
    output_path = save_expert_dataset(
        _episode_dataset_path(
            args=args,
            episode_index=episode_index,
            episode_name=str(episode_metadata["episode_name"]),
        ),
        arrays,
        episode_metadata,
    )
    return {
        "episode_index": int(episode_index),
        "manifest_entry": manifest_entry_from_episode(
            episode_index=episode_index,
            episode_metadata=episode_metadata,
            arrays=arrays,
            output_path=output_path,
            video_path=video_path,
        ),
        "num_samples": int(arrays["observations"].shape[0]),
        "output_path": output_path,
        "video_path": video_path,
    }


def _collection_worker(args_dict: dict[str, Any], episode_index: int) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    configure_native_threads(int(args.collection_worker_threads))
    register_ngsim_env()
    return collect_and_save_expert_episode(
        args,
        episode_index=int(episode_index),
        render_mode=render_mode_from_args(args),
    )


def collect_expert_episodes(args: argparse.Namespace) -> list[dict[str, Any]]:
    max_episodes = max(1, int(args.max_episodes))
    num_workers = max(1, int(args.num_collection_workers))
    if bool(args.visualize_episode) and num_workers > 1:
        warnings.warn(
            "--visualize-episode requires serial collection; forcing --num-collection-workers 1.",
            stacklevel=2,
        )
        num_workers = 1
    num_workers = min(num_workers, max_episodes)

    if num_workers == 1:
        register_ngsim_env()
        render_mode = render_mode_from_args(args)
        return [
            collect_and_save_expert_episode(
                args,
                episode_index=episode_index,
                render_mode=render_mode,
            )
            for episode_index in range(max_episodes)
        ]

    worker_args = argparse.Namespace(**vars(args))
    worker_args.disable_progress = True
    args_dict = vars(worker_args)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(_collection_worker, args_dict, episode_index): episode_index
            for episode_index in range(max_episodes)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="episodes",
            unit="episode",
        ):
            results.append(future.result())
    results.sort(key=lambda item: int(item["episode_index"]))
    return results


def main() -> None:
    args = parse_args()
    configure_native_threads(int(args.collection_worker_threads))

    collection_root = _episode_output_root(args.out)
    os.makedirs(collection_root, exist_ok=True)
    if args.save_video:
        os.makedirs(os.path.abspath(str(args.video_dir)), exist_ok=True)

    print(
        "Expert collection workers="
        f"{max(1, int(args.num_collection_workers))} "
        f"worker_threads={max(1, int(args.collection_worker_threads))}"
    )
    results = collect_expert_episodes(args)
    manifest_entries = [dict(result["manifest_entry"]) for result in results]
    total_samples = int(sum(int(result["num_samples"]) for result in results))
    for result in results:
        print(f"Saved episode dataset to: {result['output_path']}")
        if result.get("video_path"):
            print(f"Saved episode video to: {result['video_path']}")

    has_continuous_actions = any(
        entry.get("continuous_action_dim") is not None for entry in manifest_entries
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "ps_traj_observation_per_episode_collection",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene": str(args.scene),
        "prebuilt_split": str(args.prebuilt_split),
        "action_mode": str(args.expert_control_mode),
        "trajectory_state_source": str(args.trajectory_state_source),
        "allow_idm": bool(args.allow_idm),
        "control_all_vehicles": bool(args.control_all_vehicles),
        "percentage_controlled_vehicles": float(args.percentage_controlled_vehicles),
        "scene_max_vehicles": int(args.scene_max_vehicles),
        "num_collection_workers": max(1, int(args.num_collection_workers)),
        "collection_worker_threads": max(1, int(args.collection_worker_threads)),
        "continuous_action_dim": 2 if has_continuous_actions else None,
        "actions_continuous_env_key": ACTION_CONTINUOUS_ENV_KEY if has_continuous_actions else None,
        "actions_continuous_env_columns": list(ACTION_CONTINUOUS_ENV_COLUMNS)
        if has_continuous_actions
        else None,
        "actions_steering_acceleration_key": ACTION_STEERING_ACCELERATION_KEY
        if has_continuous_actions
        else None,
        "actions_steering_acceleration_columns": list(ACTION_STEERING_ACCELERATION_COLUMNS)
        if has_continuous_actions
        else None,
        "num_episodes": len(manifest_entries),
        "num_samples": int(total_samples),
        "episodes": manifest_entries,
    }
    manifest_path = os.path.join(collection_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved per-episode PS trajectory expert datasets under: {collection_root}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"episodes={len(manifest_entries)} samples={total_samples}")


if __name__ == "__main__":
    main()
