#!/usr/bin/env python3
from __future__ import annotations

"""Build raw-trajectory expert features for parameter-sharing GAIL.

The saved dataset pairs each controlled vehicle's simulator observation with
its current raw NGSIM trajectory state. The policy still learns discrete meta
actions, but the discriminator sees observation + raw motion state instead of
an inferred expert action label.
"""

import argparse
import json
import os
import re
import sys
import warnings
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
from highway_env.ngsim_utils.data.trajectory_gen import (  # noqa: E402
    trajectory_row_is_active,
)


SCHEMA_VERSION = 2


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
    parser.add_argument("--max-surrounding", default="all")
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
    return parser.parse_args()


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
    frames: list[np.ndarray] = []

    max_per_vehicle = int(args.max_samples_per_vehicle)
    trajectory_state_source = str(args.trajectory_state_source).lower()
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
    )
    episode_name = str(getattr(base, "episode_name", episode_index))

    try:
        while not done and int(base.steps) < int(args.max_steps_per_episode):
            step_index = int(base.steps)
            obs_agents = flatten_agent_observations(current_obs)
            vehicles = getattr(base, "controlled_vehicles", ())
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
            next_obs, reward, terminated, truncated, _info = env.step(action)
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
    }
    metadata = {
        "episode_name": episode_name,
        "num_samples": int(obs_arr.shape[0]),
        "observation_dim": int(obs_arr.shape[1]),
        "next_observation_dim": int(next_obs_arr.shape[1]),
        "trajectory_state_shape": list(traj_arr.shape[1:]),
        "trajectory_state_dim": int(traj_arr.shape[1]),
        "feature_dim": int(feature_arr.shape[1]),
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
        elif name in {"observations", "next_observations", "trajectory_states"}:
            arrays[name] = np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=0)
        elif name == "rewards":
            arrays[name] = np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=0)
        elif name == "dones":
            arrays[name] = np.concatenate([np.asarray(part, dtype=bool) for part in parts], axis=0)
        elif name in {"vehicle_ids", "timesteps"}:
            arrays[name] = np.concatenate([np.asarray(part, dtype=np.int64) for part in parts], axis=0)
        else:
            arrays[name] = np.concatenate(parts, axis=0)

    observations = np.asarray(arrays["observations"], dtype=np.float32)
    trajectory_states = np.asarray(arrays["trajectory_states"], dtype=np.float32)
    features = np.asarray(arrays.get("features", discriminator_features(observations, trajectory_states)), dtype=np.float32)
    arrays["features"] = features

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "ps_traj_observation_per_episode_collection",
        "num_files": len(paths),
        "num_samples": int(observations.shape[0]),
        "observation_dim": int(observations.shape[1]),
        "trajectory_state_shape": list(trajectory_states.shape[1:]),
        "trajectory_state_dim": int(trajectory_states.shape[1]),
        "feature_dim": int(features.shape[1]),
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
    return features.astype(np.float32, copy=False), metadata, arrays


def main() -> None:
    args = parse_args()
    register_ngsim_env()

    render_mode = "rgb_array" if args.save_video else ("human" if args.visualize_episode else None)
    collection_root = _episode_output_root(args.out)
    os.makedirs(collection_root, exist_ok=True)
    if args.save_video:
        os.makedirs(os.path.abspath(str(args.video_dir)), exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    total_samples = 0

    for episode_index in range(max(1, int(args.max_episodes))):
        expert_env = make_expert_scene_env(args, render_mode=render_mode)
        try:
            arrays, episode_metadata, video_path = collect_expert_episode(
                expert_env,
                args,
                episode_index=episode_index,
            )
        finally:
            expert_env.close()

        episode_metadata = {
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
            **episode_metadata,
        }
        output_path = save_expert_dataset(
            _episode_dataset_path(
                args=args,
                episode_index=episode_index,
                episode_name=str(episode_metadata["episode_name"]),
            ),
            arrays,
            episode_metadata,
        )
        total_samples += int(arrays["observations"].shape[0])
        manifest_entries.append(
            {
                "episode_index": int(episode_index),
                "episode_name": str(episode_metadata["episode_name"]),
                "dataset_file": os.path.basename(output_path),
                "video_file": os.path.basename(video_path) if video_path else None,
                "num_samples": int(arrays["observations"].shape[0]),
                "observation_dim": int(arrays["observations"].shape[1]),
                "trajectory_state_dim": int(arrays["trajectory_states"].shape[1]),
                "feature_dim": int(arrays["features"].shape[1]),
            }
        )
        print(f"Saved episode dataset to: {output_path}")
        if video_path:
            print(f"Saved episode video to: {video_path}")

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
