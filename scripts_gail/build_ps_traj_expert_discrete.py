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
import sys
import warnings
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm
from gymnasium.wrappers import RecordVideo


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import (  # noqa: E402
    ENV_ID,
    build_env_config,
    register_ngsim_env,
)
from highway_env.ngsim_utils.trajectory_gen import (  # noqa: E402
    trajectory_row_is_active,
)


SCHEMA_VERSION = 1


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
        default="expert_data/ngsim_ps_traj_expert_discrete.npz",
        help="Output expert dataset path.",
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
        default=True,
        help="Collect samples for every viable controlled vehicle in each scene.",
    )
    parser.add_argument(
        "--no-control-all-vehicles",
        dest="control_all_vehicles",
        action="store_false",
        help="Collect only --controlled-vehicles vehicles.",
    )
    parser.add_argument("--controlled-vehicles", type=int, default=4)
    parser.add_argument(
        "--max-controlled-vehicles",
        type=int,
        default=0,
        help="Optional cap for --control-all-vehicles scenes. 0 keeps every vehicle.",
    )
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--maximum-range", type=float, default=64.0)
    parser.add_argument("--simulation-frequency", type=int, default=10)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument(
        "--allow-idm",
        action="store_true",
        default=True,
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
        default="teleport",
        help=(
            "How controlled vehicles are advanced during collection. "
            "'teleport' replays raw rows exactly, while 'continuous' uses the "
            "internal expert controller to produce physically constrained motion."
        ),
    )
    parser.add_argument(
        "--trajectory-state-source",
        choices=["raw", "simulated"],
        default="raw",
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
        help="Output directory for --save-video.",
    )
    parser.add_argument(
        "--video-prefix",
        type=str,
        default="ps_traj_expert",
        help="Filename prefix for --save-video recordings.",
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
        controlled_vehicles=max(1, int(args.controlled_vehicles)),
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
    if int(args.max_controlled_vehicles) > 0:
        cfg["max_controlled_vehicles"] = int(args.max_controlled_vehicles)
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


def collect_expert_samples(env: gym.Env, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], int]:
    observations: list[np.ndarray] = []
    trajectory_states: list[np.ndarray] = []
    episode_names: list[str] = []
    vehicle_ids: list[int] = []
    timesteps: list[int] = []

    max_per_vehicle = int(args.max_samples_per_vehicle)
    trajectory_state_source = str(args.trajectory_state_source).lower()
    progress_update_interval = max(1, int(args.progress_update_interval))
    episodes = max(1, int(args.max_episodes))
    collected_episodes = 0

    for episode_idx in range(episodes):
        render_this_episode = (
            bool(args.visualize_episode)
            and episode_idx == int(args.visualize_episode_index)
        )
        manual_render_this_episode = render_this_episode and not bool(args.save_video)
        current_obs, _reset_info = env.reset(seed=int(args.seed) + episode_idx)
        if manual_render_this_episode:
            env.render()
        base = env.unwrapped
        per_vehicle_counts: dict[tuple[str, int], int] = {}
        episode_sample_count = 0
        done = False
        progress = tqdm(
            total=int(args.max_steps_per_episode),
            desc=f"episode {episode_idx + 1}/{episodes}",
            unit="step",
            leave=True,
        )

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

                for agent_idx, vehicle in enumerate(vehicles):
                    if not bool(getattr(vehicle, "scene_collection_is_active", True)):
                        continue
                    vehicle_id = int(getattr(vehicle, "vehicle_ID", -1))
                    episode_name = str(getattr(base, "episode_name", episode_idx))
                    key = (episode_name, vehicle_id)
                    if max_per_vehicle > 0 and per_vehicle_counts.get(key, 0) >= max_per_vehicle:
                        continue

                    traj = getattr(vehicle, "scene_collection_full_traj", ())
                    if len(traj) == 0:
                        continue
                    if step_index >= len(traj) or not trajectory_row_is_active(traj[step_index]):
                        continue

                    observations.append(obs_agents[agent_idx].astype(np.float32, copy=False))
                    if trajectory_state_source == "simulated":
                        trajectory_states.append(simulated_trajectory_state_from_vehicle(vehicle))
                    else:
                        trajectory_states.append(trajectory_state_from_row_fast(traj[step_index]))
                    episode_names.append(episode_name)
                    vehicle_ids.append(vehicle_id)
                    timesteps.append(step_index)
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

                action = idle_action_for_env(env, len(vehicles))
                current_obs, _reward, terminated, truncated, _info = env.step(action)
                if manual_render_this_episode:
                    env.render()
                done = bool(terminated or truncated)
        finally:
            progress.close()
        collected_episodes += 1

    if not observations:
        raise RuntimeError("No expert observation/trajectory samples could be collected.")

    obs_arr = np.stack(observations, axis=0).astype(np.float32, copy=False)
    traj_arr = np.stack(trajectory_states, axis=0).astype(np.float32, copy=False)
    feature_arr = discriminator_features(obs_arr, traj_arr)
    return (
        {
            "observations": obs_arr,
            "trajectory_states": traj_arr,
            "features": feature_arr,
            "episode_names": np.asarray(episode_names, dtype=object),
            "vehicle_ids": np.asarray(vehicle_ids, dtype=np.int64),
            "timesteps": np.asarray(timesteps, dtype=np.int64),
        },
        collected_episodes,
    )


def save_expert_dataset(path: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )
    return output_path


def load_ps_traj_expert_dataset(path: str) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as data:
        required = {"observations", "trajectory_states", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")

        metadata = json.loads(str(data["metadata_json"].item()))
        observations = np.asarray(data["observations"], dtype=np.float32)
        trajectory_states = np.asarray(data["trajectory_states"], dtype=np.float32)
        if "features" in data.files:
            features = np.asarray(data["features"], dtype=np.float32)
        else:
            features = discriminator_features(observations, trajectory_states)

        if observations.ndim != 2:
            raise ValueError(f"Expected observations [N, obs_dim], got {observations.shape}.")
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


def maybe_wrap_record_video(env: gym.Env, args: argparse.Namespace) -> tuple[gym.Env, bool]:
    if not bool(args.save_video):
        return env, False

    out_dir = os.path.abspath(str(args.video_dir))
    os.makedirs(out_dir, exist_ok=True)

    try:
        import moviepy  # noqa: F401

        wrapped = RecordVideo(
            env,
            video_folder=out_dir,
            episode_trigger=lambda ep_idx: ep_idx == int(args.visualize_episode_index),
            name_prefix=str(args.video_prefix),
        )
        return wrapped, True
    except ModuleNotFoundError:
        warnings.warn(
            "moviepy is not installed; continuing without RecordVideo.",
            stacklevel=2,
        )
        return env, False


def main() -> None:
    args = parse_args()
    register_ngsim_env()

    render_mode = "rgb_array" if args.save_video else ("human" if args.visualize_episode else None)
    expert_env = make_expert_scene_env(args, render_mode=render_mode)
    expert_env, record_video = maybe_wrap_record_video(expert_env, args)
    try:
        arrays, collected_episodes = collect_expert_samples(expert_env, args)
    finally:
        expert_env.close()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "ps_traj_observation",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_id": ENV_ID,
        "scene": str(args.scene),
        "action_mode": str(args.expert_control_mode),
        "expert_test_mode": bool(str(args.expert_control_mode) != "teleport"),
        "episode_root": os.path.abspath(args.episode_root),
        "prebuilt_split": str(args.prebuilt_split),
        "num_samples": int(arrays["observations"].shape[0]),
        "observation_dim": int(arrays["observations"].shape[1]),
        "trajectory_state_shape": list(arrays["trajectory_states"].shape[1:]),
        "trajectory_state_dim": int(arrays["trajectory_states"].shape[1]),
        "feature_dim": int(arrays["features"].shape[1]),
        "max_episodes": int(args.max_episodes),
        "collected_episodes": int(collected_episodes),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "max_samples_per_vehicle": int(args.max_samples_per_vehicle),
        "controlled_vehicles": int(args.controlled_vehicles),
        "control_all_vehicles": bool(args.control_all_vehicles),
        "max_controlled_vehicles": int(args.max_controlled_vehicles),
        "max_surrounding": args.max_surrounding,
        "simulation_frequency": int(args.simulation_frequency),
        "policy_frequency": int(args.policy_frequency),
        "trajectory_state_source": str(args.trajectory_state_source),
        "observation_config": observation_config_from_args(args),
        "allow_idm": bool(args.allow_idm),
        "save_video": bool(args.save_video),
        "video_dir": os.path.abspath(str(args.video_dir)) if args.save_video else None,
    }
    output_path = save_expert_dataset(args.out, arrays, metadata)

    print(f"Saved PS trajectory expert dataset to: {output_path}")
    print(
        f"samples={metadata['num_samples']} "
        f"observation_dim={metadata['observation_dim']} "
        f"trajectory_state_shape={tuple(metadata['trajectory_state_shape'])} "
        f"feature_dim={metadata['feature_dim']}"
    )
    if record_video:
        print(f"Saved videos to: {os.path.abspath(str(args.video_dir))}")


if __name__ == "__main__":
    main()
