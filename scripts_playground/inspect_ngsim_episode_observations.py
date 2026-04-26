#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np
from gymnasium.envs.registration import register, registry

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import ENV_ID, build_env_config  # noqa: E402
from highway_env.ngsim_utils.data.ego_trajectory import (  # noqa: E402
    get_ego_dimensions,
    load_ego_trajectory,
)
from highway_env.ngsim_utils.data.prebuilt import load_prebuilt_data  # noqa: E402
from highway_env.ngsim_utils.data.trajectory_gen import trajectory_row_is_active  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one NGSIM episode/vehicle by plotting LiDAR, lane-camera "
            "observations, vehicle state/size, and the raw expert trajectory."
        )
    )
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--episode-name", default=None, help="Example: t1118847759700")
    parser.add_argument("--vehicle-id", type=int, default=None)
    parser.add_argument("--step", type=int, default=0, help="Simulation step to inspect.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="debug/ngsim_episode_observation_inspection")
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--camera-cells", type=int, default=21)
    parser.add_argument("--maximum-range", type=float, default=64.0)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--action-mode", choices=["continuous", "discrete"], default="continuous")
    parser.add_argument(
        "--controlled-vehicle-min-occupancy",
        type=float,
        default=0.8,
        help="Set to 0.0 to inspect older sparse episode stamps.",
    )
    return parser.parse_args()


def register_env() -> None:
    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def observation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "LidarCameraObservations",
        "lidar": {
            "cells": int(args.cells),
            "maximum_range": float(args.maximum_range),
            "normalize": True,
        },
        "camera": {
            "cells": int(args.camera_cells),
            "maximum_range": float(args.maximum_range),
            "field_of_view": np.pi / 2,
            "normalize": True,
        },
    }


def choose_episode_and_vehicle(args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    _prebuilt_dir, valid_ids_by_episode, traj_all_by_episode, episodes = load_prebuilt_data(
        args.episode_root,
        args.scene,
        args.prebuilt_split,
        min_occupancy=float(args.controlled_vehicle_min_occupancy),
        cache={},
    )
    if not episodes:
        raise RuntimeError("No prebuilt episodes were found.")

    episode_name = str(args.episode_name or episodes[0])
    if episode_name not in traj_all_by_episode:
        raise ValueError(f"Episode {episode_name!r} was not found.")

    valid_ids = [int(v) for v in valid_ids_by_episode.get(episode_name, [])]
    if not valid_ids:
        raise RuntimeError(
            f"Episode {episode_name} has no valid controlled vehicles at "
            f"min_occupancy={args.controlled_vehicle_min_occupancy}."
        )

    vehicle_id = int(args.vehicle_id if args.vehicle_id is not None else valid_ids[0])
    if vehicle_id not in traj_all_by_episode[episode_name]:
        raise ValueError(f"Vehicle {vehicle_id} was not found in episode {episode_name}.")
    if vehicle_id not in valid_ids:
        print(
            f"Warning: vehicle {vehicle_id} is not in the refined valid-id list "
            f"for min_occupancy={args.controlled_vehicle_min_occupancy}."
        )

    return episode_name, vehicle_id, traj_all_by_episode[episode_name][vehicle_id]


def make_env(args: argparse.Namespace, episode_name: str, vehicle_id: int) -> gym.Env:
    obs_cfg = observation_config(args)
    action_type = "ContinuousAction" if args.action_mode == "continuous" else "DiscreteSteerMetaAction"
    cfg = build_env_config(
        scene=args.scene,
        action_mode=args.action_mode,
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        percentage_controlled_vehicles=1,
        control_all_vehicles=False,
        max_surrounding=args.max_surrounding,
        observation_config=obs_cfg,
        simulation_frequency=10,
        policy_frequency=10,
        max_episode_steps=max(300, int(args.step) + 2),
        seed=int(args.seed),
        simulation_period={"episode_name": episode_name},
        ego_vehicle_id=int(vehicle_id),
        scene_dataset_collection_mode=False,
        allow_idm=False,
    )
    cfg["observation"] = obs_cfg
    cfg["action"] = {"type": action_type}
    cfg["expert_test_mode"] = True
    cfg["controlled_vehicle_min_occupancy"] = float(args.controlled_vehicle_min_occupancy)
    cfg["offscreen_rendering"] = True
    return gym.make(ENV_ID, config=cfg)


def dummy_action(env: gym.Env) -> Any:
    action_space = env.action_space
    if hasattr(action_space, "shape") and action_space.shape is not None:
        return np.zeros(action_space.shape, dtype=np.float32)
    if hasattr(action_space, "n"):
        return 0
    return action_space.sample()


def unpack_observation(obs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(obs, tuple) or len(obs) != 3:
        raise TypeError(f"Expected (lidar, lane_camera, ego_state), got {type(obs)}.")
    lidar, lane_camera, ego_state = obs
    return (
        np.asarray(lidar, dtype=np.float32),
        np.asarray(lane_camera, dtype=np.float32),
        np.asarray(ego_state, dtype=np.float32),
    )


def denormalize_lidar(lidar: np.ndarray, maximum_range: float) -> np.ndarray:
    lidar_m = np.asarray(lidar, dtype=np.float32).copy()
    lidar_m[:, 0] *= float(maximum_range)
    lidar_m[:, 1] *= float(maximum_range)
    return lidar_m


def denormalize_lane_camera(camera: np.ndarray, maximum_range: float) -> np.ndarray:
    camera_m = np.asarray(camera, dtype=np.float32).copy()
    camera_m[:, 1:] *= float(maximum_range)
    return camera_m


def plot_observations(
    *,
    lidar: np.ndarray,
    lane_camera: np.ndarray,
    ego_state: np.ndarray,
    maximum_range: float,
    output_path: str,
    title: str,
) -> None:
    lidar_m = denormalize_lidar(lidar, maximum_range)
    camera_m = denormalize_lane_camera(lane_camera, maximum_range)

    angles = (np.arange(lidar_m.shape[0]) + 0.5) * (2.0 * np.pi / lidar_m.shape[0])
    present = camera_m[:, 0] > 0.5

    fig = plt.figure(figsize=(15, 5), dpi=150)
    polar_ax = fig.add_subplot(1, 3, 1, projection="polar")
    lane_ax = fig.add_subplot(1, 3, 2)
    text_ax = fig.add_subplot(1, 3, 3)

    polar_ax.plot(angles, lidar_m[:, 0], linewidth=1.2)
    polar_ax.scatter(angles, lidar_m[:, 0], c=lidar_m[:, 1], s=10, cmap="coolwarm")
    polar_ax.set_ylim(0.0, maximum_range)
    polar_ax.set_title("LiDAR distance / relative speed")

    lane_ax.scatter(camera_m[~present, 1], camera_m[~present, 2], s=14, color="0.8", label="empty")
    lane_ax.scatter(camera_m[present, 1], camera_m[present, 2], s=28, color="tab:green", label="lane point")
    lane_ax.scatter([0.0], [0.0], marker="x", color="black", s=50, label="ego")
    lane_ax.set_xlim(-5.0, maximum_range)
    lane_ax.set_ylim(-maximum_range / 2.0, maximum_range / 2.0)
    lane_ax.set_aspect("equal", adjustable="box")
    lane_ax.grid(True, linewidth=0.3)
    lane_ax.set_xlabel("ego-frame x forward (m)")
    lane_ax.set_ylabel("ego-frame y left (m)")
    lane_ax.set_title("Lane-camera detections")
    lane_ax.legend(loc="upper right", fontsize=8)

    text_ax.axis("off")
    text_ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                title,
                "",
                "Observation state",
                f"ego speed:   {ego_state[0]:.3f} m/s",
                f"ego heading: {ego_state[1]:.6f} rad",
                f"ego width:   {ego_state[2]:.3f} m",
                f"ego length:  {ego_state[3]:.3f} m",
                "",
                f"lidar cells: {lidar.shape[0]}",
                f"lane cells:  {lane_camera.shape[0]}",
                f"lane hits:   {int(np.count_nonzero(present))}",
                f"min lidar distance: {float(np.min(lidar_m[:, 0])):.3f} m",
            ]
        ),
        va="top",
        family="monospace",
        fontsize=9,
    )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_trajectory(
    *,
    trajectory: np.ndarray,
    active_mask: np.ndarray,
    step: int,
    vehicle_position: np.ndarray,
    output_path: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax_xy, ax_speed = axes
    active_traj = trajectory[active_mask]

    ax_xy.plot(trajectory[:, 0], trajectory[:, 1], color="0.75", linewidth=1.0, label="raw padded")
    if len(active_traj):
        ax_xy.plot(active_traj[:, 0], active_traj[:, 1], color="tab:blue", linewidth=1.5, label="active")
    if 0 <= step < len(trajectory):
        ax_xy.scatter(trajectory[step, 0], trajectory[step, 1], color="tab:orange", s=50, label=f"raw step {step}")
    ax_xy.scatter(vehicle_position[0], vehicle_position[1], marker="x", color="black", s=60, label="env vehicle")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, linewidth=0.3)
    ax_xy.legend(fontsize=8)
    ax_xy.set_title("Raw expert trajectory")

    t = np.arange(len(trajectory))
    ax_speed.plot(t, trajectory[:, 2], color="tab:purple", linewidth=1.2)
    ax_speed.axvline(step, color="tab:orange", linestyle="--", linewidth=1.0)
    ax_speed.set_xlabel("trajectory row / step")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.grid(True, linewidth=0.3)
    ax_speed.set_title("Expert speed trace")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    register_env()
    os.makedirs(args.outdir, exist_ok=True)

    episode_name, vehicle_id, ego_rec = choose_episode_and_vehicle(args)
    raw_traj = np.asarray(load_ego_trajectory(ego_rec, args.scene), dtype=np.float32)
    active_mask = np.asarray([trajectory_row_is_active(row) for row in raw_traj], dtype=bool)
    if args.step < 0:
        raise ValueError("--step must be non-negative.")
    if args.step >= len(raw_traj):
        raise ValueError(f"--step {args.step} is outside trajectory length {len(raw_traj)}.")

    env = make_env(args, episode_name, vehicle_id)
    try:
        obs, _info = env.reset(seed=int(args.seed))
        for _ in range(int(args.step)):
            obs, _reward, terminated, truncated, _info = env.step(dummy_action(env))
            if terminated or truncated:
                break

        lidar, lane_camera, ego_state = unpack_observation(obs)
        vehicle = env.unwrapped.vehicle
        position = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=float)
        velocity = np.asarray(getattr(vehicle, "velocity", np.zeros(2)), dtype=float)
        rec_length, rec_width = get_ego_dimensions(ego_rec, 3.280839895, args.scene)

        stem = f"{args.scene}_{episode_name}_veh{vehicle_id}_step{args.step}"
        obs_path = os.path.abspath(os.path.join(args.outdir, f"{stem}_observations.png"))
        traj_path = os.path.abspath(os.path.join(args.outdir, f"{stem}_trajectory.png"))
        title = f"{args.scene} {episode_name} vehicle {vehicle_id} step {args.step}"

        plot_observations(
            lidar=lidar,
            lane_camera=lane_camera,
            ego_state=ego_state,
            maximum_range=float(args.maximum_range),
            output_path=obs_path,
            title=title,
        )
        plot_trajectory(
            trajectory=raw_traj,
            active_mask=active_mask,
            step=int(args.step),
            vehicle_position=position,
            output_path=traj_path,
            title=title,
        )

        print("Selected sample")
        print(f"episode_name: {episode_name}")
        print(f"vehicle_id: {vehicle_id}")
        print(f"step: {args.step}")
        print("")
        print("Vehicle state from environment")
        print(f"position_xy_m: {position.tolist()}")
        print(f"velocity_xy_mps: {velocity.tolist()}")
        print(f"speed_mps: {float(getattr(vehicle, 'speed', 0.0)):.6f}")
        print(f"heading_rad: {float(getattr(vehicle, 'heading', 0.0)):.6f}")
        print(f"vehicle_WIDTH_m: {float(getattr(vehicle, 'WIDTH', 0.0)):.6f}")
        print(f"vehicle_LENGTH_m: {float(getattr(vehicle, 'LENGTH', 0.0)):.6f}")
        print("")
        print("Vehicle size from trajectory metadata")
        print(f"record_width_m: {float(rec_width):.6f}")
        print(f"record_length_m: {float(rec_length):.6f}")
        print("")
        print("Observation arrays")
        print(f"lidar_shape: {lidar.shape}")
        print(f"lane_camera_shape: {lane_camera.shape}")
        print(f"ego_state_speed_heading_width_length: {ego_state.tolist()}")
        print("")
        print("Raw expert trajectory")
        print(f"trajectory_shape: {raw_traj.shape}")
        print(f"active_rows: {int(np.count_nonzero(active_mask))}")
        print(f"row_at_step_x_y_speed_lane: {raw_traj[int(args.step)].tolist()}")
        print("")
        print(f"Saved observation plot: {obs_path}")
        print(f"Saved trajectory plot: {traj_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
