"""Create a Japanese NGSIM expert replay video with sensor observables.

The script records a single expert-controlled vehicle and renders a composite
MP4: the environment RGB frame on top, plus LiDAR and lane-camera observables
below. It is intended for visual sensor validation and does not modify the
environment or training logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from highway_env.imitation.expert_dataset import register_ngsim_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--episode-name", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--ego-id", type=int, default=None)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--cells", type=int, default=96)
    parser.add_argument("--range", type=float, default=60.0, dest="maximum_range")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sensor_validation/japanese_expert_sensor_validation.mp4"),
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> dict[str, Any]:
    simulation_period: int | dict[str, str]
    if args.episode_name:
        simulation_period = {"episode_name": str(args.episode_name)}
    else:
        simulation_period = int(args.episode_index)

    return {
        "scene": "japanese",
        "observation": {
            "type": "LidarCameraObservations",
            "lidar": {
                "cells": int(args.cells),
                "maximum_range": float(args.maximum_range),
                "normalize": False,
            },
            "camera": {
                "cells": 31,
                "maximum_range": float(args.maximum_range),
                "field_of_view": np.pi / 2,
                "normalize": False,
            },
        },
        "action": {"type": "DiscreteSteerMetaAction"},
        "action_mode": "teleport",
        "scene_dataset_collection_mode": True,
        "expert_test_mode": False,
        "discrete_expert_policy": "planner",
        "episode_root": "highway_env/data/processed_20s",
        "prebuilt_split": str(args.split),
        "simulation_period": simulation_period,
        "ego_vehicle_ID": args.ego_id,
        "percentage_controlled_vehicles": 1,
        "control_all_vehicles": False,
        "clip_controlled_vehicles_to_available": True,
        "max_surrounding": "all",
        "show_trajectories": True,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "max_episode_steps": int(args.steps),
        "truncate_to_trajectory_length": True,
        "allow_idm": True,
        "offscreen_rendering": True,
        "screen_width": 1000,
        "screen_height": 320,
        "scaling": 5.0,
        "centering_position": [0.35, 0.5],
    }


def first_agent_observation(obs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lidar, camera, ego-state arrays for the first controlled agent."""
    agent_obs = obs[0] if isinstance(obs, tuple) and len(obs) == 1 else obs
    if not isinstance(agent_obs, tuple) or len(agent_obs) < 3:
        raise TypeError(f"Unexpected observation structure: {type(obs)!r}")
    lidar, camera, ego_state = agent_obs[:3]
    return np.asarray(lidar), np.asarray(camera), np.asarray(ego_state)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_lidar_panel(
    lidar: np.ndarray,
    maximum_range: float,
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    center = (width // 2, height // 2 + 8)
    radius = int(min(width, height) * 0.42)

    for frac in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(panel, center, int(radius * frac), (60, 60, 60), 1, cv2.LINE_AA)

    cells = max(len(lidar), 1)
    angles = np.linspace(0.0, 2.0 * np.pi, cells, endpoint=False)
    distances = np.clip(lidar[:, 0].astype(float), 0.0, maximum_range)
    rel_speeds = lidar[:, 1].astype(float) if lidar.shape[1] > 1 else np.zeros(cells)

    for angle, distance, rel_speed in zip(angles, distances, rel_speeds):
        r = int(radius * distance / maximum_range)
        point = (
            int(center[0] + r * np.cos(angle)),
            int(center[1] + r * np.sin(angle)),
        )
        color = (90, 220, 120) if abs(rel_speed) < 0.5 else (80, 170, 250)
        cv2.line(panel, center, point, color, 1, cv2.LINE_AA)
        cv2.circle(panel, point, 2, (230, 230, 230), -1, cv2.LINE_AA)

    cv2.arrowedLine(panel, center, (center[0] + radius, center[1]), (250, 250, 250), 2, cv2.LINE_AA)
    draw_text(panel, "LiDAR returns: distance and relative radial speed", (16, 24))
    draw_text(panel, f"min {distances.min():.1f} m   max {distances.max():.1f} m", (16, height - 16), 0.5)
    return panel


def draw_camera_panel(
    camera: np.ndarray,
    ego_state: np.ndarray,
    maximum_range: float,
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    origin = (width // 2, height - 34)
    scale = min(width * 0.42 / maximum_range, height * 0.78 / maximum_range)

    cv2.line(panel, (origin[0], 16), (origin[0], height - 12), (55, 55, 55), 1, cv2.LINE_AA)
    cv2.line(panel, (10, origin[1]), (width - 10, origin[1]), (55, 55, 55), 1, cv2.LINE_AA)

    if camera.ndim == 2 and camera.shape[1] >= 3:
        visible = camera[:, 0] > 0.0
        points = camera[visible]
        for _, x_value, y_value in points:
            px = int(origin[0] + y_value * scale)
            py = int(origin[1] - x_value * scale)
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(panel, (px, py), 4, (255, 210, 90), -1, cv2.LINE_AA)

    vehicle_width = 18
    vehicle_length = 32
    cv2.rectangle(
        panel,
        (origin[0] - vehicle_width // 2, origin[1] - vehicle_length // 2),
        (origin[0] + vehicle_width // 2, origin[1] + vehicle_length // 2),
        (80, 180, 255),
        -1,
    )
    cv2.arrowedLine(panel, origin, (origin[0], origin[1] - 42), (255, 255, 255), 2, cv2.LINE_AA)

    speed = float(ego_state[0]) if ego_state.size else 0.0
    heading = float(ego_state[1]) if ego_state.size > 1 else 0.0
    draw_text(panel, "Lane-camera observable: road boundary points in ego frame", (16, 24))
    draw_text(panel, f"ego speed {speed:.1f} m/s   heading {heading:.3f} rad", (16, height - 16), 0.5)
    return panel


def compose_frame(
    env_frame: np.ndarray,
    lidar: np.ndarray,
    camera: np.ndarray,
    ego_state: np.ndarray,
    maximum_range: float,
    status: str,
) -> np.ndarray:
    env_frame = np.asarray(env_frame, dtype=np.uint8)
    height, width = env_frame.shape[:2]
    sensor_height = 260
    lidar_panel = draw_lidar_panel(lidar, maximum_range, width // 2, sensor_height)
    camera_panel = draw_camera_panel(camera, ego_state, maximum_range, width - width // 2, sensor_height)
    bottom = np.concatenate([lidar_panel, camera_panel], axis=1)
    frame = np.concatenate([env_frame, bottom], axis=0)
    draw_text(frame, status, (16, 30), 0.65, (255, 255, 255))
    return frame


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    register_ngsim_env()
    env = gym.make("NGSim-US101-v0", config=make_config(args), render_mode="rgb_array")

    try:
        obs, _ = env.reset(seed=0)
        first_frame = env.render()
        lidar, camera, ego_state = first_agent_observation(obs)
        status = (
            f"Japanese expert replay | episode {env.unwrapped.episode_name} | "
            f"ego {env.unwrapped.vehicle.vehicle_ID}"
        )
        frame = compose_frame(first_frame, lidar, camera, ego_state, args.maximum_range, status)

        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            (frame.shape[1], frame.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {args.output}")

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        for step in range(1, int(args.steps)):
            obs, _reward, terminated, truncated, info = env.step(0)
            env_frame = env.render()
            lidar, camera, ego_state = first_agent_observation(obs)
            action = "recorded replay"
            status = (
                f"Japanese expert replay | episode {env.unwrapped.episode_name} | "
                f"ego {env.unwrapped.vehicle.vehicle_ID} | step {step} | action {action}"
            )
            frame = compose_frame(env_frame, lidar, camera, ego_state, args.maximum_range, status)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated or truncated:
                break

        writer.release()
        print(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
