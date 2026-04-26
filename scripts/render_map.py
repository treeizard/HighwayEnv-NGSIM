import argparse
import os
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from PIL import Image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")

def _iter_lanes(road):
    for lane_dict in road.network.graph.values():
        for lanes in lane_dict.values():
            for lane in lanes:
                yield lane


def _road_bounds(road, samples_per_lane=200, margin_m=20.0):
    points = []
    for lane in _iter_lanes(road):
        width = float(getattr(lane, "width", lane.width_at(0.0)))
        for lateral in (-0.5 * width, 0.5 * width):
            for s in np.linspace(0.0, float(lane.length), samples_per_lane):
                points.append(lane.position(s, lateral))

    pts = np.asarray(points, dtype=float)
    min_xy = pts.min(axis=0) - margin_m
    max_xy = pts.max(axis=0) + margin_m
    return min_xy, max_xy


def _fit_view_to_road(base_env, width, height, margin_ratio=0.92):
    min_xy, max_xy = _road_bounds(base_env.road)
    span = np.maximum(max_xy - min_xy, 1.0)
    center = 0.5 * (min_xy + max_xy)
    scaling = float(min(width / span[0], height / span[1]) * margin_ratio)

    base_env.config["centering_position"] = [0.5, 0.5]
    base_env.config["scaling"] = scaling

    # Ensure the viewer tracks the topology center instead of an ego vehicle.
    if base_env.viewer is not None:
        base_env.viewer.observer_vehicle = SimpleNamespace(position=center)
        base_env.viewer.sim_surface.scaling = scaling
        base_env.viewer.sim_surface.centering_position = [0.5, 0.5]

    return center, scaling


def save_static_map_with_api(env, out_path="us101_static.png"):
    """Render the full road topology, fit it in frame, and save to PNG."""
    base_env = env.unwrapped
    if base_env.render_mode is None:
        base_env.render_mode = "rgb_array"
    base_env._create_road()
    base_env.road.vehicles = []
    base_env.vehicle = None

    width = int(base_env.config["screen_width"])
    height = int(base_env.config["screen_height"])
    original_observation_type = getattr(base_env, "observation_type", None)

    try:
        # Disable LiDAR/camera overlays for a clean topology export.
        base_env.observation_type = None

        # First render initializes the viewer; the second uses the fitted camera.
        _ = base_env.render()
        center, scaling = _fit_view_to_road(base_env, width=width, height=height)
        frame = base_env.render()
    finally:
        base_env.observation_type = original_observation_type

    Image.fromarray(frame).save(out_path)
    print(f"Saved static map to {out_path}")
    print(f"camera_center={center.tolist()}")
    print(f"camera_scaling={scaling:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a clean full-topology PNG for an NGSIM road scene."
    )
    parser.add_argument(
        "--scene",
        choices=("us-101", "japanese"),
        default="us-101",
        help="Road topology to render.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to '<scene>_static_full_topology.png'.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output image height in pixels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    default_sizes = {
        "us-101": (2400, 900),
        "japanese": (2200, 300),
    }
    default_width, default_height = default_sizes[args.scene]
    width = args.width if args.width is not None else default_width
    height = args.height if args.height is not None else default_height
    out_path = args.out or f"{args.scene.replace('-', '_')}_static_full_topology.png"

    cfg = {
        "scene": args.scene,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction"},
        "show_trajectories": False,
        "centering_position": [0.5, 0.5],

        # Renderer settings for a nice static export
        "offscreen_rendering": True,   # <- no window needed
        "screen_width": width,
        "screen_height": height,
        "scaling": 2.0,
    }

    # Pass config at construction time so scene-dependent internals are initialized correctly.
    env = gym.make("NGSim-US101-v0", config=cfg, render_mode="rgb_array")

    save_static_map_with_api(env, out_path)

    # (Optional) run a short rollout or just close
    env.close()

if __name__ == "__main__":
    main()
