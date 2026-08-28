import argparse
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from PIL import Image, ImageDraw, ImageFont


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


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_static_map_with_api(
    env,
    out_path="us101_static.png",
    *,
    road_network=None,
    title=None,
    subtitle=None,
):
    """Render the full road topology, fit it in frame, and save to PNG."""
    from highway_env.road.road import Road

    base_env = env.unwrapped
    if base_env.render_mode is None:
        base_env.render_mode = "rgb_array"
    if road_network is None:
        base_env._create_road()
    else:
        base_env.road = Road(
            network=road_network,
            np_random=getattr(base_env, "np_random", None),
        )
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

    if not np.any(frame):
        raise RuntimeError(
            "Native HighwayEnv viewer returned an all-black frame. "
            "This fork disables drawing when SDL_VIDEODRIVER=dummy; unset that "
            "variable for an offscreen topology export."
        )
    rendered = Image.fromarray(frame)
    if title or subtitle:
        caption_height = 92
        captioned = Image.new(
            "RGB", (rendered.width, rendered.height + caption_height), "white"
        )
        captioned.paste(rendered, (0, caption_height))
        draw = ImageDraw.Draw(captioned)
        if title:
            draw.text((24, 12), str(title), fill=(18, 18, 18), font=_font(28))
        if subtitle:
            draw.text((24, 52), str(subtitle), fill=(75, 75, 75), font=_font(18))
        rendered = captioned
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rendered.save(out_path)
    print(f"Saved static map to {out_path}")
    print(f"camera_center={center.tolist()}")
    print(f"camera_scaling={scaling:.4f}")
    print("equal_xy_scaling=true")

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
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--road-geometry-v3",
        type=Path,
        default=None,
        help="Render a manifest RoadGeometryV3 through the native HighwayEnv viewer.",
    )
    source_group.add_argument(
        "--japanese-source-geometry",
        type=Path,
        default=None,
        help=(
            "Render a historical source-derived Japanese geometry through the "
            "legacy native road constructor."
        ),
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
    parser.add_argument("--title", default=None, help="Optional caption above the map.")
    parser.add_argument(
        "--subtitle", default=None, help="Optional disclosure caption above the map."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    default_sizes = {
        "us-101": (2400, 900),
        "japanese": (2200, 300),
    }
    default_width, default_height = default_sizes[args.scene]
    road_network = None
    source_label = f"legacy scene={args.scene}"
    if args.road_geometry_v3 is not None:
        from highway_env.ngsim_utils.road.manifest_road import (
            RoadGeometryV3,
            build_road_network,
        )

        geometry = RoadGeometryV3.from_source(args.road_geometry_v3)
        road_network = build_road_network(geometry)
        source_label = (
            f"RoadGeometryV3 environment_id={geometry.environment_id or 'unspecified'}"
        )
    elif args.japanese_source_geometry is not None:
        from highway_env.ngsim_utils.road.gen_road import create_japanese_road

        road_network = create_japanese_road(args.japanese_source_geometry)
        source_label = "historical source-derived Japanese geometry"

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

    save_static_map_with_api(
        env,
        out_path,
        road_network=road_network,
        title=args.title,
        subtitle=args.subtitle or source_label,
    )

    # (Optional) run a short rollout or just close
    env.close()

if __name__ == "__main__":
    main()
