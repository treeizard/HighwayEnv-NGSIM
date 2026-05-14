import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from highway_env.envs.ngsim_env import NGSimEnv
from highway_env.ngsim_utils.road.gen_road import (
    create_japanese_road,
    create_ngsim_101_road,
)
from highway_env.ngsim_utils.vehicles.ego import EgoVehicle
from highway_env.road.road import Road


def make_env(scene: str, lane_index, *, start_offset_m: float) -> tuple[NGSimEnv, EgoVehicle]:
    env = NGSimEnv.__new__(NGSimEnv)
    env.config = NGSimEnv.default_config()
    env.config.update(
        {
            "scene": scene,
            "simulation_frequency": 10,
            "policy_frequency": 10,
            "offscreen_rendering": True,
            "screen_width": 1200,
            "screen_height": 608,
            "scaling": 6.5,
            "show_trajectories": False,
            "manual_control": False,
            "real_time_rendering": False,
            "render_agent": False,
            "complete_controlled_vehicles_at_road_end": True,
            "crash_controlled_vehicles_offroad": True,
        }
    )
    env.scene = scene
    env.control_mode = "continuous"
    env.render_mode = "rgb_array"
    env.viewer = None
    env.enable_auto_render = False
    env.steps = 0
    env.time = 0.0
    env.done = False
    env._record_video_wrapper = None
    env.observation_type = None
    env.action_type = None
    env.ego_ids = [1]

    net = create_japanese_road() if scene == "japanese" else create_ngsim_101_road()
    env.net = net
    env.road = Road(
        network=net,
        np_random=np.random.RandomState(0),
        record_history=False,
    )

    lane = net.get_lane(lane_index)
    start_s = max(0.0, float(lane.length) - float(start_offset_m))
    vehicle = EgoVehicle(
        road=env.road,
        position=lane.position(start_s, 0.0),
        heading=lane.heading_at(start_s),
        speed=12.0,
        target_speed=12.0,
        control_mode="continuous",
    )
    vehicle.vehicle_ID = 1
    vehicle.target_lane_index = lane_index
    vehicle.lane_index = lane_index
    vehicle.lane = lane
    env.vehicle = vehicle
    env.controlled_vehicles = [vehicle]
    env.road.vehicles.append(vehicle)
    return env, vehicle


def record(scene: str, lane_index, path: Path, *, steps: int) -> dict:
    import imageio.v2 as imageio

    env, vehicle = make_env(scene, lane_index, start_offset_m=18.0)
    frames = []
    first_completed_step = None
    try:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        for step in range(1, steps + 1):
            if not bool(getattr(vehicle, "completed", False)):
                vehicle.act({"steering": 0.0, "acceleration": 0.0})
                env.road.step(1.0 / env.config["simulation_frequency"])
                env._complete_road_end_controlled_vehicles()
                env._crash_offroad_controlled_vehicles()
                env._prune_removed_vehicles()
                env.steps += 1

            if bool(getattr(vehicle, "completed", False)) and first_completed_step is None:
                first_completed_step = step

            frame = env.render()
            if frame is not None:
                frames.append(frame)

        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(path), fps=10) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))

        return {
            "path": str(path.resolve()),
            "frames": len(frames),
            "first_completed_step": first_completed_step,
            "completed": bool(getattr(vehicle, "completed", False)),
            "crashed": bool(getattr(vehicle, "crashed", False)),
            "road_vehicle_count": len(env.road.vehicles),
            "position": [float(vehicle.position[0]), float(vehicle.position[1])],
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="logs/road_end_completion")
    parser.add_argument("--steps", type=int, default=34)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = [
        record(
            "us-101",
            ("s3", "s4", 2),
            out_dir / "us101_road_end_completion.mp4",
            steps=int(args.steps),
        ),
        record(
            "us-101",
            ("s3", "merge_out", 0),
            out_dir / "us101_merge_exit_completion.mp4",
            steps=int(args.steps),
        ),
        record(
            "japanese",
            ("c", "d", 0),
            out_dir / "japanese_road_end_completion.mp4",
            steps=int(args.steps),
        ),
    ]
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
