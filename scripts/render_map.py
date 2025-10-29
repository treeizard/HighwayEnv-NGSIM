import numpy as np
import gymnasium as gym
import os, sys
from gymnasium.envs.registration import register
from PIL import Image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")

def save_static_map_with_api(env, out_path="us101_static.png", hide_traffic=True):
    """Render one frame with highway_env's renderer and save to PNG."""
    # Optionally hide vehicles to get a clean lane map
    saved = None
    if hide_traffic and hasattr(env.unwrapped, "road"):
        saved = list(env.unwrapped.road.vehicles)
        env.unwrapped.road.vehicles = []

    # Grab a frame from the renderer
    frame = env.render() if env.render_mode == "rgb_array" else env.render(mode="rgb_array")
    Image.fromarray(frame).save(out_path)
    print(f"Saved static map to {out_path}")

    # Restore vehicles
    if saved is not None:
        env.unwrapped.road.vehicles = saved

def main():
    cfg = {
        "scene": "us-101",
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction"},
        "show_trajectories": False,

        # Renderer settings for a nice static export
        "offscreen_rendering": True,   # <- no window needed
        "screen_width": 1600,
        "screen_height": 450,
        "scaling": 2.0,                # zoom level (increase if you want a closer view)
    }

    # Use rgb_array so render() returns an image directly
    env = gym.make("NGSim-US101-v0", render_mode="rgb_array")
    env.unwrapped.configure(cfg)

    # Build road/vehicles
    obs, info = env.reset(seed=42)

    # 👉 Save a static lane map using the built-in renderer
    save_static_map_with_api(env, "us101_static.png", hide_traffic=True)

    # (Optional) run a short rollout or just close
    env.close()

if __name__ == "__main__":
    main()
