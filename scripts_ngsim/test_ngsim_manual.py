#!/usr/bin/env python3
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def get_terminal_action_discrete(env, cmd: str):
    cmd = cmd.strip().lower()

    cmd_to_act = {
        "w": "FASTER",
        "s": "SLOWER",
        "l": "STEER_LEFT",   # l = left steer
        "r": "STEER_RIGHT",  # r = right steer
        "": "IDLE",          # pressing Enter with no command = idle
        "i": "IDLE",
        "idle": "IDLE",
    }

    act = cmd_to_act.get(cmd)
    if act is None:
        return None

    if hasattr(env, "action_type") and hasattr(env.action_type, "actions_indexes"):
        return int(env.action_type.actions_indexes.get(
            act, env.action_type.actions_indexes["IDLE"]
        ))
    return act


def get_terminal_action_continuous(cmd: str):
    """
    Returns np.array([accel, steer], dtype=np.float32)
    Commands:
      w  -> +accel
      s  -> -accel
      l  -> -steer
      r  -> +steer
      wl -> +accel, -steer
      wr -> +accel, +steer
      sl -> -accel, -steer
      sr -> -accel, +steer
      enter / i -> idle
    """
    cmd = cmd.strip().lower()

    if cmd in ("", "i", "idle"):
        return np.array([0.0, 0.0], dtype=np.float32)

    accel = 0.0
    steer = 0.0

    if "w" in cmd:
        accel += 1.0
    if "s" in cmd:
        accel -= 1.0
    if "l" in cmd:
        steer -= 1.0
    if "r" in cmd:
        steer += 1.0

    if accel == 0.0 and steer == 0.0:
        return None

    return np.array([accel, steer], dtype=np.float32)


def print_help(continuous: bool):
    if continuous:
        print("""
Commands:
  w   accelerate
  s   brake
  l   steer left
  r   steer right
  wl  accelerate + left
  wr  accelerate + right
  sl  brake + left
  sr  brake + right
  i   idle
  <Enter> idle
  reset
  quit
""")
    else:
        print("""
Commands:
  w   faster
  s   slower
  l   steer left
  r   steer right
  i   idle
  <Enter> idle
  reset
  quit
""")


def run_manual_terminal(env, continuous: bool, max_steps=10_000):
    obs, info = env.reset()
    env.render()
    step_i = 0

    print("Manual terminal control started.")
    print_help(continuous)

    while step_i < max_steps:
        cmd = input(f"[step {step_i}] command> ").strip().lower()

        if cmd in ("quit", "q", "exit"):
            break

        if cmd in ("help", "h", "?"):
            print_help(continuous)
            continue

        if cmd in ("reset",):
            obs, info = env.reset()
            env.render()
            step_i = 0
            print("Episode reset.")
            continue

        if continuous:
            action = get_terminal_action_continuous(cmd)
        else:
            action = get_terminal_action_discrete(env, cmd)

        if action is None:
            print("Unknown command. Type 'help' for valid commands.")
            continue

        print("action:", action)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        v = env.unwrapped.vehicle
        print(
            f"t={step_i:04d} "
            f"x={v.position[0]:.1f} y={v.position[1]:.1f} "
            f"spd={v.speed:.2f} hdg={float(v.heading):.2f} "
            f"tgt_lane={getattr(v, 'target_lane_index', None)} "
            f"lat_off={getattr(v, 'lateral_offset', np.nan):.2f} "
            f"crash={bool(v.crashed)} "
            f"reward={reward:.4f} term={terminated} trunc={truncated}"
        )

        step_i += 1

        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            obs, info = env.reset()
            env.render()
            step_i = 0
            print("Auto-reset.")

    env.close()
    print("Exited manual terminal control.")


def main():
    DISCRETE_ACTION_TYPE_NAME = "DiscreteSteerMetaAction"

    base_cfg = {
        "scene": "us-101",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "action": {
            "type": DISCRETE_ACTION_TYPE_NAME,
            "target_speeds": list(np.arange(0.0, 35.0 + 1e-6, 2.0)),
        },
        "show_trajectories": True,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 800,
        "screen_height": 300,
        "scaling": 2.0,
        "offscreen_rendering": False,
        "ego_vehicle_ID": None,
        "episode_root": "highway_env/data/processed_10s",
        "replay_period": None,
        "max_surrounding": 0,
        "expert_test_mode": False,
    }

    env = gym.make("NGSim-US101-v0", render_mode="human", config=base_cfg)
    run_manual_terminal(env, continuous=False)


if __name__ == "__main__":
    main()
