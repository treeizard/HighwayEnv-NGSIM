#!/usr/bin/env python3
"""
Collect an expert dataset from NGSimEnv when `expert_test_mode=True` makes the environment
apply expert actions internally (i.e., the action passed into env.step() is ignored/overridden).

This script:
- Uses your NGSim-US101-v0 registration and config
- Steps the environment with a dummy action
- Reads the actually applied "expert" action from `info`
- Saves (obs, acts, next_obs, dones, ep_id) into a compressed NPZ

IMPORTANT:
- NGSimEnv.step() MUST put the applied/expert action into info["expert_action"]
  (or info["applied_action"] as a fallback).
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation

# ----------------------------------------------------------------------
# Import project root so `highway_env` is importable
# ----------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ----------------------------------------------------------------------
# Register NGSimEnv
# ----------------------------------------------------------------------
register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")


# ----------------------------------------------------------------------
# Environment construction
# ----------------------------------------------------------------------
def make_env(config: Dict[str, Any]) -> gym.Env:
    """
    Construct NGSim env using gym.make so config is passed into NGSimEnv.__init__.

    We do NOT set render_mode (defaults to None). No video recording is used.
    """
    env = gym.make(
        "NGSim-US101-v0",
        config=config,
    )
    # Flatten obs to 1D float32 vector
    env = FlattenObservation(env)
    return env


# ----------------------------------------------------------------------
# Utilities for extracting expert/applied action
# ----------------------------------------------------------------------
def _extract_applied_action(info: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Extract the action actually applied by the environment.

    Prefer info["expert_action"], fall back to info["applied_action"].
    Raise a clear error if neither exists.
    """
    if info is None:
        info = {}

    #if "expert_action" in info:
        #return np.asarray(info["expert_action"], dtype=np.float32).ravel()
    if "applied_action" in info:
        return np.asarray(info["applied_action"], dtype=np.float32).ravel()

    raise RuntimeError(
        "expert_test_mode=True but env.step() did not expose the applied/expert action.\n"
        "Please modify NGSimEnv.step() to set either:\n"
        "  info['expert_action'] = expert_action\n"
        "or\n"
        "  info['applied_action'] = action_actually_applied\n"
    )


def _dummy_action(env: gym.Env) -> np.ndarray:
    """
    Create a valid dummy action; will be ignored if expert_test_mode=True.

    For ContinuousAction, the action space is a Box; we just construct zeros of the
    appropriate shape.
    """
    shape = getattr(env.action_space, "shape", None)
    if shape is None:
        # Fallback: sample once and zero-like it
        a = env.action_space.sample()
        return np.zeros_like(np.asarray(a, dtype=np.float32)).ravel()
    return np.zeros(shape, dtype=np.float32)


# ----------------------------------------------------------------------
# Rollout collection
# ----------------------------------------------------------------------
def collect_expert_rollouts_expert_mode(
    base_cfg: Dict[str, Any],
    n_episodes: int,
    max_steps_per_episode: Optional[int],
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Collect transitions by stepping the env with a dummy action, then recording the
    expert/applied action returned in info.

    Returns:
      {
        "obs":      [N, obs_dim],
        "acts":     [N, act_dim],
        "next_obs": [N, obs_dim],
        "dones":    [N,],
        "ep_id":    [N,]  (episode index)
      }
    """
    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    next_obs_buf: List[np.ndarray] = []
    done_buf: List[bool] = []
    ep_id_buf: List[int] = []

    for ep in range(n_episodes):
        env = make_env(base_cfg)
        obs, info = env.reset(seed=seed + ep)
        obs = np.asarray(obs, dtype=np.float32).ravel()

        print(f"[ExpertCollect] Episode {ep + 1}/{n_episodes}")

        done = False
        steps_in_ep = 0

        while not done:
            if max_steps_per_episode is not None and steps_in_ep >= max_steps_per_episode:
                break

            dummy = _dummy_action(env)
            next_obs, reward, terminated, truncated, info = env.step(dummy)
            next_obs = np.asarray(next_obs, dtype=np.float32).ravel()
            done = bool(terminated or truncated)

            # Key: pull the expert-applied action out of info
            a_used = _extract_applied_action(info)

            obs_buf.append(obs.copy())
            act_buf.append(a_used.copy())
            next_obs_buf.append(next_obs.copy())
            done_buf.append(done)
            ep_id_buf.append(ep)

            obs = next_obs
            steps_in_ep += 1

        print(f"[ExpertCollect]   Collected {steps_in_ep} steps.")
        env.close()

    if len(obs_buf) == 0:
        raise RuntimeError(
            "No transitions collected. Possible causes:\n"
            "- Episodes terminate immediately\n"
            "- expert_test_mode not enabled in config\n"
            "- NGSimEnv.step() does not fill info['expert_action'] / info['applied_action']"
        )

    data = {
        "obs": np.stack(obs_buf, axis=0).astype(np.float32),
        "acts": np.stack(act_buf, axis=0).astype(np.float32),
        "next_obs": np.stack(next_obs_buf, axis=0).astype(np.float32),
        "dones": np.asarray(done_buf, dtype=bool),
        "ep_id": np.asarray(ep_id_buf, dtype=np.int32),
    }

    print(f"[ExpertCollect] Finished: {data['obs'].shape[0]} transitions.")
    print(f"[ExpertCollect] obs_dim={data['obs'].shape[1]} act_dim={data['acts'].shape[1]}")
    return data


# ----------------------------------------------------------------------
# CLI and entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect NGSim expert dataset (continuous actions) using expert_test_mode=True."
    )
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to attempt.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional cap on steps per episode.")
    parser.add_argument(
        "--out",
        type=str,
        default="expert_data/ngsim_expert_continuous.npz",
        help="Output NPZ path.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    args = parser.parse_args()

    # ---------------------- YOUR CONFIG (unchanged) ----------------------
    base_cfg: Dict[str, Any] = {
        "scene": "us-101",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalise": True,
        },
        "action": {"type": "ContinuousAction"},
        "show_trajectories": True,

        # Keep them equal for a clean 1-to-1 step mapping (recommended in expert replay mode)
        "simulation_frequency": 10,
        "policy_frequency": 10,

        # Rendering flags (kept from your config; not used for video here)
        "screen_width": 400,
        "screen_height": 150,
        "scaling": 2.0,
        "offscreen_rendering": True,

        # Random episode selection each reset
        "episode_root": "highway_env/data/processed_10s",
        "replay_period": None,
        "reset_step_offset": 1,

        # Ego override (None -> random ego by NGSimEnv)
        "ego_vehicle_ID": None,

        # Spawn volume (your existing setting)
        "max_surrounding": 20000,

        # Critical: environment applies expert actions internally
        "expert_test_mode": True,
    }

    dataset = collect_expert_rollouts_expert_mode(
        base_cfg=base_cfg,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(out_path, **dataset)
    print(f"[ExpertCollect] Saved dataset to: {out_path}")


if __name__ == "__main__":
    main()
