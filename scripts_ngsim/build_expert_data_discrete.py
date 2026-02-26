#!/usr/bin/env python3
"""
Collect a DISCRETE expert dataset from NGSimEnv.

This script:
1. Configures NGSimEnv with 'DiscreteSteerMetaAction'.
2. Enables 'expert_test_mode' + 'expert_action_mode="discrete"'.
3. Steps the environment.
4. Extracts 'info["expert_action_discrete_idx"]' (the integer label calculated by the tracker).
5. Saves the result to an NPZ file.
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
    Construct NGSim env using gym.make.
    We use FlattenObservation to ensure 'obs' is a clean 1D vector.
    """
    env = gym.make(
        "NGSim-US101-v0",
        config=config,
    )
    env = FlattenObservation(env)
    return env


# ----------------------------------------------------------------------
# Utilities for extracting expert/applied action
# ----------------------------------------------------------------------
def _extract_applied_action(info: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Extract the DISCRETE action index actually applied by the environment.
    
    In our modified NGSimEnv, this is stored in info["expert_action_discrete_idx"].
    We return it as a 1D numpy array of shape (1,) so it stacks consistently.
    """
    if info is None:
        info = {}

    # 1. Check for the specific discrete index key (Preferred)
    if "expert_action_discrete_idx" in info:
        idx = info["expert_action_discrete_idx"]
        return np.array([idx], dtype=np.float32)

    # 2. Fallback to generic applied_action
    if "applied_action" in info:
        val = info["applied_action"]
        # If it's a scalar integer/float (discrete index), wrap it
        if np.isscalar(val):
            return np.array([val], dtype=np.float32)
        # If it's already an array, ravel it
        return np.asarray(val, dtype=np.float32).ravel()

    raise RuntimeError(
        "expert_test_mode=True but env.step() did not expose 'expert_action_discrete_idx'.\n"
        "Check that your NGSimEnv.step() populates info['expert_action_discrete_idx']."
    )


def _dummy_action(env: gym.Env) -> int:
    """
    Create a valid dummy action for Discrete space.
    """
    # Simply sample a random integer. It will be ignored by the env anyway.
    return env.action_space.sample()


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

            # EXTRACT DISCRETE ACTION INDEX
            try:
                a_used = _extract_applied_action(info)
            except RuntimeError as e:
                print(f"Error extracting action at step {steps_in_ep}: {e}")
                break

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
        raise RuntimeError("No transitions collected.")

    data = {
        "obs": np.stack(obs_buf, axis=0).astype(np.float32),
        "acts": np.stack(act_buf, axis=0).astype(np.float32), # Shape [N, 1]
        "next_obs": np.stack(next_obs_buf, axis=0).astype(np.float32),
        "dones": np.asarray(done_buf, dtype=bool),
        "ep_id": np.asarray(ep_id_buf, dtype=np.int32),
    }

    print(f"[ExpertCollect] Finished: {data['obs'].shape[0]} transitions.")
    print(f"[ExpertCollect] obs_dim={data['obs'].shape[1]} act_dim={data['acts'].shape[1]}")
    
    # Sanity check for discrete actions
    unique_acts = np.unique(data['acts'])
    print(f"[ExpertCollect] Unique Discrete Actions Found: {unique_acts}")
    
    return data


# ----------------------------------------------------------------------
# CLI and entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect NGSim expert dataset (DISCRETE actions)."
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional cap on steps per episode.")
    parser.add_argument(
        "--out",
        type=str,
        default="expert_data/ngsim_expert_discrete.npz",
        help="Output NPZ path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    args = parser.parse_args()

    # ---------------------- DISCRETE CONFIG ----------------------
    base_cfg: Dict[str, Any] = {
        "scene": "us-101",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        # 1. Use Discrete Action Space
        "action": {"type": "DiscreteSteerMetaAction"},
        
        "show_trajectories": False, # Disable for speed during collection
        "simulation_frequency": 10,
        "policy_frequency": 10,

        # Rendering options (offscreen)
        "screen_width": 400,
        "screen_height": 150,
        "scaling": 2.0,
        "offscreen_rendering": True,

        "episode_root": "highway_env/data/processed_10s",
        "replay_period": None,
        "reset_step_offset": 1,
        "ego_vehicle_ID": None,
        "max_surrounding": 20000, # Reduce density slightly for speed if desired

        # 2. Enable Expert Discrete Mode
        "expert_test_mode": True,
        "expert_action_mode": "discrete",
        
        # Expert Tuning (Matches your environment settings)
        "expert_prefer_speed": False, 
        "lane_change_cooldown_steps": 10,
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