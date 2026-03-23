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
def build_unique_scenarios(env: gym.Env) -> list[tuple[str, int]]:
    """
    Extract all unique (episode_name, ego_id) pairs from the unwrapped NGSim env.
    """
    base = env.unwrapped
    scenarios = []
    for ep_name in base._episodes:
        for ego_id in base._valid_ids_by_episode[ep_name]:
            scenarios.append((str(ep_name), int(ego_id)))
    return scenarios

def collect_expert_rollouts_unique(
    base_cfg: Dict[str, Any],
    n_episodes: int,
    max_steps_per_episode: Optional[int],
    seed: int,
) -> Dict[str, np.ndarray]:
    obs_buf = []
    act_buf = []
    next_obs_buf = []
    done_buf = []
    ep_id_buf = []

    # Build scenario pool once
    probe_env = make_env(base_cfg)
    scenarios = build_unique_scenarios(probe_env)
    probe_env.close()

    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios)

    if n_episodes > len(scenarios):
        raise ValueError(
            f"Requested {n_episodes} episodes, but only {len(scenarios)} unique "
            f"(episode_name, ego_id) pairs are available."
        )

    selected = scenarios[:n_episodes]

    for ep_idx, (episode_name, ego_id) in enumerate(selected):
        cfg = dict(base_cfg)
        cfg["simulation_period"] = {"episode_name": episode_name}
        cfg["ego_vehicle_ID"] = int(ego_id)

        env = make_env(cfg)
        obs, info = env.reset(seed=seed + ep_idx)
        obs = np.asarray(obs, dtype=np.float32).ravel()

        print(
            f"[ExpertCollect] Episode {ep_idx + 1}/{n_episodes} | "
            f"episode={episode_name} ego_id={ego_id}"
        )

        done = False
        steps_in_ep = 0

        while not done:
            if max_steps_per_episode is not None and steps_in_ep >= max_steps_per_episode:
                break

            dummy = _dummy_action(env)
            next_obs, reward, terminated, truncated, info = env.step(dummy)
            next_obs = np.asarray(next_obs, dtype=np.float32).ravel()
            done = bool(terminated or truncated)

            a_used = _extract_applied_action(info)

            obs_buf.append(obs.copy())
            act_buf.append(a_used.copy())
            next_obs_buf.append(next_obs.copy())
            done_buf.append(done)
            ep_id_buf.append(ep_idx)

            obs = next_obs
            steps_in_ep += 1

        env.close()

    if len(obs_buf) == 0:
        raise RuntimeError("No transitions collected.")

    return {
        "obs": np.stack(obs_buf, axis=0).astype(np.float32),
        "acts": np.stack(act_buf, axis=0).astype(np.float32),
        "next_obs": np.stack(next_obs_buf, axis=0).astype(np.float32),
        "dones": np.asarray(done_buf, dtype=bool),
        "ep_id": np.asarray(ep_id_buf, dtype=np.int32),
    }


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
        "action_mode": "discrete",
        
        # Expert Tuning (Matches your environment settings)
        "expert_prefer_speed": False, 
        "lane_change_cooldown_steps": 10,
    }

    dataset = collect_expert_rollouts_unique(
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