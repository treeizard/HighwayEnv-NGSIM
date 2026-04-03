#!/usr/bin/env python3
"""
Collect a DISCRETE expert dataset from NGSimEnv.

This script:
1. Configures NGSimEnv with 'DiscreteSteerMetaAction'.
2. Enables expert override mode with discrete expert actions.
3. Steps the environment using a valid placeholder action.
4. Reads the actual expert discrete action chosen by the environment from
   info["expert_action_discrete_idx"].
5. Saves a non-repeating dataset over unique (episode_name, ego_id) scenarios.

Output arrays:
    obs           float32 [N, obs_dim]
    acts          int64   [N, 1]
    next_obs      float32 [N, obs_dim]
    dones         bool    [N]
    ep_id         int32   [N]
    episode_name  str     [N]
    ego_id        int32   [N]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import FlattenObservation

# ----------------------------------------------------------------------
# Import project root so `highway_env` is importable
# ----------------------------------------------------------------------
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ----------------------------------------------------------------------
# Register NGSimEnv safely
# ----------------------------------------------------------------------
ENV_ID = "NGSim-US101-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="highway_env.envs.ngsim_env:NGSimEnv")


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------
def resolve_output_path(out_arg: str) -> str:
    """
    Resolve output path robustly for local runs and SLURM jobs.

    Rules:
    - If out_arg is absolute, use it as-is.
    - If out_arg is relative and SLURM_SUBMIT_DIR exists, resolve against it.
    - Otherwise resolve against current working directory.
    """
    if os.path.isabs(out_arg):
        return os.path.abspath(out_arg)

    base_dir = os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd()
    return os.path.abspath(os.path.join(base_dir, out_arg))


# ----------------------------------------------------------------------
# Environment construction
# ----------------------------------------------------------------------
def make_env(config: Dict[str, Any]) -> gym.Env:
    """
    Construct NGSim env using gym.make and flatten the observation to 1D.
    """
    env = gym.make(ENV_ID, config=config)
    env = FlattenObservation(env)
    return env


# ----------------------------------------------------------------------
# Action extraction
# ----------------------------------------------------------------------
def extract_expert_action(info: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Extract the discrete expert action actually applied by the environment.

    Expected primary key:
        info["expert_action_discrete_idx"]

    Fallback:
        info["applied_action"]

    Returns:
        np.ndarray of shape (1,) and dtype int64.
    """
    info = info or {}

    if "expert_action_discrete_idx" in info:
        idx = int(info["expert_action_discrete_idx"])
        return np.array([idx], dtype=np.int64)

    if "applied_action" in info:
        val = info["applied_action"]
        if np.isscalar(val):
            return np.array([int(val)], dtype=np.int64)

        arr = np.asarray(val).ravel()
        if arr.size != 1:
            raise RuntimeError(
                "Expected a scalar discrete action in info['applied_action'], "
                f"but got shape {arr.shape}."
            )
        return np.array([int(arr[0])], dtype=np.int64)

    raise RuntimeError(
        "expert_test_mode=True but env.step() did not expose "
        "'expert_action_discrete_idx' or 'applied_action'."
    )


def dummy_action(env: gym.Env) -> int:
    """
    Return a valid placeholder action.

    In expert override mode, the env should ignore the semantic content of this
    input and instead apply the internal expert action.
    """
    if not hasattr(env.action_space, "sample"):
        raise RuntimeError("Environment action_space does not support sampling.")
    return int(env.action_space.sample())


# ----------------------------------------------------------------------
# Scenario enumeration
# ----------------------------------------------------------------------
def build_unique_scenarios(env: gym.Env) -> list[tuple[str, int]]:
    """
    Extract all unique (episode_name, ego_id) pairs from the unwrapped NGSim env.
    """
    base = env.unwrapped

    if not hasattr(base, "_episodes"):
        raise AttributeError("Env is missing attribute '_episodes'.")
    if not hasattr(base, "_valid_ids_by_episode"):
        raise AttributeError("Env is missing attribute '_valid_ids_by_episode'.")

    scenarios: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    for ep_name in base._episodes:
        if ep_name not in base._valid_ids_by_episode:
            continue

        for ego_id in base._valid_ids_by_episode[ep_name]:
            key = (str(ep_name), int(ego_id))
            if key in seen:
                continue
            seen.add(key)
            scenarios.append(key)

    if not scenarios:
        raise RuntimeError("No valid (episode_name, ego_id) scenarios found.")

    return scenarios


# ----------------------------------------------------------------------
# Rollout collection
# ----------------------------------------------------------------------
def collect_expert_rollouts_unique(
    base_cfg: Dict[str, Any],
    n_episodes: int,
    max_steps_per_episode: Optional[int],
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Collect expert transitions from unique (episode_name, ego_id) scenarios.
    """
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    next_obs_buf: list[np.ndarray] = []
    done_buf: list[bool] = []
    ep_id_buf: list[int] = []
    episode_name_buf: list[str] = []
    ego_id_buf: list[int] = []

    probe_env = make_env(base_cfg)
    try:
        scenarios = build_unique_scenarios(probe_env)
    finally:
        probe_env.close()

    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios)

    if n_episodes > len(scenarios):
        raise ValueError(
            f"Requested {n_episodes} episodes, but only {len(scenarios)} unique "
            "(episode_name, ego_id) pairs are available."
        )

    selected = scenarios[:n_episodes]
    used_keys: set[tuple[str, int]] = set()

    for ep_idx, (episode_name, ego_id) in enumerate(selected):
        key = (episode_name, ego_id)
        if key in used_keys:
            raise RuntimeError(f"Duplicate scenario selected unexpectedly: {key}")
        used_keys.add(key)

        cfg = dict(base_cfg)
        cfg["simulation_period"] = {"episode_name": episode_name}
        cfg["ego_vehicle_ID"] = int(ego_id)

        env = make_env(cfg)
        try:
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

                act_in = dummy_action(env)
                next_obs, reward, terminated, truncated, info = env.step(act_in)
                del reward  # reward is not used for dataset export

                next_obs = np.asarray(next_obs, dtype=np.float32).ravel()
                done = bool(terminated or truncated)

                a_used = extract_expert_action(info)

                obs_buf.append(obs.copy())
                act_buf.append(a_used.copy())
                next_obs_buf.append(next_obs.copy())
                done_buf.append(done)
                ep_id_buf.append(ep_idx)
                episode_name_buf.append(episode_name)
                ego_id_buf.append(int(ego_id))

                obs = next_obs
                steps_in_ep += 1

        finally:
            env.close()

    if not obs_buf:
        raise RuntimeError("No transitions collected.")

    data: Dict[str, np.ndarray] = {
        "obs": np.stack(obs_buf, axis=0).astype(np.float32),
        "acts": np.stack(act_buf, axis=0).astype(np.int64),   # [N, 1]
        "next_obs": np.stack(next_obs_buf, axis=0).astype(np.float32),
        "dones": np.asarray(done_buf, dtype=bool),
        "ep_id": np.asarray(ep_id_buf, dtype=np.int32),
        "episode_name": np.asarray(episode_name_buf, dtype="<U64"),
        "ego_id": np.asarray(ego_id_buf, dtype=np.int32),
    }

    print(f"[ExpertCollect] Finished: {data['obs'].shape[0]} transitions.")
    print(
        f"[ExpertCollect] obs_dim={data['obs'].shape[1]} "
        f"act_dim={data['acts'].shape[1]}"
    )
    print(f"[ExpertCollect] Unique discrete actions found: {np.unique(data['acts'].ravel())}")
    print(f"[ExpertCollect] Unique scenarios used: {len(used_keys)}")

    return data


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def build_base_config() -> Dict[str, Any]:
    """
    Build the default environment config for expert dataset collection.
    """
    return {
        "scene": "us-101",
        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "action": {
            "type": "DiscreteSteerMetaAction",
        },
        "show_trajectories": False,
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "screen_width": 400,
        "screen_height": 150,
        "scaling": 2.0,
        "offscreen_rendering": True,
        "episode_root": "highway_env/data/processed_10s",
        "replay_period": None,
        "reset_step_offset": 1,
        "ego_vehicle_ID": None,
        "max_surrounding": "all",
        "expert_test_mode": True,
        "action_mode": "discrete",   # keep this aligned with your env implementation
        "expert_prefer_speed": False,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect NGSim expert dataset with non-repeating discrete expert actions."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of unique (episode_name, ego_id) scenarios to collect.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="expert_data/ngsim_expert_discrete.npz",
        help=(
            "Output NPZ path. If relative, resolve against SLURM_SUBMIT_DIR when set, "
            "otherwise against current working directory."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed used for scenario shuffle and env reset.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print(f"[ExpertCollect] Current working directory: {os.getcwd()}")
    print(f"[ExpertCollect] SLURM_SUBMIT_DIR: {os.environ.get('SLURM_SUBMIT_DIR', '<not set>')}")

    base_cfg = build_base_config()

    dataset = collect_expert_rollouts_unique(
        base_cfg=base_cfg,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )

    out_path = resolve_output_path(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[ExpertCollect] Saving dataset to: {out_path}")
    np.savez_compressed(out_path, **dataset)
    print(f"[ExpertCollect] Saved dataset to: {out_path}")


if __name__ == "__main__":
    main()
