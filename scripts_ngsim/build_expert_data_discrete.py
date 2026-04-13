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
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry

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
    Construct NGSim env using gym.make.

    For multiple controlled expert vehicles we request multi-agent observation and
    action spaces, then flatten each per-agent observation during export.
    """
    config = dict(config)
    controlled_vehicles = int(config.get("controlled_vehicles", 1))
    if controlled_vehicles > 1:
        obs_cfg = dict(config.get("observation", {}))
        action_cfg = dict(config.get("action", {}))
        config["observation"] = {
            "type": "MultiAgentObservation",
            "observation_config": obs_cfg,
        }
        config["action"] = {
            "type": "MultiAgentAction",
            "action_config": action_cfg,
        }

    env = gym.make(ENV_ID, config=config)
    return env


# ----------------------------------------------------------------------
# Action extraction
# ----------------------------------------------------------------------
def extract_expert_actions(info: Optional[Dict[str, Any]]) -> list[np.ndarray]:
    """
    Extract the discrete expert action actually applied by the environment.

    Expected primary key:
        info["expert_action_discrete_idx"]

    Fallback:
        info["applied_action"]

    Returns:
        One discrete action per controlled expert vehicle.
    """
    info = info or {}

    if "expert_action_discrete_idx_all" in info:
        return [
            np.array([int(idx)], dtype=np.int64)
            for idx in info["expert_action_discrete_idx_all"]
        ]

    if "expert_action_discrete_idx" in info:
        idx = int(info["expert_action_discrete_idx"])
        return [np.array([idx], dtype=np.int64)]

    if "applied_actions" in info:
        actions = []
        for val in info["applied_actions"]:
            if np.isscalar(val):
                actions.append(np.array([int(val)], dtype=np.int64))
                continue

            arr = np.asarray(val).ravel()
            if arr.size != 1:
                raise RuntimeError(
                    "Expected scalar discrete actions in info['applied_actions'], "
                    f"but got shape {arr.shape}."
                )
            actions.append(np.array([int(arr[0])], dtype=np.int64))
        return actions

    if "applied_action" in info:
        val = info["applied_action"]
        if np.isscalar(val):
            return [np.array([int(val)], dtype=np.int64)]

        arr = np.asarray(val).ravel()
        if arr.size != 1:
            raise RuntimeError(
                "Expected a scalar discrete action in info['applied_action'], "
                f"but got shape {arr.shape}."
            )
        return [np.array([int(arr[0])], dtype=np.int64)]

    raise RuntimeError(
        "expert_test_mode=True but env.step() did not expose "
        "'expert_action_discrete_idx' or 'applied_action'."
    )


def dummy_action(env: gym.Env) -> Any:
    """
    Return a valid placeholder action.

    In expert override mode, the env should ignore the semantic content of this
    input and instead apply the internal expert action.
    """
    if not hasattr(env.action_space, "sample"):
        raise RuntimeError("Environment action_space does not support sampling.")
    return env.action_space.sample()


def flatten_observations(obs: Any) -> list[np.ndarray]:
    if isinstance(obs, tuple):
        return [np.asarray(obs_i, dtype=np.float32).ravel() for obs_i in obs]
    return [np.asarray(obs, dtype=np.float32).ravel()]


# ----------------------------------------------------------------------
# Scenario enumeration
# ----------------------------------------------------------------------
def build_unique_scenarios(
    env: gym.Env, controlled_vehicles: int
) -> list[tuple[str, tuple[int, ...]]]:
    """
    Extract unique scenario groups from the unwrapped NGSim env.

    For a single controlled vehicle, each scenario is a single ego id.
    For multiple controlled vehicles, use sliding windows over sorted valid ids to
    keep the scenario count tractable while still providing diverse groupings.
    """
    base = env.unwrapped

    if not hasattr(base, "_episodes"):
        raise AttributeError("Env is missing attribute '_episodes'.")
    if not hasattr(base, "_valid_ids_by_episode"):
        raise AttributeError("Env is missing attribute '_valid_ids_by_episode'.")

    scenarios: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for ep_name in base._episodes:
        if ep_name not in base._valid_ids_by_episode:
            continue

        valid_ids = sorted(int(ego_id) for ego_id in base._valid_ids_by_episode[ep_name])
        if len(valid_ids) < controlled_vehicles:
            continue

        if controlled_vehicles == 1:
            ego_groups = [(ego_id,) for ego_id in valid_ids]
        else:
            ego_groups = [
                tuple(valid_ids[i : i + controlled_vehicles])
                for i in range(len(valid_ids) - controlled_vehicles + 1)
            ]

        for ego_group in ego_groups:
            key = (str(ep_name), tuple(int(ego_id) for ego_id in ego_group))
            if key in seen:
                continue
            seen.add(key)
            scenarios.append(key)

    if not scenarios:
        raise RuntimeError("No valid expert replay scenarios found.")

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
    scenario_ids_buf: list[str] = []

    controlled_vehicles = int(base_cfg.get("controlled_vehicles", 1))

    probe_env = make_env(base_cfg)
    try:
        scenarios = build_unique_scenarios(probe_env, controlled_vehicles)
    finally:
        probe_env.close()

    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios)

    if n_episodes > len(scenarios):
        raise ValueError(
            f"Requested {n_episodes} episodes, but only {len(scenarios)} unique "
            "scenario groups are available."
        )

    selected = scenarios[:n_episodes]
    used_keys: set[tuple[str, tuple[int, ...]]] = set()

    for ep_idx, (episode_name, ego_ids) in enumerate(selected):
        key = (episode_name, ego_ids)
        if key in used_keys:
            raise RuntimeError(f"Duplicate scenario selected unexpectedly: {key}")
        used_keys.add(key)

        cfg = dict(base_cfg)
        cfg["simulation_period"] = {"episode_name": episode_name}
        cfg["ego_vehicle_ID"] = (
            int(ego_ids[0]) if controlled_vehicles == 1 else list(map(int, ego_ids))
        )
        cfg["controlled_vehicles"] = controlled_vehicles

        env = make_env(cfg)
        try:
            obs, info = env.reset(seed=seed + ep_idx)
            obs_list = flatten_observations(obs)
            scenario_label = ",".join(str(ego_id) for ego_id in ego_ids)

            print(
                f"[ExpertCollect] Episode {ep_idx + 1}/{n_episodes} | "
                f"episode={episode_name} ego_ids=[{scenario_label}]"
            )

            done = False
            steps_in_ep = 0

            while not done:
                if max_steps_per_episode is not None and steps_in_ep >= max_steps_per_episode:
                    break

                act_in = dummy_action(env)
                next_obs, reward, terminated, truncated, info = env.step(act_in)
                del reward  # reward is not used for dataset export

                next_obs_list = flatten_observations(next_obs)
                done = bool(terminated or truncated)

                actions_used = extract_expert_actions(info)
                vehicle_ids = info.get("expert_controlled_vehicle_ids") or list(ego_ids)

                if not (
                    len(obs_list) == len(next_obs_list) == len(actions_used) == len(vehicle_ids)
                ):
                    raise RuntimeError(
                        "Mismatch between controlled-vehicle observations/actions: "
                        f"obs={len(obs_list)} next_obs={len(next_obs_list)} "
                        f"acts={len(actions_used)} ids={len(vehicle_ids)}"
                    )

                for obs_i, action_i, next_obs_i, vehicle_id in zip(
                    obs_list, actions_used, next_obs_list, vehicle_ids
                ):
                    obs_buf.append(obs_i.copy())
                    act_buf.append(action_i.copy())
                    next_obs_buf.append(next_obs_i.copy())
                    done_buf.append(done)
                    ep_id_buf.append(ep_idx)
                    episode_name_buf.append(episode_name)
                    ego_id_buf.append(int(vehicle_id))
                    scenario_ids_buf.append(scenario_label)

                obs_list = next_obs_list
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
        "scenario_ego_ids": np.asarray(scenario_ids_buf, dtype="<U128"),
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
        "episode_root": "highway_env/data/processed_20s",
        "prebuilt_split": "train",
        "replay_period": None,
        "reset_step_offset": 1,
        "ego_vehicle_ID": None,
        "max_surrounding": "all",
        "expert_test_mode": True,
        "action_mode": "discrete",   # keep this aligned with your env implementation
        "expert_prefer_speed": False,
        "controlled_vehicles": 1,
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
    parser.add_argument(
        "--controlled_vehicles",
        type=int,
        default=10,
        help="Number of expert-controlled vehicles to replay per environment step.",
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
    base_cfg["controlled_vehicles"] = int(args.controlled_vehicles)

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
