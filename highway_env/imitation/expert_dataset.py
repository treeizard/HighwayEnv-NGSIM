from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register, registry
from torch.utils.data import Dataset


ENV_ID = "NGSim-US101-v0"
SCHEMA_VERSION = 1


@dataclass
class EpisodeTrajectory:
    episode_id: int
    scenario_id: str
    episode_name: str
    ego_id: int
    source_split: str
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    timesteps: np.ndarray


@dataclass
class SceneTrajectory:
    episode_id: int
    scenario_id: str
    episode_name: str
    agent_ids: np.ndarray
    source_split: str
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    timesteps: np.ndarray
    alive_mask: np.ndarray


def register_ngsim_env() -> None:
    """Register the local NGSim gym id when the caller has not done so yet."""
    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point="highway_env.envs.ngsim_env:NGSimEnv")


def default_observation_config(cells: int = 128, maximum_range: float = 64.0) -> dict[str, Any]:
    return {
        "type": "LidarObservation",
        "cells": int(cells),
        "maximum_range": float(maximum_range),
        "normalize": True,
    }


def build_env_config(
    *,
    scene: str,
    action_mode: str,
    episode_root: str,
    prebuilt_split: str = "train",
    controlled_vehicles: int,
    control_all_vehicles: bool = False,
    max_surrounding: str | int,
    observation_config: dict[str, Any] | None = None,
    simulation_frequency: int = 10,
    policy_frequency: int = 10,
    max_episode_steps: int = 300,
    show_trajectories: bool = False,
    seed: int | None = None,
    simulation_period: dict[str, Any] | None = None,
    ego_vehicle_id: int | list[int] | None = None,
) -> dict[str, Any]:
    """
    Build a repo-native NGSim config for expert replay collection.

    The dataset pipeline uses expert replay driven by processed real trajectories,
    so observations and actions match the actual environment conventions.
    """
    action_mode = str(action_mode).lower()
    if action_mode not in {"discrete", "continuous"}:
        raise ValueError(f"Unsupported action_mode={action_mode!r}")

    obs_cfg = dict(observation_config or default_observation_config())
    action_type = "DiscreteSteerMetaAction" if action_mode == "discrete" else "ContinuousAction"

    if control_all_vehicles or controlled_vehicles > 1:
        observation = {
            "type": "MultiAgentObservation",
            "observation_config": obs_cfg,
        }
        action = {
            "type": "MultiAgentAction",
            "action_config": {"type": action_type},
        }
    else:
        observation = obs_cfg
        action = {"type": action_type}

    return {
        "scene": str(scene),
        "observation": observation,
        "action": action,
        "action_mode": action_mode,
        "show_trajectories": bool(show_trajectories),
        "simulation_frequency": int(simulation_frequency),
        "policy_frequency": int(policy_frequency),
        "offscreen_rendering": True,
        "episode_root": str(episode_root),
        "prebuilt_split": str(prebuilt_split),
        "simulation_period": simulation_period,
        "ego_vehicle_ID": ego_vehicle_id,
        "controlled_vehicles": int(controlled_vehicles),
        "control_all_vehicles": bool(control_all_vehicles),
        "max_surrounding": max_surrounding,
        "expert_test_mode": True,
        "truncate_to_trajectory_length": True,
        "max_episode_steps": int(max_episode_steps),
        "seed": seed,
    }


def _flatten_action_for_storage(action: np.ndarray | int | float) -> np.ndarray:
    arr = np.asarray(action)
    if arr.ndim == 0:
        return arr.reshape(())
    return arr.astype(arr.dtype, copy=False)


def _extract_single_agent_views(value: Any, dtype: np.dtype | None = None) -> list[np.ndarray]:
    if isinstance(value, tuple):
        return [_coerce_array(v, dtype=dtype) for v in value]
    return [_coerce_array(value, dtype=dtype)]


def _coerce_array(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(arr)) and np.issubdtype(arr.dtype, np.floating):
        raise ValueError("Encountered non-finite values while building expert dataset.")
    return arr


def _extract_expert_actions(info: dict[str, Any], action_mode: str) -> list[np.ndarray]:
    action_mode = str(action_mode).lower()
    info = info or {}

    if action_mode == "continuous":
        if "expert_action_continuous_all" in info:
            return [_coerce_array(a, dtype=np.float32) for a in info["expert_action_continuous_all"]]
        if "expert_action_continuous" in info:
            return [_coerce_array(info["expert_action_continuous"], dtype=np.float32)]
    else:
        if "expert_action_discrete_idx_all" in info:
            return [np.asarray(a, dtype=np.int64).reshape(()) for a in info["expert_action_discrete_idx_all"]]
        if "expert_action_discrete_idx" in info:
            return [np.asarray(info["expert_action_discrete_idx"], dtype=np.int64).reshape(())]

    if "applied_actions" in info:
        return [_flatten_action_for_storage(a) for a in info["applied_actions"]]
    if "applied_action" in info:
        return [_flatten_action_for_storage(info["applied_action"])]

    raise RuntimeError(
        "NGSimEnv did not expose expert actions in info. "
        "Expected expert_action_* or applied_action keys."
    )


def _dummy_action(env: gym.Env) -> Any:
    action_space = env.action_space
    if hasattr(action_space, "n"):
        return 0
    if hasattr(action_space, "shape") and action_space.shape is not None:
        return np.zeros(action_space.shape, dtype=np.float32)
    if hasattr(action_space, "spaces"):
        return tuple(_dummy_action(type("TmpEnv", (), {"action_space": space})()) for space in action_space.spaces)
    if hasattr(action_space, "sample"):
        return action_space.sample()
    raise RuntimeError("Environment action_space does not support dummy action creation.")


def _available_scenarios(
    env: gym.Env,
    controlled_vehicles: int,
    control_all_vehicles: bool = False,
) -> list[tuple[str, tuple[int, ...]]]:
    base = env.unwrapped
    if not hasattr(base, "_episodes") or not hasattr(base, "_valid_ids_by_episode"):
        raise AttributeError("NGSimEnv does not expose prebuilt episode metadata.")

    scenarios: list[tuple[str, tuple[int, ...]]] = []
    for episode_name in base._episodes:
        valid_ids = [int(v) for v in base._valid_ids_by_episode.get(episode_name, [])]
        valid_ids = sorted(valid_ids)
        if control_all_vehicles:
            if valid_ids:
                scenarios.append((str(episode_name), tuple(valid_ids)))
            continue
        if len(valid_ids) < controlled_vehicles:
            continue

        if controlled_vehicles == 1:
            ego_groups = [(ego_id,) for ego_id in valid_ids]
        else:
            ego_groups = [
                tuple(valid_ids[start : start + controlled_vehicles])
                for start in range(len(valid_ids) - controlled_vehicles + 1)
            ]

        for ego_ids in ego_groups:
            scenarios.append((str(episode_name), tuple(int(ego_id) for ego_id in ego_ids)))

    if not scenarios:
        raise RuntimeError("No valid replay scenarios were found in the processed trajectory root.")
    return scenarios


def _action_dtype(action_mode: str) -> np.dtype:
    return np.int64 if str(action_mode).lower() == "discrete" else np.float32


def _stack_actions(actions: list[np.ndarray], action_mode: str) -> np.ndarray:
    action_dtype = _action_dtype(action_mode)
    first_action = np.asarray(actions[0])
    if first_action.ndim == 0:
        return np.asarray([int(np.asarray(a)) for a in actions], dtype=action_dtype)
    return np.stack([np.asarray(a, dtype=action_dtype) for a in actions], axis=0).astype(
        action_dtype,
        copy=False,
    )


def _build_episode_trajectory(
    *,
    episode_id: int,
    scenario_id: str,
    episode_name: str,
    ego_id: int,
    source_split: str,
    obs_list: list[np.ndarray],
    act_list: list[np.ndarray],
    next_obs_list: list[np.ndarray],
    done_list: list[bool],
    reward_list: list[float],
    timestep_list: list[int],
    action_mode: str,
) -> EpisodeTrajectory:
    if not obs_list:
        raise RuntimeError(f"Scenario {scenario_id} ego_id={ego_id} produced no transitions.")

    observations = np.stack(obs_list, axis=0).astype(np.float32, copy=False)
    next_observations = np.stack(next_obs_list, axis=0).astype(np.float32, copy=False)
    dones = np.asarray(done_list, dtype=bool)
    rewards = np.asarray(reward_list, dtype=np.float32)
    timesteps = np.asarray(timestep_list, dtype=np.int32)

    actions = _stack_actions(act_list, action_mode)

    return EpisodeTrajectory(
        episode_id=int(episode_id),
        scenario_id=str(scenario_id),
        episode_name=str(episode_name),
        ego_id=int(ego_id),
        source_split=str(source_split),
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        timesteps=timesteps,
    )


def _build_scene_trajectory(
    *,
    episode_id: int,
    scenario_id: str,
    episode_name: str,
    agent_ids: list[int],
    source_split: str,
    obs_steps: list[np.ndarray],
    act_steps: list[np.ndarray],
    next_obs_steps: list[np.ndarray],
    done_steps: list[bool],
    reward_steps: list[float],
    timestep_steps: list[int],
    alive_mask_steps: list[np.ndarray],
    action_mode: str,
) -> SceneTrajectory:
    if not obs_steps:
        raise RuntimeError(f"Scene scenario {scenario_id} produced no transitions.")

    observations = np.stack(obs_steps, axis=0).astype(np.float32, copy=False)
    next_observations = np.stack(next_obs_steps, axis=0).astype(np.float32, copy=False)
    actions = np.stack([_stack_actions(step_actions, action_mode) for step_actions in act_steps], axis=0)
    dones = np.asarray(done_steps, dtype=bool)
    rewards = np.asarray(reward_steps, dtype=np.float32)
    timesteps = np.asarray(timestep_steps, dtype=np.int32)
    alive_mask = np.stack(alive_mask_steps, axis=0).astype(bool, copy=False)

    return SceneTrajectory(
        episode_id=int(episode_id),
        scenario_id=str(scenario_id),
        episode_name=str(episode_name),
        agent_ids=np.asarray(agent_ids, dtype=np.int32),
        source_split=str(source_split),
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        timesteps=timesteps,
        alive_mask=alive_mask,
    )


def build_expert_dataset(
    *,
    scene: str,
    action_mode: str,
    episode_root: str,
    prebuilt_split: str = "train",
    output_path: str,
    num_episodes: int,
    fixed_episode_name: str | None = None,
    max_horizon: int | None = None,
    controlled_vehicles: int = 1,
    control_all_vehicles: bool = False,
    dataset_mode: str = "per_vehicle",
    max_surrounding: str | int = "all",
    source_split: str = "train",
    seed: int = 0,
    observation_config: dict[str, Any] | None = None,
    simulation_frequency: int = 10,
    policy_frequency: int = 10,
    max_episode_steps: int = 300,
) -> dict[str, Any]:
    """
    Build an expert dataset by replaying processed real trajectories through NGSimEnv.

    Each saved dataset episode corresponds to one controlled vehicle rollout, even when
    multiple vehicles are replayed together in the same simulator scenario. This keeps
    the saved format simple and directly usable for single-agent GAIL baselines while
    retaining scenario metadata for later grouping.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    dataset_mode = str(dataset_mode)
    if dataset_mode not in {"per_vehicle", "scene"}:
        raise ValueError("dataset_mode must be 'per_vehicle' or 'scene'.")
    if dataset_mode == "scene" and not (control_all_vehicles or controlled_vehicles > 1):
        raise ValueError("scene dataset_mode requires multi-agent replay.")
    if control_all_vehicles and max_surrounding != 0:
        max_surrounding = 0

    register_ngsim_env()
    episode_root = os.path.abspath(episode_root)
    output_path = os.path.abspath(output_path)

    probe_cfg = build_env_config(
        scene=scene,
        action_mode=action_mode,
        episode_root=episode_root,
        prebuilt_split=prebuilt_split,
        controlled_vehicles=controlled_vehicles,
        control_all_vehicles=control_all_vehicles,
        max_surrounding=max_surrounding,
        observation_config=observation_config,
        simulation_frequency=simulation_frequency,
        policy_frequency=policy_frequency,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    probe_env = gym.make(ENV_ID, config=probe_cfg)
    try:
        scenarios = _available_scenarios(
            probe_env,
            controlled_vehicles=controlled_vehicles,
            control_all_vehicles=control_all_vehicles,
        )
    finally:
        probe_env.close()

    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios)
    if fixed_episode_name is not None:
        scenarios = [
            scenario for scenario in scenarios if scenario[0] == str(fixed_episode_name)
        ]
        if not scenarios:
            raise ValueError(
                f"Episode {fixed_episode_name!r} was not found in split {prebuilt_split!r}."
            )
    selected_scenarios = scenarios[:num_episodes]
    if len(selected_scenarios) < num_episodes:
        raise RuntimeError(
            f"Requested {num_episodes} scenarios but only found {len(selected_scenarios)} valid ones."
        )

    collected: list[EpisodeTrajectory] = []
    collected_scenes: list[SceneTrajectory] = []

    for scenario_index, (episode_name, ego_ids) in enumerate(selected_scenarios):
        scenario_cfg = build_env_config(
            scene=scene,
            action_mode=action_mode,
            episode_root=episode_root,
            prebuilt_split=prebuilt_split,
            controlled_vehicles=controlled_vehicles,
            control_all_vehicles=control_all_vehicles,
            max_surrounding=max_surrounding,
            observation_config=observation_config,
            simulation_frequency=simulation_frequency,
            policy_frequency=policy_frequency,
            max_episode_steps=max_episode_steps,
            seed=seed + scenario_index,
            simulation_period={"episode_name": episode_name},
            ego_vehicle_id=list(ego_ids),
        )

        env = gym.make(ENV_ID, config=scenario_cfg)
        try:
            obs, _ = env.reset(seed=seed + scenario_index)
            obs_views = _extract_single_agent_views(obs, dtype=np.float32)
            per_vehicle = {
                int(ego_id): {
                    "obs": [],
                    "actions": [],
                    "next_obs": [],
                    "dones": [],
                    "rewards": [],
                    "timesteps": [],
                }
                for ego_id in ego_ids
            }
            scene_obs_steps: list[np.ndarray] = []
            scene_act_steps: list[np.ndarray] = []
            scene_next_obs_steps: list[np.ndarray] = []
            scene_done_steps: list[bool] = []
            scene_reward_steps: list[float] = []
            scene_timestep_steps: list[int] = []
            scene_alive_mask_steps: list[np.ndarray] = []

            step_index = 0
            done = False
            while not done:
                if max_horizon is not None and step_index >= max_horizon:
                    break

                next_obs, reward, terminated, truncated, info = env.step(_dummy_action(env))
                next_obs_views = _extract_single_agent_views(next_obs, dtype=np.float32)
                action_views = _extract_expert_actions(info, action_mode=action_mode)
                vehicle_ids = [int(v) for v in info.get("expert_controlled_vehicle_ids", ego_ids)]

                if not (
                    len(obs_views) == len(next_obs_views) == len(action_views) == len(vehicle_ids)
                ):
                    raise RuntimeError(
                        "Mismatch while collecting expert dataset: "
                        f"obs={len(obs_views)} next_obs={len(next_obs_views)} "
                        f"actions={len(action_views)} vehicle_ids={len(vehicle_ids)}"
                    )

                done = bool(terminated or truncated)
                reward_value = float(reward)
                if dataset_mode == "scene":
                    alive_ids = {int(v) for v in info.get("alive_controlled_vehicle_ids", vehicle_ids)}
                    scene_obs_steps.append(np.stack(obs_views, axis=0).astype(np.float32, copy=False))
                    scene_act_steps.append([np.asarray(a).copy() for a in action_views])
                    scene_next_obs_steps.append(
                        np.stack(next_obs_views, axis=0).astype(np.float32, copy=False)
                    )
                    scene_done_steps.append(done)
                    scene_reward_steps.append(reward_value)
                    scene_timestep_steps.append(step_index)
                    scene_alive_mask_steps.append(
                        np.asarray([vehicle_id in alive_ids for vehicle_id in vehicle_ids], dtype=bool)
                    )

                for obs_i, action_i, next_obs_i, vehicle_id in zip(
                    obs_views, action_views, next_obs_views, vehicle_ids
                ):
                    buf = per_vehicle[int(vehicle_id)]
                    buf["obs"].append(obs_i.copy())
                    buf["actions"].append(np.asarray(action_i).copy())
                    buf["next_obs"].append(next_obs_i.copy())
                    buf["dones"].append(done)
                    buf["rewards"].append(reward_value)
                    buf["timesteps"].append(step_index)

                obs_views = next_obs_views
                step_index += 1

            scenario_id = f"{episode_name}__{'-'.join(str(ego_id) for ego_id in ego_ids)}"
            if dataset_mode == "scene":
                collected_scenes.append(
                    _build_scene_trajectory(
                        episode_id=len(collected_scenes),
                        scenario_id=scenario_id,
                        episode_name=episode_name,
                        agent_ids=[int(ego_id) for ego_id in ego_ids],
                        source_split=source_split,
                        obs_steps=scene_obs_steps,
                        act_steps=scene_act_steps,
                        next_obs_steps=scene_next_obs_steps,
                        done_steps=scene_done_steps,
                        reward_steps=scene_reward_steps,
                        timestep_steps=scene_timestep_steps,
                        alive_mask_steps=scene_alive_mask_steps,
                        action_mode=action_mode,
                    )
                )
            else:
                for ego_id in ego_ids:
                    episode_id = len(collected)
                    vehicle_buf = per_vehicle[int(ego_id)]
                    collected.append(
                        _build_episode_trajectory(
                            episode_id=episode_id,
                            scenario_id=scenario_id,
                            episode_name=episode_name,
                            ego_id=int(ego_id),
                            source_split=source_split,
                            obs_list=vehicle_buf["obs"],
                            act_list=vehicle_buf["actions"],
                            next_obs_list=vehicle_buf["next_obs"],
                            done_list=vehicle_buf["dones"],
                            reward_list=vehicle_buf["rewards"],
                            timestep_list=vehicle_buf["timesteps"],
                            action_mode=action_mode,
                        )
                    )
        finally:
            env.close()

    if dataset_mode == "scene":
        if not collected_scenes:
            raise RuntimeError("No expert scene trajectories were collected.")
        observation_shape = tuple(int(v) for v in collected_scenes[0].observations.shape[2:])
        first_actions = collected_scenes[0].actions
        action_shape = tuple(int(v) for v in first_actions.shape[2:]) if first_actions.ndim > 2 else ()
        num_dataset_episodes = len(collected_scenes)
    else:
        if not collected:
            raise RuntimeError("No expert trajectories were collected.")
        observation_shape = tuple(int(v) for v in collected[0].observations.shape[1:])
        first_actions = collected[0].actions
        action_shape = tuple(int(v) for v in first_actions.shape[1:]) if first_actions.ndim > 1 else ()
        num_dataset_episodes = len(collected)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_id": ENV_ID,
        "scene": str(scene),
        "action_mode": str(action_mode),
        "episode_root": episode_root,
        "prebuilt_split": str(prebuilt_split),
        "source_split": str(source_split),
        "dataset_mode": dataset_mode,
        "num_dataset_episodes": num_dataset_episodes,
        "num_requested_scenarios": int(num_episodes),
        "controlled_vehicles_per_scenario": int(controlled_vehicles),
        "control_all_vehicles": bool(control_all_vehicles),
        "max_horizon": None if max_horizon is None else int(max_horizon),
        "max_surrounding": max_surrounding,
        "simulation_frequency": int(simulation_frequency),
        "policy_frequency": int(policy_frequency),
        "observation_config": observation_config or default_observation_config(),
        "observation_shape": list(observation_shape),
        "action_shape": list(action_shape),
        "action_dtype": str(first_actions.dtype),
    }

    if dataset_mode == "scene":
        arrays = {
            "episode_id": np.asarray([ep.episode_id for ep in collected_scenes], dtype=np.int32),
            "scenario_id": np.asarray([ep.scenario_id for ep in collected_scenes], dtype=object),
            "episode_name": np.asarray([ep.episode_name for ep in collected_scenes], dtype=object),
            "agent_ids": np.asarray([ep.agent_ids for ep in collected_scenes], dtype=object),
            "source_split": np.asarray([ep.source_split for ep in collected_scenes], dtype=object),
            "observations": np.asarray([ep.observations for ep in collected_scenes], dtype=object),
            "actions": np.asarray([ep.actions for ep in collected_scenes], dtype=object),
            "next_observations": np.asarray([ep.next_observations for ep in collected_scenes], dtype=object),
            "dones": np.asarray([ep.dones for ep in collected_scenes], dtype=object),
            "rewards": np.asarray([ep.rewards for ep in collected_scenes], dtype=object),
            "timesteps": np.asarray([ep.timesteps for ep in collected_scenes], dtype=object),
            "alive_mask": np.asarray([ep.alive_mask for ep in collected_scenes], dtype=object),
            "metadata_json": np.asarray(json.dumps(metadata), dtype=object),
        }
    else:
        arrays = {
            "episode_id": np.asarray([ep.episode_id for ep in collected], dtype=np.int32),
            "scenario_id": np.asarray([ep.scenario_id for ep in collected], dtype=object),
            "episode_name": np.asarray([ep.episode_name for ep in collected], dtype=object),
            "ego_id": np.asarray([ep.ego_id for ep in collected], dtype=np.int32),
            "source_split": np.asarray([ep.source_split for ep in collected], dtype=object),
            "observations": np.asarray([ep.observations for ep in collected], dtype=object),
            "actions": np.asarray([ep.actions for ep in collected], dtype=object),
            "next_observations": np.asarray([ep.next_observations for ep in collected], dtype=object),
            "dones": np.asarray([ep.dones for ep in collected], dtype=object),
            "rewards": np.asarray([ep.rewards for ep in collected], dtype=object),
            "timesteps": np.asarray([ep.timesteps for ep in collected], dtype=object),
            "metadata_json": np.asarray(json.dumps(metadata), dtype=object),
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {"output_path": output_path, "metadata": metadata}


def load_dataset_metadata(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "metadata_json" not in data:
            raise KeyError(f"{path} does not contain metadata_json.")
        return json.loads(str(data["metadata_json"].item()))


def load_expert_dataset(path: str) -> dict[str, Any]:
    """
    Load and validate an expert dataset saved by this module.

    Returns plain numpy arrays plus decoded metadata.
    """
    with np.load(path, allow_pickle=True) as data:
        required = {
            "episode_id",
            "scenario_id",
            "episode_name",
            "source_split",
            "observations",
            "actions",
            "next_observations",
            "dones",
            "rewards",
            "timesteps",
            "metadata_json",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"Dataset {path} is missing required fields: {missing}")

        dataset = {key: data[key] for key in data.files if key != "metadata_json"}
        dataset["metadata"] = json.loads(str(data["metadata_json"].item()))

    dataset_mode = str(dataset["metadata"].get("dataset_mode", "per_vehicle"))
    if dataset_mode == "scene":
        _validate_loaded_scene_dataset(dataset)
    else:
        _validate_loaded_dataset(dataset)
    return dataset


def _validate_loaded_dataset(dataset: dict[str, Any]) -> None:
    lengths = {
        key: len(dataset[key])
        for key in [
            "episode_id",
            "scenario_id",
            "episode_name",
            "ego_id",
            "source_split",
            "observations",
            "actions",
            "next_observations",
            "dones",
            "rewards",
            "timesteps",
        ]
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent top-level dataset lengths: {lengths}")

    obs_shape_expected = tuple(dataset["metadata"].get("observation_shape", []))
    action_shape_expected = tuple(dataset["metadata"].get("action_shape", []))
    action_mode = str(dataset["metadata"].get("action_mode", ""))

    for episode_idx in range(lengths["episode_id"]):
        observations = np.asarray(dataset["observations"][episode_idx], dtype=np.float32)
        next_observations = np.asarray(dataset["next_observations"][episode_idx], dtype=np.float32)
        if action_mode == "discrete":
            actions = np.asarray(dataset["actions"][episode_idx], dtype=np.int64)
        else:
            actions = np.asarray(dataset["actions"][episode_idx], dtype=np.float32)
        dones = np.asarray(dataset["dones"][episode_idx], dtype=bool)
        rewards = np.asarray(dataset["rewards"][episode_idx], dtype=np.float32)
        timesteps = np.asarray(dataset["timesteps"][episode_idx], dtype=np.int32)

        T = observations.shape[0]
        if T == 0:
            raise ValueError(f"Episode {episode_idx} is empty.")
        if next_observations.shape[0] != T or dones.shape[0] != T or rewards.shape[0] != T or timesteps.shape[0] != T:
            raise ValueError(f"Episode {episode_idx} has inconsistent per-step lengths.")
        if observations.shape != next_observations.shape:
            raise ValueError(f"Episode {episode_idx} observation and next_observation shapes differ.")
        if obs_shape_expected and tuple(observations.shape[1:]) != obs_shape_expected:
            raise ValueError(
                f"Episode {episode_idx} observation shape {observations.shape[1:]} "
                f"does not match metadata {obs_shape_expected}."
            )
        if timesteps[0] != 0 or not np.array_equal(timesteps, np.arange(T, dtype=np.int32)):
            raise ValueError(f"Episode {episode_idx} timesteps are not contiguous from zero.")

        if action_mode == "discrete":
            if actions.shape != (T,):
                raise ValueError(
                    f"Episode {episode_idx} discrete actions should have shape {(T,)}, got {actions.shape}."
                )
            if actions.dtype.kind not in {"i", "u"}:
                raise ValueError(f"Episode {episode_idx} discrete actions are not integer typed.")
        else:
            if action_shape_expected and tuple(actions.shape[1:]) != action_shape_expected:
                raise ValueError(
                    f"Episode {episode_idx} action shape {actions.shape[1:]} "
                    f"does not match metadata {action_shape_expected}."
                )
            if actions.shape[0] != T:
                raise ValueError(f"Episode {episode_idx} continuous actions length mismatch.")


def _validate_loaded_scene_dataset(dataset: dict[str, Any]) -> None:
    required = [
        "episode_id",
        "scenario_id",
        "episode_name",
        "agent_ids",
        "source_split",
        "observations",
        "actions",
        "next_observations",
        "dones",
        "rewards",
        "timesteps",
        "alive_mask",
    ]
    missing = [key for key in required if key not in dataset]
    if missing:
        raise KeyError(f"Scene dataset missing required fields: {missing}")

    lengths = {key: len(dataset[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent top-level scene dataset lengths: {lengths}")

    obs_shape_expected = tuple(dataset["metadata"].get("observation_shape", []))
    action_shape_expected = tuple(dataset["metadata"].get("action_shape", []))
    action_mode = str(dataset["metadata"].get("action_mode", ""))

    for episode_idx in range(lengths["episode_id"]):
        agent_ids = np.asarray(dataset["agent_ids"][episode_idx], dtype=np.int32)
        observations = np.asarray(dataset["observations"][episode_idx], dtype=np.float32)
        next_observations = np.asarray(dataset["next_observations"][episode_idx], dtype=np.float32)
        actions = (
            np.asarray(dataset["actions"][episode_idx], dtype=np.int64)
            if action_mode == "discrete"
            else np.asarray(dataset["actions"][episode_idx], dtype=np.float32)
        )
        dones = np.asarray(dataset["dones"][episode_idx], dtype=bool)
        rewards = np.asarray(dataset["rewards"][episode_idx], dtype=np.float32)
        timesteps = np.asarray(dataset["timesteps"][episode_idx], dtype=np.int32)
        alive_mask = np.asarray(dataset["alive_mask"][episode_idx], dtype=bool)

        if observations.ndim < 3:
            raise ValueError(f"Scene episode {episode_idx} observations must be [T, N, ...].")
        T, N = observations.shape[:2]
        if T == 0 or N == 0:
            raise ValueError(f"Scene episode {episode_idx} is empty.")
        if len(agent_ids) != N:
            raise ValueError(f"Scene episode {episode_idx} agent_ids length does not match N.")
        if next_observations.shape != observations.shape:
            raise ValueError(f"Scene episode {episode_idx} next_observations shape mismatch.")
        if alive_mask.shape != (T, N):
            raise ValueError(f"Scene episode {episode_idx} alive_mask shape mismatch.")
        if dones.shape != (T,) or rewards.shape != (T,) or timesteps.shape != (T,):
            raise ValueError(f"Scene episode {episode_idx} scalar step arrays shape mismatch.")
        if obs_shape_expected and tuple(observations.shape[2:]) != obs_shape_expected:
            raise ValueError(
                f"Scene episode {episode_idx} observation shape {observations.shape[2:]} "
                f"does not match metadata {obs_shape_expected}."
            )
        if timesteps[0] != 0 or not np.array_equal(timesteps, np.arange(T, dtype=np.int32)):
            raise ValueError(f"Scene episode {episode_idx} timesteps are not contiguous from zero.")

        if action_mode == "discrete":
            if actions.shape != (T, N):
                raise ValueError(f"Scene episode {episode_idx} discrete actions must be [T, N].")
        else:
            expected = (T, N, *action_shape_expected) if action_shape_expected else None
            if expected is not None and actions.shape != expected:
                raise ValueError(
                    f"Scene episode {episode_idx} action shape {actions.shape} does not match {expected}."
                )


class SceneTransitionDataset(Dataset):
    """
    Flatten a scene-level multi-agent dataset into per-agent transitions.

    This keeps full-scene demonstrations on disk but exposes a parameter-sharing
    training view suitable for PS-GAIL style policies.
    """

    def __init__(
        self,
        path: str,
        *,
        flatten_observations: bool = False,
        include_next_observation: bool = True,
        skip_inactive_agents: bool = True,
    ) -> None:
        self.path = os.path.abspath(path)
        self.flatten_observations = bool(flatten_observations)
        self.include_next_observation = bool(include_next_observation)
        self.skip_inactive_agents = bool(skip_inactive_agents)
        self.data = load_expert_dataset(self.path)
        self.metadata = self.data["metadata"]
        if str(self.metadata.get("dataset_mode", "")) != "scene":
            raise ValueError("SceneTransitionDataset requires a scene-mode expert dataset.")

        self._index: list[tuple[int, int, int]] = []
        for episode_idx, observations in enumerate(self.data["observations"]):
            scene_obs = np.asarray(observations)
            T, N = scene_obs.shape[:2]
            alive_mask = np.asarray(self.data["alive_mask"][episode_idx], dtype=bool)
            for step_idx in range(T):
                for agent_idx in range(N):
                    if self.skip_inactive_agents and not bool(alive_mask[step_idx, agent_idx]):
                        continue
                    self._index.append((episode_idx, step_idx, agent_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        episode_idx, step_idx, agent_idx = self._index[index]
        obs = np.asarray(
            self.data["observations"][episode_idx][step_idx][agent_idx],
            dtype=np.float32,
        )
        next_obs = np.asarray(
            self.data["next_observations"][episode_idx][step_idx][agent_idx],
            dtype=np.float32,
        )
        action = (
            np.asarray(self.data["actions"][episode_idx][step_idx][agent_idx], dtype=np.int64)
            if str(self.metadata.get("action_mode", "")) == "discrete"
            else np.asarray(self.data["actions"][episode_idx][step_idx][agent_idx], dtype=np.float32)
        )

        if self.flatten_observations:
            obs = obs.reshape(-1)
            next_obs = next_obs.reshape(-1)

        item: dict[str, torch.Tensor | int | str] = {
            "observation": torch.as_tensor(obs, dtype=torch.float32),
            "action": torch.as_tensor(action),
            "reward": torch.tensor(
                float(self.data["rewards"][episode_idx][step_idx]),
                dtype=torch.float32,
            ),
            "done": torch.tensor(bool(self.data["dones"][episode_idx][step_idx]), dtype=torch.bool),
            "episode_id": int(self.data["episode_id"][episode_idx]),
            "timestep": int(self.data["timesteps"][episode_idx][step_idx]),
            "agent_id": int(self.data["agent_ids"][episode_idx][agent_idx]),
            "agent_index": int(agent_idx),
            "episode_name": str(self.data["episode_name"][episode_idx]),
            "scenario_id": str(self.data["scenario_id"][episode_idx]),
            "alive": torch.tensor(
                bool(self.data["alive_mask"][episode_idx][step_idx][agent_idx]),
                dtype=torch.bool,
            ),
        }
        if self.include_next_observation:
            item["next_observation"] = torch.as_tensor(next_obs, dtype=torch.float32)
        return item


class ExpertTransitionDataset(Dataset):
    """
    Flat transition dataset backed by episode-preserving expert trajectories.

    This gives a PyTorch-friendly interface for pointwise GAIL baselines while keeping
    the original episode structure available for future sequence models.
    """

    def __init__(
        self,
        path: str,
        *,
        flatten_observations: bool = False,
        include_next_observation: bool = True,
    ) -> None:
        self.path = os.path.abspath(path)
        self.flatten_observations = bool(flatten_observations)
        self.include_next_observation = bool(include_next_observation)
        self.data = load_expert_dataset(self.path)
        self.metadata = self.data["metadata"]

        self._index: list[tuple[int, int]] = []
        for episode_idx, observations in enumerate(self.data["observations"]):
            T = int(np.asarray(observations).shape[0])
            self._index.extend((episode_idx, step_idx) for step_idx in range(T))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        episode_idx, step_idx = self._index[index]
        obs = np.asarray(self.data["observations"][episode_idx][step_idx], dtype=np.float32)
        if str(self.metadata.get("action_mode", "")) == "discrete":
            action = np.asarray(self.data["actions"][episode_idx][step_idx], dtype=np.int64)
        else:
            action = np.asarray(self.data["actions"][episode_idx][step_idx], dtype=np.float32)
        reward = float(self.data["rewards"][episode_idx][step_idx])
        done = bool(self.data["dones"][episode_idx][step_idx])

        if self.flatten_observations:
            obs = obs.reshape(-1)

        item: dict[str, torch.Tensor | int | str] = {
            "observation": torch.as_tensor(obs, dtype=torch.float32),
            "action": torch.as_tensor(action),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.bool),
            "episode_id": int(self.data["episode_id"][episode_idx]),
            "timestep": int(self.data["timesteps"][episode_idx][step_idx]),
            "episode_name": str(self.data["episode_name"][episode_idx]),
            "ego_id": int(self.data["ego_id"][episode_idx]),
            "scenario_id": str(self.data["scenario_id"][episode_idx]),
        }

        if self.include_next_observation:
            next_obs = np.asarray(
                self.data["next_observations"][episode_idx][step_idx],
                dtype=np.float32,
            )
            if self.flatten_observations:
                next_obs = next_obs.reshape(-1)
            item["next_observation"] = torch.as_tensor(next_obs, dtype=torch.float32)

        return item

    def episode(self, episode_idx: int) -> dict[str, Any]:
        """Return one full episode as numpy arrays plus metadata."""
        return {
            "episode_id": int(self.data["episode_id"][episode_idx]),
            "scenario_id": str(self.data["scenario_id"][episode_idx]),
            "episode_name": str(self.data["episode_name"][episode_idx]),
            "ego_id": int(self.data["ego_id"][episode_idx]),
            "source_split": str(self.data["source_split"][episode_idx]),
            "observations": np.asarray(self.data["observations"][episode_idx], dtype=np.float32),
            "actions": (
                np.asarray(self.data["actions"][episode_idx], dtype=np.int64)
                if str(self.metadata.get("action_mode", "")) == "discrete"
                else np.asarray(self.data["actions"][episode_idx], dtype=np.float32)
            ),
            "next_observations": np.asarray(self.data["next_observations"][episode_idx], dtype=np.float32),
            "dones": np.asarray(self.data["dones"][episode_idx], dtype=bool),
            "rewards": np.asarray(self.data["rewards"][episode_idx], dtype=np.float32),
            "timesteps": np.asarray(self.data["timesteps"][episode_idx], dtype=np.int32),
        }
