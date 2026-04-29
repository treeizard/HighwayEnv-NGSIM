from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env

from .config import PSGAILConfig


def observation_config(cfg: PSGAILConfig) -> dict[str, Any]:
    return {
        "type": "LidarCameraObservations",
        "lidar": {
            "cells": int(cfg.cells),
            "maximum_range": float(cfg.maximum_range),
            "normalize": True,
        },
        "camera": {
            "cells": 21,
            "maximum_range": float(cfg.maximum_range),
            "field_of_view": np.pi / 2,
            "normalize": True,
        },
    }


def make_training_env(cfg: PSGAILConfig, *, render_mode: str | None = None) -> gym.Env:
    register_ngsim_env()
    env_cfg = build_env_config(
        scene=cfg.scene,
        action_mode=str(cfg.action_mode),
        episode_root=cfg.episode_root,
        prebuilt_split=cfg.prebuilt_split,
        percentage_controlled_vehicles=cfg.percentage_controlled_vehicles,
        control_all_vehicles=cfg.control_all_vehicles,
        max_surrounding=cfg.max_surrounding,
        observation_config=observation_config(cfg),
        simulation_frequency=cfg.simulation_frequency,
        policy_frequency=cfg.policy_frequency,
        max_episode_steps=cfg.max_episode_steps,
        seed=None,
        scene_dataset_collection_mode=False,
        allow_idm=cfg.allow_idm,
    )
    env_cfg["expert_test_mode"] = False
    env_cfg["disable_controlled_vehicle_collisions"] = not bool(cfg.enable_collision)
    env_cfg["terminate_when_all_controlled_crashed"] = bool(
        cfg.terminate_when_all_controlled_crashed
    )
    env_cfg["allow_idm"] = bool(cfg.allow_idm)
    env_cfg["crash_controlled_vehicles_offroad"] = True
    action_cfg = env_cfg.get("action", {})
    if str(cfg.action_mode).lower() == "discrete":
        nested_action_cfg = (
            action_cfg.get("action_config", {})
            if action_cfg.get("type") == "MultiAgentAction"
            else action_cfg
        )
        if nested_action_cfg.get("type") != "DiscreteSteerMetaAction":
            raise RuntimeError(
                "Discrete PS-GAIL must use DiscreteSteerMetaAction, got "
                f"{nested_action_cfg.get('type')!r}."
            )
    return gym.make(ENV_ID, render_mode=render_mode, config=env_cfg)


def controlled_vehicle_snapshot(env: gym.Env) -> tuple[list[int], np.ndarray]:
    vehicles = list(getattr(env.unwrapped, "controlled_vehicles", []))
    ids = [int(getattr(vehicle, "vehicle_ID", idx)) for idx, vehicle in enumerate(vehicles)]
    states = np.asarray(
        [
            [
                float(np.asarray(vehicle.position, dtype=np.float32)[0]),
                float(np.asarray(vehicle.position, dtype=np.float32)[1]),
                float(getattr(vehicle, "speed", 0.0)),
            ]
            for vehicle in vehicles
        ],
        dtype=np.float32,
    )
    return ids, states
