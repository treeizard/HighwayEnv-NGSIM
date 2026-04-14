#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import logger as imitation_logger
from imitation.util.networks import RunningNorm


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from highway_env.imitation.expert_dataset import (
    ExpertTransitionDataset,
    SceneTransitionDataset,
    load_expert_dataset,
    register_ngsim_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a discrete-action PS-GAIL style baseline on NGSIM using the repo's "
            "expert replay environment and saved expert dataset."
        )
    )
    parser.add_argument(
        "--expert-data",
        type=str,
        default="expert_data/ngsim_single_train_episode_discrete.npz",
        help="Path to a saved expert dataset (.npz).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="ps_gail_discrete",
        help="Run name under logs/ps_gail_discrete/.",
    )
    parser.add_argument("--scene", type=str, default="us-101")
    parser.add_argument("--episode-root", type=str, default="highway_env/data/processed_20s")
    parser.add_argument(
        "--prebuilt-split",
        choices=["train", "val", "test"],
        default="train",
        help="Replay split used for generator rollouts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--simulation-frequency", type=int, default=10)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--max-surrounding", default=32)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--maximum-range", type=float, default=64.0)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--demo-batch-size", type=int, default=512)
    parser.add_argument("--disc-updates-per-round", type=int, default=4)
    parser.add_argument("--gen-replay-buffer-capacity", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument(
        "--dummy-vec",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv for easier debugging.",
    )
    parser.add_argument(
        "--skip-inactive-agents",
        action="store_true",
        default=True,
        help=(
            "When loading a scene dataset, remove transitions whose alive_mask is false "
            "before training the discriminator and policy."
        ),
    )
    parser.add_argument(
        "--keep-inactive-agents",
        dest="skip_inactive_agents",
        action="store_false",
        help="Keep inactive padded scene slots in the expert transition set.",
    )
    return parser.parse_args()


def _dataset_items_to_transitions(items: list[dict[str, Any]]) -> Transitions:
    obs_parts: list[np.ndarray] = []
    act_parts: list[np.ndarray] = []
    next_obs_parts: list[np.ndarray] = []
    done_parts: list[np.ndarray] = []
    info_parts: list[dict[str, Any]] = []

    for item in items:
        obs_parts.append(np.asarray(item["observation"], dtype=np.float32).reshape(-1))
        act_parts.append(np.asarray(item["action"], dtype=np.int64).reshape(()))
        next_obs_parts.append(np.asarray(item["next_observation"], dtype=np.float32).reshape(-1))
        done_parts.append(np.asarray(item["done"], dtype=bool).reshape(()))

        info = {
            "episode_id": int(item["episode_id"]),
            "timestep": int(item["timestep"]),
            "episode_name": str(item["episode_name"]),
            "scenario_id": str(item["scenario_id"]),
        }
        if "ego_id" in item:
            info["ego_id"] = int(item["ego_id"])
        if "agent_id" in item:
            info["agent_id"] = int(item["agent_id"])
            info["agent_index"] = int(item["agent_index"])
        if "alive" in item:
            info["alive"] = bool(item["alive"])
        info_parts.append(info)

    if not obs_parts:
        raise RuntimeError("Expert dataset produced zero transitions after flattening.")

    obs = np.asarray(obs_parts, dtype=np.float32)
    acts = np.asarray(act_parts, dtype=np.int64)
    next_obs = np.asarray(next_obs_parts, dtype=np.float32)
    dones = np.asarray(done_parts, dtype=bool)
    infos = np.asarray(info_parts, dtype=object)
    return Transitions(obs=obs, acts=acts, infos=infos, next_obs=next_obs, dones=dones)


def load_expert_transitions(
    path: str,
    *,
    skip_inactive_agents: bool = True,
) -> tuple[Transitions, dict[str, Any], dict[str, Any]]:
    dataset = load_expert_dataset(path)
    metadata = dataset["metadata"]
    if str(metadata.get("action_mode", "")).lower() != "discrete":
        raise ValueError("This training script expects a discrete-action expert dataset.")
    dataset_mode = str(metadata.get("dataset_mode", "per_vehicle"))

    summary: dict[str, Any] = {"dataset_mode": dataset_mode}
    if dataset_mode == "scene":
        raw_alive = np.concatenate(
            [
                np.asarray(dataset["alive_mask"][episode_idx], dtype=bool).reshape(-1)
                for episode_idx in range(len(dataset["alive_mask"]))
            ]
        )
        summary["alive_true_count"] = int(raw_alive.sum())
        summary["alive_total_count"] = int(raw_alive.size)
        summary["alive_ratio"] = (
            float(raw_alive.sum()) / float(raw_alive.size) if raw_alive.size else 0.0
        )

        transition_dataset = SceneTransitionDataset(
            path,
            flatten_observations=True,
            include_next_observation=True,
            skip_inactive_agents=skip_inactive_agents,
        )
        transitions = _dataset_items_to_transitions(
            [transition_dataset[idx] for idx in range(len(transition_dataset))]
        )
        summary["flattened_transition_count"] = int(len(transition_dataset))
        summary["skip_inactive_agents"] = bool(skip_inactive_agents)
    elif dataset_mode == "per_vehicle":
        transition_dataset = ExpertTransitionDataset(
            path,
            flatten_observations=True,
            include_next_observation=True,
        )
        transitions = _dataset_items_to_transitions(
            [transition_dataset[idx] for idx in range(len(transition_dataset))]
        )
        summary["flattened_transition_count"] = int(len(transition_dataset))
    else:
        raise ValueError(f"Unsupported dataset_mode={dataset_mode!r}")
    return transitions, metadata, summary


def make_env_factory(
    *,
    scene: str,
    episode_root: str,
    prebuilt_split: str,
    max_surrounding: str | int,
    cells: int,
    maximum_range: float,
    simulation_frequency: int,
    policy_frequency: int,
    max_episode_steps: int,
    seed: int,
):
    def _factory(rank: int):
        def _init():
            config = {
                "scene": str(scene),
                "episode_root": str(episode_root),
                "prebuilt_split": str(prebuilt_split),
                "observation": {
                    "type": "LidarObservation",
                    "cells": int(cells),
                    "maximum_range": float(maximum_range),
                    "normalize": True,
                },
                "action": {"type": "DiscreteSteerMetaAction"},
                "action_mode": "discrete",
                "simulation_frequency": int(simulation_frequency),
                "policy_frequency": int(policy_frequency),
                "max_episode_steps": int(max_episode_steps),
                "expert_test_mode": False,
                "truncate_to_trajectory_length": True,
                "max_surrounding": max_surrounding,
                "offscreen_rendering": True,
                "seed": int(seed + rank),
            }
            env = gym.make("NGSim-US101-v0", config=config)
            env = FlattenObservation(env)
            env.reset(seed=seed + rank)
            return env

        return _init

    return _factory


def build_vec_env(args: argparse.Namespace, training: bool):
    factory = make_env_factory(
        scene=args.scene,
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        max_surrounding=args.max_surrounding,
        cells=args.cells,
        maximum_range=args.maximum_range,
        simulation_frequency=args.simulation_frequency,
        policy_frequency=args.policy_frequency,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed + (0 if training else 10_000),
    )

    if args.dummy_vec or args.n_envs <= 1:
        env = DummyVecEnv([factory(0)])
    else:
        env = SubprocVecEnv([factory(rank) for rank in range(args.n_envs)])
    return VecMonitor(env)


def evaluate_generator(learner: PPO, eval_env, n_eval_episodes: int) -> tuple[float, float]:
    mean_reward, std_reward = evaluate_policy(
        model=learner,
        env=eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        return_episode_rewards=False,
    )
    return float(mean_reward), float(std_reward)


def main() -> None:
    args = parse_args()
    register_ngsim_env()

    expert_data_path = os.path.abspath(args.expert_data)
    run_dir = os.path.abspath(os.path.join("logs", "ps_gail_discrete", args.run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    demos, metadata, data_summary = load_expert_transitions(
        expert_data_path,
        skip_inactive_agents=args.skip_inactive_agents,
    )
    effective_demo_batch_size = min(int(args.demo_batch_size), int(len(demos.obs)))
    if effective_demo_batch_size < int(args.demo_batch_size):
        print(
            f"Reducing demo_batch_size from {args.demo_batch_size} to "
            f"{effective_demo_batch_size} because the dataset is small."
        )
    print(f"Loaded expert dataset: {expert_data_path}")
    print(
        f"expert_transitions={len(demos.obs)} "
        f"dataset_episodes={metadata['num_dataset_episodes']} "
        f"dataset_mode={metadata.get('dataset_mode', 'per_vehicle')} "
        f"observation_shape={tuple(metadata['observation_shape'])}"
    )
    if metadata.get("dataset_mode") == "scene":
        print(
            f"alive_mask true/total={data_summary['alive_true_count']}/"
            f"{data_summary['alive_total_count']} "
            f"(ratio={data_summary['alive_ratio']:.3f}) "
            f"skip_inactive_agents={data_summary['skip_inactive_agents']}"
        )

    train_env = build_vec_env(args, training=True)
    eval_env = build_vec_env(args, training=False)

    learner = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=run_dir,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        policy_kwargs={"net_arch": [256, 256]},
    )

    reward_net = BasicRewardNet(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        normalize_input_layer=RunningNorm,
    )

    gail_logger = imitation_logger.configure(
        folder=os.path.join(run_dir, "gail_metrics"),
        format_strs=["stdout", "tensorboard"],
    )

    trainer = GAIL(
        demonstrations=demos,
        demo_batch_size=effective_demo_batch_size,
        gen_replay_buffer_capacity=args.gen_replay_buffer_capacity,
        n_disc_updates_per_round=args.disc_updates_per_round,
        venv=train_env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        custom_logger=gail_logger,
    )

    print(f"Training discrete PS-GAIL baseline for {args.total_timesteps} timesteps")
    print(f"TensorBoard: tensorboard --logdir {run_dir}")

    best_eval_reward = -np.inf

    def _on_round(round_idx: int) -> None:
        sampled_steps = int((round_idx + 1) * trainer.gen_train_timesteps)
        if sampled_steps % max(1, args.checkpoint_every) == 0:
            learner.save(os.path.join(ckpt_dir, f"ppo_generator_step_{sampled_steps}"))

        if sampled_steps % max(1, args.eval_every) == 0:
            mean_reward, std_reward = evaluate_generator(
                learner=learner,
                eval_env=eval_env,
                n_eval_episodes=args.eval_episodes,
            )
            nonlocal_best = _on_round.best_eval_reward
            print(
                f"[eval] sampled_steps={sampled_steps} "
                f"mean_reward={mean_reward:.3f} std_reward={std_reward:.3f}"
            )
            if mean_reward > nonlocal_best:
                _on_round.best_eval_reward = mean_reward
                learner.save(os.path.join(eval_dir, "best_model"))

    _on_round.best_eval_reward = best_eval_reward

    try:
        trainer.train(total_timesteps=args.total_timesteps, callback=_on_round)
    finally:
        learner.save(os.path.join(run_dir, "ppo_generator_final"))
        train_env.close()
        eval_env.close()

    print(f"Saved final generator policy under: {run_dir}")


if __name__ == "__main__":
    main()
