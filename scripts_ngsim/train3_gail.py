import os, sys
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from highway_env.envs.ngsim_env import NGSimEnv

SEED = 42


def load_npz_as_transitions(path: str) -> Transitions:
    data = np.load(path)
    obs = data["obs"].astype(np.float32)
    acts = data["acts"].astype(np.float32)
    next_obs = data["next_obs"].astype(np.float32)
    dones = data["dones"].astype(bool)

    # Optional: if you need to swap action columns due to old dataset ordering:
    # acts = acts[:, [1, 0]]

    infos = np.array([{} for _ in range(len(obs))], dtype=object)
    return Transitions(obs=obs, acts=acts, next_obs=next_obs, dones=dones, infos=infos)


def make_ngsim_env():
    config = {
        "scene": "us-101",
        "episode_root": "highway_env/data/processed_10s",

        "observation": {
            "type": "LidarObservation",
            "cells": 128,
            "maximum_range": 64,
            "normalize": True,
        },
        "action": {"type": "ContinuousAction"},

        "simulation_frequency": 10,
        "policy_frequency": 10,
        "max_episode_steps": 300,

        "expert_test_mode": False,
        "max_surrounding": 200000,
    }
    env = NGSimEnv(config=config)
    env = FlattenObservation(env)
    return env


def main():
    env = make_vec_env(make_ngsim_env, n_envs=8, seed=SEED)
    print("env.observation_space:", env.observation_space)
    print("env.action_space:", env.action_space)

    demos = load_npz_as_transitions("expert_data/ngsim_expert_continuous.npz")
    print("demos.obs.shape:", demos.obs.shape)
    print("demos.acts.shape:", demos.acts.shape)

    # Hard checks: must match exactly
    assert demos.obs.shape[1:] == env.observation_space.shape, (demos.obs.shape, env.observation_space.shape)
    assert demos.acts.shape[1:] == env.action_space.shape, (demos.acts.shape, env.action_space.shape)

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        n_steps=32,
        batch_size=256,
        n_epochs=5,
        ent_coef=0.0,
        learning_rate=4e-4,
        gamma=0.95,
        seed=SEED,
        verbose=1,
        tensorboard_log="logs/ppo_ngsim",
    )

    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    gail_trainer = GAIL(
        demonstrations=demos,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=50_000,
        n_disc_updates_per_round=4,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    gail_trainer.train(total_timesteps=200_000)


if __name__ == "__main__":
    main()
