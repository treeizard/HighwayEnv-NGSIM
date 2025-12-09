import os, sys
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


# Add in the Project Roots
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from highway_env.envs.ngsim_env import NGSimEnv  # adjust path as needed

SEED = 42


def load_npz_as_transitions(path: str) -> Transitions:
    data = np.load(path)

    obs = data["obs"]
    acts = data["acts"]
    next_obs = data["next_obs"]
    dones = data["dones"]

    infos = np.array([{} for _ in range(len(obs))], dtype=object)

    return Transitions(
        obs=obs,
        acts=acts,
        next_obs=next_obs,
        dones=dones,
        infos=infos,
    )


def make_ngsim_env():
    config = {
        "scene": "us-101",
        "episode_root": "highway_env/data/processed_10s",
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "max_episode_steps": 300,
        "log_overlaps": False,
    }
    env = NGSimEnv(config=config)
    env = FlattenObservation(env)
    return env


def main():
    rng = np.random.default_rng(SEED)

    # ---- Vec env ----
    env = make_vec_env(make_ngsim_env, n_envs=8, seed=SEED)
    print("env.observation_space.shape:", env.observation_space.shape)

    demos = load_npz_as_transitions("expert_data/ngsim_expert_continuous.npz")
    print("demos.obs.shape:", demos.obs.shape)
    print("demos.obs.ndim:", demos.obs.ndim)

    # ---- Learner (PPO) ----
    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=4e-4,
        gamma=0.95,
        n_steps = 1,
        n_epochs=5,
        seed=SEED,
        verbose=1,
        tensorboard_log="logs/ppo_ngsim"
    )

    # ---- Reward net (discriminator) ----
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # ---- GAIL trainer ----
    gail_trainer = GAIL(
        demonstrations=demos,              # <--- our Transitions object
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon = True
    )

    # ---- Evaluate before training ----
    env.seed(SEED)
    rew_before, _ = evaluate_policy(
        learner, env, n_eval_episodes=10, return_episode_rewards=True
    )

    print("mean reward BEFORE training:", np.mean(rew_before))

    # ---- Train ----
    gail_trainer.train(total_timesteps=200_000)  # bump up later if stable

    # ---- Evaluate after training ----
    env.seed(SEED)
    rew_after, _ = evaluate_policy(
        learner, env, n_eval_episodes=10, return_episode_rewards=True
    )

    print("mean reward AFTER training:", np.mean(rew_after))


if __name__ == "__main__":
    main()
