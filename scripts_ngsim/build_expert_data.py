#!/usr/bin/env python3
"""
Generate an expert dataset from NGSimEnv using continuous expert actions.

- Uses NGSimEnv.expert_action_at(policy_step) as the expert policy.
- Steps the environment with those continuous actions.
- Records (obs, act, next_obs, done) tuples.
- Saves to a compressed NPZ file, ready for GAIL (imitation library) consumption.
"""

from __future__ import annotations

import sys
import os
import argparse
from typing import Dict, List

import numpy as np
from gymnasium.wrappers import FlattenObservation
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from highway_env.envs.ngsim_env import NGSimEnv  # adjust path if needed


def collect_expert_rollouts(
    env: NGSimEnv,
    n_episodes: int = 200,
    max_steps_per_episode: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Run NGSimEnv with expert actions and collect transitions.

    We assume:
      - env.reset() -> (obs, info)
      - env.step(action) -> (obs, reward, terminated, truncated, info)
      - env.expert_action_at(policy_step) -> np.array([steering, accel])

    Returns:
      {
        "obs":      [N, obs_dim],
        "acts":     [N, act_dim],
        "next_obs": [N, obs_dim],
        "dones":    [N,],
      }
    """

    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    next_obs_buf: List[np.ndarray] = []
    done_buf: List[bool] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        obs = np.asarray(obs, dtype=np.float32).ravel()

        done = False
        t = 0
        steps_in_ep = 0

        print(f"[ExpertCollect] Episode {ep + 1}/{n_episodes}")

        # Quick check: ensure this episode actually has expert actions
        try:
            _ = env.expert_action_at(policy_step=0)
        except RuntimeError as e:
            print(f"[ExpertCollect]   Skipping episode: no expert actions ({e})")
            continue

        while not done:
            # ----- 1) Expert action for this policy step -----
            try:
                a = env.expert_action_at(policy_step=t)
            except RuntimeError as e:
                print(f"[ExpertCollect]   Expert missing at step {t}: {e}")
                break

            a = np.asarray(a, dtype=np.float32)

            # ----- 2) Step environment with expert action -----
            next_obs, reward, terminated, truncated, info = env.step(a)
            next_obs = np.asarray(next_obs, dtype=np.float32).ravel()

            done = bool(terminated or truncated)
            steps_in_ep += 1

            # ----- 3) Store transition -----
            obs_buf.append(obs.copy())
            act_buf.append(a.copy())
            next_obs_buf.append(next_obs.copy())
            done_buf.append(done)

            obs = next_obs
            t += 1

            if max_steps_per_episode is not None and steps_in_ep >= max_steps_per_episode:
                break

        print(f"[ExpertCollect]   Collected {steps_in_ep} steps in this episode.")

    if len(obs_buf) == 0:
        raise RuntimeError("No expert transitions were collected. Check your episodes / actions.")

    data = {
        "obs": np.stack(obs_buf, axis=0).astype(np.float32),
        "acts": np.stack(act_buf, axis=0).astype(np.float32),
        "next_obs": np.stack(next_obs_buf, axis=0).astype(np.float32),
        "dones": np.array(done_buf, dtype=bool),
    }

    print(
        f"[ExpertCollect] Finished: {data['obs'].shape[0]} transitions "
        f"from up to {n_episodes} episodes."
    )
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate NGSim expert dataset (continuous actions).")
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of expert episodes to roll out.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode (policy steps).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="expert_data/ngsim_expert_continuous.npz",
        help="Output NPZ path.",
    )
    args = parser.parse_args()

    # ----- Configure NGSimEnv -----
    config = {
        "scene": "us-101",
        "episode_root": "highway_env/data/processed_10s",
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "max_episode_steps": 300,
        "log_overlaps": False,
        # "ego_vehicle_ID": None,  # keep random or fix if you want
    }

    env = NGSimEnv(config=config)
    env = FlattenObservation(env)

    # ----- Collect expert transitions -----
    dataset = collect_expert_rollouts(
        env,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
    )

    # ----- Save dataset -----
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    print(f"[ExpertCollect] Saved dataset to: {out_path}")


if __name__ == "__main__":
    main()
