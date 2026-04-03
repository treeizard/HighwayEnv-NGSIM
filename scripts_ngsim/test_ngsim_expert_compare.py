#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import register

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")
except Exception:
    pass


def sample_case(scene: str, rng: np.random.Generator, episode_root: str) -> tuple[str, int]:
    prebuilt = np.load(
        os.path.join(episode_root, scene, "prebuilt", "veh_ids_train.npy"),
        allow_pickle=True,
    ).item()
    episodes = sorted(prebuilt)
    episode_name = str(rng.choice(episodes))
    ego_id = int(rng.choice(prebuilt[episode_name]))
    return episode_name, ego_id


def run_case(
    scene: str,
    episode_name: str,
    ego_id: int,
    episode_root: str,
    max_steps: int = 80,
    max_surrounding: int = 40,
) -> dict:
    cfg = {
        "scene": scene,
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteSteerMetaAction"},
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "episode_root": episode_root,
        "simulation_period": {"episode_name": episode_name},
        "ego_vehicle_ID": ego_id,
        "max_surrounding": max_surrounding,
        "expert_test_mode": True,
        "show_trajectories": False,
        "offscreen_rendering": True,
    }

    env = gym.make("NGSim-US101-v0", render_mode="rgb_array", config=cfg)
    env.reset(seed=0)

    steps = 0
    while steps < max_steps:
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        if terminated or truncated:
            break

    metrics = env.unwrapped.expert_replay_metrics()
    ref = np.asarray(env.unwrapped._expert_ref_xy_pol, dtype=float)
    rep = np.asarray(env.unwrapped._replay_xy_pol, dtype=float)
    crashed = bool(env.unwrapped.vehicle.crashed)
    env.close()

    return {
        "scene": scene,
        "episode_name": episode_name,
        "ego_id": ego_id,
        "steps": steps,
        "metrics": metrics,
        "ref": ref,
        "rep": rep,
        "crashed": crashed,
    }


def plot_case(result: dict, out_dir: str) -> str:
    scene = result["scene"]
    episode_name = result["episode_name"]
    ego_id = result["ego_id"]
    metrics = result["metrics"]
    ref = result["ref"]
    rep = result["rep"]

    plt.figure(figsize=(10, 6))
    plt.plot(ref[:, 0], ref[:, 1], "k--", linewidth=2, alpha=0.7, label="Reference")
    plt.plot(rep[:, 0], rep[:, 1], color="tab:blue", linewidth=1.6, label="Discrete Expert Replay")
    if len(rep) > 0:
        plt.scatter(rep[0, 0], rep[0, 1], color="green", s=24, label="Start")
        plt.scatter(rep[-1, 0], rep[-1, 1], color="red", s=32, marker="x", label="End")

    plt.title(
        f"{scene} | {episode_name} | ego={ego_id} | "
        f"ADE={metrics['ADE_m']:.3f} FDE={metrics['FDE_m']:.3f}"
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"expert_compare_{scene}_{episode_name}_ego{ego_id}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    episode_root = "highway_env/data/processed_10s"
    out_dir = os.path.abspath("./expert_compare_outputs")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(7)
    scenes = ["us-101", "japanese"]

    for scene in scenes:
        episode_name, ego_id = sample_case(scene, rng, episode_root)
        result = run_case(scene, episode_name, ego_id, episode_root)
        plot_path = plot_case(result, out_dir)
        metrics = result["metrics"]
        print(
            f"{scene}: episode={episode_name} ego_id={ego_id} "
            f"crashed={result['crashed']} steps={result['steps']} T={metrics['T']} "
            f"ADE={metrics['ADE_m']:.3f} FDE={metrics['FDE_m']:.3f}"
        )
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
