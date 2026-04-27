#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scripts_gail.ps_gail.config import PSGAILConfig  # noqa: E402
from scripts_gail.ps_gail.envs import make_training_env  # noqa: E402
from scripts_gail.ps_gail.models import SharedActorCritic  # noqa: E402
from scripts_gail.ps_gail.observations import (  # noqa: E402
    flatten_agent_observations,
    policy_observations_from_flat,
)
from scripts_gail.ps_gail.trainer import infer_policy_obs_dim  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the PS-GAIL training environment: collision flags, episode "
            "sampling, surrounding vehicle spawning, termination, and optional video."
        )
    )
    parser.add_argument("--expert-data", default="expert_data/ngsim_ps_traj_expert_discrete_54902119")
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--max-expert-samples", type=int, default=100_000)
    parser.add_argument(
        "--trajectory-frame",
        choices=["relative", "absolute"],
        default="relative",
        help="Trajectory coordinate frame used by the discriminator.",
    )
    parser.add_argument(
        "--compare-trajectory-frames",
        action="store_true",
        help="Print absolute-vs-relative expert trajectory-state statistics.",
    )
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--control-all-vehicles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--percentage-controlled-vehicles", type=float, default=0.1)
    parser.add_argument("--enable-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-idm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--terminate-when-all-controlled-crashed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-terminate-when-all-controlled-crashed to test early termination on any controlled crash.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--simulation-frequency", type=int, default=10)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--maximum-range", type=float, default=64.0)
    parser.add_argument("--checkpoint", default=None, help="Optional PS-GAIL checkpoint/final.pt for policy actions.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-path", default="logs/simple_ps_gail/diagnostics/policy_interaction.mp4")
    parser.add_argument("--video-episode-index", type=int, default=0)
    parser.add_argument("--screen-width", type=int, default=1200)
    parser.add_argument("--screen-height", type=int, default=600)
    parser.add_argument("--scaling", type=float, default=5.5)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def dataset_files(path: str) -> list[str]:
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            return [
                os.path.join(path, item["dataset_file"])
                for item in manifest.get("episodes", [])
                if "dataset_file" in item
            ]
        return sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".npz"))
    return [path]


def manifest_items(path: str) -> list[dict[str, Any]] | None:
    if not os.path.isdir(path):
        return None
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    items = []
    for entry in manifest.get("episodes", []):
        dataset_file = entry.get("dataset_file")
        if not dataset_file:
            continue
        items.append(
            {
                "path": os.path.join(path, dataset_file),
                "episode_name": str(entry.get("episode_name", os.path.basename(dataset_file))),
                "num_samples": int(entry.get("num_samples", 0)),
                "num_controlled_vehicle_ids": int(entry.get("controlled_vehicles", 0) or 0),
                "num_done_samples": 0,
            }
        )
    return items


def metadata_from_npz(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        metadata = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
        observations = np.asarray(data["observations"])
        vehicle_ids = np.asarray(data["vehicle_ids"]) if "vehicle_ids" in data.files else np.asarray([])
        dones = np.asarray(data["dones"], dtype=bool) if "dones" in data.files else np.asarray([], dtype=bool)
    return {
        "path": path,
        "episode_name": str(metadata.get("episode_name", os.path.basename(path))),
        "num_samples": int(observations.shape[0]),
        "num_controlled_vehicle_ids": int(len(set(map(int, vehicle_ids.tolist())))) if vehicle_ids.size else 0,
        "num_done_samples": int(dones.sum()) if dones.size else 0,
    }


def uniform_sample_counts(items: list[dict[str, Any]], max_samples: int, seed: int) -> list[int]:
    total = int(sum(item["num_samples"] for item in items))
    if total <= 0:
        return [0 for _item in items]
    if int(max_samples) <= 0 or int(max_samples) >= total:
        return [int(item["num_samples"]) for item in items]

    rng = np.random.default_rng(int(seed))
    global_indices = np.sort(
        rng.choice(total, size=int(max_samples), replace=False).astype(np.int64, copy=False)
    )
    used_counts = []
    cursor = 0
    for item in items:
        count = int(item["num_samples"])
        start = cursor
        end = cursor + count
        left = int(np.searchsorted(global_indices, start, side="left"))
        right = int(np.searchsorted(global_indices, end, side="left"))
        used_counts.append(int(right - left))
        cursor = end
    return used_counts


def audit_expert_data(path: str, max_samples: int, seed: int) -> dict[str, Any]:
    items = manifest_items(path)
    if items is None:
        files = dataset_files(path)
        items = [metadata_from_npz(file_path) for file_path in files]

    used_counts = uniform_sample_counts(items, max_samples=max_samples, seed=seed)
    used: list[dict[str, Any]] = []
    for item, take in zip(items, used_counts):
        if int(take) <= 0:
            continue
        used.append({**item, "samples_used_by_loader": int(take)})
    episode_counts = Counter(item["episode_name"] for item in items)
    used_episode_counts = Counter(item["episode_name"] for item in used)
    return {
        "num_files": len(items),
        "total_samples": int(sum(item["num_samples"] for item in items)),
        "unique_episode_names": len(episode_counts),
        "duplicate_episode_names": {key: value for key, value in episode_counts.items() if value > 1},
        "max_samples": int(max_samples),
        "sampling": "uniform_without_replacement",
        "files_used_by_loader": len(used),
        "samples_used_by_loader": int(sum(item["samples_used_by_loader"] for item in used)),
        "unique_episode_names_used_by_loader": len(used_episode_counts),
        "used_files": used,
    }


def build_cfg(args: argparse.Namespace) -> PSGAILConfig:
    return PSGAILConfig(
        expert_data=args.expert_data,
        scene=args.scene,
        episode_root=args.episode_root,
        prebuilt_split=args.prebuilt_split,
        seed=int(args.seed),
        rollout_steps=int(args.steps_per_episode),
        max_expert_samples=int(args.max_expert_samples),
        trajectory_frame=str(args.trajectory_frame),
        max_surrounding=args.max_surrounding,
        control_all_vehicles=bool(args.control_all_vehicles),
        percentage_controlled_vehicles=float(args.percentage_controlled_vehicles),
        enable_collision=bool(args.enable_collision),
        terminate_when_all_controlled_crashed=bool(args.terminate_when_all_controlled_crashed),
        allow_idm=bool(args.allow_idm),
        cells=int(args.cells),
        maximum_range=float(args.maximum_range),
        simulation_frequency=int(args.simulation_frequency),
        policy_frequency=int(args.policy_frequency),
        max_episode_steps=int(args.max_episode_steps),
        device=args.device,
    )


def load_policy(checkpoint_path: str | None, obs_dim: int, hidden_size: int, device: torch.device):
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    policy = SharedActorCritic(int(obs_dim), int(hidden_size)).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def policy_or_random_action(env: gym.Env, policy, device: torch.device, obs: Any) -> tuple[int, ...]:
    if policy is None:
        sample = env.action_space.sample()
        return tuple(int(v) for v in sample) if isinstance(sample, tuple) else (int(sample),)
    obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
    with torch.no_grad():
        logits, _values = policy(torch.as_tensor(obs_agents, dtype=torch.float32, device=device))
        actions = Categorical(logits=logits).sample().cpu().numpy().astype(int)
    return tuple(int(action) for action in actions.tolist())


def save_video(path: str, frames: list[np.ndarray], fps: int) -> str | None:
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError:
        print("video: imageio is not installed; skipping MP4 export")
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with imageio.get_writer(path, fps=max(1, int(fps))) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return os.path.abspath(path)


def audit_env(args: argparse.Namespace, cfg: PSGAILConfig) -> dict[str, Any]:
    render_mode = "rgb_array" if args.save_video else None
    env_cfg = replace(cfg)
    env = make_training_env(env_cfg)
    if args.save_video:
        env.close()
        env_cfg = replace(cfg)
        env = make_training_env(env_cfg)
        env.unwrapped.render_mode = render_mode
        env.unwrapped.config["offscreen_rendering"] = True
        env.unwrapped.config["screen_width"] = int(args.screen_width)
        env.unwrapped.config["screen_height"] = int(args.screen_height)
        env.unwrapped.config["scaling"] = float(args.scaling)

    device = resolve_device(args.device)
    obs_dim = infer_policy_obs_dim(env)
    policy = load_policy(args.checkpoint, obs_dim, cfg.hidden_size, device)
    frames: list[np.ndarray] = []
    episode_summaries: list[dict[str, Any]] = []

    try:
        for episode_idx in range(int(args.episodes)):
            obs, _info = env.reset(seed=int(args.seed) + episode_idx)
            base = env.unwrapped
            if args.save_video and episode_idx == int(args.video_episode_index):
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))

            controlled = list(getattr(base, "controlled_vehicles", []))
            road_vehicles = list(getattr(base.road, "vehicles", []))
            surrounding = [vehicle for vehicle in road_vehicles if vehicle not in controlled]
            collision_flags_ok = all(
                bool(getattr(vehicle, "check_collisions", False))
                and bool(getattr(vehicle, "collidable", False))
                for vehicle in controlled
            )
            episode = {
                "episode_name": str(getattr(base, "episode_name", "")),
                "controlled_count": len(controlled),
                "surrounding_count_at_reset": len(surrounding),
                "road_vehicle_count_at_reset": len(road_vehicles),
                "collision_flags_ok_at_reset": bool(collision_flags_ok),
                "steps": 0,
                "terminated": False,
                "truncated": False,
                "any_controlled_crash": False,
                "all_controlled_crashed": False,
                "first_crash_step": None,
                "final_alive_count": len(controlled),
            }

            for step_idx in range(int(args.steps_per_episode)):
                action = policy_or_random_action(env, policy, device, obs)
                obs, _reward, terminated, truncated, info = env.step(action)
                if args.save_video and episode_idx == int(args.video_episode_index):
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))

                crashes = [bool(flag) for flag in info.get("controlled_vehicle_crashes", [])]
                any_crash = any(crashes)
                if any_crash and episode["first_crash_step"] is None:
                    episode["first_crash_step"] = int(step_idx + 1)
                episode["any_controlled_crash"] = bool(episode["any_controlled_crash"] or any_crash)
                episode["all_controlled_crashed"] = bool(crashes and all(crashes))
                episode["final_alive_count"] = len(info.get("alive_controlled_vehicle_ids", []))
                episode["steps"] = int(step_idx + 1)
                episode["terminated"] = bool(terminated)
                episode["truncated"] = bool(truncated)
                if terminated or truncated:
                    break
            episode_summaries.append(episode)
    finally:
        env.close()

    video_path = save_video(args.video_path, frames, fps=int(cfg.policy_frequency)) if args.save_video else None
    return {
        "policy_source": os.path.abspath(args.checkpoint) if args.checkpoint else "random action_space.sample()",
        "obs_dim": int(obs_dim),
        "episodes": episode_summaries,
        "video_path": video_path,
    }


def print_report(dataset_report: dict[str, Any], env_report: dict[str, Any], cfg: PSGAILConfig) -> None:
    print("\n=== Expert Dataset ===")
    print(
        f"files={dataset_report['num_files']} total_samples={dataset_report['total_samples']} "
        f"unique_episode_names={dataset_report['unique_episode_names']}"
    )
    print(
        f"loader sampling={dataset_report['sampling']} max_samples={dataset_report['max_samples']} uses "
        f"{dataset_report['files_used_by_loader']} files, "
        f"{dataset_report['samples_used_by_loader']} samples, "
        f"{dataset_report['unique_episode_names_used_by_loader']} unique episode names"
    )
    if dataset_report["duplicate_episode_names"]:
        print(f"duplicate episode names in expert folder: {dataset_report['duplicate_episode_names']}")
    print("first files used by current loader order:")
    for item in dataset_report["used_files"][:10]:
        print(
            f"  {os.path.basename(item['path'])}: episode={item['episode_name']} "
            f"samples={item['num_samples']} used={item['samples_used_by_loader']} "
            f"vehicles={item['num_controlled_vehicle_ids']}"
        )

    print("\n=== Training Env ===")
    print(
        f"collision_enabled={cfg.enable_collision} "
        f"terminate_when_all_controlled_crashed={cfg.terminate_when_all_controlled_crashed} "
        f"control_all_vehicles={cfg.control_all_vehicles} max_surrounding={cfg.max_surrounding}"
    )
    print(f"policy_source={env_report['policy_source']} obs_dim={env_report['obs_dim']}")
    for idx, episode in enumerate(env_report["episodes"]):
        print(
            f"  ep{idx:02d} {episode['episode_name']}: "
            f"controlled={episode['controlled_count']} surrounding={episode['surrounding_count_at_reset']} "
            f"road={episode['road_vehicle_count_at_reset']} collision_flags_ok={episode['collision_flags_ok_at_reset']} "
            f"steps={episode['steps']} term={episode['terminated']} trunc={episode['truncated']} "
            f"any_crash={episode['any_controlled_crash']} first_crash={episode['first_crash_step']} "
            f"alive_final={episode['final_alive_count']}"
        )
    if env_report.get("video_path"):
        print(f"\nvideo={env_report['video_path']}")


def print_trajectory_frame_comparison(args: argparse.Namespace) -> None:
    from scripts_gail.ps_gail.data import load_expert_policy_and_disc_data

    print("\n=== Trajectory Frame Comparison ===")
    for frame in ("absolute", "relative"):
        _policy_obs, features, metadata = load_expert_policy_and_disc_data(
            args.expert_data,
            max_samples=int(args.max_expert_samples),
            seed=int(args.seed),
            trajectory_frame=frame,
        )
        traj = features[:, -3:]
        xy = traj[:, :2]
        print(
            f"{frame}: "
            f"samples={metadata['num_samples']} files={metadata['num_files_loaded']} "
            f"x_mean={xy[:, 0].mean():.2f} x_std={xy[:, 0].std():.2f} "
            f"x_range=({xy[:, 0].min():.2f},{xy[:, 0].max():.2f}) "
            f"y_mean={xy[:, 1].mean():.2f} y_std={xy[:, 1].std():.2f} "
            f"y_range=({xy[:, 1].min():.2f},{xy[:, 1].max():.2f}) "
            f"v_mean={traj[:, 2].mean():.2f} v_std={traj[:, 2].std():.2f}"
        )


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    dataset_report = audit_expert_data(
        args.expert_data,
        max_samples=int(args.max_expert_samples),
        seed=int(args.seed),
    )
    env_report = audit_env(args, cfg)
    print_report(dataset_report, env_report, cfg)
    if args.compare_trajectory_frames:
        print_trajectory_frame_comparison(args)


if __name__ == "__main__":
    main()
