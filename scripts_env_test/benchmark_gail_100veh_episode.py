#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.envs import make_training_env
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.observations import flatten_agent_observations, policy_observations_from_flat
from scripts_gail.ps_gail.trainer import (
    _actions_to_env_tuple,
    _masked_discrete_logits,
    _sample_policy_actions,
    central_critic_observation_dim,
    central_critic_observations,
    discrete_action_masks_from_env,
    infer_continuous_action_dim,
    policy_action_dim,
    policy_distribution_values_memory,
    recurrent_policy_enabled,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time one GAIL-style NGSIM rollout episode with a requested 100 controlled vehicles "
            "and 200 env steps."
        )
    )
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--requested-controlled-vehicles", type=float, default=100.0)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--policy-model", default="recurrent_transformer", choices=("mlp", "transformer", "recurrent_transformer"))
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-temporal-module", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transformer-temporal-kernel-size", type=int, default=5)
    parser.add_argument("--transformer-temporal-layers", type=int, default=1)
    parser.add_argument("--transformer-memory-tokens", type=int, default=8)
    parser.add_argument("--transformer-memory-context-length", type=int, default=32)
    parser.add_argument("--transformer-use-causal-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--centralized-critic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--central-critic-max-vehicles", type=int, default=64)
    parser.add_argument("--central-critic-include-local-obs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--central-critic-pooling", default="flat")
    parser.add_argument("--central-critic-attention-heads", type=int, default=4)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--worker-threads", type=int, default=2)
    parser.add_argument("--output", default="")
    parser.add_argument("--disable-progress", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def configure_threads(threads: int) -> None:
    threads = max(1, int(threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)


def make_cfg(args: argparse.Namespace) -> PSGAILConfig:
    return PSGAILConfig(
        action_mode="continuous",
        scene=str(args.scene),
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.split),
        seed=int(args.seed),
        control_all_vehicles=False,
        percentage_controlled_vehicles=float(args.requested_controlled_vehicles),
        max_surrounding=str(args.max_surrounding),
        max_episode_steps=int(args.steps),
        rollout_steps=int(args.steps),
        rollout_full_episodes=True,
        rollout_min_episodes=1,
        num_rollout_workers=1,
        rollout_worker_threads=max(1, int(args.worker_threads)),
        policy_model=str(args.policy_model),
        hidden_size=int(args.hidden_size),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_dropout=float(args.transformer_dropout),
        transformer_temporal_module=bool(args.transformer_temporal_module),
        transformer_temporal_kernel_size=int(args.transformer_temporal_kernel_size),
        transformer_temporal_layers=int(args.transformer_temporal_layers),
        transformer_memory_tokens=int(args.transformer_memory_tokens),
        transformer_memory_context_length=int(args.transformer_memory_context_length),
        transformer_use_causal_attention=bool(args.transformer_use_causal_attention),
        centralized_critic=bool(args.centralized_critic),
        central_critic_max_vehicles=int(args.central_critic_max_vehicles),
        central_critic_include_local_obs=bool(args.central_critic_include_local_obs),
        central_critic_pooling=str(args.central_critic_pooling),
        central_critic_attention_heads=int(args.central_critic_attention_heads),
        device=str(args.device),
    )


def load_policy_checkpoint(policy: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    path = str(checkpoint or "").strip()
    if not path:
        return
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload.get("policy_state_dict") if isinstance(payload, dict) else None
    if state_dict is None:
        raise RuntimeError(f"Checkpoint has no policy_state_dict: {path}")
    policy.load_state_dict(state_dict)


def summarize_times(values: list[float]) -> dict[str, float]:
    if not values:
        return {"total_s": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "total_s": float(arr.sum()),
        "mean_ms": float(arr.mean() * 1000.0),
        "p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "max_ms": float(arr.max() * 1000.0),
    }


def cuda_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}
    return {
        "allocated_mb": float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)),
        "reserved_mb": float(torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
    }


def main() -> None:
    args = parse_args()
    configure_threads(int(args.worker_threads))
    cfg = make_cfg(args)
    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: dict[str, list[float]] = {
        "obs_flatten": [],
        "critic_obs": [],
        "action_mask": [],
        "policy_forward": [],
        "action_to_env": [],
        "env_step": [],
        "step_total": [],
    }

    make_start = time.perf_counter()
    env = make_training_env(cfg)
    make_seconds = time.perf_counter() - make_start
    try:
        cfg.continuous_action_dim = infer_continuous_action_dim(env)
        reset_start = time.perf_counter()
        obs, info = env.reset(seed=int(cfg.seed))
        reset_seconds = time.perf_counter() - reset_start

        obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
        policy_obs_dim = int(obs_agents.shape[1])
        critic_obs_dim = central_critic_observation_dim(policy_obs_dim, cfg)
        policy = make_actor_critic(
            cfg.policy_model,
            policy_obs_dim,
            cfg.hidden_size,
            action_mode=str(cfg.action_mode),
            continuous_action_dim=int(cfg.continuous_action_dim),
            transformer_layers=int(cfg.transformer_layers),
            transformer_heads=int(cfg.transformer_heads),
            transformer_dropout=float(cfg.transformer_dropout),
            transformer_temporal_module=bool(cfg.transformer_temporal_module),
            transformer_temporal_kernel_size=int(cfg.transformer_temporal_kernel_size),
            transformer_temporal_layers=int(cfg.transformer_temporal_layers),
            transformer_memory_tokens=int(cfg.transformer_memory_tokens),
            transformer_memory_context_length=int(cfg.transformer_memory_context_length),
            transformer_use_causal_attention=bool(cfg.transformer_use_causal_attention),
            centralized_critic=bool(cfg.centralized_critic),
            critic_obs_dim=int(critic_obs_dim),
            central_critic_pooling=str(cfg.central_critic_pooling),
            central_critic_max_vehicles=int(cfg.central_critic_max_vehicles),
            central_critic_attention_heads=int(cfg.central_critic_attention_heads),
        ).to(device)
        load_policy_checkpoint(policy, str(args.checkpoint), device)
        policy.eval()

        controlled = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
        road = getattr(env.unwrapped, "road", None)
        road_vehicles = list(getattr(road, "vehicles", ()) or ()) if road is not None else []
        memory = (
            policy.initial_memory(len(obs_agents), device=device, dtype=torch.float32)
            if recurrent_policy_enabled(policy)
            else None
        )

        total_reward = 0.0
        terminated = False
        truncated = False
        steps_completed = 0
        crash_steps = 0
        offroad_steps = 0
        iterator = range(max(1, int(args.steps)))
        if tqdm is not None and not bool(args.disable_progress):
            iterator = tqdm(iterator, total=max(1, int(args.steps)), desc="GAIL episode", dynamic_ncols=True)
        for _step in iterator:
            step_start = time.perf_counter()

            t0 = time.perf_counter()
            obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
            timings["obs_flatten"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            critic_obs_agents = central_critic_observations(env, cfg, obs_agents)
            timings["critic_obs"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            action_masks = None
            if str(cfg.action_mode).lower() == "discrete":
                masks = discrete_action_masks_from_env(
                    env,
                    num_agents=len(obs_agents),
                    num_actions=policy_action_dim(policy),
                    enabled=bool(cfg.enable_action_masking),
                )
                action_masks = torch.as_tensor(masks, dtype=torch.bool, device=device)
            timings["action_mask"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
                critic_obs_tensor = torch.as_tensor(critic_obs_agents, dtype=torch.float32, device=device)
                dist, _values, memory = policy_distribution_values_memory(
                    policy,
                    obs_tensor,
                    cfg,
                    action_masks,
                    critic_obs_tensor=critic_obs_tensor,
                    memory=memory,
                    return_memory=True,
                )
                actions, _log_probs = _sample_policy_actions(dist, cfg)
                if str(cfg.action_mode).lower() == "discrete" and action_masks is not None:
                    actions = torch.argmax(_masked_discrete_logits(actions, action_masks), dim=-1)
            timings["policy_forward"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            action_tuple = _actions_to_env_tuple(actions, cfg)
            timings["action_to_env"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action_tuple)
            timings["env_step"].append(time.perf_counter() - t0)
            timings["step_total"].append(time.perf_counter() - step_start)

            total_reward += float(np.asarray(reward, dtype=np.float32).mean())
            crash_steps += int(sum(bool(flag) for flag in info.get("controlled_vehicle_crashes", []) or []))
            offroad_steps += int(sum(bool(flag) for flag in info.get("controlled_vehicle_offroad", []) or []))
            steps_completed += 1
            if tqdm is not None and not bool(args.disable_progress):
                iterator.set_postfix(env_ms=f"{timings['env_step'][-1] * 1000.0:.1f}", agents=len(obs_agents))
            if terminated or truncated:
                break

        result: dict[str, Any] = {
            "config": {
                "scene": cfg.scene,
                "split": cfg.prebuilt_split,
                "requested_controlled_vehicles": float(args.requested_controlled_vehicles),
                "max_steps": int(args.steps),
                "policy_model": cfg.policy_model,
                "device": str(device),
                "worker_threads": max(1, int(args.worker_threads)),
                "checkpoint": str(args.checkpoint or ""),
            },
            "counts": {
                "initial_controlled_vehicles": int(len(controlled)),
                "initial_road_vehicles": int(len(road_vehicles)),
                "final_controlled_vehicles": int(len(getattr(env.unwrapped, "controlled_vehicles", ()) or ())),
                "steps_completed": int(steps_completed),
                "crash_flag_steps": int(crash_steps),
                "offroad_flag_steps": int(offroad_steps),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
            "reward": {"mean_step_reward_sum": float(total_reward)},
            "setup_seconds": {
                "env_make": float(make_seconds),
                "env_reset": float(reset_seconds),
            },
            "timings": {name: summarize_times(values) for name, values in timings.items()},
            "cuda": cuda_memory(device),
            "reset_info_keys": sorted(str(key) for key in info.keys()) if isinstance(info, dict) else [],
        }
    finally:
        env.close()

    print(json.dumps(result, indent=2, sort_keys=True))
    output = str(args.output or "").strip()
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved benchmark JSON: {path}")


if __name__ == "__main__":
    main()
