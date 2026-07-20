#!/usr/bin/env python3
"""Validation-first online IQ-Learn for recurrent-transformer driving policies."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.envs import controlled_vehicle_snapshot, make_training_env
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.observations import flatten_agent_observations, policy_observations_from_flat
from scripts_gail.ps_gail.recurrent_bc import build_sequence_windows, evaluate_recurrent_bc
from scripts_gail.ps_gail.recurrent_iq import (
    RecurrentTwinQNetwork,
    append_memory,
    sample_recurrent_policy,
    trainable_parameters,
    update_recurrent_iq,
)
from scripts_gail.ps_gail.trainer import infer_continuous_action_dim, infer_policy_obs_dim, resolve_device
from scripts_gail.train_simple_ps_gail import evaluate_policy_survival


IQ_REFERENCE_REPOSITORY = "https://github.com/Div-Infinity/IQ-Learn"
IQ_REFERENCE_COMMIT = "1f5492dd26348ef11dea3467037baf4eff65c178"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--domain", default="us", choices=["us", "japanese"])
    parser.add_argument("--scene", default="us-101", choices=["us-101", "japanese"])
    parser.add_argument("--episode-root", default="data/highway_env/processed_20s")
    parser.add_argument("--prebuilt-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=20260716)
    parser.add_argument("--max-expert-samples", type=int, default=300_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-policy-checkpoint", required=True)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, required=True, choices=[2, 3])
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-norm-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-tokens", type=int, default=8)
    parser.add_argument("--memory-context-length", type=int, default=32)
    parser.add_argument("--training-context-length", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--validation-sequence-length", type=int, default=32)
    parser.add_argument("--sequences-per-update", type=int, default=4)
    parser.add_argument("--micro-batch-sequences", type=int, default=16)

    parser.add_argument("--updates", type=int, default=2_000)
    parser.add_argument("--minimum-joint-updates", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--early-stopping-evaluations", type=int, default=6)
    parser.add_argument("--policy-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--q-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--entropy-temperature", type=float, default=0.01)
    parser.add_argument("--chi2-alpha", type=float, default=0.5)
    parser.add_argument(
        "--chi2-regularization",
        choices=["expert", "mixture"],
        default="expert",
        help="expert matches the official IQ-Learn default; mixture is an explicitly labelled variant.",
    )
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--target-q-clip", type=float, default=20.0)
    parser.add_argument("--bc-coef", type=float, default=10.0)
    parser.add_argument("--q-only-updates", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--max-q-abs", type=float, default=50.0)
    parser.add_argument("--initial-log-std", type=float, default=-2.5)
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=-1.5)

    parser.add_argument("--initial-policy-replay", type=int, default=1_280)
    parser.add_argument("--policy-replay-capacity", type=int, default=100_000)
    parser.add_argument("--collect-steps", type=int, default=512)
    parser.add_argument("--collect-every", type=int, default=64)
    parser.add_argument("--training-enable-collision", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--min-validation-skill", type=float, default=0.10)
    parser.add_argument("--max-initial-skill-regression", type=float, default=0.02)
    parser.add_argument("--max-validation-mae", type=float, default=0.35)
    parser.add_argument("--learning-action-index", type=int, default=0)
    parser.add_argument("--min-learning-action-std-ratio", type=float, default=0.25)
    parser.add_argument("--min-learning-action-correlation", type=float, default=0.50)
    parser.add_argument("--min-rollout-steps", type=int, default=100)
    parser.add_argument("--max-crash-fraction", type=float, default=0.34)
    parser.add_argument("--max-offroad-fraction", type=float, default=0.34)
    parser.add_argument("--max-collision-proxy-fraction", type=float, default=0.34)
    parser.add_argument("--max-rollout-mean-length-regression", type=float, default=20.0)
    parser.add_argument("--max-rollout-fraction-regression", type=float, default=0.0)
    parser.add_argument("--capability-failure-mode", choices=["error", "report"], default="error")
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write local TensorBoard events in addition to metrics.jsonl.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="TensorBoard event directory (default: <out-dir>/tensorboard).",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _tensorboard_scalars(payload: Any, *, prefix: str = "") -> dict[str, float]:
    """Flatten finite numeric diagnostics into stable TensorBoard scalar tags."""
    scalars: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"phase", "update"}:
                continue
            child = f"{prefix}/{key}" if prefix else str(key)
            scalars.update(_tensorboard_scalars(value, prefix=child))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            child = f"{prefix}/action_{index}"
            scalars.update(_tensorboard_scalars(value, prefix=child))
    elif isinstance(payload, (bool, int, float, np.number)):
        value = float(payload)
        if np.isfinite(value):
            scalars[prefix] = value
    return scalars


def write_metric_row(
    path: Path,
    payload: dict[str, Any],
    tensorboard_writer: Any | None,
) -> None:
    append_jsonl(path, payload)
    if tensorboard_writer is None:
        return
    step = int(payload.get("update", 0))
    phase = str(payload.get("phase", "metrics"))
    for tag, value in _tensorboard_scalars(payload).items():
        tensorboard_writer.add_scalar(f"{phase}/{tag}", value, step)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_sidecar(path: Path) -> str:
    actual = sha256_file(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"Initial checkpoint SHA-256 sidecar is missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if expected != actual:
        raise RuntimeError(f"Initial checkpoint SHA-256 mismatch: {expected} != {actual} ({path})")
    return actual


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(path)
    path.with_name(f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def make_config(args: argparse.Namespace) -> PSGAILConfig:
    return PSGAILConfig(
        expert_data=str(Path(args.expert_data).resolve()),
        run_name=f"online_recurrent_iq_{args.domain}_{args.transformer_layers}layer_seed_{args.seed}",
        scene=str(args.scene),
        action_mode="continuous",
        episode_root=str(Path(args.episode_root).resolve()),
        prebuilt_split=str(args.prebuilt_split),
        seed=int(args.seed),
        max_expert_samples=int(args.max_expert_samples),
        trajectory_frame="relative",
        max_surrounding="all",
        control_all_vehicles=False,
        percentage_controlled_vehicles=1.0,
        allow_idm=True,
        enable_collision=bool(args.training_enable_collision),
        terminate_when_all_controlled_crashed=True,
        cells=128,
        maximum_range=64.0,
        simulation_frequency=10,
        policy_frequency=10,
        max_episode_steps=200,
        policy_model="recurrent_transformer",
        hidden_size=int(args.hidden_size),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_dropout=float(args.transformer_dropout),
        transformer_norm_first=bool(args.transformer_norm_first),
        transformer_memory_tokens=int(args.memory_tokens),
        transformer_memory_context_length=int(args.memory_context_length),
        transformer_recurrent_sequence_length=int(args.sequence_length),
        transformer_recurrent_sequences_per_batch=int(args.sequences_per_update),
        transformer_recurrent_micro_batch_sequences=int(args.micro_batch_sequences),
        transformer_use_causal_attention=True,
        bc_pretrain_eval_deterministic=True,
        device=str(args.device),
    )


def policy_architecture(cfg: PSGAILConfig, obs_dim: int, action_dim: int) -> dict[str, Any]:
    return {
        "policy_model": "recurrent_transformer",
        "obs_dim": int(obs_dim),
        "hidden_size": int(cfg.hidden_size),
        "action_mode": "continuous",
        "continuous_action_dim": int(action_dim),
        "transformer_layers": int(cfg.transformer_layers),
        "transformer_heads": int(cfg.transformer_heads),
        "transformer_dropout": float(cfg.transformer_dropout),
        "transformer_norm_first": bool(cfg.transformer_norm_first),
        "transformer_memory_tokens": int(cfg.transformer_memory_tokens),
        "transformer_memory_context_length": int(cfg.transformer_memory_context_length),
        "transformer_use_causal_attention": True,
    }


def load_initial_policy_checkpoint(
    policy: torch.nn.Module,
    path: Path,
    *,
    obs_dim: int,
    action_dim: int,
    cfg: PSGAILConfig,
    domain: str | None = None,
    seed: int | None = None,
    min_validation_skill: float = 0.10,
    max_validation_mae: float = 0.35,
    min_std_ratio: float = 0.25,
    min_correlation: float = 0.50,
) -> dict[str, Any]:
    digest = verify_sha256_sidecar(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "policy_state_dict" not in payload:
        raise RuntimeError(f"Initial policy checkpoint has no policy_state_dict: {path}")
    if payload.get("checkpoint_kind") != "behaviour_cloning_best":
        raise RuntimeError(f"IQ production requires a full BC best checkpoint, got {payload.get('checkpoint_kind')!r}.")
    architecture = payload.get("policy_architecture") or {}
    expected = {
        "policy_model": "recurrent_transformer",
        "obs_dim": int(obs_dim),
        "continuous_action_dim": int(action_dim),
        "hidden_size": int(cfg.hidden_size),
        "transformer_layers": int(cfg.transformer_layers),
        "transformer_heads": int(cfg.transformer_heads),
        "transformer_dropout": float(cfg.transformer_dropout),
        "transformer_norm_first": bool(cfg.transformer_norm_first),
        "transformer_memory_tokens": int(cfg.transformer_memory_tokens),
        "transformer_memory_context_length": int(cfg.transformer_memory_context_length),
        "transformer_use_causal_attention": True,
    }
    for key, value in expected.items():
        if architecture.get(key) != value:
            raise RuntimeError(f"Initial checkpoint architecture mismatch for {key}: {architecture.get(key)!r} != {value!r} ({path})")
    config = payload.get("config") or {}
    if seed is not None and int(config.get("seed", -1)) != int(seed):
        raise RuntimeError(f"Initial checkpoint policy seed mismatch: {config.get('seed')} != {seed}")
    expected_scene = "us-101" if domain == "us" else "japanese" if domain == "japanese" else None
    if expected_scene is not None and str(config.get("scene")) != expected_scene:
        raise RuntimeError(f"Initial checkpoint domain/scene mismatch: {config.get('scene')!r} != {expected_scene!r}")
    stats = payload.get("bc_stats") or {}
    ratios = list(stats.get("validation_prediction_std_ratio") or [])
    correlations = list(stats.get("validation_prediction_target_correlation") or [])
    if not ratios or not correlations:
        raise RuntimeError("Initial BC checkpoint lacks anti-collapse validation diagnostics.")
    qualified = bool(
        float(stats.get("validation_skill", float("-inf"))) >= float(min_validation_skill)
        and float(stats.get("validation_mae", float("inf"))) <= float(max_validation_mae)
        and float(ratios[0]) >= float(min_std_ratio)
        and float(correlations[0]) >= float(min_correlation)
    )
    if not qualified:
        raise RuntimeError(f"Initial BC checkpoint is not metric-qualified: {path}")
    data_split = payload.get("data_split") or {}
    split_ids = data_split.get("trajectory_ids")
    if not isinstance(split_ids, dict) or not all(key in split_ids for key in ("train", "validation", "test")):
        raise RuntimeError("Initial BC checkpoint lacks its trajectory-level train/validation/test split.")
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "bc_stats": stats,
        "split_trajectory_ids": split_ids,
        "source_provenance": payload.get("provenance"),
    }


def validation_baseline_mse(transitions: Any, train_windows: list[Any], validation_windows: list[Any]) -> float:
    actions = np.asarray(transitions.actions_continuous_env, dtype=np.float32)
    train_indices = np.concatenate([window.train_indices for window in train_windows]).astype(np.int64)
    validation_indices = np.concatenate([window.train_indices for window in validation_windows]).astype(np.int64)
    action_mean = actions[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    return float(np.mean(np.square(actions[validation_indices] - action_mean)))


def validation_metrics(
    policy: torch.nn.Module,
    transitions: Any,
    windows: list[Any],
    *,
    device: torch.device,
    micro_batch_sequences: int,
    baseline_mse: float,
) -> dict[str, Any]:
    metrics = evaluate_recurrent_bc(
        policy, transitions, windows, device=device, micro_batch_sequences=int(micro_batch_sequences)
    )
    return {
        "validation_mse": float(metrics["mse"]),
        "validation_mae": float(metrics["mae"]),
        "validation_skill": 1.0 - float(metrics["mse"]) / max(float(baseline_mse), 1.0e-12),
        "validation_samples": float(metrics["samples"]),
        "validation_action_mse": metrics["action_mse"],
        "validation_action_mae": metrics["action_mae"],
        "validation_prediction_std_ratio": metrics["prediction_std_ratio"],
        "validation_prediction_target_correlation": metrics["prediction_target_correlation"],
        "validation_prediction_saturation_fraction": metrics["prediction_saturation_fraction"],
    }


def offline_capability(metrics: dict[str, Any], initial: dict[str, Any], args: argparse.Namespace) -> bool:
    index = int(args.learning_action_index)
    ratios = metrics["validation_prediction_std_ratio"]
    correlations = metrics["validation_prediction_target_correlation"]
    return bool(
        metrics["validation_skill"] >= float(args.min_validation_skill)
        and metrics["validation_skill"] >= initial["validation_skill"] - float(args.max_initial_skill_regression)
        and metrics["validation_mae"] <= float(args.max_validation_mae)
        and 0 <= index < len(ratios)
        and float(ratios[index]) >= float(args.min_learning_action_std_ratio)
        and float(correlations[index]) >= float(args.min_learning_action_correlation)
    )


def rollout_thresholds(
    args: argparse.Namespace,
    baseline: dict[str, float] | None = None,
) -> dict[str, float]:
    thresholds = {
        "min_mean_episode_length": float(args.min_rollout_steps),
        "max_crash_episode_fraction": float(args.max_crash_fraction),
        "max_offroad_episode_fraction": float(args.max_offroad_fraction),
        "max_collision_proxy_episode_fraction": float(args.max_collision_proxy_fraction),
    }
    if baseline:
        thresholds["min_mean_episode_length"] = max(
            thresholds["min_mean_episode_length"],
            float(baseline["bc_eval/mean_episode_length"])
            - float(args.max_rollout_mean_length_regression),
        )
        tolerance = float(args.max_rollout_fraction_regression)
        for metric, threshold in (
            ("bc_eval/crash_episode_fraction", "max_crash_episode_fraction"),
            ("bc_eval/offroad_episode_fraction", "max_offroad_episode_fraction"),
            ("bc_eval/collision_proxy_episode_fraction", "max_collision_proxy_episode_fraction"),
        ):
            thresholds[threshold] = min(
                1.0,
                max(thresholds[threshold], float(baseline[metric]) + tolerance),
            )
    return thresholds


def rollout_capability(
    metrics: dict[str, float],
    args: argparse.Namespace,
    *,
    baseline: dict[str, float] | None = None,
) -> bool:
    thresholds = rollout_thresholds(args, baseline)
    return bool(
        metrics
        and metrics["bc_eval/mean_episode_length"] >= thresholds["min_mean_episode_length"]
        and metrics["bc_eval/crash_episode_fraction"] <= thresholds["max_crash_episode_fraction"]
        and metrics["bc_eval/offroad_episode_fraction"] <= thresholds["max_offroad_episode_fraction"]
        and metrics["bc_eval/collision_proxy_episode_fraction"]
        <= thresholds["max_collision_proxy_episode_fraction"]
    )


def sample_policy_replay_action(
    policy: torch.nn.Module,
    observations: np.ndarray,
    memory: torch.Tensor,
    *,
    device: torch.device,
    log_std_min: float,
    log_std_max: float,
) -> tuple[tuple[np.ndarray, ...], torch.Tensor]:
    """Sample the same squashed Gaussian optimized by SAC and advance all memory."""
    observations = np.asarray(observations, dtype=np.float32)
    if int(memory.shape[0]) != len(observations):
        memory = policy.initial_memory(len(observations), device=device, dtype=torch.float32)
    with torch.no_grad():
        actions, _log_prob, _mean_action, step_memory = sample_recurrent_policy(
            policy,
            torch.as_tensor(observations, dtype=torch.float32, device=device),
            memory,
            log_std_min=float(log_std_min),
            log_std_max=float(log_std_max),
        )
        updated_memory = append_memory(
            memory,
            step_memory,
            max_context=int(policy.memory_context_length),
        )
    action_array = actions.detach().cpu().numpy().astype(np.float32, copy=False)
    return tuple(action.copy() for action in action_array), updated_memory


def collect_policy_transitions(
    policy: torch.nn.Module,
    cfg: PSGAILConfig,
    device: torch.device,
    *,
    target_steps: int,
    seed: int,
    collection_id: int,
    log_std_min: float,
    log_std_max: float,
) -> SimpleNamespace:
    """Collect learner replay with the exact recurrent SAC policy distribution."""
    rows: dict[str, list[Any]] = {
        "policy_observations": [], "next_policy_observations": [], "actions": [],
        "trajectory_ids": [], "vehicle_ids": [], "timesteps": [], "dones": [], "terminals": [],
    }
    environment = make_training_env(cfg)
    trajectory_counter = 0
    try:
        while len(rows["actions"]) < max(1, int(target_steps)):
            obs, _info = environment.reset(seed=int(seed) + trajectory_counter)
            obs_agents = policy_observations_from_flat(flatten_agent_observations(obs))
            memory = policy.initial_memory(len(obs_agents), device=device, dtype=torch.float32)
            trajectory_ids = [f"online_{collection_id}_{trajectory_counter}_{index}" for index in range(len(obs_agents))]
            for step in range(max(1, int(cfg.max_episode_steps))):
                current_agents = policy_observations_from_flat(flatten_agent_observations(obs))
                vehicle_ids, _trajectory_states = controlled_vehicle_snapshot(environment)
                action_tuple, new_memory = sample_policy_replay_action(
                    policy,
                    current_agents,
                    memory,
                    device=device,
                    log_std_min=float(log_std_min),
                    log_std_max=float(log_std_max),
                )
                next_obs, _reward, terminated, truncated, _step_info = environment.step(action_tuple)
                next_agents = policy_observations_from_flat(flatten_agent_observations(next_obs))
                if next_agents.shape != current_agents.shape:
                    next_agents = current_agents.copy()
                segment_done = bool(terminated or truncated or step + 1 >= int(cfg.max_episode_steps))
                true_terminal = bool(terminated)
                for index in range(len(current_agents)):
                    rows["policy_observations"].append(current_agents[index].copy())
                    rows["next_policy_observations"].append(next_agents[index].copy())
                    rows["actions"].append(np.asarray(action_tuple[index], dtype=np.float32).copy())
                    rows["trajectory_ids"].append(trajectory_ids[index])
                    rows["vehicle_ids"].append(int(vehicle_ids[index]) if index < len(vehicle_ids) else index)
                    rows["timesteps"].append(step)
                    rows["dones"].append(segment_done)
                    rows["terminals"].append(true_terminal)
                obs, memory = next_obs, new_memory
                if segment_done:
                    break
            trajectory_counter += 1
    finally:
        environment.close()
    return SimpleNamespace(
        policy_observations=np.asarray(rows["policy_observations"], dtype=np.float32),
        next_policy_observations=np.asarray(rows["next_policy_observations"], dtype=np.float32),
        actions_continuous_env=np.asarray(rows["actions"], dtype=np.float32),
        trajectory_ids=np.asarray(rows["trajectory_ids"], dtype=object),
        vehicle_ids=np.asarray(rows["vehicle_ids"], dtype=np.int64),
        timesteps=np.asarray(rows["timesteps"], dtype=np.int64),
        dones=np.asarray(rows["dones"], dtype=bool),
        terminals=np.asarray(rows["terminals"], dtype=bool),
    )


class RecurrentPolicyReplay:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.chunks: deque[SimpleNamespace] = deque()
        self.transitions = 0

    def add(self, chunk: SimpleNamespace) -> None:
        self.chunks.append(chunk)
        self.transitions += len(chunk.dones)
        while len(self.chunks) > 1 and self.transitions > self.capacity:
            self.transitions -= len(self.chunks.popleft().dones)

    def merged(self) -> SimpleNamespace:
        if not self.chunks:
            raise RuntimeError("Learner replay is empty.")
        names = (
            "policy_observations", "next_policy_observations", "actions_continuous_env",
            "trajectory_ids", "vehicle_ids", "timesteps", "dones", "terminals",
        )
        return SimpleNamespace(**{
            name: np.concatenate([getattr(chunk, name) for chunk in self.chunks], axis=0)
            for name in names
        })


def parameter_delta(before: dict[str, torch.Tensor], policy: torch.nn.Module) -> float:
    squares = torch.zeros((), dtype=torch.float64)
    for name, value in policy.state_dict().items():
        if name == "log_std":
            continue
        squares += (value.detach().cpu().to(torch.float64) - before[name].to(torch.float64)).square().sum()
    return float(torch.sqrt(squares))


def checkpoint_payload(
    policy: torch.nn.Module,
    q_net: RecurrentTwinQNetwork,
    target_q_net: RecurrentTwinQNetwork,
    cfg: PSGAILConfig,
    args: argparse.Namespace,
    *,
    obs_dim: int,
    action_dim: int,
    update: int,
    selection: dict[str, Any],
    initialization: dict[str, Any],
    split_ids: dict[str, list[str]],
    checkpoint_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "checkpoint_kind": checkpoint_kind,
        "method": "online_recurrent_iq_learn_chi2",
        "update": int(update),
        "policy_state_dict": policy.state_dict(),
        "q_state_dict": q_net.state_dict(),
        "target_q_state_dict": target_q_net.state_dict(),
        "config": vars(cfg),
        "policy_architecture": policy_architecture(cfg, obs_dim, action_dim),
        "iq_config": vars(args),
        "selection": selection,
        "initialization": initialization,
        "data_split": {"method": "trajectory_level_reused_from_bc", "trajectory_ids": split_ids},
        "reference": {"repository": IQ_REFERENCE_REPOSITORY, "commit": IQ_REFERENCE_COMMIT},
        "provenance": {"command": list(map(str, os.sys.argv)), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= int(args.training_context_length) <= int(args.memory_context_length):
        raise ValueError("training-context-length must be in [1, memory-context-length].")
    if min(int(args.sequence_length), int(args.validation_sequence_length), int(args.sequences_per_update)) < 1:
        raise ValueError("training/validation sequence lengths and sequences-per-update must be positive.")
    if int(args.minimum_joint_updates) > max(0, int(args.updates) - int(args.q_only_updates)):
        raise ValueError("minimum-joint-updates exceeds the available joint IQ updates.")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_path, best_path, final_path = out_dir / "initial.pt", out_dir / "best.pt", out_dir / "final.pt"
    for path in (initial_path, best_path, final_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing IQ-Learn checkpoint: {path}")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    tensorboard_writer = None
    tensorboard_dir: Path | None = None
    if bool(getattr(args, "tensorboard", True)):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            print(
                "WARNING: TensorBoard is unavailable; continuing with metrics.jsonl only "
                f"({exc}).",
                file=os.sys.stderr,
                flush=True,
            )
        else:
            tensorboard_dir = (
                Path(args.tensorboard_dir).resolve()
                if getattr(args, "tensorboard_dir", None) is not None
                else out_dir / "tensorboard"
            )
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir), flush_secs=10)
            tensorboard_writer.add_text("run/command", " ".join(map(str, os.sys.argv)), 0)
            tensorboard_writer.add_text("run/config", json.dumps(vars(args), indent=2, default=str), 0)
            print(f"TensorBoard logs: {tensorboard_dir}", flush=True)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = resolve_device(str(args.device))
    cfg = make_config(args)
    environment = make_training_env(cfg)
    try:
        obs_dim = infer_policy_obs_dim(environment)
        action_dim = infer_continuous_action_dim(environment)
    finally:
        environment.close()
    cfg.continuous_action_dim = int(action_dim)
    transitions = load_expert_transition_data(
        str(args.expert_data), max_samples=int(args.max_expert_samples), seed=int(args.data_seed), trajectory_frame="relative"
    )
    if int(transitions.policy_observations.shape[1]) != int(obs_dim):
        raise RuntimeError(f"Expert/env observation mismatch: {transitions.policy_observations.shape[1]} != {obs_dim}.")

    policy = make_actor_critic(
        "recurrent_transformer", obs_dim, int(args.hidden_size), action_mode="continuous",
        continuous_action_dim=action_dim, transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads), transformer_dropout=float(args.transformer_dropout),
        transformer_norm_first=bool(args.transformer_norm_first),
        transformer_memory_tokens=int(args.memory_tokens),
        transformer_memory_context_length=int(args.memory_context_length), transformer_use_causal_attention=True,
    ).to(device)
    initialization = load_initial_policy_checkpoint(
        policy, Path(args.initial_policy_checkpoint).resolve(), obs_dim=obs_dim, action_dim=action_dim, cfg=cfg,
        domain=str(args.domain), seed=int(args.seed), min_validation_skill=float(args.min_validation_skill),
        max_validation_mae=float(args.max_validation_mae), min_std_ratio=float(args.min_learning_action_std_ratio),
        min_correlation=float(args.min_learning_action_correlation),
    )
    with torch.no_grad():
        policy.log_std.fill_(float(args.initial_log_std))
    split_ids = initialization.pop("split_trajectory_ids")
    train_windows = build_sequence_windows(
        transitions,
        split_ids["train"],
        sequence_length=int(args.sequence_length),
        context_length=int(args.training_context_length),
    )
    validation_windows = build_sequence_windows(
        transitions,
        split_ids["validation"],
        sequence_length=int(args.validation_sequence_length),
        context_length=int(args.memory_context_length),
    )
    baseline_mse = validation_baseline_mse(transitions, train_windows, validation_windows)
    initial_validation = validation_metrics(
        policy, transitions, validation_windows, device=device,
        micro_batch_sequences=int(args.micro_batch_sequences), baseline_mse=baseline_mse,
    )
    # Compare every learned checkpoint to the matched initializer under the
    # same episodes and seeds.  Existing BC models can miss absolute survival
    # thresholds, so an absolute-only gate would make no-regression IQ models
    # impossible to qualify.
    initial_validation_rollouts = evaluate_policy_survival(
        policy,
        replace(cfg, prebuilt_split="val", enable_collision=False, bc_pretrain_eval_deterministic=True),
        device,
        episodes=int(args.evaluation_episodes),
        seed_offset=20_000,
    )
    initial_test_rollouts = evaluate_policy_survival(
        policy,
        replace(cfg, prebuilt_split="test", enable_collision=False, bc_pretrain_eval_deterministic=True),
        device,
        episodes=int(args.evaluation_episodes),
        seed_offset=40_000,
    )

    q_kwargs = {
        "transformer_layers": int(args.transformer_layers), "transformer_heads": int(args.transformer_heads),
        "transformer_dropout": float(args.transformer_dropout), "transformer_norm_first": bool(args.transformer_norm_first),
        "memory_tokens": int(args.memory_tokens), "memory_context_length": int(args.memory_context_length),
        "use_causal_attention": True,
    }
    q_net = RecurrentTwinQNetwork(obs_dim, action_dim, int(args.hidden_size), **q_kwargs).to(device)
    q_net.initialize_encoders_from_policy(policy)
    target_q_net = RecurrentTwinQNetwork(obs_dim, action_dim, int(args.hidden_size), **q_kwargs).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    target_q_net.requires_grad_(False)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.policy_learning_rate), weight_decay=1.0e-5)
    q_optimizer = torch.optim.AdamW(trainable_parameters(q_net), lr=float(args.q_learning_rate), weight_decay=1.0e-5)
    actor_joint_start = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}

    initial_selection = {"stage": "bc_initialization", "update": 0, **initial_validation}
    initial_sha256 = save_checkpoint(
        initial_path,
        checkpoint_payload(
            policy, q_net, target_q_net, cfg, args, obs_dim=obs_dim, action_dim=action_dim, update=0,
            selection=initial_selection, initialization=initialization, split_ids=split_ids,
            checkpoint_kind="recurrent_iq_learn_initialization",
        ),
    )
    write_metric_row(
        metrics_path,
        {"phase": "iq_validation", "update": 0, "eligible": False, **initial_validation},
        tensorboard_writer,
    )

    replay = RecurrentPolicyReplay(int(args.policy_replay_capacity))
    collection_id = 0
    while replay.transitions < int(args.initial_policy_replay):
        replay.add(collect_policy_transitions(
            policy, cfg, device, target_steps=int(args.collect_steps),
            seed=int(args.seed) + 30_000 + collection_id * 1_000, collection_id=collection_id,
            log_std_min=float(args.log_std_min), log_std_max=float(args.log_std_max),
        ))
        collection_id += 1

    rng = np.random.default_rng(int(args.seed) + 9_001)
    best_update: int | None = None
    best_validation_mse = float("inf")
    best_sha256: str | None = None
    best_selection: dict[str, Any] | None = None
    evaluations_without_improvement = 0
    latest_stats: dict[str, float] = {}
    completed_updates = 0
    total_policy_steps = replay.transitions

    for update in range(1, max(1, int(args.updates)) + 1):
        completed_updates = update
        if update > 1 and update % max(1, int(args.collect_every)) == 0:
            chunk = collect_policy_transitions(
                policy, cfg, device, target_steps=int(args.collect_steps),
                seed=int(args.seed) + 30_000 + collection_id * 1_000, collection_id=collection_id,
                log_std_min=float(args.log_std_min), log_std_max=float(args.log_std_max),
            )
            replay.add(chunk)
            total_policy_steps += len(chunk.dones)
            collection_id += 1
        replay_transitions = replay.merged()
        replay_windows = build_sequence_windows(
            replay_transitions, set(map(str, replay_transitions.trajectory_ids.tolist())),
            sequence_length=int(args.sequence_length), context_length=int(args.training_context_length),
        )
        count = min(int(args.sequences_per_update), len(train_windows), len(replay_windows))
        expert_indices = rng.choice(len(train_windows), size=count, replace=False)
        learner_indices = rng.choice(len(replay_windows), size=count, replace=False)
        update_started = time.perf_counter()
        stats = update_recurrent_iq(
            policy, q_net, target_q_net, policy_optimizer, q_optimizer,
            transitions, [train_windows[int(index)] for index in expert_indices],
            policy_transitions=replay_transitions,
            policy_windows=[replay_windows[int(index)] for index in learner_indices],
            device=device, gamma=float(args.gamma), entropy_temperature=float(args.entropy_temperature),
            chi2_alpha=float(args.chi2_alpha), target_tau=float(args.target_tau), bc_coef=float(args.bc_coef),
            max_grad_norm=float(args.max_grad_norm), update_actor=update > int(args.q_only_updates),
            chi2_on_mixture=str(args.chi2_regularization) == "mixture", target_q_clip=float(args.target_q_clip),
            log_std_min=float(args.log_std_min), log_std_max=float(args.log_std_max),
        )
        update_wall_seconds = time.perf_counter() - update_started
        latest_stats = stats.as_dict()
        if not all(np.isfinite(value) for value in latest_stats.values()):
            raise RuntimeError(f"Non-finite recurrent IQ metric at update {update}: {latest_stats}")
        if float(stats.q_abs_max) > float(args.max_q_abs):
            raise RuntimeError(f"Recurrent IQ Q scale exceeded bound at update {update}: {stats.q_abs_max:.4f} > {float(args.max_q_abs):.4f}.")
        row: dict[str, Any] = {
            "phase": "iq_update", "update": update, "replay_transitions": replay.transitions,
            "total_policy_steps": total_policy_steps, "replay_collections": collection_id,
            "update_wall_seconds": update_wall_seconds, **latest_stats,
        }
        evaluation_due = update == int(args.q_only_updates) or update % max(1, int(args.eval_every)) == 0 or update == int(args.updates)
        if evaluation_due:
            current_validation = validation_metrics(
                policy, transitions, validation_windows, device=device,
                micro_batch_sequences=int(args.micro_batch_sequences), baseline_mse=baseline_mse,
            )
            joint_updates = max(0, update - int(args.q_only_updates))
            val_rollouts: dict[str, float] = {}
            offline_passed = offline_capability(current_validation, initial_validation, args)
            if joint_updates > 0 and offline_passed:
                val_rollouts = evaluate_policy_survival(
                    policy, replace(cfg, prebuilt_split="val", enable_collision=False, bc_pretrain_eval_deterministic=True),
                    device, episodes=int(args.evaluation_episodes), seed_offset=20_000,
                )
            delta = parameter_delta(actor_joint_start, policy)
            eligible = bool(
                joint_updates >= int(args.minimum_joint_updates)
                and offline_passed
                and rollout_capability(val_rollouts, args, baseline=initial_validation_rollouts)
                and delta > 0.0
                and float(stats.actor_rl_grad_norm) > 0.0
            )
            improved = bool(eligible and float(current_validation["validation_mse"]) < best_validation_mse)
            row.update(current_validation)
            row.update({"joint_updates": joint_updates, "actor_parameter_delta": delta, "offline_capability_passed": offline_passed,
                        "validation_rollouts": val_rollouts,
                        "validation_rollout_thresholds": rollout_thresholds(args, initial_validation_rollouts),
                        "rollout_capability_passed": rollout_capability(
                            val_rollouts, args, baseline=initial_validation_rollouts
                        ),
                        "eligible": eligible, "is_best": improved})
            if improved:
                best_validation_mse = float(current_validation["validation_mse"])
                best_update = update
                best_selection = dict(row)
                best_sha256 = save_checkpoint(
                    best_path,
                    checkpoint_payload(
                        policy, q_net, target_q_net, cfg, args, obs_dim=obs_dim, action_dim=action_dim,
                        update=update, selection=best_selection, initialization=initialization, split_ids=split_ids,
                        checkpoint_kind="online_recurrent_iq_learn_best",
                    ),
                )
                evaluations_without_improvement = 0
            elif joint_updates >= int(args.minimum_joint_updates):
                evaluations_without_improvement += 1
            print(
                f"[online recurrent iq {update:06d}] q={stats.q_loss:.5f} bc={stats.bc_loss:.5f} "
                f"reward={stats.recovered_reward:.5f} q_abs={stats.q_abs_max:.3f} "
                f"val_skill={current_validation['validation_skill']:.4f} eligible={eligible} best={best_update}", flush=True,
            )
            if best_update is not None and evaluations_without_improvement >= int(args.early_stopping_evaluations):
                write_metric_row(metrics_path, row, tensorboard_writer)
                break
        write_metric_row(metrics_path, row, tensorboard_writer)

    final_sha256 = save_checkpoint(
        final_path,
        checkpoint_payload(
            policy, q_net, target_q_net, cfg, args, obs_dim=obs_dim, action_dim=action_dim,
            update=completed_updates, selection={"stage": "final", "update": completed_updates, **latest_stats},
            initialization=initialization, split_ids=split_ids, checkpoint_kind="online_recurrent_iq_learn_final",
        ),
    )

    test_rollouts: dict[str, float] = {}
    best_validation: dict[str, Any] = {}
    test_capability = False
    if best_path.is_file():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        policy.load_state_dict(best_payload["policy_state_dict"], strict=True)
        best_validation = validation_metrics(
            policy, transitions, validation_windows, device=device,
            micro_batch_sequences=int(args.micro_batch_sequences), baseline_mse=baseline_mse,
        )
        test_rollouts = evaluate_policy_survival(
            policy, replace(cfg, prebuilt_split="test", enable_collision=False, bc_pretrain_eval_deterministic=True),
            device, episodes=int(args.evaluation_episodes), seed_offset=40_000,
        )
        test_capability = rollout_capability(test_rollouts, args, baseline=initial_test_rollouts)
    capability_passed = bool(best_update is not None and offline_capability(best_validation, initial_validation, args) and test_capability)
    summary = {
        "schema_version": 2, "method": "online_recurrent_iq_learn_chi2", "domain": str(args.domain),
        "scene": str(args.scene), "seed": int(args.seed), "transformer_layers": int(args.transformer_layers),
        "reference": {"repository": IQ_REFERENCE_REPOSITORY, "commit": IQ_REFERENCE_COMMIT},
        "chi2_regularization": str(args.chi2_regularization), "initialization": initialization,
        "initial_checkpoint": str(initial_path), "initial_checkpoint_sha256": initial_sha256,
        "initial_validation": initial_validation,
        "initial_validation_rollouts": initial_validation_rollouts,
        "initial_test_rollouts": initial_test_rollouts,
        "validation_rollout_thresholds": rollout_thresholds(args, initial_validation_rollouts),
        "test_rollout_thresholds": rollout_thresholds(args, initial_test_rollouts),
        "completed_updates": completed_updates,
        "total_policy_steps": total_policy_steps, "best_update": best_update,
        "best_validation": best_validation, "best_selection": best_selection,
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "best_checkpoint_sha256": best_sha256, "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": final_sha256, "latest_iq_stats": latest_stats,
        "held_out_test_rollouts": test_rollouts, "metric_capability_passed": bool(best_validation and offline_capability(best_validation, initial_validation, args)),
        "rollout_capability_passed": test_capability, "capability_passed": capability_passed,
        "validation_history": str(metrics_path), "split_trajectory_ids": split_ids,
        "tensorboard_log_dir": str(tensorboard_dir) if tensorboard_dir is not None else None,
        "replay": {
            "capacity": int(args.policy_replay_capacity),
            "retained_transitions": replay.transitions,
            "collections": collection_id,
            "policy_distribution": "tanh_squashed_gaussian_pre_squash_mean",
            "recurrent_memory_update": "append_full_context",
            "memory_context_length": int(args.memory_context_length),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "replay_manifest.json", summary["replay"])
    print(json.dumps({
        "method": summary["method"],
        "domain": summary["domain"],
        "seed": summary["seed"],
        "transformer_layers": summary["transformer_layers"],
        "completed_updates": summary["completed_updates"],
        "best_update": summary["best_update"],
        "best_checkpoint": summary["best_checkpoint"],
        "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
        "initial_validation_skill": initial_validation["validation_skill"],
        "best_validation_skill": best_validation.get("validation_skill"),
        "metric_capability_passed": summary["metric_capability_passed"],
        "rollout_capability_passed": summary["rollout_capability_passed"],
        "capability_passed": summary["capability_passed"],
        "summary": str(out_dir / "summary.json"),
    }, indent=2, sort_keys=True))
    if tensorboard_writer is not None:
        tensorboard_writer.flush()
        tensorboard_writer.close()
    if not capability_passed and str(args.capability_failure_mode) == "error":
        raise RuntimeError("Online recurrent IQ-Learn did not produce a nonzero-update checkpoint passing all capability gates.")
    return summary


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
