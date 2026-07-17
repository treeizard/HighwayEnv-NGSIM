#!/usr/bin/env python3
"""Train an interpretation-ready recurrent-transformer BC policy."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from scripts_gail.pretrain_continuous_bc_policy import (
    build_policy_for_env,
    default_scenario_from_expert_folder,
    make_selected_replay_env,
    render_selected_replay,
)
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.recurrent_bc import train_recurrent_behavior_clone
from scripts_gail.ps_gail.trainer import resolve_device
from scripts_gail.train_simple_ps_gail import evaluate_policy_survival


CHECKPOINT_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--domain", required=True, choices=["us", "japanese"])
    parser.add_argument("--scene", required=True, choices=["us-101", "japanese"])
    parser.add_argument("--episode-root", default="data/highway_env/processed_20s")
    parser.add_argument("--prebuilt-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=20260716)
    parser.add_argument("--split-seed", type=int, default=20260716)
    parser.add_argument("--max-expert-samples", type=int, default=300_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--checkpoint-purpose",
        choices=["policy", "warm_start"],
        default="policy",
        help="Use warm_start for a deliberately under-trained GAIL/AIRL stabilizer.",
    )
    parser.add_argument("--max-warmup-epochs", type=int, default=5)
    parser.add_argument("--min-warmup-relative-improvement", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--min-validation-skill", type=float, default=0.05)
    parser.add_argument("--max-validation-mae", type=float, default=0.35)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, required=True, choices=[2, 3])
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--memory-tokens", type=int, default=8)
    parser.add_argument("--memory-context-length", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequences-per-batch", type=int, default=16)
    parser.add_argument("--micro-batch-sequences", type=int, default=4)

    parser.add_argument("--render-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-steps", type=int, default=200)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument(
        "--evaluation-split",
        choices=["val", "test"],
        default="test",
        help="Prebuilt replay split used for the post-training BC evaluation.",
    )
    parser.add_argument(
        "--evaluation-enable-collision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable collision physics during post-training evaluation (disabled by default).",
    )
    parser.add_argument("--min-rollout-steps", type=int, default=100)
    parser.add_argument("--max-crash-fraction", type=float, default=0.34)
    parser.add_argument("--max-offroad-fraction", type=float, default=0.34)
    parser.add_argument(
        "--capability-failure-mode",
        choices=["error", "report"],
        default="error",
        help=(
            "error exits non-zero when the combined metric/rollout gate fails; "
            "report records the rejection in summary.json and exits successfully. "
            "Code and data failures always exit non-zero."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_worktree_provenance(path: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    )
    return {
        "head": git_head(path),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_short": status.stdout.splitlines() if status.returncode == 0 else [],
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
    }


def expert_provenance(path: Path) -> dict[str, Any]:
    manifest = path / "manifest.json" if path.is_dir() else None
    payload: dict[str, Any] = {"path": str(path.resolve())}
    if manifest is not None and manifest.is_file():
        payload["manifest_path"] = str(manifest.resolve())
        payload["manifest_sha256"] = sha256_file(manifest)
        with manifest.open(encoding="utf-8") as handle:
            data = json.load(handle)
        payload.update(
            {
                "schema_version": data.get("schema_version"),
                "scene": data.get("scene"),
                "prebuilt_split": data.get("prebuilt_split"),
                "num_episodes": data.get("num_episodes"),
                "num_samples": data.get("num_samples"),
            }
        )
    return payload


def make_config(args: argparse.Namespace) -> PSGAILConfig:
    return PSGAILConfig(
        expert_data=str(Path(args.expert_data).resolve()),
        run_name=f"bc_{args.domain}_recurrent_transformer_{args.transformer_layers}layer_seed_{args.seed}",
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
        transformer_memory_tokens=int(args.memory_tokens),
        transformer_memory_context_length=int(args.memory_context_length),
        transformer_recurrent_sequence_length=int(args.sequence_length),
        transformer_recurrent_sequences_per_batch=int(args.sequences_per_batch),
        transformer_recurrent_micro_batch_sequences=int(args.micro_batch_sequences),
        transformer_use_causal_attention=True,
        bc_pretrain_epochs=int(args.epochs),
        bc_pretrain_learning_rate=float(args.learning_rate),
        bc_pretrain_weight_decay=float(args.weight_decay),
        device=str(args.device),
    )


def checkpoint_payload(
    policy: torch.nn.Module,
    cfg: PSGAILConfig,
    *,
    args: argparse.Namespace,
    obs_dim: int,
    action_dim: int,
    training_summary: dict[str, Any],
    split_trajectory_ids: dict[str, list[str]],
    scenario: tuple[str | None, int | None],
) -> dict[str, Any]:
    component_root = Path(__file__).resolve().parents[1]
    project_root = Path(os.environ.get("VFI_PROJECT_ROOT", component_root)).resolve()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": (
            "behaviour_cloning_warm_start"
            if args.checkpoint_purpose == "warm_start"
            else "behaviour_cloning_best"
        ),
        "policy_state_dict": policy.state_dict(),
        "config": vars(cfg),
        "policy_architecture": {
            "policy_model": "recurrent_transformer",
            "obs_dim": int(obs_dim),
            "hidden_size": int(cfg.hidden_size),
            "action_mode": "continuous",
            "continuous_action_dim": int(action_dim),
            "transformer_layers": int(cfg.transformer_layers),
            "transformer_heads": int(cfg.transformer_heads),
            "transformer_dropout": float(cfg.transformer_dropout),
            "transformer_memory_tokens": int(cfg.transformer_memory_tokens),
            "transformer_memory_context_length": int(cfg.transformer_memory_context_length),
            "transformer_use_causal_attention": True,
        },
        "bc_stats": training_summary,
        "data_split": {
            "method": "trajectory_level",
            "seed": int(args.split_seed),
            "trajectory_ids": split_trajectory_ids,
        },
        "expert_data": expert_provenance(Path(args.expert_data)),
        "provenance": {
            "component_git": git_worktree_provenance(component_root),
            "project_git": git_worktree_provenance(project_root),
            "source_sha256": {
                "recurrent_bc.py": sha256_file(component_root / "scripts_gail" / "ps_gail" / "recurrent_bc.py"),
                "train_recurrent_bc_policy.py": sha256_file(Path(__file__).resolve()),
                "models.py": sha256_file(component_root / "scripts_gail" / "ps_gail" / "models.py"),
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "command": [str(value) for value in sys.argv],
        },
        "default_video_scenario": {"episode_name": scenario[0], "vehicle_id": scenario[1]},
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append and flush one progress record so interrupted runs retain history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_checkpoint_artifacts(
    checkpoint_path: Path,
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    """Atomically persist a metric-qualified checkpoint and its integrity metadata."""
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    (checkpoint_path.parent / f"{checkpoint_path.name}.sha256").write_text(
        f"{checkpoint_sha256}  {checkpoint_path.name}\n",
        encoding="utf-8",
    )
    summary["checkpoint_saved"] = True
    summary["checkpoint_sha256"] = checkpoint_sha256
    return checkpoint_sha256


def main() -> None:
    args = parse_args()
    if args.checkpoint_purpose == "warm_start" and not (
        1 <= int(args.epochs) <= int(args.max_warmup_epochs)
    ):
        raise ValueError(
            "A BC warm start must use between 1 and "
            f"{int(args.max_warmup_epochs)} epochs; got {int(args.epochs)}."
        )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing policy checkpoint: {checkpoint_path}")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = resolve_device(str(args.device))
    cfg = make_config(args)

    scenario = default_scenario_from_expert_folder(str(args.expert_data))
    if not scenario[0] or scenario[1] is None:
        raise RuntimeError("Could not infer an episode and vehicle from the expert dataset.")
    environment = make_selected_replay_env(
        cfg,
        episode_name=str(scenario[0]),
        vehicle_id=int(scenario[1]),
        render_mode=None,
    )
    try:
        policy, obs_dim, action_dim = build_policy_for_env(cfg, environment, device)
    finally:
        environment.close()
    cfg.continuous_action_dim = int(action_dim)

    transitions = load_expert_transition_data(
        str(args.expert_data),
        max_samples=int(args.max_expert_samples),
        seed=int(args.data_seed),
        trajectory_frame="relative",
    )
    if int(transitions.policy_observations.shape[1]) != int(obs_dim):
        raise RuntimeError(
            f"Expert and environment observation dimensions differ: "
            f"{transitions.policy_observations.shape[1]} != {obs_dim}."
        )
    if int(transitions.actions_continuous_env.shape[1]) != int(action_dim):
        raise RuntimeError(
            f"Expert and environment action dimensions differ: "
            f"{transitions.actions_continuous_env.shape[1]} != {action_dim}."
        )

    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    result = train_recurrent_behavior_clone(
        policy,
        transitions,
        device=device,
        seed=int(args.seed),
        split_seed=int(args.split_seed),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        sequence_length=int(args.sequence_length),
        sequences_per_batch=int(args.sequences_per_batch),
        micro_batch_sequences=int(args.micro_batch_sequences),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        max_grad_norm=float(args.max_grad_norm),
        early_stopping_patience=int(args.early_stopping_patience),
        epoch_callback=lambda row: append_jsonl(metrics_path, row),
    )
    policy.load_state_dict(result.best_state_dict, strict=True)
    policy.eval()
    summary = {
        **result.summary,
        "domain": str(args.domain),
        "scene": str(args.scene),
        "seed": int(args.seed),
        "checkpoint_purpose": str(args.checkpoint_purpose),
        "transformer_layers": int(args.transformer_layers),
        "checkpoint": str(checkpoint_path),
        "expert_data": expert_provenance(Path(args.expert_data)),
        "capability_threshold": float(args.min_validation_skill),
        "max_validation_mae": float(args.max_validation_mae),
        "validation_history": str(metrics_path),
        "validation_history_epochs": len(result.history),
    }

    metric_capability = bool(
        result.summary["validation_skill"] >= float(args.min_validation_skill)
        and result.summary["validation_mae"] <= float(args.max_validation_mae)
    )
    warm_start_passed = bool(
        args.checkpoint_purpose == "warm_start"
        and np.isfinite(result.summary["initial_validation_mse"])
        and np.isfinite(result.summary["validation_mse"])
        and result.summary["relative_validation_improvement"]
        >= float(args.min_warmup_relative_improvement)
    )
    summary["metric_capability_passed"] = metric_capability
    summary["warm_start_passed"] = warm_start_passed
    summary["warm_start_gate"] = {
        "configured_epochs": int(args.epochs),
        "maximum_epochs": int(args.max_warmup_epochs),
        "relative_validation_improvement": float(result.summary["relative_validation_improvement"]),
        "minimum_relative_validation_improvement": float(args.min_warmup_relative_improvement),
    }
    summary["checkpoint_saved"] = False
    summary["checkpoint_eligibility"] = (
        "warm_start_passed" if args.checkpoint_purpose == "warm_start" else "full_policy_best_validation"
    )
    checkpoint_eligible = warm_start_passed if args.checkpoint_purpose == "warm_start" else True
    write_json(out_dir / "split_manifest.json", result.split_trajectory_ids)

    # A full BC study retains its best-validation model even when capability
    # gates reject it; those gates control promotion, not artifact retention.
    # Short GAIL/AIRL warm-ups still have their own stabilization gate.
    if checkpoint_eligible:
        payload = checkpoint_payload(
            policy,
            cfg,
            args=args,
            obs_dim=obs_dim,
            action_dim=action_dim,
            training_summary=dict(result.summary),
            split_trajectory_ids=result.split_trajectory_ids,
            scenario=scenario,
        )
        save_checkpoint_artifacts(checkpoint_path, payload, summary)
    write_json(out_dir / "summary.json", summary)

    evaluation_cfg = replace(
        cfg,
        prebuilt_split=str(args.evaluation_split),
        enable_collision=bool(args.evaluation_enable_collision),
        bc_pretrain_eval_deterministic=True,
    )
    survival_stats = evaluate_policy_survival(
        policy,
        evaluation_cfg,
        device,
        episodes=int(args.evaluation_episodes),
        seed_offset=10_000,
    )
    summary["held_out_rollouts"] = survival_stats
    summary["held_out_evaluation"] = {
        "prebuilt_split": str(args.evaluation_split),
        "collision_physics_enabled": bool(args.evaluation_enable_collision),
        "episodes": int(args.evaluation_episodes),
        "metrics": survival_stats,
    }

    rollout_stats: dict[str, Any] | None = None
    if bool(args.render_video):
        rollout_stats = render_selected_replay(
            policy,
            cfg,
            episode_name=str(scenario[0]),
            vehicle_id=int(scenario[1]),
            device=device,
            video_path=str(out_dir / "evaluation.mp4"),
            steps=int(args.video_steps),
            deterministic=True,
            screen_width=1200,
            screen_height=608,
            scaling=5.5,
        )
        summary["rollout"] = rollout_stats

    rollout_capability = bool(
        survival_stats
        and float(survival_stats["bc_eval/mean_episode_length"]) >= float(args.min_rollout_steps)
        and float(survival_stats["bc_eval/crash_episode_fraction"]) <= float(args.max_crash_fraction)
        and float(survival_stats["bc_eval/offroad_episode_fraction"]) <= float(args.max_offroad_fraction)
    )
    summary["rollout_capability_passed"] = rollout_capability
    summary["capability_passed"] = bool(metric_capability and rollout_capability)
    write_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    requested_gate_passed = (
        warm_start_passed if args.checkpoint_purpose == "warm_start" else summary["capability_passed"]
    )
    if not requested_gate_passed and args.capability_failure_mode == "error":
        if args.checkpoint_purpose == "warm_start":
            raise RuntimeError(
                "BC warm-start stabilization gate failed: "
                f"relative_validation_improvement={result.summary['relative_validation_improvement']:.4f} "
                f"(minimum {float(args.min_warmup_relative_improvement):.4f})."
            )
        raise RuntimeError(
            "BC capability gate failed: "
            f"validation_skill={result.summary['validation_skill']:.4f} "
            f"(minimum {float(args.min_validation_skill):.4f}), "
            f"validation_mae={result.summary['validation_mae']:.4f} "
            f"(maximum {float(args.max_validation_mae):.4f}), "
            f"rollout_passed={rollout_capability}."
        )

if __name__ == "__main__":
    main()
