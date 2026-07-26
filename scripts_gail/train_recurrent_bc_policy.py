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
from scripts_gail.ps_gail.contracts import validate_training_data_contracts
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.recurrent_bc import (
    PreparedRecurrentBCData,
    train_recurrent_behavior_clone,
)
from scripts_gail.ps_gail.trainer import (
    evaluate_policy_matched_trajectories,
    resolve_device,
)
from scripts_gail.ps_gail.validation import (
    PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
    paper_driver_model_validation_overrides,
    scored_validation_metrics,
)
from scripts_gail.train_simple_ps_gail import evaluate_policy_survival


CHECKPOINT_SCHEMA_VERSION = 1


def _comma_separated_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _comma_separated_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def training_artifact_is_complete(summary: dict[str, Any], out_dir: Path) -> bool:
    """Return whether a BC run produced a finite, integrity-checkable artifact."""
    finite_scalars = (
        summary.get("initial_validation_mse"),
        summary.get("validation_mse"),
        summary.get("validation_mae"),
    )
    diagnostic_vectors = (
        summary.get("validation_prediction_std_ratio"),
        summary.get("validation_prediction_target_correlation"),
    )
    return bool(
        summary.get("checkpoint_saved")
        and str(summary.get("checkpoint_sha256") or "")
        and (out_dir / "best.pt").is_file()
        and (out_dir / "best.pt.sha256").is_file()
        and (out_dir / "split_manifest.json").is_file()
        and all(value is not None and np.isfinite(float(value)) for value in finite_scalars)
        and all(
            isinstance(values, (list, tuple))
            and len(values) > 0
            for values in diagnostic_vectors
        )
    )


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
    parser.add_argument(
        "--action-loss-weights",
        type=_comma_separated_floats,
        default=[1.0, 1.0],
        help="Comma-separated BC optimization weights in [acceleration, steering] order.",
    )
    parser.add_argument(
        "--action-loss-weighting",
        choices=["fixed", "inverse_variance"],
        default="fixed",
        help=(
            "Use fixed weights directly or divide them by each training-split "
            "action variance before normalization."
        ),
    )
    parser.add_argument(
        "--correlation-loss-weight",
        type=float,
        default=0.0,
        help="Weight on the per-micro-batch action correlation anti-collapse loss.",
    )
    parser.add_argument(
        "--variance-loss-weight",
        type=float,
        default=0.0,
        help="Weight on the one-sided prediction standard-deviation anti-collapse loss.",
    )
    parser.add_argument(
        "--training-min-prediction-std-ratios",
        type=_comma_separated_floats,
        default=[],
        help="Per-action training targets for the one-sided prediction variance loss.",
    )
    parser.add_argument(
        "--learning-action-index",
        type=int,
        default=0,
        help="Action dimension used by anti-collapse prediction-variance and correlation gates.",
    )
    parser.add_argument(
        "--min-learning-action-std-ratio",
        type=float,
        default=0.0,
        help="Minimum validation prediction/target standard-deviation ratio for the learning action.",
    )
    parser.add_argument(
        "--min-learning-action-correlation",
        type=float,
        default=-1.0,
        help="Minimum validation prediction/target correlation for the learning action.",
    )
    parser.add_argument(
        "--learning-action-indices",
        type=_comma_separated_ints,
        default=[],
        help="Optional comma-separated action dimensions that must all pass.",
    )
    parser.add_argument(
        "--min-learning-action-std-ratios",
        type=_comma_separated_floats,
        default=[],
        help="Per-dimension minimum prediction/target standard-deviation ratios.",
    )
    parser.add_argument(
        "--min-learning-action-correlations",
        type=_comma_separated_floats,
        default=[],
        help="Per-dimension minimum prediction/target correlations.",
    )
    parser.add_argument(
        "--require-explicit-data-contracts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, required=True, choices=[2, 3])
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument(
        "--transformer-norm-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pre-norm transformer encoder blocks for optimization stability.",
    )
    parser.add_argument(
        "--transformer-observation-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit and persist a train-split-only standardizer for policy observations.",
    )
    parser.add_argument(
        "--transformer-observation-tokenization",
        choices=["semantic", "dense_temporal"],
        default="semantic",
        help=(
            "Use sensor-wise current tokens or one normalized observation token "
            "with transformer attention over temporal memory."
        ),
    )
    parser.add_argument(
        "--policy-head-init-std",
        type=float,
        default=-1.0,
        help="Reinitialize the continuous policy head with this std; negative preserves framework initialization.",
    )
    parser.add_argument("--memory-tokens", type=int, default=8)
    parser.add_argument("--memory-context-length", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequences-per-batch", type=int, default=16)
    parser.add_argument("--micro-batch-sequences", type=int, default=4)

    parser.add_argument("--render-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-steps", type=int, default=200)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument(
        "--validation-evaluation-episodes",
        type=int,
        default=3,
        help="All-vehicle validation episodes used for shared BC/GAIL ranking.",
    )
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
    parser.add_argument(
        "--matched-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the same fixed-horizon matched-trajectory evaluator used by GAIL.",
    )
    parser.add_argument(
        "--evaluation-vehicle-mode",
        choices=["single", "training_count", "all"],
        default="single",
    )
    parser.add_argument("--min-rollout-steps", type=int, default=100)
    parser.add_argument("--max-crash-fraction", type=float, default=0.34)
    parser.add_argument("--max-offroad-fraction", type=float, default=0.34)
    parser.add_argument(
        "--capability-failure-mode",
        choices=["error", "report"],
        default="error",
        help=(
            "For full policy training, error exits non-zero only when a finite "
            "checkpoint artifact is not produced. Offline and closed-loop "
            "quality are always reported. Warm starts retain their stabilization "
            "gate. Code and data failures always exit non-zero."
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
    policy_model = str(getattr(args, "policy_model", "recurrent_transformer"))
    return PSGAILConfig(
        expert_data=str(Path(args.expert_data).resolve()),
        run_name=f"bc_{args.domain}_{policy_model}_{args.transformer_layers}layer_seed_{args.seed}",
        scene=str(args.scene),
        action_mode="continuous",
        episode_root=str(Path(args.episode_root).resolve()),
        prebuilt_split=str(args.prebuilt_split),
        seed=int(args.seed),
        max_expert_samples=int(args.max_expert_samples),
        require_explicit_data_contracts=bool(
            args.require_explicit_data_contracts
        ),
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
        road_query_mode="spatial",
        collision_check_mode="broadphase",
        record_replay_diagnostics=False,
        sensor_road_edge_mode="batched",
        reuse_pre_reset_spaces=True,
        policy_model=policy_model,
        hidden_size=int(args.hidden_size),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_dropout=float(args.transformer_dropout),
        transformer_norm_first=bool(args.transformer_norm_first),
        transformer_observation_normalization=bool(
            args.transformer_observation_normalization
        ),
        transformer_observation_tokenization=str(
            args.transformer_observation_tokenization
        ),
        transformer_memory_tokens=int(args.memory_tokens),
        transformer_memory_context_length=int(args.memory_context_length),
        transformer_recurrent_sequence_length=int(args.sequence_length),
        transformer_recurrent_sequences_per_batch=int(args.sequences_per_batch),
        transformer_recurrent_micro_batch_sequences=int(args.micro_batch_sequences),
        transformer_use_causal_attention=True,
        evaluation_num_workers=1,
        evaluation_worker_threads=2,
        **paper_driver_model_validation_overrides(),
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
    training_data_contract: dict[str, object],
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
            "policy_model": str(cfg.policy_model),
            "obs_dim": int(obs_dim),
            "hidden_size": int(cfg.hidden_size),
            "action_mode": "continuous",
            "continuous_action_dim": int(action_dim),
            "transformer_layers": int(cfg.transformer_layers),
            "transformer_heads": int(cfg.transformer_heads),
            "transformer_dropout": float(cfg.transformer_dropout),
            "transformer_norm_first": bool(cfg.transformer_norm_first),
            "transformer_observation_normalization": bool(
                cfg.transformer_observation_normalization
            ),
            "transformer_observation_tokenization": str(
                cfg.transformer_observation_tokenization
            ),
            "policy_head_init_std": float(args.policy_head_init_std),
            "transformer_memory_tokens": int(cfg.transformer_memory_tokens),
            "transformer_memory_context_length": int(cfg.transformer_memory_context_length),
            "transformer_use_causal_attention": True,
        },
        "bc_stats": training_summary,
        "training_data_contract": training_data_contract,
        "policy_output_action_contract": training_data_contract[
            "continuous_action"
        ],
        "policy_observation_contract": training_data_contract[
            "policy_observation"
        ],
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


def run_training(
    args: argparse.Namespace,
    *,
    transitions: Any | None = None,
    prepared_data: PreparedRecurrentBCData | None = None,
) -> dict[str, Any]:
    """Run one BC cell, optionally reusing already-loaded and prepared data."""
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

    if float(args.policy_head_init_std) >= 0.0:
        policy_head = getattr(policy, "policy_head", None)
        if not isinstance(policy_head, torch.nn.Linear):
            raise TypeError("Policy-head initialization requires a linear continuous policy head.")
        torch.nn.init.normal_(policy_head.weight, mean=0.0, std=float(args.policy_head_init_std))
        torch.nn.init.zeros_(policy_head.bias)

    if transitions is None:
        transitions = load_expert_transition_data(
            str(args.expert_data),
            max_samples=int(args.max_expert_samples),
            seed=int(args.data_seed),
            trajectory_frame="relative",
        )
    training_data_contract = validate_training_data_contracts(
        transitions.metadata,
        lidar_cells=int(cfg.cells),
        maximum_range=float(cfg.maximum_range),
        require_explicit=bool(args.require_explicit_data_contracts),
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
        selection_min_validation_skill=float(args.min_validation_skill),
        action_loss_weights=list(args.action_loss_weights),
        action_loss_weighting=str(args.action_loss_weighting),
        correlation_loss_weight=float(args.correlation_loss_weight),
        variance_loss_weight=float(args.variance_loss_weight),
        minimum_prediction_std_ratios=(
            list(args.training_min_prediction_std_ratios)
            if args.training_min_prediction_std_ratios
            else None
        ),
        selection_min_prediction_std_ratios=(
            list(args.min_learning_action_std_ratios)
            if args.min_learning_action_std_ratios
            else None
        ),
        selection_min_prediction_correlations=(
            list(args.min_learning_action_correlations)
            if args.min_learning_action_correlations
            else None
        ),
        epoch_callback=lambda row: append_jsonl(metrics_path, row),
        prepared_data=prepared_data,
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
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "transformer_dropout": float(args.transformer_dropout),
        "transformer_norm_first": bool(args.transformer_norm_first),
        "transformer_observation_normalization": bool(
            args.transformer_observation_normalization
        ),
        "transformer_observation_tokenization": str(
            args.transformer_observation_tokenization
        ),
        "policy_head_init_std": float(args.policy_head_init_std),
        "early_stopping_patience": int(args.early_stopping_patience),
        "configured_action_loss_weights": list(args.action_loss_weights),
        "action_loss_weighting": str(args.action_loss_weighting),
        "correlation_loss_weight": float(args.correlation_loss_weight),
        "variance_loss_weight": float(args.variance_loss_weight),
        "training_min_prediction_std_ratios": list(
            args.training_min_prediction_std_ratios
        ),
        "training_data_contract": training_data_contract,
        "policy_output_action_contract": training_data_contract[
            "continuous_action"
        ],
        "policy_observation_contract": training_data_contract[
            "policy_observation"
        ],
    }

    validation_std_ratios = result.summary["validation_prediction_std_ratio"]
    validation_correlations = result.summary["validation_prediction_target_correlation"]
    learning_action_indices = (
        list(args.learning_action_indices)
        if args.learning_action_indices
        else [int(args.learning_action_index)]
    )
    minimum_std_ratios = (
        list(args.min_learning_action_std_ratios)
        if args.min_learning_action_std_ratios
        else [float(args.min_learning_action_std_ratio)]
    )
    minimum_correlations = (
        list(args.min_learning_action_correlations)
        if args.min_learning_action_correlations
        else [float(args.min_learning_action_correlation)]
    )
    if not (
        len(learning_action_indices)
        == len(minimum_std_ratios)
        == len(minimum_correlations)
    ):
        raise ValueError(
            "Learning action indices and per-action gate thresholds must have "
            "the same length."
        )
    action_names = ("acceleration_norm", "steering_norm")
    learning_signal_gates: list[dict[str, object]] = []
    for action_index, minimum_std_ratio, minimum_correlation in zip(
        learning_action_indices,
        minimum_std_ratios,
        minimum_correlations,
        strict=True,
    ):
        if not 0 <= int(action_index) < len(validation_std_ratios):
            raise ValueError(
                f"learning_action_index={action_index} is outside the action "
                f"dimension [0, {len(validation_std_ratios)})."
            )
        std_ratio = float(validation_std_ratios[int(action_index)])
        correlation = float(validation_correlations[int(action_index)])
        passed = bool(
            std_ratio >= float(minimum_std_ratio)
            and correlation >= float(minimum_correlation)
        )
        learning_signal_gates.append(
            {
                "action_index": int(action_index),
                "action_name": action_names[int(action_index)],
                "prediction_std_ratio": std_ratio,
                "minimum_prediction_std_ratio": float(minimum_std_ratio),
                "prediction_target_correlation": correlation,
                "minimum_prediction_target_correlation": float(
                    minimum_correlation
                ),
                "passed": passed,
            }
        )
    learning_signal_passed = all(
        bool(gate["passed"]) for gate in learning_signal_gates
    )
    metric_capability = bool(
        result.summary["validation_skill"] >= float(args.min_validation_skill)
        and result.summary["validation_mae"] <= float(args.max_validation_mae)
        and learning_signal_passed
    )
    summary["learning_signal_passed"] = learning_signal_passed
    summary["learning_signal_gate"] = {
        "aggregation": "all_required_actions",
        "actions": learning_signal_gates,
        # Compatibility fields for older tuning/report readers. The ``actions``
        # list is authoritative and all listed dimensions must pass.
        "action_index": int(learning_signal_gates[0]["action_index"]),
        "prediction_std_ratio": float(
            learning_signal_gates[0]["prediction_std_ratio"]
        ),
        "minimum_prediction_std_ratio": float(
            learning_signal_gates[0]["minimum_prediction_std_ratio"]
        ),
        "prediction_target_correlation": float(
            learning_signal_gates[0]["prediction_target_correlation"]
        ),
        "minimum_prediction_target_correlation": float(
            learning_signal_gates[0][
                "minimum_prediction_target_correlation"
            ]
        ),
    }
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
            training_summary={
                **result.summary,
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "max_grad_norm": float(args.max_grad_norm),
                "transformer_dropout": float(args.transformer_dropout),
                "transformer_norm_first": bool(args.transformer_norm_first),
                "transformer_observation_normalization": bool(
                    args.transformer_observation_normalization
                ),
                "transformer_observation_tokenization": str(
                    args.transformer_observation_tokenization
                ),
                "policy_head_init_std": float(args.policy_head_init_std),
                "early_stopping_patience": int(args.early_stopping_patience),
                "learning_signal_passed": learning_signal_passed,
                "learning_signal_gate": dict(summary["learning_signal_gate"]),
                "configured_action_loss_weights": list(
                    args.action_loss_weights
                ),
                "action_loss_weighting": str(args.action_loss_weighting),
                "correlation_loss_weight": float(args.correlation_loss_weight),
                "variance_loss_weight": float(args.variance_loss_weight),
                "training_min_prediction_std_ratios": list(
                    args.training_min_prediction_std_ratios
                ),
            },
            training_data_contract=training_data_contract,
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
    survival_stats: dict[str, Any] = {}
    matched_evaluation_passed = True
    matched_evaluation_complete = not bool(args.matched_evaluation)
    if bool(args.matched_evaluation):
        # BC and GAIL use this exact fixed-horizon, collision-enabled evaluator
        # and scoring function. Collision termination is ignored so poor BC
        # behavior remains measurable through the requested horizon.
        matched_cfg = replace(
            evaluation_cfg,
            enable_collision=True,
            **paper_driver_model_validation_overrides(),
        )
        validation_stats = evaluate_policy_matched_trajectories(
            policy,
            matched_cfg,
            device,
            split="val",
            episodes=int(args.validation_evaluation_episodes),
            prefix="validation",
        )
        validation_stats, validation_cost, validation_score = (
            scored_validation_metrics(
                validation_stats,
                matched_cfg,
                prefix="validation",
            )
        )
        test_stats = evaluate_policy_matched_trajectories(
            policy,
            matched_cfg,
            device,
            split="test",
            episodes=int(args.evaluation_episodes),
            prefix="test",
        )
        score_horizon = int(matched_cfg.validation_score_horizon_seconds)
        validation_coverage = float(
            validation_stats.get(
                f"validation/horizon_coverage_{score_horizon}s",
                float("nan"),
            )
        )
        test_coverage = float(
            test_stats.get(
                f"test/horizon_coverage_{score_horizon}s",
                float("nan"),
            )
        )
        matched_evaluation_passed = bool(
            np.isfinite(validation_score)
            and np.isfinite(validation_coverage)
            and validation_coverage
            >= float(matched_cfg.validation_min_horizon_coverage)
            and np.isfinite(test_coverage)
            and test_coverage
            >= float(matched_cfg.validation_min_horizon_coverage)
        )
        matched_evaluation_complete = bool(
            validation_stats
            and test_stats
            and np.isfinite(validation_coverage)
            and np.isfinite(test_coverage)
        )
        summary["gail_aligned_matched_evaluation"] = {
            "validation_split": "val",
            "test_split": "test",
            "collision_physics_enabled": True,
            "collision_termination_enabled": False,
            "vehicle_mode": str(matched_cfg.test_vehicle_mode),
            "validation_episodes": int(args.validation_evaluation_episodes),
            "test_episodes": int(args.evaluation_episodes),
            "score_horizon_seconds": score_horizon,
            "minimum_horizon_coverage": float(
                matched_cfg.validation_min_horizon_coverage
            ),
            "checkpoint_selection": {
                "framework": PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
                "validation_cost": float(validation_cost),
                "validation_score": float(validation_score),
                "components": {
                    key: value
                    for key, value in validation_stats.items()
                    if key.startswith("validation/score_component_")
                },
            },
            "validation_metrics": validation_stats,
            "test_metrics": test_stats,
            "passed": matched_evaluation_passed,
        }
        summary["paper_validation_cost"] = float(validation_cost)
        summary["paper_validation_score"] = float(validation_score)
        summary["held_out_rollouts"] = test_stats
        summary["held_out_evaluation"] = {
            "prebuilt_split": "test",
            "collision_physics_enabled": True,
            "collision_termination_enabled": False,
            "vehicle_mode": str(matched_cfg.test_vehicle_mode),
            "episodes": int(args.evaluation_episodes),
            "metrics": test_stats,
        }
    else:
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

    rollout_capability = (
        bool(matched_evaluation_passed)
        if bool(args.matched_evaluation)
        else bool(
            survival_stats
            and float(survival_stats["bc_eval/mean_episode_length"])
            >= float(args.min_rollout_steps)
            and float(survival_stats["bc_eval/crash_episode_fraction"])
            <= float(args.max_crash_fraction)
            and float(survival_stats["bc_eval/offroad_episode_fraction"])
            <= float(args.max_offroad_fraction)
        )
    )
    summary["rollout_capability_passed"] = rollout_capability
    summary["matched_evaluation_passed"] = matched_evaluation_passed
    summary["capability_passed"] = bool(
        metric_capability
        and rollout_capability
        and matched_evaluation_passed
    )
    summary["closed_loop_quality_passed"] = bool(
        rollout_capability and matched_evaluation_passed
    )
    summary["closed_loop_evaluation_complete"] = bool(
        matched_evaluation_complete
        if bool(args.matched_evaluation)
        else survival_stats
    )
    summary["training_artifact_complete"] = training_artifact_is_complete(
        summary,
        out_dir,
    )
    summary["interpretability_baseline_eligible"] = bool(
        summary["training_artifact_complete"] and metric_capability
    )
    summary["benchmark_contract"] = {
        "objective": "behavior_cloning_interpretability_reference",
        "matrix_completion_gate": "all_training_artifacts_complete",
        "interpretability_eligibility_gate": "offline_imitation_learning",
        "closed_loop_metrics_role": "descriptive_non_terminal",
        "shared_validation_framework": PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
        "validation_vehicle_mode": (
            "all" if bool(args.matched_evaluation) else str(args.evaluation_vehicle_mode)
        ),
        "collision_termination_enabled": False,
        "closed_loop_metrics_include": [
            "position_rmse_by_horizon",
            "speed_rmse_by_horizon",
            "lane_offset_rmse_by_horizon",
            "vehicle_crash_rate",
            "vehicle_offroad_rate",
            "hard_brake_agent_step_rate",
        ],
    }
    write_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    requested_gate_passed = (
        warm_start_passed
        if args.checkpoint_purpose == "warm_start"
        else summary["training_artifact_complete"]
    )
    if not requested_gate_passed and args.capability_failure_mode == "error":
        if args.checkpoint_purpose == "warm_start":
            raise RuntimeError(
                "BC warm-start stabilization gate failed: "
                f"relative_validation_improvement={result.summary['relative_validation_improvement']:.4f} "
                f"(minimum {float(args.min_warmup_relative_improvement):.4f})."
            )
        raise RuntimeError(
            "BC training artifact is incomplete or non-finite. Closed-loop "
            "quality metrics are descriptive and cannot trigger this error."
        )

    return summary


def main() -> None:
    run_training(parse_args())

if __name__ == "__main__":
    main()
