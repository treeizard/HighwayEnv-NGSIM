#!/usr/bin/env python3
"""Load expert data once and diagnose three-layer recurrent-BC optimization."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.recurrent_bc import (
    PreparedRecurrentBCData,
    SequenceWindow,
    prepare_recurrent_bc_data,
    train_recurrent_behavior_clone,
)
from scripts_gail.ps_gail.trainer import resolve_device


CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "dense_temporal_transformer_2layer",
        "policy_model": "recurrent_transformer",
        "transformer_layers": 2,
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "transformer_observation_tokenization": "dense_temporal",
        "transformer_memory_tokens": 1,
        "policy_head_init_std": 1.0e-2,
        "action_loss_weights": [1.0, 1.0],
        "action_loss_weighting": "inverse_variance",
        "correlation_loss_weight": 0.0,
        "variance_loss_weight": 0.0,
        "training_min_prediction_std_ratios": [0.0, 0.0],
        "selection_min_prediction_std_ratios": [0.25, 0.10],
        "selection_min_prediction_correlations": [0.50, 0.20],
    },
    {
        "candidate_id": "dense_temporal_transformer",
        "policy_model": "recurrent_transformer",
        "transformer_layers": 3,
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "transformer_observation_tokenization": "dense_temporal",
        "transformer_memory_tokens": 1,
        "policy_head_init_std": 1.0e-2,
        "action_loss_weights": [1.0, 1.0],
        "action_loss_weighting": "inverse_variance",
        "correlation_loss_weight": 0.0,
        "variance_loss_weight": 0.0,
        "training_min_prediction_std_ratios": [0.0, 0.0],
        "selection_min_prediction_std_ratios": [0.25, 0.10],
        "selection_min_prediction_correlations": [0.50, 0.20],
    },
    {
        "candidate_id": "recovery_v3_simple_gru",
        "policy_model": "recurrent_gru",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "policy_head_init_std": 1.0e-2,
        "action_loss_weights": [1.0, 1.0],
        "action_loss_weighting": "inverse_variance",
        "correlation_loss_weight": 0.0,
        "variance_loss_weight": 0.0,
        "training_min_prediction_std_ratios": [0.0, 0.0],
        "selection_min_prediction_std_ratios": [0.25, 0.10],
        "selection_min_prediction_correlations": [0.50, 0.20],
    },
    {
        "candidate_id": "recovery_v2_moment_prenorm",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "policy_head_init_std": 1.0e-2,
        "action_loss_weights": [1.0, 1.0],
        "action_loss_weighting": "inverse_variance",
        "correlation_loss_weight": 0.05,
        "variance_loss_weight": 0.05,
        "training_min_prediction_std_ratios": [0.25, 0.10],
        "selection_min_prediction_std_ratios": [0.25, 0.10],
        "selection_min_prediction_correlations": [0.50, 0.20],
    },
    {
        "candidate_id": "baseline_no_early_stop",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 0.5,
        "transformer_dropout": 0.1,
        "transformer_norm_first": False,
        "policy_head_init_std": -1.0,
    },
    {
        "candidate_id": "clip1_no_dropout",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 1.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": -1.0,
    },
    {
        "candidate_id": "small_head",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 1.0,
        "transformer_dropout": 0.1,
        "transformer_norm_first": False,
        "policy_head_init_std": 1.0e-3,
    },
    {
        "candidate_id": "small_head_prenorm",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 1.0,
        "transformer_dropout": 0.1,
        "transformer_norm_first": True,
        "policy_head_init_std": 1.0e-3,
    },
    {
        "candidate_id": "head01_clip5",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": 1.0e-2,
    },
    {
        "candidate_id": "head01_clip5_prenorm",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "policy_head_init_std": 1.0e-2,
    },
    {
        "candidate_id": "head01_lr1em3_clip5",
        "learning_rate": 1.0e-3,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": 1.0e-2,
    },
    {
        "candidate_id": "head01_lr1em4_clip5",
        "learning_rate": 1.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": 1.0e-2,
    },
    {
        "candidate_id": "head05_clip5",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": 5.0e-2,
    },
    {
        "candidate_id": "default_head_clip5",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": -1.0,
    },
    {
        "candidate_id": "head01_clip1",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 1.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": 1.0e-2,
    },
    {
        "candidate_id": "head01_clip5_drop01",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.1,
        "transformer_norm_first": False,
        "policy_head_init_std": 1.0e-2,
    },
    {
        "candidate_id": "head01_clip5_nowd",
        "learning_rate": 3.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": 1.0e-2,
        "weight_decay": 0.0,
    },
    {
        "candidate_id": "head05_lr1em3_clip5",
        "learning_rate": 1.0e-3,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": 5.0e-2,
    },
    {
        "candidate_id": "head05_lr1em4_clip5",
        "learning_rate": 1.0e-4,
        "max_grad_norm": 5.0,
        "transformer_dropout": 0.0,
        "transformer_norm_first": False,
        "policy_head_init_std": 5.0e-2,
    },
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def representative_windows(
    windows: list[SequenceWindow],
    actions: np.ndarray,
    *,
    limit: int,
) -> list[SequenceWindow]:
    """Select windows evenly across the acceleration-action distribution."""
    if limit <= 0 or limit >= len(windows):
        return list(windows)
    scored = sorted(
        windows,
        key=lambda window: (
            float(np.mean(actions[window.train_indices, 0])),
            window.trajectory_id,
            int(window.train_indices[0]),
        ),
    )
    positions = np.linspace(0, len(scored) - 1, num=int(limit), dtype=np.int64)
    return [scored[int(position)] for position in positions]


def uniform_windows(
    windows: list[SequenceWindow],
    *,
    limit: int,
    seed: int,
) -> list[SequenceWindow]:
    """Select a deterministic uniform subset without changing split membership."""
    if limit <= 0 or limit >= len(windows):
        return list(windows)
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(windows), size=int(limit), replace=False))
    return [windows[int(index)] for index in selected]


def diagnostic_view(
    prepared: PreparedRecurrentBCData,
    *,
    train_windows: int,
    validation_windows: int,
    test_windows: int,
    window_selection: str = "uniform",
    window_seed: int = 20260719,
) -> PreparedRecurrentBCData:
    limits = {
        "train": int(train_windows),
        "validation": int(validation_windows),
        "test": int(test_windows),
    }
    selected: dict[str, list[SequenceWindow]] = {}
    for offset, split in enumerate(("train", "validation", "test")):
        if window_selection == "stratified":
            selected[split] = representative_windows(
                prepared.split_windows[split], prepared.actions, limit=limits[split]
            )
        elif window_selection == "uniform":
            selected[split] = uniform_windows(
                prepared.split_windows[split],
                limit=limits[split],
                seed=int(window_seed) + offset,
            )
        else:
            raise ValueError(f"Unsupported window selection: {window_selection!r}")
    train_indices = np.concatenate([window.train_indices for window in selected["train"]])
    validation_indices = np.concatenate([window.train_indices for window in selected["validation"]])
    action_mean = prepared.actions[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    baseline = float(np.mean(np.square(prepared.actions[validation_indices] - action_mean)))
    return replace(prepared, split_windows=selected, validation_baseline_mse=baseline)


def make_policy(candidate: dict[str, Any], *, obs_dim: int, action_dim: int, device: torch.device) -> torch.nn.Module:
    policy = make_actor_critic(
        str(candidate.get("policy_model", "recurrent_transformer")),
        obs_dim=obs_dim,
        hidden_size=256,
        action_mode="continuous",
        continuous_action_dim=action_dim,
        transformer_layers=int(candidate.get("transformer_layers", 3)),
        transformer_heads=4,
        transformer_dropout=float(candidate["transformer_dropout"]),
        transformer_norm_first=bool(candidate["transformer_norm_first"]),
        transformer_observation_normalization=bool(
            candidate.get("transformer_observation_normalization", False)
        ),
        transformer_observation_tokenization=str(
            candidate.get("transformer_observation_tokenization", "semantic")
        ),
        transformer_memory_tokens=int(candidate.get("transformer_memory_tokens", 8)),
        transformer_memory_context_length=32,
        transformer_use_causal_attention=True,
    ).to(device)
    init_std = float(candidate["policy_head_init_std"])
    if init_std >= 0.0:
        torch.nn.init.normal_(policy.policy_head.weight, mean=0.0, std=init_std)
        torch.nn.init.zeros_(policy.policy_head.bias)
    return policy


def promotion_record(candidate: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Build a validation-only screen record; locked test remains unopened."""
    std_ratio = float(summary["validation_prediction_std_ratio"][0])
    correlation = float(summary["validation_prediction_target_correlation"][0])
    steering_std_ratio = float(summary["validation_prediction_std_ratio"][1])
    steering_correlation = float(summary["validation_prediction_target_correlation"][1])
    skill = float(summary["validation_skill"])
    saturation = float(summary["validation_prediction_saturation_fraction"][0])
    metrics_finite = bool(
        np.isfinite(
            [
                skill,
                std_ratio,
                correlation,
                steering_std_ratio,
                steering_correlation,
                saturation,
            ]
        ).all()
    )
    passed = bool(
        metrics_finite
        and skill >= 0.10
        and std_ratio >= 0.25
        and correlation >= 0.50
        and steering_std_ratio >= 0.10
        and steering_correlation >= 0.20
        and saturation <= 0.05
    )
    gate_margin = (
        min(
            skill / 0.10,
            std_ratio / 0.25,
            correlation / 0.50,
            steering_std_ratio / 0.10,
            steering_correlation / 0.20,
        )
        if metrics_finite and saturation <= 0.05
        else None
    )
    return {
        **candidate,
        "validation_skill": skill,
        "validation_mse": float(summary["validation_mse"]),
        "validation_mae": float(summary["validation_mae"]),
        "acceleration_std_ratio": std_ratio,
        "acceleration_correlation": correlation,
        "steering_std_ratio": steering_std_ratio,
        "steering_correlation": steering_correlation,
        "acceleration_saturation_fraction": saturation,
        "test_evaluation_status": "pending_deferred",
        "test_mse": None,
        "test_mae": None,
        "test_action_mse": None,
        "test_action_mae": None,
        "test_prediction_std_ratio": None,
        "test_prediction_target_correlation": None,
        "test_prediction_saturation_fraction": None,
        "metrics_finite": metrics_finite,
        "minimum_normalized_gate_margin": gate_margin,
        "initial_validation_mse": float(summary["initial_validation_mse"]),
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "last_gradient_clipped_fraction": float(summary["history"][-1]["gradient_clipped_fraction"]),
        "last_gradient_group_norm_mean": summary["history"][-1]["gradient_group_norm_mean"],
        "effective_action_loss_weights": summary["action_loss_weights"],
        "training_action_variance": summary["training_action_variance"],
        "validation_screen_passed": passed,
        "promotion_passed": False,
        "promotion_status": "pending_locked_test",
    }


def infer_diagnosis(records: list[dict[str, Any]], selected: dict[str, Any] | None) -> str:
    if selected is None:
        return "No local candidate escaped the constant-action basin; do not launch the production matrix."
    candidate_id = str(selected["candidate_id"])
    if candidate_id == "baseline_no_early_stop":
        return "The original depth failure is consistent with premature early stopping in a long constant-action basin."
    if candidate_id == "clip1_no_dropout":
        return "The three-layer model required less destructive global clipping and deterministic residual flow."
    if candidate_id == "small_head":
        return "The continuous tanh head initialization and global clipping trapped the deeper model near saturation/constant actions."
    if candidate_id == "head01_clip5":
        return "A moderate action-head initialization preserved encoder gradients, while relaxed clipping allowed the delayed learning transition."
    if candidate_id == "head01_clip5_prenorm":
        return "Small action-head initialization plus pre-norm restored learning in the deeper transformer."
    if candidate_id == "head01_lr1em3_clip5":
        return "Moderate action-head initialization, relaxed clipping, and a higher learning rate escaped the constant-action basin."
    return "The selected candidate restored observation-dependent learning."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-expert-samples", type=int, default=300_000)
    parser.add_argument("--data-seed", type=int, default=20260716)
    parser.add_argument("--split-seed", type=int, default=20260716)
    parser.add_argument("--seed", type=int, default=0, help="Policy initialization and batch-order seed.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-windows", type=int, default=512)
    parser.add_argument("--validation-windows", type=int, default=128)
    parser.add_argument("--test-windows", type=int, default=128)
    parser.add_argument(
        "--window-selection",
        choices=["uniform", "stratified"],
        default="uniform",
        help="How to choose bounded diagnostic windows within each trajectory-disjoint split.",
    )
    parser.add_argument("--window-seed", type=int, default=20260719)
    parser.add_argument("--candidate-limit", type=int, default=len(CANDIDATES))
    parser.add_argument("--candidate", action="append", dest="candidates", default=[])
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Use the same representative windows for all splits to test optimization capacity.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {out}")
    out.mkdir(parents=True)
    device = resolve_device(str(args.device))
    if device.type != "cuda":
        raise RuntimeError("Three-layer recovery diagnosis requires CUDA.")

    transitions = load_expert_transition_data(
        str(args.expert_data),
        max_samples=int(args.max_expert_samples),
        seed=int(args.data_seed),
        trajectory_frame="relative",
    )
    prepared = prepare_recurrent_bc_data(
        transitions,
        split_seed=int(args.split_seed),
        sequence_length=32,
        context_length=32,
    )
    view = diagnostic_view(
        prepared,
        train_windows=int(args.train_windows),
        validation_windows=int(args.validation_windows),
        test_windows=int(args.test_windows),
        window_selection=str(args.window_selection),
        window_seed=int(args.window_seed),
    )
    if bool(args.overfit):
        shared = list(view.split_windows["train"])
        shared_indices = np.concatenate([window.train_indices for window in shared])
        shared_actions = view.actions[shared_indices]
        shared_mean = shared_actions.mean(axis=0, dtype=np.float64).astype(np.float32)
        shared_baseline = float(np.mean(np.square(shared_actions - shared_mean)))
        view = replace(
            view,
            split_windows={"train": shared, "validation": shared, "test": shared},
            validation_baseline_mse=shared_baseline,
        )
    records: list[dict[str, Any]] = []
    requested = set(str(value) for value in args.candidates)
    candidates = [
        candidate
        for candidate in CANDIDATES
        if not requested or str(candidate["candidate_id"]) in requested
    ][: int(args.candidate_limit)]
    if not candidates:
        raise ValueError("No requested diagnostic candidates were found.")
    for candidate in candidates:
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        policy = make_policy(
            candidate,
            obs_dim=int(view.observations.shape[1]),
            action_dim=int(view.actions.shape[1]),
            device=device,
        )
        history: list[dict[str, Any]] = []
        result = train_recurrent_behavior_clone(
            policy,
            transitions,
            prepared_data=view,
            device=device,
            seed=int(args.seed),
            split_seed=int(args.split_seed),
            epochs=int(args.epochs),
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate.get("weight_decay", 1.0e-5)),
            sequence_length=32,
            sequences_per_batch=16,
            micro_batch_sequences=16,
            max_grad_norm=float(candidate["max_grad_norm"]),
            early_stopping_patience=0,
            selection_min_validation_skill=0.10,
            action_loss_weights=candidate.get("action_loss_weights"),
            action_loss_weighting=str(
                candidate.get("action_loss_weighting", "fixed")
            ),
            correlation_loss_weight=float(
                candidate.get("correlation_loss_weight", 0.0)
            ),
            variance_loss_weight=float(
                candidate.get("variance_loss_weight", 0.0)
            ),
            minimum_prediction_std_ratios=candidate.get(
                "training_min_prediction_std_ratios"
            ),
            selection_min_prediction_std_ratios=candidate.get(
                "selection_min_prediction_std_ratios"
            ),
            selection_min_prediction_correlations=candidate.get(
                "selection_min_prediction_correlations"
            ),
            epoch_callback=history.append,
        )
        summary = {**result.summary, "history": history}
        record = promotion_record(candidate, summary)
        records.append(record)
        write_json(out / str(candidate["candidate_id"]) / "summary.json", record)
        write_json(
            out / str(candidate["candidate_id"]) / "training_history.json",
            history,
        )
        del result, policy
        torch.cuda.empty_cache()

    def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        margin = row.get("minimum_normalized_gate_margin")
        return (
            float(margin) if margin is not None else float("-inf"),
            float(row["validation_skill"]),
            float(row["acceleration_correlation"]),
            float(row["acceleration_std_ratio"]),
        )

    ranked = sorted(records, key=rank_key, reverse=True)
    passing = [
        record for record in ranked if record["validation_screen_passed"]
    ]
    passing.sort(
        key=lambda row: (
            float(row["validation_skill"]),
            float(row["acceleration_correlation"]),
            float(row["acceleration_std_ratio"]),
        ),
        reverse=True,
    )
    selected = passing[0] if passing else None
    diagnosis = {
        "schema_version": 1,
        "study": "bc_3layer_local_load_once_diagnosis",
        "device": str(device),
        "expert_data": str(args.expert_data.resolve()),
        "expert_loader_calls": 1,
        "loaded_samples": int(len(transitions.policy_observations)),
        "policy_seed": int(args.seed),
        "window_selection": str(args.window_selection),
        "window_seed": int(args.window_seed),
        "screen_windows": {name: len(windows) for name, windows in view.split_windows.items()},
        "candidate_count": len(records),
        "overfit_mode": bool(args.overfit),
        "passing_count": len(passing),
        "selection_evidence": "validation_only",
        "locked_test_status": "pending_deferred",
        "promotion_status": "pending_locked_test",
        "ranked_candidate_ids": [str(record["candidate_id"]) for record in ranked],
        "selected_candidate": selected,
        "diagnosis": infer_diagnosis(records, selected),
        "candidates": records,
    }
    write_json(out / "diagnosis.json", diagnosis)
    print(json.dumps(diagnosis, indent=2, sort_keys=True))
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
