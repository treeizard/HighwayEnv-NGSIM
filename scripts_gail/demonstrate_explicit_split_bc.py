#!/usr/bin/env python3
"""Demonstrate simple recurrent BC on collected train/validation/test roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import scripts_gail.ps_gail.recurrent_bc as recurrent_bc_module
from scripts_gail.diagnose_bc_3layer import (
    CANDIDATES,
    diagnostic_view,
    make_policy,
    promotion_record,
    write_json,
)
from scripts_gail.ps_gail.data import load_expert_transition_data
from scripts_gail.ps_gail.recurrent_bc import (
    PreparedRecurrentBCData,
    prepare_recurrent_bc_data_from_explicit_splits,
    train_recurrent_behavior_clone,
)
from scripts_gail.ps_gail.trainer import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--domain", choices=["us", "japanese"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=300_000)
    parser.add_argument("--max-validation-samples", type=int, default=100_000)
    parser.add_argument("--max-test-samples", type=int, default=100_000)
    parser.add_argument("--data-seed", type=int, default=20260716)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-windows", type=int, default=2048)
    parser.add_argument("--validation-windows", type=int, default=256)
    parser.add_argument("--test-windows", type=int, default=256)
    parser.add_argument("--window-seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _indices(prepared: PreparedRecurrentBCData, split: str) -> np.ndarray:
    return np.concatenate(
        [
            window.train_indices
            for window in prepared.split_windows[split]
        ]
    ).astype(np.int64, copy=False)


def _constant_mean_baseline(
    prepared: PreparedRecurrentBCData,
    split: str,
) -> dict[str, Any]:
    train_actions = prepared.actions[_indices(prepared, "train")]
    target_actions = prepared.actions[_indices(prepared, split)]
    train_mean = train_actions.mean(axis=0, dtype=np.float64)
    action_mse = np.mean(
        np.square(target_actions - train_mean),
        axis=0,
        dtype=np.float64,
    )
    return {
        "train_action_mean": train_mean.tolist(),
        "action_mse": action_mse.tolist(),
        "mse": float(action_mse.mean()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite demonstration output: {output}")
    output.mkdir(parents=True)
    device = resolve_device(str(args.device))
    if device.type != "cuda":
        raise RuntimeError("The local BC demonstration requires a CUDA device.")

    roots = {
        "train": args.collection_root / args.domain / "train",
        "validation": args.collection_root / args.domain / "val",
        "test": args.collection_root / args.domain / "test",
    }
    limits = {
        "train": int(args.max_train_samples),
        "validation": int(args.max_validation_samples),
        "test": int(args.max_test_samples),
    }
    source_transitions = {
        split: load_expert_transition_data(
            str(root),
            max_samples=limits[split],
            seed=int(args.data_seed) + offset,
            trajectory_frame="relative",
        )
        for offset, (split, root) in enumerate(roots.items())
    }
    prepared = prepare_recurrent_bc_data_from_explicit_splits(
        source_transitions,
        sequence_length=32,
        context_length=32,
    )
    view = diagnostic_view(
        prepared,
        train_windows=int(args.train_windows),
        validation_windows=int(args.validation_windows),
        test_windows=int(args.test_windows),
        window_selection="uniform",
        window_seed=int(args.window_seed),
    )
    candidate = next(
        row
        for row in CANDIDATES
        if row["candidate_id"] == "recovery_v3_simple_gru"
    )
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
        view.transitions,
        prepared_data=view,
        device=device,
        seed=int(args.seed),
        split_seed=0,
        epochs=int(args.epochs),
        learning_rate=float(candidate["learning_rate"]),
        weight_decay=float(candidate.get("weight_decay", 1.0e-5)),
        sequence_length=32,
        sequences_per_batch=16,
        micro_batch_sequences=16,
        max_grad_norm=float(candidate["max_grad_norm"]),
        early_stopping_patience=0,
        selection_min_validation_skill=0.10,
        action_loss_weights=candidate["action_loss_weights"],
        action_loss_weighting=str(candidate["action_loss_weighting"]),
        correlation_loss_weight=0.0,
        variance_loss_weight=0.0,
        minimum_prediction_std_ratios=candidate[
            "training_min_prediction_std_ratios"
        ],
        selection_min_prediction_std_ratios=candidate[
            "selection_min_prediction_std_ratios"
        ],
        selection_min_prediction_correlations=candidate[
            "selection_min_prediction_correlations"
        ],
        evaluate_test=True,
        epoch_callback=history.append,
    )
    summary = {**result.summary, "history": history}
    record = promotion_record(candidate, summary)
    validation_baseline = _constant_mean_baseline(view, "validation")
    test_baseline = _constant_mean_baseline(view, "test")
    record.update(
        {
            "schema_version": 1,
            "study": "explicit_time_split_simple_gru_bc_local_demonstration",
            "domain": str(args.domain),
            "device": str(device),
            "policy_seed": int(args.seed),
            "data_seed": int(args.data_seed),
            "window_seed": int(args.window_seed),
            "split_method": "explicit_collected_train_validation_test_directories",
            "source_roots": {
                split: str(root.resolve())
                for split, root in roots.items()
            },
            "loaded_samples": {
                split: int(len(transitions.policy_observations))
                for split, transitions in source_transitions.items()
            },
            "source_samples": {
                split: int(transitions.metadata["num_source_samples"])
                for split, transitions in source_transitions.items()
            },
            "screen_windows": {
                split: len(windows)
                for split, windows in view.split_windows.items()
            },
            "constant_train_mean_baseline": {
                "validation": validation_baseline,
                "test": test_baseline,
            },
            "test_skill_vs_train_mean": (
                1.0
                - float(record["test_mse"])
                / max(float(test_baseline["mse"]), 1.0e-12)
            ),
            "claim_boundary": (
                "Offline time-held-out action imitation only; closed-loop "
                "collision/off-road realism remains separately unqualified."
            ),
            "source_lock": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": _sha256(Path(__file__).resolve()),
                "recurrent_bc_module": str(
                    Path(recurrent_bc_module.__file__).resolve()
                ),
                "recurrent_bc_module_sha256": _sha256(
                    Path(recurrent_bc_module.__file__).resolve()
                ),
            },
        }
    )

    checkpoint = output / "offline_best.pt"
    temporary = output / ".offline_best.pt.tmp"
    try:
        torch.save(
            {
                "schema_version": 1,
                "checkpoint_kind": "offline_explicit_split_bc_demonstration",
                "policy_state_dict": result.best_state_dict,
                "policy_architecture": {
                    "policy_model": "recurrent_gru",
                    "obs_dim": int(view.observations.shape[1]),
                    "hidden_size": 256,
                    "continuous_action_dim": int(view.actions.shape[1]),
                    "memory_context_length": 32,
                    "observation_normalization": True,
                },
                "source_roots": record["source_roots"],
                "offline_metrics": record,
                "closed_loop_qualified": False,
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    record["checkpoint"] = str(checkpoint)
    record["checkpoint_sha256"] = _sha256(checkpoint)
    (output / "offline_best.pt.sha256").write_text(
        f"{record['checkpoint_sha256']}  offline_best.pt\n",
        encoding="utf-8",
    )
    write_json(output / "summary.json", record)
    write_json(output / "training_history.json", history)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not bool(record["promotion_passed"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
