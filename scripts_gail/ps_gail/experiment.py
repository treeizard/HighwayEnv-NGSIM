"""Experiment identity, provenance, and machine-readable result helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import torch

from .config import PSGAILConfig


ALGORITHM_VARIANTS = {
    "gail_bce",
    "gail_wgan_gp",
    "airl_bce",
    "airl_wgan_gp",
}


def resolve_algorithm_variant(cfg: PSGAILConfig, *, trainer: str) -> PSGAILConfig:
    """Validate and apply an explicit algorithm objective contract."""
    trainer = str(trainer).strip().lower()
    if trainer not in {"gail", "airl"}:
        raise ValueError(f"trainer must be 'gail' or 'airl', got {trainer!r}.")
    variant = str(getattr(cfg, "algorithm_variant", "auto") or "auto").strip().lower()
    if variant == "auto":
        return cfg
    if variant not in ALGORITHM_VARIANTS:
        raise ValueError(
            f"Unsupported algorithm_variant={variant!r}; expected one of "
            f"{sorted(ALGORITHM_VARIANTS)} or 'auto'."
        )
    if not variant.startswith(f"{trainer}_"):
        raise ValueError(f"{trainer} trainer cannot run algorithm_variant={variant!r}.")
    if variant == "gail_bce":
        return replace(
            cfg,
            algorithm_variant=variant,
            discriminator_loss="bce",
            normalize_gail_reward=False,
            allow_wgan_reward_normalization=False,
        )
    if variant == "gail_wgan_gp":
        return replace(cfg, algorithm_variant=variant, discriminator_loss="wgan_gp")
    if variant == "airl_bce":
        return replace(
            cfg,
            algorithm_variant=variant,
            discriminator_loss="airl_bce",
            airl_policy_reward_mode="discriminator",
            normalize_gail_reward=False,
            allow_wgan_reward_normalization=False,
        )
    return replace(cfg, algorithm_variant=variant, discriminator_loss="wgan_gp")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def config_hash(cfg: PSGAILConfig) -> str:
    encoded = json.dumps(_jsonable(vars(cfg)), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _path_provenance(path: str) -> dict[str, Any]:
    expanded = os.path.abspath(os.path.expanduser(str(path))) if path else ""
    return {
        "path": expanded,
        "exists": bool(expanded and os.path.exists(expanded)),
        "kind": "directory" if expanded and os.path.isdir(expanded) else "file",
    }


def write_run_manifest(run_dir: str, cfg: PSGAILConfig, *, trainer: str) -> str:
    """Write immutable-at-start run identity before expensive data loading."""
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trainer": str(trainer),
        "algorithm_variant": str(cfg.algorithm_variant),
        "seed": int(cfg.seed),
        "config_hash": config_hash(cfg),
        "config": _jsonable(vars(cfg)),
        "data": {
            "expert": _path_provenance(cfg.expert_data),
            "episode_root": _path_provenance(cfg.episode_root),
            "prebuilt_split": str(cfg.prebuilt_split),
            "validation_split": str(cfg.validation_prebuilt_split),
            "test_split": str(cfg.test_prebuilt_split),
        },
        "initialization": {
            "initial_policy_checkpoint": _path_provenance(cfg.initial_policy_checkpoint),
            "resume_checkpoint": _path_provenance(cfg.resume_checkpoint),
            "bc_pretrain_epochs": int(cfg.bc_pretrain_epochs),
        },
        "collision": {
            "enable_collision": bool(cfg.enable_collision),
            "schedule": str(cfg.collision_mode_schedule),
            "rollout_fixed_horizon": bool(
                getattr(cfg, "rollout_fixed_horizon", False)
            ),
            "mixed_on_fraction": float(cfg.collision_mixed_on_fraction),
            "proxy_penalty_coef": float(cfg.collision_proxy_penalty_coef),
            "vehicle_increase_soft_collision_rounds": int(
                cfg.vehicle_increase_soft_collision_rounds
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": _git_metadata(),
        },
    }
    path = os.path.join(run_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def write_evaluation_summary(
    run_dir: str,
    cfg: PSGAILConfig,
    *,
    trainer: str,
    best_validation_score: float,
    best_validation_round: int,
    final_validation_metrics: dict[str, float] | None,
    stress_metrics: dict[str, float] | None,
    test_metrics: dict[str, float] | None,
    initial_validation_metrics: dict[str, float] | None = None,
    selected_validation_metrics: dict[str, float] | None = None,
    initializer_test_metrics: dict[str, float] | None = None,
    final_policy_test_metrics: dict[str, float] | None = None,
    policy_relative_l2_delta: float | None = None,
    selected_checkpoint: str = "best.pt",
    best_full_load_score: float | None = None,
    best_full_load_round: int = 0,
) -> str:
    payload = {
        "schema_version": 2,
        "trainer": str(trainer),
        "algorithm_variant": str(cfg.algorithm_variant),
        "seed": int(cfg.seed),
        "config_hash": config_hash(cfg),
        "best_validation_score": _jsonable(float(best_validation_score)),
        "best_validation_round": int(best_validation_round),
        "initial_validation": _jsonable(initial_validation_metrics or {}),
        "final_validation": _jsonable(final_validation_metrics or {}),
        "selected_checkpoint": str(selected_checkpoint),
        "selected_validation": _jsonable(selected_validation_metrics or {}),
        "validation_stress": _jsonable(stress_metrics or {}),
        "initializer_test": _jsonable(initializer_test_metrics or {}),
        "test": _jsonable(test_metrics or {}),
        "final_policy_test": _jsonable(final_policy_test_metrics or {}),
        "policy_relative_l2_delta": _jsonable(policy_relative_l2_delta),
    }
    if best_full_load_score is not None:
        payload["best_full_load_score"] = _jsonable(best_full_load_score)
        payload["best_full_load_round"] = int(best_full_load_round)
    path = os.path.join(run_dir, "evaluation_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def load_verified_policy_state(path: str) -> dict[str, torch.Tensor]:
    """Load a policy state only after validating the checkpoint sidecar."""
    from .checkpoints import verify_checkpoint_sidecar

    verify_checkpoint_sidecar(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("policy_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint does not contain a policy state: {path}")
    return {str(key): value.detach().cpu().clone() for key, value in state.items()}


def policy_relative_l2_delta(
    selected: dict[str, torch.Tensor],
    initializer: dict[str, torch.Tensor],
) -> float:
    """Return a scale-normalized parameter distance for learning audits."""
    if set(selected) != set(initializer):
        raise RuntimeError("Selected and initializer policy state keys differ.")
    squared_delta = 0.0
    squared_reference = 0.0
    for key in sorted(selected):
        current = selected[key].detach().to(dtype=torch.float64, device="cpu")
        reference = initializer[key].detach().to(dtype=torch.float64, device="cpu")
        if current.shape != reference.shape:
            raise RuntimeError(f"Policy tensor shape mismatch for {key}.")
        squared_delta += float(torch.sum((current - reference) ** 2).item())
        squared_reference += float(torch.sum(reference ** 2).item())
    return float(np.sqrt(squared_delta) / max(np.sqrt(squared_reference), 1.0e-12))


def write_training_failure(
    run_dir: str,
    cfg: PSGAILConfig,
    *,
    trainer: str,
    round_idx: int,
    reasons: list[str],
) -> str:
    payload = {
        "schema_version": 1,
        "trainer": str(trainer),
        "algorithm_variant": str(cfg.algorithm_variant),
        "seed": int(cfg.seed),
        "config_hash": config_hash(cfg),
        "round": int(round_idx),
        "reasons": list(reasons),
    }
    path = os.path.join(run_dir, "training_failure.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


__all__ = [
    "ALGORITHM_VARIANTS",
    "config_hash",
    "load_verified_policy_state",
    "policy_relative_l2_delta",
    "resolve_algorithm_variant",
    "write_evaluation_summary",
    "write_run_manifest",
    "write_training_failure",
]
