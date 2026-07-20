"""Atomic, integrity-checked checkpoint helpers shared by GAIL and AIRL."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any
import warnings

import numpy as np
import torch

from .config import PSGAILConfig


_RUNTIME_CONFIG_FIELDS = {
    "device",
    "allow_legacy_model_only_resume",
    "allow_unverified_resume_checkpoint",
    "initial_policy_checkpoint",
    "resume_checkpoint",
    "run_name",
    "run_root",
    "stop_after_round",
    "expected_resume_round",
    "study_stage",
    "wandb_entity",
    "wandb_group",
    "wandb_mode",
    "wandb_project",
    "wandb_tags",
    "wandb_watch",
}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_config_hash(cfg: PSGAILConfig) -> str:
    """Hash scientific configuration while excluding location/runtime-only fields."""
    payload = {
        key: value
        for key, value in vars(cfg).items()
        if key not in _RUNTIME_CONFIG_FIELDS and not key.startswith("wandb_")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def resume_config_hash(cfg: PSGAILConfig) -> str:
    """Hash the scientific continuation contract, excluding stage/runtime controls."""
    payload = {
        key: value
        for key, value in vars(cfg).items()
        if key not in _RUNTIME_CONFIG_FIELDS and not key.startswith("wandb_")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def study_domain(cfg: PSGAILConfig) -> str:
    explicit = str(getattr(cfg, "study_domain", "") or "").strip().lower()
    if explicit:
        return explicit
    scene = str(cfg.scene).strip().lower()
    expert = str(cfg.expert_data).strip().lower()
    if "japan" in scene or "japan" in expert:
        return "japanese"
    if scene in {"us-101", "i-80"} or "ngsim" in expert:
        return "us"
    return scene.replace("_", "-")


def checkpoint_metadata(
    cfg: PSGAILConfig,
    *,
    method: str,
    checkpoint_kind: str,
) -> dict[str, Any]:
    method = str(method).strip().lower()
    checkpoint_kind = str(checkpoint_kind).strip().lower()
    cell_kind = _study_checkpoint_kind(checkpoint_kind)
    cell = {
        "method": method,
        "domain": study_domain(cfg),
        "transformer_layers": int(cfg.transformer_layers),
        "policy_seed": int(cfg.seed),
        "checkpoint_kind": cell_kind,
    }
    canonical_index = int(getattr(cfg, "study_cell_index", -1))
    identity_key = "study_cell" if canonical_index >= 0 else "trial_identity"
    if canonical_index >= 0:
        cell["canonical_index"] = canonical_index
        stage = int(getattr(cfg, "study_stage", 0))
        cell["campaign_stage"] = stage
        cell["official"] = stage == 2
    return {
        "checkpoint_schema_version": 2,
        "checkpoint_kind": checkpoint_kind,
        "method": method,
        "identity_scope": "campaign_cell" if canonical_index >= 0 else "trial",
        "normalized_config_hash": normalized_config_hash(cfg),
        "resume_config_hash": resume_config_hash(cfg),
        identity_key: cell,
    }


def _study_checkpoint_kind(checkpoint_kind: str) -> str:
    kind = str(checkpoint_kind).strip().lower()
    for prefix in ("gail_", "airl_"):
        if kind.startswith(prefix):
            kind = kind[len(prefix) :]
            break
    return "warm_start" if kind == "bc_pretrained" else kind


def set_checkpoint_kind(payload: dict[str, Any], checkpoint_kind: str) -> dict[str, Any]:
    """Return a shallow copy with synchronized top-level/study-cell kind metadata."""
    result = dict(payload)
    kind = str(checkpoint_kind).strip().lower()
    result["checkpoint_kind"] = kind
    identity_key = "study_cell" if "study_cell" in result else "trial_identity"
    cell = dict(result.get(identity_key) or {})
    cell["checkpoint_kind"] = _study_checkpoint_kind(kind)
    result[identity_key] = cell
    return result


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_torch_save(
    payload: object,
    path: str | os.PathLike[str],
    *,
    refuse_overwrite: bool = False,
) -> str:
    """Atomically save a checkpoint and matching ``.sha256`` sidecar."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and target.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    digest = sha256_file(target)
    _atomic_write_text(target.with_name(f"{target.name}.sha256"), f"{digest}  {target.name}\n")
    return digest


def atomic_copy_verified_checkpoint(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    refuse_overwrite: bool = True,
) -> str:
    """Copy a checkpoint only after validating its SHA sidecar, then write atomically."""
    source_path = Path(source)
    target_path = Path(target)
    expected = verify_checkpoint_sidecar(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and target_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".tmp", dir=target_path.parent
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source_path, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        actual = sha256_file(temporary_name)
        if actual != expected:
            raise RuntimeError(
                f"Checkpoint changed during verified copy: {expected} != {actual} ({source_path})"
            )
        os.replace(temporary_name, target_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    _atomic_write_text(
        target_path.with_name(f"{target_path.name}.sha256"),
        f"{expected}  {target_path.name}\n",
    )
    return expected


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(f"Exact resume RNG state is incomplete: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_states = list(state.get("torch_cuda") or ())
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from exact-resume checkpoint: "
                f"{len(cuda_states)} != {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in cuda_states]
        )


def exact_training_state(
    *,
    completed_round: int,
    optimizers: dict[str, torch.optim.Optimizer | None],
    trainer_state: dict[str, Any],
) -> dict[str, Any]:
    """Capture all mutable state needed to continue at ``completed_round + 1``."""
    return {
        "schema_version": 1,
        "completed_round": int(completed_round),
        "round_complete": True,
        "optimizer_state_dicts": {
            name: optimizer.state_dict()
            for name, optimizer in optimizers.items()
            if optimizer is not None
        },
        "rng_state": capture_rng_state(),
        "trainer_state": dict(trainer_state),
    }


def restore_exact_training_state(
    checkpoint: dict[str, Any],
    *,
    optimizers: dict[str, torch.optim.Optimizer | None],
    expected_resume_config_hash: str,
    expected_method: str | None = None,
    allow_legacy_model_only: bool = False,
) -> dict[str, Any] | None:
    """Restore optimizer/RNG/trainer state, or explicitly allow legacy weights-only load."""
    state = checkpoint.get("training_state")
    stored_hash = str(checkpoint.get("resume_config_hash") or "")
    stored_method = str(checkpoint.get("method") or "").strip().lower()
    if expected_method is not None and stored_method != str(expected_method).strip().lower():
        if not allow_legacy_model_only or stored_method:
            raise RuntimeError(
                f"Resume checkpoint method mismatch: {stored_method or 'missing'} != {expected_method}"
            )
    if not isinstance(state, dict) or not stored_hash:
        if not allow_legacy_model_only:
            raise RuntimeError(
                "Resume checkpoint lacks exact training state/config identity. "
                "Use --allow-legacy-model-only-resume only for an intentional ablation."
            )
        warnings.warn(
            "Using legacy model-only resume: optimizer, RNG, replay, curriculum, and health state "
            "will restart from round 1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if stored_hash != str(expected_resume_config_hash):
        raise RuntimeError(
            "Resume configuration hash mismatch: "
            f"{stored_hash} != {expected_resume_config_hash}"
        )
    if not bool(state.get("round_complete", False)):
        raise RuntimeError("Resume checkpoint was not captured at a completed round boundary.")
    completed_round = int(state.get("completed_round", -1))
    if completed_round < 0 or completed_round != int(checkpoint.get("round", -2)):
        raise RuntimeError("Resume checkpoint round and exact training-state round disagree.")
    optimizer_states = dict(state.get("optimizer_state_dicts") or {})
    required_optimizers = {name for name, optimizer in optimizers.items() if optimizer is not None}
    missing_optimizers = sorted(required_optimizers.difference(optimizer_states))
    if missing_optimizers:
        raise RuntimeError(f"Resume checkpoint is missing optimizer states: {missing_optimizers}")
    for name, optimizer in optimizers.items():
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_states[name])
    trainer_state = state.get("trainer_state")
    if not isinstance(trainer_state, dict):
        raise RuntimeError("Resume checkpoint is missing trainer runtime state.")
    restore_rng_state(dict(state.get("rng_state") or {}))
    return {
        "completed_round": completed_round,
        "start_round": completed_round + 1,
        "trainer_state": trainer_state,
    }


def verify_resume_checkpoint(
    path: str | os.PathLike[str],
    *,
    allow_unverified: bool = False,
) -> str | None:
    """Verify an existing sidecar; missing sidecars require an explicit ablation flag."""
    target = Path(path)
    sidecar = target.with_name(f"{target.name}.sha256")
    if sidecar.is_file():
        return verify_checkpoint_sidecar(target)
    if not allow_unverified:
        raise FileNotFoundError(
            f"Resume checkpoint SHA-256 sidecar is missing: {sidecar}. "
            "Use --allow-unverified-resume-checkpoint only for an intentional ablation."
        )
    warnings.warn(
        f"Loading resume checkpoint without SHA-256 verification: {target}",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def verify_checkpoint_sidecar(path: str | os.PathLike[str]) -> str:
    target = Path(path)
    sidecar = target.with_name(f"{target.name}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Checkpoint SHA-256 sidecar is missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(target)
    if expected != actual:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {expected} != {actual} ({target})")
    return actual


__all__ = [
    "atomic_copy_verified_checkpoint",
    "atomic_torch_save",
    "capture_rng_state",
    "checkpoint_metadata",
    "exact_training_state",
    "normalized_config_hash",
    "restore_exact_training_state",
    "restore_rng_state",
    "resume_config_hash",
    "set_checkpoint_kind",
    "sha256_file",
    "study_domain",
    "verify_checkpoint_sidecar",
    "verify_resume_checkpoint",
]
