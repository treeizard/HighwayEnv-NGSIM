"""Atomic, integrity-checked checkpoint helpers shared by GAIL and AIRL."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from .config import PSGAILConfig


_RUNTIME_CONFIG_FIELDS = {
    "device",
    "resume_checkpoint",
    "run_name",
    "run_root",
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
    return {
        "checkpoint_schema_version": 2,
        "checkpoint_kind": checkpoint_kind,
        "method": method,
        "normalized_config_hash": normalized_config_hash(cfg),
        "study_cell": {
            "method": method,
            "domain": study_domain(cfg),
            "transformer_layers": int(cfg.transformer_layers),
            "policy_seed": int(cfg.seed),
            "checkpoint_kind": cell_kind,
        },
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
    cell = dict(result.get("study_cell") or {})
    cell["checkpoint_kind"] = _study_checkpoint_kind(kind)
    result["study_cell"] = cell
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
    "atomic_torch_save",
    "checkpoint_metadata",
    "normalized_config_hash",
    "set_checkpoint_kind",
    "sha256_file",
    "study_domain",
    "verify_checkpoint_sidecar",
]
