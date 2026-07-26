#!/usr/bin/env python3
"""Attach verified exact-resume checkpoints to a fresh scratch GAIL manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from scripts_gail.ps_gail.checkpoints import (
    _RUNTIME_CONFIG_FIELDS,
    resume_config_hash,
    sha256_file,
    verify_checkpoint_sidecar,
)
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.experiment import resolve_algorithm_variant
from scripts_gail.ps_gail.pilot import load_manifest, select_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--failed-manifest", required=True, type=Path)
    parser.add_argument("--depth2-checkpoint", required=True, type=Path)
    parser.add_argument("--depth3-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load_checkpoint(
    path: Path,
    *,
    method: str,
    depth: int,
    seed: int,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    checkpoint_sha = verify_checkpoint_sidecar(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("training_state")
    if not isinstance(state, dict) or state.get("round_complete") is not True:
        raise RuntimeError(f"Checkpoint is not a completed-round boundary: {path}")
    completed_round = int(state.get("completed_round", -1))
    if completed_round != int(payload.get("round", -2)):
        raise RuntimeError(f"Checkpoint round metadata is inconsistent: {path}")
    expected = {
        "method": str(method),
        "transformer_layers": int(depth),
        "policy_seed": int(seed),
    }
    cell = dict(payload.get("study_cell") or payload.get("trial_identity") or {})
    observed = {
        "method": str(payload.get("method") or cell.get("method") or ""),
        "transformer_layers": int(cell.get("transformer_layers", -1)),
        "policy_seed": int(cell.get("policy_seed", -1)),
    }
    if observed != expected:
        raise RuntimeError(
            f"Checkpoint identity mismatch for {path}: {observed} != {expected}"
        )
    config = resolve_algorithm_variant(PSGAILConfig(**arguments), trainer=method)
    expected_config_hash = resume_config_hash(config)
    if str(payload.get("resume_config_hash") or "") != expected_config_hash:
        raise RuntimeError(
            f"Checkpoint continuation config differs from the fresh trial: {path}"
        )
    if completed_round >= int(arguments["total_rounds"]):
        raise RuntimeError(f"Checkpoint already reached total_rounds: {path}")
    return payload, checkpoint_sha


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_path = args.base_manifest.resolve()
    base = load_manifest(base_path)
    failed_path = args.failed_manifest.resolve()
    failed = load_manifest(failed_path)
    scope = dict(base["scope"])
    if scope.get("methods") != ["gail"] or scope.get("uses_bc_initialization") is not False:
        raise RuntimeError("Recovery base must be the two-cell scratch GAIL manifest.")
    checkpoints = {
        2: args.depth2_checkpoint.resolve(),
        3: args.depth3_checkpoint.resolve(),
    }
    failed_jobs = {2: "58508531", 3: "58508532"}
    recovered_trials: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for depth, seed in ((2, 0), (3, 1)):
        trial = select_trial(base, method="gail", depth=depth)
        failed_trial = select_trial(failed, method="gail", depth=depth)
        fresh_arguments = dict(trial["arguments"])
        failed_arguments = dict(failed_trial["arguments"])
        arguments = {
            name: (
                fresh_arguments[name]
                if name in _RUNTIME_CONFIG_FIELDS or name.startswith("wandb_")
                else failed_arguments[name]
            )
            for name in fresh_arguments
        }
        if set(arguments) != set(failed_arguments):
            raise RuntimeError(
                f"Fresh and failed depth-{depth} manifests have different config fields."
            )
        payload, checkpoint_sha = _load_checkpoint(
            checkpoints[depth],
            method="gail",
            depth=depth,
            seed=seed,
            arguments=arguments,
        )
        completed_round = int(payload["training_state"]["completed_round"])
        arguments.update(
            {
                "initial_policy_checkpoint": "",
                "resume_checkpoint": str(checkpoints[depth]),
                "expected_resume_round": completed_round,
            }
        )
        recovered = dict(trial)
        recovered.update(
            {
                "initialization": "random_seeded_exact_resume",
                "phase": "failed_job_recovery",
                "arguments": arguments,
                "replaces_failed_job": failed_jobs[depth],
            }
        )
        recovered_trials.append(recovered)
        provenance.append(
            {
                "depth": depth,
                "seed": seed,
                "failed_job": failed_jobs[depth],
                "resume_checkpoint": str(checkpoints[depth]),
                "resume_checkpoint_sha256": checkpoint_sha,
                "completed_round": completed_round,
                "next_round": completed_round + 1,
            }
        )
    scope.update(
        {
            "policy_initialization": "random_seeded_exact_resume",
            "exact_resume_recovery": True,
            "reason": (
                "Continue the original random-seeded policies from verified "
                "completed-round checkpoints after infrastructure/health-gate repairs."
            ),
        }
    )
    result = {
        **base,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "trials": recovered_trials,
        "recovery_provenance": {
            "base_manifest": str(base_path),
            "base_manifest_sha256": sha256_file(base_path),
            "failed_manifest": str(failed_path),
            "failed_manifest_sha256": sha256_file(failed_path),
            "replaces_failed_jobs": ["58508531", "58508532"],
            "repairs": [
                "discriminator saturation is diagnostic unless a fatal collapse gate also fails",
                "runtime projection converts aggregate evaluation work to wall time using evaluation_num_workers",
            ],
            "trials": provenance,
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recovery manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    load_manifest(output)
    return {
        "manifest": str(output),
        "trial_count": len(recovered_trials),
        "recovery_provenance": provenance,
    }


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
