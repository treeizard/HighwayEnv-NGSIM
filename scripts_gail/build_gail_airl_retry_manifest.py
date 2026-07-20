#!/usr/bin/env python3
"""Create one isolated, exact-resume retry manifest without reusing a failed run directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--new-run-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    sidecar = path.with_name(f"{path.name}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Exact retry requires checkpoint and SHA sidecar: {path}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Retry checkpoint SHA mismatch: {actual} != {expected}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("training_state")
    if not isinstance(state, dict) or state.get("round_complete") is not True:
        raise RuntimeError("Retry checkpoint is not an exact completed-round boundary.")
    if int(state.get("completed_round", -1)) != int(payload.get("round", -2)):
        raise RuntimeError("Retry checkpoint round metadata is inconsistent.")
    if not payload.get("resume_config_hash"):
        raise RuntimeError("Retry checkpoint has no continuation config hash.")
    return payload, actual


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_manifest.resolve()
    with source_path.open(encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("manifest_kind") != "gail_airl_final_stage":
        raise RuntimeError("Retry source must be a locked final-stage manifest, not screening input.")
    rows = list(source.get("trials") or ())
    if args.trial_index < 0 or args.trial_index >= len(rows):
        raise IndexError(args.trial_index)
    source_row = dict(rows[args.trial_index])
    checkpoint, checkpoint_sha = _verify_checkpoint(args.resume_checkpoint)
    method = str(source.get("method"))
    if str(checkpoint.get("method")) != method:
        raise RuntimeError("Retry checkpoint method differs from the source trial.")
    arguments = dict(source_row.get("arguments") or {})
    completed_round = int(checkpoint["training_state"]["completed_round"])
    total_rounds = int(arguments["total_rounds"])
    stop_after_round = int(arguments.get("stop_after_round", 0))
    intended_end = stop_after_round if stop_after_round > 0 else total_rounds
    if completed_round >= intended_end:
        raise RuntimeError(
            f"Source stage already reached its declared boundary ({completed_round} >= {intended_end})."
        )
    run_name = PurePosixPath(str(args.new_run_name).strip())
    if not str(run_name) or run_name.is_absolute() or ".." in run_name.parts:
        raise ValueError("new-run-name must be a safe non-empty relative path.")
    run_root = Path(str(arguments["run_root"])).resolve()
    target = (run_root / run_name).resolve()
    try:
        target.relative_to(run_root)
    except ValueError as exc:
        raise RuntimeError(f"Retry output escapes run root: {target}") from exc
    if target.exists():
        raise FileExistsError(f"Refusing to reuse retry output: {target}")
    arguments.update(
        {
            "run_name": str(run_name),
            "initial_policy_checkpoint": "",
            "resume_checkpoint": str(args.resume_checkpoint.resolve()),
            "expected_resume_round": completed_round,
        }
    )
    row = dict(source_row)
    row.update(
        {
            "index": 0,
            "trial_id": f"{source_row['trial_id']}_retry_after_{completed_round:04d}",
            "phase": f"{source_row['phase']}_retry",
            "initialization": "resume",
            "run_name": str(run_name),
            "arguments": arguments,
        }
    )
    payload = {
        **{key: value for key, value in source.items() if key != "trials"},
        "selection": "retry",
        "canonical_indices": [int(arguments["study_cell_index"])],
        "trial_count": 1,
        "retry_provenance": {
            "source_manifest": str(source_path),
            "source_manifest_sha256": _sha256(source_path),
            "source_trial_index": int(args.trial_index),
            "resume_checkpoint": str(args.resume_checkpoint.resolve()),
            "resume_checkpoint_sha256": checkpoint_sha,
            "completed_round": completed_round,
            "intended_end_round": intended_end,
        },
        "trials": [row],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite retry manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "manifest": str(output),
        "completed_round": completed_round,
        "next_round": completed_round + 1,
        "new_run_dir": str(target),
    }


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
