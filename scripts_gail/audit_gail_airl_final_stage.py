#!/usr/bin/env python3
"""Fail-closed artifact audit for a completed final-campaign stage-2 selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Missing checkpoint/sidecar: {path}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Checkpoint hash mismatch: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    return payload, actual


def audit(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("manifest_kind") != "gail_airl_final_stage" or int(manifest.get("stage", -1)) != 2:
        raise RuntimeError("Completion audit requires a final stage-2 manifest.")
    method = str(manifest["method"])
    canonical_indices = list(manifest.get("canonical_indices") or ())
    rows = list(manifest.get("trials") or ())
    if len(rows) != len(canonical_indices) or not rows:
        raise RuntimeError("Stage-2 manifest rows/indices are inconsistent.")
    records: list[dict[str, Any]] = []
    for local_index, (canonical_index, row) in enumerate(zip(canonical_indices, rows, strict=True)):
        if int(row["index"]) != local_index:
            raise RuntimeError("Stage-2 array indices are not contiguous.")
        arguments = dict(row["arguments"])
        if int(arguments["study_cell_index"]) != int(canonical_index):
            raise RuntimeError("Stage-2 canonical identity mismatch.")
        run_root = Path(str(arguments["run_root"])).resolve()
        run_dir = (run_root / str(row["run_name"])).resolve()
        failure = run_dir / "training_failure.json"
        if failure.exists():
            raise RuntimeError(f"Training failure artifact exists: {failure}")
        best_payload, best_sha = _verified_checkpoint(run_dir / "best.pt")
        final_payload, final_sha = _verified_checkpoint(run_dir / "final.pt")
        resume_payload, resume_sha = _verified_checkpoint(run_dir / "resume_latest.pt")
        if int(final_payload.get("round", -1)) != 800 or int(resume_payload.get("round", -1)) != 800:
            raise RuntimeError(f"Cell {canonical_index} did not reach round 800.")
        if final_payload.get("method") != method or resume_payload.get("method") != method:
            raise RuntimeError(f"Cell {canonical_index} method mismatch.")
        final_identity = dict(final_payload.get("study_cell") or {})
        best_identity = dict(best_payload.get("study_cell") or {})
        if (
            int(final_identity.get("canonical_index", -1)) != int(canonical_index)
            or int(final_identity.get("campaign_stage", -1)) != 2
            or final_identity.get("official") is not True
        ):
            raise RuntimeError(f"Cell {canonical_index} final identity is not official stage2.")
        if (
            int(best_identity.get("canonical_index", -1)) != int(canonical_index)
            or int(best_identity.get("campaign_stage", -1)) != 2
            or best_identity.get("official") is not True
        ):
            raise RuntimeError(f"Cell {canonical_index} best identity is not official stage2.")
        exact_state = resume_payload.get("training_state")
        if (
            not isinstance(exact_state, dict)
            or exact_state.get("round_complete") is not True
            or int(exact_state.get("completed_round", -1)) != 800
        ):
            raise RuntimeError(f"Cell {canonical_index} resume_latest is not exact round 800 state.")
        if "training_state" in final_payload or "training_state" in best_payload:
            raise RuntimeError(f"Cell {canonical_index} inference artifacts are not lean.")
        evaluation_path = run_dir / "evaluation_summary.json"
        if not evaluation_path.is_file():
            raise FileNotFoundError(evaluation_path)
        with evaluation_path.open(encoding="utf-8") as handle:
            evaluation = json.load(handle)
        if evaluation.get("trainer") != method or not isinstance(evaluation.get("test"), dict):
            raise RuntimeError(f"Cell {canonical_index} evaluation summary is incomplete.")
        records.append(
            {
                "canonical_index": int(canonical_index),
                "domain": arguments["study_domain"],
                "transformer_layers": int(arguments["transformer_layers"]),
                "policy_seed": int(arguments["seed"]),
                "best_checkpoint": str(run_dir / "best.pt"),
                "best_checkpoint_sha256": best_sha,
                "best_carried_from_stage1": "carried_forward_from" in best_payload,
                "final_checkpoint": str(run_dir / "final.pt"),
                "final_checkpoint_sha256": final_sha,
                "resume_checkpoint": str(run_dir / "resume_latest.pt"),
                "resume_checkpoint_sha256": resume_sha,
                "normalized_config_hash": final_payload.get("normalized_config_hash"),
                "evaluation_summary": str(evaluation_path),
            }
        )
    result = {
        "schema_version": 1,
        "audit_kind": "gail_airl_final_stage2_completion",
        "status": "passed",
        "terminal": True,
        "method": method,
        "selection": manifest.get("selection"),
        "simulator_profile": manifest.get("simulator_profile"),
        "source_code": manifest.get("source_code"),
        "canonical_indices": canonical_indices,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "artifacts": records,
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.output)
    print(json.dumps({"status": result["status"], "cells": len(result["artifacts"])}, indent=2))


if __name__ == "__main__":
    main()
