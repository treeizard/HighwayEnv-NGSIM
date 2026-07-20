#!/usr/bin/env python3
"""Combine canary and remaining stage-2 audits into one terminal 12-cell method audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine(
    campaign_path: Path,
    canary_audit_path: Path,
    remaining_audit_path: Path,
) -> dict[str, Any]:
    campaign_path = campaign_path.resolve()
    canary_audit_path = canary_audit_path.resolve()
    remaining_audit_path = remaining_audit_path.resolve()
    campaign = _read(campaign_path)
    canary = _read(canary_audit_path)
    remaining = _read(remaining_audit_path)
    if campaign.get("manifest_kind") != "gail_airl_final_12" or int(campaign.get("cell_count", -1)) != 12:
        raise RuntimeError("Expected a final 12-cell campaign manifest.")
    method = str(campaign["method"])
    simulator_profile = campaign.get("simulator_profile")
    expected = {
        "canary": [0, 3, 6, 9],
        "remaining": [1, 2, 4, 5, 7, 8, 10, 11],
    }
    audits = {"canary": canary, "remaining": remaining}
    all_artifacts: list[dict[str, Any]] = []
    for selection, audit in audits.items():
        if (
            audit.get("audit_kind") != "gail_airl_final_stage2_completion"
            or audit.get("status") != "passed"
            or audit.get("terminal") is not True
            or audit.get("method") != method
            or audit.get("selection") != selection
            or audit.get("simulator_profile") != simulator_profile
            or audit.get("canonical_indices") != expected[selection]
            or audit.get("source_code") != campaign.get("source_code")
        ):
            raise RuntimeError(f"{selection} audit does not match the campaign contract.")
        locked_manifest = Path(campaign["launch_manifests"][f"stage2_{selection}"]).resolve()
        if (
            Path(str(audit.get("manifest"))).resolve() != locked_manifest
            or audit.get("manifest_sha256") != _sha256(locked_manifest)
        ):
            raise RuntimeError(f"{selection} audit is not bound to the campaign stage2 manifest.")
        all_artifacts.extend(dict(row) for row in audit.get("artifacts") or ())
    indices = [int(row["canonical_index"]) for row in all_artifacts]
    if sorted(indices) != list(range(12)) or len(set(indices)) != 12:
        raise RuntimeError(f"Final audit does not contain exactly one artifact per canonical cell: {indices}")
    by_index = {int(row["canonical_index"]): row for row in all_artifacts}
    campaign_cells = {int(row["canonical_index"]): row for row in campaign.get("cells") or ()}
    if set(campaign_cells) != set(range(12)):
        raise RuntimeError("Campaign cell table is incomplete.")
    registry_inputs = []
    for index in range(12):
        artifact = by_index[index]
        cell = campaign_cells[index]
        if (
            artifact["best_checkpoint"] != cell["official_best_checkpoint"]
            or artifact["final_checkpoint"] != cell["official_final_checkpoint"]
        ):
            raise RuntimeError(f"Official artifact path mismatch for canonical cell {index}.")
        registry_inputs.extend(
            (
                {
                    "canonical_index": index,
                    "checkpoint_kind": "best",
                    "path": artifact["best_checkpoint"],
                    "sha256": artifact["best_checkpoint_sha256"],
                },
                {
                    "canonical_index": index,
                    "checkpoint_kind": "final",
                    "path": artifact["final_checkpoint"],
                    "sha256": artifact["final_checkpoint_sha256"],
                },
            )
        )
    return {
        "schema_version": 1,
        "audit_kind": "gail_airl_final_12_completion",
        "status": "passed",
        "terminal": True,
        "method": method,
        "simulator_profile": simulator_profile,
        "canonical_indices": list(range(12)),
        "campaign_manifest": str(campaign_path),
        "campaign_manifest_sha256": _sha256(campaign_path),
        "source_code": campaign.get("source_code"),
        "component_audits": {
            "canary": {"path": str(canary_audit_path), "sha256": _sha256(canary_audit_path)},
            "remaining": {
                "path": str(remaining_audit_path), "sha256": _sha256(remaining_audit_path)
            },
        },
        "artifacts": [by_index[index] for index in range(12)],
        "official_registry_inputs": registry_inputs,
        "registry_rule": "Consume only these 24 declared stage2 paths; do not recursively scan stage1.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--canary-audit", required=True, type=Path)
    parser.add_argument("--remaining-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = combine(args.campaign, args.canary_audit, args.remaining_audit)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(output), "method": result["method"], "cells": 12}, indent=2))


if __name__ == "__main__":
    main()
