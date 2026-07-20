#!/usr/bin/env python3
"""Promote a five-seed safety-eligible confirmation arm to a locked final recipe."""

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


def promote(method: str, report_path: Path, manifest_path: Path) -> dict[str, Any]:
    method = str(method).lower()
    if method not in {"gail", "airl"}:
        raise ValueError(method)
    report_path = report_path.resolve()
    manifest_path = manifest_path.resolve()
    report = _read(report_path)
    manifest = _read(manifest_path)
    trials = list(manifest.get("trials") or ())
    confirmation_rows = [row for row in trials if row.get("method") == method]
    if len(confirmation_rows) != 10 or int(report.get("completed", -1)) != len(trials):
        raise RuntimeError("Confirmation report is not terminal for the full 20-run comparison.")
    groups = [
        row for row in list(report.get("configuration_groups") or ())
        if row.get("method") == method and row.get("phase") == "confirmation_winner"
    ]
    if len(groups) != 1:
        raise RuntimeError(f"Expected one {method} confirmation_winner group.")
    group = groups[0]
    if group.get("seed_gate_passed") is not True or int(group.get("eligible_seeds", -1)) != 5:
        raise RuntimeError(f"{method} confirmation winner did not pass all five safety seeds.")
    trial_ids = set(group.get("trial_ids") or ())
    expected_ids = {
        row["trial_id"]
        for row in confirmation_rows
        if row.get("phase") == "confirmation_winner"
    }
    if trial_ids != expected_ids or len(expected_ids) != 5:
        raise RuntimeError("Confirmation report trial IDs do not match the locked winner arm.")
    representative = dict(group.get("representative") or {})
    if representative.get("eligible") is not True or representative.get("method") != method:
        raise RuntimeError("Confirmation representative is not safety eligible.")
    arguments = dict(representative.get("arguments") or {})
    for runtime_key in (
        "run_name", "run_root", "seed", "resume_checkpoint", "stop_after_round",
        "expected_resume_round", "study_domain", "study_cell_index", "study_stage",
    ):
        arguments.pop(runtime_key, None)
    variant = str(arguments.get("algorithm_variant") or "")
    if not variant.startswith(f"{method}_"):
        raise RuntimeError(f"Confirmed recipe variant does not match {method}: {variant!r}")
    return {
        "schema_version": 1,
        "recipe_kind": "gail_airl_confirmed_final",
        "status": "passed",
        "terminal": True,
        "safety_eligible": True,
        "method": method,
        "confirmation_report": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "confirmation_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "confirmation_group": {
            "configuration_id": group.get("configuration_id"),
            "eligible_seeds": 5,
            "trial_ids": sorted(trial_ids),
            "mean_normalized_trajectory_error": group.get("mean_normalized_trajectory_error"),
        },
        "arguments": arguments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("gail", "airl"))
    parser.add_argument("--confirmation-report", required=True, type=Path)
    parser.add_argument("--confirmation-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = promote(args.method, args.confirmation_report, args.confirmation_manifest)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite confirmed recipe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(output), "method": args.method, "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
