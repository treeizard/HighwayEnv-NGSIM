#!/usr/bin/env python3
"""Run the confirmation-first 12-cell recurrent IQ-Learn study serially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from scripts_gail.iq_study import (
    bc_checkpoint, cell_relative_path, expert_for_domain, matrix_cells, read_locked_recipe, trainer_command,
)
from scripts_gail.train_recurrent_iq_learn import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--bc-root", type=Path, required=True)
    parser.add_argument("--us-expert", type=Path, required=True)
    parser.add_argument("--japanese-expert", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--model-limit", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def completed_cell_record(
    run_dir: Path,
    *,
    domain: str,
    depth: int,
    seed: int,
    relative: Path,
) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    checkpoint = run_dir / "best.pt"
    sidecar = run_dir / "best.pt.sha256"
    if not (summary_path.is_file() and checkpoint.is_file() and sidecar.is_file()):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(checkpoint)
    passed = bool(
        summary.get("capability_passed")
        and summary.get("best_update")
        and summary.get("best_checkpoint_sha256") == actual
        and expected == actual
    )
    if not passed:
        return None
    return {
        "domain": domain,
        "transformer_layers": depth,
        "seed": seed,
        "relative_path": str(relative),
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual,
        "best_update": summary.get("best_update"),
        "capability_passed": True,
        "returncode": 0,
        "resumed": True,
    }


def preserve_incomplete_run(run_dir: Path) -> Path:
    attempt = 1
    while True:
        destination = run_dir.with_name(f"{run_dir.name}_incomplete_attempt_{attempt:02d}")
        if not destination.exists():
            run_dir.rename(destination)
            return destination
        attempt += 1


def main() -> None:
    args = parse_args()
    recipe = read_locked_recipe(args.recipe.resolve())
    policy_root = args.policy_root.resolve()
    if policy_root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to reuse IQ policy root: {policy_root}")
    policy_root.mkdir(parents=True, exist_ok=bool(args.resume))
    cells = matrix_cells(confirmation_first=True)[: int(args.model_limit)]
    if not 1 <= int(args.model_limit) <= 12:
        raise ValueError("model-limit must be in [1, 12]")
    records: list[dict[str, Any]] = []
    progress_path = policy_root / "matrix_progress.json"
    for index, (domain, depth, seed) in enumerate(cells):
        relative = cell_relative_path(domain, depth, seed)
        run_dir = policy_root / relative
        if args.resume:
            resumed_record = completed_cell_record(
                run_dir, domain=domain, depth=depth, seed=seed, relative=relative,
            )
            if resumed_record is not None:
                records.append(resumed_record)
                continue
            if run_dir.exists():
                preserve_incomplete_run(run_dir)
        initial = bc_checkpoint(args.bc_root.resolve(), domain, depth, seed)
        if not initial.is_file():
            raise FileNotFoundError(f"Matched BC initializer is missing: {initial}")
        command = trainer_command(
            recipe, domain=domain, depth=depth, seed=seed,
            expert_data=expert_for_domain(domain, args.us_expert.resolve(), args.japanese_expert.resolve()),
            episode_root=args.episode_root.resolve(), initial_checkpoint=initial,
            out_dir=run_dir, device=str(args.device), capability_failure_mode="report",
        )
        completed = subprocess.run(command, check=False)
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        passed = bool(completed.returncode == 0 and summary.get("capability_passed") and summary.get("best_update"))
        record = {
            "domain": domain, "transformer_layers": depth, "seed": seed,
            "relative_path": str(relative), "summary": str(summary_path),
            "checkpoint": summary.get("best_checkpoint"), "checkpoint_sha256": summary.get("best_checkpoint_sha256"),
            "best_update": summary.get("best_update"), "capability_passed": passed,
            "returncode": completed.returncode,
        }
        records.append(record)
        write_json(progress_path, {
            "schema_version": 1, "study": "online_recurrent_iq_domain_depth_v1",
            "study_id": str(args.study_id), "status": "training",
            "confirmation_first": True, "requested_model_count": len(cells),
            "completed_model_count": len(records), "capability_passed_count": sum(int(row["capability_passed"]) for row in records),
            "cells": records,
        })
        if index < 2 and not passed:
            write_json(policy_root / "confirmation_failure.json", {
                "schema_version": 1, "status": "confirmation_failed", "failed_cell": record, "cells": records,
            })
            raise RuntimeError(f"IQ confirmation failed for {domain}, depth={depth}, seed={seed}")
    manifest = {
        "schema_version": 1, "study": "online_recurrent_iq_domain_depth_v1",
        "study_id": str(args.study_id), "recipe": str(args.recipe.resolve()),
        "recipe_payload": recipe, "model_count": len(records),
        "capability_passed_count": sum(int(row["capability_passed"]) for row in records), "cells": records,
    }
    write_json(policy_root / "matrix_manifest.json", manifest)
    write_json(progress_path, {
        **manifest, "status": "learning_qualified" if manifest["capability_passed_count"] == len(records) else "trained_with_failures",
        "completed_model_count": len(records), "requested_model_count": len(cells),
    })
    if manifest["capability_passed_count"] != len(records):
        raise RuntimeError(f"Only {manifest['capability_passed_count']}/{len(records)} IQ cells passed capability gates.")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
