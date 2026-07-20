#!/usr/bin/env python3
"""Execute one indexed trial from a generated GAIL/AIRL study manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts_gail.ps_gail.study import StudyTrial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--expected-method", choices=("gail", "airl"), default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_trial(path: str, index: int) -> StudyTrial:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("trials") or []
    if index < 0 or index >= len(rows):
        raise IndexError(f"trial-index {index} outside [0, {len(rows) - 1}].")
    return StudyTrial(**rows[index])


def verify_source_lock(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    source = dict(payload.get("source_code") or {})
    hashes = dict(source.get("files_sha256") or {})
    if not hashes:
        raise RuntimeError("Method-specific manifest is missing its source-code hash lock.")
    expected_repo = Path(os.environ.get("REPODIR", os.getcwd())).resolve()
    recorded_repo = Path(str(source.get("repo") or "")).resolve()
    if recorded_repo != expected_repo:
        raise RuntimeError(f"Manifest source repo mismatch: {recorded_repo} != {expected_repo}")
    for relative, expected in hashes.items():
        digest = hashlib.sha256()
        with (expected_repo / relative).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Submission-locked source changed: {relative} ({expected} != {actual})"
            )


def main() -> None:
    args = parse_args()
    trial = load_trial(args.manifest, args.trial_index)
    if args.expected_method and trial.method != args.expected_method:
        raise RuntimeError(
            f"Method-specific array expected {args.expected_method}, got {trial.method} "
            f"for trial {trial.trial_id}."
        )
    if args.expected_method:
        verify_source_lock(args.manifest)
        run_root = Path(str(trial.arguments.get("run_root", "") or "")).expanduser()
        if not run_root.is_absolute():
            raise RuntimeError(f"Trial {trial.trial_id} requires an absolute --run-root.")
        run_dir = (run_root / str(trial.run_name)).resolve()
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing trial output: {run_dir}")
        rollout_workers = int(trial.arguments.get("num_rollout_workers", 0))
        rollout_threads = int(trial.arguments.get("rollout_worker_threads", 0))
        evaluation_workers = int(trial.arguments.get("evaluation_num_workers", 0))
        evaluation_threads = int(trial.arguments.get("evaluation_worker_threads", 0))
        if min(rollout_workers, rollout_threads, evaluation_workers, evaluation_threads) <= 0:
            raise RuntimeError(f"Trial {trial.trial_id} is missing explicit worker geometry.")
        allocated_cpus = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "32")))
        if (
            rollout_workers * rollout_threads > allocated_cpus
            or evaluation_workers * evaluation_threads > allocated_cpus
        ):
            raise RuntimeError(
                f"Trial {trial.trial_id} worker geometry exceeds {allocated_cpus} allocated CPUs."
            )
    bc_path = str(trial.arguments.get("initial_policy_checkpoint", "") or "")
    if trial.initialization == "bc" and not os.path.isfile(bc_path):
        raise FileNotFoundError(f"Trial {trial.trial_id} requires BC checkpoint: {bc_path}")
    command = trial.command(python=args.python)
    print(command, flush=True)
    if not args.dry_run:
        subprocess.run(trial.argv(python=args.python), check=True)


if __name__ == "__main__":
    main()
