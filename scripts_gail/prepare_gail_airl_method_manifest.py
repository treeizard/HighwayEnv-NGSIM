#!/usr/bin/env python3
"""Create one overwrite-safe, resource-explicit GAIL or AIRL array manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("gail", "airl"))
    parser.add_argument("--selection", choices=("all", "canary", "production"), default="all")
    parser.add_argument("--canary-count", type=int, default=4)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--study-domain", default="")
    parser.add_argument("--num-rollout-workers", type=int, default=16)
    parser.add_argument("--rollout-worker-threads", type=int, default=2)
    parser.add_argument("--evaluation-num-workers", type=int, default=16)
    parser.add_argument("--evaluation-worker-threads", type=int, default=2)
    parser.add_argument("--cpus-per-task", type=int, default=32)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_prefix(value: str) -> str:
    path = PurePosixPath(str(value).strip())
    if not str(path) or path.is_absolute() or ".." in path.parts:
        raise ValueError("run-prefix must be a non-empty relative path without '..'.")
    return str(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite method manifest: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    output = args.output.resolve()
    if not args.run_root.is_absolute():
        raise ValueError("run-root must be an absolute path.")
    run_root = args.run_root.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source study manifest does not exist: {source}")
    with source.open(encoding="utf-8") as handle:
        source_payload = json.load(handle)
    rows = [row for row in source_payload.get("trials", ()) if row.get("method") == args.method]
    rows.sort(key=lambda row: int(row.get("index", 0)))
    if not rows:
        raise RuntimeError(f"Source manifest contains no {args.method} trials: {source}")
    canary_count = min(len(rows), max(1, int(args.canary_count)))
    if args.selection == "canary":
        rows = rows[:canary_count]
    elif args.selection == "production":
        rows = rows[canary_count:]
    if not rows:
        raise RuntimeError(f"selection={args.selection!r} produced no {args.method} trials.")

    geometry = {
        "num_rollout_workers": max(1, int(args.num_rollout_workers)),
        "rollout_worker_threads": max(1, int(args.rollout_worker_threads)),
        "evaluation_num_workers": max(1, int(args.evaluation_num_workers)),
        "evaluation_worker_threads": max(1, int(args.evaluation_worker_threads)),
        "cpus_per_task": max(1, int(args.cpus_per_task)),
    }
    allocated_cpus = geometry["cpus_per_task"]
    if geometry["num_rollout_workers"] * geometry["rollout_worker_threads"] > allocated_cpus:
        raise ValueError("rollout worker geometry exceeds the Slurm CPU allocation.")
    if geometry["evaluation_num_workers"] * geometry["evaluation_worker_threads"] > allocated_cpus:
        raise ValueError("evaluation worker geometry exceeds the Slurm CPU allocation.")

    prefix = _validate_prefix(args.run_prefix)
    prepared: list[dict[str, Any]] = []
    target_paths: set[Path] = set()
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        trial_id = str(row.get("trial_id") or f"{args.method}_{index:03d}")
        run_name = str(PurePosixPath(prefix) / args.method / trial_id)
        target = (run_root / run_name).resolve()
        if target in target_paths:
            raise RuntimeError(f"Duplicate output directory in method manifest: {target}")
        if target.exists():
            raise FileExistsError(f"Refusing to reuse existing trial output directory: {target}")
        target_paths.add(target)
        arguments = dict(row.get("arguments") or {})
        arguments.update({key: value for key, value in geometry.items() if key != "cpus_per_task"})
        arguments.update(
            {
                "run_name": run_name,
                "run_root": str(run_root),
                "rollout_cache_envs": True,
                "rollout_max_cached_envs_per_worker": 2,
                "rollout_profile": True,
                "road_query_mode": "spatial",
                "collision_check_mode": "broadphase",
                "record_replay_diagnostics": False,
            }
        )
        if args.study_domain:
            arguments["study_domain"] = str(args.study_domain).strip().lower()
        row.update(index=index, method=args.method, run_name=run_name, arguments=arguments)
        prepared.append(row)

    source_code: dict[str, Any] = {}
    source_repo_arg = getattr(args, "source_repo", None)
    if source_repo_arg is not None:
        source_repo = Path(source_repo_arg).resolve()
        source_files = (
            "scripts_gail/train_simple_ps_gail.py",
            "scripts_gail/train_simple_airl.py",
            "scripts_gail/ps_gail/training/rollouts.py",
            "scripts_gail/ps_gail/checkpoints.py",
            "scripts_gail/ps_gail/envs.py",
            "scripts_gail/run_gail_airl_study_trial.py",
            "highway_env/envs/ngsim_env.py",
            "highway_env/road/road.py",
        )
        source_code = {
            "repo": str(source_repo),
            "revision": str(getattr(args, "source_revision", "") or ""),
            "files_sha256": {
                relative: _sha256(source_repo / relative)
                for relative in source_files
            },
        }
    payload = {
        "schema_version": 2,
        "method": args.method,
        "selection": args.selection,
        "trial_count": len(prepared),
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256(source),
        "source_code": source_code,
        "run_root": str(run_root),
        "run_prefix": prefix,
        "resource_geometry": geometry,
        "trials": prepared,
    }
    _atomic_json(output, payload)
    return {
        "manifest": str(output),
        "method": args.method,
        "selection": args.selection,
        "trial_count": len(prepared),
        "run_root": str(run_root),
        "run_prefix": prefix,
    }


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
