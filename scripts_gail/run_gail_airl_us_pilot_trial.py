#!/usr/bin/env python3
"""Verify and execute one depth from the locked US GAIL/AIRL pilot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys

from scripts_gail.ps_gail.pilot import (
    load_manifest,
    select_trial,
    trial_argv,
    verify_manifest_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("gail", "airl"))
    parser.add_argument("--depth", required=True, type=int, choices=(2, 3))
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--include-large-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    verify_manifest_inputs(
        manifest,
        repo=args.repo,
        include_large_data=bool(args.include_large_data),
    )
    trial = select_trial(manifest, method=args.method, depth=args.depth)
    run_root = Path(str(trial["arguments"]["run_root"])).resolve()
    run_dir = (run_root / str(trial["arguments"]["run_name"])).resolve()
    try:
        run_dir.relative_to(run_root)
    except ValueError as exc:
        raise RuntimeError(f"Pilot output escapes run root: {run_dir}") from exc
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {run_dir}")
    argv = trial_argv(trial, python=args.python)
    print(shlex.join(argv), flush=True)
    if args.verify_only:
        return
    os.execv(str(args.python), argv)


if __name__ == "__main__":
    main()
