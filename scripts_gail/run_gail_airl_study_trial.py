#!/usr/bin/env python3
"""Execute one indexed trial from a generated GAIL/AIRL study manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from scripts_gail.ps_gail.study import StudyTrial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_trial(path: str, index: int) -> StudyTrial:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("trials") or []
    if index < 0 or index >= len(rows):
        raise IndexError(f"trial-index {index} outside [0, {len(rows) - 1}].")
    return StudyTrial(**rows[index])


def main() -> None:
    args = parse_args()
    trial = load_trial(args.manifest, args.trial_index)
    bc_path = str(trial.arguments.get("initial_policy_checkpoint", "") or "")
    if trial.initialization == "bc" and not os.path.isfile(bc_path):
        raise FileNotFoundError(f"Trial {trial.trial_id} requires BC checkpoint: {bc_path}")
    command = trial.command(python=args.python)
    print(command, flush=True)
    if not args.dry_run:
        subprocess.run(trial.argv(python=args.python), check=True)


if __name__ == "__main__":
    main()
