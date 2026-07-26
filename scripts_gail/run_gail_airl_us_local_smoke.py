#!/usr/bin/env python3
"""Run all four locked pilot shapes as bounded local-GPU integration smokes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import torch

from scripts_gail.ps_gail.pilot import (
    load_manifest,
    source_fingerprint,
    trial_argv,
    verify_manifest_inputs,
)
from scripts_gail.ps_gail.checkpoints import policy_architecture_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _smoke_trial(
    trial: dict[str, Any], *, output_root: Path, rounds: int
) -> dict[str, Any]:
    trial = json.loads(json.dumps(trial))
    args = trial["arguments"]
    args.update(
        {
            "run_root": str(output_root.resolve()),
            "run_name": str(trial["trial_id"]),
            "total_rounds": rounds,
            "max_expert_samples": 1024,
            "controlled_vehicle_schedule": f"1:{rounds}:2:2",
            "initial_controlled_vehicles": 2.0,
            "final_controlled_vehicles": 2.0,
            "controlled_vehicle_curriculum_rounds": rounds,
            "rollout_target_agent_steps": 64,
            "rollout_target_agent_steps_schedule": f"1:{rounds}:64:64",
            "rollout_max_episode_steps": 20,
            "max_episode_steps": 20,
            "num_rollout_workers": 1,
            "rollout_worker_threads": 1,
            "evaluation_num_workers": 0,
            "evaluation_worker_threads": 1,
            "batch_size": 128,
            "disc_batch_size": 128,
            "transformer_recurrent_sequences_per_batch": 4,
            "transformer_recurrent_micro_batch_sequences": 2,
            "warmup_rounds": 1,
            "collision_mode_schedule": f"1:{rounds}:full",
            "validation_every": 1,
            "validation_episodes": 1,
            "validation_stress_every": 0,
            "validation_stress_episodes": 0,
            "test_episodes": 1,
            "evaluation_horizons_seconds": "1",
            "validation_score_horizon_seconds": 1,
            "validation_min_horizon_coverage": 1.0,
            "health_learning_gate_round": 0,
            "abort_on_health_failure": False,
            "checkpoint_every": 1,
            "wandb_mode": "disabled",
            "wandb_group": "local_gpu_smoke",
        }
    )
    if trial["method"] == "airl":
        trial["trainer_arguments"] = {
            "reward_batch_size": 128,
            "airl_log_prob_batch_size": 64,
        }
    return trial


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return dict(value)


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    manifest = load_manifest(args.manifest)
    verify_manifest_inputs(manifest, repo=args.repo, include_large_data=True)
    if not args.dry_run:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("Local pilot smoke requires a CUDA GPU.")
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "dry-run"
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse local smoke root: {output_root}")
    output_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for raw_trial in manifest["trials"]:
        trial = _smoke_trial(dict(raw_trial), output_root=output_root, rounds=args.rounds)
        argv = trial_argv(trial, python=args.python)
        log_path = output_root / f"{trial['trial_id']}.log"
        print(shlex.join(argv), flush=True)
        if args.dry_run:
            results.append({"trial_id": trial["trial_id"], "passed": True, "dry_run": True})
            continue
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                argv,
                cwd=args.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        run_dir = output_root / str(trial["trial_id"])
        failures: list[str] = []
        if completed.returncode != 0:
            failures.append(f"exit_code_{completed.returncode}")
        summary_path = run_dir / "evaluation_summary.json"
        final_path = run_dir / "final.pt"
        if not summary_path.is_file():
            failures.append("missing_evaluation_summary")
            summary: dict[str, Any] = {}
        else:
            summary = _load_json(summary_path)
        if not final_path.is_file() or not final_path.with_name("final.pt.sha256").is_file():
            failures.append("missing_final_checkpoint_or_sha")
        elif final_path.is_file():
            try:
                payload = torch.load(
                    final_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(final_path, map_location="cpu")
            expected_architecture = policy_architecture_contract(
                trial["arguments"]
            )
            actual_architecture = policy_architecture_contract(payload)
            if actual_architecture != expected_architecture:
                failures.append("final_checkpoint_architecture_drift")
        selected = dict(summary.get("selected_validation") or {})
        test = dict(summary.get("test") or {})
        policy_delta = summary.get("policy_relative_l2_delta")
        if (
            not isinstance(policy_delta, (int, float))
            or not math.isfinite(float(policy_delta))
            or float(policy_delta) <= 0.0
        ):
            failures.append(
                "nonpositive_or_missing:policy_relative_l2_delta"
            )
        for key in (
            "selected_validation/cost",
            "selected_validation/horizon_coverage_1s",
        ):
            value = selected.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"nonfinite_or_missing:{key}")
        for key in ("test/cost", "test/acceleration_action_std", "test/steering_action_std"):
            value = test.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"nonfinite_or_missing:{key}")
        results.append(
            {
                "trial_id": trial["trial_id"],
                "method": trial["method"],
                "depth": int(trial["depth"]),
                "seed": int(trial["seed"]),
                "exit_code": int(completed.returncode),
                "run_dir": str(run_dir),
                "log": str(log_path),
                "passed": not failures,
                "failures": failures,
            }
        )
        if failures:
            break
    expected_trials = len(manifest["trials"])
    passed = (
        len(results) == expected_trials
        and all(row["passed"] for row in results)
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "dry_run": bool(args.dry_run),
        "gpu_name": gpu_name,
        "rounds": int(args.rounds),
        "expected_trials": int(expected_trials),
        "source_fingerprint": source_fingerprint(manifest),
        "results": results,
    }
    report_path = output_root / "smoke_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
