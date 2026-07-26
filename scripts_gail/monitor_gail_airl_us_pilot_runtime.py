#!/usr/bin/env python3
"""Enforce the round-20 conservative walltime projection for one pilot method."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from scripts_gail.ps_gail.pilot import load_manifest, select_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("gail", "airl"))
    parser.add_argument(
        "--depth",
        type=int,
        choices=(2, 3),
        help="Monitor one depth when depth trials are scheduled sequentially.",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-hours", type=float, default=36.0)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _finite(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def project_run(
    rows: list[dict[str, object]],
    runtime: dict[str, object],
    *,
    evaluation_parallelism: int = 1,
    measurement_start_round: int = 0,
) -> dict[str, float]:
    measurement_start_round = max(0, int(measurement_start_round))
    measurement_end_round = measurement_start_round + int(
        runtime.get("gate_round", 20)
    )
    round_seconds = [
        value
        for row in rows
        if (
            measurement_start_round + 2
            <= int(row.get("step", -1))
            <= measurement_end_round - 1
        )
        for value in [_finite(row, "perf/round_seconds")]
        if value is not None and value > 0.0
    ]
    if len(round_seconds) < 10:
        raise RuntimeError(
            f"Runtime gate requires at least 10 non-validation rounds, found {len(round_seconds)}."
        )
    p95_round_seconds = float(np.percentile(np.asarray(round_seconds), 95.0))
    eval_seconds = 0.0
    eval_vehicle_trajectories = 0.0
    for row in rows:
        step = int(row.get("step", -1))
        if step <= measurement_start_round or step > measurement_end_round:
            continue
        vehicle_count = _finite(row, "validation/vehicle_episodes")
        timing = sum(
            value or 0.0
            for value in (
                _finite(row, "validation/eval_env_reset_seconds"),
                _finite(row, "validation/eval_policy_forward_seconds"),
                _finite(row, "validation/eval_env_step_seconds"),
            )
        )
        if vehicle_count is not None and vehicle_count > 0.0 and timing > 0.0:
            eval_seconds += timing
            eval_vehicle_trajectories += vehicle_count
    if eval_vehicle_trajectories <= 0.0:
        raise RuntimeError("Runtime gate has no measured validation timing per vehicle trajectory.")
    eval_seconds_per_vehicle = eval_seconds / eval_vehicle_trajectories
    safety_factor = float(runtime["safety_factor"])
    evaluation_parallelism = max(1, int(evaluation_parallelism))
    # The measured timing fields are summed worker timings.  The original
    # projection charged every planned vehicle trajectory serially even though
    # final evaluation uses an explicit process pool.  Convert that aggregate
    # work estimate back to elapsed wall time using the locked worker count.
    projected_evaluation_wall_seconds = (
        float(runtime["planned_evaluation_vehicle_trajectories"])
        * eval_seconds_per_vehicle
        / float(evaluation_parallelism)
    )
    projected_seconds = safety_factor * (
        float(runtime["conservative_training_rounds"]) * p95_round_seconds
        + projected_evaluation_wall_seconds
    )
    return {
        "round_samples": float(len(round_seconds)),
        "measurement_start_round": float(measurement_start_round),
        "measurement_end_round": float(measurement_end_round),
        "p95_nonvalidation_round_seconds": p95_round_seconds,
        "measured_eval_seconds": eval_seconds,
        "measured_eval_vehicle_trajectories": eval_vehicle_trajectories,
        "eval_seconds_per_vehicle_trajectory": eval_seconds_per_vehicle,
        "evaluation_parallelism": float(evaluation_parallelism),
        "projected_evaluation_wall_seconds": projected_evaluation_wall_seconds,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600.0,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    runtime = dict(manifest["runtime_projection"])
    depths = (int(args.depth),) if args.depth is not None else (2, 3)
    trials = [select_trial(manifest, method=args.method, depth=depth) for depth in depths]
    run_dirs = [
        Path(str(trial["arguments"]["run_root"])) / str(trial["arguments"]["run_name"])
        for trial in trials
    ]
    output_name = (
        f"runtime_projection_depth{int(args.depth)}.json"
        if args.depth is not None
        else "runtime_projection.json"
    )
    output = Path(str(trials[0]["arguments"]["run_root"])) / args.method / output_name
    deadline = time.monotonic() + max(1.0, float(args.timeout_hours)) * 3600.0
    while time.monotonic() < deadline:
        for run_dir in run_dirs:
            if (run_dir / "training_failure.json").is_file():
                raise RuntimeError(f"Training health failure before runtime gate: {run_dir}")
        rows_by_run = [_read_rows(run_dir / "metrics.jsonl") for run_dir in run_dirs]
        measurement_starts = [
            max(0, int(trial["arguments"].get("expected_resume_round", 0)))
            for trial in trials
        ]
        if all(
            any(
                int(row.get("step", -1))
                >= start + int(runtime["gate_round"])
                for row in rows
            )
            for rows, start in zip(rows_by_run, measurement_starts)
        ):
            projections = {
                str(trial["trial_id"]): project_run(
                    rows,
                    runtime,
                    evaluation_parallelism=int(
                        trial["arguments"].get("evaluation_num_workers", 1)
                    ),
                    measurement_start_round=start,
                )
                for trial, rows, start in zip(
                    trials,
                    rows_by_run,
                    measurement_starts,
                )
            }
            maximum = float(runtime["maximum_projected_hours"])
            passed = all(item["projected_hours"] <= maximum for item in projections.values())
            payload: dict[str, object] = {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "method": args.method,
                "depths": list(depths),
                "gate_round": int(runtime["gate_round"]),
                "maximum_projected_hours": maximum,
                "passed": passed,
                "projections": projections,
                "formula": (
                    "safety_factor * (conservative_training_rounds * "
                    "p95_nonvalidation_round_seconds + "
                    "planned_evaluation_vehicle_trajectories * "
                    "eval_seconds_per_vehicle_trajectory / evaluation_num_workers)"
                ),
            }
            _atomic_json(output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
            if not passed:
                raise SystemExit(42)
            return
        time.sleep(max(1.0, min(float(args.poll_seconds), 60.0)))
    raise TimeoutError(f"Round-20 runtime evidence did not arrive within {args.timeout_hours}h.")


if __name__ == "__main__":
    main()
