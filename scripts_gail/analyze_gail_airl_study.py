#!/usr/bin/env python3
"""Apply finite/safety gates and rank completed GAIL/AIRL study trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from scripts_gail.ps_gail.study import assess_evaluation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--expert-crash-rate", type=float, required=True)
    parser.add_argument("--expert-offroad-rate", type=float, required=True)
    parser.add_argument("--position-error-span", type=float, required=True)
    parser.add_argument("--speed-error-span", type=float, required=True)
    parser.add_argument("--lane-error-span", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        trials = json.load(handle).get("trials") or []
    results = []
    for trial in trials:
        trainer_dir = "simple_ps_gail" if trial["method"] == "gail" else "airl"
        summary_path = os.path.join(
            args.log_root,
            trainer_dir,
            trial["run_name"],
            "evaluation_summary.json",
        )
        if not os.path.isfile(summary_path):
            failure_path = os.path.join(os.path.dirname(summary_path), "training_failure.json")
            if os.path.isfile(failure_path):
                with open(failure_path, encoding="utf-8") as handle:
                    failure = json.load(handle)
                results.append(
                    {
                        **trial,
                        "status": "training_failure",
                        "summary_path": summary_path,
                        "failure": failure,
                    }
                )
            else:
                results.append({**trial, "status": "not_run", "summary_path": summary_path})
            continue
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        assessment = assess_evaluation_summary(
            summary,
            expert_crash_rate=args.expert_crash_rate,
            expert_offroad_rate=args.expert_offroad_rate,
            position_error_span=args.position_error_span,
            speed_error_span=args.speed_error_span,
            lane_error_span=args.lane_error_span,
        )
        config_arguments = {
            key: value
            for key, value in trial["arguments"].items()
            if key not in {"seed", "run_name"}
        }
        configuration_id = hashlib.sha256(
            json.dumps(config_arguments, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        results.append(
            {
                **trial,
                **assessment,
                "status": "eligible" if assessment["eligible"] else "rejected",
                "summary_path": summary_path,
                "best_validation_score": summary.get("best_validation_score"),
                "configuration_id": configuration_id,
            }
        )
    eligible = [row for row in results if row.get("eligible")]
    eligible.sort(
        key=lambda row: (
            float(row.get("normalized_trajectory_error", float("inf"))),
            float(row.get("vehicle_crash_rate", float("inf"))),
        )
    )
    groups = []
    for configuration_id in sorted(
        {row["configuration_id"] for row in eligible if row.get("configuration_id")}
    ):
        rows = [row for row in eligible if row.get("configuration_id") == configuration_id]
        scores = [float(row["normalized_trajectory_error"]) for row in rows]
        required_seeds = {
            "factorial": 3,
            "objective": 2,
            "hpo": 1,
            "confirmation_winner": 5,
            "confirmation_current_baseline": 5,
        }.get(rows[0]["phase"], 1)
        groups.append(
            {
                "configuration_id": configuration_id,
                "method": rows[0]["method"],
                "phase": rows[0]["phase"],
                "variant": rows[0]["variant"],
                "initialization": rows[0]["initialization"],
                "collision_training": rows[0]["collision_training"],
                "eligible_seeds": len(rows),
                "required_eligible_seeds": required_seeds,
                "seed_gate_passed": len(rows) >= required_seeds,
                "mean_normalized_trajectory_error": sum(scores) / len(scores),
                "trial_ids": [row["trial_id"] for row in rows],
                "representative": rows[0],
            }
        )
    groups.sort(key=lambda row: row["mean_normalized_trajectory_error"])
    winners_by_method = {}
    for method in ("gail", "airl"):
        candidates = [
            row
            for row in groups
            if row["method"] == method and row["seed_gate_passed"]
        ]
        winners_by_method[method] = (
            candidates[0]["representative"] if candidates else None
        )
    payload = {
        "schema_version": 1,
        "completed": sum(row["status"] != "not_run" for row in results),
        "eligible": len(eligible),
        "winner": eligible[0] if eligible else None,
        "winners_by_method": winners_by_method,
        "configuration_groups": groups,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {key: payload[key] for key in ("completed", "eligible", "winners_by_method")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
