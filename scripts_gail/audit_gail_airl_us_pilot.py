#!/usr/bin/env python3
"""Fail-closed acceptance audit for the four-run US GAIL/AIRL pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import torch

from scripts_gail.ps_gail.checkpoints import verify_checkpoint_sidecar
from scripts_gail.ps_gail.pilot import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("gail", "airl"), default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _finite(mapping: dict[str, Any], key: str, failures: list[str]) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        failures.append(f"nonfinite_or_missing:{key}")
        return float("nan")
    return float(value)


def _checkpoint_round(path: Path) -> int:
    verify_checkpoint_sidecar(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return int(payload.get("round", -1))


def audit_trial(trial: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, Any]:
    args = dict(trial["arguments"])
    run_dir = Path(str(args["run_root"])) / str(args["run_name"])
    failures: list[str] = []
    if (run_dir / "training_failure.json").is_file():
        failures.append("training_failure_present")
    summary_path = run_dir / "evaluation_summary.json"
    if not summary_path.is_file():
        return {
            "trial_id": trial["trial_id"],
            "run_dir": str(run_dir),
            "passed": False,
            "failures": ["missing_evaluation_summary"],
        }
    summary = _load_json(summary_path)
    if int(summary.get("schema_version", -1)) != 2:
        failures.append("evaluation_summary_schema_not_2")
    if summary.get("selected_checkpoint") != "best.pt":
        failures.append("selected_checkpoint_is_not_best")
    best_round = int(summary.get("best_validation_round", -1))
    if best_round < int(acceptance["minimum_best_round"]):
        failures.append("best_round_below_minimum")
    try:
        final_round = _checkpoint_round(run_dir / "final.pt")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        failures.append(f"invalid_final_checkpoint:{exc}")
        final_round = -1
    try:
        _checkpoint_round(run_dir / "best.pt")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        failures.append(f"invalid_best_checkpoint:{exc}")
    if final_round != int(acceptance["required_completed_round"]):
        failures.append("required_round_not_completed")

    initial = dict(summary.get("initial_validation") or {})
    selected_validation = dict(summary.get("selected_validation") or {})
    initializer_test = dict(summary.get("initializer_test") or {})
    selected_test = dict(summary.get("test") or {})
    initial_cost = _finite(initial, "validation/cost", failures)
    selected_cost = _finite(selected_validation, "selected_validation/cost", failures)
    coverage = _finite(
        selected_validation,
        "selected_validation/horizon_coverage_20s",
        failures,
    )
    relative_improvement = (
        (initial_cost - selected_cost) / max(abs(initial_cost), 1.0e-12)
        if math.isfinite(initial_cost) and math.isfinite(selected_cost)
        else float("nan")
    )
    if not math.isfinite(relative_improvement) or relative_improvement < float(
        acceptance["minimum_relative_validation_cost_improvement"]
    ):
        failures.append("validation_improvement_below_minimum")
    if not math.isfinite(coverage) or coverage < float(
        acceptance["minimum_20s_horizon_coverage"]
    ):
        failures.append("selected_validation_horizon_coverage_below_minimum")

    test_cost = _finite(selected_test, "test/cost", failures)
    initializer_test_cost = _finite(initializer_test, "initializer_test/cost", failures)
    if (
        math.isfinite(test_cost)
        and math.isfinite(initializer_test_cost)
        and test_cost > initializer_test_cost + 1.0e-12
    ):
        failures.append("selected_test_cost_exceeds_initializer")
    crash_rate = _finite(selected_test, "test/vehicle_crash_rate", failures)
    offroad_rate = _finite(selected_test, "test/vehicle_offroad_rate", failures)
    if math.isfinite(crash_rate) and crash_rate > float(
        acceptance["maximum_test_vehicle_crash_rate"]
    ):
        failures.append("test_vehicle_crash_rate_above_limit")
    if math.isfinite(offroad_rate) and offroad_rate > float(
        acceptance["maximum_test_vehicle_offroad_rate"]
    ):
        failures.append("test_vehicle_offroad_rate_above_limit")

    action_evidence: dict[str, dict[str, float]] = {}
    for name in ("acceleration", "steering"):
        selected_std = _finite(selected_test, f"test/{name}_action_std", failures)
        initializer_std = _finite(
            initializer_test,
            f"initializer_test/{name}_action_std",
            failures,
        )
        threshold = max(
            float(acceptance["minimum_action_std_absolute"]),
            float(acceptance["minimum_action_std_initializer_fraction"])
            * initializer_std,
        )
        action_evidence[name] = {
            "selected_std": selected_std,
            "initializer_std": initializer_std,
            "minimum": threshold,
        }
        if not math.isfinite(selected_std) or selected_std < threshold:
            failures.append(f"{name}_action_variation_below_minimum")

    policy_delta = summary.get("policy_relative_l2_delta")
    if not isinstance(policy_delta, (int, float)) or not math.isfinite(float(policy_delta)):
        failures.append("nonfinite_or_missing:policy_relative_l2_delta")
        policy_delta = float("nan")
    elif float(policy_delta) <= float(acceptance["minimum_policy_relative_l2_delta"]):
        failures.append("policy_parameter_delta_below_minimum")

    core_test_keys = (
        "test/rmse_position_20s",
        "test/rmse_speed_20s",
        "test/rmse_lane_offset_20s",
        "test/horizon_coverage_20s",
        "test/score",
    )
    for key in core_test_keys:
        _finite(selected_test, key, failures)

    metrics = _rows(run_dir / "metrics.jsonl")
    final_rows = [
        row
        for row in metrics
        if int(row.get("step", -1)) == int(acceptance["required_completed_round"])
    ]
    if not final_rows:
        failures.append("missing_durable_final_round_metrics")
        final_row: dict[str, Any] = {}
    else:
        final_row = final_rows[-1]
    reward_key = (
        "rollout/raw_gail_reward_std"
        if trial["method"] == "gail"
        else "rollout/raw_airl_reward_std"
    )
    reward_std = _finite(final_row, reward_key, failures)
    if math.isfinite(reward_std) and reward_std < float(
        acceptance["minimum_adversarial_reward_std"]
    ):
        failures.append("adversarial_reward_std_below_minimum")
    for key in ("policy/loss", "policy/value_loss", "policy/approx_kl"):
        _finite(final_row, key, failures)

    wandb_path = run_dir / "wandb_run.json"
    if not wandb_path.is_file():
        failures.append("missing_wandb_identity")
        wandb_identity: dict[str, Any] = {}
    else:
        wandb_identity = _load_json(wandb_path)
    if bool(acceptance.get("require_wandb_online", False)):
        if wandb_identity.get("effective_mode") != "online":
            failures.append("wandb_not_online")
        if wandb_identity.get("project") != acceptance.get("wandb_project"):
            failures.append("wandb_project_mismatch")
        if not wandb_identity.get("run_id"):
            failures.append("wandb_run_id_missing")

    runtime_path = run_dir.parent / "runtime_projection.json"
    if not runtime_path.is_file() or _load_json(runtime_path).get("passed") is not True:
        failures.append("runtime_projection_not_passed")

    return {
        "trial_id": trial["trial_id"],
        "method": trial["method"],
        "depth": int(trial["depth"]),
        "seed": int(trial["seed"]),
        "run_dir": str(run_dir),
        "passed": not failures,
        "failures": sorted(set(failures)),
        "evidence": {
            "completed_round": final_round,
            "best_round": best_round,
            "initial_validation_cost": initial_cost,
            "selected_validation_cost": selected_cost,
            "relative_validation_cost_improvement": relative_improvement,
            "selected_validation_20s_coverage": coverage,
            "selected_test_cost": test_cost,
            "initializer_test_cost": initializer_test_cost,
            "test_vehicle_crash_rate": crash_rate,
            "test_vehicle_offroad_rate": offroad_rate,
            "action_variation": action_evidence,
            "policy_relative_l2_delta": policy_delta,
            "final_adversarial_reward_std": reward_std,
            "wandb": wandb_identity,
        },
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    trials = [
        dict(row)
        for row in manifest["trials"]
        if not args.method or row["method"] == args.method
    ]
    results = [audit_trial(row, dict(manifest["acceptance"])) for row in trials]
    required_count = 2 if args.method else 4
    passed = len(results) == required_count and all(row["passed"] for row in results)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": args.method or "all",
        "passed": passed,
        "automatic_expansion_authorized": False,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
