#!/usr/bin/env python3
"""Run all four locked pilot shapes as bounded local-GPU integration smokes."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from scripts_gail.ps_gail.checkpoints import policy_architecture_contract
from scripts_gail.ps_gail.pilot import (
    load_manifest,
    source_fingerprint,
    trial_argv,
    verify_manifest_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=20,
        help="Fixed simulator steps per local rollout episode (production remains 200).",
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=2,
        help=(
            "Controlled vehicles requested in the local rollout. The simulator "
            "clips this to all valid vehicles when a sampled scene contains fewer."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _smoke_trial(
    trial: dict[str, Any],
    *,
    output_root: Path,
    rounds: int,
    episode_steps: int,
    controlled_vehicles: int = 2,
) -> dict[str, Any]:
    trial = json.loads(json.dumps(trial))
    args = trial["arguments"]
    policy_frequency = max(1, int(args.get("policy_frequency", 10)))
    score_horizon_seconds = max(1, int(episode_steps) // policy_frequency)
    evaluation_horizons = [
        seconds
        for seconds in (1, 5, 10, 20)
        if seconds <= score_horizon_seconds
    ]
    if score_horizon_seconds not in evaluation_horizons:
        evaluation_horizons.append(score_horizon_seconds)
    target_agent_steps = max(
        64,
        int(episode_steps) * int(controlled_vehicles),
    )
    args.update(
        {
            "run_root": str(output_root.resolve()),
            "run_name": str(trial["trial_id"]),
            "total_rounds": rounds,
            "max_expert_samples": 1024,
            "initial_controlled_vehicles": float(controlled_vehicles),
            "final_controlled_vehicles": float(controlled_vehicles),
            "controlled_vehicle_curriculum_rounds": rounds,
            "controlled_vehicle_schedule": (
                f"1:{rounds}:{controlled_vehicles}:{controlled_vehicles}"
            ),
            "rollout_steps": int(episode_steps),
            "rollout_target_agent_steps": target_agent_steps,
            "rollout_target_agent_steps_schedule": (
                f"1:{rounds}:{target_agent_steps}:{target_agent_steps}"
            ),
            "rollout_max_episode_steps": int(episode_steps),
            "max_episode_steps": int(episode_steps),
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
            "validation_vehicle_mode": "single",
            "validation_stress_every": 0,
            "validation_stress_episodes": 0,
            "test_episodes": 1,
            "test_vehicle_mode": "single",
            "evaluation_horizons_seconds": ",".join(
                str(value) for value in sorted(evaluation_horizons)
            ),
            "validation_score_horizon_seconds": score_horizon_seconds,
            "validation_min_horizon_coverage": 1.0,
            "health_learning_gate_round": 0,
            "abort_on_health_failure": False,
            "checkpoint_every": 1,
            "wandb_mode": "disabled",
            "wandb_group": "local_gpu_smoke",
        }
    )
    if int(args.get("full_load_selection_start_round", 0)) > 0:
        args["full_load_selection_start_round"] = 1
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


def _last_training_metrics(path: Path, *, expected_step: int) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("step", -1)) == int(expected_step):
                selected = dict(row)
    if not selected:
        raise RuntimeError(
            f"No metrics row for training step {expected_step}: {path}"
        )
    return selected


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    if args.episode_steps < 10:
        raise ValueError("--episode-steps must be at least 10")
    if args.controlled_vehicles < 1:
        raise ValueError("--controlled-vehicles must be positive")
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
        trial = _smoke_trial(
            dict(raw_trial),
            output_root=output_root,
            rounds=args.rounds,
            episode_steps=args.episode_steps,
            controlled_vehicles=args.controlled_vehicles,
        )
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
        runtime_metrics: dict[str, Any] = {}
        if not summary_path.is_file():
            failures.append("missing_evaluation_summary")
            summary: dict[str, Any] = {}
        else:
            summary = _load_json(summary_path)
            if (
                int(
                    trial["arguments"].get(
                        "full_load_selection_start_round", 0
                    )
                )
                > 0
                and summary.get("selected_checkpoint")
                != "best_full_load.pt"
            ):
                failures.append("full_load_checkpoint_not_selected")
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
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            failures.append("missing_metrics_jsonl")
        else:
            try:
                runtime_metrics = _last_training_metrics(
                    metrics_path,
                    expected_step=args.rounds,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(f"invalid_training_metrics:{exc}")
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
        score_horizon_seconds = int(
            trial["arguments"]["validation_score_horizon_seconds"]
        )
        for key in (
            "selected_validation/cost",
            f"selected_validation/horizon_coverage_{score_horizon_seconds}s",
        ):
            value = selected.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"nonfinite_or_missing:{key}")
        for key in ("test/cost", "test/acceleration_action_std", "test/steering_action_std"):
            value = test.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"nonfinite_or_missing:{key}")
        expected_runtime = {
            "train/discriminator_loss_type_wgan_gp": 1.0,
            "train/rollout_fixed_horizon": 1.0,
            "rollout/mean_episode_length": float(args.episode_steps),
            "rollout/min_episode_length": float(args.episode_steps),
            "rollout/max_episode_length": float(args.episode_steps),
            "rollout/terminated": 0.0,
            "health/target_kl_warning": 0.0,
            "health/target_kl_consecutive_violations": 0.0,
        }
        for key, expected in expected_runtime.items():
            value = runtime_metrics.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != expected
            ):
                failures.append(
                    f"runtime_contract_mismatch:{key}={value!r}!={expected!r}"
                )
        for key in (
            "policy/ppo_optimizer_steps",
            "policy/ppo_minibatch_early_stopped_kl",
        ):
            value = runtime_metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"nonfinite_or_missing:{key}")
        achieved_controlled_vehicles = runtime_metrics.get(
            "rollout/mean_controlled_vehicles"
        )
        if (
            not isinstance(achieved_controlled_vehicles, (int, float))
            or not math.isfinite(float(achieved_controlled_vehicles))
            or float(achieved_controlled_vehicles) <= 0.0
            or float(achieved_controlled_vehicles)
            > float(args.controlled_vehicles)
        ):
            failures.append(
                "invalid_achieved_controlled_vehicle_count:"
                f"{achieved_controlled_vehicles!r}"
            )
        for key in (
            "discriminator/critic_gap",
            "discriminator/gradient_penalty",
            "discriminator/wgan_loss",
            "rollout/raw_gail_reward_std",
            "rollout/normalized_gail_reward_std",
            "rollout/reward_std",
            "policy/approx_kl",
            "policy/post_update_approx_kl",
            "policy/action_std_param_mean",
        ):
            value = runtime_metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"nonfinite_or_missing:{key}")
        if float(runtime_metrics.get("discriminator/gradient_penalty", 0.0)) <= 0.0:
            failures.append("nonpositive:discriminator/gradient_penalty")
        if float(runtime_metrics.get("rollout/raw_gail_reward_std", 0.0)) <= 0.0:
            failures.append("nonpositive:rollout/raw_gail_reward_std")
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
                "stability_metrics": {
                    key: runtime_metrics.get(key)
                    for key in (
                        "rollout/episodes",
                        "rollout/env_steps",
                        "rollout/agent_steps",
                        "rollout/mean_episode_length",
                        "rollout/controlled_vehicle_fraction",
                        "rollout/mean_controlled_vehicles",
                        "rollout/mean_road_vehicles",
                        "rollout/terminated",
                        "discriminator/critic_gap",
                        "discriminator/gradient_penalty",
                        "rollout/raw_gail_reward_std",
                        "rollout/normalized_gail_reward_std",
                        "policy/approx_kl",
                        "policy/post_update_approx_kl",
                        "health/target_kl_warning",
                        "health/target_kl_consecutive_violations",
                        "policy/action_std_param_mean",
                    )
                },
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
        "episode_steps": int(args.episode_steps),
        "controlled_vehicles": int(args.controlled_vehicles),
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
