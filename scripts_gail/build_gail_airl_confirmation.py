#!/usr/bin/env python3
"""Generate five-seed full-curriculum confirmation runs for study finalists."""

from __future__ import annotations

import argparse
import json

from scripts_gail.ps_gail.study import StudyTrial, write_study_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def _full_curriculum(arguments: dict, *, seed: int, run_name: str) -> dict:
    result = dict(arguments)
    result.update(
        {
            "seed": seed,
            "run_name": run_name,
            "total_rounds": 800,
            "initial_controlled_vehicles": 10,
            "final_controlled_vehicles": 100,
            "controlled_vehicle_curriculum_rounds": 800,
            "controlled_vehicle_schedule": (
                "1:120:10:10;121:240:20:20;241:360:30:30;"
                "361:480:40:40;481:600:50:50;601:650:60:60;"
                "651:700:70:70;701:750:85:85;751:800:100:100"
            ),
            "initial_rollout_target_agent_steps": 10_000,
            "final_rollout_target_agent_steps": 40_000,
            "rollout_target_agent_steps_curriculum_rounds": 800,
            "rollout_target_agent_steps_schedule": (
                "1:500:10000:10000;501:600:10000:20000;"
                "601:700:20000:30000;701:800:30000:40000"
            ),
            "validation_every": 20,
            "validation_episodes": 10,
            "validation_stress_every": 100,
            "validation_stress_episodes": 10,
            "test_episodes": 30,
            "checkpoint_every": 20,
        }
    )
    return result


def _baseline_arguments(method: str, finalist: dict) -> dict:
    args = dict(finalist["arguments"])
    args.update(
        {
            "algorithm_variant": f"{method}_wgan_gp",
            "initial_policy_checkpoint": "",
            "policy_bc_regularization_coef": 0.0,
            "collision_mode_schedule": "1:800:full",
            "wgan_gp_lambda": 2.0,
            "discriminator_spectral_norm": True,
            "wgan_reward_center": False,
            "wgan_reward_clip": 0.0,
            "normalize_gail_reward": True,
            "allow_wgan_reward_normalization": True,
            "disc_learning_rate": 4.0e-4,
            "disc_updates_per_round": 2,
            "ppo_epochs": 6,
            "clip_range": 0.20,
            "target_kl": 0.0,
            "learning_rate": 4.0e-4 if method == "gail" else 2.0e-4,
            "entropy_coef": 0.0015 if method == "gail" else 0.001,
        }
    )
    if method == "airl":
        args["airl_policy_reward_mode"] = "shaped"
    return args


def main() -> None:
    args = parse_args()
    with open(args.screening_report, encoding="utf-8") as handle:
        report = json.load(handle)
    winners = report.get("winners_by_method") or {}
    trials = []
    for method in ("gail", "airl"):
        finalist = winners.get(method)
        if not finalist:
            raise RuntimeError(f"No safety-eligible {method} finalist in screening report.")
        for arm in ("winner", "current_baseline"):
            source_args = (
                dict(finalist["arguments"])
                if arm == "winner"
                else _baseline_arguments(method, finalist)
            )
            for seed in range(10, 15):
                index = len(trials)
                trial_id = f"{method}_confirm_{arm}_s{seed}"
                run_name = f"gail_airl_confirmation/{trial_id}"
                trial_args = _full_curriculum(
                    source_args,
                    seed=seed,
                    run_name=run_name,
                )
                trials.append(
                    StudyTrial(
                        index=index,
                        trial_id=trial_id,
                        method=method,
                        phase=f"confirmation_{arm}",
                        seed=seed,
                        variant=str(trial_args["algorithm_variant"]),
                        initialization=(
                            "bc" if trial_args.get("initial_policy_checkpoint") else "cold"
                        ),
                        collision_training=(
                            "full"
                            if trial_args["collision_mode_schedule"] == "1:800:full"
                            else "curriculum"
                        ),
                        module=(
                            "scripts_gail.train_simple_ps_gail"
                            if method == "gail"
                            else "scripts_gail.train_simple_airl"
                        ),
                        run_name=run_name,
                        arguments=trial_args,
                    )
                )
    paths = write_study_files(args.output_dir, trials, python=args.python)
    print(json.dumps({"trial_count": len(trials), **paths}, indent=2))


if __name__ == "__main__":
    main()
