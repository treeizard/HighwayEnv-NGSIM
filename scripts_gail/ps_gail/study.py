"""Deterministic GAIL/AIRL correctness and training-study design."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shlex
from typing import Any


@dataclass(frozen=True)
class StudyTrial:
    index: int
    trial_id: str
    method: str
    phase: str
    seed: int
    variant: str
    initialization: str
    collision_training: str
    module: str
    run_name: str
    arguments: dict[str, Any]

    def argv(self, python: str = "python") -> list[str]:
        parts = [str(python), "-m", self.module]
        for name, value in self.arguments.items():
            flag = "--" + str(name).replace("_", "-")
            if isinstance(value, bool):
                parts.append(flag if value else "--no-" + str(name).replace("_", "-"))
            elif value is not None and value != "":
                parts.extend([flag, str(value)])
        return parts

    def command(self, python: str = "python") -> str:
        return " ".join(shlex.quote(part) for part in self.argv(python=python))


def _collision_schedule(mode: str, rounds: int) -> str:
    if mode in {"full", "soft"}:
        return f"1:{int(rounds)}:{mode}"
    first = max(1, int(rounds) // 3)
    second = max(first + 1, 2 * int(rounds) // 3)
    return f"1:{first}:soft;{first + 1}:{second}:mixed;{second + 1}:{int(rounds)}:full"


def _base_arguments(
    *,
    method: str,
    variant: str,
    seed: int,
    run_name: str,
    collision_training: str,
    initialization: str,
    expert_data: str,
    episode_root: str,
    bc_checkpoint: str,
    bc_policy_config: dict[str, Any] | None,
    rounds: int,
) -> dict[str, Any]:
    policy_lr = 3.0e-5 if method == "gail" else 1.0e-5
    entropy = 0.002 if method == "gail" else 0.003
    args: dict[str, Any] = {
        "algorithm_variant": variant,
        "run_name": run_name,
        "expert_data": expert_data,
        "episode_root": episode_root,
        "prebuilt_split": "train",
        "validation_prebuilt_split": "val",
        "test_prebuilt_split": "test",
        "seed": seed,
        "action_mode": "continuous",
        "policy_model": "recurrent_transformer",
        "total_rounds": rounds,
        "controlled_vehicle_curriculum": True,
        "initial_controlled_vehicles": 10,
        "final_controlled_vehicles": 10,
        "controlled_vehicle_curriculum_rounds": rounds,
        "rollout_target_agent_steps": 10_000,
        "initial_rollout_target_agent_steps": 10_000,
        "final_rollout_target_agent_steps": 10_000,
        "rollout_target_agent_steps_curriculum_rounds": rounds,
        "learning_rate": policy_lr,
        "disc_learning_rate": 1.0e-4,
        "warmup_rounds": min(5, int(rounds)),
        "warmup_learning_rate": 5.0e-6,
        "warmup_disc_learning_rate": 5.0e-5,
        "warmup_clip_range": 0.05,
        "clip_range": 0.10,
        "value_clip_range": 0.20,
        "target_kl": 0.005,
        "abort_on_health_failure": True,
        "ppo_epochs": 2,
        "batch_size": 4096,
        "disc_batch_size": 4096,
        "disc_updates_per_round": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coef": entropy,
        # The shared BC checkpoint is only a short numerical stabilization.
        # Adversarial training takes over immediately instead of continuing to
        # pull the policy toward BC for dozens of rounds.
        "policy_bc_regularization_coef": 0.0,
        "policy_bc_regularization_final_coef": 0.0,
        "policy_bc_regularization_decay_rounds": 50,
        "initial_policy_checkpoint": bc_checkpoint if initialization == "bc" else "",
        "enable_collision": True,
        "collision_mode_schedule": _collision_schedule(collision_training, rounds),
        "collision_mixed_on_fraction": 0.5,
        "collision_proxy_penalty_coef": 1.0,
        "collision_penalty": 2.0,
        "offroad_penalty": 2.0,
        "validation_every": 10,
        "validation_episodes": 10,
        "evaluate_initial_policy": True,
        "validation_max_score_drop": 5.0,
        "validation_regression_patience": 2,
        "validation_stress_every": 0,
        "validation_stress_episodes": 0,
        "test_episodes": 30,
        "test_vehicle_mode": "training_count",
        "checkpoint_every": 20,
        "save_checkpoint_video": False,
        "wandb_mode": "disabled",
    }
    if variant.endswith("wgan_gp"):
        args.update(
            {
                "wgan_gp_lambda": 10.0,
                "discriminator_spectral_norm": False,
                "wgan_reward_center": True,
                "wgan_reward_clip": 5.0,
                "normalize_gail_reward": False,
                "allow_wgan_reward_normalization": False,
            }
        )
    if method == "airl":
        args["airl_policy_reward_mode"] = (
            "discriminator" if variant == "airl_bce" else "shaped"
        )
    for name in (
        "policy_model",
        "hidden_size",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "transformer_memory_tokens",
        "transformer_memory_context_length",
        "transformer_use_causal_attention",
        "continuous_action_dim",
    ):
        if bc_policy_config is not None and name in bc_policy_config:
            args[name] = bc_policy_config[name]
    return args


def build_screening_trials(
    *,
    expert_data: str,
    episode_root: str,
    bc_checkpoint: str,
    bc_policy_config: dict[str, Any] | None = None,
    rounds: int = 60,
) -> list[StudyTrial]:
    """Return exactly 24 short trials per method (48 total)."""
    trials: list[StudyTrial] = []

    def add(
        method: str,
        phase: str,
        seed: int,
        variant: str,
        initialization: str,
        collision_training: str,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        index = len(trials)
        trial_id = f"{method}_{index:03d}_{phase}_s{seed}"
        run_name = f"gail_airl_study/{trial_id}"
        args = _base_arguments(
            method=method,
            variant=variant,
            seed=seed,
            run_name=run_name,
            collision_training=collision_training,
            initialization=initialization,
            expert_data=expert_data,
            episode_root=episode_root,
            bc_checkpoint=bc_checkpoint,
            bc_policy_config=bc_policy_config,
            rounds=rounds,
        )
        args.update(overrides or {})
        trials.append(
            StudyTrial(
                index=index,
                trial_id=trial_id,
                method=method,
                phase=phase,
                seed=seed,
                variant=variant,
                initialization=initialization,
                collision_training=collision_training,
                module=(
                    "scripts_gail.train_simple_ps_gail"
                    if method == "gail"
                    else "scripts_gail.train_simple_airl"
                ),
                run_name=run_name,
                arguments=args,
            )
        )

    hpo_points = [
        (3e-5, 5e-5, 0.10, 3, 2048, 0.005, 0.97, 0.90, 0.001, 1),
        (5e-5, 1e-4, 0.15, 4, 4096, 0.010, 0.99, 0.95, 0.002, 2),
        (1e-4, 2e-4, 0.20, 6, 2048, 0.020, 0.995, 0.97, 0.003, 1),
        (2e-4, 3e-4, 0.10, 4, 4096, 0.005, 0.99, 0.90, 0.005, 2),
        (3e-4, 5e-5, 0.15, 3, 2048, 0.010, 0.995, 0.95, 0.001, 4),
        (7e-5, 1.5e-4, 0.20, 4, 4096, 0.020, 0.97, 0.97, 0.004, 1),
        (1.5e-4, 7e-5, 0.10, 6, 2048, 0.010, 0.99, 0.97, 0.002, 2),
        (2.5e-4, 2.5e-4, 0.15, 4, 4096, 0.005, 0.995, 0.90, 0.003, 4),
    ]
    for method in ("gail", "airl"):
        canonical = f"{method}_bce"
        wgan = f"{method}_wgan_gp"
        # 12 paired factorial trials: warm start x collision physics x three seeds.
        for seed in range(3):
            for initialization in ("cold", "bc"):
                for collision_training in ("full", "soft"):
                    add(
                        method,
                        "factorial",
                        seed,
                        canonical,
                        initialization,
                        collision_training,
                    )
        # Four matched objective trials under the recommended curriculum.
        for seed in (3, 4):
            for variant in (canonical, wgan):
                add(method, "objective", seed, variant, "bc", "curriculum")
        # Eight deterministic space-filling hyperparameter trials.
        for offset, point in enumerate(hpo_points):
            (
                policy_lr,
                disc_lr,
                clip,
                epochs,
                batch,
                target_kl,
                gamma,
                gae_lambda,
                entropy,
                disc_updates,
            ) = point
            add(
                method,
                "hpo",
                100 + offset,
                canonical,
                "bc",
                "curriculum",
                {
                    "learning_rate": policy_lr,
                    "disc_learning_rate": disc_lr,
                    "clip_range": clip,
                    "ppo_epochs": epochs,
                    "batch_size": batch,
                    "target_kl": target_kl,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "entropy_coef": entropy,
                    "disc_updates_per_round": disc_updates,
                },
            )
    assert len(trials) == 48
    assert sum(trial.method == "gail" for trial in trials) == 24
    assert sum(trial.method == "airl" for trial in trials) == 24
    return trials


def write_study_files(output_dir: str, trials: list[StudyTrial], *, python: str = "python") -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "trials.json"
    commands_path = output / "commands.txt"
    manifest = {
        "schema_version": 1,
        "trial_count": len(trials),
        "trials": [asdict(trial) for trial in trials],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commands_path.write_text(
        "\n".join(trial.command(python=python) for trial in trials) + "\n",
        encoding="utf-8",
    )
    return {"manifest": str(manifest_path), "commands": str(commands_path)}


def safety_thresholds(*, expert_crash_rate: float, expert_offroad_rate: float) -> tuple[float, float]:
    return max(float(expert_crash_rate) + 0.02, 0.05), max(float(expert_offroad_rate) + 0.01, 0.02)


def assess_evaluation_summary(
    summary: dict[str, Any],
    *,
    expert_crash_rate: float,
    expert_offroad_rate: float,
    position_error_span: float = 1.0,
    speed_error_span: float = 1.0,
    lane_error_span: float = 1.0,
) -> dict[str, Any]:
    """Apply the pre-registered finite/safety eligibility gates."""
    test = dict(summary.get("test") or {})
    crash = test.get("test/vehicle_crash_rate", test.get("test/collision_rate"))
    offroad = test.get("test/vehicle_offroad_rate", test.get("test/offroad_duration_rate"))
    position = test.get("test/rmse_position_20s", test.get("test/rmse_position_final"))
    speed = test.get("test/rmse_speed_20s", test.get("test/rmse_speed_final"))
    lane = test.get("test/rmse_lane_offset_20s", test.get("test/rmse_lane_offset_final"))
    required = (crash, offroad, position, speed, lane)
    finite = all(value is not None and math.isfinite(float(value)) for value in required)
    crash_gate, offroad_gate = safety_thresholds(
        expert_crash_rate=expert_crash_rate,
        expert_offroad_rate=expert_offroad_rate,
    )
    eligible = bool(
        finite
        and float(crash) <= crash_gate
        and float(offroad) <= offroad_gate
    )
    trajectory_error_score = (
        float(position) / max(float(position_error_span), 1.0e-12)
        + float(speed) / max(float(speed_error_span), 1.0e-12)
        + float(lane) / max(float(lane_error_span), 1.0e-12)
        if finite
        else None
    )
    return {
        "eligible": eligible,
        "finite": finite,
        "vehicle_crash_rate": None if crash is None else float(crash),
        "vehicle_offroad_rate": None if offroad is None else float(offroad),
        "position_rmse_20s": None if position is None else float(position),
        "speed_rmse_20s": None if speed is None else float(speed),
        "lane_rmse_20s": None if lane is None else float(lane),
        "normalized_trajectory_error": trajectory_error_score,
        "crash_gate": crash_gate,
        "offroad_gate": offroad_gate,
    }


__all__ = [
    "StudyTrial",
    "assess_evaluation_summary",
    "build_screening_trials",
    "safety_thresholds",
    "write_study_files",
]
