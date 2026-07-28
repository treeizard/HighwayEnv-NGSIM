"""Locked configuration and integrity helpers for the four-run US GAIL/AIRL pilot."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .checkpoints import (
    policy_architecture_contract,
    sha256_file,
    shared_interpretable_transformer_architecture,
    verify_checkpoint_sidecar,
)
from .config import PSGAILConfig
from .validation import paper_driver_model_validation_overrides

PILOT_SCHEMA_VERSION = 1
BC_JOB_ID = "58391443"
TOTAL_ROUNDS = 600
SCRATCH_GAIL_TOTAL_ROUNDS = 800
GAIL_TRAINING_PROFILES = ("legacy_bce", "realistic_wgan_v1")
BC_INITIALIZER_QUALIFICATIONS = (
    "legacy_zero_event_v1",
    "matched_training_artifact_v1",
)
PLANNED_EVALUATION_VEHICLE_TRAJECTORIES = 7_440
MAX_PROJECTED_HOURS = 108.0


def gail_training_profile_overrides(profile: str) -> dict[str, Any]:
    """Return an additive GAIL recipe while retaining the historical default."""
    profile = str(profile).strip().lower()
    if profile not in GAIL_TRAINING_PROFILES:
        raise ValueError(
            f"Unsupported GAIL training profile {profile!r}; expected one of "
            f"{list(GAIL_TRAINING_PROFILES)}."
        )
    if profile == "legacy_bce":
        return {
            "algorithm_variant": "gail_bce",
            "learning_rate": 3.0e-5,
            "entropy_coef": 0.002,
        }
    return {
        "algorithm_variant": "gail_wgan_gp",
        "discriminator_input": "action",
        "wgan_gp_lambda": 2.0,
        "normalize_discriminator_features": True,
        "discriminator_feature_clip": 10.0,
        "disc_updates_per_round": 2,
        "discriminator_replay_rounds": 3,
        "discriminator_replay_max_samples": 120_000,
        "terminate_when_all_controlled_crashed": False,
        "rollout_fixed_horizon": True,
        "evaluation_terminate_when_all_controlled_crashed": False,
        "normalize_gail_reward": True,
        "allow_wgan_reward_normalization": True,
        "wgan_reward_center": False,
        "wgan_reward_clip": 0.0,
        "wgan_reward_scale": 1.0,
        "wgan_reward_norm_min_std": 1.0e-3,
        "wgan_reward_norm_clip": 5.0,
        "learning_rate": 1.0e-5,
        "entropy_coef": 5.0e-4,
        "policy_bc_regularization_coef": 0.02,
        "policy_bc_regularization_final_coef": 0.0,
        "policy_bc_regularization_decay_rounds": 50,
        "initial_action_std": "0.10,0.05",
        "minimum_action_std": "0.02,0.01",
        "maximum_action_std": "0.30,0.15",
        "full_load_selection_start_round": 701,
    }


def _json_load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _file_record(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {"path": str(path), "bytes": int(path.stat().st_size)}
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def _source_paths(repo: Path) -> list[Path]:
    paths: set[Path] = set()
    for root in (repo / "scripts_gail", repo / "highway_env"):
        paths.update(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    for relative in (
        "hpc/slurm/project_env.bash",
        "hpc/slurm/script_full_training/run_gail_airl_us_pilot.bash",
        "hpc/slurm/script_full_training/submit_gail_airl_us_pilot.bash",
        "hpc/slurm/script_full_training/run_gail_us_scratch_depth.bash",
        "hpc/slurm/script_full_training/submit_gail_us_scratch.bash",
        "hpc/slurm/script_full_training/submit_gail_us_realistic_wgan.bash",
        "hpc/slurm/script_full_training/submit_gail_us_realistic_wgan_kl_control_repair.bash",
        "hpc/slurm/script_full_training/submit_failed_policy_recovery.bash",
    ):
        path = repo / relative
        if path.is_file():
            paths.add(path)
    return sorted(paths)


def source_lock(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    revision = subprocess.check_output(
        ("git", "-C", str(repo), "rev-parse", "HEAD"), text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ("git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"),
            text=True,
        ).strip()
    )
    hashes = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in _source_paths(repo)
    }
    if not hashes:
        raise RuntimeError(f"No execution source files found under {repo}")
    return {
        "repo": str(repo),
        "revision": revision,
        "dirty_when_built": dirty,
        "files_sha256": hashes,
    }


def verify_source_lock(manifest: dict[str, Any], repo: Path) -> None:
    repo = repo.resolve()
    source = dict(manifest.get("source") or {})
    if Path(str(source.get("repo") or "")).resolve() != repo:
        raise RuntimeError("Pilot manifest repository does not match REPODIR.")
    actual_revision = subprocess.check_output(
        ("git", "-C", str(repo), "rev-parse", "HEAD"), text=True
    ).strip()
    if actual_revision != str(source.get("revision") or ""):
        raise RuntimeError("Pilot source revision changed after manifest creation.")
    hashes = dict(source.get("files_sha256") or {})
    if not hashes:
        raise RuntimeError("Pilot manifest has no source hash lock.")
    for relative, expected in hashes.items():
        path = repo / str(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise RuntimeError(f"Pilot source hash mismatch: {relative}")


def _validate_bc_candidate(
    checkpoint: Path,
    *,
    depth: int,
    seed: int,
    qualification: str = "legacy_zero_event_v1",
) -> dict[str, Any]:
    qualification = str(qualification).strip().lower()
    if qualification not in BC_INITIALIZER_QUALIFICATIONS:
        raise ValueError(
            f"Unsupported BC initializer qualification {qualification!r}; "
            f"expected one of {list(BC_INITIALIZER_QUALIFICATIONS)}."
        )
    checkpoint = checkpoint.resolve()
    summary_path = checkpoint.with_name("summary.json")
    summary = _json_load(summary_path)
    actual_sha = verify_checkpoint_sidecar(checkpoint)
    try:
        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint_payload, dict):
        raise RuntimeError(f"BC initializer lacks checkpoint metadata: {checkpoint}")
    architecture = policy_architecture_contract(checkpoint_payload)
    failures: list[str] = []
    expected_values = {
        "domain": "us",
        "scene": "us-101",
        "seed": int(seed),
        "transformer_layers": int(depth),
        "checkpoint_purpose": "policy",
        "checkpoint_eligibility": "full_policy_best_validation",
    }
    for key, expected in expected_values.items():
        if summary.get(key) != expected:
            failures.append(f"{key}={summary.get(key)!r}, expected {expected!r}")
    for key in (
        "checkpoint_saved",
        "capability_passed",
        "metric_capability_passed",
        "rollout_capability_passed",
        "learning_signal_passed",
    ):
        if summary.get(key) is not True:
            failures.append(f"{key} did not pass")
    if int(summary.get("completed_epochs", 0)) != int(summary.get("configured_epochs", -1)):
        failures.append("BC training did not complete its configured epochs")
    if str(summary.get("checkpoint_sha256") or "") != actual_sha:
        failures.append("summary checkpoint SHA does not match checkpoint")
    held = dict(summary.get("held_out_evaluation") or {})
    held_metrics = dict(held.get("metrics") or {})
    qualification_metrics: dict[str, float]
    if qualification == "legacy_zero_event_v1":
        for key in ("bc_eval/crash_episode_fraction", "bc_eval/offroad_episode_fraction"):
            if float(held_metrics.get(key, 1.0)) != 0.0:
                failures.append(f"held-out {key} is nonzero")
        if float(held_metrics.get("bc_eval/mean_episode_length", 0.0)) < 200.0:
            failures.append("held-out BC policy did not cover the full 20-second horizon")
        qualification_metrics = {
            "held_out_crash_episode_fraction": float(
                held_metrics.get("bc_eval/crash_episode_fraction", float("nan"))
            ),
            "held_out_offroad_episode_fraction": float(
                held_metrics.get("bc_eval/offroad_episode_fraction", float("nan"))
            ),
            "held_out_mean_episode_length": float(
                held_metrics.get("bc_eval/mean_episode_length", float("nan"))
            ),
        }
    else:
        for key in (
            "training_artifact_complete",
            "interpretability_baseline_eligible",
            "matched_evaluation_passed",
            "closed_loop_evaluation_complete",
        ):
            if summary.get(key) is not True:
                failures.append(f"{key} did not pass")
        expected_held_values = {
            "prebuilt_split": "test",
            "collision_physics_enabled": True,
            "collision_termination_enabled": False,
            "vehicle_mode": "all",
        }
        for key, expected in expected_held_values.items():
            if held.get(key) != expected:
                failures.append(
                    f"held-out {key}={held.get(key)!r}, expected {expected!r}"
                )
        metric_keys = (
            "test/mean_episode_length",
            "test/horizon_coverage_20s",
            "test/vehicle_crash_rate",
            "test/vehicle_offroad_rate",
            "test/acceleration_action_std",
            "test/steering_action_std",
        )
        parsed_metrics: dict[str, float] = {}
        for key in metric_keys:
            try:
                value = float(held_metrics[key])
            except (KeyError, TypeError, ValueError):
                failures.append(f"held-out {key} is missing or invalid")
                continue
            if not torch.isfinite(torch.tensor(value)).item():
                failures.append(f"held-out {key} is non-finite")
            parsed_metrics[key] = value
        if parsed_metrics.get("test/mean_episode_length", 0.0) < 200.0:
            failures.append("held-out BC evaluation did not run for 200 timesteps")
        if parsed_metrics.get("test/horizon_coverage_20s", 0.0) <= 0.0:
            failures.append("held-out BC evaluation has no 20-second coverage")
        for key in ("test/acceleration_action_std", "test/steering_action_std"):
            if parsed_metrics.get(key, 0.0) <= 1.0e-3:
                failures.append(f"held-out {key} shows collapsed actions")
        qualification_metrics = {
            "held_out_vehicle_crash_rate": parsed_metrics.get(
                "test/vehicle_crash_rate", float("nan")
            ),
            "held_out_vehicle_offroad_rate": parsed_metrics.get(
                "test/vehicle_offroad_rate", float("nan")
            ),
            "held_out_horizon_coverage_20s": parsed_metrics.get(
                "test/horizon_coverage_20s", float("nan")
            ),
            "held_out_mean_episode_length": parsed_metrics.get(
                "test/mean_episode_length", float("nan")
            ),
            "held_out_acceleration_action_std": parsed_metrics.get(
                "test/acceleration_action_std", float("nan")
            ),
            "held_out_steering_action_std": parsed_metrics.get(
                "test/steering_action_std", float("nan")
            ),
        }
    if failures:
        raise RuntimeError(f"Unqualified BC initializer {checkpoint}: {'; '.join(failures)}")
    expert = dict(summary.get("expert_data") or {})
    return {
        "depth": int(depth),
        "seed": int(seed),
        "checkpoint": _file_record(checkpoint),
        "sidecar": _file_record(checkpoint.with_name(f"{checkpoint.name}.sha256")),
        "summary": _file_record(summary_path),
        "qualification": {
            "contract": qualification,
            "checkpoint_eligibility": summary["checkpoint_eligibility"],
            "capability_passed": True,
            "learning_signal_passed": True,
            **qualification_metrics,
        },
        "policy_architecture": architecture,
        "expert_manifest_sha256": str(expert.get("manifest_sha256") or ""),
    }


def _common_arguments(
    *,
    expert_data: Path,
    episode_root: Path,
    run_root: Path,
    campaign_id: str,
    require_explicit_data_contracts: bool,
    include_100_vehicle_phase: bool = False,
) -> dict[str, Any]:
    total_rounds = (
        SCRATCH_GAIL_TOTAL_ROUNDS
        if include_100_vehicle_phase
        else TOTAL_ROUNDS
    )
    final_controlled_vehicles = 100.0 if include_100_vehicle_phase else 50.0
    controlled_vehicle_schedule = (
        (
            "1:120:10:10;121:240:20:20;241:360:30:30;"
            "361:480:40:40;481:600:50:50;"
            "601:640:50:70;641:680:70:90;681:700:90:100;"
            "701:800:100:100"
        )
        if include_100_vehicle_phase
        else (
            "1:120:10:10;121:240:20:20;241:360:30:30;"
            "361:480:40:40;481:600:50:50"
        )
    )
    rollout_target_agent_steps_schedule = (
        (
            "1:500:10000:10000;501:600:10000:20000;"
            "601:640:20000:25000;641:680:25000:32000;"
            "681:700:32000:40000;701:800:40000:40000"
        )
        if include_100_vehicle_phase
        else "1:500:10000:10000;501:600:10000:20000"
    )
    return {
        "expert_data": str(expert_data.resolve()),
        "episode_root": str(episode_root.resolve()),
        "scene": "us-101",
        "prebuilt_split": "train",
        "validation_prebuilt_split": "val",
        "test_prebuilt_split": "test",
        "run_root": str(run_root.resolve()),
        "device": "cuda",
        "action_mode": "continuous",
        "continuous_action_dim": 2,
        "require_explicit_data_contracts": bool(
            require_explicit_data_contracts
        ),
        "transformer_recurrent_sequence_length": 32,
        "transformer_recurrent_sequences_per_batch": 16,
        "transformer_recurrent_micro_batch_sequences": 16,
        "transformer_memory_storage_dtype": "float16",
        "transformer_use_causal_attention": True,
        "centralized_critic": False,
        "bc_pretrain_epochs": 0,
        "policy_bc_regularization_coef": 0.0,
        "policy_bc_regularization_final_coef": 0.0,
        "policy_bc_regularization_decay_rounds": 0,
        "total_rounds": total_rounds,
        "max_expert_samples": 100_000,
        "controlled_vehicle_curriculum": True,
        "initial_controlled_vehicles": 10.0,
        "final_controlled_vehicles": final_controlled_vehicles,
        "controlled_vehicle_curriculum_rounds": total_rounds,
        "controlled_vehicle_schedule": controlled_vehicle_schedule,
        "rollout_target_agent_steps": 10_000,
        "rollout_target_agent_steps_schedule": (
            rollout_target_agent_steps_schedule
        ),
        "rollout_min_episodes": 1,
        "rollout_full_episodes": True,
        "rollout_target_aware_episodes": True,
        "rollout_target_min_episodes": 1,
        "rollout_max_episode_steps": 200,
        "max_episode_steps": 200,
        "rollout_cache_envs": True,
        "rollout_max_cached_envs_per_worker": 2,
        "num_rollout_workers": 8,
        "rollout_worker_threads": 2,
        "evaluation_num_workers": 8,
        "evaluation_worker_threads": 2,
        "evaluation_cache_envs": True,
        "evaluation_max_cached_envs_per_worker": 4,
        "road_query_mode": "spatial",
        "collision_check_mode": "broadphase",
        "record_replay_diagnostics": False,
        "sensor_road_edge_mode": "batched",
        "reuse_pre_reset_spaces": True,
        "disc_learning_rate": 1.0e-4,
        "warmup_rounds": 5,
        "warmup_learning_rate": 5.0e-6,
        "warmup_disc_learning_rate": 5.0e-5,
        "warmup_clip_range": 0.05,
        "clip_range": 0.10,
        "value_clip_range": 0.20,
        "target_kl": 0.005,
        "ppo_epochs": 2,
        "batch_size": 4096,
        "disc_batch_size": 4096,
        "disc_updates_per_round": 1,
        "discriminator_replay_rounds": 0,
        "discriminator_replay_max_samples": 0,
        "collision_mode_schedule": (
            f"1:200:soft;201:400:mixed;401:{total_rounds}:full"
        ),
        "enable_collision": True,
        "collision_mixed_on_fraction": 0.5,
        "collision_proxy_penalty_coef": 1.0,
        "vehicle_increase_soft_collision_rounds": 5,
        "collision_penalty": 2.0,
        "offroad_penalty": 2.0,
        "evaluate_initial_policy": True,
        "validation_every": 20,
        "validation_episodes": 4,
        "validation_stress_every": 100,
        "validation_stress_episodes": 2,
        "validation_stress_vehicle_mode": "all",
        "test_episodes": 12,
        **paper_driver_model_validation_overrides(),
        "validation_min_delta": 0.0,
        "validation_max_score_drop": 0.0,
        "validation_regression_patience": 0,
        "abort_on_health_failure": True,
        "health_kl_patience": 5,
        "health_discriminator_patience": 5,
        "health_reward_std_patience": 5,
        "health_min_reward_std": 1.0e-3,
        "health_min_action_std": 1.0e-3,
        "health_learning_gate_round": 100,
        "health_min_best_round": 20,
        "health_min_relative_validation_improvement": 0.02,
        "checkpoint_every": 20,
        "save_best_checkpoint": True,
        "save_checkpoint_video": False,
        "wandb_mode": "online",
        "wandb_project": "highwayenv-ps-gail",
        "wandb_group": campaign_id,
        "wandb_tags": "us,gail-airl,pilot,recurrent-transformer,locked",
        "wandb_watch": False,
        "wandb_compact_metrics": True,
        "study_domain": "us",
        "study_stage": 0,
    }


def build_manifest(
    *,
    repo: Path,
    project_root: Path,
    run_root: Path,
    campaign_id: str,
    methods: tuple[str, ...] = ("gail", "airl"),
    use_bc_initialization: bool = True,
    num_rollout_workers: int = 8,
    expert_data: Path | None = None,
    bc_root: Path | None = None,
    policy_recipe: Path | None = None,
    shared_policy_seed: int | None = None,
    require_explicit_data_contracts: bool = False,
    gail_training_profile: str = "legacy_bce",
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", campaign_id):
        raise ValueError(f"Unsafe campaign id: {campaign_id!r}")
    repo = repo.resolve()
    project_root = project_root.resolve()
    run_root = run_root.resolve()
    policy_recipe = (
        policy_recipe.resolve()
        if policy_recipe is not None
        else repo / "configs/bc_gail_aligned_accel5_v4.json"
    )
    policy_recipe_payload = _json_load(policy_recipe)
    policy_recipe_record = _file_record(policy_recipe)
    if not methods or any(method not in {"gail", "airl"} for method in methods):
        raise ValueError(f"Unsupported method set: {methods!r}")
    if len(set(methods)) != len(methods):
        raise ValueError(f"Duplicate method in method set: {methods!r}")
    if int(num_rollout_workers) < 1:
        raise ValueError("num_rollout_workers must be positive")
    gail_training_profile = str(gail_training_profile).strip().lower()
    gail_profile_overrides = gail_training_profile_overrides(
        gail_training_profile
    )
    if gail_training_profile != "legacy_bce":
        if tuple(methods) != ("gail",):
            raise ValueError(
                "The realistic WGAN profile is scoped to a GAIL-only manifest."
            )
        if not use_bc_initialization:
            raise ValueError(
                "The realistic WGAN profile requires a verified matched BC artifact."
            )
    expert_data = (
        expert_data.resolve()
        if expert_data is not None
        else project_root
        / "data/expert/ngsim_ps_unified_expert_continuous_55145982"
    )
    episode_root = project_root / "data/highway_env/processed_20s"
    bc_root = (
        bc_root.resolve()
        if bc_root is not None
        else project_root
        / "results/runs/policies/bc/domain_depth_recovered_58391443/us"
    )
    depth_seed_pairs = (
        ((2, int(shared_policy_seed)), (3, int(shared_policy_seed)))
        if shared_policy_seed is not None
        else ((2, 0), (3, 1))
    )
    bc_specs = tuple(
        (
            depth,
            seed,
            bc_root
            / f"recurrent_transformer_{depth}layer"
            / f"policy_seed_{seed}"
            / "best.pt",
        )
        for depth, seed in depth_seed_pairs
    )
    initializer_qualification = (
        "matched_training_artifact_v1"
        if gail_training_profile == "realistic_wgan_v1"
        else "legacy_zero_event_v1"
    )
    initializers = (
        [
            _validate_bc_candidate(
                path,
                depth=depth,
                seed=seed,
                qualification=initializer_qualification,
            )
            for depth, seed, path in bc_specs
        ]
        if use_bc_initialization
        else []
    )
    expert_manifest = _file_record(expert_data / "manifest.json")
    if use_bc_initialization:
        expected_expert_hashes = {item["expert_manifest_sha256"] for item in initializers}
        if expected_expert_hashes != {expert_manifest["sha256"]}:
            raise RuntimeError(
                "BC initializers and requested expert dataset do not share one manifest."
            )
    prebuilt = episode_root / "us-101/prebuilt"
    data_files = {
        path.name: _file_record(path)
        for split in ("train", "val", "test")
        for path in (
            prebuilt / f"trajectory_{split}.npy",
            prebuilt / f"veh_ids_{split}.npy",
        )
    }
    common = _common_arguments(
        expert_data=expert_data,
        episode_root=episode_root,
        run_root=run_root,
        campaign_id=campaign_id,
        require_explicit_data_contracts=bool(
            require_explicit_data_contracts
        ),
        include_100_vehicle_phase=(
            tuple(methods) == ("gail",)
            and (
                not use_bc_initialization
                or gail_training_profile == "realistic_wgan_v1"
            )
        ),
    )
    common.update(
        {
            "num_rollout_workers": int(num_rollout_workers),
            "evaluation_num_workers": int(num_rollout_workers),
        }
    )
    if not use_bc_initialization:
        common["wandb_tags"] = (
            "us,gail,pilot,recurrent-transformer,locked,scratch-init,no-bc-init"
        )
        # A random policy commonly terminates before 20 s. Requiring exact
        # 20-second coverage during optimization turns that expected poor
        # performance into -inf and aborts at the first validation gate. Use
        # finite final-state errors plus explicit crash/off-road penalties for
        # checkpoint selection; the terminal audit still enforces the planned
        # 20-second coverage acceptance gate.
        common["validation_require_exact_horizon"] = False
        common["validation_min_horizon_coverage"] = 0.0
    elif gail_training_profile == "realistic_wgan_v1":
        common["wandb_tags"] = (
            "us,gail,ps-gail,wgan-gp,bc-initialized,recurrent-transformer,"
            "realistic-driving,full-load-100,locked"
        )
    initializer_by_depth = {int(item["depth"]): item for item in initializers}
    trials: list[dict[str, Any]] = []
    for method in methods:
        for depth, seed in depth_seed_pairs:
            arguments = dict(common)
            arguments.update(
                shared_interpretable_transformer_architecture(
                    policy_recipe_payload,
                    depth=depth,
                )
            )
            arguments.update(
                {
                    "run_name": (
                        f"{method}/recurrent_transformer_{depth}layer_seed_{seed}"
                    ),
                    "seed": seed,
                    "transformer_layers": depth,
                    "algorithm_variant": f"{method}_bce",
                    "learning_rate": 3.0e-5 if method == "gail" else 1.0e-5,
                    "entropy_coef": 0.002 if method == "gail" else 0.003,
                }
            )
            if method == "gail":
                arguments.update(gail_profile_overrides)
            if use_bc_initialization:
                initializer = initializer_by_depth[depth]
                arguments.update(initializer["policy_architecture"])
                arguments["initial_policy_checkpoint"] = initializer["checkpoint"][
                    "path"
                ]
            trials.append(
                {
                    "index": len(trials),
                    "trial_id": f"{method}_us_d{depth}_s{seed}",
                    "method": method,
                    "depth": depth,
                    "seed": seed,
                    "initializer_depth": depth if use_bc_initialization else None,
                    "arguments": arguments,
                    "trainer_arguments": (
                        {
                            "reward_batch_size": 4096,
                            "airl_log_prob_batch_size": 512,
                        }
                        if method == "airl"
                        else {}
                    ),
                }
            )
    required_completed_round = max(
        int(dict(trial["arguments"])["total_rounds"]) for trial in trials
    )
    conservative_training_rounds = required_completed_round + 50
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "scope": {
            "domain": "us",
            "methods": list(methods),
            "depth_seed_pairs": [
                [int(depth), int(seed)] for depth, seed in depth_seed_pairs
            ],
            "trial_count": len(trials),
            "policy_initialization": (
                (
                    "verified_matched_bc_checkpoint"
                    if initializer_qualification == "matched_training_artifact_v1"
                    else "qualified_bc_checkpoint"
                )
                if use_bc_initialization
                else "random_seeded"
            ),
            "bc_initializer_qualification": (
                initializer_qualification if use_bc_initialization else None
            ),
            "uses_bc_initialization": bool(use_bc_initialization),
            "gail_training_profile": gail_training_profile,
            "depth_comparison_is_exploratory": True,
            "reason": (
                (
                    (
                        "depths use one paired verified matched BC artifact"
                        if shared_policy_seed is not None
                        else "depths use different verified matched BC artifacts"
                    )
                    if initializer_qualification == "matched_training_artifact_v1"
                    else (
                        "depths use one paired qualified BC policy seed"
                        if shared_policy_seed is not None
                        else "depths use different qualified BC policy seeds"
                    )
                )
                if use_bc_initialization
                else (
                    "depths are independently random-initialized with one paired seed"
                    if shared_policy_seed is not None
                    else "depths are independently random-initialized with frozen seeds"
                )
            ),
        },
        "source": source_lock(repo),
        "data": {
            "expert_root": str(expert_data.resolve()),
            "expert_manifest": expert_manifest,
            "policy_recipe": policy_recipe_record,
            "architecture_contract_id": str(
                policy_recipe_payload["architecture_contract_id"]
            ),
            "explicit_contracts_required": bool(
                require_explicit_data_contracts
            ),
            "episode_root": str(episode_root.resolve()),
            "prebuilt_files": data_files,
        },
        "initializers": initializers,
        "trials": trials,
        "resources": {
            "jobs": (
                len(trials)
                if (
                    not use_bc_initialization
                    or gail_training_profile == "realistic_wgan_v1"
                )
                else 2
            ),
            "job_shape": (
                "one independent depth per job; depths may run concurrently"
                if (
                    not use_bc_initialization
                    or gail_training_profile == "realistic_wgan_v1"
                )
                else "one method per job; two depths concurrent"
            ),
            "gpus_per_job": (
                1
                if (
                    not use_bc_initialization
                    or gail_training_profile == "realistic_wgan_v1"
                )
                else 2
            ),
            "gpu_type": "L40S",
            "cpus_per_job": 32,
            "memory_per_job_gb": 64,
            "walltime": "4-12:00:00",
            "walltime_hours": 108,
            "arrays": False,
            "internal_dependencies": False,
            "requeue": False,
            "automatic_expansion": False,
            "automatic_retry": False,
        },
        "blocking_gate": (
            {
                "bc_job_id": None,
                "required_terminal_state": None,
                "reason": (
                    "Matched BC artifacts are verified directly by checksum, "
                    "metadata, and held-out evaluation."
                ),
            }
            if gail_training_profile == "realistic_wgan_v1"
            else {
                "bc_job_id": BC_JOB_ID,
                "required_terminal_state": "COMPLETED",
            }
            if use_bc_initialization
            else {
                "bc_job_id": None,
                "required_terminal_state": None,
                "reason": "BC initialization is intentionally disabled",
            }
        ),
        "runtime_projection": {
            "gate_round": 20,
            "safety_factor": 1.10,
            "conservative_training_rounds": conservative_training_rounds,
            "planned_evaluation_vehicle_trajectories": (
                PLANNED_EVALUATION_VEHICLE_TRAJECTORIES
            ),
            "maximum_projected_hours": MAX_PROJECTED_HOURS,
            "formula": (
                f"1.10 * ({conservative_training_rounds} * "
                "p95(round_seconds for rounds 2..19) + "
                "eval_seconds_per_vehicle_trajectory * 7440 / "
                "evaluation_num_workers)"
            ),
        },
        "acceptance": {
            "required_completed_round": required_completed_round,
            "minimum_best_round": 20,
            "minimum_relative_validation_cost_improvement": 0.02,
            "minimum_20s_horizon_coverage": 0.95,
            "selected_test_cost_must_not_exceed_initializer": bool(
                use_bc_initialization
            ),
            "maximum_test_vehicle_crash_rate": 0.05,
            "maximum_test_vehicle_offroad_rate": 0.02,
            "minimum_action_std_absolute": 1.0e-3,
            "minimum_action_std_initializer_fraction": (
                0.25 if use_bc_initialization else None
            ),
            "minimum_policy_relative_l2_delta": 1.0e-6,
            "minimum_adversarial_reward_std": 1.0e-3,
            "require_all_four": len(trials) == 4,
            "require_wandb_online": True,
            "wandb_project": "highwayenv-ps-gail",
        },
        "simulator_speed_evidence": {
            "us_50_controlled_exact_full_step_speedup": 1.387,
            "us_50_controlled_sensor_speedup": 2.736,
            "us_100_controlled_exact_full_step_speedup": 1.484,
            "us_100_controlled_sensor_speedup": 2.849,
            "projected_end_to_end_gail_speedup": [1.20, 1.35],
            "projected_end_to_end_airl_speedup": [1.18, 1.25],
            "projection_requires_live_measurement": True,
        },
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = _json_load(path.resolve())
    if int(payload.get("schema_version", -1)) != PILOT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported GAIL/AIRL pilot manifest schema.")
    scope = dict(payload.get("scope") or {})
    methods = tuple(str(method) for method in list(scope.get("methods") or []))
    if not methods or any(method not in {"gail", "airl"} for method in methods):
        raise RuntimeError(f"Pilot contains unsupported methods: {methods!r}")
    if len(set(methods)) != len(methods):
        raise RuntimeError(f"Pilot contains duplicate methods: {methods!r}")
    trials = list(payload.get("trials") or [])
    expected_count = 2 * len(methods)
    if len(trials) != expected_count or int(scope.get("trial_count", -1)) != expected_count:
        raise RuntimeError(
            f"Pilot must contain exactly {expected_count} trials, found {len(trials)}."
        )
    observed = {
        (str(row.get("method")), int(row.get("depth", -1)), int(row.get("seed", -1)))
        for row in trials
    }
    raw_depth_seed_pairs = list(
        scope.get("depth_seed_pairs") or [[2, 0], [3, 1]]
    )
    depth_seed_pairs = {
        (int(pair[0]), int(pair[1]))
        for pair in raw_depth_seed_pairs
        if isinstance(pair, list) and len(pair) == 2
    }
    if {depth for depth, _seed in depth_seed_pairs} != {2, 3}:
        raise RuntimeError(
            f"Pilot must contain transformer depths 2 and 3: {raw_depth_seed_pairs!r}"
        )
    expected = {
        (method, depth, seed)
        for method in methods
        for depth, seed in depth_seed_pairs
    }
    if observed != expected:
        raise RuntimeError(f"Pilot trial set changed: {sorted(observed)}")
    uses_bc = bool(scope.get("uses_bc_initialization", True))
    if not uses_bc:
        if list(payload.get("initializers") or []):
            raise RuntimeError("Scratch-initialized pilot must not contain BC initializers.")
        exact_resume_recovery = bool(scope.get("exact_resume_recovery", False))
        for row in trials:
            arguments = dict(row.get("arguments") or {})
            if exact_resume_recovery:
                if arguments.get("initial_policy_checkpoint"):
                    raise RuntimeError(
                        "Scratch recovery must not replace the random-seeded initializer."
                    )
                if not arguments.get("resume_checkpoint"):
                    raise RuntimeError(
                        "Scratch recovery trial has no exact resume checkpoint."
                    )
                if int(arguments.get("expected_resume_round", 0)) <= 0:
                    raise RuntimeError(
                        "Scratch recovery trial has no positive expected resume round."
                    )
            else:
                prohibited = {
                    name: arguments.get(name)
                    for name in ("initial_policy_checkpoint", "resume_checkpoint")
                    if arguments.get(name)
                }
                if prohibited:
                    raise RuntimeError(
                        f"Scratch-initialized trial contains a checkpoint: {prohibited}"
                    )
            required_zero = (
                "bc_pretrain_epochs",
                "policy_bc_regularization_coef",
                "policy_bc_regularization_final_coef",
                "policy_bc_regularization_decay_rounds",
            )
            nonzero = {
                name: arguments.get(name)
                for name in required_zero
                if float(arguments.get(name, 0)) != 0.0
            }
            if nonzero:
                raise RuntimeError(
                    f"Scratch-initialized trial enables BC training/regularization: {nonzero}"
                )
    return payload


def verify_manifest_inputs(
    manifest: dict[str, Any],
    *,
    repo: Path,
    include_large_data: bool = True,
) -> None:
    verify_source_lock(manifest, repo)
    for initializer in list(manifest.get("initializers") or []):
        checkpoint = Path(initializer["checkpoint"]["path"])
        if verify_checkpoint_sidecar(checkpoint) != initializer["checkpoint"]["sha256"]:
            raise RuntimeError(f"BC initializer hash changed: {checkpoint}")
        for name in ("summary", "sidecar"):
            record = dict(initializer[name])
            if sha256_file(record["path"]) != str(record["sha256"]):
                raise RuntimeError(f"BC initializer {name} changed: {record['path']}")
    expert_manifest = dict(dict(manifest.get("data") or {}).get("expert_manifest") or {})
    if sha256_file(expert_manifest["path"]) != str(expert_manifest["sha256"]):
        raise RuntimeError("Expert-data manifest changed after pilot lock.")
    policy_recipe = dict(dict(manifest.get("data") or {}).get("policy_recipe") or {})
    if policy_recipe and sha256_file(policy_recipe["path"]) != str(
        policy_recipe["sha256"]
    ):
        raise RuntimeError("Shared BC/GAIL policy recipe changed after pilot lock.")
    if include_large_data:
        for record in dict(dict(manifest["data"])["prebuilt_files"]).values():
            if sha256_file(record["path"]) != str(record["sha256"]):
                raise RuntimeError(f"Prebuilt evaluation data changed: {record['path']}")


def select_trial(
    manifest: dict[str, Any], *, method: str, depth: int
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in list(manifest.get("trials") or [])
        if row.get("method") == method and int(row.get("depth", -1)) == int(depth)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {method} depth-{depth} pilot trial.")
    return matches[0]


def trial_argv(trial: dict[str, Any], *, python: str = sys.executable) -> list[str]:
    method = str(trial.get("method") or "")
    module = (
        "scripts_gail.train_simple_ps_gail"
        if method == "gail"
        else "scripts_gail.train_simple_airl"
    )
    if method not in {"gail", "airl"}:
        raise RuntimeError(f"Unsupported pilot method: {method!r}")
    defaults = PSGAILConfig()
    field_map = {field.name: field for field in fields(PSGAILConfig)}
    arguments = dict(trial.get("arguments") or {})
    unknown = sorted(set(arguments).difference(field_map))
    if unknown:
        raise RuntimeError(f"Unknown PSGAILConfig pilot arguments: {unknown}")
    argv = [str(python), "-m", module]
    for name in sorted(arguments):
        value = arguments[name]
        option = "--" + name.replace("_", "-")
        default_value = getattr(defaults, name)
        if isinstance(default_value, bool):
            argv.append(option if bool(value) else "--no-" + name.replace("_", "-"))
        else:
            argv.extend((option, str(value)))
    for name, value in sorted(dict(trial.get("trainer_arguments") or {}).items()):
        argv.extend(("--" + str(name).replace("_", "-"), str(value)))
    return argv


def source_fingerprint(manifest: dict[str, Any]) -> str:
    hashes = dict(dict(manifest.get("source") or {}).get("files_sha256") or {})
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BC_JOB_ID",
    "GAIL_TRAINING_PROFILES",
    "MAX_PROJECTED_HOURS",
    "PLANNED_EVALUATION_VEHICLE_TRAJECTORIES",
    "SCRATCH_GAIL_TOTAL_ROUNDS",
    "TOTAL_ROUNDS",
    "build_manifest",
    "gail_training_profile_overrides",
    "load_manifest",
    "select_trial",
    "source_fingerprint",
    "trial_argv",
    "verify_manifest_inputs",
    "verify_source_lock",
    "write_manifest",
]
