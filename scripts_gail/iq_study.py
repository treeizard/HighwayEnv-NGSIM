"""Shared contracts for the recurrent IQ-Learn pilot and 12-cell study."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


DOMAINS = ("us", "japanese")
DEPTHS = (2, 3)
SEEDS = (0, 1, 2)
CONFIRMATIONS = (("us", 3, 0), ("japanese", 3, 1))


def matrix_cells(*, confirmation_first: bool = True) -> list[tuple[str, int, int]]:
    cells = [(domain, depth, seed) for domain in DOMAINS for depth in DEPTHS for seed in SEEDS]
    if not confirmation_first:
        return cells
    return [*CONFIRMATIONS, *(cell for cell in cells if cell not in CONFIRMATIONS)]


def scene_for_domain(domain: str) -> str:
    return "us-101" if domain == "us" else "japanese"


def expert_for_domain(domain: str, us_expert: Path, japanese_expert: Path) -> Path:
    return us_expert if domain == "us" else japanese_expert


def cell_relative_path(domain: str, depth: int, seed: int) -> Path:
    return Path(domain) / f"recurrent_transformer_{depth}layer" / f"policy_seed_{seed}"


def bc_checkpoint(bc_root: Path, domain: str, depth: int, seed: int) -> Path:
    return bc_root / cell_relative_path(domain, depth, seed) / "best.pt"


def read_locked_recipe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1 or payload.get("status") != "locked":
        raise ValueError(f"IQ recipe is not locked schema v1: {path}")
    for section in ("architecture", "optimization", "replay", "data", "gates", "evaluation"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"IQ recipe is missing {section!r}: {path}")
    return payload


def trainer_command(
    recipe: dict[str, Any],
    *,
    domain: str,
    depth: int,
    seed: int,
    expert_data: Path,
    episode_root: Path,
    initial_checkpoint: Path,
    out_dir: Path,
    device: str,
    capability_failure_mode: str = "report",
) -> list[str]:
    architecture = recipe["architecture"]
    optimization = recipe["optimization"]
    replay = recipe["replay"]
    data = recipe["data"]
    gates = recipe["gates"]
    evaluation = recipe["evaluation"]
    command = [
        sys.executable, "-m", "scripts_gail.train_recurrent_iq_learn",
        "--expert-data", str(expert_data), "--out-dir", str(out_dir),
        "--domain", domain, "--scene", scene_for_domain(domain),
        "--episode-root", str(episode_root), "--prebuilt-split", "train",
        "--seed", str(seed), "--data-seed", str(data["data_seed"]),
        "--max-expert-samples", str(data["max_expert_samples"]), "--device", device,
        "--initial-policy-checkpoint", str(initial_checkpoint),
        "--hidden-size", str(architecture["hidden_size"]), "--transformer-layers", str(depth),
        "--transformer-heads", str(architecture["transformer_heads"]),
        "--transformer-dropout", str(architecture["transformer_dropout"]),
        "--memory-tokens", str(architecture["memory_tokens"]),
        "--memory-context-length", str(architecture["memory_context_length"]),
        "--training-context-length", str(optimization["training_context_length"]),
        "--sequence-length", str(optimization["sequence_length"]),
        "--validation-sequence-length", str(evaluation["validation_sequence_length"]),
        "--sequences-per-update", str(optimization["sequences_per_update"]),
        "--micro-batch-sequences", str(optimization["micro_batch_sequences"]),
        "--updates", str(optimization["updates"]),
        "--minimum-joint-updates", str(optimization["minimum_joint_updates"]),
        "--eval-every", str(optimization["eval_every"]),
        "--early-stopping-evaluations", str(optimization["early_stopping_evaluations"]),
        "--policy-learning-rate", str(optimization["policy_learning_rate"]),
        "--q-learning-rate", str(optimization["q_learning_rate"]),
        "--gamma", str(optimization["gamma"]),
        "--entropy-temperature", str(optimization["entropy_temperature"]),
        "--chi2-alpha", str(optimization["chi2_alpha"]),
        "--chi2-regularization", str(optimization["chi2_regularization"]),
        "--target-tau", str(optimization["target_tau"]),
        "--target-q-clip", str(optimization["target_q_clip"]),
        "--bc-coef", str(optimization["bc_coef"]),
        "--q-only-updates", str(optimization["q_only_updates"]),
        "--max-grad-norm", str(optimization["max_grad_norm"]),
        "--max-q-abs", str(optimization["max_q_abs"]),
        "--initial-log-std", str(optimization["initial_log_std"]),
        "--log-std-min", str(optimization["log_std_min"]),
        "--log-std-max", str(optimization["log_std_max"]),
        "--initial-policy-replay", str(replay["initial_policy_replay"]),
        "--policy-replay-capacity", str(replay["capacity"]),
        "--collect-steps", str(replay["collect_steps"]),
        "--collect-every", str(replay["collect_every"]),
        "--evaluation-episodes", str(evaluation["episodes"]),
        "--min-validation-skill", str(gates["min_validation_skill"]),
        "--max-initial-skill-regression", str(gates["max_initial_skill_regression"]),
        "--max-validation-mae", str(gates["max_validation_mae"]),
        "--learning-action-index", str(gates["learning_action_index"]),
        "--min-learning-action-std-ratio", str(gates["min_learning_action_std_ratio"]),
        "--min-learning-action-correlation", str(gates["min_learning_action_correlation"]),
        "--min-rollout-steps", str(evaluation["min_rollout_steps"]),
        "--max-crash-fraction", str(evaluation["max_crash_fraction"]),
        "--max-offroad-fraction", str(evaluation["max_offroad_fraction"]),
        "--max-collision-proxy-fraction", str(evaluation["max_collision_proxy_fraction"]),
        "--max-rollout-mean-length-regression", str(evaluation["max_mean_length_regression"]),
        "--max-rollout-fraction-regression", str(evaluation["max_fraction_regression"]),
        "--capability-failure-mode", capability_failure_mode,
    ]
    command.append("--transformer-norm-first" if architecture["transformer_norm_first"] else "--no-transformer-norm-first")
    command.append("--training-enable-collision" if replay["training_enable_collision"] else "--no-training-enable-collision")
    return command
