"""Stateful training-health gates used by screening experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import PSGAILConfig

RECOVERABLE_HEALTH_REASONS = frozenset(
    {
        "discriminator_saturated",
        "target_kl_repeatedly_exceeded",
    }
)


@dataclass
class TrainingHealthMonitor:
    kl_violations: int = 0
    discriminator_saturation: int = 0
    reward_collapse: int = 0
    validation_regressions: int = 0

    def state_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

    def load_state_dict(self, state: dict[str, int]) -> None:
        required = set(asdict(self))
        missing = sorted(required.difference(state))
        if missing:
            raise RuntimeError(f"Training health state is incomplete: {missing}")
        for key in required:
            setattr(self, key, int(state[key]))

    def observe(
        self,
        cfg: PSGAILConfig,
        *,
        approx_kl: float,
        expert_accuracy: float,
        generator_accuracy: float,
        reward_std: float,
        action_std: float,
        extra_metrics: dict[str, float] | None = None,
    ) -> list[str]:
        values = {
            "approx_kl": approx_kl,
            "expert_accuracy": expert_accuracy,
            "generator_accuracy": generator_accuracy,
            "reward_std": reward_std,
            "action_std": action_std,
        }
        values.update(extra_metrics or {})
        nonfinite = [name for name, value in values.items() if not math.isfinite(float(value))]
        if nonfinite:
            return ["nonfinite:" + ",".join(nonfinite)]

        target_kl = max(0.0, float(getattr(cfg, "target_kl", 0.0)))
        self.kl_violations = (
            self.kl_violations + 1
            if target_kl > 0.0 and float(approx_kl) > target_kl
            else 0
        )
        self.discriminator_saturation = (
            self.discriminator_saturation + 1
            if float(expert_accuracy) > 0.95 and float(generator_accuracy) > 0.95
            else 0
        )
        self.reward_collapse = (
            self.reward_collapse + 1
            if float(reward_std) < float(getattr(cfg, "health_min_reward_std", 1.0e-3))
            else 0
        )
        reasons = []
        if self.kl_violations >= max(1, int(getattr(cfg, "health_kl_patience", 2))):
            reasons.append("target_kl_repeatedly_exceeded")
        if self.discriminator_saturation >= max(
            1, int(getattr(cfg, "health_discriminator_patience", 5))
        ):
            reasons.append("discriminator_saturated")
        if self.reward_collapse >= max(1, int(getattr(cfg, "health_reward_std_patience", 5))):
            reasons.append("adversarial_reward_collapsed")
        if float(action_std) < float(getattr(cfg, "health_min_action_std", 1.0e-3)):
            reasons.append("policy_action_variance_collapsed")
        return reasons

    def observe_validation(
        self,
        cfg: PSGAILConfig,
        *,
        score: float,
        best_score: float,
    ) -> list[str]:
        if not math.isfinite(float(score)):
            return ["nonfinite:validation_score"]
        max_drop = float(getattr(cfg, "validation_max_score_drop", 0.0))
        patience = int(getattr(cfg, "validation_regression_patience", 0))
        if max_drop <= 0.0 or patience <= 0 or not math.isfinite(float(best_score)):
            self.validation_regressions = 0
            return []
        self.validation_regressions = (
            self.validation_regressions + 1
            if float(score) < float(best_score) - max_drop
            else 0
        )
        if self.validation_regressions >= patience:
            return ["validation_score_repeatedly_regressed"]
        return []

    def observe_learning(
        self,
        cfg: PSGAILConfig,
        *,
        round_idx: int,
        initial_score: float,
        best_score: float,
        best_round: int,
    ) -> list[str]:
        """Require measurable post-initialization improvement at a fixed round."""
        gate_round = int(getattr(cfg, "health_learning_gate_round", 0))
        if gate_round <= 0 or int(round_idx) < gate_round:
            return []
        if not math.isfinite(float(initial_score)) or not math.isfinite(float(best_score)):
            return ["nonfinite:learning_gate_score"]
        minimum_best_round = max(1, int(getattr(cfg, "health_min_best_round", 1)))
        if int(best_round) < minimum_best_round:
            return ["no_post_initialization_best_checkpoint"]
        initial_cost = -float(initial_score)
        best_cost = -float(best_score)
        denominator = max(abs(initial_cost), 1.0e-12)
        relative_improvement = (initial_cost - best_cost) / denominator
        minimum_improvement = max(
            0.0,
            float(getattr(cfg, "health_min_relative_validation_improvement", 0.0)),
        )
        if relative_improvement + 1.0e-12 < minimum_improvement:
            return ["validation_did_not_meet_learning_improvement_gate"]
        return []


def partition_health_reasons(reasons: list[str]) -> tuple[list[str], list[str]]:
    """Separate fatal numerical/policy failures from adversarial diagnostics.

    A highly accurate discriminator is evidence that expert and generator
    distributions remain separable; by itself it is not evidence that
    optimization is numerically invalid or that the policy has collapsed. A
    finite repeated target-KL excess is likewise an update-size diagnostic:
    PPO already stops the remaining optimizer epochs for that update. Both
    conditions are recorded as warnings and allowed to recover. Non-finite
    values and the remaining policy/learning gates retain fail-closed
    behaviour.
    """

    warnings = [
        str(reason) for reason in reasons if str(reason) in RECOVERABLE_HEALTH_REASONS
    ]
    fatal = [
        str(reason) for reason in reasons if str(reason) not in RECOVERABLE_HEALTH_REASONS
    ]
    return fatal, warnings


def health_warning_metrics(
    monitor: TrainingHealthMonitor,
    warnings: list[str],
) -> dict[str, int]:
    """Return durable audit metrics for recoverable health conditions."""

    return {
        "health/discriminator_saturation_warning": int(
            "discriminator_saturated" in warnings
        ),
        "health/target_kl_warning": int(
            "target_kl_repeatedly_exceeded" in warnings
        ),
        "health/target_kl_consecutive_violations": int(monitor.kl_violations),
    }


__all__ = [
    "RECOVERABLE_HEALTH_REASONS",
    "TrainingHealthMonitor",
    "health_warning_metrics",
    "partition_health_reasons",
]
