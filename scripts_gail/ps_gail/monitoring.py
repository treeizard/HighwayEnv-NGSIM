"""Support PS-GAIL imitation-learning experiments on NGSIM environments."""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import torch.nn as nn

from .config import PSGAILConfig

WANDB_PROJECT = "highwayenv-ps-gail"


class WandbMonitor:
    def __init__(self, cfg: PSGAILConfig, run_dir: str, *, trainer: str = "") -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.trainer = str(trainer or "").strip().lower()
        self.enabled = str(cfg.wandb_mode).lower() != "disabled"
        self._wandb: Any | None = None
        self._run: Any | None = None
        self._effective_mode = "disabled"
        self._pending_step: int | None = None
        self._pending_metrics: dict[str, float | int] = {}
        self.metrics_path = os.path.join(os.path.abspath(self.run_dir), "metrics.jsonl")

    def start(self) -> None:
        if not self.enabled:
            return
        wandb_root = os.path.abspath(os.path.join(self.run_dir, "wandb"))
        for name, relative in {
            "WANDB_DIR": ".",
            "WANDB_CONFIG_DIR": "config",
            "WANDB_CACHE_DIR": "cache",
            "WANDB_DATA_DIR": "data",
            "WANDB_ARTIFACT_DIR": "artifacts",
            "TMPDIR": "tmp",
            "TEMP": "tmp",
            "TMP": "tmp",
        }.items():
            path = os.path.abspath(os.path.join(wandb_root, relative))
            os.makedirs(path, exist_ok=True)
            os.environ.setdefault(name, path)
        try:
            import numpy as np

            if not hasattr(np, "float_"):
                np.float_ = np.float64  # type: ignore[attr-defined]
            if not hasattr(np, "complex_"):
                np.complex_ = np.complex128  # type: ignore[attr-defined]
            if not hasattr(np, "string_"):
                np.string_ = np.bytes_  # type: ignore[attr-defined]
            if not hasattr(np, "unicode_"):
                np.unicode_ = np.str_  # type: ignore[attr-defined]
            if not hasattr(np, "Inf"):
                np.Inf = np.inf  # type: ignore[attr-defined]
            import wandb
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "wandb is not installed. Install it with `pip install wandb`, "
                "or run with `--wandb-mode disabled`."
            ) from exc

        self._wandb = wandb
        mode = str(self.cfg.wandb_mode).lower()
        project = str(self.cfg.wandb_project or WANDB_PROJECT).strip() or WANDB_PROJECT
        tags = [tag.strip() for tag in str(self.cfg.wandb_tags).split(",") if tag.strip()]
        settings = wandb.Settings(
            _service_wait=120,
            init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "30")),
        )
        init_kwargs = {
            "project": project,
            "entity": self.cfg.wandb_entity or None,
            "group": self.cfg.wandb_group or None,
            "name": self.cfg.run_name,
            "dir": os.path.abspath(self.run_dir),
            "tags": tags or None,
            "config": vars(self.cfg),
            "settings": settings,
        }
        try:
            self._run = wandb.init(mode=mode, **init_kwargs)
            self._effective_mode = mode
        except Exception as exc:
            comm_error = getattr(getattr(wandb, "errors", None), "CommError", None)
            is_comm_error = (
                isinstance(exc, comm_error)
                if comm_error is not None
                else exc.__class__.__name__ == "CommError"
            )
            if not is_comm_error or mode in {"offline", "disabled"}:
                raise
            print(
                "wandb online initialization failed; falling back to local offline W&B logging "
                f"under {os.path.abspath(self.run_dir)!r}. "
                "Sync later with `wandb sync` if needed.",
                file=sys.stderr,
                flush=True,
            )
            self._run = wandb.init(mode="offline", **init_kwargs)
            self._effective_mode = "offline"
        self._write_wandb_identity(project=project, requested_mode=mode)

    def _write_wandb_identity(self, *, project: str, requested_mode: str) -> None:
        run = self._run
        payload = {
            "schema_version": 1,
            "requested_mode": str(requested_mode),
            "effective_mode": str(self._effective_mode),
            "project": str(project),
            "entity": str(self.cfg.wandb_entity or ""),
            "group": str(self.cfg.wandb_group or ""),
            "name": str(self.cfg.run_name),
            "run_id": str(getattr(run, "id", "") or ""),
            "url": str(getattr(run, "url", "") or ""),
        }
        path = os.path.join(os.path.abspath(self.run_dir), "wandb_run.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    def watch(self, policy: nn.Module, discriminator: nn.Module) -> None:
        if not self.enabled or not self.cfg.wandb_watch or self._wandb is None:
            return
        self._wandb.watch(
            [policy, discriminator],
            log="gradients",
            log_freq=max(1, int(self.cfg.checkpoint_every)),
        )

    @staticmethod
    def _finite_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
        return {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }

    def _compact_metrics(self, metrics: dict[str, float | int]) -> dict[str, float | int]:
        metrics = self._finite_metrics(metrics)
        if not bool(getattr(self.cfg, "wandb_compact_metrics", True)):
            return metrics
        compact: dict[str, float | int] = {}
        direct = {
            "rollout/agent_steps",
            "rollout/training_agent_steps",
            "rollout/env_steps",
            "rollout/episodes",
            "rollout/mean_episode_length",
            "rollout/acceleration_action_mean",
            "rollout/acceleration_action_std",
            "rollout/steering_action_mean",
            "rollout/steering_action_std",
            "policy/loss",
            "policy/value_loss",
            "policy/entropy",
            "policy/approx_kl",
            "policy/post_update_approx_kl",
            "policy/clip_fraction",
            "policy/ppo_epochs_completed",
            "policy/ppo_early_stopped_kl",
            "policy/action_std_param_mean",
            "train/policy_learning_rate",
            "train/disc_learning_rate",
            "train/reward_learning_rate",
            "train/entropy_coef",
            "train/clip_range",
            "perf/round_seconds",
            "perf/collect_rollouts_seconds",
            "perf/policy_update_seconds",
            "perf/cuda_max_memory_mb",
            "rollout/env_steps_per_second",
            "rollout/agent_steps_per_second",
        }
        aliases = {
            "rollout/mean_controlled_vehicles": "rollout/actual_controlled_vehicles",
            "rollout/crash_agent_fraction": "rollout/crash_transition_fraction",
            "rollout/offroad_agent_fraction": "rollout/offroad_transition_fraction",
            "rollout/crash_episodes": "rollout/crash_episode_count",
            "rollout/offroad_episodes": "rollout/offroad_episode_count",
            "rollout/mean_reward": "reward/training_mean",
            "rollout/reward_std": "reward/training_std",
            "rollout/mean_env_penalty": "reward/environment_penalty_mean",
            "rollout/mean_raw_gail_reward": "reward/adversarial_raw_mean",
            "rollout/raw_gail_reward_std": "reward/adversarial_raw_std",
            "rollout/mean_normalized_gail_reward": "reward/adversarial_policy_mean",
            "rollout/normalized_gail_reward_std": "reward/adversarial_policy_std",
            "rollout/mean_raw_airl_reward": "reward/adversarial_raw_mean",
            "rollout/raw_airl_reward_std": "reward/adversarial_raw_std",
            "rollout/mean_normalized_airl_reward": "reward/adversarial_policy_mean",
            "rollout/normalized_airl_reward_std": "reward/adversarial_policy_std",
        }
        for key in direct:
            if key in metrics:
                compact[key] = metrics[key]
        for source, target in aliases.items():
            if source in metrics:
                compact[target] = metrics[source]
        requested = metrics.get("rollout/controlled_vehicle_fraction")
        if requested is not None:
            target = (
                "rollout/requested_controlled_fraction"
                if abs(float(requested)) <= 1.0
                else "rollout/requested_controlled_vehicles"
            )
            compact[target] = requested

        loss_type = str(getattr(self.cfg, "discriminator_loss", "")).lower()
        if self.trainer == "airl":
            airl_names = (
                ("airl/reward_loss", "reward_model/loss"),
                ("airl/expert_reward", "reward_model/expert_reward_mean"),
                ("airl/gen_reward", "reward_model/generator_reward_mean"),
            )
            objective_names = (
                (("airl/bce_loss", "reward_model/bce_loss"),
                 ("airl/expert_acc", "reward_model/expert_accuracy"),
                 ("airl/gen_acc", "reward_model/generator_accuracy"))
                if loss_type in {"bce", "airl_bce"}
                else (("airl/wgan_loss", "reward_model/wgan_loss"),
                      ("airl/gradient_penalty", "reward_model/gradient_penalty"),
                      ("airl/critic_gap", "reward_model/critic_gap"))
            )
            for source, target in (*airl_names, *objective_names):
                if source in metrics:
                    compact[target] = metrics[source]
            for key in (
                "perf/airl_reward_total_seconds",
                "perf/airl_log_prob_seconds",
                "perf/airl_reward_update_seconds",
            ):
                if key in metrics:
                    compact[key] = metrics[key]
        else:
            common_disc = (
                ("discriminator/loss", "discriminator/loss"),
                ("discriminator/critic_gap", "discriminator/critic_gap"),
            )
            objective_disc = (
                (("discriminator/bce_loss", "discriminator/bce_loss"),
                 ("discriminator/expert_prob_mean", "discriminator/expert_probability"),
                 ("discriminator/gen_prob_mean", "discriminator/generator_probability"),
                 ("discriminator/expert_acc", "discriminator/expert_accuracy"),
                 ("discriminator/gen_acc", "discriminator/generator_accuracy"))
                if loss_type in {"bce", "airl_bce"}
                else (("discriminator/wgan_loss", "discriminator/wgan_loss"),
                      ("discriminator/gradient_penalty", "discriminator/gradient_penalty"),
                      ("discriminator/expert_score_mean", "discriminator/expert_score"),
                      ("discriminator/gen_score_mean", "discriminator/generator_score"))
            )
            for source, target in (*common_disc, *objective_disc):
                if source in metrics:
                    compact[target] = metrics[source]

        for prefix in (
            "validation",
            "validation_stress",
            "selected_validation",
            "initializer_test",
            "test",
        ):
            for suffix in (
                "score",
                "cost",
                "horizon_coverage_20s",
                "rmse_position_20s",
                "rmse_speed_20s",
                "rmse_lane_offset_20s",
                "vehicle_crash_rate",
                "vehicle_offroad_rate",
                "hard_brake_rate",
                "episodes",
                "vehicle_episodes",
                "mean_episode_length",
                "acceleration_action_std",
                "steering_action_std",
            ):
                key = f"{prefix}/{suffix}"
                if key in metrics:
                    compact[key] = metrics[key]
        return compact

    def log(self, metrics: dict[str, float | int], *, step: int) -> None:
        step = int(step)
        if self._pending_step is not None and step != self._pending_step:
            self.flush()
        if self._pending_step is None:
            self._pending_step = step
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._pending_metrics[str(key)] = value

    def flush(self) -> None:
        if self._pending_step is None:
            return
        record: dict[str, object] = {"step": int(self._pending_step)}
        for key, value in self._pending_metrics.items():
            record[key] = value if math.isfinite(float(value)) else None
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.enabled and self._wandb is not None:
            compact = self._compact_metrics(self._pending_metrics)
            if compact:
                self._wandb.log(compact, step=int(self._pending_step))
        self._pending_step = None
        self._pending_metrics = {}

    def save(self, path: str) -> None:
        if not self.enabled or self._wandb is None:
            return
        basename = os.path.basename(path)
        if basename not in {
            "best.pt",
            "final.pt",
            "evaluation_summary.json",
            "training_failure.json",
            "run_manifest.json",
            "wandb_run.json",
        } and not basename.endswith(".sha256"):
            return
        self._wandb.save(os.path.abspath(path), base_path=os.path.abspath(self.run_dir))

    def log_video(self, key: str, path: str, *, step: int, fps: int) -> None:
        self.flush()
        if not self.enabled or self._wandb is None:
            return
        abs_path = os.path.abspath(path)
        self._wandb.log(
            {key: self._wandb.Video(abs_path, fps=max(1, int(fps)), format="mp4")},
            step=int(step),
        )

    def finish(self) -> None:
        self.flush()
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()
