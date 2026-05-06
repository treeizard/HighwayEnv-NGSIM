from __future__ import annotations

import os
import sys
from typing import Any

import torch.nn as nn

from .config import PSGAILConfig

WANDB_PROJECT = "highwayenv-ps-gail"


class WandbMonitor:
    def __init__(self, cfg: PSGAILConfig, run_dir: str) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.enabled = str(cfg.wandb_mode).lower() != "disabled"
        self._wandb: Any | None = None
        self._run: Any | None = None

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
            import wandb
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "wandb is not installed. Install it with `pip install wandb`, "
                "or run with `--wandb-mode disabled`."
            ) from exc

        self._wandb = wandb
        mode = str(self.cfg.wandb_mode).lower()
        self.cfg.wandb_project = WANDB_PROJECT
        tags = [tag.strip() for tag in str(self.cfg.wandb_tags).split(",") if tag.strip()]
        settings = wandb.Settings(
            _service_wait=120,
            init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "30")),
        )
        init_kwargs = {
            "project": WANDB_PROJECT,
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

    def watch(self, policy: nn.Module, discriminator: nn.Module) -> None:
        if not self.enabled or not self.cfg.wandb_watch or self._wandb is None:
            return
        self._wandb.watch(
            [policy, discriminator],
            log="gradients",
            log_freq=max(1, int(self.cfg.checkpoint_every)),
        )

    def log(self, metrics: dict[str, float | int], *, step: int) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(metrics, step=int(step))

    def save(self, path: str) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.save(os.path.abspath(path), base_path=os.path.abspath(self.run_dir))

    def log_video(self, key: str, path: str, *, step: int, fps: int) -> None:
        if not self.enabled or self._wandb is None:
            return
        abs_path = os.path.abspath(path)
        self._wandb.log(
            {key: self._wandb.Video(abs_path, fps=max(1, int(fps)), format="mp4")},
            step=int(step),
        )
        self.save(abs_path)

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()
