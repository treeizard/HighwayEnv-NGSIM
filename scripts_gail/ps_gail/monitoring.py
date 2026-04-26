from __future__ import annotations

import os
from typing import Any

import torch.nn as nn

from .config import PSGAILConfig


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
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "wandb is not installed. Install it with `pip install wandb`, "
                "or run with `--wandb-mode disabled`."
            ) from exc

        self._wandb = wandb
        mode = str(self.cfg.wandb_mode).lower()
        tags = [tag.strip() for tag in str(self.cfg.wandb_tags).split(",") if tag.strip()]
        self._run = wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity or None,
            group=self.cfg.wandb_group or None,
            name=self.cfg.run_name,
            dir=os.path.abspath(self.run_dir),
            mode=mode,
            tags=tags or None,
            config=vars(self.cfg),
        )

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

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()
