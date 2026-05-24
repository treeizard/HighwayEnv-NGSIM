"""Torch distribution, device, and optimizer helpers for PS-GAIL training."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import OrderedDict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env
from highway_env.ngsim_utils.core.constants import MAX_ACCEL
from highway_env.ngsim_utils.data.prebuilt import load_prebuilt_data

from ..config import PSGAILConfig
from ..data import (
    SCENE_FEATURE_DIM_PER_VEHICLE,
    build_sequence_windows,
    discriminator_features,
    normalize_trajectory_frame,
    scene_snapshot_features,
    standardize_features,
    transform_sequence_features,
)
from ..models import NUM_DISCRETE_META_ACTIONS, make_actor_critic
from ..observations import flatten_agent_observations, policy_observations_from_flat

class SquashedNormal:
    """Tanh-squashed diagonal Gaussian over bounded continuous actions."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        self.normal = Normal(mean, std)
        self.eps = float(eps)

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.normal.sample())

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(actions, -1.0 + self.eps, 1.0 - self.eps)
        raw = torch.atanh(clipped)
        correction = torch.log1p(-clipped.pow(2) + self.eps)
        return (self.normal.log_prob(raw) - correction).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # The tanh-transformed entropy has no simple closed form. The base
        # entropy is a stable proxy for PPO's entropy bonus and diagnostics.
        return self.normal.entropy().sum(dim=-1)

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    learning_rate = float(learning_rate)
    for group in optimizer.param_groups:
        if float(group.get("lr", learning_rate)) != learning_rate:
            group["lr"] = learning_rate

def _as_device_tensor(array: np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=dtype)
    if device.type == "cuda":
        tensor = tensor.pin_memory().to(device=device, non_blocking=True)
    else:
        tensor = tensor.to(device=device)
    return tensor

__all__ = [
    'SquashedNormal',
    'resolve_device',
    'set_optimizer_lr',
    '_as_device_tensor'
]
