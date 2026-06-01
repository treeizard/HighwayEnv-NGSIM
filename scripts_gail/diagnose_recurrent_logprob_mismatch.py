"""Diagnose recurrent-policy log-prob differences with and without memory.

This script is intentionally synthetic: it isolates the policy likelihood
calculation used by AIRL from the environment so it can run quickly.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.distributions import Normal

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts_gail.ps_gail.models import make_actor_critic


class SquashedNormal:
    """Local copy of the training distribution to avoid importing Gym."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1.0e-6) -> None:
        self.normal = Normal(mean, std)
        self.eps = float(eps)

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.normal.sample())

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(actions, -1.0 + self.eps, 1.0 - self.eps)
        raw = torch.atanh(clipped)
        correction = torch.log1p(-clipped.pow(2) + self.eps)
        return (self.normal.log_prob(raw) - correction).sum(dim=-1)


def policy_dist(policy: torch.nn.Module, policy_out: torch.Tensor) -> SquashedNormal:
    log_std = getattr(policy, "log_std", None)
    if log_std is None:
        raise RuntimeError("Expected continuous policy with log_std.")
    return SquashedNormal(policy_out, torch.exp(log_std).expand_as(policy_out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--obs-dim", type=int, default=12)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    policy = make_actor_critic(
        "recurrent_transformer",
        obs_dim=int(args.obs_dim),
        hidden_size=32,
        action_mode="continuous",
        continuous_action_dim=2,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        transformer_memory_tokens=4,
        transformer_memory_context_length=4,
        transformer_use_causal_attention=True,
    ).eval()

    obs = torch.randn(int(args.steps), int(args.obs_dim))
    memory = policy.initial_memory(1, dtype=torch.float32)

    memory_logps: list[float] = []
    blank_logps: list[float] = []
    memory_abs_sums: list[float] = []

    for step in range(int(args.steps)):
        step_obs = obs[step : step + 1]
        policy_out_with_memory, _values, step_memory = policy(
            step_obs,
            memory=memory,
            return_memory=True,
        )
        dist_with_memory = policy_dist(policy, policy_out_with_memory)

        action = dist_with_memory.sample()
        memory_logps.append(float(dist_with_memory.log_prob(action).detach().item()))
        memory_abs_sums.append(float(memory.abs().sum().detach().item()))

        # This is the same effective path used by
        # train_simple_airl._policy_log_probs: recurrent policy forward with
        # memory=None, so the model constructs blank initial memory internally.
        policy_out_blank, _values_blank = policy(step_obs)
        dist_blank = policy_dist(policy, policy_out_blank)
        blank_logps.append(float(dist_blank.log_prob(action).detach().item()))

        memory = torch.cat([memory[:, 1:], step_memory.unsqueeze(1)], dim=1)

    memory_logps_np = np.asarray(memory_logps, dtype=np.float64)
    blank_logps_np = np.asarray(blank_logps, dtype=np.float64)
    delta = blank_logps_np - memory_logps_np

    print("policy_model=recurrent_transformer")
    print("dropout=0.0")
    print("policy_eval_mode=True")
    print("airl_helper_path=policy_distribution_and_values(policy, obs, cfg, None)")
    print("gail_optional_bc_path=policy(obs_tensor) with no memory")
    print("step,memory_abs_sum_before,logp_with_memory,logp_blank_memory,blank_minus_memory")
    for step, (memory_abs, memory_logp, blank_logp, diff) in enumerate(
        zip(memory_abs_sums, memory_logps_np, blank_logps_np, delta)
    ):
        print(f"{step},{memory_abs:.6f},{memory_logp:.6f},{blank_logp:.6f},{diff:+.6f}")

    noninitial = delta[1:] if delta.size > 1 else delta
    print(
        "summary_noninitial_steps="
        f"mean_abs_delta={float(np.mean(np.abs(noninitial))):.6f},"
        f"max_abs_delta={float(np.max(np.abs(noninitial))):.6f},"
        f"std_delta={float(np.std(noninitial)):.6f}"
    )


if __name__ == "__main__":
    main()
