from __future__ import annotations

import torch
import torch.nn as nn


NUM_DISCRETE_META_ACTIONS = 5


class SharedActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        num_actions: int = NUM_DISCRETE_META_ACTIONS,
        *,
        action_mode: str = "discrete",
        continuous_action_dim: int = 2,
    ) -> None:
        super().__init__()
        self.action_mode = str(action_mode).lower()
        self.continuous_action_dim = int(continuous_action_dim)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        if self.action_mode == "continuous":
            self.policy_head = nn.Linear(hidden_size, self.continuous_action_dim)
            self.log_std = nn.Parameter(torch.full((self.continuous_action_dim,), -0.5))
        elif self.action_mode == "discrete":
            self.policy_head = nn.Linear(hidden_size, num_actions)
            self.log_std = None
        else:
            raise ValueError(f"Unsupported action_mode={action_mode!r}.")
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        policy_out = self.policy_head(encoded)
        if self.action_mode == "continuous":
            policy_out = torch.tanh(policy_out)
        return policy_out, self.value_head(encoded).squeeze(-1)


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
