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


class TransformerActorCritic(nn.Module):
    """Small transformer policy over the flattened policy-observation feature sequence."""

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        num_actions: int = NUM_DISCRETE_META_ACTIONS,
        *,
        action_mode: str = "discrete",
        continuous_action_dim: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_mode = str(action_mode).lower()
        self.continuous_action_dim = int(continuous_action_dim)
        obs_dim = int(obs_dim)
        hidden_size = int(hidden_size)
        num_heads = max(1, int(num_heads))
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by transformer_heads={num_heads}."
            )
        self.input_proj = nn.Linear(1, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, obs_dim + 1, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(num_layers)),
            norm=nn.LayerNorm(hidden_size),
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
        tokens = self.input_proj(obs.unsqueeze(-1))
        cls = self.cls_token.expand(obs.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)[:, 0]
        policy_out = self.policy_head(encoded)
        if self.action_mode == "continuous":
            policy_out = torch.tanh(policy_out)
        return policy_out, self.value_head(encoded).squeeze(-1)


def make_actor_critic(
    policy_model: str,
    obs_dim: int,
    hidden_size: int,
    *,
    action_mode: str = "discrete",
    continuous_action_dim: int = 2,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    transformer_dropout: float = 0.1,
) -> nn.Module:
    model_name = str(policy_model).lower()
    if model_name == "mlp":
        return SharedActorCritic(
            obs_dim,
            hidden_size,
            action_mode=action_mode,
            continuous_action_dim=continuous_action_dim,
        )
    if model_name == "transformer":
        return TransformerActorCritic(
            obs_dim,
            hidden_size,
            action_mode=action_mode,
            continuous_action_dim=continuous_action_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
        )
    raise ValueError(f"Unsupported policy_model={policy_model!r}. Expected 'mlp' or 'transformer'.")


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


class SceneDiscriminator(TrajectoryDiscriminator):
    """MLP discriminator for one full-road traffic snapshot."""


class SequenceTrajectoryDiscriminator(nn.Module):
    """Autoregressive discriminator over fixed-length trajectory feature windows."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # WGAN-GP differentiates through the discriminator input gradients.
        # CuDNN RNN kernels do not support that double-backward path, so use
        # PyTorch's differentiable GRU implementation for this discriminator.
        if features.requires_grad:
            with torch.backends.cudnn.flags(enabled=False):
                _sequence, hidden = self.encoder(features)
        else:
            _sequence, hidden = self.encoder(features)
        return self.head(hidden[-1]).squeeze(-1)
