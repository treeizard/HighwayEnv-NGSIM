from __future__ import annotations

import torch
import torch.nn as nn


NUM_DISCRETE_META_ACTIONS = 5
DEFAULT_CRITIC_HIDDEN_SIZES = (128, 128, 64)
DEFAULT_CRITIC_DROPOUT = 0.0
CENTRAL_CRITIC_VEHICLE_FEATURE_DIM = 5
CENTRAL_CRITIC_CONTEXT_DIM = 4


def parse_hidden_sizes(hidden_sizes: str | int | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if hidden_sizes is None:
        return DEFAULT_CRITIC_HIDDEN_SIZES
    if isinstance(hidden_sizes, int):
        return (int(hidden_sizes), int(hidden_sizes))
    if isinstance(hidden_sizes, str):
        values = tuple(int(value.strip()) for value in hidden_sizes.split(",") if value.strip())
    else:
        values = tuple(int(value) for value in hidden_sizes)
    if not values:
        raise ValueError("At least one critic hidden size is required.")
    if any(value <= 0 for value in values):
        raise ValueError(f"Critic hidden sizes must be positive, got {values!r}.")
    return values


def make_relu_mlp(
    input_dim: int,
    hidden_sizes: str | int | tuple[int, ...] | list[int] | None,
    output_dim: int,
    *,
    dropout: float = DEFAULT_CRITIC_DROPOUT,
    spectral_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    dropout = float(dropout)
    for hidden_dim in parse_hidden_sizes(hidden_sizes):
        linear = nn.Linear(in_dim, int(hidden_dim))
        layers.append(nn.utils.spectral_norm(linear) if spectral_norm else linear)
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        in_dim = int(hidden_dim)
    linear = nn.Linear(in_dim, int(output_dim))
    layers.append(nn.utils.spectral_norm(linear) if spectral_norm else linear)
    return nn.Sequential(*layers)


class AttentionCriticEncoder(nn.Module):
    """Attention-pool fixed-slot vehicle features for a centralized traffic critic."""

    def __init__(
        self,
        critic_obs_dim: int,
        hidden_size: int,
        *,
        max_vehicles: int,
        vehicle_feature_dim: int = CENTRAL_CRITIC_VEHICLE_FEATURE_DIM,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.critic_obs_dim = int(critic_obs_dim)
        self.max_vehicles = max(1, int(max_vehicles))
        self.vehicle_feature_dim = max(1, int(vehicle_feature_dim))
        self.scene_dim = self.max_vehicles * self.vehicle_feature_dim
        if self.critic_obs_dim < self.scene_dim:
            raise ValueError(
                "Attention critic expects critic_obs_dim to include "
                f"{self.scene_dim} scene features, got {self.critic_obs_dim}."
            )
        self.tail_dim = max(1, self.critic_obs_dim - self.scene_dim)
        attention_heads = max(1, int(attention_heads))
        if int(hidden_size) % attention_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by central_critic_attention_heads={attention_heads}."
            )
        self.vehicle_proj = nn.Sequential(
            nn.Linear(self.vehicle_feature_dim, hidden_size),
            nn.Tanh(),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(self.tail_dim, hidden_size),
            nn.Tanh(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=int(hidden_size),
            num_heads=attention_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(int(hidden_size))
        self.out = nn.Sequential(
            nn.Linear(int(hidden_size) * 2, int(hidden_size)),
            nn.Tanh(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.Tanh(),
        )

    def forward(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.ndim != 2 or critic_obs.shape[1] != self.critic_obs_dim:
            raise ValueError(
                f"Attention critic expected shape [B, {self.critic_obs_dim}], got {tuple(critic_obs.shape)}."
            )
        scene = critic_obs[:, : self.scene_dim].reshape(
            critic_obs.shape[0],
            self.max_vehicles,
            self.vehicle_feature_dim,
        )
        tail = critic_obs[:, self.scene_dim :]
        if tail.shape[1] == 0:
            tail = torch.zeros((critic_obs.shape[0], self.tail_dim), dtype=critic_obs.dtype, device=critic_obs.device)
        vehicle_tokens = self.vehicle_proj(scene)
        query = self.query_proj(tail).unsqueeze(1)
        key_padding_mask = scene[:, :, 0] <= 0.5
        if key_padding_mask.any():
            all_padded = key_padding_mask.all(dim=1)
            if all_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded, 0] = False
        pooled, _weights = self.attention(
            query,
            vehicle_tokens,
            vehicle_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.norm(query + pooled).squeeze(1)
        return self.out(torch.cat([query.squeeze(1), pooled], dim=-1))


def make_centralized_critic_encoder(
    critic_obs_dim: int,
    hidden_size: int,
    *,
    pooling: str = "flat",
    max_vehicles: int = 64,
    attention_heads: int = 4,
) -> nn.Module:
    pooling = str(pooling).lower()
    if pooling in {"flat", "mlp"}:
        return nn.Sequential(
            nn.Linear(int(critic_obs_dim), int(hidden_size)),
            nn.Tanh(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.Tanh(),
        )
    if pooling in {"attention", "attn"}:
        return AttentionCriticEncoder(
            int(critic_obs_dim),
            int(hidden_size),
            max_vehicles=int(max_vehicles),
            attention_heads=int(attention_heads),
        )
    raise ValueError(f"Unsupported central_critic_pooling={pooling!r}. Expected 'flat' or 'attention'.")


class SharedActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        num_actions: int = NUM_DISCRETE_META_ACTIONS,
        *,
        action_mode: str = "discrete",
        continuous_action_dim: int = 2,
        centralized_critic: bool = False,
        critic_obs_dim: int | None = None,
        central_critic_pooling: str = "flat",
        central_critic_max_vehicles: int = 64,
        central_critic_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.action_mode = str(action_mode).lower()
        self.continuous_action_dim = int(continuous_action_dim)
        self.centralized_critic = bool(centralized_critic)
        self.critic_obs_dim = int(critic_obs_dim if critic_obs_dim is not None else obs_dim)
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
        self.critic_encoder = (
            make_centralized_critic_encoder(
                self.critic_obs_dim,
                hidden_size,
                pooling=central_critic_pooling,
                max_vehicles=central_critic_max_vehicles,
                attention_heads=central_critic_attention_heads,
            )
            if self.centralized_critic
            else None
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def _encode_actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_actor(obs)
        policy_out = self.policy_head(encoded)
        if self.action_mode == "continuous":
            policy_out = torch.tanh(policy_out)
        return policy_out

    def value(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
        *,
        encoded_actor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.centralized_critic:
            if critic_obs is None:
                critic_obs = torch.zeros(
                    (obs.shape[0], self.critic_obs_dim),
                    dtype=obs.dtype,
                    device=obs.device,
                )
            if self.critic_encoder is None:
                raise RuntimeError("Centralized critic is enabled without a critic encoder.")
            encoded = self.critic_encoder(critic_obs)
        else:
            encoded = encoded_actor if encoded_actor is not None else self._encode_actor(obs)
        return self.value_head(encoded).squeeze(-1)

    def forward(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encode_actor(obs)
        policy_out = self.policy_head(encoded)
        if self.action_mode == "continuous":
            policy_out = torch.tanh(policy_out)
        return policy_out, self.value(obs, critic_obs, encoded_actor=encoded)


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
        temporal_module: bool = False,
        temporal_kernel_size: int = 5,
        temporal_layers: int = 1,
        centralized_critic: bool = False,
        critic_obs_dim: int | None = None,
        central_critic_pooling: str = "flat",
        central_critic_max_vehicles: int = 64,
        central_critic_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.action_mode = str(action_mode).lower()
        self.continuous_action_dim = int(continuous_action_dim)
        obs_dim = int(obs_dim)
        hidden_size = int(hidden_size)
        self.centralized_critic = bool(centralized_critic)
        self.critic_obs_dim = int(critic_obs_dim if critic_obs_dim is not None else obs_dim)
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
        temporal_blocks: list[nn.Module] = []
        if bool(temporal_module):
            kernel_size = max(1, int(temporal_kernel_size))
            if kernel_size % 2 == 0:
                kernel_size += 1
            for _ in range(max(1, int(temporal_layers))):
                temporal_blocks.extend(
                    [
                        nn.Conv1d(
                            hidden_size,
                            hidden_size,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            groups=hidden_size,
                        ),
                        nn.GELU(),
                        nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                        nn.Dropout(p=float(dropout)),
                    ]
                )
        self.temporal_mixer = nn.Sequential(*temporal_blocks) if temporal_blocks else None
        if self.action_mode == "continuous":
            self.policy_head = nn.Linear(hidden_size, self.continuous_action_dim)
            self.log_std = nn.Parameter(torch.full((self.continuous_action_dim,), -0.5))
        elif self.action_mode == "discrete":
            self.policy_head = nn.Linear(hidden_size, num_actions)
            self.log_std = None
        else:
            raise ValueError(f"Unsupported action_mode={action_mode!r}.")
        self.critic_encoder = (
            make_centralized_critic_encoder(
                self.critic_obs_dim,
                hidden_size,
                pooling=central_critic_pooling,
                max_vehicles=central_critic_max_vehicles,
                attention_heads=central_critic_attention_heads,
            )
            if self.centralized_critic
            else None
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def _encode_actor(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(obs.unsqueeze(-1))
        if self.temporal_mixer is not None:
            mixed = self.temporal_mixer(tokens.transpose(1, 2)).transpose(1, 2)
            tokens = tokens + mixed
        cls = self.cls_token.expand(obs.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1]]
        return self.encoder(tokens)[:, 0]

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_actor(obs)
        policy_out = self.policy_head(encoded)
        if self.action_mode == "continuous":
            policy_out = torch.tanh(policy_out)
        return policy_out

    def value(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
        *,
        encoded_actor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.centralized_critic:
            if critic_obs is None:
                critic_obs = torch.zeros(
                    (obs.shape[0], self.critic_obs_dim),
                    dtype=obs.dtype,
                    device=obs.device,
                )
            if self.critic_encoder is None:
                raise RuntimeError("Centralized critic is enabled without a critic encoder.")
            encoded = self.critic_encoder(critic_obs)
        else:
            encoded = encoded_actor if encoded_actor is not None else self._encode_actor(obs)
        return self.value_head(encoded).squeeze(-1)

    def forward(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._encode_actor(obs)
        policy_out = self.policy_head(encoded)
        if self.action_mode == "continuous":
            policy_out = torch.tanh(policy_out)
        return policy_out, self.value(obs, critic_obs, encoded_actor=encoded)


class RecurrentTransformerActorCritic(nn.Module):
    """Attention policy over current sensor tokens plus bounded per-agent memory.

    The memory interface intentionally stores multiple transformer tokens rather
    than one recurrent hidden vector. Rollout code owns the per-vehicle cache and
    passes it back into the model on the next decision.
    """

    supports_recurrent_memory = True

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
        memory_tokens: int = 8,
        memory_context_length: int = 32,
        use_causal_attention: bool = True,
        centralized_critic: bool = False,
        critic_obs_dim: int | None = None,
        central_critic_pooling: str = "flat",
        central_critic_max_vehicles: int = 64,
        central_critic_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.action_mode = str(action_mode).lower()
        self.continuous_action_dim = int(continuous_action_dim)
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.memory_tokens = max(1, int(memory_tokens))
        self.memory_context_length = max(1, int(memory_context_length))
        self.use_causal_attention = bool(use_causal_attention)
        self.centralized_critic = bool(centralized_critic)
        self.critic_obs_dim = int(critic_obs_dim if critic_obs_dim is not None else obs_dim)

        num_heads = max(1, int(num_heads))
        if self.hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by transformer_heads={num_heads}."
            )

        self.ego_dim = 3
        self.lane_camera_cells = 21
        self.lane_feature_dim = 3
        lane_flat_dim = self.lane_camera_cells * self.lane_feature_dim
        sensor_dim = self.obs_dim - self.ego_dim
        lidar_flat_dim = sensor_dim - lane_flat_dim
        self.semantic_tokenization = sensor_dim > lane_flat_dim and lidar_flat_dim > 0 and lidar_flat_dim % 2 == 0
        if self.semantic_tokenization:
            self.lidar_feature_dim = 2
            self.lidar_cells = lidar_flat_dim // self.lidar_feature_dim
            self.lane_flat_dim = lane_flat_dim
            self.lidar_flat_dim = lidar_flat_dim
            self.max_current_tokens = 1 + self.ego_dim + self.lidar_cells + self.lane_camera_cells
        else:
            self.lidar_feature_dim = 0
            self.lidar_cells = 0
            self.lane_flat_dim = 0
            self.lidar_flat_dim = 0
            self.max_current_tokens = 1 + self.obs_dim

        self.policy_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.type_embedding = nn.Embedding(5, self.hidden_size)
        self.current_position_embedding = nn.Parameter(
            torch.zeros(1, self.max_current_tokens, self.hidden_size)
        )
        self.memory_position_embedding = nn.Parameter(
            torch.zeros(1, self.memory_context_length, 1, self.hidden_size)
        )
        self.memory_slot_embedding = nn.Parameter(
            torch.zeros(1, 1, self.memory_tokens, self.hidden_size)
        )
        self.scalar_proj = nn.Linear(1, self.hidden_size)
        self.ego_proj = nn.Linear(1, self.hidden_size)
        self.lidar_proj = nn.Linear(2, self.hidden_size)
        self.lane_proj = nn.Linear(3, self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(num_layers)),
            norm=nn.LayerNorm(self.hidden_size),
        )
        self.memory_query = nn.Parameter(torch.zeros(1, self.memory_tokens, self.hidden_size))
        self.memory_pool = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.memory_norm = nn.LayerNorm(self.hidden_size)

        if self.action_mode == "continuous":
            self.policy_head = nn.Linear(self.hidden_size, self.continuous_action_dim)
            self.log_std = nn.Parameter(torch.full((self.continuous_action_dim,), -0.5))
        elif self.action_mode == "discrete":
            self.policy_head = nn.Linear(self.hidden_size, num_actions)
            self.log_std = None
        else:
            raise ValueError(f"Unsupported action_mode={action_mode!r}.")
        self.critic_encoder = (
            make_centralized_critic_encoder(
                self.critic_obs_dim,
                self.hidden_size,
                pooling=central_critic_pooling,
                max_vehicles=central_critic_max_vehicles,
                attention_heads=central_critic_attention_heads,
            )
            if self.centralized_critic
            else None
        )
        self.value_head = nn.Linear(self.hidden_size, 1)

        nn.init.normal_(self.policy_token, std=0.02)
        nn.init.normal_(self.current_position_embedding, std=0.02)
        nn.init.normal_(self.memory_position_embedding, std=0.02)
        nn.init.normal_(self.memory_slot_embedding, std=0.02)
        nn.init.normal_(self.memory_query, std=0.02)

    def initial_memory(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(
            (
                int(batch_size),
                self.memory_context_length,
                self.memory_tokens,
                self.hidden_size,
            ),
            dtype=dtype or self.policy_token.dtype,
            device=device or self.policy_token.device,
        )

    def _type_tokens(self, token_type: int, batch_size: int, count: int, device: torch.device) -> torch.Tensor:
        ids = torch.full((int(batch_size), int(count)), int(token_type), dtype=torch.long, device=device)
        return self.type_embedding(ids)

    def _build_current_tokens(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = int(obs.shape[0])
        policy = self.policy_token.expand(batch_size, -1, -1)
        policy = policy + self._type_tokens(0, batch_size, 1, obs.device)
        if self.semantic_tokenization:
            lidar = obs[:, : self.lidar_flat_dim].reshape(batch_size, self.lidar_cells, 2)
            lane_start = self.lidar_flat_dim
            lane_end = lane_start + self.lane_flat_dim
            lane = obs[:, lane_start:lane_end].reshape(batch_size, self.lane_camera_cells, 3)
            ego = obs[:, -self.ego_dim :].reshape(batch_size, self.ego_dim, 1)
            tokens = torch.cat(
                [
                    policy,
                    self.ego_proj(ego) + self._type_tokens(1, batch_size, self.ego_dim, obs.device),
                    self.lidar_proj(lidar) + self._type_tokens(2, batch_size, self.lidar_cells, obs.device),
                    self.lane_proj(lane) + self._type_tokens(3, batch_size, self.lane_camera_cells, obs.device),
                ],
                dim=1,
            )
        else:
            scalar_tokens = self.scalar_proj(obs.unsqueeze(-1))
            scalar_tokens = scalar_tokens + self._type_tokens(4, batch_size, self.obs_dim, obs.device)
            tokens = torch.cat([policy, scalar_tokens], dim=1)
        return tokens + self.current_position_embedding[:, : tokens.shape[1]]

    def _prepare_memory(self, obs: torch.Tensor, memory: torch.Tensor | None) -> torch.Tensor:
        batch_size = int(obs.shape[0])
        if memory is None:
            memory = self.initial_memory(batch_size, device=obs.device, dtype=obs.dtype)
        else:
            memory = memory.to(device=obs.device, dtype=obs.dtype)
            if memory.ndim == 3:
                memory = memory.unsqueeze(1)
            if memory.ndim != 4:
                raise ValueError(
                    "Recurrent transformer memory must have shape "
                    f"[B, T, M, H] or [B, M, H], got {tuple(memory.shape)}."
                )
            if memory.shape[0] != batch_size:
                raise ValueError(f"Memory batch mismatch: {memory.shape[0]} != {batch_size}.")
            if memory.shape[2] != self.memory_tokens or memory.shape[3] != self.hidden_size:
                raise ValueError(
                    "Memory token shape mismatch: expected "
                    f"[*, *, {self.memory_tokens}, {self.hidden_size}], got {tuple(memory.shape)}."
                )
            if memory.shape[1] > self.memory_context_length:
                memory = memory[:, -self.memory_context_length :]
        if memory.shape[1] < self.memory_context_length:
            pad = torch.zeros(
                (
                    batch_size,
                    self.memory_context_length - int(memory.shape[1]),
                    self.memory_tokens,
                    self.hidden_size,
                ),
                dtype=memory.dtype,
                device=memory.device,
            )
            memory = torch.cat([pad, memory], dim=1)
        positions = self.memory_position_embedding[:, -int(memory.shape[1]) :]
        slots = self.memory_slot_embedding[:, :, : self.memory_tokens]
        memory = memory + positions + slots
        memory = memory + self.type_embedding(
            torch.full(
                memory.shape[:-1],
                4,
                dtype=torch.long,
                device=memory.device,
            )
        )
        return memory.reshape(batch_size, -1, self.hidden_size)

    def _attention_mask(self, memory_token_count: int, current_token_count: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_causal_attention or memory_token_count <= 0:
            return None
        total = int(memory_token_count) + int(current_token_count)
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:memory_token_count, memory_token_count:] = True
        return mask

    def _encode_actor(
        self,
        obs: torch.Tensor,
        memory: torch.Tensor | None = None,
        *,
        return_memory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        current_tokens = self._build_current_tokens(obs)
        memory_tokens = self._prepare_memory(obs, memory)
        tokens = torch.cat([memory_tokens, current_tokens], dim=1)
        attention_mask = self._attention_mask(memory_tokens.shape[1], current_tokens.shape[1], obs.device)
        encoded = self.encoder(tokens, mask=attention_mask)
        current_encoded = encoded[:, memory_tokens.shape[1] :]
        actor_encoded = current_encoded[:, 0]
        if not return_memory:
            return actor_encoded
        queries = self.memory_query.expand(obs.shape[0], -1, -1)
        pooled, _weights = self.memory_pool(queries, current_encoded, current_encoded, need_weights=False)
        new_memory = self.memory_norm(queries + pooled)
        return actor_encoded, new_memory

    def actor(self, obs: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self._encode_actor(obs, memory)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        policy_out = self.policy_head(encoded)
        if self.action_mode == "continuous":
            policy_out = torch.tanh(policy_out)
        return policy_out

    def value(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
        *,
        encoded_actor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.centralized_critic:
            if critic_obs is None:
                critic_obs = torch.zeros(
                    (obs.shape[0], self.critic_obs_dim),
                    dtype=obs.dtype,
                    device=obs.device,
                )
            if self.critic_encoder is None:
                raise RuntimeError("Centralized critic is enabled without a critic encoder.")
            encoded = self.critic_encoder(critic_obs)
        else:
            encoded = encoded_actor if encoded_actor is not None else self._encode_actor(obs)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
        return self.value_head(encoded).squeeze(-1)

    def forward(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
        *,
        memory: torch.Tensor | None = None,
        return_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self._encode_actor(obs, memory, return_memory=return_memory)
        if return_memory:
            actor_encoded, new_memory = encoded
        else:
            actor_encoded = encoded
            new_memory = None
        policy_out = self.policy_head(actor_encoded)
        if self.action_mode == "continuous":
            policy_out = torch.tanh(policy_out)
        values = self.value(obs, critic_obs, encoded_actor=actor_encoded)
        if return_memory:
            if new_memory is None:
                raise RuntimeError("Recurrent transformer did not produce memory.")
            return policy_out, values, new_memory
        return policy_out, values


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
    transformer_temporal_module: bool = False,
    transformer_temporal_kernel_size: int = 5,
    transformer_temporal_layers: int = 1,
    transformer_memory_tokens: int = 8,
    transformer_memory_context_length: int = 32,
    transformer_use_causal_attention: bool = True,
    centralized_critic: bool = False,
    critic_obs_dim: int | None = None,
    central_critic_pooling: str = "flat",
    central_critic_max_vehicles: int = 64,
    central_critic_attention_heads: int = 4,
) -> nn.Module:
    model_name = str(policy_model).lower()
    if model_name == "mlp":
        return SharedActorCritic(
            obs_dim,
            hidden_size,
            action_mode=action_mode,
            continuous_action_dim=continuous_action_dim,
            centralized_critic=centralized_critic,
            critic_obs_dim=critic_obs_dim,
            central_critic_pooling=central_critic_pooling,
            central_critic_max_vehicles=central_critic_max_vehicles,
            central_critic_attention_heads=central_critic_attention_heads,
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
            temporal_module=transformer_temporal_module,
            temporal_kernel_size=transformer_temporal_kernel_size,
            temporal_layers=transformer_temporal_layers,
            centralized_critic=centralized_critic,
            critic_obs_dim=critic_obs_dim,
            central_critic_pooling=central_critic_pooling,
            central_critic_max_vehicles=central_critic_max_vehicles,
            central_critic_attention_heads=central_critic_attention_heads,
        )
    if model_name == "recurrent_transformer":
        return RecurrentTransformerActorCritic(
            obs_dim,
            hidden_size,
            action_mode=action_mode,
            continuous_action_dim=continuous_action_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
            memory_tokens=transformer_memory_tokens,
            memory_context_length=transformer_memory_context_length,
            use_causal_attention=transformer_use_causal_attention,
            centralized_critic=centralized_critic,
            critic_obs_dim=critic_obs_dim,
            central_critic_pooling=central_critic_pooling,
            central_critic_max_vehicles=central_critic_max_vehicles,
            central_critic_attention_heads=central_critic_attention_heads,
        )
    raise ValueError(
        f"Unsupported policy_model={policy_model!r}. "
        "Expected 'mlp', 'transformer', or 'recurrent_transformer'."
    )


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int | None = None,
        *,
        hidden_sizes: str | int | tuple[int, ...] | list[int] | None = None,
        dropout: float = DEFAULT_CRITIC_DROPOUT,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.net = make_relu_mlp(
            input_dim,
            hidden_sizes if hidden_sizes is not None else hidden_size,
            1,
            dropout=dropout,
            spectral_norm=spectral_norm,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class SceneDiscriminator(TrajectoryDiscriminator):
    """MLP discriminator for one full-road traffic snapshot."""


class SequenceTrajectoryDiscriminator(nn.Module):
    """Autoregressive discriminator over fixed-length trajectory feature windows."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int | None = None,
        *,
        hidden_sizes: str | int | tuple[int, ...] | list[int] | None = None,
        dropout: float = DEFAULT_CRITIC_DROPOUT,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        recurrent_and_head_sizes = parse_hidden_sizes(
            hidden_sizes if hidden_sizes is not None else hidden_size
        )
        recurrent_size = int(recurrent_and_head_sizes[0])
        head_sizes = recurrent_and_head_sizes[1:] or (recurrent_size,)
        self.encoder = nn.GRU(
            input_size=int(input_dim),
            hidden_size=recurrent_size,
            batch_first=True,
        )
        self.head = make_relu_mlp(
            recurrent_size,
            head_sizes,
            1,
            dropout=dropout,
            spectral_norm=spectral_norm,
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
