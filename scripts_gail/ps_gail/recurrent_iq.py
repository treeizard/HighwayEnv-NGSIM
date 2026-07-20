"""Sequence-aware online IQ-Learn for recurrent transformer policies.

The loss follows the continuous-control reference implementation at
https://github.com/Div-Infinity/IQ-Learn (commit recorded by the caller):
the inverse Bellman term is evaluated on expert transitions, the value term is
evaluated on an equal expert/policy mixture, and the actor uses a SAC update.
The reference's default chi-squared penalty is expert-only; mixture
regularization is available as an explicitly labelled stability variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.recurrent_bc import SequenceWindow


@dataclass(frozen=True)
class RecurrentIQUpdateStats:
    q_loss: float
    q1_loss: float
    q2_loss: float
    inverse_softq_loss: float
    value_loss: float
    chi2_loss: float
    actor_loss: float
    actor_rl_loss: float
    bc_loss: float
    expert_q: float
    policy_q: float
    q_disagreement: float
    recovered_reward: float
    entropy: float
    q_abs_max: float
    actor_grad_norm: float
    actor_rl_grad_norm: float
    actor_bc_grad_norm: float
    critic_grad_norm: float
    expert_samples: int
    policy_samples: int
    samples: int

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in vars(self).items()}


class RecurrentTransformerQNetwork(nn.Module):
    """One action-conditioned recurrent Q network."""

    supports_recurrent_memory = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        *,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        transformer_norm_first: bool = False,
        memory_tokens: int,
        memory_context_length: int,
        use_causal_attention: bool,
    ) -> None:
        super().__init__()
        self.encoder = make_actor_critic(
            "recurrent_transformer",
            int(obs_dim),
            int(hidden_size),
            action_mode="continuous",
            continuous_action_dim=int(action_dim),
            transformer_layers=int(transformer_layers),
            transformer_heads=int(transformer_heads),
            transformer_dropout=float(transformer_dropout),
            transformer_norm_first=bool(transformer_norm_first),
            transformer_memory_tokens=int(memory_tokens),
            transformer_memory_context_length=int(memory_context_length),
            transformer_use_causal_attention=bool(use_causal_attention),
        )
        # The actor/value heads only provide the recurrent encoder shell.
        for module in (self.encoder.policy_head, self.encoder.value_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.encoder.log_std is not None:
            self.encoder.log_std.requires_grad_(False)
        self.q_head = nn.Sequential(
            nn.Linear(int(hidden_size) + int(action_dim), int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), 1),
        )
        self.memory_context_length = int(memory_context_length)
        self.reset_q_head()

    def reset_q_head(self) -> None:
        for module in self.q_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def initialize_encoder_from_policy(self, policy: nn.Module) -> None:
        """Warm-start the recurrent representation while keeping a fresh Q head."""
        self.encoder.load_state_dict(policy.state_dict(), strict=True)
        for module in (self.encoder.policy_head, self.encoder.value_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.encoder.log_std is not None:
            self.encoder.log_std.requires_grad_(False)

    def initial_memory(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.encoder.initial_memory(batch_size, device=device, dtype=dtype)

    def encode(
        self,
        obs: torch.Tensor,
        memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder._encode_actor(obs, memory, return_memory=True)
        if not isinstance(encoded, tuple):
            raise RuntimeError("Recurrent Q encoder did not return updated memory.")
        return encoded

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        memory: torch.Tensor | None = None,
        *,
        return_memory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded, new_memory = self.encode(obs, memory)
        q_value = self.q_head(torch.cat([encoded, action], dim=-1)).squeeze(-1)
        if return_memory:
            return q_value, new_memory
        return q_value


class RecurrentTwinQNetwork(nn.Module):
    """Independent twin recurrent critics, matching continuous IQ-Learn/SAC."""

    supports_recurrent_memory = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.q1 = RecurrentTransformerQNetwork(*args, **kwargs)
        self.q2 = RecurrentTransformerQNetwork(*args, **kwargs)

    def initialize_encoders_from_policy(self, policy: nn.Module) -> None:
        self.q1.initialize_encoder_from_policy(policy)
        self.q2.initialize_encoder_from_policy(policy)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        memories: tuple[torch.Tensor, torch.Tensor],
        *,
        return_memories: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[
        tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]:
        q1, memory1 = self.q1(obs, action, memories[0], return_memory=True)
        q2, memory2 = self.q2(obs, action, memories[1], return_memory=True)
        if return_memories:
            return (q1, q2), (memory1, memory2)
        return torch.minimum(q1, q2)


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def append_memory(memory: torch.Tensor, new_memory: torch.Tensor, *, max_context: int) -> torch.Tensor:
    if memory.ndim != 4 or new_memory.ndim != 3:
        raise ValueError(
            f"Expected memory [B,T,M,H] and new memory [B,M,H], got {tuple(memory.shape)} and "
            f"{tuple(new_memory.shape)}."
        )
    combined = torch.cat([memory, new_memory.unsqueeze(1)], dim=1)
    return combined[:, -max(1, int(max_context)) :]


def sample_recurrent_policy(
    policy: nn.Module,
    obs: torch.Tensor,
    memory: torch.Tensor,
    *,
    log_std_min: float,
    log_std_max: float,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = policy._encode_actor(obs, memory, return_memory=True)
    if not isinstance(encoded, tuple):
        raise RuntimeError("Recurrent policy did not return updated memory.")
    actor_encoded, new_memory = encoded
    location = policy.policy_head(actor_encoded)
    log_std = torch.clamp(policy.log_std, float(log_std_min), float(log_std_max)).expand_as(location)
    distribution = Normal(location, torch.exp(log_std))
    raw_action = location if deterministic else distribution.rsample()
    action = torch.tanh(raw_action)
    log_prob = distribution.log_prob(raw_action) - torch.log(1.0 - action.square() + 1.0e-6)
    return action, log_prob.sum(dim=-1), torch.tanh(location), new_memory


def _step_indices(windows: list[SequenceWindow], step: int) -> tuple[np.ndarray, np.ndarray]:
    fallback = np.asarray([int(window.train_indices[0]) for window in windows], dtype=np.int64)
    active = np.asarray([step < len(window.train_indices) for window in windows], dtype=bool)
    indices = fallback.copy()
    for row, window in enumerate(windows):
        if active[row]:
            indices[row] = int(window.train_indices[step])
    return indices, active


def _advance_selected_memory(
    memory: torch.Tensor,
    new_memory: torch.Tensor,
    active: torch.Tensor,
    *,
    max_context: int,
) -> torch.Tensor:
    shifted = append_memory(memory, new_memory, max_context=max_context)
    return torch.where(active[:, None, None, None], shifted, memory)


def _warm_memories(
    policy: nn.Module,
    networks: Iterable[RecurrentTransformerQNetwork],
    observations: np.ndarray,
    windows: list[SequenceWindow],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    networks = list(networks)
    batch_size = len(windows)
    policy_memory = policy.initial_memory(batch_size, device=device, dtype=torch.float32)
    q_memories = [network.initial_memory(batch_size, device=device) for network in networks]
    max_context = max(len(window.context_indices) for window in windows)
    fallback = np.asarray([int(window.train_indices[0]) for window in windows], dtype=np.int64)
    with torch.no_grad():
        for offset in range(-max_context, 0):
            active_np = np.asarray([len(window.context_indices) + offset >= 0 for window in windows], dtype=bool)
            if not bool(active_np.any()):
                continue
            indices = fallback.copy()
            for row, window in enumerate(windows):
                position = len(window.context_indices) + offset
                if position >= 0:
                    indices[row] = int(window.context_indices[position])
            obs = torch.as_tensor(observations[indices], dtype=torch.float32, device=device)
            active = torch.as_tensor(active_np, dtype=torch.bool, device=device)
            _encoded, policy_new = policy._encode_actor(obs, policy_memory, return_memory=True)
            policy_memory = _advance_selected_memory(
                policy_memory,
                policy_new,
                active,
                max_context=int(policy.memory_context_length),
            )
            for index, network in enumerate(networks):
                _q_encoded, q_new = network.encode(obs, q_memories[index])
                q_memories[index] = _advance_selected_memory(
                    q_memories[index], q_new, active, max_context=network.memory_context_length
                )
    return policy_memory, q_memories


def _actions(transitions: Any) -> np.ndarray:
    source = getattr(transitions, "actions_continuous_env", None)
    if source is None:
        source = getattr(transitions, "actions", None)
    if source is None:
        raise AttributeError("Transitions require actions_continuous_env or actions.")
    return np.clip(np.asarray(source, dtype=np.float32), -1.0, 1.0)


def _terminals(transitions: Any) -> np.ndarray:
    source = getattr(transitions, "terminals", None)
    if source is None:
        source = transitions.dones
    return np.asarray(source, dtype=np.float32)


def _grad_norm(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
    squares = [gradient.detach().square().sum() for gradient in grads if gradient is not None]
    if not squares:
        return torch.zeros(())
    return torch.sqrt(torch.stack(squares).sum())


def update_recurrent_iq(
    policy: nn.Module,
    q_net: RecurrentTwinQNetwork,
    target_q_net: RecurrentTwinQNetwork,
    policy_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    expert_transitions: Any,
    expert_windows: list[SequenceWindow],
    *,
    device: torch.device,
    gamma: float,
    entropy_temperature: float,
    chi2_alpha: float,
    target_tau: float,
    bc_coef: float,
    max_grad_norm: float,
    policy_transitions: Any | None = None,
    policy_windows: list[SequenceWindow] | None = None,
    update_actor: bool = True,
    actor_bc_only: bool = False,
    chi2_on_mixture: bool = False,
    target_q_clip: float = 0.0,
    log_std_min: float = -5.0,
    log_std_max: float = 0.5,
) -> RecurrentIQUpdateStats:
    """Apply one recurrent twin-Q IQ-Learn update.

    Equal-sized expert and learner window batches implement the reference
    online 50/50 replay mixture. When no learner batch is supplied the function
    intentionally reduces to the reference offline ``value_expert`` variant.
    """
    if not expert_windows:
        raise ValueError("At least one expert recurrent IQ window is required.")
    online = policy_transitions is not None and bool(policy_windows)
    if (policy_transitions is None) != (not policy_windows):
        raise ValueError("policy_transitions and policy_windows must be supplied together.")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be between zero and one.")
    if float(chi2_alpha) <= 0.0 or float(max_grad_norm) <= 0.0:
        raise ValueError("chi2_alpha and max_grad_norm must be positive.")
    if not 0.0 < float(target_tau) <= 1.0:
        raise ValueError("target_tau must be in (0, 1].")
    if float(entropy_temperature) < 0.0 or float(bc_coef) < 0.0:
        raise ValueError("entropy_temperature and bc_coef must be non-negative.")

    critics = [q_net.q1, q_net.q2]
    targets = [target_q_net.q1, target_q_net.q2]

    def critic_source(transitions: Any, windows: list[SequenceWindow]) -> dict[str, Any]:
        observations = np.asarray(transitions.policy_observations, dtype=np.float32)
        next_observations = np.asarray(transitions.next_policy_observations, dtype=np.float32)
        replay_actions = _actions(transitions)
        terminals = _terminals(transitions)
        policy_memory, memories = _warm_memories(
            policy, [*critics, *targets], observations, windows, device=device
        )
        q1_memory, q2_memory, target1_memory, target2_memory = memories
        totals = {
            "reward1": torch.zeros((), device=device),
            "reward2": torch.zeros((), device=device),
            "reward1_square": torch.zeros((), device=device),
            "reward2_square": torch.zeros((), device=device),
            "value": torch.zeros((), device=device),
            "expert_q": torch.zeros((), device=device),
            "policy_q": torch.zeros((), device=device),
            "disagreement": torch.zeros((), device=device),
            "q_abs_max": torch.zeros((), device=device),
        }
        samples = 0
        max_steps = max(len(window.train_indices) for window in windows)
        for step in range(max_steps):
            indices, active_np = _step_indices(windows, step)
            if not bool(active_np.any()):
                continue
            active = torch.as_tensor(active_np, dtype=torch.bool, device=device)
            obs = torch.as_tensor(observations[indices], dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_observations[indices], dtype=torch.float32, device=device)
            replay_action = torch.as_tensor(replay_actions[indices], dtype=torch.float32, device=device)
            terminal = torch.as_tensor(terminals[indices], dtype=torch.float32, device=device)
            with torch.no_grad():
                sampled_action, log_prob, _mean, policy_new = sample_recurrent_policy(
                    policy, obs, policy_memory, log_std_min=log_std_min, log_std_max=log_std_max
                )
                policy_after = _advance_selected_memory(
                    policy_memory, policy_new, active, max_context=int(policy.memory_context_length)
                )
                next_action, next_log_prob, _next_mean, _next_new = sample_recurrent_policy(
                    policy, next_obs, policy_after, log_std_min=log_std_min, log_std_max=log_std_max
                )
                _encoded1, target1_new = targets[0].encode(obs, target1_memory)
                _encoded2, target2_new = targets[1].encode(obs, target2_memory)
                target1_after = _advance_selected_memory(
                    target1_memory, target1_new, active, max_context=targets[0].memory_context_length
                )
                target2_after = _advance_selected_memory(
                    target2_memory, target2_new, active, max_context=targets[1].memory_context_length
                )
                next_q1 = targets[0](next_obs, next_action, target1_after)
                next_q2 = targets[1](next_obs, next_action, target2_after)
                next_q = torch.minimum(next_q1, next_q2)
                if float(target_q_clip) > 0.0:
                    next_q = next_q.clamp(-float(target_q_clip), float(target_q_clip))
                y = (1.0 - terminal) * float(gamma) * (
                    next_q - float(entropy_temperature) * next_log_prob
                )

            replay_q1, q1_new = critics[0](obs, replay_action, q1_memory, return_memory=True)
            replay_q2, q2_new = critics[1](obs, replay_action, q2_memory, return_memory=True)
            policy_q1 = critics[0](obs, sampled_action, q1_memory)
            policy_q2 = critics[1](obs, sampled_action, q2_memory)
            current_v = torch.minimum(policy_q1, policy_q2) - float(entropy_temperature) * log_prob
            reward1 = replay_q1 - y
            reward2 = replay_q2 - y
            count = int(active_np.sum())
            totals["reward1"] += reward1[active].mean() * count
            totals["reward2"] += reward2[active].mean() * count
            totals["reward1_square"] += reward1[active].square().mean() * count
            totals["reward2_square"] += reward2[active].square().mean() * count
            totals["value"] += (current_v - y)[active].mean() * count
            totals["expert_q"] += torch.minimum(replay_q1, replay_q2)[active].mean().detach() * count
            totals["policy_q"] += torch.minimum(policy_q1, policy_q2)[active].mean().detach() * count
            totals["disagreement"] += (replay_q1 - replay_q2)[active].abs().mean().detach() * count
            totals["q_abs_max"] = torch.maximum(
                totals["q_abs_max"],
                torch.stack((replay_q1[active].abs().max(), replay_q2[active].abs().max(), next_q[active].abs().max())).max().detach(),
            )
            samples += count
            policy_memory = policy_after
            q1_memory = _advance_selected_memory(
                q1_memory, q1_new, active, max_context=critics[0].memory_context_length
            )
            q2_memory = _advance_selected_memory(
                q2_memory, q2_new, active, max_context=critics[1].memory_context_length
            )
            target1_memory, target2_memory = target1_after, target2_after
        if samples <= 0:
            raise RuntimeError("Recurrent IQ source batch contained no active samples.")
        return {**{key: value / samples for key, value in totals.items() if key != "q_abs_max"}, "q_abs_max": totals["q_abs_max"], "samples": samples}

    expert = critic_source(expert_transitions, expert_windows)
    learner = critic_source(policy_transitions, policy_windows) if online else None
    value_loss = expert["value"] if learner is None else 0.5 * (expert["value"] + learner["value"])
    chi1 = expert["reward1_square"]
    chi2 = expert["reward2_square"]
    if learner is not None and bool(chi2_on_mixture):
        chi1 = 0.5 * (chi1 + learner["reward1_square"])
        chi2 = 0.5 * (chi2 + learner["reward2_square"])
    q1_loss = -expert["reward1"] + value_loss + chi1 / (4.0 * float(chi2_alpha))
    q2_loss = -expert["reward2"] + value_loss + chi2 / (4.0 * float(chi2_alpha))
    q_loss = 0.5 * (q1_loss + q2_loss)
    q_optimizer.zero_grad(set_to_none=True)
    q_loss.backward()
    critic_grad = nn.utils.clip_grad_norm_(trainable_parameters(q_net), float(max_grad_norm))
    q_optimizer.step()

    def actor_source(transitions: Any, windows: list[SequenceWindow], *, include_bc: bool) -> dict[str, Any]:
        observations = np.asarray(transitions.policy_observations, dtype=np.float32)
        replay_actions = _actions(transitions)
        policy_memory, q_memories = _warm_memories(policy, critics, observations, windows, device=device)
        q1_memory, q2_memory = q_memories
        rl_sum = torch.zeros((), device=device)
        bc_sum = torch.zeros((), device=device)
        entropy_sum = torch.zeros((), device=device)
        samples = 0
        for step in range(max(len(window.train_indices) for window in windows)):
            indices, active_np = _step_indices(windows, step)
            if not bool(active_np.any()):
                continue
            active = torch.as_tensor(active_np, dtype=torch.bool, device=device)
            obs = torch.as_tensor(observations[indices], dtype=torch.float32, device=device)
            expert_action = torch.as_tensor(replay_actions[indices], dtype=torch.float32, device=device)
            sampled_action, log_prob, mean_action, policy_new = sample_recurrent_policy(
                policy, obs, policy_memory, log_std_min=log_std_min, log_std_max=log_std_max
            )
            actor_q1, q1_new = critics[0](obs, sampled_action, q1_memory, return_memory=True)
            actor_q2, q2_new = critics[1](obs, sampled_action, q2_memory, return_memory=True)
            rl = (float(entropy_temperature) * log_prob - torch.minimum(actor_q1, actor_q2))[active].mean()
            bc = F.mse_loss(mean_action[active], expert_action[active]) if include_bc else torch.zeros((), device=device)
            count = int(active_np.sum())
            rl_sum += rl * count
            bc_sum += bc * count
            entropy_sum += (-log_prob[active]).mean().detach() * count
            samples += count
            policy_memory = _advance_selected_memory(
                policy_memory, policy_new, active, max_context=int(policy.memory_context_length)
            )
            q1_memory = _advance_selected_memory(
                q1_memory, q1_new, active, max_context=critics[0].memory_context_length
            )
            q2_memory = _advance_selected_memory(
                q2_memory, q2_new, active, max_context=critics[1].memory_context_length
            )
        return {"rl": rl_sum / samples, "bc": bc_sum / samples, "entropy": entropy_sum / samples, "samples": samples}

    actor_loss = torch.zeros((), device=device)
    actor_rl = torch.zeros((), device=device)
    bc_loss = torch.zeros((), device=device)
    actor_entropy = torch.zeros((), device=device)
    actor_grad = torch.zeros((), device=device)
    rl_grad_norm = torch.zeros((), device=device)
    bc_grad_norm = torch.zeros((), device=device)
    if bool(update_actor):
        # Q-only stabilization should not build and immediately discard the
        # expensive recurrent actor graph.  Once joint training starts, freeze
        # the critic weights while retaining dQ/da for the SAC actor update.
        for parameter in q_net.parameters():
            parameter.requires_grad_(False)
        expert_actor = actor_source(expert_transitions, expert_windows, include_bc=True)
        learner_actor = actor_source(policy_transitions, policy_windows, include_bc=False) if online else None
        actor_rl = expert_actor["rl"] if learner_actor is None else 0.5 * (
            expert_actor["rl"] + learner_actor["rl"]
        )
        bc_loss = expert_actor["bc"]
        actor_loss = float(bc_coef) * bc_loss if actor_bc_only else actor_rl + float(bc_coef) * bc_loss
        actor_entropy = expert_actor["entropy"] if learner_actor is None else 0.5 * (
            expert_actor["entropy"] + learner_actor["entropy"]
        )
        policy_optimizer.zero_grad(set_to_none=True)
        parameters = trainable_parameters(policy)
        rl_grads = torch.autograd.grad(actor_rl, parameters, retain_graph=True, allow_unused=True)
        bc_grads = torch.autograd.grad(bc_loss, parameters, retain_graph=True, allow_unused=True)
        rl_grad_norm = _grad_norm(rl_grads).to(device)
        bc_grad_norm = _grad_norm(bc_grads).to(device)
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
        policy_optimizer.step()
        for parameter in q_net.parameters():
            parameter.requires_grad_(True)
        for network in critics:
            for module in (network.encoder.policy_head, network.encoder.value_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            if network.encoder.log_std is not None:
                network.encoder.log_std.requires_grad_(False)

    with torch.no_grad():
        for target_parameter, parameter in zip(target_q_net.parameters(), q_net.parameters()):
            target_parameter.mul_(1.0 - float(target_tau)).add_(parameter, alpha=float(target_tau))

    learner_samples = int(learner["samples"]) if learner is not None else 0
    policy_q = expert["policy_q"] if learner is None else 0.5 * (expert["policy_q"] + learner["policy_q"])
    disagreement = expert["disagreement"] if learner is None else 0.5 * (
        expert["disagreement"] + learner["disagreement"]
    )
    q_abs = expert["q_abs_max"] if learner is None else torch.maximum(expert["q_abs_max"], learner["q_abs_max"])
    return RecurrentIQUpdateStats(
        q_loss=float(q_loss.detach().cpu()),
        q1_loss=float(q1_loss.detach().cpu()),
        q2_loss=float(q2_loss.detach().cpu()),
        inverse_softq_loss=float((-0.5 * (expert["reward1"] + expert["reward2"])).detach().cpu()),
        value_loss=float(value_loss.detach().cpu()),
        chi2_loss=float((0.5 * (chi1 + chi2) / (4.0 * float(chi2_alpha))).detach().cpu()),
        actor_loss=float(actor_loss.detach().cpu()),
        actor_rl_loss=float(actor_rl.detach().cpu()),
        bc_loss=float(bc_loss.detach().cpu()),
        expert_q=float(expert["expert_q"].detach().cpu()),
        policy_q=float(policy_q.detach().cpu()),
        q_disagreement=float(disagreement.detach().cpu()),
        recovered_reward=float((0.5 * (expert["reward1"] + expert["reward2"])).detach().cpu()),
        entropy=float(actor_entropy.detach().cpu()),
        q_abs_max=float(q_abs.detach().cpu()),
        actor_grad_norm=float(torch.as_tensor(actor_grad).detach().cpu()),
        actor_rl_grad_norm=float(rl_grad_norm.detach().cpu()),
        actor_bc_grad_norm=float(bc_grad_norm.detach().cpu()),
        critic_grad_norm=float(torch.as_tensor(critic_grad).detach().cpu()),
        expert_samples=int(expert["samples"]),
        policy_samples=learner_samples,
        samples=int(expert["samples"]) + learner_samples,
    )
