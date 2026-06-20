"""Discriminator sampling and optimization utilities for PS-GAIL."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PSGAILConfig

from .torch_utils import _as_device_tensor

def _wgan_gradient_penalty(
    discriminator: nn.Module,
    expert: torch.Tensor,
    generator: torch.Tensor,
    *,
    gp_lambda: float,
) -> torch.Tensor:
    alpha_shape = (expert.shape[0],) + (1,) * (expert.ndim - 1)
    alpha = torch.rand(alpha_shape, dtype=expert.dtype, device=expert.device)
    interpolated = (alpha * expert + (1.0 - alpha) * generator).requires_grad_(True)
    scores = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.shape[0], -1)
    gradient_norm = gradients.norm(2, dim=1)
    return float(gp_lambda) * torch.square(gradient_norm - 1.0).mean()

def _sample_array_indices(
    n: int,
    size: int,
    *,
    replace: bool,
) -> np.ndarray:
    n = int(n)
    size = max(0, int(size))
    if n <= 0 or size <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.random.choice(n, size=size, replace=bool(replace)).astype(np.int64)

def _score_discriminator_candidates(
    discriminator: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    was_training = discriminator.training
    discriminator.eval()
    try:
        with torch.no_grad():
            scores = discriminator(
                _as_device_tensor(features.astype(np.float32, copy=False), dtype=torch.float32, device=device)
            )
    finally:
        if was_training:
            discriminator.train()
    return scores.detach().cpu().numpy().astype(np.float32).reshape(-1)

def _soft_hard_indices(
    hardness: np.ndarray,
    *,
    selected_count: int,
    selected_fraction: float,
    uniform_mix: float,
    temperature: float,
) -> np.ndarray:
    hardness = np.asarray(hardness, dtype=np.float64).reshape(-1)
    n = int(hardness.size)
    selected_count = min(n, max(1, int(selected_count)))
    if n <= selected_count:
        return np.arange(n, dtype=np.int64)
    uniform_count = int(round(selected_count * min(1.0, max(0.0, float(uniform_mix)))))
    uniform_count = min(selected_count, max(0, uniform_count))
    hard_count = selected_count - uniform_count
    chosen: list[np.ndarray] = []
    if hard_count > 0:
        top_pool_count = min(n, max(hard_count, int(np.ceil(max(0.0, min(1.0, selected_fraction)) * n))))
        top_pool = np.argpartition(-hardness, top_pool_count - 1)[:top_pool_count]
        pool_hardness = hardness[top_pool]
        temp = max(1.0e-6, float(temperature))
        logits = (pool_hardness - float(pool_hardness.max())) / temp
        weights = np.exp(logits)
        weight_sum = float(weights.sum())
        probs = None if weight_sum <= 0.0 or not np.isfinite(weight_sum) else weights / weight_sum
        chosen.append(
            np.random.choice(top_pool, size=hard_count, replace=False, p=probs).astype(np.int64)
        )
    if uniform_count > 0:
        excluded = np.concatenate(chosen, axis=0) if chosen else np.zeros((0,), dtype=np.int64)
        available = np.setdiff1d(np.arange(n, dtype=np.int64), excluded, assume_unique=False)
        if available.size < uniform_count:
            available = np.arange(n, dtype=np.int64)
        chosen.append(
            np.random.choice(available, size=uniform_count, replace=False).astype(np.int64)
        )
    return np.concatenate(chosen, axis=0).astype(np.int64)

def select_hard_discriminator_examples(
    discriminator: nn.Module,
    expert_features: np.ndarray,
    generator_features: np.ndarray,
    cfg: PSGAILConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not bool(getattr(cfg, "enable_hard_example_selection", False)):
        expert_idx = np.random.choice(
            len(expert_features),
            size=len(generator_features),
            replace=len(expert_features) < len(generator_features),
        )
        return (
            expert_features[expert_idx],
            generator_features,
            {
                "hard_selector_enabled": 0.0,
                "hard_selector_candidate_samples": float(len(generator_features)),
                "hard_selector_selected_samples": float(len(generator_features)),
                "hard_selector_selected_fraction": 1.0,
                "hard_selector_expert_full_score_mean": float("nan"),
                "hard_selector_gen_full_score_mean": float("nan"),
                "hard_selector_expert_selected_score_mean": float("nan"),
                "hard_selector_gen_selected_score_mean": float("nan"),
            },
        )

    candidate_cap = max(1, int(getattr(cfg, "hard_example_candidate_samples", 65_536)))
    candidate_count = min(candidate_cap, max(1, int(len(generator_features))))
    expert_candidate_idx = _sample_array_indices(
        len(expert_features),
        candidate_count,
        replace=len(expert_features) < candidate_count,
    )
    generator_candidate_idx = _sample_array_indices(
        len(generator_features),
        candidate_count,
        replace=len(generator_features) < candidate_count,
    )
    expert_candidates = expert_features[expert_candidate_idx].astype(np.float32, copy=False)
    generator_candidates = generator_features[generator_candidate_idx].astype(np.float32, copy=False)
    expert_scores = _score_discriminator_candidates(discriminator, expert_candidates, device)
    generator_scores = _score_discriminator_candidates(discriminator, generator_candidates, device)

    selected_fraction = min(1.0, max(0.0, float(getattr(cfg, "hard_example_selected_fraction", 0.35))))
    min_samples = max(1, int(getattr(cfg, "hard_example_min_samples", 4_096)))
    selected_count = min(candidate_count, max(min_samples, int(np.ceil(selected_fraction * candidate_count))))
    if selected_fraction < 1.0 and candidate_count > 1:
        selected_count = min(selected_count, candidate_count - 1)
    loss_type = str(getattr(cfg, "discriminator_loss", "bce")).lower()
    expert_hardness = -expert_scores
    generator_hardness = generator_scores
    if loss_type != "wgan_gp":
        expert_hardness = -expert_scores
        generator_hardness = generator_scores
    expert_selected_idx = _soft_hard_indices(
        expert_hardness,
        selected_count=selected_count,
        selected_fraction=selected_fraction,
        uniform_mix=float(getattr(cfg, "hard_example_uniform_mix", 0.25)),
        temperature=float(getattr(cfg, "hard_example_temperature", 1.0)),
    )
    generator_selected_idx = _soft_hard_indices(
        generator_hardness,
        selected_count=selected_count,
        selected_fraction=selected_fraction,
        uniform_mix=float(getattr(cfg, "hard_example_uniform_mix", 0.25)),
        temperature=float(getattr(cfg, "hard_example_temperature", 1.0)),
    )
    selected_count = min(len(expert_selected_idx), len(generator_selected_idx))
    expert_selected_idx = expert_selected_idx[:selected_count]
    generator_selected_idx = generator_selected_idx[:selected_count]
    return (
        expert_candidates[expert_selected_idx],
        generator_candidates[generator_selected_idx],
        {
            "hard_selector_enabled": 1.0,
            "hard_selector_candidate_samples": float(candidate_count),
            "hard_selector_selected_samples": float(selected_count),
            "hard_selector_selected_fraction": float(selected_count) / float(max(1, candidate_count)),
            "hard_selector_expert_full_score_mean": float(expert_scores.mean()) if expert_scores.size else float("nan"),
            "hard_selector_gen_full_score_mean": float(generator_scores.mean()) if generator_scores.size else float("nan"),
            "hard_selector_expert_selected_score_mean": (
                float(expert_scores[expert_selected_idx].mean()) if selected_count else float("nan")
            ),
            "hard_selector_gen_selected_score_mean": (
                float(generator_scores[generator_selected_idx].mean()) if selected_count else float("nan")
            ),
        },
    )

def update_discriminator(
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    expert_features: np.ndarray,
    generator_features: np.ndarray,
    cfg: PSGAILConfig,
    device: torch.device,
) -> dict[str, float]:
    expert, generator_train_features, selector_stats = select_hard_discriminator_examples(
        discriminator,
        expert_features,
        generator_features,
        cfg,
        device,
    )
    expert_tensor = _as_device_tensor(expert.astype(np.float32, copy=False), dtype=torch.float32, device=device)
    generator_tensor = _as_device_tensor(
        generator_train_features.astype(np.float32, copy=False),
        dtype=torch.float32,
        device=device,
    )
    discriminator.train()
    losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []
    cgail_penalties: list[torch.Tensor] = []
    wgan_losses: list[torch.Tensor] = []
    gradient_penalties: list[torch.Tensor] = []
    prob_means: list[torch.Tensor] = []
    prob_stds: list[torch.Tensor] = []
    expert_prob_means: list[torch.Tensor] = []
    gen_prob_means: list[torch.Tensor] = []
    expert_score_means: list[torch.Tensor] = []
    gen_score_means: list[torch.Tensor] = []
    critic_gaps: list[torch.Tensor] = []
    expert_accs: list[torch.Tensor] = []
    gen_accs: list[torch.Tensor] = []
    expert_centered_accs: list[torch.Tensor] = []
    gen_centered_accs: list[torch.Tensor] = []
    expert_positive_fracs: list[torch.Tensor] = []
    gen_negative_fracs: list[torch.Tensor] = []
    cgail_k = max(0.0, float(getattr(cfg, "cgail_k", 0.0)))
    loss_type = str(getattr(cfg, "discriminator_loss", "bce")).lower()
    batch_size = max(1, int(cfg.disc_batch_size))
    num_pairs = int(generator_tensor.shape[0])
    if loss_type not in {"bce", "wgan_gp"}:
        raise ValueError(f"Unsupported discriminator_loss={loss_type!r}. Expected 'bce' or 'wgan_gp'.")

    if loss_type == "wgan_gp":
        gp_lambda = float(getattr(cfg, "wgan_gp_lambda", 2.0))
        for _ in range(int(cfg.disc_updates_per_round)):
            permutation = torch.randperm(num_pairs, device=device)
            for start in range(0, num_pairs, batch_size):
                batch_idx = permutation[start : start + batch_size]
                batch_expert = expert_tensor[batch_idx]
                batch_generator = generator_tensor[batch_idx]
                expert_scores = discriminator(batch_expert)
                gen_scores = discriminator(batch_generator)
                wgan_loss = gen_scores.mean() - expert_scores.mean()
                gradient_penalty = _wgan_gradient_penalty(
                    discriminator,
                    batch_expert,
                    batch_generator,
                    gp_lambda=gp_lambda,
                )
                loss = wgan_loss + gradient_penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    scores = torch.cat([expert_scores, gen_scores], dim=0)
                    probs = torch.sigmoid(scores)
                    centered_threshold = 0.5 * (expert_scores.mean() + gen_scores.mean())
                    prob_means.append(probs.mean().detach())
                    prob_stds.append(probs.std(unbiased=False).detach())
                    expert_prob_means.append(torch.sigmoid(expert_scores).mean().detach())
                    gen_prob_means.append(torch.sigmoid(gen_scores).mean().detach())
                    expert_score_means.append(expert_scores.mean().detach())
                    gen_score_means.append(gen_scores.mean().detach())
                    critic_gaps.append((expert_scores.mean() - gen_scores.mean()).detach())
                    expert_positive_fracs.append((expert_scores > 0.0).float().mean().detach())
                    gen_negative_fracs.append((gen_scores < 0.0).float().mean().detach())
                    expert_centered = (expert_scores > centered_threshold).float().mean().detach()
                    gen_centered = (gen_scores < centered_threshold).float().mean().detach()
                    expert_accs.append(expert_centered)
                    gen_accs.append(gen_centered)
                    expert_centered_accs.append(expert_centered)
                    gen_centered_accs.append(gen_centered)
                losses.append(loss.detach())
                wgan_losses.append(wgan_loss.detach())
                gradient_penalties.append(gradient_penalty.detach())
    else:
        x = torch.cat([expert_tensor, generator_tensor], dim=0)
        expert_label = float(cfg.disc_expert_label)
        generator_label = float(cfg.disc_generator_label)
        y = torch.cat(
            [
                torch.full((len(expert_tensor),), expert_label, dtype=torch.float32, device=device),
                torch.full((len(generator_tensor),), generator_label, dtype=torch.float32, device=device),
            ],
            dim=0,
        )
        num_samples = int(x.shape[0])
        for _ in range(int(cfg.disc_updates_per_round)):
            permutation = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, batch_size):
                batch_idx = permutation[start : start + batch_size]
                batch_x = x[batch_idx]
                batch_y = y[batch_idx]
                logits = discriminator(batch_x)
                bce_loss = F.binary_cross_entropy_with_logits(logits, batch_y)
                probs = torch.sigmoid(logits)
                cgail_penalty = 0.5 * cgail_k * torch.square(probs - 0.5).mean()
                loss = bce_loss + cgail_penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred = probs >= 0.5
                    expert_mask = batch_y > 0.5
                    gen_mask = batch_y < 0.5
                    prob_means.append(probs.mean().detach())
                    prob_stds.append(probs.std(unbiased=False).detach())
                    if expert_mask.any():
                        expert_accs.append((pred[expert_mask] == 1).float().mean().detach())
                        expert_centered_accs.append((pred[expert_mask] == 1).float().mean().detach())
                        expert_prob_means.append(probs[expert_mask].mean().detach())
                        expert_score_means.append(logits[expert_mask].mean().detach())
                    if gen_mask.any():
                        gen_accs.append((pred[gen_mask] == 0).float().mean().detach())
                        gen_centered_accs.append((pred[gen_mask] == 0).float().mean().detach())
                        gen_prob_means.append(probs[gen_mask].mean().detach())
                        gen_score_means.append(logits[gen_mask].mean().detach())
                    if expert_mask.any() and gen_mask.any():
                        critic_gaps.append(
                            (logits[expert_mask].mean() - logits[gen_mask].mean()).detach()
                        )
                losses.append(loss.detach())
                bce_losses.append(bce_loss.detach())
                cgail_penalties.append(cgail_penalty.detach())

    def mean_or_nan(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().cpu().item()) if values else float("nan")

    return {
        "disc_loss": mean_or_nan(losses),
        "disc_bce_loss": mean_or_nan(bce_losses),
        "cgail_penalty": mean_or_nan(cgail_penalties),
        "wgan_loss": mean_or_nan(wgan_losses),
        "gradient_penalty": mean_or_nan(gradient_penalties),
        "expert_score_mean": mean_or_nan(expert_score_means),
        "gen_score_mean": mean_or_nan(gen_score_means),
        "critic_gap": mean_or_nan(critic_gaps),
        "disc_prob_mean": mean_or_nan(prob_means),
        "disc_prob_std": mean_or_nan(prob_stds),
        "expert_prob_mean": mean_or_nan(expert_prob_means),
        "gen_prob_mean": mean_or_nan(gen_prob_means),
        "expert_acc": mean_or_nan(expert_accs),
        "gen_acc": mean_or_nan(gen_accs),
        "expert_centered_acc": mean_or_nan(expert_centered_accs),
        "gen_centered_acc": mean_or_nan(gen_centered_accs),
        "expert_positive_frac": mean_or_nan(expert_positive_fracs),
        "gen_negative_frac": mean_or_nan(gen_negative_fracs),
        **selector_stats,
    }

__all__ = [
    '_wgan_gradient_penalty',
    '_sample_array_indices',
    '_score_discriminator_candidates',
    '_soft_hard_indices',
    'select_hard_discriminator_examples',
    'update_discriminator'
]
