"""Sequence-aware behaviour cloning for recurrent transformer policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SequenceWindow:
    trajectory_id: str
    context_indices: np.ndarray
    train_indices: np.ndarray


@dataclass
class RecurrentBCResult:
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    history: list[dict[str, Any]]
    summary: dict[str, float]
    split_trajectory_ids: dict[str, list[str]]


def trajectory_segments(transitions: Any) -> list[tuple[str, np.ndarray]]:
    """Return ordered trajectory segments without crossing terminal boundaries."""
    trajectory_ids = np.asarray(transitions.trajectory_ids, dtype=object)
    timesteps = np.asarray(transitions.timesteps, dtype=np.int64)
    dones = np.asarray(transitions.dones, dtype=bool)
    if not (len(trajectory_ids) == len(timesteps) == len(dones)):
        raise ValueError("Trajectory ids, timesteps, and dones must have equal length.")

    segments: list[tuple[str, np.ndarray]] = []
    for trajectory_id in sorted({str(value) for value in trajectory_ids.tolist()}):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        indices = indices[np.argsort(timesteps[indices], kind="stable")]
        if indices.size == 0:
            continue
        segment_start = 0
        for position in range(1, len(indices)):
            previous = int(indices[position - 1])
            current = int(indices[position])
            boundary = bool(dones[previous]) or int(timesteps[current]) != int(timesteps[previous]) + 1
            if boundary:
                segment = indices[segment_start:position]
                if segment.size:
                    segments.append((trajectory_id, segment.astype(np.int64, copy=False)))
                segment_start = position
        segment = indices[segment_start:]
        if segment.size:
            segments.append((trajectory_id, segment.astype(np.int64, copy=False)))
    return segments


def split_trajectory_ids(
    transitions: Any,
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    """Create a deterministic trajectory-level train/validation/test split."""
    identifiers = sorted({trajectory_id for trajectory_id, _indices in trajectory_segments(transitions)})
    if len(identifiers) < 3:
        raise ValueError("Recurrent BC requires at least three trajectories for leakage-free splits.")
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than one.")

    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(identifiers, dtype=object)[rng.permutation(len(identifiers))].tolist()
    train_count = max(1, int(round(len(shuffled) * train_fraction)))
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    if train_count + validation_count >= len(shuffled):
        train_count = max(1, len(shuffled) - 2)
        validation_count = 1
    return {
        "train": sorted(str(value) for value in shuffled[:train_count]),
        "validation": sorted(str(value) for value in shuffled[train_count : train_count + validation_count]),
        "test": sorted(str(value) for value in shuffled[train_count + validation_count :]),
    }


def build_sequence_windows(
    transitions: Any,
    trajectory_ids: list[str] | set[str],
    *,
    sequence_length: int,
    context_length: int,
) -> list[SequenceWindow]:
    """Build non-overlapping training windows with preceding warm-up context."""
    keep = set(str(value) for value in trajectory_ids)
    sequence_length = max(1, int(sequence_length))
    context_length = max(0, int(context_length))
    windows: list[SequenceWindow] = []
    for trajectory_id, indices in trajectory_segments(transitions):
        if trajectory_id not in keep:
            continue
        for start in range(0, len(indices), sequence_length):
            end = min(len(indices), start + sequence_length)
            context_start = max(0, start - context_length)
            windows.append(
                SequenceWindow(
                    trajectory_id=trajectory_id,
                    context_indices=indices[context_start:start].astype(np.int64, copy=False),
                    train_indices=indices[start:end].astype(np.int64, copy=False),
                )
            )
    if not windows:
        raise RuntimeError("No recurrent BC sequence windows were built.")
    return windows


def _shift_memory(memory: torch.Tensor, new_memory: torch.Tensor) -> torch.Tensor:
    return torch.cat([memory[:, 1:], new_memory.unsqueeze(1)], dim=1)


def _batched_sequence_loss(
    policy: torch.nn.Module,
    observations: np.ndarray,
    actions: np.ndarray,
    windows: list[SequenceWindow],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Warm up memory, then unroll differentiably across each training window."""
    if not windows:
        raise ValueError("At least one sequence window is required.")
    batch_size = len(windows)
    memory = policy.initial_memory(batch_size, device=device, dtype=torch.float32)
    fallback = np.asarray([int(window.train_indices[0]) for window in windows], dtype=np.int64)

    max_context = max(len(window.context_indices) for window in windows)
    with torch.no_grad():
        for offset in range(-max_context, 0):
            active_np = np.asarray([len(window.context_indices) + offset >= 0 for window in windows], dtype=bool)
            if not bool(active_np.any()):
                continue
            step_indices = fallback.copy()
            for row, window in enumerate(windows):
                position = len(window.context_indices) + offset
                if position >= 0:
                    step_indices[row] = int(window.context_indices[position])
            obs = torch.as_tensor(observations[step_indices], dtype=torch.float32, device=device)
            _predictions, _values, step_memory = policy(obs, memory=memory, return_memory=True)
            shifted = _shift_memory(memory, step_memory)
            active = torch.as_tensor(active_np, dtype=torch.bool, device=device)[:, None, None, None]
            memory = torch.where(active, shifted, memory)

    squared_error_sum = torch.zeros((), dtype=torch.float32, device=device)
    absolute_error_sum = torch.zeros((), dtype=torch.float32, device=device)
    valid_count = 0
    max_steps = max(len(window.train_indices) for window in windows)
    for step in range(max_steps):
        active_np = np.asarray([step < len(window.train_indices) for window in windows], dtype=bool)
        step_indices = fallback.copy()
        for row, window in enumerate(windows):
            if active_np[row]:
                step_indices[row] = int(window.train_indices[step])
        obs = torch.as_tensor(observations[step_indices], dtype=torch.float32, device=device)
        target = torch.as_tensor(actions[step_indices], dtype=torch.float32, device=device)
        predictions, _values, step_memory = policy(obs, memory=memory, return_memory=True)
        active = torch.as_tensor(active_np, dtype=torch.bool, device=device)
        error = predictions[active] - target[active]
        squared_error_sum = squared_error_sum + error.square().mean(dim=1).sum()
        absolute_error_sum = absolute_error_sum + error.abs().mean(dim=1).sum()
        valid_count += int(active_np.sum())
        shifted = _shift_memory(memory, step_memory)
        memory = torch.where(active[:, None, None, None], shifted, memory)
    if valid_count <= 0:
        raise RuntimeError("Recurrent BC batch contained no valid timesteps.")
    return squared_error_sum / valid_count, absolute_error_sum / valid_count, valid_count


def evaluate_recurrent_bc(
    policy: torch.nn.Module,
    transitions: Any,
    windows: list[SequenceWindow],
    *,
    device: torch.device,
    micro_batch_sequences: int,
) -> dict[str, float]:
    observations = np.asarray(transitions.policy_observations, dtype=np.float32)
    actions = np.clip(np.asarray(transitions.actions_continuous_env, dtype=np.float32), -1.0, 1.0)
    micro_batch_sequences = max(1, int(micro_batch_sequences))
    was_training = policy.training
    policy.eval()
    weighted_mse = 0.0
    weighted_mae = 0.0
    total = 0
    with torch.no_grad():
        for start in range(0, len(windows), micro_batch_sequences):
            mse, mae, count = _batched_sequence_loss(
                policy,
                observations,
                actions,
                windows[start : start + micro_batch_sequences],
                device=device,
            )
            weighted_mse += float(mse.detach().cpu()) * count
            weighted_mae += float(mae.detach().cpu()) * count
            total += count
    if was_training:
        policy.train()
    return {
        "mse": weighted_mse / max(1, total),
        "mae": weighted_mae / max(1, total),
        "samples": float(total),
        "windows": float(len(windows)),
    }


def _window_indices(windows: list[SequenceWindow]) -> np.ndarray:
    return np.concatenate([window.train_indices for window in windows]).astype(np.int64, copy=False)


def train_recurrent_behavior_clone(
    policy: torch.nn.Module,
    transitions: Any,
    *,
    device: torch.device,
    seed: int,
    split_seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    sequence_length: int,
    sequences_per_batch: int,
    micro_batch_sequences: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    max_grad_norm: float = 0.5,
    early_stopping_patience: int = 10,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RecurrentBCResult:
    """Train a recurrent policy and select the checkpoint by validation MSE."""
    if not bool(getattr(policy, "supports_recurrent_memory", False)):
        raise TypeError("Sequence-aware BC requires a recurrent policy.")
    observations = np.asarray(transitions.policy_observations, dtype=np.float32)
    actions = np.clip(np.asarray(transitions.actions_continuous_env, dtype=np.float32), -1.0, 1.0)
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions):
        raise ValueError("Recurrent BC requires aligned rank-2 observations and actions.")

    split_ids = split_trajectory_ids(
        transitions,
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
        seed=int(split_seed),
    )
    context_length = int(getattr(policy, "memory_context_length", sequence_length))
    split_windows = {
        name: build_sequence_windows(
            transitions,
            identifiers,
            sequence_length=int(sequence_length),
            context_length=context_length,
        )
        for name, identifiers in split_ids.items()
    }
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed) + 1701)
    sequences_per_batch = max(1, int(sequences_per_batch))
    micro_batch_sequences = max(1, min(sequences_per_batch, int(micro_batch_sequences)))

    train_actions = actions[_window_indices(split_windows["train"])]
    validation_actions = actions[_window_indices(split_windows["validation"])]
    action_mean = train_actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    baseline_validation_mse = float(np.mean(np.square(validation_actions - action_mean)))
    initial_validation = evaluate_recurrent_bc(
        policy,
        transitions,
        split_windows["validation"],
        device=device,
        micro_batch_sequences=micro_batch_sequences,
    )

    best_epoch = 0
    best_validation_mse = float("inf")
    best_state_dict: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    for epoch in range(1, max(1, int(epochs)) + 1):
        policy.train()
        order = rng.permutation(len(split_windows["train"]))
        ordered = [split_windows["train"][int(index)] for index in order]
        epoch_mse = 0.0
        epoch_mae = 0.0
        epoch_samples = 0
        for batch_start in range(0, len(ordered), sequences_per_batch):
            batch = ordered[batch_start : batch_start + sequences_per_batch]
            batch_samples = sum(len(window.train_indices) for window in batch)
            optimizer.zero_grad(set_to_none=True)
            batch_mse = 0.0
            batch_mae = 0.0
            for micro_start in range(0, len(batch), micro_batch_sequences):
                micro = batch[micro_start : micro_start + micro_batch_sequences]
                mse, mae, count = _batched_sequence_loss(
                    policy,
                    observations,
                    actions,
                    micro,
                    device=device,
                )
                (mse * (float(count) / float(batch_samples))).backward()
                batch_mse += float(mse.detach().cpu()) * count
                batch_mae += float(mae.detach().cpu()) * count
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(max_grad_norm))
            optimizer.step()
            epoch_mse += batch_mse
            epoch_mae += batch_mae
            epoch_samples += batch_samples

        validation = evaluate_recurrent_bc(
            policy,
            transitions,
            split_windows["validation"],
            device=device,
            micro_batch_sequences=micro_batch_sequences,
        )
        validation_mse = float(validation["mse"])
        validation_skill = 1.0 - validation_mse / max(baseline_validation_mse, 1.0e-12)
        improved = validation_mse < best_validation_mse
        row: dict[str, Any] = {
            "epoch": float(epoch),
            "train_mse": epoch_mse / max(1, epoch_samples),
            "train_mae": epoch_mae / max(1, epoch_samples),
            "validation_mse": validation_mse,
            "validation_mae": float(validation["mae"]),
            "validation_baseline_mse": baseline_validation_mse,
            "validation_skill": validation_skill,
            "is_best_so_far": improved,
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(dict(row))
        print(
            f"[recurrent bc {epoch:03d}/{int(epochs):03d}] "
            f"train_mse={row['train_mse']:.6f} train_mae={row['train_mae']:.6f} "
            f"val_mse={row['validation_mse']:.6f} val_mae={row['validation_mae']:.6f} "
            f"val_skill={row['validation_skill']:.6f} best={row['is_best_so_far']}"
        )
        if improved:
            best_validation_mse = row["validation_mse"]
            best_epoch = epoch
            best_state_dict = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if int(early_stopping_patience) > 0 and stale_epochs >= int(early_stopping_patience):
            print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
            break

    if not best_state_dict:
        raise RuntimeError("Recurrent BC did not produce a best checkpoint.")
    policy.load_state_dict(best_state_dict, strict=True)
    policy.to(device)
    train_metrics = evaluate_recurrent_bc(
        policy,
        transitions,
        split_windows["train"],
        device=device,
        micro_batch_sequences=micro_batch_sequences,
    )
    validation_metrics = evaluate_recurrent_bc(
        policy,
        transitions,
        split_windows["validation"],
        device=device,
        micro_batch_sequences=micro_batch_sequences,
    )
    test_metrics = evaluate_recurrent_bc(
        policy,
        transitions,
        split_windows["test"],
        device=device,
        micro_batch_sequences=micro_batch_sequences,
    )
    skill = 1.0 - float(validation_metrics["mse"]) / max(baseline_validation_mse, 1.0e-12)
    initial_validation_mse = float(initial_validation["mse"])
    summary = {
        "best_epoch": float(best_epoch),
        "configured_epochs": float(epochs),
        "completed_epochs": float(len(history)),
        "initial_validation_mse": initial_validation_mse,
        "initial_validation_mae": float(initial_validation["mae"]),
        "relative_validation_improvement": (
            initial_validation_mse - float(validation_metrics["mse"])
        )
        / max(initial_validation_mse, 1.0e-12),
        "train_mse": float(train_metrics["mse"]),
        "train_mae": float(train_metrics["mae"]),
        "validation_mse": float(validation_metrics["mse"]),
        "validation_mae": float(validation_metrics["mae"]),
        "test_mse": float(test_metrics["mse"]),
        "test_mae": float(test_metrics["mae"]),
        "validation_baseline_mse": baseline_validation_mse,
        "validation_skill": float(skill),
        "train_samples": float(train_metrics["samples"]),
        "validation_samples": float(validation_metrics["samples"]),
        "test_samples": float(test_metrics["samples"]),
        "train_windows": float(train_metrics["windows"]),
        "validation_windows": float(validation_metrics["windows"]),
        "test_windows": float(test_metrics["windows"]),
    }
    return RecurrentBCResult(
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        history=history,
        summary=summary,
        split_trajectory_ids=split_ids,
    )
