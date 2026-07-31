"""Sequence-aware behaviour cloning for recurrent transformer policies."""

from __future__ import annotations

from copy import deepcopy
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
    summary: dict[str, Any]
    split_trajectory_ids: dict[str, list[str]]


@dataclass(frozen=True)
class PreparedRecurrentBCData:
    """Immutable, reusable recurrent-BC arrays and trajectory windows.

    Preparing this object is deliberately separate from optimization so a
    multi-trial process can load a heavy expert dataset once and guarantee that
    every candidate sees the same sampled transitions and held-out split.
    """

    transitions: Any
    observations: np.ndarray
    actions: np.ndarray
    split_trajectory_ids: dict[str, list[str]]
    split_windows: dict[str, list[SequenceWindow]]
    validation_baseline_mse: float
    warmup_mode: str


@dataclass(frozen=True)
class ExplicitSplitRecurrentTransitions:
    """Minimal concatenated transition view with source-defined split labels."""

    policy_observations: np.ndarray
    actions_continuous_env: np.ndarray
    trajectory_ids: np.ndarray
    timesteps: np.ndarray
    dones: np.ndarray
    metadata: dict[str, Any]


def _validated_normalized_actions(actions: Any, *, context: str) -> np.ndarray:
    """Return normalized action labels without silently changing their values."""

    values = np.asarray(actions, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context} contains non-finite normalized actions.")
    outside = (values < -1.0) | (values > 1.0)
    if bool(np.any(outside)):
        first = tuple(int(index) for index in np.argwhere(outside)[0])
        raise ValueError(
            f"{context} violates the normalized [-1, 1] action contract at "
            f"index {first}: {float(values[first])}. Labels are never clipped."
        )
    return values


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
    warmup_mode: str = "full_prefix",
) -> list[SequenceWindow]:
    """Build non-overlapping targets with rollout-faithful memory warm-up.

    Recurrent memory contains summaries generated by earlier recurrent states,
    not merely the last ``context_length`` raw observations.  Replaying only a
    bounded raw-observation suffix from zero therefore does not reproduce the
    memory used in a live rollout.  ``full_prefix`` is the qualifying mode: it
    reconstructs the exact recurrent state by replaying every earlier row in
    the trajectory segment.  ``bounded_raw_history`` is retained only for
    explicit legacy diagnostics.
    """
    keep = set(str(value) for value in trajectory_ids)
    sequence_length = max(1, int(sequence_length))
    context_length = max(0, int(context_length))
    warmup_mode = str(warmup_mode).strip().lower()
    if warmup_mode not in {"full_prefix", "bounded_raw_history"}:
        raise ValueError(
            "warmup_mode must be 'full_prefix' or "
            f"'bounded_raw_history', got {warmup_mode!r}."
        )
    windows: list[SequenceWindow] = []
    for trajectory_id, indices in trajectory_segments(transitions):
        if trajectory_id not in keep:
            continue
        for start in range(0, len(indices), sequence_length):
            end = min(len(indices), start + sequence_length)
            context_start = (
                0
                if warmup_mode == "full_prefix"
                else max(0, start - context_length)
            )
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


def _center_sequence_steps(
    sequence_steps: list[list[torch.Tensor]],
) -> torch.Tensor:
    """Center timesteps within each sequence before pooling them."""
    centered: list[torch.Tensor] = []
    for steps in sequence_steps:
        if not steps:
            raise ValueError("Every recurrent BC sequence must contain a timestep.")
        values = torch.stack(steps, dim=0)
        centered.append(values - values.mean(dim=0))
    return torch.cat(centered, dim=0)


def mirror_policy_observations(
    observations: np.ndarray,
) -> np.ndarray:
    """Reflect the frozen 322-D policy observation across the world x-axis.

    The lidar is world-fixed, so cell ``i`` maps to ``127-i``.  The lane
    camera is ego-aligned, so its angular bins reverse and lateral coordinates
    change sign.  Vehicle length and speed are invariant; heading changes
    sign.  This is a training-data transformation, never an evaluation-time
    action modification.
    """

    values = np.asarray(observations, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != 322:
        raise ValueError(
            "Mirror augmentation requires the frozen 322-D policy observation."
        )
    result = np.array(values, dtype=np.float32, copy=True)
    lidar = result[..., :256].reshape(*result.shape[:-1], 128, 2)
    result[..., :256] = np.flip(lidar, axis=-2).reshape(
        *result.shape[:-1],
        256,
    )
    camera = result[..., 256:319].reshape(*result.shape[:-1], 21, 3)
    reflected_camera = np.flip(camera, axis=-2).copy()
    reflected_camera[..., 2] *= -1.0
    result[..., 256:319] = reflected_camera.reshape(
        *result.shape[:-1],
        63,
    )
    result[..., 321] *= -1.0
    return result


def mirror_normalized_actions(actions: np.ndarray) -> np.ndarray:
    """Reflect normalized ``[acceleration, steering]`` actions."""

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError(
            "Mirror augmentation requires [acceleration, steering] actions."
        )
    result = np.array(values, dtype=np.float32, copy=True)
    result[..., 1] *= -1.0
    return result


def _mixture_mean_variance(
    original: np.ndarray,
    transformed: np.ndarray,
    *,
    transformed_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute moments of the actual stochastic augmentation distribution."""

    source = np.asarray(original, dtype=np.float64)
    augmented = np.asarray(transformed, dtype=np.float64)
    probability = float(transformed_probability)
    if (
        source.ndim != 2
        or augmented.shape != source.shape
        or source.shape[0] == 0
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(augmented))
    ):
        raise ValueError(
            "Augmentation moment inputs must be aligned, non-empty, finite "
            "rank-2 arrays."
        )
    if not 0.0 <= probability <= 1.0:
        raise ValueError("transformed_probability must be between zero and one.")
    source_weight = 1.0 - probability
    mean = (
        source_weight * source.mean(axis=0, dtype=np.float64)
        + probability * augmented.mean(axis=0, dtype=np.float64)
    )
    second_moment = (
        source_weight * np.square(source).mean(axis=0, dtype=np.float64)
        + probability
        * np.square(augmented).mean(axis=0, dtype=np.float64)
    )
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    return mean, variance


def _batched_sequence_loss(
    policy: torch.nn.Module,
    observations: np.ndarray,
    actions: np.ndarray,
    windows: list[SequenceWindow],
    *,
    device: torch.device,
    diagnostics: dict[str, Any] | None = None,
    action_loss_weights: np.ndarray | None = None,
    correlation_loss_weight: float = 0.0,
    variance_loss_weight: float = 0.0,
    minimum_prediction_std_ratios: np.ndarray | None = None,
    mirror_mask: np.ndarray | None = None,
    loss_diagnostics: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Warm up memory, then unroll differentiably across each training window.

    The optional moment losses are deliberately computed inside each optimizer
    micro-batch.  MSE alone can select a nearly constant conditional-mean
    policy on imbalanced driving data.  The correlation term rewards temporal
    action shape after centering every sequence independently, so unrelated
    trajectory offsets cannot satisfy it.  The one-sided standard-deviation
    term prevents a low-variance shortcut without rewarding arbitrarily large
    action variance.
    """
    if not windows:
        raise ValueError("At least one sequence window is required.")
    batch_size = len(windows)
    if mirror_mask is None:
        mirrored = np.zeros(batch_size, dtype=bool)
    else:
        mirrored = np.asarray(mirror_mask, dtype=bool)
        if mirrored.shape != (batch_size,):
            raise ValueError(
                "mirror_mask must contain one boolean per sequence window."
            )

    def _selected_observations(indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(observations[indices], dtype=np.float32)
        if not bool(mirrored.any()):
            return selected
        selected = np.array(selected, copy=True)
        selected[mirrored] = mirror_policy_observations(
            selected[mirrored]
        )
        return selected

    def _selected_actions(indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(actions[indices], dtype=np.float32)
        if not bool(mirrored.any()):
            return selected
        selected = np.array(selected, copy=True)
        selected[mirrored] = mirror_normalized_actions(
            selected[mirrored]
        )
        return selected

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
            obs = torch.as_tensor(
                _selected_observations(step_indices),
                dtype=torch.float32,
                device=device,
            )
            _predictions, _values, step_memory = policy(obs, memory=memory, return_memory=True)
            shifted = _shift_memory(memory, step_memory)
            active = torch.as_tensor(active_np, dtype=torch.bool, device=device)[:, None, None, None]
            memory = torch.where(active, shifted, memory)

    squared_error_sum = torch.zeros((), dtype=torch.float32, device=device)
    absolute_error_sum = torch.zeros((), dtype=torch.float32, device=device)
    active_prediction_steps: list[torch.Tensor] = []
    active_target_steps: list[torch.Tensor] = []
    sequence_prediction_steps: list[list[torch.Tensor]] = [
        [] for _window in windows
    ]
    sequence_target_steps: list[list[torch.Tensor]] = [
        [] for _window in windows
    ]
    valid_count = 0
    max_steps = max(len(window.train_indices) for window in windows)
    for step in range(max_steps):
        active_np = np.asarray([step < len(window.train_indices) for window in windows], dtype=bool)
        step_indices = fallback.copy()
        for row, window in enumerate(windows):
            if active_np[row]:
                step_indices[row] = int(window.train_indices[step])
        obs = torch.as_tensor(
            _selected_observations(step_indices),
            dtype=torch.float32,
            device=device,
        )
        target = torch.as_tensor(
            _selected_actions(step_indices),
            dtype=torch.float32,
            device=device,
        )
        predictions, _values, step_memory = policy(obs, memory=memory, return_memory=True)
        active = torch.as_tensor(active_np, dtype=torch.bool, device=device)
        active_predictions = predictions[active]
        active_targets = target[active]
        active_prediction_steps.append(active_predictions)
        active_target_steps.append(active_targets)
        for row in np.flatnonzero(active_np):
            sequence_prediction_steps[int(row)].append(predictions[int(row)])
            sequence_target_steps[int(row)].append(target[int(row)])
        error = active_predictions - active_targets
        if action_loss_weights is None:
            squared_error = error.square().mean(dim=1)
        else:
            weights = torch.as_tensor(
                action_loss_weights,
                dtype=error.dtype,
                device=error.device,
            )
            if weights.shape != (error.shape[1],):
                raise ValueError(
                    "action_loss_weights must match the action dimension: "
                    f"{tuple(weights.shape)} != {(error.shape[1],)}."
                )
            squared_error = (error.square() * weights).mean(dim=1)
        squared_error_sum = squared_error_sum + squared_error.sum()
        absolute_error_sum = absolute_error_sum + error.abs().mean(dim=1).sum()
        valid_count += int(active_np.sum())
        if diagnostics is not None:
            detached_predictions = active_predictions.detach().to(dtype=torch.float64)
            detached_targets = active_targets.detach().to(dtype=torch.float64)
            diagnostics["count"] += int(detached_predictions.shape[0])
            diagnostics["prediction_sum"].add_(detached_predictions.sum(dim=0))
            diagnostics["prediction_square_sum"].add_(detached_predictions.square().sum(dim=0))
            diagnostics["target_sum"].add_(detached_targets.sum(dim=0))
            diagnostics["target_square_sum"].add_(detached_targets.square().sum(dim=0))
            diagnostics["prediction_target_sum"].add_((detached_predictions * detached_targets).sum(dim=0))
            diagnostics["squared_error_sum"].add_((detached_predictions - detached_targets).square().sum(dim=0))
            diagnostics["absolute_error_sum"].add_((detached_predictions - detached_targets).abs().sum(dim=0))
            diagnostics["saturated_count"].add_((detached_predictions.abs() >= 0.98).sum(dim=0))
        shifted = _shift_memory(memory, step_memory)
        memory = torch.where(active[:, None, None, None], shifted, memory)
    if valid_count <= 0:
        raise RuntimeError("Recurrent BC batch contained no valid timesteps.")
    base_mse = squared_error_sum / valid_count
    mae = absolute_error_sum / valid_count
    objective = base_mse
    correlation_loss = torch.zeros((), dtype=base_mse.dtype, device=device)
    variance_loss = torch.zeros((), dtype=base_mse.dtype, device=device)
    prediction_std_ratio = torch.zeros(
        actions.shape[1], dtype=base_mse.dtype, device=device
    )
    correlation = torch.zeros_like(prediction_std_ratio)
    if float(correlation_loss_weight) > 0.0 or float(variance_loss_weight) > 0.0:
        batch_predictions = torch.cat(active_prediction_steps, dim=0)
        batch_targets = torch.cat(active_target_steps, dim=0)
        prediction_centered = batch_predictions - batch_predictions.mean(dim=0)
        target_centered = batch_targets - batch_targets.mean(dim=0)
        prediction_variance = prediction_centered.square().mean(dim=0)
        target_variance = target_centered.square().mean(dim=0)
        temporal_predictions = _center_sequence_steps(sequence_prediction_steps)
        temporal_targets = _center_sequence_steps(sequence_target_steps)
        temporal_prediction_variance = temporal_predictions.square().mean(dim=0)
        temporal_target_variance = temporal_targets.square().mean(dim=0)
        covariance = (temporal_predictions * temporal_targets).mean(dim=0)
        epsilon = torch.finfo(batch_predictions.dtype).eps
        prediction_std = torch.sqrt(prediction_variance + epsilon)
        target_std = torch.sqrt(target_variance + epsilon)
        temporal_prediction_std = torch.sqrt(
            temporal_prediction_variance + epsilon
        )
        temporal_target_std = torch.sqrt(temporal_target_variance + epsilon)
        correlation = covariance / (
            temporal_prediction_std * temporal_target_std
        )
        correlation = correlation.clamp(min=-1.0, max=1.0)
        prediction_std_ratio = prediction_std / target_std
        correlation_loss = (1.0 - correlation).mean()
        if minimum_prediction_std_ratios is None:
            minimum_ratios = torch.zeros_like(prediction_std_ratio)
        else:
            minimum_ratios = torch.as_tensor(
                minimum_prediction_std_ratios,
                dtype=prediction_std_ratio.dtype,
                device=prediction_std_ratio.device,
            )
            if minimum_ratios.shape != prediction_std_ratio.shape:
                raise ValueError(
                    "minimum_prediction_std_ratios must match the action dimension: "
                    f"{tuple(minimum_ratios.shape)} != {tuple(prediction_std_ratio.shape)}."
                )
        variance_loss = F.relu(minimum_ratios - prediction_std_ratio).square().mean()
        objective = (
            base_mse
            + float(correlation_loss_weight) * correlation_loss
            + float(variance_loss_weight) * variance_loss
        )
    if loss_diagnostics is not None:
        loss_diagnostics.update(
            {
                "objective": float(objective.detach().cpu()),
                "base_mse": float(base_mse.detach().cpu()),
                "correlation_loss": float(correlation_loss.detach().cpu()),
                "variance_loss": float(variance_loss.detach().cpu()),
                "prediction_std_ratio": prediction_std_ratio.detach().cpu().tolist(),
                "prediction_target_correlation": correlation.detach().cpu().tolist(),
            }
        )
    return objective, mae, valid_count


def evaluate_recurrent_bc(
    policy: torch.nn.Module,
    transitions: Any,
    windows: list[SequenceWindow],
    *,
    device: torch.device,
    micro_batch_sequences: int,
) -> dict[str, Any]:
    observations = np.asarray(transitions.policy_observations, dtype=np.float32)
    actions = _validated_normalized_actions(
        transitions.actions_continuous_env,
        context="Recurrent BC evaluation data",
    )
    micro_batch_sequences = max(1, int(micro_batch_sequences))
    was_training = policy.training
    policy.eval()
    weighted_mse = 0.0
    weighted_mae = 0.0
    total = 0
    action_dim = int(actions.shape[1])
    diagnostics: dict[str, Any] = {
        "count": 0,
        "prediction_sum": torch.zeros(action_dim, dtype=torch.float64, device=device),
        "prediction_square_sum": torch.zeros(action_dim, dtype=torch.float64, device=device),
        "target_sum": torch.zeros(action_dim, dtype=torch.float64, device=device),
        "target_square_sum": torch.zeros(action_dim, dtype=torch.float64, device=device),
        "prediction_target_sum": torch.zeros(action_dim, dtype=torch.float64, device=device),
        "squared_error_sum": torch.zeros(action_dim, dtype=torch.float64, device=device),
        "absolute_error_sum": torch.zeros(action_dim, dtype=torch.float64, device=device),
        "saturated_count": torch.zeros(action_dim, dtype=torch.float64, device=device),
    }
    with torch.no_grad():
        for start in range(0, len(windows), micro_batch_sequences):
            mse, mae, count = _batched_sequence_loss(
                policy,
                observations,
                actions,
                windows[start : start + micro_batch_sequences],
                device=device,
                diagnostics=diagnostics,
            )
            weighted_mse += float(mse.detach().cpu()) * count
            weighted_mae += float(mae.detach().cpu()) * count
            total += count
    if was_training:
        policy.train()
    diagnostic_count = max(1, int(diagnostics["count"]))
    for key in (
        "prediction_sum",
        "prediction_square_sum",
        "target_sum",
        "target_square_sum",
        "prediction_target_sum",
        "squared_error_sum",
        "absolute_error_sum",
        "saturated_count",
    ):
        diagnostics[key] = diagnostics[key].cpu().numpy()
    prediction_mean = diagnostics["prediction_sum"] / diagnostic_count
    target_mean = diagnostics["target_sum"] / diagnostic_count
    prediction_variance = np.maximum(
        diagnostics["prediction_square_sum"] / diagnostic_count - np.square(prediction_mean),
        0.0,
    )
    target_variance = np.maximum(
        diagnostics["target_square_sum"] / diagnostic_count - np.square(target_mean),
        0.0,
    )
    prediction_std = np.sqrt(prediction_variance)
    target_std = np.sqrt(target_variance)
    covariance = diagnostics["prediction_target_sum"] / diagnostic_count - prediction_mean * target_mean
    correlation_denominator = prediction_std * target_std
    correlation = np.divide(
        covariance,
        correlation_denominator,
        out=np.zeros_like(covariance),
        where=correlation_denominator > 1.0e-12,
    )
    prediction_std_ratio = np.divide(
        prediction_std,
        target_std,
        out=np.zeros_like(prediction_std),
        where=target_std > 1.0e-12,
    )
    return {
        "mse": weighted_mse / max(1, total),
        "mae": weighted_mae / max(1, total),
        "samples": float(total),
        "windows": float(len(windows)),
        "action_mse": (diagnostics["squared_error_sum"] / diagnostic_count).tolist(),
        "action_mae": (diagnostics["absolute_error_sum"] / diagnostic_count).tolist(),
        "prediction_mean": prediction_mean.tolist(),
        "prediction_std": prediction_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "prediction_std_ratio": prediction_std_ratio.tolist(),
        "prediction_target_correlation": correlation.tolist(),
        "prediction_saturation_fraction": (diagnostics["saturated_count"] / diagnostic_count).tolist(),
    }


def _window_indices(windows: list[SequenceWindow]) -> np.ndarray:
    return np.concatenate([window.train_indices for window in windows]).astype(np.int64, copy=False)


def prepare_recurrent_bc_data_from_explicit_splits(
    split_transitions: dict[str, Any],
    *,
    sequence_length: int,
    context_length: int = 32,
    warmup_mode: str = "full_prefix",
) -> PreparedRecurrentBCData:
    """Prepare BC windows from independently collected train/validation/test roots.

    Unlike :func:`prepare_recurrent_bc_data`, this function never creates a
    random split. Source trajectory ids are prefixed with their split name
    before concatenation so a vehicle recurring in separate time windows
    cannot accidentally join sequence context across those windows.
    """
    required = ("train", "validation")
    has_test = "test" in split_transitions
    expected = (*required, "test") if has_test else required
    if set(split_transitions) != set(expected):
        raise ValueError(
            "Explicit recurrent BC splits must contain train/validation and "
            "optionally test; "
            f"got {sorted(split_transitions)}."
        )

    observation_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    trajectory_id_parts: list[np.ndarray] = []
    timestep_parts: list[np.ndarray] = []
    done_parts: list[np.ndarray] = []
    source_metadata: dict[str, Any] = {}
    split_ids: dict[str, list[str]] = {}
    observation_dim: int | None = None
    action_dim: int | None = None

    for split in expected:
        transitions = split_transitions[split]
        observations = np.asarray(
            transitions.policy_observations,
            dtype=np.float32,
        )
        actions = np.asarray(
            transitions.actions_continuous_env,
            dtype=np.float32,
        )
        trajectory_ids = np.asarray(transitions.trajectory_ids, dtype=object)
        timesteps = np.asarray(transitions.timesteps, dtype=np.int64)
        dones = np.asarray(transitions.dones, dtype=bool)
        lengths = {
            len(observations),
            len(actions),
            len(trajectory_ids),
            len(timesteps),
            len(dones),
        }
        if (
            observations.ndim != 2
            or actions.ndim != 2
            or len(lengths) != 1
            or not len(observations)
        ):
            raise ValueError(
                f"Explicit split {split!r} requires non-empty aligned rank-2 "
                "observations/actions and rank-1 trajectory arrays."
            )
        if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Explicit split {split!r} contains non-finite observations or actions.")
        if observation_dim is None:
            observation_dim = int(observations.shape[1])
            action_dim = int(actions.shape[1])
        elif (
            int(observations.shape[1]) != observation_dim
            or int(actions.shape[1]) != action_dim
        ):
            raise ValueError(
                f"Explicit split {split!r} has incompatible dimensions "
                f"{observations.shape[1]}/{actions.shape[1]}; expected "
                f"{observation_dim}/{action_dim}."
            )

        prefixed_ids = np.asarray(
            [f"{split}:{value}" for value in trajectory_ids],
            dtype=object,
        )
        identifiers = sorted({str(value) for value in prefixed_ids.tolist()})
        if not identifiers:
            raise ValueError(f"Explicit split {split!r} contains no trajectories.")
        split_ids[split] = identifiers
        observation_parts.append(observations)
        action_parts.append(
            _validated_normalized_actions(
                actions,
                context=f"Explicit recurrent BC split {split!r}",
            )
        )
        trajectory_id_parts.append(prefixed_ids)
        timestep_parts.append(timesteps)
        done_parts.append(dones)
        source_metadata[split] = getattr(transitions, "metadata", {})

    combined = ExplicitSplitRecurrentTransitions(
        policy_observations=np.concatenate(observation_parts, axis=0).astype(
            np.float32,
            copy=False,
        ),
        actions_continuous_env=np.concatenate(action_parts, axis=0).astype(
            np.float32,
            copy=False,
        ),
        trajectory_ids=np.concatenate(trajectory_id_parts, axis=0).astype(
            object,
            copy=False,
        ),
        timesteps=np.concatenate(timestep_parts, axis=0).astype(
            np.int64,
            copy=False,
        ),
        dones=np.concatenate(done_parts, axis=0).astype(bool, copy=False),
        metadata={
            "split_method": (
                "explicit_source_directories"
                if has_test
                else "explicit_source_directories_deferred_test"
            ),
            "sources": source_metadata,
            "recurrent_warmup_mode": str(warmup_mode),
        },
    )
    split_windows = {
        split: build_sequence_windows(
            combined,
            identifiers,
            sequence_length=int(sequence_length),
            context_length=int(context_length),
            warmup_mode=str(warmup_mode),
        )
        for split, identifiers in split_ids.items()
    }
    if not has_test:
        split_ids["test"] = []
        split_windows["test"] = []
    actions = combined.actions_continuous_env
    train_actions = actions[_window_indices(split_windows["train"])]
    validation_actions = actions[_window_indices(split_windows["validation"])]
    action_mean = train_actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    baseline_validation_mse = float(
        np.mean(np.square(validation_actions - action_mean))
    )
    return PreparedRecurrentBCData(
        transitions=combined,
        observations=combined.policy_observations,
        actions=actions,
        split_trajectory_ids=split_ids,
        split_windows=split_windows,
        validation_baseline_mse=baseline_validation_mse,
        warmup_mode=str(warmup_mode),
    )


def prepare_recurrent_bc_data(
    transitions: Any,
    *,
    split_seed: int,
    sequence_length: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    context_length: int = 32,
    warmup_mode: str = "full_prefix",
) -> PreparedRecurrentBCData:
    """Prepare deterministic splits/windows once for one or more BC trials."""
    observations = np.asarray(transitions.policy_observations, dtype=np.float32)
    actions = _validated_normalized_actions(
        transitions.actions_continuous_env,
        context="Recurrent BC training data",
    )
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions):
        raise ValueError("Recurrent BC requires aligned rank-2 observations and actions.")

    split_ids = split_trajectory_ids(
        transitions,
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
        seed=int(split_seed),
    )
    split_windows = {
        name: build_sequence_windows(
            transitions,
            identifiers,
            sequence_length=int(sequence_length),
            context_length=int(context_length),
            warmup_mode=str(warmup_mode),
        )
        for name, identifiers in split_ids.items()
    }
    train_actions = actions[_window_indices(split_windows["train"])]
    validation_actions = actions[_window_indices(split_windows["validation"])]
    action_mean = train_actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    baseline_validation_mse = float(np.mean(np.square(validation_actions - action_mean)))
    return PreparedRecurrentBCData(
        transitions=transitions,
        observations=observations,
        actions=actions,
        split_trajectory_ids=split_ids,
        split_windows=split_windows,
        validation_baseline_mse=baseline_validation_mse,
        warmup_mode=str(warmup_mode),
    )


def _gradient_group_norms(policy: torch.nn.Module) -> dict[str, float]:
    squared: dict[str, float] = {}
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("encoder.layers."):
            parts = name.split(".")
            group = ".".join(parts[:3])
        elif name.startswith("policy_head"):
            group = "policy_head"
        elif name.startswith(("lidar_proj", "lane_proj", "ego_proj", "scalar_proj")):
            group = "input_projection"
        elif name.startswith(("memory_pool", "memory_norm", "memory_query")):
            group = "memory_pool"
        else:
            group = "other"
        value = float(parameter.grad.detach().float().square().sum().cpu())
        squared[group] = squared.get(group, 0.0) + value
    return {name: float(value ** 0.5) for name, value in sorted(squared.items())}


def train_recurrent_behavior_clone(
    policy: torch.nn.Module,
    transitions: Any | None,
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
    early_stopping_min_epochs: int = 0,
    early_stopping_min_delta_relative: float = 0.001,
    selection_min_validation_skill: float | None = None,
    action_loss_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    action_loss_weighting: str = "fixed",
    correlation_loss_weight: float = 0.0,
    variance_loss_weight: float = 0.0,
    minimum_prediction_std_ratios: list[float] | tuple[float, ...] | np.ndarray | None = None,
    selection_min_prediction_std_ratios: list[float] | tuple[float, ...] | np.ndarray | None = None,
    selection_min_prediction_correlations: list[float] | tuple[float, ...] | np.ndarray | None = None,
    checkpoint_selection_rule: str = "validation_loss",
    mirror_augmentation_probability: float = 0.0,
    warmup_mode: str = "full_prefix",
    evaluate_test: bool = False,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    prepared_data: PreparedRecurrentBCData | None = None,
) -> RecurrentBCResult:
    """Train recurrent BC and select a checkpoint using validation evidence only.

    Qualification diagnostics are reported independently from checkpoint
    selection. ``evaluate_test=False`` leaves all offline-test fields pending so
    candidate methods and seeds can be compared without opening the test split.
    """
    if not bool(getattr(policy, "supports_recurrent_memory", False)):
        raise TypeError("Sequence-aware BC requires a recurrent policy.")
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError(f"epochs must be at least one; got {epochs}.")
    early_stopping_patience = int(early_stopping_patience)
    if early_stopping_patience < 0:
        raise ValueError(
            "early_stopping_patience must be non-negative; "
            f"got {early_stopping_patience}."
        )
    early_stopping_min_epochs = int(early_stopping_min_epochs)
    if not 0 <= early_stopping_min_epochs <= epochs:
        raise ValueError(
            "early_stopping_min_epochs must be between zero and epochs; "
            f"got {early_stopping_min_epochs} for {epochs} epochs."
        )
    early_stopping_min_delta_relative = float(
        early_stopping_min_delta_relative
    )
    if (
        not np.isfinite(early_stopping_min_delta_relative)
        or early_stopping_min_delta_relative < 0.0
    ):
        raise ValueError(
            "early_stopping_min_delta_relative must be finite and "
            "non-negative; "
            f"got {early_stopping_min_delta_relative}."
        )
    evaluate_test = bool(evaluate_test)
    if prepared_data is None:
        if transitions is None:
            raise ValueError("transitions is required when prepared_data is not supplied.")
        prepared_data = prepare_recurrent_bc_data(
            transitions,
            split_seed=int(split_seed),
            sequence_length=int(sequence_length),
            train_fraction=float(train_fraction),
            validation_fraction=float(validation_fraction),
            context_length=int(getattr(policy, "memory_context_length", sequence_length)),
            warmup_mode=str(warmup_mode),
        )
    elif transitions is not None and transitions is not prepared_data.transitions:
        raise ValueError("transitions and prepared_data.transitions must refer to the same dataset.")
    transitions = prepared_data.transitions
    observations = prepared_data.observations
    actions = prepared_data.actions
    split_ids = prepared_data.split_trajectory_ids
    split_windows = prepared_data.split_windows
    if evaluate_test and not split_windows.get("test"):
        raise ValueError(
            "evaluate_test=True requires a non-empty prepared test split."
        )
    action_loss_weighting = str(action_loss_weighting).lower()
    if action_loss_weighting not in {"fixed", "inverse_variance"}:
        raise ValueError(
            "action_loss_weighting must be 'fixed' or 'inverse_variance'; "
            f"got {action_loss_weighting!r}."
        )
    checkpoint_selection_rule = str(checkpoint_selection_rule)
    if checkpoint_selection_rule not in {
        "qualification_then_loss",
        "validation_loss",
    }:
        raise ValueError(
            "checkpoint_selection_rule must be 'qualification_then_loss' or "
            f"'validation_loss'; got {checkpoint_selection_rule!r}."
        )
    mirror_augmentation_probability = float(
        mirror_augmentation_probability
    )
    if not 0.0 <= mirror_augmentation_probability <= 1.0:
        raise ValueError(
            "mirror_augmentation_probability must be between zero and one."
        )
    base_action_loss_weights = np.asarray(
        (
            action_loss_weights
            if action_loss_weights is not None
            else [1.0] * int(actions.shape[1])
        ),
        dtype=np.float64,
    )
    if (
        base_action_loss_weights.shape != (actions.shape[1],)
        or not np.isfinite(base_action_loss_weights).all()
        or np.any(base_action_loss_weights <= 0.0)
    ):
        raise ValueError(
            "action_loss_weights must contain one finite positive value per "
            f"action dimension; got {base_action_loss_weights}."
        )
    train_trajectory_ids = set(split_ids["train"])
    training_mask = np.asarray(
        [
            str(trajectory_id) in train_trajectory_ids
            for trajectory_id in np.asarray(transitions.trajectory_ids)
        ],
        dtype=bool,
    )
    if not np.any(training_mask):
        raise RuntimeError("No training actions were available for loss weighting.")
    training_actions = np.asarray(actions[training_mask], dtype=np.float32)
    if mirror_augmentation_probability > 0.0:
        _, training_action_variance = _mixture_mean_variance(
            training_actions,
            mirror_normalized_actions(training_actions),
            transformed_probability=mirror_augmentation_probability,
        )
    else:
        training_action_variance = np.var(
            training_actions,
            axis=0,
            dtype=np.float64,
        )
    observation_normalization = bool(
        getattr(policy, "observation_normalization", False)
    )
    observation_normalizer_mean: np.ndarray | None = None
    observation_normalizer_std: np.ndarray | None = None
    if observation_normalization:
        training_observations = np.asarray(
            observations[training_mask],
            dtype=np.float32,
        )
        if mirror_augmentation_probability > 0.0:
            normalizer_mean, normalizer_variance = _mixture_mean_variance(
                training_observations,
                mirror_policy_observations(training_observations),
                transformed_probability=mirror_augmentation_probability,
            )
            observation_normalizer_mean = normalizer_mean.astype(
                np.float32
            )
            observation_normalizer_std = np.sqrt(
                normalizer_variance
            ).astype(np.float32)
        else:
            observation_normalizer_mean = training_observations.mean(
                axis=0, dtype=np.float64
            ).astype(np.float32)
            observation_normalizer_std = training_observations.std(
                axis=0, dtype=np.float64
            ).astype(np.float32)
        observation_normalizer_std = np.maximum(
            observation_normalizer_std,
            1.0e-6,
        ).astype(np.float32, copy=False)
        setter = getattr(policy, "set_observation_normalizer", None)
        if not callable(setter):
            raise TypeError(
                "Policy enables observation normalization without a normalizer setter."
            )
        setter(observation_normalizer_mean, observation_normalizer_std)
    effective_action_loss_weights = base_action_loss_weights.copy()
    if action_loss_weighting == "inverse_variance":
        # Equalize standardized per-action error without allowing a near-zero
        # variance dimension to produce an unbounded optimizer weight.
        variance_floor = max(
            1.0e-8,
            float(np.max(training_action_variance)) * 1.0e-4,
        )
        effective_action_loss_weights = (
            effective_action_loss_weights
            / np.maximum(training_action_variance, variance_floor)
        )
    normalized_action_loss_weights = (
        effective_action_loss_weights
        / float(effective_action_loss_weights.mean())
    ).astype(np.float32, copy=False)
    if float(correlation_loss_weight) < 0.0 or float(variance_loss_weight) < 0.0:
        raise ValueError("Anti-collapse loss weights must be non-negative.")

    def _action_vector(
        values: list[float] | tuple[float, ...] | np.ndarray | None,
        *,
        name: str,
        default: float,
    ) -> np.ndarray:
        result = np.asarray(
            [default] * int(actions.shape[1]) if values is None else values,
            dtype=np.float64,
        )
        if result.shape != (actions.shape[1],) or not np.isfinite(result).all():
            raise ValueError(
                f"{name} must contain one finite value per action dimension; got {result}."
            )
        return result

    training_minimum_std_ratios = _action_vector(
        minimum_prediction_std_ratios,
        name="minimum_prediction_std_ratios",
        default=0.0,
    )
    selection_minimum_std_ratios = _action_vector(
        selection_min_prediction_std_ratios,
        name="selection_min_prediction_std_ratios",
        default=0.0,
    )
    selection_minimum_correlations = _action_vector(
        selection_min_prediction_correlations,
        name="selection_min_prediction_correlations",
        default=-1.0,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed) + 1701)
    sequences_per_batch = max(1, int(sequences_per_batch))
    micro_batch_sequences = max(1, min(sequences_per_batch, int(micro_batch_sequences)))

    baseline_validation_mse = float(prepared_data.validation_baseline_mse)
    initial_validation = evaluate_recurrent_bc(
        policy,
        transitions,
        split_windows["validation"],
        device=device,
        micro_batch_sequences=micro_batch_sequences,
    )

    best_epoch = 0
    best_validation_selection_mse = float("inf")
    best_selection_eligible = False
    best_fallback_gate_margin = float("-inf")
    best_state_dict: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    early_stopping_reference_mse = float("inf")
    stopping_reason = "configured_epochs_completed"
    for epoch in range(1, epochs + 1):
        policy.train()
        order = rng.permutation(len(split_windows["train"]))
        ordered = [split_windows["train"][int(index)] for index in order]
        epoch_mse = 0.0
        epoch_mae = 0.0
        epoch_objective = 0.0
        epoch_correlation_loss = 0.0
        epoch_variance_loss = 0.0
        epoch_samples = 0
        epoch_gradient_norm_sum = 0.0
        epoch_gradient_norm_max = 0.0
        epoch_clipped_steps = 0
        epoch_optimizer_steps = 0
        epoch_mirrored_sequences = 0
        epoch_group_norm_sums: dict[str, float] = {}
        for batch_start in range(0, len(ordered), sequences_per_batch):
            batch = ordered[batch_start : batch_start + sequences_per_batch]
            batch_samples = sum(len(window.train_indices) for window in batch)
            optimizer.zero_grad(set_to_none=True)
            batch_mse = 0.0
            batch_mae = 0.0
            batch_objective = 0.0
            batch_correlation_loss = 0.0
            batch_variance_loss = 0.0
            batch_mirrored_sequences = 0
            for micro_start in range(0, len(batch), micro_batch_sequences):
                micro = batch[micro_start : micro_start + micro_batch_sequences]
                mirror_mask = (
                    rng.random(len(micro))
                    < mirror_augmentation_probability
                )
                batch_mirrored_sequences += int(mirror_mask.sum())
                loss_diagnostics: dict[str, Any] = {}
                objective, mae, count = _batched_sequence_loss(
                    policy,
                    observations,
                    actions,
                    micro,
                    device=device,
                    action_loss_weights=normalized_action_loss_weights,
                    correlation_loss_weight=float(correlation_loss_weight),
                    variance_loss_weight=float(variance_loss_weight),
                    minimum_prediction_std_ratios=training_minimum_std_ratios,
                    mirror_mask=mirror_mask,
                    loss_diagnostics=loss_diagnostics,
                )
                (objective * (float(count) / float(batch_samples))).backward()
                batch_objective += float(loss_diagnostics["objective"]) * count
                batch_mse += float(loss_diagnostics["base_mse"]) * count
                batch_mae += float(mae.detach().cpu()) * count
                batch_correlation_loss += (
                    float(loss_diagnostics["correlation_loss"]) * count
                )
                batch_variance_loss += (
                    float(loss_diagnostics["variance_loss"]) * count
                )
            for group, norm in _gradient_group_norms(policy).items():
                epoch_group_norm_sums[group] = epoch_group_norm_sums.get(group, 0.0) + norm
            clipping_threshold = float(max_grad_norm)
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                clipping_threshold if clipping_threshold > 0.0 else float("inf"),
            )
            gradient_norm = float(gradient_norm_tensor.detach().cpu())
            epoch_gradient_norm_sum += gradient_norm
            epoch_gradient_norm_max = max(epoch_gradient_norm_max, gradient_norm)
            epoch_optimizer_steps += 1
            if clipping_threshold > 0.0 and gradient_norm > clipping_threshold:
                epoch_clipped_steps += 1
            optimizer.step()
            epoch_objective += batch_objective
            epoch_mse += batch_mse
            epoch_mae += batch_mae
            epoch_correlation_loss += batch_correlation_loss
            epoch_variance_loss += batch_variance_loss
            epoch_samples += batch_samples
            epoch_mirrored_sequences += batch_mirrored_sequences

        validation = evaluate_recurrent_bc(
            policy,
            transitions,
            split_windows["validation"],
            device=device,
            micro_batch_sequences=micro_batch_sequences,
        )
        validation_mse = float(validation["mse"])
        validation_weighted_objective_mse = float(
            np.mean(
                np.asarray(validation["action_mse"], dtype=np.float64)
                * (
                    normalized_action_loss_weights
                    if normalized_action_loss_weights is not None
                    else np.ones(actions.shape[1], dtype=np.float32)
                )
            )
        )
        # All factorial arms must be selected on the same estimand.  The
        # inverse-variance weights are a training treatment, not a license to
        # change the validation criterion used to choose a checkpoint.
        validation_selection_mse = validation_mse
        validation_skill = 1.0 - validation_mse / max(baseline_validation_mse, 1.0e-12)
        validation_std_ratios = np.asarray(
            validation["prediction_std_ratio"], dtype=np.float64
        )
        validation_correlations = np.asarray(
            validation["prediction_target_correlation"], dtype=np.float64
        )
        skill_threshold = (
            float(selection_min_validation_skill)
            if selection_min_validation_skill is not None
            else float("-inf")
        )
        selection_eligible = bool(
            validation_skill >= skill_threshold
            and np.all(validation_std_ratios >= selection_minimum_std_ratios)
            and np.all(validation_correlations >= selection_minimum_correlations)
        )
        fallback_penalty = float(
            np.square(
                np.maximum(
                    selection_minimum_std_ratios - validation_std_ratios,
                    0.0,
                )
            ).sum()
            + np.square(
                np.maximum(
                    selection_minimum_correlations - validation_correlations,
                    0.0,
                )
            ).sum()
        )
        normalized_gate_margins: list[float] = []
        if np.isfinite(skill_threshold) and skill_threshold > 0.0:
            normalized_gate_margins.append(validation_skill / skill_threshold)
        normalized_gate_margins.extend(
            float(value / threshold)
            for value, threshold in zip(
                validation_std_ratios,
                selection_minimum_std_ratios,
                strict=True,
            )
            if threshold > 0.0
        )
        normalized_gate_margins.extend(
            float(value / threshold)
            for value, threshold in zip(
                validation_correlations,
                selection_minimum_correlations,
                strict=True,
            )
            if threshold > 0.0
        )
        fallback_gate_margin = min(normalized_gate_margins, default=0.0)
        if checkpoint_selection_rule == "validation_loss":
            improved = bool(
                validation_selection_mse < best_validation_selection_mse
            )
        elif selection_eligible:
            improved = bool(
                not best_selection_eligible
                or validation_selection_mse < best_validation_selection_mse
            )
        else:
            improved = bool(
                not best_selection_eligible
                and (
                    fallback_gate_margin > best_fallback_gate_margin + 1.0e-12
                    or (
                        abs(fallback_gate_margin - best_fallback_gate_margin)
                        <= 1.0e-12
                        and validation_selection_mse < best_validation_selection_mse
                    )
                )
            )
        if np.isfinite(early_stopping_reference_mse):
            early_stopping_relative_improvement = float(
                (
                    early_stopping_reference_mse
                    - validation_selection_mse
                )
                / max(abs(early_stopping_reference_mse), 1.0e-12)
            )
            early_stopping_significant_improvement = bool(
                validation_selection_mse < early_stopping_reference_mse
                and early_stopping_relative_improvement
                >= early_stopping_min_delta_relative
            )
        else:
            early_stopping_relative_improvement = None
            early_stopping_significant_improvement = True
        if early_stopping_significant_improvement:
            early_stopping_reference_mse = validation_selection_mse
            stale_epochs = 0
        else:
            stale_epochs += 1

        row: dict[str, Any] = {
            "epoch": float(epoch),
            "train_objective": epoch_objective / max(1, epoch_samples),
            "train_mse": epoch_mse / max(1, epoch_samples),
            "train_mae": epoch_mae / max(1, epoch_samples),
            "train_correlation_loss": epoch_correlation_loss / max(1, epoch_samples),
            "train_variance_loss": epoch_variance_loss / max(1, epoch_samples),
            "validation_mse": validation_mse,
            "validation_selection_mse": validation_selection_mse,
            "validation_weighted_objective_mse": (
                validation_weighted_objective_mse
            ),
            "validation_mae": float(validation["mae"]),
            "validation_baseline_mse": baseline_validation_mse,
            "validation_skill": validation_skill,
            "validation_action_mse": validation["action_mse"],
            "validation_action_mae": validation["action_mae"],
            "validation_prediction_std": validation["prediction_std"],
            "validation_target_std": validation["target_std"],
            "validation_prediction_std_ratio": validation["prediction_std_ratio"],
            "validation_prediction_target_correlation": validation["prediction_target_correlation"],
            "validation_prediction_saturation_fraction": validation["prediction_saturation_fraction"],
            "gradient_norm_mean": epoch_gradient_norm_sum / max(1, epoch_optimizer_steps),
            "gradient_norm_max": epoch_gradient_norm_max,
            "gradient_clipped_fraction": epoch_clipped_steps / max(1, epoch_optimizer_steps),
            "gradient_group_norm_mean": {
                group: total / max(1, epoch_optimizer_steps)
                for group, total in sorted(epoch_group_norm_sums.items())
            },
            "optimizer_steps": float(epoch_optimizer_steps),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "max_grad_norm": float(max_grad_norm),
            "action_loss_weights": (
                normalized_action_loss_weights.tolist()
            ),
            "action_loss_weighting": action_loss_weighting,
            "observation_normalization": observation_normalization,
            "correlation_loss_weight": float(correlation_loss_weight),
            "variance_loss_weight": float(variance_loss_weight),
            "mirror_augmentation_probability": (
                mirror_augmentation_probability
            ),
            "mirrored_sequence_fraction": float(
                epoch_mirrored_sequences / max(1, len(ordered))
            ),
            "minimum_prediction_std_ratios": training_minimum_std_ratios.tolist(),
            "selection_min_prediction_std_ratios": selection_minimum_std_ratios.tolist(),
            "selection_min_prediction_correlations": selection_minimum_correlations.tolist(),
            "selection_min_validation_skill": (
                float(selection_min_validation_skill)
                if selection_min_validation_skill is not None
                else None
            ),
            "selection_eligible": selection_eligible,
            "checkpoint_selection_rule": checkpoint_selection_rule,
            "selection_fallback_penalty": fallback_penalty,
            "selection_fallback_gate_margin": fallback_gate_margin,
            "is_best_so_far": improved,
            "early_stopping_significant_improvement": (
                early_stopping_significant_improvement
            ),
            "early_stopping_relative_improvement": (
                early_stopping_relative_improvement
            ),
            "early_stopping_reference_mse": (
                early_stopping_reference_mse
            ),
            "early_stopping_stale_epochs": float(stale_epochs),
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(dict(row))
        print(
            f"[recurrent bc {epoch:03d}/{epochs:03d}] "
            f"train_mse={row['train_mse']:.6f} train_mae={row['train_mae']:.6f} "
            f"val_mse={row['validation_mse']:.6f} val_mae={row['validation_mae']:.6f} "
            f"val_skill={row['validation_skill']:.6f} "
            f"action0_std_ratio={row['validation_prediction_std_ratio'][0]:.4f} "
            f"grad_norm={row['gradient_norm_mean']:.4f} "
            f"clipped={row['gradient_clipped_fraction']:.3f} best={row['is_best_so_far']}"
        )
        if improved:
            best_validation_selection_mse = row["validation_selection_mse"]
            best_selection_eligible = bool(selection_eligible)
            best_fallback_gate_margin = float(fallback_gate_margin)
            best_epoch = epoch
            best_state_dict = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
        if (
            early_stopping_patience > 0
            and epoch >= early_stopping_min_epochs
            and (
                best_selection_eligible
                or checkpoint_selection_rule == "validation_loss"
            )
            and stale_epochs >= early_stopping_patience
        ):
            stopping_reason = "early_stopping_patience_exhausted"
            print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
            break

    if not best_state_dict:
        raise RuntimeError("Recurrent BC did not produce a best checkpoint.")
    selected_history_row = next(
        deepcopy(row)
        for row in history
        if int(row["epoch"]) == int(best_epoch)
    )
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
    test_metrics = (
        evaluate_recurrent_bc(
            policy,
            transitions,
            split_windows["test"],
            device=device,
            micro_batch_sequences=micro_batch_sequences,
        )
        if evaluate_test
        else None
    )
    validation_weighted_objective_mse = float(
        np.mean(
            np.asarray(validation_metrics["action_mse"], dtype=np.float64)
            * (
                normalized_action_loss_weights
                if normalized_action_loss_weights is not None
                else np.ones(actions.shape[1], dtype=np.float32)
            )
        )
    )
    validation_selection_mse = float(validation_metrics["mse"])
    skill = 1.0 - float(validation_metrics["mse"]) / max(baseline_validation_mse, 1.0e-12)
    initial_validation_mse = float(initial_validation["mse"])
    summary = {
        "best_epoch": float(best_epoch),
        "configured_epochs": float(epochs),
        "completed_epochs": float(len(history)),
        "stopping_reason": stopping_reason,
        "stop_epoch": float(len(history)),
        "early_stopping_enabled": bool(early_stopping_patience > 0),
        "early_stopping_patience": float(early_stopping_patience),
        "early_stopping_min_epochs": float(early_stopping_min_epochs),
        "early_stopping_min_delta_relative": (
            early_stopping_min_delta_relative
        ),
        "stale_epochs_at_stop": float(stale_epochs),
        "selected_history_row": selected_history_row,
        "checkpoint_selection_metric": "unweighted_validation_mse",
        "checkpoint_selection_tie_tolerance": 0.0,
        "initial_validation_mse": initial_validation_mse,
        "initial_validation_mae": float(initial_validation["mae"]),
        "relative_validation_improvement": (
            initial_validation_mse - float(validation_metrics["mse"])
        )
        / max(initial_validation_mse, 1.0e-12),
        "train_mse": float(train_metrics["mse"]),
        "train_mae": float(train_metrics["mae"]),
        "validation_mse": float(validation_metrics["mse"]),
        "validation_selection_mse": validation_selection_mse,
        "validation_weighted_objective_mse": (
            validation_weighted_objective_mse
        ),
        "validation_mae": float(validation_metrics["mae"]),
        "offline_test_evaluated": evaluate_test,
        "offline_test_status": (
            "evaluated_after_validation_selection"
            if evaluate_test
            else "pending_deferred"
        ),
        "test_mse": float(test_metrics["mse"]) if test_metrics is not None else None,
        "test_mae": float(test_metrics["mae"]) if test_metrics is not None else None,
        "validation_baseline_mse": baseline_validation_mse,
        "validation_skill": float(skill),
        "action_loss_weights": (
            normalized_action_loss_weights.tolist()
        ),
        "action_loss_weighting": action_loss_weighting,
        "correlation_loss_weight": float(correlation_loss_weight),
        "variance_loss_weight": float(variance_loss_weight),
        "mirror_augmentation_probability": mirror_augmentation_probability,
        "training_moments_include_mirror_mixture": bool(
            mirror_augmentation_probability > 0.0
        ),
        "minimum_prediction_std_ratios": training_minimum_std_ratios.tolist(),
        "selection_min_prediction_std_ratios": selection_minimum_std_ratios.tolist(),
        "selection_min_prediction_correlations": selection_minimum_correlations.tolist(),
        "selection_min_validation_skill": (
            float(selection_min_validation_skill)
            if selection_min_validation_skill is not None
            else None
        ),
        "best_selection_eligible": bool(best_selection_eligible),
        "checkpoint_selection_rule": checkpoint_selection_rule,
        "best_fallback_gate_margin": float(best_fallback_gate_margin),
        "training_action_variance": training_action_variance.tolist(),
        "observation_normalization": observation_normalization,
        "recurrent_warmup_mode": str(prepared_data.warmup_mode),
        "observation_normalizer_mean": (
            observation_normalizer_mean.tolist()
            if observation_normalizer_mean is not None
            else None
        ),
        "observation_normalizer_std": (
            observation_normalizer_std.tolist()
            if observation_normalizer_std is not None
            else None
        ),
        "train_action_mse": train_metrics["action_mse"],
        "train_action_mae": train_metrics["action_mae"],
        "train_prediction_std": train_metrics["prediction_std"],
        "train_target_std": train_metrics["target_std"],
        "train_prediction_std_ratio": train_metrics["prediction_std_ratio"],
        "train_prediction_target_correlation": train_metrics["prediction_target_correlation"],
        "train_prediction_saturation_fraction": train_metrics["prediction_saturation_fraction"],
        "validation_action_mse": validation_metrics["action_mse"],
        "validation_action_mae": validation_metrics["action_mae"],
        "validation_prediction_std": validation_metrics["prediction_std"],
        "validation_target_std": validation_metrics["target_std"],
        "validation_prediction_std_ratio": validation_metrics["prediction_std_ratio"],
        "validation_prediction_target_correlation": validation_metrics["prediction_target_correlation"],
        "validation_prediction_saturation_fraction": validation_metrics["prediction_saturation_fraction"],
        "test_action_mse": test_metrics["action_mse"] if test_metrics is not None else None,
        "test_action_mae": test_metrics["action_mae"] if test_metrics is not None else None,
        "test_prediction_std": test_metrics["prediction_std"] if test_metrics is not None else None,
        "test_target_std": test_metrics["target_std"] if test_metrics is not None else None,
        "test_prediction_std_ratio": (
            test_metrics["prediction_std_ratio"] if test_metrics is not None else None
        ),
        "test_prediction_target_correlation": (
            test_metrics["prediction_target_correlation"]
            if test_metrics is not None
            else None
        ),
        "test_prediction_saturation_fraction": (
            test_metrics["prediction_saturation_fraction"]
            if test_metrics is not None
            else None
        ),
        "train_samples": float(train_metrics["samples"]),
        "validation_samples": float(validation_metrics["samples"]),
        "test_samples": float(test_metrics["samples"]) if test_metrics is not None else None,
        "train_windows": float(train_metrics["windows"]),
        "validation_windows": float(validation_metrics["windows"]),
        "test_windows": float(test_metrics["windows"]) if test_metrics is not None else None,
    }
    return RecurrentBCResult(
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        history=history,
        summary=summary,
        split_trajectory_ids=split_ids,
    )
