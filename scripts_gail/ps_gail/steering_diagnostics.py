"""Monitoring-only steering action diagnostics."""

from __future__ import annotations

import numpy as np

from .data import build_sequence_windows
from .vendi import safe_sequence_window_mask, vendi_score


STEERING_EVENT_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)
STEERING_QUANTILES = (50, 75, 90, 95, 99)
STEERING_HISTOGRAM_BINS = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
_PAIRWISE_EPS = 1.0e-12
_RBF_KERNEL_MAX_PAIRS = 1_000_000


def normalized_steering(actions: np.ndarray | None) -> np.ndarray:
    """Return finite normalized steering column from continuous [accel, steer] actions."""

    if actions is None:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0,), dtype=np.float32)
    steering = arr[:, 1].reshape(-1)
    return steering[np.isfinite(steering)].astype(np.float32, copy=False)


def _finite_steering_with_ids(
    actions: np.ndarray | None,
    trajectory_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if actions is None:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    steering = arr[:, 1].reshape(-1)
    finite = np.isfinite(steering)
    if trajectory_ids is None:
        ids = np.arange(steering.shape[0], dtype=np.int64)
    else:
        ids = np.asarray(trajectory_ids).reshape(-1)
        if ids.shape[0] != steering.shape[0]:
            return (
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
    original_indices = np.arange(steering.shape[0], dtype=np.int64)
    return (
        steering[finite].reshape(-1, 1).astype(np.float32, copy=False),
        ids[finite],
        original_indices[finite],
    )


def steering_summary_metrics(
    actions: np.ndarray | None,
    *,
    prefix: str,
) -> dict[str, float]:
    """Summarize normalized steering magnitude and event-rate diagnostics."""

    steering = normalized_steering(actions)
    metrics: dict[str, float] = {
        f"steering/{prefix}_count": float(steering.size),
    }
    if steering.size == 0:
        metrics.update(
            {
                f"steering/{prefix}_steer_mean": float("nan"),
                f"steering/{prefix}_steer_std": float("nan"),
                f"steering/{prefix}_abs_steer_mean": float("nan"),
            }
        )
        for quantile in STEERING_QUANTILES:
            metrics[f"steering/{prefix}_abs_steer_p{quantile}"] = float("nan")
        for threshold in STEERING_EVENT_THRESHOLDS:
            metrics[f"steering/{prefix}_event_rate_abs_ge_{_threshold_key(threshold)}"] = 0.0
        for idx in range(len(STEERING_HISTOGRAM_BINS) - 1):
            metrics[f"steering/{prefix}_hist_bin_{idx:02d}"] = 0.0
        return metrics

    abs_steering = np.abs(steering)
    metrics[f"steering/{prefix}_steer_mean"] = float(steering.mean())
    metrics[f"steering/{prefix}_steer_std"] = float(steering.std())
    metrics[f"steering/{prefix}_abs_steer_mean"] = float(abs_steering.mean())
    for quantile in STEERING_QUANTILES:
        metrics[f"steering/{prefix}_abs_steer_p{quantile}"] = float(
            np.percentile(abs_steering, quantile)
        )
    for threshold in STEERING_EVENT_THRESHOLDS:
        metrics[f"steering/{prefix}_event_rate_abs_ge_{_threshold_key(threshold)}"] = float(
            np.mean(abs_steering >= float(threshold))
        )

    clipped = np.clip(steering, -1.0, 1.0)
    hist, _edges = np.histogram(clipped, bins=STEERING_HISTOGRAM_BINS)
    fractions = hist.astype(np.float64) / float(max(1, hist.sum()))
    for idx, value in enumerate(fractions):
        metrics[f"steering/{prefix}_hist_bin_{idx:02d}"] = float(value)
    return metrics


def steering_windows(
    actions: np.ndarray | None,
    trajectory_ids: np.ndarray | None,
    *,
    sequence_length: int,
    stride: int = 1,
    return_window_indices: bool = False,
):
    """Build [N, T, 1] steering windows grouped by trajectory."""

    steering, ids, original_indices = _finite_steering_with_ids(actions, trajectory_ids)
    result = build_sequence_windows(
        steering,
        ids,
        sequence_length=max(1, int(sequence_length)),
        stride=max(1, int(stride)),
        return_window_indices=return_window_indices,
    )
    if not return_window_indices:
        windows, last_indices = result
        return windows, original_indices[last_indices] if last_indices.size else last_indices
    windows, last_indices, window_indices = result
    mapped_last_indices = original_indices[last_indices] if last_indices.size else last_indices
    mapped_window_indices = (
        original_indices[window_indices] if window_indices.size else window_indices
    )
    return windows, mapped_last_indices, mapped_window_indices


def steering_vendi_metrics(
    actions: np.ndarray | None,
    trajectory_ids: np.ndarray | None,
    *,
    prefix: str,
    sequence_length: int,
    stride: int,
    max_windows: int,
    seed: int,
    rbf_sigma: float,
) -> dict[str, float]:
    """Compute Vendi metrics over normalized steering action windows."""

    windows, _last_indices = steering_windows(
        actions,
        trajectory_ids,
        sequence_length=sequence_length,
        stride=stride,
    )
    return vendi_score(
        windows,
        max_windows=int(max_windows),
        seed=int(seed),
        rbf_sigma=float(rbf_sigma),
    ).as_dict(f"vendi/{prefix}_steering")


def policy_safe_steering_vendi_metrics(
    actions: np.ndarray | None,
    trajectory_ids: np.ndarray | None,
    env_penalties: np.ndarray,
    *,
    prefix: str,
    sequence_length: int,
    stride: int,
    max_windows: int,
    seed: int,
    rbf_sigma: float,
) -> dict[str, float]:
    """Compute Vendi metrics over safe policy steering windows only."""

    windows, _last_indices, window_indices = steering_windows(
        actions,
        trajectory_ids,
        sequence_length=sequence_length,
        stride=stride,
        return_window_indices=True,
    )
    safe_mask = safe_sequence_window_mask(window_indices, env_penalties)
    metrics: dict[str, float] = {}
    if safe_mask.shape[0] == windows.shape[0]:
        safe_metrics = vendi_score(
            windows[safe_mask],
            max_windows=int(max_windows),
            seed=int(seed),
            rbf_sigma=float(rbf_sigma),
        )
        metrics.update(safe_metrics.as_dict(f"vendi/{prefix}_steering_safe"))
        metrics[f"vendi/{prefix}_steering_safe_fraction"] = float(safe_mask.mean()) if safe_mask.size else 0.0
    else:
        metrics[f"vendi/{prefix}_steering_safe_available"] = 0.0
    return metrics


def safe_policy_steering_actions(
    actions: np.ndarray | None,
    trajectory_ids: np.ndarray | None,
    env_penalties: np.ndarray,
    *,
    sequence_length: int,
    stride: int,
) -> np.ndarray:
    """Return policy actions referenced by zero-penalty steering windows."""

    if actions is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    _windows, _last_indices, window_indices = steering_windows(
        arr,
        trajectory_ids,
        sequence_length=sequence_length,
        stride=stride,
        return_window_indices=True,
    )
    safe_mask = safe_sequence_window_mask(window_indices, env_penalties)
    if safe_mask.shape[0] != window_indices.shape[0] or not bool(safe_mask.any()):
        return np.zeros((0, arr.shape[1]), dtype=np.float32)
    safe_indices = np.unique(window_indices[safe_mask].reshape(-1))
    safe_indices = safe_indices[(safe_indices >= 0) & (safe_indices < arr.shape[0])]
    return arr[safe_indices].astype(np.float32, copy=False)


def steering_mmd_rbf(
    left_actions: np.ndarray | None,
    right_actions: np.ndarray | None,
    *,
    max_samples: int = 2048,
    seed: int = 0,
    rbf_sigma: float = 0.0,
) -> float:
    """Return biased RBF-MMD^2 between normalized steering samples."""

    left_values = normalized_steering(left_actions)
    right_values = normalized_steering(right_actions)
    same_values = left_values.shape == right_values.shape and np.array_equal(left_values, right_values)
    left = _subsample_1d(left_values, max_samples=max_samples, seed=seed)
    right = left.copy() if same_values else _subsample_1d(right_values, max_samples=max_samples, seed=seed + 1)
    if left.size == 0 or right.size == 0:
        return float("nan")
    sigma = float(rbf_sigma) if float(rbf_sigma) > 0.0 else _median_pairwise_bandwidth(left, right)
    k_xx = _rbf_kernel_mean(left, left, sigma)
    k_yy = _rbf_kernel_mean(right, right, sigma)
    k_xy = _rbf_kernel_mean(left, right, sigma)
    return float(max(0.0, k_xx + k_yy - 2.0 * k_xy))


def _subsample_1d(values: np.ndarray, *, max_samples: int, seed: int) -> np.ndarray:
    max_samples = max(0, int(max_samples))
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if max_samples <= 0 or arr.size <= max_samples:
        return arr
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(arr.size, size=max_samples, replace=False)
    indices.sort()
    return arr[indices]


def _median_pairwise_bandwidth(left: np.ndarray, right: np.ndarray) -> float:
    values = np.concatenate([left.reshape(-1), right.reshape(-1)]).astype(np.float64, copy=False)
    if values.size <= 1:
        return 1.0
    values.sort()
    max_distance = float(values[-1] - values[0])
    if max_distance <= _PAIRWISE_EPS:
        return 1.0
    total_pairs = values.size * (values.size - 1) // 2
    zero_like_pairs = _count_pairwise_distances_leq(values, _PAIRWISE_EPS)
    positive_pairs = int(total_pairs - zero_like_pairs)
    if positive_pairs <= 0:
        return 1.0
    lower_k = (positive_pairs - 1) // 2
    upper_k = positive_pairs // 2
    lower = _select_positive_pairwise_distance(values, lower_k, zero_like_pairs, max_distance)
    upper = _select_positive_pairwise_distance(values, upper_k, zero_like_pairs, max_distance)
    return max(float(0.5 * (lower + upper)), 1.0e-6)


def _count_pairwise_distances_leq(sorted_values: np.ndarray, radius: float) -> int:
    upper = np.searchsorted(sorted_values, sorted_values + float(radius), side="right")
    counts = upper - np.arange(sorted_values.size, dtype=np.int64) - 1
    return int(np.maximum(counts, 0).sum())


def _select_positive_pairwise_distance(
    sorted_values: np.ndarray,
    k: int,
    zero_like_pairs: int,
    max_distance: float,
) -> float:
    lo = _PAIRWISE_EPS
    hi = float(max_distance)
    target = int(k)
    for _iteration in range(64):
        mid = 0.5 * (lo + hi)
        positive_leq_mid = _count_pairwise_distances_leq(sorted_values, mid) - int(zero_like_pairs)
        if positive_leq_mid > target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def _rbf_kernel_mean(left: np.ndarray, right: np.ndarray, sigma: float) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size == 0 or right.size == 0:
        return float("nan")
    sigma_sq = float(sigma) * float(sigma)
    rows_per_chunk = max(1, int(_RBF_KERNEL_MAX_PAIRS) // int(right.size))
    total = 0.0
    count = 0
    for start in range(0, int(left.size), rows_per_chunk):
        chunk = left[start : start + rows_per_chunk].reshape(-1, 1)
        dist_sq = (chunk - right.reshape(1, -1)) ** 2
        total += float(np.exp(-dist_sq / (2.0 * sigma_sq)).sum())
        count += int(dist_sq.size)
    return float(total / float(count)) if count else float("nan")


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.2f}".replace(".", "p")


__all__ = [
    "STEERING_EVENT_THRESHOLDS",
    "STEERING_HISTOGRAM_BINS",
    "STEERING_QUANTILES",
    "normalized_steering",
    "policy_safe_steering_vendi_metrics",
    "safe_policy_steering_actions",
    "steering_mmd_rbf",
    "steering_summary_metrics",
    "steering_vendi_metrics",
    "steering_windows",
]
