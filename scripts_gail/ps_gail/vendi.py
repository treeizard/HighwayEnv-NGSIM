"""Vendi-score diagnostics for trajectory-window diversity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VendiMetrics:
    vendi: float
    log_vendi: float
    effective_ratio: float
    n_windows: int
    sigma: float

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}": float(self.vendi),
            f"{prefix}_log": float(self.log_vendi),
            f"{prefix}_ratio": float(self.effective_ratio),
            f"{prefix}_windows": float(self.n_windows),
            f"{prefix}_sigma": float(self.sigma),
        }


def _flatten_windows(sequence_features: np.ndarray) -> np.ndarray:
    features = np.asarray(sequence_features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected sequence features [N, T, D], got {features.shape}.")
    flat_dim = int(np.prod(features.shape[1:]))
    return features.reshape(features.shape[0], flat_dim).astype(np.float32, copy=False)


def _deterministic_subsample(
    features: np.ndarray,
    *,
    max_windows: int,
    seed: int,
) -> np.ndarray:
    max_windows = max(0, int(max_windows))
    if max_windows <= 0 or len(features) <= max_windows:
        return features
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(features), size=max_windows, replace=False)
    indices.sort()
    return features[indices]


def _squared_distance_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    norms = np.sum(x * x, axis=1, keepdims=True)
    dist_sq = norms + norms.T - 2.0 * (x @ x.T)
    return np.maximum(dist_sq, 0.0)


def _median_distance_bandwidth(dist_sq: np.ndarray) -> float:
    if dist_sq.shape[0] <= 1:
        return 1.0
    upper = dist_sq[np.triu_indices(dist_sq.shape[0], k=1)]
    positive = upper[upper > 1.0e-12]
    if positive.size == 0:
        return 1.0
    sigma = float(np.sqrt(np.median(positive)))
    return max(sigma, 1.0e-6)


def vendi_score(
    sequence_features: np.ndarray,
    *,
    max_windows: int = 2048,
    seed: int = 0,
    rbf_sigma: float = 0.0,
) -> VendiMetrics:
    """Return Vendi metrics for fixed-length trajectory windows.

    Empty inputs return finite zero-valued diagnostics. A single window has
    effective diversity one. For larger batches, Vendi is computed from the
    eigenvalue entropy of the normalized RBF similarity matrix.
    """

    flattened = _flatten_windows(sequence_features)
    flattened = _deterministic_subsample(flattened, max_windows=max_windows, seed=seed)
    n_windows = int(flattened.shape[0])
    if n_windows <= 0:
        return VendiMetrics(
            vendi=0.0,
            log_vendi=0.0,
            effective_ratio=0.0,
            n_windows=0,
            sigma=0.0,
        )
    if n_windows == 1:
        return VendiMetrics(
            vendi=1.0,
            log_vendi=0.0,
            effective_ratio=1.0,
            n_windows=1,
            sigma=float(rbf_sigma) if float(rbf_sigma) > 0.0 else 1.0,
        )

    dist_sq = _squared_distance_matrix(flattened)
    sigma = float(rbf_sigma) if float(rbf_sigma) > 0.0 else _median_distance_bandwidth(dist_sq)
    kernel = np.exp(-dist_sq / (2.0 * sigma * sigma))
    np.fill_diagonal(kernel, 1.0)
    rho = kernel / float(n_windows)
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return VendiMetrics(
            vendi=0.0,
            log_vendi=0.0,
            effective_ratio=0.0,
            n_windows=n_windows,
            sigma=sigma,
        )
    eigenvalues = eigenvalues / total
    positive = eigenvalues[eigenvalues > 1.0e-12]
    entropy = float(-np.sum(positive * np.log(positive)))
    vendi = float(np.exp(entropy))
    return VendiMetrics(
        vendi=vendi,
        log_vendi=entropy,
        effective_ratio=float(vendi / float(n_windows)),
        n_windows=n_windows,
        sigma=sigma,
    )


def safe_sequence_window_mask(
    sequence_transition_indices: np.ndarray,
    env_penalties: np.ndarray,
    *,
    penalty_tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Return windows whose referenced transitions have no env penalty."""

    windows = np.asarray(sequence_transition_indices, dtype=np.int64)
    penalties = np.asarray(env_penalties, dtype=np.float32).reshape(-1)
    if windows.ndim != 2:
        raise ValueError(f"Expected sequence transition indices [N, T], got {windows.shape}.")
    if windows.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    valid = (windows >= 0) & (windows < len(penalties))
    safe = np.zeros((windows.shape[0],), dtype=bool)
    for row_idx, row_valid in enumerate(valid):
        if not bool(row_valid.any()):
            continue
        row_indices = windows[row_idx, row_valid]
        safe[row_idx] = bool(np.all(np.abs(penalties[row_indices]) <= float(penalty_tolerance)))
    return safe


__all__ = ["VendiMetrics", "safe_sequence_window_mask", "vendi_score"]
