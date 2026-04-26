from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .observations import policy_observations_from_flat


def discriminator_features(policy_observations: np.ndarray, trajectory_states: np.ndarray) -> np.ndarray:
    policy_observations = np.asarray(policy_observations, dtype=np.float32)
    trajectory_states = np.asarray(trajectory_states, dtype=np.float32)
    if policy_observations.ndim != 2 or trajectory_states.ndim != 2:
        raise ValueError("Discriminator inputs must be rank-2 arrays.")
    if len(policy_observations) != len(trajectory_states):
        raise ValueError(
            f"Observation/trajectory count mismatch: {len(policy_observations)} != {len(trajectory_states)}"
        )
    if trajectory_states.shape[1] != 3:
        raise ValueError(f"Expected trajectory states [x, y, v], got {trajectory_states.shape}.")
    return np.concatenate([policy_observations, trajectory_states], axis=1).astype(np.float32, copy=False)


def _dataset_files(path: str) -> list[str]:
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            files = [
                os.path.join(path, item["dataset_file"])
                for item in manifest.get("episodes", [])
                if "dataset_file" in item
            ]
        else:
            files = sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".npz"))
        if not files:
            raise FileNotFoundError(f"No expert .npz files found under {path}.")
        return files
    return [path]


def load_expert_policy_and_disc_data(
    path: str,
    *,
    max_samples: int = 100_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    files = _dataset_files(path)
    remaining = int(max_samples) if int(max_samples) > 0 else None
    obs_parts: list[np.ndarray] = []
    traj_parts: list[np.ndarray] = []
    metadata_items: list[dict[str, Any]] = []

    for file_path in files:
        if remaining is not None and remaining <= 0:
            break
        with np.load(file_path, allow_pickle=True) as data:
            required = {"observations", "trajectory_states"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise KeyError(f"{file_path} is missing arrays: {missing}")
            obs = np.asarray(data["observations"], dtype=np.float32)
            traj = np.asarray(data["trajectory_states"], dtype=np.float32)
            if "metadata_json" in data.files:
                metadata_items.append(json.loads(str(data["metadata_json"].item())))
            count = len(obs)
            take = count if remaining is None else min(count, remaining)
            if take < count:
                idx = np.sort(rng.choice(count, size=take, replace=False))
                obs = obs[idx]
                traj = traj[idx]
            obs_parts.append(policy_observations_from_flat(obs))
            traj_parts.append(traj.astype(np.float32, copy=False))
            if remaining is not None:
                remaining -= take

    if not obs_parts:
        raise RuntimeError(f"No expert samples were loaded from {path}.")

    policy_obs = np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)
    trajectory_states = np.concatenate(traj_parts, axis=0).astype(np.float32, copy=False)
    features = discriminator_features(policy_obs, trajectory_states)
    metadata = {
        "source_path": os.path.abspath(path),
        "num_files_seen": len(metadata_items) or len(files),
        "num_samples": int(policy_obs.shape[0]),
        "policy_observation_dim": int(policy_obs.shape[1]),
        "trajectory_state_dim": int(trajectory_states.shape[1]),
        "feature_dim": int(features.shape[1]),
        "episodes": metadata_items,
    }
    return policy_obs, features, metadata
