from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .observations import policy_observations_from_flat


SCENE_FEATURE_DIM_PER_VEHICLE = 5


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


def scene_snapshot_features(
    vehicles: list[Any] | tuple[Any, ...],
    *,
    max_vehicles: int,
    origin: np.ndarray | None = None,
) -> np.ndarray:
    """Encode one road snapshot as fixed top-K [presence, rel_x, rel_y, vx, vy]."""
    max_vehicles = max(1, int(max_vehicles))
    rows: list[tuple[float, int, np.ndarray]] = []
    if origin is None:
        positions = [
            np.asarray(getattr(vehicle, "position"), dtype=np.float32)
            for vehicle in vehicles
            if getattr(vehicle, "position", None) is not None
        ]
        origin = np.mean(np.stack(positions, axis=0), axis=0) if positions else np.zeros(2, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32).reshape(2)

    for index, vehicle in enumerate(vehicles):
        if hasattr(vehicle, "visible") and not bool(getattr(vehicle, "visible", True)):
            continue
        if hasattr(vehicle, "appear") and not bool(getattr(vehicle, "appear", True)):
            continue
        if hasattr(vehicle, "scene_collection_is_active") and not bool(
            getattr(vehicle, "scene_collection_is_active", True)
        ):
            continue
        position = getattr(vehicle, "position", None)
        if position is None:
            continue
        position_arr = np.asarray(position, dtype=np.float32).reshape(2)
        if not np.all(np.isfinite(position_arr)):
            continue
        velocity = np.asarray(getattr(vehicle, "velocity", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
        if not np.all(np.isfinite(velocity)):
            velocity = np.zeros(2, dtype=np.float32)
        rel = position_arr - origin
        vehicle_id = int(getattr(vehicle, "vehicle_ID", index))
        feature = np.asarray([1.0, rel[0], rel[1], velocity[0], velocity[1]], dtype=np.float32)
        rows.append((float(np.linalg.norm(rel)), vehicle_id, feature))

    rows.sort(key=lambda item: (item[0], item[1]))
    out = np.zeros((max_vehicles, SCENE_FEATURE_DIM_PER_VEHICLE), dtype=np.float32)
    for row_idx, (_dist, _vehicle_id, feature) in enumerate(rows[:max_vehicles]):
        out[row_idx] = feature
    return out.reshape(-1).astype(np.float32, copy=False)


def build_sequence_windows(
    features: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    sequence_length: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    trajectory_ids = np.asarray(trajectory_ids)
    if features.ndim != 2:
        raise ValueError(f"Expected rank-2 features, got {features.shape}.")
    if len(features) != len(trajectory_ids):
        raise ValueError(
            f"Feature/trajectory id count mismatch: {len(features)} != {len(trajectory_ids)}."
        )
    sequence_length = max(1, int(sequence_length))
    stride = max(1, int(stride))
    windows: list[np.ndarray] = []
    last_indices: list[int] = []
    for trajectory_id in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        if indices.size < sequence_length:
            continue
        for start in range(0, indices.size - sequence_length + 1, stride):
            window_indices = indices[start : start + sequence_length]
            windows.append(features[window_indices])
            last_indices.append(int(window_indices[-1]))
    if not windows:
        return (
            np.zeros((0, sequence_length, features.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.stack(windows, axis=0).astype(np.float32, copy=False),
        np.asarray(last_indices, dtype=np.int64),
    )


def normalize_trajectory_frame(
    trajectory_states: np.ndarray,
    trajectory_keys: np.ndarray | list[int] | None,
    *,
    frame: str,
) -> np.ndarray:
    frame = str(frame).lower()
    trajectory_states = np.asarray(trajectory_states, dtype=np.float32)
    if trajectory_states.ndim != 2 or trajectory_states.shape[1] != 3:
        raise ValueError(f"Expected trajectory_states [N, 3], got {trajectory_states.shape}.")
    if frame == "absolute":
        return trajectory_states.astype(np.float32, copy=True)
    if frame != "relative":
        raise ValueError(f"Unsupported trajectory_frame={frame!r}. Expected 'absolute' or 'relative'.")

    relative = trajectory_states.astype(np.float32, copy=True)
    if len(relative) == 0:
        return relative
    if trajectory_keys is None:
        relative[:, :2] -= relative[0:1, :2]
        return relative

    keys = np.asarray(trajectory_keys)
    if keys.shape[0] != relative.shape[0]:
        raise ValueError(
            f"trajectory_keys length {keys.shape[0]} does not match states length {relative.shape[0]}."
        )
    for key in np.unique(keys):
        mask = keys == key
        first_idx = int(np.flatnonzero(mask)[0])
        relative[mask, :2] -= relative[first_idx, :2]
    return relative


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


def _dataset_file_counts(path: str, files: list[str]) -> dict[str, int]:
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            counts = {
                os.path.join(path, item["dataset_file"]): int(item["num_samples"])
                for item in manifest.get("episodes", [])
                if "dataset_file" in item and "num_samples" in item
            }
            if all(file_path in counts for file_path in files):
                return counts

    counts = {}
    for file_path in files:
        with np.load(file_path, allow_pickle=True) as data:
            if "observations" not in data.files:
                raise KeyError(f"{file_path} is missing array: observations")
            counts[file_path] = int(len(data["observations"]))
    return counts


def _uniform_file_sample_plan(
    files: list[str],
    counts: dict[str, int],
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | None]:
    total = int(sum(counts[file_path] for file_path in files))
    if total <= 0:
        raise RuntimeError("Expert dataset contains no samples.")
    if int(max_samples) <= 0 or int(max_samples) >= total:
        return {file_path: None for file_path in files}

    # Sample from the conceptual concatenation of all files, then map back to
    # per-file row ids. This is uniform over expert transitions without loading
    # the full dataset into memory.
    global_indices = np.sort(
        rng.choice(total, size=int(max_samples), replace=False).astype(np.int64, copy=False)
    )
    plan: dict[str, np.ndarray | None] = {}
    cursor = 0
    for file_path in files:
        count = int(counts[file_path])
        start = cursor
        end = cursor + count
        left = int(np.searchsorted(global_indices, start, side="left"))
        right = int(np.searchsorted(global_indices, end, side="left"))
        local_indices = global_indices[left:right] - start
        if local_indices.size:
            plan[file_path] = local_indices.astype(np.int64, copy=False)
        cursor = end
    return plan


def load_expert_policy_and_disc_data(
    path: str,
    *,
    max_samples: int = 100_000,
    seed: int = 0,
    trajectory_frame: str = "relative",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    files = _dataset_files(path)
    counts = _dataset_file_counts(path, files)
    sample_plan = _uniform_file_sample_plan(
        files,
        counts,
        max_samples=int(max_samples),
        rng=rng,
    )
    obs_parts: list[np.ndarray] = []
    traj_parts: list[np.ndarray] = []
    metadata_items: list[dict[str, Any]] = []
    samples_by_file: list[dict[str, Any]] = []

    for file_path in files:
        idx = sample_plan.get(file_path)
        if file_path not in sample_plan:
            continue
        with np.load(file_path, allow_pickle=True) as data:
            required = {"observations", "trajectory_states"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise KeyError(f"{file_path} is missing arrays: {missing}")
            obs = np.asarray(data["observations"], dtype=np.float32)
            traj = np.asarray(data["trajectory_states"], dtype=np.float32)
            trajectory_keys = np.asarray(data["vehicle_ids"]) if "vehicle_ids" in data.files else None
            traj = normalize_trajectory_frame(
                traj,
                trajectory_keys,
                frame=trajectory_frame,
            )
            if "metadata_json" in data.files:
                metadata_items.append(json.loads(str(data["metadata_json"].item())))
            if idx is not None:
                obs = obs[idx]
                traj = traj[idx]
            take = int(len(obs))
            if take <= 0:
                continue
            obs_parts.append(policy_observations_from_flat(obs))
            traj_parts.append(traj.astype(np.float32, copy=False))
            samples_by_file.append(
                {
                    "file": os.path.basename(file_path),
                    "source_samples": int(counts[file_path]),
                    "loaded_samples": take,
                }
            )

    if not obs_parts:
        raise RuntimeError(f"No expert samples were loaded from {path}.")

    policy_obs = np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)
    trajectory_states = np.concatenate(traj_parts, axis=0).astype(np.float32, copy=False)
    features = discriminator_features(policy_obs, trajectory_states)
    metadata = {
        "source_path": os.path.abspath(path),
        "num_files_seen": len(metadata_items) or len(files),
        "num_files_loaded": len(samples_by_file),
        "num_source_samples": int(sum(counts[file_path] for file_path in files)),
        "num_samples": int(policy_obs.shape[0]),
        "policy_observation_dim": int(policy_obs.shape[1]),
        "trajectory_state_dim": int(trajectory_states.shape[1]),
        "feature_dim": int(features.shape[1]),
        "trajectory_frame": str(trajectory_frame).lower(),
        "sampling": "uniform_without_replacement",
        "max_samples": int(max_samples),
        "samples_by_file": samples_by_file,
        "episodes": metadata_items,
    }
    return policy_obs, features, metadata


def load_expert_scene_data(
    path: str,
    *,
    max_samples: int = 100_000,
    seed: int = 0,
    scene_max_vehicles: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    files = _dataset_files(path)
    parts: list[np.ndarray] = []
    for file_path in files:
        with np.load(file_path, allow_pickle=True) as data:
            if "scene_features" not in data.files:
                raise KeyError(
                    f"{file_path} is missing scene_features. Rebuild the expert data with the updated "
                    "scripts_gail/build_ps_traj_expert_discrete.py before enabling the scene discriminator."
                )
            scene = np.asarray(data["scene_features"], dtype=np.float32)
            if scene.ndim != 2:
                raise ValueError(f"{file_path} scene_features must be rank-2, got {scene.shape}.")
            expected_dim = int(scene_max_vehicles) * SCENE_FEATURE_DIM_PER_VEHICLE
            if scene.shape[1] != expected_dim:
                raise ValueError(
                    f"{file_path} scene feature dim {scene.shape[1]} does not match configured "
                    f"scene_max_vehicles={scene_max_vehicles} ({expected_dim})."
                )
            if len(scene):
                parts.append(scene)
    if not parts:
        raise RuntimeError(f"No expert scene features were loaded from {path}.")
    features = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    source_samples = int(len(features))
    if 0 < int(max_samples) < source_samples:
        idx = np.sort(rng.choice(source_samples, size=int(max_samples), replace=False))
        features = features[idx]
    return features, {
        "num_scene_samples": int(len(features)),
        "num_source_scene_samples": source_samples,
        "scene_feature_dim": int(features.shape[1]),
    }


def load_expert_sequence_data(
    path: str,
    *,
    max_samples: int = 100_000,
    seed: int = 0,
    trajectory_frame: str = "relative",
    sequence_length: int = 8,
    sequence_stride: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    files = _dataset_files(path)
    windows: list[np.ndarray] = []
    source_windows = 0
    for file_idx, file_path in enumerate(files):
        with np.load(file_path, allow_pickle=True) as data:
            required = {"observations", "trajectory_states", "vehicle_ids"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise KeyError(f"{file_path} is missing arrays for sequence discriminator: {missing}")
            obs = policy_observations_from_flat(np.asarray(data["observations"], dtype=np.float32))
            traj = normalize_trajectory_frame(
                np.asarray(data["trajectory_states"], dtype=np.float32),
                np.asarray(data["vehicle_ids"]),
                frame=trajectory_frame,
            )
            features = discriminator_features(obs, traj)
            vehicle_ids = np.asarray(data["vehicle_ids"], dtype=np.int64)
            trajectory_ids = np.asarray(
                [f"{file_idx}:{int(vehicle_id)}" for vehicle_id in vehicle_ids],
                dtype=object,
            )
            file_windows, _last_indices = build_sequence_windows(
                features,
                trajectory_ids,
                sequence_length=sequence_length,
                stride=sequence_stride,
            )
            source_windows += int(len(file_windows))
            if len(file_windows):
                windows.append(file_windows)
    if not windows:
        raise RuntimeError(
            f"No expert sequence windows of length {sequence_length} were built from {path}."
        )
    sequences = np.concatenate(windows, axis=0).astype(np.float32, copy=False)
    if 0 < int(max_samples) < len(sequences):
        idx = np.sort(rng.choice(len(sequences), size=int(max_samples), replace=False))
        sequences = sequences[idx]
    return sequences, {
        "num_sequence_samples": int(len(sequences)),
        "num_source_sequence_samples": int(source_windows),
        "sequence_length": int(sequence_length),
        "sequence_feature_dim": int(sequences.shape[-1]),
    }
