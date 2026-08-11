from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import policy.data.audit_expert as audit_module
from highway_env.envs.common.observations.lidar import LidarObservation
from highway_env.road.lane import PolyLaneFixedWidth
from policy.contracts.imitation import (
    assert_compatible_observation_contracts,
    policy_observation_contract,
    runtime_continuous_action_contract,
    validate_declared_raw_observation_space,
)
from policy.data.expert import load_expert_transition_data


ROOT = Path(__file__).resolve().parents[1]


def valid_raw_observations(rows: int = 3) -> np.ndarray:
    observations = np.zeros((rows, 323), dtype=np.float32)
    lidar = observations[:, :256].reshape(rows, 128, 2)
    lidar[:, :, 0] = 0.5
    camera = observations[:, 256:319].reshape(rows, 21, 3)
    camera[:, :, 0] = 1.0
    camera[:, :, 1] = 0.25
    camera[:, :, 2] = -0.25
    observations[:, 319:] = np.asarray(
        [1.0e6, 0.0, 2.0, 5.0],
        dtype=np.float32,
    )
    return observations


@pytest.mark.parametrize(
    ("column", "value", "field"),
    [
        (0, -0.01, "lidar.distance_norm"),
        (1, 1.01, "lidar.relative_speed_norm"),
        (256, 1.01, "lane_camera.presence"),
        (257, -1.01, "lane_camera.relative_x_norm"),
        (258, 1.01, "lane_camera.relative_y_norm"),
    ],
)
def test_declared_raw_observation_validator_rejects_only_sensor_space_violations(
    column,
    value,
    field,
):
    contract = policy_observation_contract(
        lidar_cells=128,
        maximum_range=64.0,
    )
    observations = valid_raw_observations()
    receipt = validate_declared_raw_observation_space(
        observations,
        contract,
    )
    assert contract["schema_version"] == 2
    assert contract["raw_observation_components"][0]["low"] == [0.0, -1.0]
    assert contract["raw_observation_components"][0]["high"] == [1.0, 1.0]
    assert receipt["status"] == "passed"
    assert receipt["row_count"] == 3
    # Ego speed is intentionally not assigned an invented empirical bound.
    assert observations[0, 319] == pytest.approx(1.0e6)

    observations[1, column] = value
    with pytest.raises(ValueError, match=field):
        validate_declared_raw_observation_space(
            observations,
            contract,
            context="unit expert observations",
        )


def test_runtime_lidar_space_and_contract_reject_bound_drift():
    lidar = object.__new__(LidarObservation)
    lidar.cells = 2
    lidar.normalize = True
    lidar.maximum_range = 64.0
    space = lidar.space()
    np.testing.assert_array_equal(
        space.low,
        np.asarray([[0.0, -1.0], [0.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        space.high,
        np.ones((2, 2), dtype=np.float32),
    )

    reference = policy_observation_contract(
        lidar_cells=128,
        maximum_range=64.0,
    )
    drifted = deepcopy(reference)
    drifted["raw_observation_components"][0]["high"][1] = 100.0
    with pytest.raises(
        ValueError,
        match="raw_observation_components",
    ):
        assert_compatible_observation_contracts(reference, drifted)


def test_polyline_lane_vectorized_on_road_matches_scalar_highwayenv_semantics():
    lane = PolyLaneFixedWidth(
        [(0.0, 0.0), (40.0, 2.0), (80.0, -1.0), (120.0, 1.0)],
        width=3.7,
    )
    lidar = object.__new__(LidarObservation)
    lidar._lane_geometry_cache = {}
    rng = np.random.default_rng(20260806)
    random_points = np.column_stack(
        (rng.uniform(-8.0, 128.0, 4000), rng.uniform(-8.0, 8.0, 4000))
    )
    boundary_points = np.asarray(
        [
            lane.position(longitudinal, lateral)
            for longitudinal in (-5.0, 0.0, 20.0, 60.0, lane.length, lane.length + 4.9)
            for lateral in (-1.85, 0.0, 1.85)
        ]
    )
    points = np.vstack((random_points, boundary_points))

    vectorized = lidar._lane_on_points_vectorized(lane, points)
    scalar = np.asarray([lane.on_lane(point) for point in points], dtype=bool)

    np.testing.assert_array_equal(vectorized, scalar)


def write_action_conditioned_expert(
    path: Path,
    observations: np.ndarray,
) -> None:
    action_contract = runtime_continuous_action_contract()
    normalized = np.asarray(
        [[-0.2, 0.1], [0.0, 0.0], [0.2, -0.1]],
        dtype=np.float32,
    )
    acceleration_scale, steering_scale = action_contract["scales"]
    physical = np.column_stack(
        (
            normalized[:, 1] * float(steering_scale),
            normalized[:, 0] * float(acceleration_scale),
        )
    ).astype(np.float32)
    metadata = {
        "schema_version": 3,
        "scene": "us-101",
        "episode_name": "unit_episode",
        "continuous_action_contract": action_contract,
        "policy_observation_contract": policy_observation_contract(
            lidar_cells=128,
            maximum_range=64.0,
        ),
    }
    np.savez_compressed(
        path,
        observations=observations,
        next_observations=observations.copy(),
        trajectory_states=np.zeros((3, 3), dtype=np.float32),
        actions_continuous_env=normalized,
        actions_steering_acceleration=physical,
        dones=np.asarray([False, False, True]),
        rewards=np.zeros(3, dtype=np.float32),
        vehicle_ids=np.asarray([7, 7, 7], dtype=np.int64),
        timesteps=np.asarray([0, 1, 2], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )


def test_action_conditioned_loader_validates_full_file_before_sampling(tmp_path):
    expert = tmp_path / "expert.npz"
    observations = valid_raw_observations()
    write_action_conditioned_expert(expert, observations)

    loaded = load_expert_transition_data(
        str(expert),
        max_samples=1,
        seed=17,
    )
    validation = loaded.metadata["raw_observation_space_validation"]
    assert validation["status"] == "passed"
    assert validation["full_source_files_checked"] == 1
    assert validation["source_rows_checked_per_array"] == 3
    assert validation["checked_before_row_sampling"] is True

    observations[2, 1] = 75.0
    write_action_conditioned_expert(expert, observations)
    with pytest.raises(
        ValueError,
        match="lidar.relative_speed_norm.*count=1",
    ):
        load_expert_transition_data(
            str(expert),
            max_samples=1,
            seed=17,
        )


def test_domain_audit_train_val_subset_never_resolves_test(
    tmp_path,
    monkeypatch,
):
    opened: list[Path] = []
    opened_manifests: list[Path] = []

    def fake_read_json(path: Path):
        opened_manifests.append(path)
        return {"policy_observation_contract": {"contract_id": "test"}}

    def fake_audit_split(root: Path, *, reference_observation):
        del reference_observation
        opened.append(root)
        identity = root.as_posix().replace("/", "_")
        return {
            "continuous_action_contract": runtime_continuous_action_contract(),
            "episode_count": 1,
            "row_count": 3,
            "vehicle_ids": [7],
            "files": [
                {
                    "file": f"{root.name}.npz",
                    "sha256": identity,
                }
            ],
        }

    monkeypatch.setattr(audit_module, "read_json", fake_read_json)
    monkeypatch.setattr(audit_module, "audit_split", fake_audit_split)
    result = audit_module.audit_collection(
        tmp_path / "collection",
        splits=audit_module.normalize_requested_splits(["train,val"]),
    )

    assert len(opened) == 4
    assert len(opened_manifests) == 1
    assert opened_manifests[0].parent.name == "train"
    assert all(path.name in {"train", "val"} for path in opened)
    assert not any(path.parent.name == "test" for path in opened_manifests)
    assert not any(path.name == "test" for path in opened)
    assert result["audited_splits"] == ["train", "val"]
    assert result["not_opened_splits"] == ["test"]
    assert result["test_data_status"] == "not_opened"
    assert result["domain_split_count"] == 4
