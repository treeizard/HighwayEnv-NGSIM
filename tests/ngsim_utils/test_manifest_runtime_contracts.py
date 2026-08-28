from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from highway_env.envs.ngsim_env import NGSimEnv
from highway_env.ngsim_utils.data.prebuilt import (
    DatasetManifestV2,
    EpisodeStoreV2,
    load_prebuilt_data,
)
from highway_env.ngsim_utils.road.manifest_road import (
    RoadGeometryV3,
    build_road_network,
)
from highway_env.ngsim_utils.vehicles.replay import NGSIMVehicle
from highway_env.road.lane import LineType, PolyLaneFixedWidth


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _episode_arrays(
    *, offset: float = 0.0, frames: int = 20, include_i80_reference: bool = False
) -> dict[str, np.ndarray]:
    timestamps_ms = np.arange(frames, dtype=np.int64) * 100 + 1_000
    vehicle_ids = np.asarray([101, 202], dtype=np.int64)
    dimensions_m = np.asarray([[4.5, 1.8], [4.8, 1.9]], dtype=np.float64)
    states = np.zeros((frames, 2, 4), dtype=np.float64)
    states[:, 0, 0] = offset + 10.0 + np.arange(frames)
    states[:, 0, 1] = 0.0
    states[:, 0, 2] = 10.0
    states[:, 0, 3] = 1.0
    states[:, 1, 0] = offset + 12.0 + np.arange(frames)
    states[:, 1, 1] = 4.0
    states[:, 1, 2] = 10.0
    states[:, 1, 3] = 2.0
    result = {
        "timestamps_ms": timestamps_ms,
        "vehicle_ids": vehicle_ids,
        "dimensions_m": dimensions_m,
        "states": states,
        "active_mask": np.ones((frames, 2), dtype=bool),
        "provider_mask": np.ones((frames, 2), dtype=np.int8),
    }
    if include_i80_reference:
        heading_rad = np.zeros((frames, 2), dtype=np.float64)
        heading_rad[:, 1] = 0.2
        result.update(
            {
                "source_front_center_xy_m": states[:, :, :2]
                + 0.5
                * dimensions_m[None, :, 0, None]
                * np.stack((np.cos(heading_rad), np.sin(heading_rad)), axis=-1),
                "heading_rad": heading_rad,
                "heading_valid_mask": np.ones((frames, 2), dtype=bool),
                "heading_derivation": np.ones((frames, 2), dtype=np.int8),
            }
        )
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_episode_store(
    root: Path,
    *,
    episodes: tuple[str, ...] = ("ep-1",),
    site_id: str = "i80-site",
    split: str = "train",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_number, episode_id in enumerate(episodes):
        relative_path = f"episodes/{episode_id}.npz"
        shard = root / relative_path
        shard.parent.mkdir(parents=True, exist_ok=True)
        arrays = _episode_arrays(
            offset=100.0 * episode_number,
            include_i80_reference=site_id == "i-80",
        )
        np.savez(shard, **arrays)
        rows.append(
            {
                "episode_id": episode_id,
                "relative_path": relative_path,
                "sha256": _sha256(shard),
                "session_id": f"session-{episode_number}",
                "split": split,
                "start_time_ms": int(arrays["timestamps_ms"][0]),
                "frame_count": int(arrays["states"].shape[0]),
                "vehicle_count": int(arrays["states"].shape[1]),
                "producer_extra": "accepted",
            }
        )
    index_path = root / "episodes.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_payload = {
        "schema_version": 2,
        "contract_id": "episode_store_v2",
        "site_id": site_id,
        "split": split,
        "index_file": index_path.name,
        "episode_count": len(rows),
        "producer_extra": {"accepted": True},
    }
    if site_id == "i-80":
        manifest_payload.update(
            {
                "state_position_reference": "vehicle_geometric_body_center",
                "source_position_reference": "vehicle_front_center",
                "reference_point_conversion": {"future_actor_rows_read": 0},
            }
        )
    return _write_json(
        root / "EPISODE_STORE.json",
        manifest_payload,
    )


def _road_geometry_payload(*, site_id: str = "i80-site") -> dict:
    return {
        "schema_version": 3,
        "contract_id": "road_geometry_v3",
        "site_id": site_id,
        "coordinate_frame": {"name": "site_local_xy", "units": "m"},
        "fit_split": "train",
        "test_rows_used": False,
        "nodes": [{"node_id": "s1"}, {"node_id": "s2"}],
        "edges": [
            {
                "edge_id": "mainline-1",
                "from_node": "s1",
                "to_node": "s2",
                "lanes": [
                    {
                        "lane_id": "mainline-1-lane-1",
                        "polyline_xy_m": [[0.0, 0.0], [300.0, 0.0]],
                        "width_m": 4.0,
                        "raw_lane_ids": [1],
                        "successors": [],
                        "forbidden": False,
                    },
                    {
                        "lane_id": "mainline-1-lane-2",
                        "polyline_xy_m": [[0.0, 4.0], [300.0, 4.0]],
                        "width_m": 4.0,
                        "raw_lane_ids": [2],
                        "successors": [],
                        "forbidden": False,
                    },
                ],
            }
        ],
        "producer_extra": "accepted",
    }


def _site_manifest_payload() -> dict:
    return {
        "schema_version": 2,
        "contract_id": "dataset_manifest_v2",
        "dataset_id": "i80-dataset",
        "site_id": "i80-site",
        "scene": "custom-manifest-site",
        "source_artifacts": [],
        "sessions": [],
        "coordinate_frame": {"name": "site_local_xy", "units": "m"},
        "split_contract": {"method": "session_locked"},
        "episode_store": {"train": "store/EPISODE_STORE.json"},
        "road_geometry": "ROAD_GEOMETRY.json",
        "test_accessed": False,
        "producer_extra": ["accepted"],
    }


def test_dataset_manifest_v2_is_extra_tolerant_and_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = DatasetManifestV2.from_source(
        _write_json(tmp_path / "DATASET_MANIFEST.json", _site_manifest_payload())
    )

    assert manifest.dataset_id == "i80-dataset"
    assert manifest.payload["producer_extra"] == ["accepted"]
    assert manifest.episode_store_path("train") == (
        tmp_path / "store/EPISODE_STORE.json"
    )
    assert manifest.road_geometry_path() == tmp_path / "ROAD_GEOMETRY.json"

    invalid = _site_manifest_payload()
    invalid.pop("sessions")
    with pytest.raises(ValueError, match="sessions"):
        DatasetManifestV2.from_source(invalid)

    invalid = _site_manifest_payload()
    invalid["test_accessed"] = 0
    with pytest.raises(ValueError, match="boolean test_accessed"):
        DatasetManifestV2.from_source(invalid)


def test_episode_store_is_lazy_checksum_verified_and_lru_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_episode_store(tmp_path, episodes=("ep-1", "ep-2"))
    store = EpisodeStoreV2(manifest_path, max_cached_episodes=1)
    calls: list[Path] = []
    original_sha256 = store._sha256

    def recording_sha256(path: Path) -> str:
        calls.append(path)
        return original_sha256(path)

    monkeypatch.setattr(store, "_sha256", recording_sha256)

    assert store.episode_ids == ["ep-1", "ep-2"]
    assert store.loaded_episode_ids == ()
    first = store.load_episode("ep-1")
    assert calls == [tmp_path / "episodes/ep-1.npz"]
    assert store.loaded_episode_ids == ("ep-1",)
    assert first.states.flags.writeable is False
    assert store["ep-1"] is first
    np.testing.assert_allclose(first.dimensions_m[0], [4.5, 1.8])

    store.load_episode("ep-2")
    assert store.loaded_episode_ids == ("ep-2",)

    rows = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    second = json.loads(rows[1])
    second["sha256"] = "0" * 64
    rows[1] = json.dumps(second, sort_keys=True)
    (tmp_path / "episodes.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    corrupt_store = EpisodeStoreV2(manifest_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        corrupt_store.load_episode("ep-2")


def test_episode_store_rejects_unsafe_paths_and_invalid_arrays(tmp_path: Path) -> None:
    manifest_path = _write_episode_store(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_file"] = "../outside.jsonl"
    with pytest.raises(ValueError, match="safe relative path"):
        EpisodeStoreV2(manifest, base_dir=tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = tmp_path / "episodes/ep-1.npz"
    arrays = _episode_arrays()
    arrays["states"] = arrays["states"][:, :, :3]
    np.savez(shard, **arrays)
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    rows[0]["sha256"] = _sha256(shard)
    (tmp_path / "episodes.jsonl").write_text(json.dumps(rows[0]) + "\n")
    store = EpisodeStoreV2(manifest_path)
    with pytest.raises(ValueError, match=r"shape \(20, 2, 4\)"):
        store.load_episode("ep-1")


def test_official_i80_store_requires_and_exposes_reference_point_contract(
    tmp_path: Path,
) -> None:
    manifest_path = _write_episode_store(tmp_path, site_id="i-80")
    episode = EpisodeStoreV2(manifest_path).load_episode("ep-1")

    assert episode.source_front_center_xy_m is not None
    assert episode.heading_rad is not None
    assert episode.heading_valid_mask is not None
    assert episode.heading_derivation is not None
    assert episode.heading_rad.flags.writeable is False
    record = episode.as_legacy_trajectory_dict()[101]
    assert record["heading_rad"][0] == 0.0
    assert record["heading_derivation"][0] == 1
    reconstructed = (
        episode.states[:, :, :2]
        + 0.5
        * episode.dimensions_m[None, :, 0, None]
        * np.stack(
            (np.cos(episode.heading_rad), np.sin(episode.heading_rad)), axis=-1
        )
    )
    np.testing.assert_allclose(
        reconstructed,
        episode.source_front_center_xy_m,
        rtol=0.0,
        atol=1e-9,
    )

    shard = tmp_path / "episodes/ep-1.npz"
    arrays = _episode_arrays(include_i80_reference=True)
    arrays.pop("heading_derivation")
    np.savez(shard, **arrays)
    row = json.loads((tmp_path / "episodes.jsonl").read_text(encoding="utf-8"))
    row["sha256"] = _sha256(shard)
    (tmp_path / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="partial I-80 reference-point contract"):
        EpisodeStoreV2(manifest_path).load_episode("ep-1")


def test_official_i80_store_rejects_front_body_reconstruction_drift(
    tmp_path: Path,
) -> None:
    manifest_path = _write_episode_store(tmp_path, site_id="i-80")
    shard = tmp_path / "episodes/ep-1.npz"
    arrays = _episode_arrays(include_i80_reference=True)
    arrays["source_front_center_xy_m"][0, 0, 0] += 0.01
    np.savez(shard, **arrays)
    row = json.loads((tmp_path / "episodes.jsonl").read_text(encoding="utf-8"))
    row["sha256"] = _sha256(shard)
    (tmp_path / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        EpisodeStoreV2(manifest_path).load_episode("ep-1")


@pytest.mark.parametrize(
    ("active_value", "provider_value", "message"),
    [
        (False, 1, "inactive states require provider_mask=-1"),
        (True, -1, r"active states require provider_mask in \{0,1\}"),
    ],
)
def test_episode_store_provider_mask_matches_activity_fail_closed(
    tmp_path: Path,
    active_value: bool,
    provider_value: int,
    message: str,
) -> None:
    manifest_path = _write_episode_store(tmp_path)
    shard = tmp_path / "episodes/ep-1.npz"
    arrays = _episode_arrays()
    arrays["active_mask"][0, 0] = active_value
    arrays["provider_mask"][0, 0] = provider_value
    np.savez(shard, **arrays)
    row = json.loads((tmp_path / "episodes.jsonl").read_text(encoding="utf-8"))
    row["sha256"] = _sha256(shard)
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        EpisodeStoreV2(manifest_path).load_episode("ep-1")


def test_road_geometry_v3_builds_lane_metadata_and_rejects_leakage() -> None:
    geometry = RoadGeometryV3.from_source(_road_geometry_payload())
    network = build_road_network(geometry)

    assert network.road_geometry_contract == "road_geometry_v3"
    assert network.manifest_lane_index_by_id["mainline-1-lane-2"] == (
        "s1",
        "s2",
        1,
    )
    assert network.manifest_raw_lane_indexes[2] == (("s1", "s2", 1),)
    np.testing.assert_allclose(network.get_lane(("s1", "s2", 0)).position(10, 0), [10, 0])
    assert geometry.edges[0].lanes[0].marking_profile is None
    assert network.get_lane(("s1", "s2", 0)).marking_profile is None

    invalid = _road_geometry_payload()
    invalid["test_rows_used"] = True
    with pytest.raises(ValueError, match="test_rows_used=false"):
        RoadGeometryV3.from_source(invalid)

    invalid = _road_geometry_payload()
    invalid["edges"][0]["lanes"][0]["successors"] = ["missing-lane"]
    with pytest.raises(ValueError, match="undeclared successors"):
        RoadGeometryV3.from_source(invalid)


def test_road_geometry_v3_preserves_marking_profile_in_runtime_lane() -> None:
    payload = _road_geometry_payload()
    lane_payload = payload["edges"][0]["lanes"][0]
    lane_payload["line_types"] = [LineType.CONTINUOUS_LINE, LineType.NONE]
    lane_payload["marking_profile"] = [
        {
            "start_s_m": 0.0,
            "end_s_m": 100.0,
            "line_types": [LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
        },
        {
            "start_s_m": 100.0,
            "end_s_m": 300.0,
            "line_types": [LineType.NONE, LineType.STRIPED],
        },
    ]

    geometry = RoadGeometryV3.from_source(payload)
    lane_contract = geometry.edges[0].lanes[0]
    runtime_lane = build_road_network(geometry).get_lane(("s1", "s2", 0))

    assert lane_contract.line_types == (LineType.CONTINUOUS_LINE, LineType.NONE)
    assert runtime_lane.line_types == [LineType.CONTINUOUS_LINE, LineType.NONE]
    assert [interval.to_config() for interval in lane_contract.marking_profile] == lane_payload["marking_profile"]
    assert runtime_lane.marking_profile == lane_contract.marking_profile


def test_marking_profile_exact_length_survives_manifest_runtime_handoff(tmp_path: Path) -> None:
    points = [[float(index), float(np.sin(index / 7.0))] for index in range(128)]
    lane_length = PolyLaneFixedWidth(points).length
    payload = _road_geometry_payload()
    lane_payload = payload["edges"][0]["lanes"][0]
    lane_payload["polyline_xy_m"] = points
    lane_payload["marking_profile"] = [
        {
            "start_s_m": 0.0,
            "end_s_m": lane_length,
            "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
        }
    ]

    geometry = RoadGeometryV3.from_source(_write_json(tmp_path / "ROAD_GEOMETRY.json", payload))
    runtime_lane = build_road_network(geometry).get_lane(("s1", "s2", 0))

    assert runtime_lane.length == lane_length
    assert runtime_lane.marking_profile == geometry.edges[0].lanes[0].marking_profile


@pytest.mark.parametrize(
    "marking_profile",
    [
        None,
        [],
        [
            {
                "start_s_m": 1.0,
                "end_s_m": 300.0,
                "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
            }
        ],
        [
            {
                "start_s_m": 0.0,
                "end_s_m": 100.0,
                "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
            },
            {
                "start_s_m": 101.0,
                "end_s_m": 300.0,
                "line_types": [LineType.NONE, LineType.STRIPED],
            },
        ],
        [
            {
                "start_s_m": 0.0,
                "end_s_m": 100.0,
                "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
            },
            {
                "start_s_m": 100.0000000001,
                "end_s_m": 300.0,
                "line_types": [LineType.NONE, LineType.STRIPED],
            },
        ],
        [
            {
                "start_s_m": 0.0,
                "end_s_m": 100.0,
                "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
            },
            {
                "start_s_m": 99.0,
                "end_s_m": 300.0,
                "line_types": [LineType.NONE, LineType.STRIPED],
            },
        ],
        [
            {
                "start_s_m": 0.0,
                "end_s_m": 0.0,
                "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
            },
            {
                "start_s_m": 0.0,
                "end_s_m": 300.0,
                "line_types": [LineType.NONE, LineType.STRIPED],
            },
        ],
        [
            {
                "start_s_m": 0.0,
                "end_s_m": 299.0,
                "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
            }
        ],
        [
            {
                "start_s_m": 0.0,
                "end_s_m": 301.0,
                "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
            }
        ],
        [
            {
                "start_s_m": 0.0,
                "end_s_m": 300.0,
                "line_types": [LineType.CONTINUOUS_LINE],
            }
        ],
        [
            {
                "start_s_m": 0.0,
                "end_s_m": 300.0,
                "line_types": [LineType.CONTINUOUS_LINE, 99],
            }
        ],
    ],
)
def test_road_geometry_v3_rejects_invalid_marking_profile(marking_profile) -> None:
    payload = _road_geometry_payload()
    payload["edges"][0]["lanes"][0]["marking_profile"] = marking_profile

    with pytest.raises(ValueError, match="marking_profile"):
        RoadGeometryV3.from_source(payload)


def test_polylane_marking_profile_round_trips_and_rejects_gaps() -> None:
    profile = [
        {
            "start_s_m": 0.0,
            "end_s_m": 4.0,
            "line_types": [LineType.CONTINUOUS_LINE, LineType.NONE],
        },
        {
            "start_s_m": 4.0,
            "end_s_m": 10.5,
            "line_types": [LineType.NONE, LineType.STRIPED],
        },
    ]
    lane = PolyLaneFixedWidth(
        [(0.0, 0.0), (10.5, 0.0)],
        line_types=[LineType.CONTINUOUS_LINE, LineType.NONE],
        marking_profile=profile,
    )

    config = lane.to_config()["config"]
    restored = PolyLaneFixedWidth.from_config(config)

    assert config["marking_profile"] == profile
    assert restored.marking_profile == lane.marking_profile
    assert restored.line_types == lane.line_types

    invalid = [dict(interval) for interval in profile]
    invalid[1]["start_s_m"] = 4.1
    with pytest.raises(ValueError, match="exact contiguous partition"):
        PolyLaneFixedWidth(
            [(0.0, 0.0), (10.5, 0.0)],
            marking_profile=invalid,
        )


def test_ngsim_env_uses_explicit_store_si_units_and_generic_road(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = _write_episode_store(tmp_path / "store")
    road_path = _write_json(tmp_path / "ROAD_GEOMETRY.json", _road_geometry_payload())
    site_path = _write_json(tmp_path / "DATASET_MANIFEST.json", _site_manifest_payload())
    sealed_test_bytes = tmp_path / "sealed-test-episode.npz"
    sealed_test_bytes.write_bytes(b"must not be opened by a train reset")
    hashed_paths: list[Path] = []
    original_sha256 = EpisodeStoreV2._sha256

    def recording_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return original_sha256(path)

    monkeypatch.setattr(EpisodeStoreV2, "_sha256", staticmethod(recording_sha256))

    env = NGSimEnv(
        {
            "scene": "custom-manifest-site",
            "site_manifest": str(site_path),
            "episode_store": str(store_path),
            "road_geometry": str(road_path),
            "prebuilt_split": "train",
            "simulation_period": {"episode_name": "ep-1"},
            "ego_vehicle_ID": 101,
            "percentage_controlled_vehicles": 1,
            "max_surrounding": 1,
            "show_trajectories": False,
        }
    )
    try:
        assert env._episode_store.loaded_episode_ids == ("ep-1",)
        assert env._traj_all_by_episode == {}
        np.testing.assert_allclose(env.vehicle.position, [10.0, 0.0])
        assert env.vehicle.LENGTH == pytest.approx(4.5)
        assert env.vehicle.WIDTH == pytest.approx(1.8)
        surroundings = [
            vehicle
            for vehicle in env.road.vehicles
            if isinstance(vehicle, NGSIMVehicle)
        ]
        assert len(surroundings) == 1
        assert surroundings[0].real_length == pytest.approx(4.8)
        assert surroundings[0].real_width == pytest.approx(1.9)
        assert set(hashed_paths) == {
            tmp_path / "store/episodes.jsonl",
            tmp_path / "store/episodes/ep-1.npz",
        }
        assert sealed_test_bytes not in hashed_paths
    finally:
        env.close()


def test_legacy_prebuilt_loader_keeps_four_tuple_contract(tmp_path: Path) -> None:
    prebuilt = tmp_path / "us-101/prebuilt"
    prebuilt.mkdir(parents=True)
    raw_ids = {"ep": np.asarray([7], dtype=np.int64)}
    raw_trajectory = np.asarray(
        [[6.0, 30.0, 32.81, 1.0], [6.0, 33.281, 32.81, 1.0]],
        dtype=float,
    )
    trajectories = {
        "ep": {
            7: {
                "length": 14.0,
                "width": 6.0,
                "trajectory": raw_trajectory,
            }
        }
    }
    np.save(prebuilt / "veh_ids_train.npy", raw_ids, allow_pickle=True)
    np.save(prebuilt / "trajectory_train.npy", trajectories, allow_pickle=True)

    result = load_prebuilt_data(
        str(tmp_path),
        "us-101",
        "train",
        min_occupancy=0.5,
        cache={},
    )

    assert isinstance(result, tuple) and len(result) == 4
    prebuilt_dir, valid_ids, loaded_trajectories, episodes = result
    assert prebuilt_dir == str(prebuilt)
    assert episodes == ["ep"]
    np.testing.assert_array_equal(valid_ids["ep"], [7])
    np.testing.assert_array_equal(
        loaded_trajectories["ep"][7]["trajectory"], raw_trajectory
    )
