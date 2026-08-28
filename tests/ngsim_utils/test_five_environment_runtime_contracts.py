from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from highway_env.envs.ngsim_env import NGSimEnv
from highway_env.ngsim_utils.core.config import resolve_idm_parameters
from highway_env.ngsim_utils.core.constants import (
    MORINOMIYA_MANIFEST_ENVIRONMENT_IDS,
)
from highway_env.ngsim_utils.data.prebuilt import (
    ELIGIBILITY_REASON_BITS,
    DatasetManifestV2,
    EpisodeStoreV2,
)
from highway_env.ngsim_utils.road.manifest_road import (
    RoadGeometryV3,
    build_road_network,
)
from highway_env.ngsim_utils.vehicles.replay import NGSIMVehicle

HEADING_CONTRACT = "causal_source_motion_or_train_road_tangent_v1"
MASK_CONTRACT = "morinomiya_section_eligibility_masks_v1"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _geometry_payload(environment_id: str) -> dict:
    lanes = []
    for raw_lane_id in (1, 2, 3):
        lateral = 4.0 * (raw_lane_id - 1)
        lanes.append(
            {
                "lane_id": f"lane-{raw_lane_id}",
                "polyline_xy_m": [[0.0, lateral], [300.0, lateral]],
                "width_m": 4.0,
                "raw_lane_ids": [raw_lane_id],
                "successors": [],
                "forbidden": False,
            }
        )
    return {
        "schema_version": 3,
        "contract_id": "road_geometry_v3",
        "site_id": "morinomiya",
        "environment_id": environment_id,
        "coordinate_frame": {"name": "morinomiya_local_xy", "units": "m"},
        "fit_split": "train",
        "test_rows_used": False,
        "nodes": [{"node_id": "section_start"}, {"node_id": "section_end"}],
        "edges": [
            {
                "edge_id": "section",
                "from_node": "section_start",
                "to_node": "section_end",
                "lanes": lanes,
            }
        ],
    }


def _episode_arrays(frames: int = 200) -> dict[str, np.ndarray]:
    vehicles = 3
    timestamps_ms = 10_000 + 100 * np.arange(frames, dtype=np.int64)
    vehicle_ids = np.asarray([101, 202, 303], dtype=np.int64)
    dimensions_m = np.asarray(
        [[4.5, 1.8], [4.7, 1.9], [4.3, 1.75]], dtype=np.float64
    )
    states = np.zeros((frames, vehicles, 4), dtype=np.float64)
    for column, raw_lane_id in enumerate((1, 2, 3)):
        states[:, column, 0] = 10.0 + 0.5 * np.arange(frames)
        states[:, column, 1] = 4.0 * column
        states[:, column, 2] = 5.0
        states[:, column, 3] = raw_lane_id
    active = np.ones((frames, vehicles), dtype=bool)
    provider = np.ones((frames, vehicles), dtype=np.int8)
    provider[1::2, 0] = 0
    road_valid = np.ones((frames, vehicles), dtype=bool)
    road_valid[50, 1] = False
    heading_valid = np.ones((frames, vehicles), dtype=bool)
    heading_derivation = np.ones((frames, vehicles), dtype=np.int8)
    reasons = np.zeros((frames, vehicles), dtype=np.uint16)
    reasons[provider == 0] |= np.uint16(
        ELIGIBILITY_REASON_BITS["provider_interpolated"]
    )
    reasons[50, 1] |= np.uint16(
        ELIGIBILITY_REASON_BITS["unsupported_or_forbidden_lane"]
    )
    reasons[:, 2] |= np.uint16(
        ELIGIBILITY_REASON_BITS["section_context_or_boundary_only"]
    )
    training_eligible = active & road_valid & (reasons == 0)
    return {
        "timestamps_ms": timestamps_ms,
        "vehicle_ids": vehicle_ids,
        "dimensions_m": dimensions_m,
        "states": states,
        "active_mask": active,
        "provider_mask": provider,
        "controlled_vehicle_eligible_mask": np.asarray(
            [True, False, False], dtype=bool
        ),
        "road_valid_mask": road_valid,
        "training_eligible_mask": training_eligible,
        "eligibility_reason_mask": reasons,
        "heading_rad": np.zeros((frames, vehicles), dtype=np.float64),
        "heading_valid_mask": heading_valid,
        "heading_derivation": heading_derivation,
    }


def _write_contracts(
    tmp_path: Path,
    environment_id: str,
    *,
    controlled_mask: np.ndarray | None = None,
    frame_count: int = 200,
) -> tuple[Path, Path, Path]:
    geometry = _geometry_payload(environment_id)
    road_path = _write_json(tmp_path / "ROAD_GEOMETRY.json", geometry)
    store_root = tmp_path / "store"
    shard_path = store_root / "episodes" / "validation-episode.npz"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _episode_arrays(frame_count)
    if controlled_mask is not None:
        arrays["controlled_vehicle_eligible_mask"] = np.asarray(
            controlled_mask, dtype=bool
        )
    np.savez(shard_path, **arrays)
    sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    shared_metadata = {
        "environment_id": environment_id,
        "state_position_reference": "vehicle_body_center",
        "source_position_reference": "vehicle_body_center",
        "heading_contract": HEADING_CONTRACT,
        "mask_contract": MASK_CONTRACT,
        "eligibility_reason_bits": ELIGIBILITY_REASON_BITS,
    }
    index_path = store_root / "index.jsonl"
    _write_json(
        index_path,
        {
            "episode_id": "validation-episode",
            "relative_path": "episodes/validation-episode.npz",
            "sha256": sha256,
            "session_id": "validation-session",
            "split": "validation",
            "start_time_ms": int(arrays["timestamps_ms"][0]),
            "frame_count": frame_count,
            "vehicle_count": 3,
            "controlled_candidate_vehicle_count": int(
                np.count_nonzero(arrays["controlled_vehicle_eligible_mask"])
            ),
            **shared_metadata,
        },
    )
    index_path.write_text(index_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    store_path = _write_json(
        store_root / "EPISODE_STORE.json",
        {
            "schema_version": 2,
            "contract_id": "episode_store_v2",
            "site_id": "morinomiya",
            "split": "validation",
            "index_file": "index.jsonl",
            "index_sha256": index_sha256,
            "episode_count": 1,
            "controlled_candidate_vehicle_count": int(
                np.count_nonzero(arrays["controlled_vehicle_eligible_mask"])
            ),
            **shared_metadata,
        },
    )
    site_path = _write_json(
        tmp_path / "DATASET_MANIFEST.json",
        {
            "schema_version": 2,
            "contract_id": "dataset_manifest_v2",
            "dataset_id": f"{environment_id}-canary",
            "site_id": "morinomiya",
            "scene": environment_id,
            "environment_id": environment_id,
            "source_artifacts": [],
            "sessions": [],
            "coordinate_frame": geometry["coordinate_frame"],
            "split_contract": {"method": "session_locked"},
            "episode_store": {
                "validation": {
                    "relative_path": "store/EPISODE_STORE.json",
                    "sha256": hashlib.sha256(store_path.read_bytes()).hexdigest(),
                }
            },
            "road_geometry": {
                "relative_path": "ROAD_GEOMETRY.json",
                "sha256": hashlib.sha256(road_path.read_bytes()).hexdigest(),
            },
            "test_accessed": False,
        },
    )
    return site_path, store_path, road_path


@pytest.mark.parametrize(
    "environment_id", sorted(MORINOMIYA_MANIFEST_ENVIRONMENT_IDS)
)
def test_four_morinomiya_ids_bind_exactly_and_use_japanese_idm(
    tmp_path: Path, environment_id: str
) -> None:
    site_path, store_path, road_path = _write_contracts(tmp_path, environment_id)
    site = DatasetManifestV2.from_source(site_path)
    store = EpisodeStoreV2(store_path)
    road = RoadGeometryV3.from_source(road_path)
    road.validate_dataset(site)

    assert (site.environment_id, store.environment_id, road.environment_id) == (
        environment_id,
        environment_id,
        environment_id,
    )
    assert resolve_idm_parameters(environment_id, {})["profile"] == "japanese"
    episode = store.load_episode("validation-episode")
    assert episode.road_valid_mask.dtype == np.bool_
    assert episode.training_eligible_mask.dtype == np.bool_
    assert episode.eligibility_reason_mask.dtype == np.uint16
    assert episode.controlled_vehicle_eligible_mask.dtype == np.bool_
    assert "heading_rad" in episode.as_legacy_trajectory_dict()[101]


def test_controlled_selection_uses_200_frame_road_and_heading_window_not_training_mask(
    tmp_path: Path,
) -> None:
    environment_id = "morinomiya_0_800"
    site_path, store_path, road_path = _write_contracts(tmp_path, environment_id)
    env = NGSimEnv(
        {
            "scene": environment_id,
            "site_manifest": str(site_path),
            "episode_store": str(store_path),
            "road_geometry": str(road_path),
            "prebuilt_split": "validation",
            "simulation_period": {"episode_name": "validation-episode"},
            "ego_vehicle_ID": 101,
            "percentage_controlled_vehicles": 1.0,
            "max_surrounding": "all",
            "action_mode": "teleport",
            "scene_dataset_collection_mode": True,
            "disable_scene_collection_spawn_safety": True,
            "show_trajectories": False,
        }
    )
    try:
        assert env.ego_ids == [101]
        assert int(np.count_nonzero(env._current_episode_data.training_eligible_mask[:, 0])) == 100
        surrounding_ids = {
            int(vehicle.vehicle_ID)
            for vehicle in env.road.vehicles
            if isinstance(vehicle, NGSIMVehicle)
        }
        assert surrounding_ids == {202, 303}
        diagnostics = env.manifest_runtime_diagnostics()
        assert diagnostics["training_eligible_state_count"] == 299
        assert diagnostics["controlled_vehicle_eligible_count"] == 1
        assert diagnostics["render_equal_aspect"] is None
        assert diagnostics["loaded_episode_ids"] == ("validation-episode",)
    finally:
        env.close()

    with pytest.raises(ValueError, match="uniquely core-anchor eligible"):
        NGSimEnv(
            {
                "scene": environment_id,
                "site_manifest": str(site_path),
                "episode_store": str(store_path),
                "road_geometry": str(road_path),
                "prebuilt_split": "validation",
                "simulation_period": {"episode_name": "validation-episode"},
                "ego_vehicle_ID": 303,
                "show_trajectories": False,
            }
        )


def test_sample_anchor_v2_accepts_explicit_episode_member_outside_controlled_mask(
    tmp_path: Path,
) -> None:
    environment_id = "morinomiya_0_800"
    site_path, store_path, road_path = _write_contracts(tmp_path, environment_id)
    env = NGSimEnv(
        {
            "scene": environment_id,
            "site_manifest": str(site_path),
            "episode_store": str(store_path),
            "road_geometry": str(road_path),
            "prebuilt_split": "validation",
            "simulation_period": {"episode_name": "validation-episode"},
            "ego_vehicle_ID": 303,
            "percentage_controlled_vehicles": 1.0,
            "max_surrounding": "all",
            "action_mode": "teleport",
            "scene_dataset_collection_mode": True,
            "runtime_eligibility_mode": "sample_anchor_v2",
            "controlled_vehicle_min_occupancy": 0.0,
            "disable_scene_collection_spawn_safety": True,
            "disable_background_replay_spawn_safety": True,
            "record_vehicle_lifecycle": True,
            "allow_idm": False,
            "show_trajectories": False,
        }
    )
    try:
        assert env.ego_ids == [303]
        assert 303 in env._valid_ids_by_episode["validation-episode"]
    finally:
        env.close()

    with pytest.raises(ValueError, match="present in the selected EpisodeStore episode"):
        NGSimEnv(
            {
                "scene": environment_id,
                "site_manifest": str(site_path),
                "episode_store": str(store_path),
                "road_geometry": str(road_path),
                "prebuilt_split": "validation",
                "simulation_period": {"episode_name": "validation-episode"},
                "ego_vehicle_ID": 999,
                "percentage_controlled_vehicles": 1.0,
                "runtime_eligibility_mode": "sample_anchor_v2",
                "show_trajectories": False,
            }
        )


def test_sample_anchor_v2_uses_52_frame_minimum_without_relaxing_legacy_window(
    tmp_path: Path,
) -> None:
    environment_id = "morinomiya_0_800"
    site_path, store_path, road_path = _write_contracts(
        tmp_path / "valid-short",
        environment_id,
        frame_count=187,
    )
    config = {
        "scene": environment_id,
        "site_manifest": str(site_path),
        "episode_store": str(store_path),
        "road_geometry": str(road_path),
        "prebuilt_split": "validation",
        "simulation_period": {"episode_name": "validation-episode"},
        "ego_vehicle_ID": 303,
        "percentage_controlled_vehicles": 1.0,
        "max_surrounding": "all",
        "action_mode": "teleport",
        "scene_dataset_collection_mode": True,
        "disable_scene_collection_spawn_safety": True,
        "show_trajectories": False,
    }
    env = NGSimEnv({**config, "runtime_eligibility_mode": "sample_anchor_v2"})
    try:
        assert env.ego_ids == [303]
        assert len(env._current_episode_data.timestamps_ms) == 187
    finally:
        env.close()

    with pytest.raises(ValueError, match="at least 200 frames"):
        NGSimEnv(config)

    too_short_site, too_short_store, too_short_road = _write_contracts(
        tmp_path / "invalid-short",
        environment_id,
        frame_count=51,
    )
    with pytest.raises(ValueError, match="at least 52 frames for sample_anchor_v2"):
        NGSimEnv(
            {
                **config,
                "site_manifest": str(too_short_site),
                "episode_store": str(too_short_store),
                "road_geometry": str(too_short_road),
                "runtime_eligibility_mode": "sample_anchor_v2",
            }
        )


def test_atomic_contract_and_environment_identity_fail_closed(tmp_path: Path) -> None:
    site_path, store_path, road_path = _write_contracts(
        tmp_path, "morinomiya_800_1600"
    )
    with pytest.raises(ValueError, match="atomic runtime contract"):
        NGSimEnv(
            {
                "scene": "morinomiya_800_1600",
                "site_manifest": str(site_path),
                "episode_store": str(store_path),
            }
        )

    site_payload = json.loads(site_path.read_text(encoding="utf-8"))
    site_payload.pop("environment_id")
    with pytest.raises(ValueError, match="requires environment_id"):
        DatasetManifestV2.from_source(site_payload)

    road_payload = json.loads(road_path.read_text(encoding="utf-8"))
    road_payload["environment_id"] = "morinomiya_1600_2400"
    _write_json(road_path, road_payload)
    with pytest.raises(ValueError, match="environment_id values do not match"):
        NGSimEnv(
            {
                "scene": "morinomiya_800_1600",
                "site_manifest": str(site_path),
                "episode_store": str(store_path),
                "road_geometry": str(road_path),
                "prebuilt_split": "validation",
            }
        )


def test_explicit_terminal_successor_does_not_infer_cross_section_handoff() -> None:
    geometry = RoadGeometryV3.from_source(
        _geometry_payload("morinomiya_2400_end")
    )
    network = build_road_network(geometry)
    lane_index = network.manifest_lane_index_by_id["lane-1"]
    # Even a caller-provided route cannot override the manifest's empty
    # successor list. This is the no-section-handoff contract.
    assert network.next_lane(
        lane_index,
        route=[lane_index, ("section_end", "elsewhere", 0)],
        position=np.asarray([300.0, 0.0]),
    ) == lane_index

    invalid = _geometry_payload("morinomiya_2400_end")
    invalid["nodes"].append({"node_id": "disconnected"})
    invalid["edges"].append(
        {
            "edge_id": "other",
            "from_node": "disconnected",
            "to_node": "section_start",
            "lanes": [
                {
                    "lane_id": "other-lane",
                    "polyline_xy_m": [[-20.0, 0.0], [0.0, 0.0]],
                    "width_m": 4.0,
                    "raw_lane_ids": [9],
                    "successors": [],
                    "forbidden": False,
                }
            ],
        }
    )
    invalid["edges"][0]["lanes"][0]["successors"] = ["other-lane"]
    with pytest.raises(ValueError, match="do not start at node"):
        RoadGeometryV3.from_source(invalid)


def test_adjacent_store_controlled_ownership_is_unique_and_context_actor_stays_background(
    tmp_path: Path,
) -> None:
    first_paths = _write_contracts(
        tmp_path / "first",
        "morinomiya_0_800",
        controlled_mask=np.asarray([True, False, False]),
    )
    second_paths = _write_contracts(
        tmp_path / "second",
        "morinomiya_800_1600",
        controlled_mask=np.asarray([False, True, False]),
    )
    first = EpisodeStoreV2(first_paths[1]).load_episode("validation-episode")
    second = EpisodeStoreV2(second_paths[1]).load_episode("validation-episode")
    assert not np.any(
        first.controlled_vehicle_eligible_mask
        & second.controlled_vehicle_eligible_mask
    )
    assert first.as_legacy_trajectory_dict()[303][
        "controlled_vehicle_eligible"
    ] is False


def test_controlled_eligibility_shape_dtype_and_store_support_fail_closed(
    tmp_path: Path,
) -> None:
    _, store_path, _ = _write_contracts(tmp_path, "morinomiya_2400_end")
    shard = tmp_path / "store/episodes/validation-episode.npz"
    arrays = _episode_arrays()
    arrays["controlled_vehicle_eligible_mask"] = np.asarray([1, 0, 0], dtype=np.int8)
    np.savez(shard, **arrays)
    row = json.loads((tmp_path / "store/index.jsonl").read_text(encoding="utf-8"))
    row["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    index_path = tmp_path / "store/index.jsonl"
    index_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads(store_path.read_text(encoding="utf-8"))
    manifest["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    _write_json(store_path, manifest)
    with pytest.raises(ValueError, match="controlled_vehicle_eligible_mask.*dtype bool"):
        EpisodeStoreV2(store_path).load_episode("validation-episode")

    manifest["controlled_candidate_vehicle_count"] = 0
    row["controlled_candidate_vehicle_count"] = 0
    index_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    _write_json(store_path, manifest)
    with pytest.raises(ValueError, match="requires at least one controlled candidate"):
        EpisodeStoreV2(store_path)


def test_new_store_mask_dtype_and_invariants_fail_closed(tmp_path: Path) -> None:
    _, store_path, _ = _write_contracts(tmp_path, "morinomiya_1600_2400")
    shard = tmp_path / "store/episodes/validation-episode.npz"
    arrays = _episode_arrays()
    arrays["eligibility_reason_mask"] = arrays["eligibility_reason_mask"].astype(
        np.int64
    )
    np.savez(shard, **arrays)
    row = json.loads((tmp_path / "store/index.jsonl").read_text(encoding="utf-8"))
    row["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    index_path = tmp_path / "store/index.jsonl"
    index_path.write_text(
        json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = json.loads(store_path.read_text(encoding="utf-8"))
    manifest["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    _write_json(store_path, manifest)
    with pytest.raises(ValueError, match="eligibility_reason_mask must have dtype uint16"):
        EpisodeStoreV2(store_path).load_episode("validation-episode")


def test_five_environment_artifact_and_index_hashes_fail_closed(tmp_path: Path) -> None:
    site_path, store_path, road_path = _write_contracts(
        tmp_path, "morinomiya_0_800"
    )
    index_path = tmp_path / "store/index.jsonl"
    index_path.write_text(
        index_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="index_sha256 mismatch"):
        EpisodeStoreV2(store_path)

    # Restore the index hash so the independent DatasetManifest road binding is
    # the first failure reached.
    store_payload = json.loads(store_path.read_text(encoding="utf-8"))
    store_payload["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    _write_json(store_path, store_payload)
    site_payload = json.loads(site_path.read_text(encoding="utf-8"))
    site_payload["episode_store"]["validation"]["sha256"] = hashlib.sha256(
        store_path.read_bytes()
    ).hexdigest()
    _write_json(site_path, site_payload)
    road_payload = json.loads(road_path.read_text(encoding="utf-8"))
    road_payload["producer_extra"] = "tampered-after-manifest"
    _write_json(road_path, road_payload)
    with pytest.raises(ValueError, match="road_geometry SHA-256"):
        NGSimEnv(
            {
                "scene": "morinomiya_0_800",
                "site_manifest": str(site_path),
                "episode_store": str(store_path),
                "road_geometry": str(road_path),
                "prebuilt_split": "validation",
            }
        )
