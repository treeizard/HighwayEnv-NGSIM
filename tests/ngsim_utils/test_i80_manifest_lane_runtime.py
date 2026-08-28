from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from highway_env.envs.ngsim_env import NGSimEnv
from highway_env.ngsim_utils.data.prebuilt import ELIGIBILITY_REASON_BITS
from highway_env.ngsim_utils.road.lane_mapping import (
    resolve_target_lane_index_from_row,
    target_lane_index_from_position_and_lane_id,
)
from highway_env.ngsim_utils.road.manifest_road import (
    RoadGeometryV3,
    build_road_network,
)
from highway_env.ngsim_utils.vehicles.replay import (
    NGSIMVehicle,
    road_entity_pose_polygon,
)
from highway_env.road.road import Road


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _i80_geometry_payload() -> dict:
    def lane(
        lane_id: str,
        raw_lane_id: int | list[int],
        points: list[list[float]],
        successors: list[str],
    ) -> dict:
        return {
            "lane_id": lane_id,
            "polyline_xy_m": points,
            "width_m": 3.7,
            "raw_lane_ids": (
                [raw_lane_id] if isinstance(raw_lane_id, int) else raw_lane_id
            ),
            "successors": successors,
            "forbidden": False,
        }

    upstream = [
        lane(
            f"main_up_{raw_lane_id}",
            raw_lane_id,
            [[4.0 * (raw_lane_id - 1), 0.0], [4.0 * (raw_lane_id - 1), 80.0]],
            [f"merge_{raw_lane_id}"],
        )
        for raw_lane_id in range(1, 7)
    ]
    merge = [
        lane(
            f"merge_{raw_lane_id}",
            raw_lane_id,
            [[4.0 * (raw_lane_id - 1), 80.0], [4.0 * (raw_lane_id - 1), 120.0]],
            [f"main_down_{raw_lane_id}"],
        )
        for raw_lane_id in range(1, 7)
    ]
    merge.append(
        lane(
            "merge_7",
            [6, 7],
            [[24.0, 80.0], [22.0, 100.0], [20.0, 120.0]],
            ["main_down_6"],
        )
    )
    downstream = [
        lane(
            f"main_down_{raw_lane_id}",
            raw_lane_id,
            [[4.0 * (raw_lane_id - 1), 120.0], [4.0 * (raw_lane_id - 1), 300.0]],
            [],
        )
        for raw_lane_id in range(1, 7)
    ]
    ramp = lane(
        "ramp_approach_7",
        7,
        [[30.0, 0.0], [26.25, 50.0], [24.0, 80.0]],
        ["merge_7"],
    )
    return {
        "schema_version": 3,
        "contract_id": "road_geometry_v3",
        "site_id": "i-80",
        "environment_id": "i-80",
        "coordinate_frame": {
            "name": "i80_local_xy_m",
            "units": "m",
            "x_axis": "lateral",
            "y_axis": "longitudinal",
        },
        "fit_split": "train",
        "test_rows_used": False,
        "nodes": [
            {"node_id": "mainline_start"},
            {"node_id": "merge_start"},
            {"node_id": "merge_complete"},
            {"node_id": "mainline_end"},
            {"node_id": "ramp_start"},
        ],
        "edges": [
            {
                "edge_id": "mainline_upstream",
                "from_node": "mainline_start",
                "to_node": "merge_start",
                "lanes": upstream,
            },
            {
                "edge_id": "shared_merge_zone",
                "from_node": "merge_start",
                "to_node": "merge_complete",
                "lanes": merge,
            },
            {
                "edge_id": "mainline_downstream",
                "from_node": "merge_complete",
                "to_node": "mainline_end",
                "lanes": downstream,
            },
            {
                "edge_id": "on_ramp",
                "from_node": "ramp_start",
                "to_node": "merge_start",
                "lanes": [ramp],
            },
        ],
    }


def _ramp_episode_arrays(frames: int = 200) -> dict[str, np.ndarray]:
    timestamps_ms = 1_000 + 100 * np.arange(frames, dtype=np.int64)
    vehicle_ids = np.asarray([101, 202], dtype=np.int64)
    dimensions_m = np.asarray([[4.5, 1.8], [4.8, 1.9]], dtype=np.float64)
    states = np.zeros((frames, 2, 4), dtype=np.float64)
    y = np.linspace(0.0, 199.0, frames)
    on_ramp = y < 80.0
    in_taper = (y >= 80.0) & (y < 120.0)
    states[:, 0, 0] = np.where(
        on_ramp,
        30.0 - 0.075 * y,
        np.where(in_taper, 24.0 - 0.1 * (y - 80.0), 20.0),
    )
    states[:, 0, 1] = y
    states[:, 0, 2] = 10.0
    states[:, 0, 3] = np.where(y < 120.0, 7.0, 6.0)
    states[:, 1, 0] = 16.0
    states[:, 1, 1] = y
    states[:, 1, 2] = 10.0
    states[:, 1, 3] = 5.0
    heading_rad = np.empty((frames, 2), dtype=np.float64)
    heading_rad[:, 0] = 1.25 + 0.001 * np.arange(frames, dtype=np.float64)
    heading_rad[:, 1] = 1.10 + 0.0005 * np.arange(frames, dtype=np.float64)
    heading_valid_mask = np.ones((frames, 2), dtype=bool)
    heading_derivation = np.ones((frames, 2), dtype=np.int8)
    source_front_center_xy_m = states[:, :, :2] + 0.5 * dimensions_m[
        None, :, 0, None
    ] * np.stack((np.cos(heading_rad), np.sin(heading_rad)), axis=-1)
    return {
        "timestamps_ms": timestamps_ms,
        "vehicle_ids": vehicle_ids,
        "dimensions_m": dimensions_m,
        "states": states,
        "active_mask": np.ones((frames, 2), dtype=bool),
        "provider_mask": np.ones((frames, 2), dtype=np.int8),
        "controlled_vehicle_eligible_mask": np.ones(2, dtype=bool),
        "road_valid_mask": np.ones((frames, 2), dtype=bool),
        "training_eligible_mask": np.ones((frames, 2), dtype=bool),
        "eligibility_reason_mask": np.zeros((frames, 2), dtype=np.uint16),
        "source_front_center_xy_m": source_front_center_xy_m,
        "heading_rad": heading_rad,
        "heading_valid_mask": heading_valid_mask,
        "heading_derivation": heading_derivation,
    }


def _write_runtime_contracts(tmp_path: Path) -> tuple[Path, Path, Path]:
    geometry_payload = _i80_geometry_payload()
    geometry_path = tmp_path / "ROAD_GEOMETRY.json"
    geometry_path.write_text(json.dumps(geometry_payload), encoding="utf-8")

    store_root = tmp_path / "episode_store" / "train"
    episode_path = store_root / "episodes" / "ep-ramp.npz"
    episode_path.parent.mkdir(parents=True)
    arrays = _ramp_episode_arrays()
    np.savez(episode_path, **arrays)
    index_path = store_root / "episodes.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "episode_id": "ep-ramp",
                "relative_path": "episodes/ep-ramp.npz",
                "sha256": _sha256(episode_path),
                "session_id": "0400pm-0415pm",
                "split": "train",
                "environment_id": "i-80",
                "state_position_reference": "vehicle_geometric_body_center",
                "source_position_reference": "vehicle_front_center",
                "heading_contract": "exact_backward_source_displacement_then_current_lane_tangent_then_past_hold_v2",
                "mask_contract": "source_row_present_v1",
                "eligibility_reason_bits": ELIGIBILITY_REASON_BITS,
                "start_time_ms": int(arrays["timestamps_ms"][0]),
                "frame_count": 200,
                "vehicle_count": 2,
                "controlled_candidate_vehicle_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store_path = store_root / "EPISODE_STORE.json"
    store_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "contract_id": "episode_store_v2",
                "site_id": "i-80",
                "environment_id": "i-80",
                "split": "train",
                "index_file": "episodes.jsonl",
                "index_sha256": _sha256(index_path),
                "episode_count": 1,
                "controlled_candidate_vehicle_count": 2,
                "state_position_reference": "vehicle_geometric_body_center",
                "source_position_reference": "vehicle_front_center",
                "heading_contract": "exact_backward_source_displacement_then_current_lane_tangent_then_past_hold_v2",
                "mask_contract": "source_row_present_v1",
                "eligibility_reason_bits": ELIGIBILITY_REASON_BITS,
                "reference_point_conversion": {
                    "future_actor_rows_read": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    site_path = tmp_path / "DATASET_MANIFEST.json"
    site_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "contract_id": "dataset_manifest_v2",
                "dataset_id": "i80-runtime-canary",
                "site_id": "i-80",
                "scene": "i-80",
                "environment_id": "i-80",
                "source_artifacts": [],
                "sessions": [],
                "coordinate_frame": geometry_payload["coordinate_frame"],
                "split_contract": {"method": "session_locked"},
                "episode_store": {
                    "train": {
                        "relative_path": "episode_store/train/EPISODE_STORE.json",
                        "sha256": _sha256(store_path),
                    }
                },
                "road_geometry": {
                    "relative_path": "ROAD_GEOMETRY.json",
                    "sha256": _sha256(geometry_path),
                },
                "test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return site_path, store_path, geometry_path


def test_manifest_lane_mapping_uses_current_full_pose_and_successor_topology() -> None:
    network = build_road_network(RoadGeometryV3.from_source(_i80_geometry_payload()))
    upstream_6 = network.manifest_lane_index_by_id["main_up_6"]
    merge_6 = network.manifest_lane_index_by_id["merge_6"]
    merge_7 = network.manifest_lane_index_by_id["merge_7"]
    downstream_6 = network.manifest_lane_index_by_id["main_down_6"]
    ramp_7 = network.manifest_lane_index_by_id["ramp_approach_7"]

    assert target_lane_index_from_position_and_lane_id(
        network, "i-80", np.asarray([20.0, 50.0]), 6
    ) == upstream_6
    assert target_lane_index_from_position_and_lane_id(
        network, "i-80", np.asarray([20.0, 100.0]), 6
    ) == merge_6
    assert target_lane_index_from_position_and_lane_id(
        network, "i-80", np.asarray([20.0, 150.0]), 6
    ) == downstream_6
    assert resolve_target_lane_index_from_row(
        network, "i-80", np.asarray([26.25, 50.0, 10.0, 7.0]), vehicle_width_m=1.8
    )[0] == ramp_7
    assert resolve_target_lane_index_from_row(
        network, "i-80", np.asarray([22.0, 100.0, 10.0, 7.0]), vehicle_width_m=1.8
    )[0] == merge_7

    # Mapping is causal: changing a hypothetical future row cannot affect the
    # current-pose-only helper because no future row is accepted by its API.
    current = np.asarray([20.0, 150.0])
    assert target_lane_index_from_position_and_lane_id(network, "i-80", current, 6) == (
        downstream_6
    )
    assert network.next_lane(ramp_7, route=None, position=np.asarray([24.0, 80.0])) == (
        merge_7
    )
    assert network.next_lane(merge_7, route=None, position=np.asarray([20.0, 120.0])) == (
        downstream_6
    )


def test_manifest_lane_mapping_covers_non_ego_replay_across_merge() -> None:
    network = build_road_network(RoadGeometryV3.from_source(_i80_geometry_payload()))
    road = Road(network=network)
    trajectory = _ramp_episode_arrays()["states"][:, 0, :]
    arrays = _ramp_episode_arrays()
    replay = NGSIMVehicle.create(
        road,
        vehicle_ID=101,
        position=trajectory[0, :2],
        v_length=4.5,
        v_width=1.8,
        ngsim_traj=trajectory,
        scene="i-80",
        allow_idm=False,
        ngsim_heading_rad=arrays["heading_rad"][:, 0],
        ngsim_heading_valid_mask=arrays["heading_valid_mask"][:, 0],
        ngsim_heading_derivation=arrays["heading_derivation"][:, 0],
    )

    replay.sim_steps = 50
    replay._update_from_trajectory()
    assert replay.lane_index == network.manifest_lane_index_by_id["ramp_approach_7"]
    assert replay.heading == arrays["heading_rad"][50, 0]
    assert replay.current_heading_derivation == 1
    replay.sim_steps = 100
    replay._update_from_trajectory()
    assert replay.lane_index == network.manifest_lane_index_by_id["merge_7"]
    assert replay.heading == arrays["heading_rad"][100, 0]
    replay.sim_steps = 150
    replay._update_from_trajectory()
    assert replay.lane_index == network.manifest_lane_index_by_id["main_down_6"]
    assert replay.heading == arrays["heading_rad"][150, 0]


def test_persisted_current_pose_and_render_polygon_ignore_future_actor_rows() -> None:
    network = build_road_network(RoadGeometryV3.from_source(_i80_geometry_payload()))
    arrays = _ramp_episode_arrays()
    baseline_traj = arrays["states"][:, 0, :].copy()
    changed_future_traj = baseline_traj.copy()
    changed_future_traj[1:, :2] += np.asarray([500.0, -700.0])
    baseline_heading = arrays["heading_rad"][:, 0].copy()
    changed_future_heading = baseline_heading.copy()
    changed_future_heading[1:] = -0.75

    current_poses: list[tuple[np.ndarray, float, np.ndarray]] = []
    for trajectory, headings in (
        (baseline_traj, baseline_heading),
        (changed_future_traj, changed_future_heading),
    ):
        vehicle = NGSIMVehicle.create(
            Road(network=network),
            vehicle_ID=101,
            position=trajectory[0, :2],
            v_length=4.5,
            v_width=1.8,
            ngsim_traj=trajectory,
            scene="i-80",
            allow_idm=False,
            ngsim_heading_rad=headings,
            ngsim_heading_valid_mask=arrays["heading_valid_mask"][:, 0],
            ngsim_heading_derivation=arrays["heading_derivation"][:, 0],
        )
        vehicle._update_from_trajectory()
        polygon = road_entity_pose_polygon(
            vehicle.position,
            vehicle.heading,
            vehicle.LENGTH,
            vehicle.WIDTH,
        )
        current_poses.append((vehicle.position.copy(), float(vehicle.heading), polygon))

    np.testing.assert_array_equal(current_poses[0][0], current_poses[1][0])
    assert current_poses[0][1] == current_poses[1][1] == baseline_heading[0]
    np.testing.assert_array_equal(current_poses[0][2], current_poses[1][2])


def test_ngsim_env_teleports_i80_ramp_to_mainline_with_finite_329d_observation(
    tmp_path: Path,
) -> None:
    site_path, store_path, geometry_path = _write_runtime_contracts(tmp_path)
    env = NGSimEnv(
        {
            "scene": "i-80",
            "site_manifest": str(site_path),
            "episode_store": str(store_path),
            "road_geometry": str(geometry_path),
            "prebuilt_split": "train",
            "simulation_period": {"episode_name": "ep-ramp"},
            "ego_vehicle_ID": 101,
            "percentage_controlled_vehicles": 1,
            "max_surrounding": 1,
            "show_trajectories": False,
            "scene_dataset_collection_mode": True,
            "action_mode": "teleport",
            "simulation_frequency": 10,
            "policy_frequency": 10,
            "max_episode_steps": 200,
            "truncate_to_trajectory_length": True,
            "disable_scene_collection_spawn_safety": True,
            "observation": {
                "type": "LidarCameraObservations",
                "lidar": {
                    "cells": 64,
                    "maximum_range": 64.0,
                    "normalize": True,
                    "ego_centric": True,
                    "separate_road_edge_return": True,
                },
                "camera": {
                    "cells": 21,
                    "maximum_range": 64.0,
                    "field_of_view": float(np.pi / 2.0),
                    "normalize": True,
                },
                "ego_state_version": "route_free_v2",
            },
        },
        render_mode="rgb_array",
    )
    try:
        expected_arrays = _ramp_episode_arrays()
        expected_states = expected_arrays["states"][:, 0, :]
        expected_headings = expected_arrays["heading_rad"][:, 0]
        observation, _ = env.reset(seed=0)
        flat = np.concatenate([np.asarray(part).reshape(-1) for part in observation])
        assert flat.shape == (329,)
        assert np.all(np.isfinite(flat))
        np.testing.assert_array_equal(env.vehicle.position, [30.0, 0.0])
        assert env.vehicle.speed == 10.0
        assert env.vehicle.heading == expected_headings[0]
        assert env.vehicle.scene_collection_current_heading_derivation == 1
        assert env.vehicle.lane_index == env.road.network.manifest_lane_index_by_id[
            "ramp_approach_7"
        ]
        image = np.asarray(env.render())
        assert image.ndim == 3 and image.shape[2] == 3
        assert env.viewer.sim_surface.pix(1.0) > 0
        origin_pixels = env.viewer.sim_surface.pos2pix(0.0, 0.0)
        x_pixels = abs(
            env.viewer.sim_surface.pos2pix(1.0, 0.0)[0] - origin_pixels[0]
        )
        y_pixels = abs(
            env.viewer.sim_surface.pos2pix(0.0, 1.0)[1] - origin_pixels[1]
        )
        assert abs(x_pixels - y_pixels) <= 1  # integer pixel quantization only
        diagnostics = env.manifest_runtime_diagnostics()
        assert diagnostics["render_equal_aspect"] is True
        assert diagnostics["road_valid_state_count"] == 400
        assert diagnostics["training_eligible_state_count"] == 400
        assert (env.vehicle.LENGTH, env.vehicle.WIDTH) == (4.5, 1.8)

        for frame in range(1, 200):
            observation, _, _, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
            flat = np.concatenate([np.asarray(part).reshape(-1) for part in observation])
            assert flat.shape == (329,)
            assert np.all(np.isfinite(flat))
            expected = expected_states[frame]
            np.testing.assert_array_equal(env.vehicle.position, expected[:2])
            assert env.vehicle.speed == expected[2]
            assert env.vehicle.heading == expected_headings[frame]
            expected_front = expected_arrays["source_front_center_xy_m"][frame, 0]
            reconstructed_front = env.vehicle.position + 0.5 * env.vehicle.LENGTH * np.asarray(
                [np.cos(env.vehicle.heading), np.sin(env.vehicle.heading)]
            )
            np.testing.assert_allclose(reconstructed_front, expected_front, rtol=0.0, atol=1e-9)
            assert truncated is False

        assert env.vehicle.lane_index == env.road.network.manifest_lane_index_by_id[
            "main_down_6"
        ]
        observation, _, _, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        flat = np.concatenate([np.asarray(part).reshape(-1) for part in observation])
        assert flat.shape == (329,)
        assert np.all(np.isfinite(flat))
        assert truncated is True
    finally:
        env.close()
