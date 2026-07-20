from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts_env_test import benchmark_ngsim_fastpath_equivalence as benchmark
from scripts_env_test import audit_ngsim_fastpath_evidence as evidence_audit
from scripts_env_test.benchmark_ngsim_fastpath_equivalence import (
    _mode_order,
    build_mode_config,
    build_report,
    compare_values,
    parse_args,
    summarize_step_times,
)


def test_parser_defaults_are_headless_dry_and_cover_scale_points() -> None:
    args = parse_args([])

    assert args.episode_root == "data/highway_env/processed_20s"
    assert args.vehicle_counts == [1, 10, 50, 100]
    assert args.action_pattern == "zero"
    assert args.output == ""
    assert args.strict_parity is True
    assert args.sensor_reference is True
    assert _mode_order(args, 0) == ("legacy", "optimized")
    assert _mode_order(args, 1) == ("optimized", "legacy")


def test_native_domain_reports_aggregate_into_source_bound_release_evidence(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        evidence_audit,
        "_git",
        lambda _repo, *args: "" if args[0] == "status" else ("revision" if args[-1] == "HEAD" else "tree"),
    )
    reports = {}
    for scene in ("us-101", "japanese"):
        episode_root = tmp_path / scene
        episode_root.mkdir()
        cases = []
        for vehicles in (50, 100):
            cases.append(
                {
                    "requested_vehicle_count": vehicles,
                    "parity": {
                        "passed": True,
                        "setup": {"equal": True},
                        "observations": {"equal": True, "max_abs_diff": 0.0},
                        "outcomes": {"equal": True},
                        "sensor_reference": {
                            "passed": True,
                            "shared_speedup_x": 1.5,
                            "reset_comparison": {"max_abs_diff": 0.0},
                            "repeat_comparison": {"max_abs_diff": 0.0},
                        },
                    },
                    "modes": {
                        "legacy": {"initial_controlled_vehicles": vehicles},
                        "optimized": {"initial_controlled_vehicles": vehicles},
                    },
                    "optimized_speedup": {"agent_steps_per_second_x": 1.25},
                }
            )
        report = {
            "schema_version": 1,
            "benchmark": "ngsim_exact_fastpath_equivalence",
            "source_code": {
                "repo": str(repo), "revision": "revision", "tree": "tree", "clean": True
            },
            "config": {
                "scene": scene, "episode_root": str(episode_root), "split": "train",
                "observation_atol": 0.0,
                "sensor_reference": True,
            },
            "parity_passed": True,
            "cases": cases,
        }
        path = tmp_path / f"{scene}.json"
        path.write_text(json.dumps(report))
        reports[scene] = path
    evidence = evidence_audit.aggregate(
        reports=reports, source_repo=repo, minimum_100_vehicle_speedup=1.1
    )
    assert evidence["status"] == "passed"
    assert evidence["source_code"]["revision"] == "revision"
    assert {(row["scene"], row["controlled_vehicles"]) for row in evidence["cases"]} == {
        ("us-101", 50), ("us-101", 100), ("japanese", 50), ("japanese", 100)
    }


def test_parser_accepts_custom_counts_and_rejects_invalid_frequency() -> None:
    args = parse_args(
        [
            "--scene",
            "japanese",
            "--vehicle-counts",
            "3",
            "17",
            "--max-surrounding",
            "25",
            "--steps",
            "7",
            "--action-pattern",
            "uniform",
            "--no-strict-parity",
            "--mode-order",
            "optimized-first",
        ]
    )
    assert args.vehicle_counts == [3, 17]
    assert args.max_surrounding == 25
    assert args.steps == 7
    assert args.strict_parity is False
    assert _mode_order(args, 4) == ("optimized", "legacy")

    with pytest.raises(ValueError, match="divisible"):
        parse_args(["--simulation-frequency", "10", "--policy-frequency", "3"])


def test_mode_configs_only_change_exact_fast_path_switches() -> None:
    args = parse_args(["--vehicle-counts", "12", "--steps", "5"])
    legacy = build_mode_config(args, requested_vehicle_count=12, mode="legacy")
    optimized = build_mode_config(args, requested_vehicle_count=12, mode="optimized")

    assert legacy["percentage_controlled_vehicles"] == 12.0
    assert legacy["show_trajectories"] is False
    assert legacy["offscreen_rendering"] is True
    assert legacy["road_query_mode"] == "legacy"
    assert legacy["collision_check_mode"] == "legacy"
    assert legacy["record_replay_diagnostics"] is True
    assert legacy["sensor_road_edge_mode"] == "per_vehicle"
    assert optimized["road_query_mode"] == "spatial"
    assert optimized["collision_check_mode"] == "broadphase"
    assert optimized["record_replay_diagnostics"] is False
    assert optimized["sensor_road_edge_mode"] == "batched"

    ignored = {
        "road_query_mode",
        "road_query_cell_size",
        "collision_check_mode",
        "collision_broadphase_cell_size",
        "collision_broadphase_min_entities",
        "record_replay_diagnostics",
        "sensor_road_edge_mode",
    }
    assert {key: value for key, value in legacy.items() if key not in ignored} == {
        key: value for key, value in optimized.items() if key not in ignored
    }


def test_nested_comparison_reports_exactness_and_numeric_difference() -> None:
    left = {
        "observations": (
            np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
            np.asarray([1, 2], dtype=np.int32),
        ),
        "done": False,
    }
    equal = compare_values(left, left, atol=0.0)
    assert equal["equal"] is True
    assert equal["mismatches"] == 0

    right = {
        "observations": (
            np.asarray([[1.0, np.nan], [3.0, 4.01]], dtype=np.float32),
            np.asarray([1, 2], dtype=np.int32),
        ),
        "done": False,
    }
    mismatch = compare_values(left, right, atol=0.0)
    assert mismatch["equal"] is False
    assert mismatch["mismatches"] == 1
    assert mismatch["max_abs_diff"] == pytest.approx(0.0100002289)
    assert compare_values(left, right, atol=0.011)["equal"] is True


def test_timing_and_report_results_are_json_serializable() -> None:
    timing = summarize_step_times(
        [0.1, 0.2, 0.3],
        env_steps=3,
        agent_steps=30,
    )
    assert timing["p50_seconds"] == pytest.approx(0.2)
    assert timing["p95_seconds"] == pytest.approx(0.29)
    assert timing["env_steps_per_second"] == pytest.approx(5.0)
    assert timing["agent_steps_per_second"] == pytest.approx(50.0)

    args = parse_args(["--vehicle-counts", "10", "--steps", "3"])
    case = {
        "requested_vehicle_count": 10,
        "parity": {"passed": True},
        "modes": {
            "legacy": {"timing": timing},
            "optimized": {"timing": timing},
        },
    }
    report = build_report(args, [case])
    encoded = json.dumps(report, sort_keys=True)

    assert report["schema_version"] == 1
    assert report["parity_passed"] is True
    assert report["config"]["render_mode"] is None
    assert report["config"]["show_trajectories"] is False
    assert "ngsim_exact_fastpath_equivalence" in encoded


def test_case_result_pipeline_uses_matched_actions_without_dataset(monkeypatch) -> None:
    class FakeVehicle:
        def __init__(self, vehicle_id: int):
            self.vehicle_ID = vehicle_id
            self.position = np.asarray([float(vehicle_id), 0.0], dtype=np.float64)
            self.speed = 0.0
            self.heading = 0.0
            self.crashed = False
            self.completed = False

    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.controlled_vehicles = [FakeVehicle(10), FakeVehicle(20)]
            self.step_index = 0
            self.closed = False

        def step(self, action):
            assert len(action) == len(self.controlled_vehicles)
            self.step_index += 1
            observation = np.full((2, 3), self.step_index, dtype=np.float32)
            truncated = self.step_index >= 2
            info = {
                "controlled_vehicle_crashes": [False, False],
                "controlled_vehicle_completed": [False, False],
                "controlled_vehicle_on_road": [True, True],
                "controlled_vehicle_offroad": [False, False],
                "alive_controlled_vehicle_ids": [10, 20],
            }
            return observation, 0.0, False, truncated, info

        def close(self):
            self.closed = True

    created: list[FakeEnv] = []

    def fake_create_mode_env(args, *, requested_vehicle_count, mode):
        assert requested_vehicle_count == 2
        env = FakeEnv()
        created.append(env)
        metadata = {
            "mode": mode,
            "cold_make_seconds": 0.1,
            "seeded_reset_seconds": 0.01,
            "episode": "fixture_episode",
            "initial_controlled_vehicle_ids": [10, 20],
            "initial_controlled_vehicles": 2,
            "initial_road_vehicles": 4,
            "reset_info_keys": [],
            "step_seconds": [],
            "agent_steps": 0,
            "steps_completed": 0,
            "final_outcome": None,
        }
        return env, np.zeros((2, 3), dtype=np.float32), metadata

    monkeypatch.setattr(benchmark, "_create_mode_env", fake_create_mode_env)
    args = parse_args(
        ["--vehicle-counts", "2", "--steps", "3", "--no-sensor-reference"]
    )

    case = benchmark.run_case(
        args,
        requested_vehicle_count=2,
        case_index=0,
    )

    assert case["parity"]["passed"] is True
    assert case["parity"]["observations"]["comparisons"] == 3
    assert case["parity"]["outcomes"]["comparisons"] == 2
    assert case["modes"]["legacy"]["timing"]["env_steps"] == 2
    assert case["modes"]["legacy"]["timing"]["agent_steps"] == 4
    assert all(env.closed for env in created)
