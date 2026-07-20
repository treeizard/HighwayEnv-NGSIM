#!/usr/bin/env python3
"""Benchmark exact NGSIM simulator fast paths against the legacy oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = 1
MODES = ("legacy", "optimized")
OUTCOME_INFO_KEYS = (
    "controlled_vehicle_crashes",
    "controlled_vehicle_completed",
    "controlled_vehicle_on_road",
    "controlled_vehicle_offroad",
    "alive_controlled_vehicle_ids",
)


def _source_code_identity() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", "-C", str(repo), *arguments), text=True
        ).strip()

    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "repo": str(repo),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": not bool(git("status", "--porcelain", "--untracked-files=all")),
        "benchmark_file_sha256": digest,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _max_surrounding(value: str) -> str | int:
    if str(value).lower() == "all":
        return "all"
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("max surrounding must be non-negative or 'all'")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched seeded/action NGSIM rollouts with legacy and exact "
            "optimized road paths, report throughput, and verify parity."
        )
    )
    parser.add_argument("--scene", default="us-101", choices=("us-101", "i-80", "japanese"))
    parser.add_argument("--episode-root", default="data/highway_env/processed_20s")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument(
        "--episode-name",
        default="",
        help="Optional fixed episode; use a verified high-occupancy episode for 100-vehicle evidence.",
    )
    parser.add_argument("--vehicle-counts", nargs="+", type=_positive_int, default=[1, 10, 50, 100])
    parser.add_argument("--max-surrounding", type=_max_surrounding, default="all")
    parser.add_argument("--steps", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--action-seed", type=int, default=20260721)
    parser.add_argument("--action-pattern", choices=("zero", "uniform"), default="zero")
    parser.add_argument("--simulation-frequency", type=_positive_int, default=10)
    parser.add_argument("--policy-frequency", type=_positive_int, default=10)
    parser.add_argument("--lidar-cells", type=_positive_int, default=128)
    parser.add_argument("--maximum-range", type=_positive_float, default=64.0)
    parser.add_argument("--query-cell-size", type=_positive_float, default=25.0)
    parser.add_argument("--collision-cell-size", type=_positive_float, default=12.0)
    parser.add_argument("--collision-min-entities", type=int, default=32)
    parser.add_argument("--observation-atol", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--sensor-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compare the shared training sensor with the original per-agent "
            "MultiAgentObservation implementation at reset."
        ),
    )
    parser.add_argument(
        "--mode-order",
        choices=("alternate", "legacy-first", "optimized-first"),
        default="alternate",
        help="Balance filesystem-cache order effects across vehicle-count cases.",
    )
    parser.add_argument(
        "--strict-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return a non-zero status after writing the report if parity fails.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path. By default the dry, no-render benchmark only prints JSON.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.simulation_frequency < args.policy_frequency:
        raise ValueError("simulation frequency must be at least policy frequency")
    if args.simulation_frequency % args.policy_frequency != 0:
        raise ValueError("simulation frequency must be divisible by policy frequency")
    args.collision_min_entities = max(0, int(args.collision_min_entities))
    return args


def build_mode_config(
    args: argparse.Namespace,
    *,
    requested_vehicle_count: int,
    mode: str,
) -> dict[str, Any]:
    """Build the same headless environment config with only fast-path switches changed."""
    if mode not in MODES:
        raise ValueError(f"Unsupported benchmark mode: {mode!r}")
    from highway_env.imitation.expert_dataset import (
        build_env_config,
        default_observation_config,
    )

    config = build_env_config(
        scene=str(args.scene),
        action_mode="continuous",
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.split),
        percentage_controlled_vehicles=float(requested_vehicle_count),
        control_all_vehicles=False,
        max_surrounding=args.max_surrounding,
        observation_config=default_observation_config(
            cells=int(args.lidar_cells),
            maximum_range=float(args.maximum_range),
        ),
        simulation_frequency=int(args.simulation_frequency),
        policy_frequency=int(args.policy_frequency),
        max_episode_steps=int(args.steps),
        show_trajectories=False,
        seed=None,
        simulation_period=(
            {"episode_name": str(args.episode_name)}
            if str(args.episode_name).strip()
            else None
        ),
        scene_dataset_collection_mode=False,
        allow_idm=True,
        clip_controlled_vehicles_to_available=True,
    )
    config.update(
        {
            "expert_test_mode": False,
            "disable_controlled_vehicle_collisions": False,
            "terminate_when_all_controlled_crashed": True,
            "crash_controlled_vehicles_offroad": True,
            "offscreen_rendering": True,
            "road_query_mode": "legacy" if mode == "legacy" else "spatial",
            "road_query_cell_size": float(args.query_cell_size),
            "collision_check_mode": "legacy" if mode == "legacy" else "broadphase",
            "collision_broadphase_cell_size": float(args.collision_cell_size),
            "collision_broadphase_min_entities": int(args.collision_min_entities),
            "record_replay_diagnostics": mode == "legacy",
            "sensor_road_edge_mode": "per_vehicle" if mode == "legacy" else "batched",
            "reuse_pre_reset_spaces": mode != "legacy",
        }
    )
    return config


def clear_ngsim_process_caches() -> None:
    """Make each construction cold with respect to NGSimEnv's in-process caches."""
    from highway_env.envs.ngsim_env import NGSimEnv

    for name in (
        "_PREBUILT_CACHE",
        "_NETWORK_CACHE",
        "_PROCESSED_TRAJECTORY_CACHE",
        "_EXPERT_REFERENCE_CACHE",
    ):
        cache = getattr(NGSimEnv, name, None)
        if hasattr(cache, "clear"):
            cache.clear()


def _mode_order(args: argparse.Namespace, case_index: int) -> tuple[str, str]:
    if args.mode_order == "legacy-first":
        return MODES
    if args.mode_order == "optimized-first":
        return tuple(reversed(MODES))
    return MODES if case_index % 2 == 0 else tuple(reversed(MODES))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def compare_values(left: Any, right: Any, *, atol: float = 0.0) -> dict[str, Any]:
    """Compare nested simulator values and return a JSON-safe exactness summary."""
    summary: dict[str, Any] = {
        "equal": True,
        "values_compared": 0,
        "mismatches": 0,
        "structure_mismatch": False,
        "max_abs_diff": 0.0,
    }

    def mismatch(*, structure: bool = False) -> None:
        summary["equal"] = False
        summary["mismatches"] += 1
        summary["structure_mismatch"] = bool(summary["structure_mismatch"] or structure)

    def visit(first: Any, second: Any) -> None:
        if isinstance(first, dict) or isinstance(second, dict):
            if not isinstance(first, dict) or not isinstance(second, dict):
                mismatch(structure=True)
                return
            if set(first) != set(second):
                mismatch(structure=True)
                return
            for key in sorted(first, key=str):
                visit(first[key], second[key])
            return
        if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
            if type(first) is not type(second) or len(first) != len(second):
                mismatch(structure=True)
                return
            for item_first, item_second in zip(first, second, strict=True):
                visit(item_first, item_second)
            return
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
                mismatch(structure=True)
                return
            if first.shape != second.shape or first.dtype != second.dtype:
                mismatch(structure=True)
                return
            summary["values_compared"] += int(first.size)
            if np.issubdtype(first.dtype, np.number):
                equal = np.allclose(first, second, rtol=0.0, atol=atol, equal_nan=True)
                finite = np.isfinite(first) & np.isfinite(second)
                if np.any(finite):
                    max_diff = float(
                        np.max(
                            np.abs(
                                first[finite].astype(np.float64)
                                - second[finite].astype(np.float64)
                            )
                        )
                    )
                    summary["max_abs_diff"] = max(summary["max_abs_diff"], max_diff)
            else:
                equal = np.array_equal(first, second)
            if not bool(equal):
                mismatch()
            return

        summary["values_compared"] += 1
        if isinstance(first, (int, float, complex, np.number)) and isinstance(
            second, (int, float, complex, np.number)
        ):
            if not bool(np.isclose(first, second, rtol=0.0, atol=atol, equal_nan=True)):
                mismatch()
            elif np.isfinite(first) and np.isfinite(second):
                summary["max_abs_diff"] = max(
                    summary["max_abs_diff"], float(abs(first - second))
                )
        elif type(first) is not type(second) or first != second:
            mismatch(structure=type(first) is not type(second))

    visit(left, right)
    return summary


def _new_parity_accumulator() -> dict[str, Any]:
    return {
        "equal": True,
        "comparisons": 0,
        "mismatches": 0,
        "structure_mismatch": False,
        "max_abs_diff": 0.0,
        "first_mismatch_step": None,
    }


def _merge_parity(
    accumulator: dict[str, Any], comparison: dict[str, Any], *, step: int | str
) -> None:
    accumulator["comparisons"] += 1
    accumulator["mismatches"] += int(comparison["mismatches"])
    accumulator["max_abs_diff"] = max(
        float(accumulator["max_abs_diff"]), float(comparison["max_abs_diff"])
    )
    accumulator["structure_mismatch"] = bool(
        accumulator["structure_mismatch"] or comparison["structure_mismatch"]
    )
    if not comparison["equal"]:
        accumulator["equal"] = False
        if accumulator["first_mismatch_step"] is None:
            accumulator["first_mismatch_step"] = step


def summarize_step_times(
    step_seconds: Sequence[float], *, env_steps: int, agent_steps: int
) -> dict[str, float | int]:
    values = np.asarray(step_seconds, dtype=np.float64)
    total = float(values.sum()) if values.size else 0.0
    return {
        "env_steps": int(env_steps),
        "agent_steps": int(agent_steps),
        "total_seconds": total,
        "mean_seconds": float(values.mean()) if values.size else 0.0,
        "p50_seconds": float(np.percentile(values, 50)) if values.size else 0.0,
        "p95_seconds": float(np.percentile(values, 95)) if values.size else 0.0,
        "env_steps_per_second": float(env_steps / total) if total > 0.0 else 0.0,
        "agent_steps_per_second": float(agent_steps / total) if total > 0.0 else 0.0,
    }


def _controlled_state(env: Any) -> list[dict[str, Any]]:
    vehicles = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
    return [
        {
            "vehicle_id": int(getattr(vehicle, "vehicle_ID", index)),
            "position": np.asarray(vehicle.position).copy(),
            "speed": float(getattr(vehicle, "speed", 0.0)),
            "heading": float(getattr(vehicle, "heading", 0.0)),
            "crashed": bool(getattr(vehicle, "crashed", False)),
            "completed": bool(getattr(vehicle, "completed", False)),
        }
        for index, vehicle in enumerate(vehicles)
    ]


def _outcome(
    *,
    env: Any,
    reward: Any,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": {key: info.get(key) for key in OUTCOME_INFO_KEYS if key in info},
        "controlled_vehicle_state": _controlled_state(env),
    }


def _controlled_ids(env: Any) -> list[int]:
    vehicles = list(getattr(env.unwrapped, "controlled_vehicles", ()) or ())
    return [int(getattr(vehicle, "vehicle_ID", index)) for index, vehicle in enumerate(vehicles)]


def _action(
    *, pattern: str, rng: np.random.Generator, agent_count: int
) -> tuple[np.ndarray, ...]:
    if pattern == "uniform":
        values = rng.uniform(-1.0, 1.0, size=(agent_count, 2)).astype(np.float32)
    else:
        values = np.zeros((agent_count, 2), dtype=np.float32)
    return tuple(values[index].copy() for index in range(agent_count))


def _clone_action(action: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(value.copy() for value in action)


def _create_mode_env(
    args: argparse.Namespace,
    *,
    requested_vehicle_count: int,
    mode: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import gymnasium as gym

    from highway_env.imitation.expert_dataset import ENV_ID, register_ngsim_env

    clear_ngsim_process_caches()
    register_ngsim_env()
    config = build_mode_config(
        args,
        requested_vehicle_count=requested_vehicle_count,
        mode=mode,
    )
    make_started = time.perf_counter()
    env = gym.make(ENV_ID, render_mode=None, config=config)
    cold_make_seconds = time.perf_counter() - make_started
    reset_started = time.perf_counter()
    observation, reset_info = env.reset(seed=int(args.seed))
    reset_seconds = time.perf_counter() - reset_started
    road = getattr(env.unwrapped, "road", None)
    metadata = {
        "mode": mode,
        "cold_make_seconds": float(cold_make_seconds),
        "seeded_reset_seconds": float(reset_seconds),
        "episode": getattr(env.unwrapped, "episode_name", None),
        "initial_controlled_vehicle_ids": _controlled_ids(env),
        "initial_controlled_vehicles": len(_controlled_ids(env)),
        "initial_road_vehicles": len(getattr(road, "vehicles", ()) or ()),
        "reset_info_keys": sorted(str(key) for key in (reset_info or {})),
        "step_seconds": [],
        "agent_steps": 0,
        "steps_completed": 0,
        "final_outcome": None,
    }
    return env, observation, metadata


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def compare_reference_sensor(
    env: Any,
    shared_observation: Any,
    *,
    atol: float,
) -> dict[str, Any]:
    """Compare the shared sensor to the original per-agent implementation."""
    from highway_env.envs.common.observations.classic import MultiAgentObservation

    base = env.unwrapped
    configured = dict(base.config.get("observation") or {})
    inner = configured.get("observation_config")
    if configured.get("type") != "MultiAgentObservation" or not isinstance(inner, dict):
        raise RuntimeError("Sensor reference requires a configured MultiAgentObservation.")

    build_started = time.perf_counter()
    reference_observer = MultiAgentObservation(base, observation_config=inner)
    build_seconds = time.perf_counter() - build_started
    reference_started = time.perf_counter()
    reference_observation = reference_observer.observe()
    reference_seconds = time.perf_counter() - reference_started
    shared_started = time.perf_counter()
    repeated_shared = base.observation_type.observe()
    shared_seconds = time.perf_counter() - shared_started
    reset_comparison = compare_values(shared_observation, reference_observation, atol=atol)
    repeat_comparison = compare_values(repeated_shared, reference_observation, atol=atol)
    return {
        "implementation": "MultiAgentObservation(LidarCameraObservations per agent)",
        "passed": bool(reset_comparison["equal"] and repeat_comparison["equal"]),
        "reference_build_seconds": float(build_seconds),
        "reference_observe_seconds": float(reference_seconds),
        "shared_observe_seconds": float(shared_seconds),
        "shared_speedup_x": _safe_ratio(reference_seconds, shared_seconds),
        "reset_comparison": reset_comparison,
        "repeat_comparison": repeat_comparison,
    }


def run_case(
    args: argparse.Namespace,
    *,
    requested_vehicle_count: int,
    case_index: int,
) -> dict[str, Any]:
    envs: dict[str, Any] = {}
    observations: dict[str, Any] = {}
    modes: dict[str, dict[str, Any]] = {}
    observation_parity = _new_parity_accumulator()
    outcome_parity = _new_parity_accumulator()
    setup_parity = _new_parity_accumulator()
    action_rng = np.random.default_rng(int(args.action_seed) + int(case_index))

    try:
        construction_order = _mode_order(args, case_index)
        for mode in construction_order:
            env, observation, metadata = _create_mode_env(
                args,
                requested_vehicle_count=requested_vehicle_count,
                mode=mode,
            )
            envs[mode] = env
            observations[mode] = observation
            modes[mode] = metadata

        _merge_parity(
            setup_parity,
            compare_values(modes["legacy"]["episode"], modes["optimized"]["episode"]),
            step="reset",
        )
        _merge_parity(
            setup_parity,
            compare_values(
                modes["legacy"]["initial_controlled_vehicle_ids"],
                modes["optimized"]["initial_controlled_vehicle_ids"],
            ),
            step="reset",
        )
        _merge_parity(
            observation_parity,
            compare_values(
                observations["legacy"],
                observations["optimized"],
                atol=float(args.observation_atol),
            ),
            step="reset",
        )
        sensor_reference = (
            compare_reference_sensor(
                envs["optimized"],
                observations["optimized"],
                atol=float(args.observation_atol),
            )
            if bool(args.sensor_reference)
            else {"implementation": None, "passed": True, "skipped": True}
        )

        legacy_count = int(modes["legacy"]["initial_controlled_vehicles"])
        optimized_count = int(modes["optimized"]["initial_controlled_vehicles"])
        if legacy_count == optimized_count:
            for step in range(int(args.steps)):
                action = _action(
                    pattern=str(args.action_pattern),
                    rng=action_rng,
                    agent_count=legacy_count,
                )
                transitions: dict[str, tuple[Any, Any, bool, bool, dict[str, Any]]] = {}
                step_order = MODES if step % 2 == 0 else tuple(reversed(MODES))
                for mode in step_order:
                    started = time.perf_counter()
                    transition = envs[mode].step(_clone_action(action))
                    modes[mode]["step_seconds"].append(time.perf_counter() - started)
                    modes[mode]["steps_completed"] += 1
                    modes[mode]["agent_steps"] += legacy_count
                    transitions[mode] = transition

                (
                    legacy_obs,
                    legacy_reward,
                    legacy_terminated,
                    legacy_truncated,
                    legacy_info,
                ) = transitions["legacy"]
                (
                    optimized_obs,
                    optimized_reward,
                    optimized_terminated,
                    optimized_truncated,
                    optimized_info,
                ) = transitions["optimized"]
                _merge_parity(
                    observation_parity,
                    compare_values(
                        legacy_obs,
                        optimized_obs,
                        atol=float(args.observation_atol),
                    ),
                    step=step,
                )
                legacy_outcome = _outcome(
                    env=envs["legacy"],
                    reward=legacy_reward,
                    terminated=legacy_terminated,
                    truncated=legacy_truncated,
                    info=legacy_info,
                )
                optimized_outcome = _outcome(
                    env=envs["optimized"],
                    reward=optimized_reward,
                    terminated=optimized_terminated,
                    truncated=optimized_truncated,
                    info=optimized_info,
                )
                _merge_parity(
                    outcome_parity,
                    compare_values(legacy_outcome, optimized_outcome),
                    step=step,
                )
                modes["legacy"]["final_outcome"] = _json_safe(legacy_outcome)
                modes["optimized"]["final_outcome"] = _json_safe(optimized_outcome)
                if (
                    legacy_terminated
                    or legacy_truncated
                    or optimized_terminated
                    or optimized_truncated
                ):
                    break
        else:
            _merge_parity(
                setup_parity,
                compare_values(legacy_count, optimized_count),
                step="reset",
            )

        for mode in MODES:
            modes[mode]["timing"] = summarize_step_times(
                modes[mode].pop("step_seconds"),
                env_steps=int(modes[mode]["steps_completed"]),
                agent_steps=int(modes[mode]["agent_steps"]),
            )

        legacy_timing = modes["legacy"]["timing"]
        optimized_timing = modes["optimized"]["timing"]
        parity_passed = bool(
            setup_parity["equal"]
            and observation_parity["equal"]
            and outcome_parity["equal"]
            and sensor_reference["passed"]
        )
        return {
            "requested_vehicle_count": int(requested_vehicle_count),
            "construction_order": list(construction_order),
            "modes": modes,
            "parity": {
                "passed": parity_passed,
                "setup": setup_parity,
                "observations": observation_parity,
                "outcomes": outcome_parity,
                "sensor_reference": sensor_reference,
            },
            "optimized_speedup": {
                "step_p50_x": _safe_ratio(
                    float(legacy_timing["p50_seconds"]),
                    float(optimized_timing["p50_seconds"]),
                ),
                "step_p95_x": _safe_ratio(
                    float(legacy_timing["p95_seconds"]),
                    float(optimized_timing["p95_seconds"]),
                ),
                "env_steps_per_second_x": _safe_ratio(
                    float(optimized_timing["env_steps_per_second"]),
                    float(legacy_timing["env_steps_per_second"]),
                ),
                "agent_steps_per_second_x": _safe_ratio(
                    float(optimized_timing["agent_steps_per_second"]),
                    float(legacy_timing["agent_steps_per_second"]),
                ),
            },
        }
    finally:
        for env in envs.values():
            env.close()


def build_report(args: argparse.Namespace, cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    parity_passed = bool(cases) and all(
        bool(case.get("parity", {}).get("passed", False)) for case in cases
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "ngsim_exact_fastpath_equivalence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code": _source_code_identity(),
        "config": {
            "scene": str(args.scene),
            "episode_root": str(args.episode_root),
            "split": str(args.split),
            "episode_name": str(args.episode_name),
            "vehicle_counts": [int(value) for value in args.vehicle_counts],
            "max_surrounding": args.max_surrounding,
            "steps": int(args.steps),
            "seed": int(args.seed),
            "action_seed": int(args.action_seed),
            "action_pattern": str(args.action_pattern),
            "render_mode": None,
            "show_trajectories": False,
            "observation_atol": float(args.observation_atol),
            "sensor_reference": bool(args.sensor_reference),
            "cold_cache_scope": "NGSimEnv in-process caches",
        },
        "parity_passed": parity_passed,
        "cases": list(cases),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = [
        run_case(
            args,
            requested_vehicle_count=int(vehicle_count),
            case_index=case_index,
        )
        for case_index, vehicle_count in enumerate(args.vehicle_counts)
    ]
    return build_report(args, cases)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    text = json.dumps(_json_safe(report), indent=2, sort_keys=True)
    print(text)
    if str(args.output).strip():
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if bool(args.strict_parity) and not bool(report["parity_passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
