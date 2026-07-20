#!/usr/bin/env python3
"""Aggregate native US/Japanese fast-path reports into a production release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *arguments), text=True).strip()


def aggregate(
    *,
    reports: dict[str, Path],
    source_repo: Path,
    minimum_100_vehicle_speedup: float,
    minimum_large_case_vehicles: int,
) -> dict[str, Any]:
    source_repo = source_repo.resolve()
    current_source = {
        "repo": str(source_repo),
        "revision": _git(source_repo, "rev-parse", "HEAD"),
        "tree": _git(source_repo, "rev-parse", "HEAD^{tree}"),
    }
    if _git(source_repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Fast-path evidence must be audited in a clean source checkout.")
    normalized_cases: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    episode_data: dict[str, dict[str, str]] = {}
    expected_scenes = {"us-101", "japanese"}
    if set(reports) != expected_scenes:
        raise RuntimeError(f"Expected reports for {sorted(expected_scenes)}")
    for scene in ("us-101", "japanese"):
        path = reports[scene].resolve()
        report = _read(path)
        if int(report.get("schema_version", -1)) != 1 or report.get("benchmark") != "ngsim_exact_fastpath_equivalence":
            raise RuntimeError(f"Not a native exact-fastpath report: {path}")
        config = dict(report.get("config") or {})
        if (
            config.get("scene") != scene
            or float(config.get("observation_atol", -1.0)) != 0.0
            or config.get("sensor_reference") is not True
            or not str(config.get("episode_name") or "").strip()
        ):
            raise RuntimeError(f"Report scene/tolerance mismatch: {path}")
        if report.get("parity_passed") is not True:
            raise RuntimeError(f"Report did not pass parity: {path}")
        report_source = dict(report.get("source_code") or {})
        if report_source.get("clean") is not True:
            raise RuntimeError(f"Benchmark report came from a dirty checkout: {path}")
        for key in ("repo", "revision", "tree"):
            if report_source.get(key) != current_source[key]:
                raise RuntimeError(f"Benchmark source {key} mismatch: {path}")
        cases = list(report.get("cases") or ())
        for vehicle_count in (50, 100):
            matches = [
                case for case in cases
                if int(case.get("requested_vehicle_count", -1)) == vehicle_count
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Report needs exactly one {scene}/{vehicle_count} case.")
            case = matches[0]
            parity = dict(case.get("parity") or {})
            observations = dict(parity.get("observations") or {})
            sensor_reference = dict(parity.get("sensor_reference") or {})
            if (
                parity.get("passed") is not True
                or dict(parity.get("setup") or {}).get("equal") is not True
                or observations.get("equal") is not True
                or dict(parity.get("outcomes") or {}).get("equal") is not True
                or float(observations.get("max_abs_diff", -1.0)) != 0.0
                or sensor_reference.get("passed") is not True
                or float(
                    dict(sensor_reference.get("reset_comparison") or {}).get(
                        "max_abs_diff", -1.0
                    )
                )
                != 0.0
                or float(
                    dict(sensor_reference.get("repeat_comparison") or {}).get(
                        "max_abs_diff", -1.0
                    )
                )
                != 0.0
            ):
                raise RuntimeError(f"Strict parity failed for {scene}/{vehicle_count}.")
            modes = dict(case.get("modes") or {})
            actual_counts = []
            for mode in ("legacy", "optimized"):
                actual_counts.append(
                    int(dict(modes.get(mode) or {}).get("initial_controlled_vehicles", -1))
                )
            if actual_counts[0] != actual_counts[1]:
                raise RuntimeError(
                    f"{scene}/{vehicle_count} instantiated different controlled counts: {actual_counts}"
                )
            if vehicle_count == 100 and actual_counts[0] < int(minimum_large_case_vehicles):
                raise RuntimeError(
                    f"{scene}/100 safely instantiated only {actual_counts[0]} controlled vehicles; "
                    f"minimum is {minimum_large_case_vehicles}."
                )
            speedup = float(
                dict(case.get("optimized_speedup") or {}).get(
                    "agent_steps_per_second_x", 0.0
                )
            )
            if vehicle_count == 100 and speedup < float(minimum_100_vehicle_speedup):
                raise RuntimeError(
                    f"100-vehicle speedup failed for {scene}: {speedup} < {minimum_100_vehicle_speedup}"
                )
            normalized_cases.append(
                {
                    "scene": scene,
                    "controlled_vehicles": vehicle_count,
                    "actual_controlled_vehicles": actual_counts[0],
                    "strict_parity": True,
                    "observation_max_abs_error": 0.0,
                    "sensor_reference_speedup_ratio": float(
                        sensor_reference.get("shared_speedup_x", 0.0)
                    ),
                    "speedup_ratio": speedup,
                }
            )
        episode_data[scene] = {
            "episode_root": str(Path(str(config["episode_root"])).resolve()),
            "split": str(config["split"]),
        }
        inputs.append({"scene": scene, "path": str(path), "sha256": _sha256(path)})
    return {
        "schema_version": 1,
        "benchmark_kind": "ngsim_live_data_parity",
        "status": "passed",
        "live_data": True,
        "strict_parity": True,
        "minimum_100_vehicle_speedup": float(minimum_100_vehicle_speedup),
        "minimum_large_case_vehicles": int(minimum_large_case_vehicles),
        "source_code": current_source,
        "episode_data": episode_data,
        "input_reports": inputs,
        "cases": normalized_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--us-report", required=True, type=Path)
    parser.add_argument("--japanese-report", required=True, type=Path)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--minimum-100-vehicle-speedup", type=float, required=True)
    parser.add_argument("--minimum-large-case-vehicles", type=int, default=50)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate(
        reports={"us-101": args.us_report, "japanese": args.japanese_report},
        source_repo=args.source_repo,
        minimum_100_vehicle_speedup=args.minimum_100_vehicle_speedup,
        minimum_large_case_vehicles=args.minimum_large_case_vehicles,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(output), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
