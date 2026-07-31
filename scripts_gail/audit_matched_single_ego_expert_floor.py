#!/usr/bin/env python3
"""Measure the pairable single-ego expert feasibility floor on validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail.training.evaluation import (
    _evaluation_scenarios,
    clear_evaluation_worker_caches,
    evaluate_expert_replay_matched_single_vehicle_floor,
)
from scripts_gail.ps_gail.validation import (
    PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
    paper_driver_model_validation_overrides,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluation_config(
    *,
    scene: str,
    episode_root: Path,
    split: str,
    seed: int,
) -> PSGAILConfig:
    return replace(
        PSGAILConfig(),
        scene=str(scene),
        action_mode="continuous",
        episode_root=str(episode_root),
        prebuilt_split=str(split),
        seed=int(seed),
        trajectory_frame="relative",
        max_surrounding="all",
        control_all_vehicles=False,
        percentage_controlled_vehicles=1.0,
        allow_idm=True,
        enable_collision=True,
        cells=128,
        maximum_range=64.0,
        simulation_frequency=10,
        policy_frequency=10,
        max_episode_steps=200,
        road_query_mode="spatial",
        collision_check_mode="broadphase",
        record_replay_diagnostics=True,
        sensor_road_edge_mode="batched",
        reuse_pre_reset_spaces=True,
        evaluation_num_workers=1,
        evaluation_worker_threads=2,
        **paper_driver_model_validation_overrides(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--domain",
        choices=["us", "japanese", "all"],
        default="all",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_root = args.episode_root.resolve()
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Episode root not found: {episode_root}")
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive.")
    if int(args.num_shards) <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards).")

    domains: dict[str, str] = {
        "us": "us-101",
        "japanese": "japanese",
    }
    if str(args.domain) != "all":
        domains = {str(args.domain): domains[str(args.domain)]}
    results: dict[str, object] = {}
    for domain, scene in domains.items():
        clear_evaluation_worker_caches()
        cfg = evaluation_config(
            scene=scene,
            episode_root=episode_root,
            split=str(args.split),
            seed=int(args.seed),
        )
        all_scenarios = _evaluation_scenarios(
            cfg,
            split=str(args.split),
            episodes=int(args.episodes),
        )
        selected_scenarios = [
            (scenario_index, episode_name, vehicle_id)
            for scenario_index, (episode_name, vehicle_id) in enumerate(
                all_scenarios
            )
            if scenario_index % int(args.num_shards)
            == int(args.shard_index)
        ]
        results[domain] = (
            evaluate_expert_replay_matched_single_vehicle_floor(
                cfg,
                split=str(args.split),
                episodes=len(selected_scenarios),
                prefix="expert_floor",
                scenarios=selected_scenarios,
            )
        )
    clear_evaluation_worker_caches()

    source_paths = {
        "audit_script": Path(__file__).resolve(),
        "evaluation": ROOT
        / "scripts_gail/ps_gail/training/evaluation.py",
        "environment": ROOT / "highway_env/envs/ngsim_env.py",
        "collision_object": ROOT
        / "highway_env/vehicle/objects.py",
        "road_generator": ROOT
        / "highway_env/ngsim_utils/road/gen_road.py",
    }
    payload = {
        "schema_version": 2,
        "audit": "matched_single_ego_expert_floor_v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_framework": PAPER_DRIVER_MODEL_VALIDATION_FRAMEWORK,
        "scope": (
            "Route-conditioned synthetic tracker feasibility floor on the "
            "same single-ego evaluator used by policies; not a human-driver "
            "or surveyed-road baseline."
        ),
        "episode_root": str(episode_root),
        "split": str(args.split),
        "test_split_opened": False,
        "requested_scenarios_per_domain": int(args.episodes),
        "domain_selection": str(args.domain),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "seed": int(args.seed),
        "runtime_contract": {
            "action_mode": "continuous",
            "action_order": ["acceleration_norm", "steering_norm"],
            "acceleration_limit_mps2": 5.0,
            "vehicle_mode": "single_requested_ego",
            "collision_physics_enabled": True,
            "collision_termination_enabled": False,
            "road_query_mode": "spatial",
            "collision_check_mode": "broadphase",
            "horizons_seconds": [1, 5, 10, 20],
            "pairing": (
                "same scenario list and reset seed offset as single-ego "
                "policy evaluation"
            ),
        },
        "source_sha256": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        },
        "domains": results,
    }
    output_path = args.out.resolve()
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
