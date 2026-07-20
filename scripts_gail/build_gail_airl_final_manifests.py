#!/usr/bin/env python3
"""Build locked two-stage manifests for the final 12-cell GAIL and AIRL matrices."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from typing import Any

import torch

from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.promote_gail_airl_final_recipe import promote


DOMAINS = ("us", "japanese")
DEPTHS = (2, 3)
SEEDS = (0, 1, 2)
CANARY_INDICES = (0, 3, 6, 9)
CONTROLLED_VEHICLE_SCHEDULE = (
    "1:120:10:10;121:240:20:20;241:360:30:30;361:480:40:40;"
    "481:600:50:50;601:650:60:60;651:700:70:70;701:750:85:85;751:800:100:100"
)
ROLLOUT_TARGET_SCHEDULE = (
    "1:500:10000:10000;501:600:10000:20000;"
    "601:700:20000:30000;701:800:30000:40000"
)
SOURCE_LOCK_FILES = (
    "scripts_gail/train_simple_ps_gail.py",
    "scripts_gail/train_simple_airl.py",
    "scripts_gail/ps_gail/config.py",
    "scripts_gail/ps_gail/checkpoints.py",
    "scripts_gail/ps_gail/health.py",
    "scripts_gail/ps_gail/envs.py",
    "scripts_gail/ps_gail/training/rollouts.py",
    "scripts_gail/ps_gail/training/evaluation.py",
    "scripts_gail/run_gail_airl_study_trial.py",
    "highway_env/envs/ngsim_env.py",
    "highway_env/envs/common/observations/base.py",
    "highway_env/envs/common/observations/camera.py",
    "highway_env/envs/common/observations/factory.py",
    "highway_env/envs/common/observations/lidar.py",
    "highway_env/ngsim_utils/vehicles/replay.py",
    "highway_env/road/road.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gail-recipe", required=True, type=Path)
    parser.add_argument("--airl-recipe", required=True, type=Path)
    parser.add_argument("--bc-audit", required=True, type=Path)
    parser.add_argument("--us-expert", required=True, type=Path)
    parser.add_argument("--japanese-expert", required=True, type=Path)
    parser.add_argument("--us-episode-root", required=True, type=Path)
    parser.add_argument("--japanese-episode-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--simulator-profile", choices=("legacy", "optimized"), default="legacy")
    parser.add_argument("--canary-cpus-per-task", type=int, default=16)
    parser.add_argument("--canary-memory-per-task", default="64G")
    parser.add_argument("--canary-rollout-workers", type=int, default=8)
    parser.add_argument("--canary-evaluation-workers", type=int, default=8)
    parser.add_argument("--production-cpus-per-task", required=True, type=int)
    parser.add_argument("--production-memory-per-task", required=True)
    parser.add_argument("--production-rollout-workers", required=True, type=int)
    parser.add_argument("--production-evaluation-workers", required=True, type=int)
    parser.add_argument("--worker-threads", type=int, default=2)
    parser.add_argument("--evaluation-cache-limit", type=int, default=4)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sidecar(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Warm start is missing SHA-256 sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if expected != actual:
        raise RuntimeError(f"Warm-start SHA-256 mismatch: {expected} != {actual} ({path})")
    return actual


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def source_code_lock(repo: Path, requested_revision: str = "") -> dict[str, Any]:
    repo = repo.resolve()
    head = _git(repo, "rev-parse", "HEAD")
    if requested_revision and requested_revision != head:
        raise RuntimeError(f"Requested source revision is not checked out: {requested_revision} != {head}")
    missing = [relative for relative in SOURCE_LOCK_FILES if not (repo / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Source-lock files are missing: {missing}")
    return {
        "repo": str(repo),
        "revision": head,
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "require_clean_worktree": True,
        "files_sha256": {
            relative: _sha256(repo / relative)
            for relative in SOURCE_LOCK_FILES
        },
    }


def _load_recipe(path: Path, method: str) -> dict[str, Any]:
    payload = _read_json(path.resolve())
    if "trials" in payload or int(payload.get("trial_count", 0)) > 0:
        raise ValueError(f"Final campaign generator rejects screening/study manifests: {path}")
    if (
        payload.get("recipe_kind") != "gail_airl_confirmed_final"
        or payload.get("status") != "passed"
        or payload.get("terminal") is not True
        or payload.get("safety_eligible") is not True
        or payload.get("method") != method
    ):
        raise RuntimeError(f"Final recipe is not a terminal safety-eligible {method} promotion: {path}")
    report_path = Path(str(dict(payload.get("confirmation_report") or {}).get("path", "")))
    manifest_path = Path(str(dict(payload.get("confirmation_manifest") or {}).get("path", "")))
    expected = promote(method, report_path, manifest_path)
    if payload != expected:
        raise RuntimeError(f"Final recipe differs from deterministic confirmation promotion: {path}")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise TypeError(f"Recipe arguments must be a JSON object: {path}")
    ignored_metadata = {
        "schema_version", "recipe_kind", "method", "status", "notes", "provenance"
    }
    result = {key: value for key, value in arguments.items() if key not in ignored_metadata}
    allowed = {field.name for field in fields(PSGAILConfig)}
    if method == "airl":
        allowed.update(("reward_batch_size", "airl_log_prob_batch_size"))
    unknown = sorted(set(result).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {method} recipe arguments: {unknown}")
    variant = str(result.get("algorithm_variant") or f"{method}_bce").strip().lower()
    if not variant.startswith(f"{method}_"):
        raise ValueError(f"{method} recipe has incompatible algorithm_variant={variant!r}")
    result["algorithm_variant"] = variant
    return result


def _load_bc_warm_starts(path: Path) -> tuple[dict[tuple[str, int], Path], dict[str, Any]]:
    audit_path = path.resolve()
    payload = _read_json(audit_path)
    if payload.get("terminal") is not True or str(payload.get("status", "")).lower() not in {
        "complete", "passed"
    }:
        raise RuntimeError("BC audit must record terminal=true and status=complete|passed.")
    rows = payload.get("warm_starts")
    if not isinstance(rows, list):
        raise RuntimeError("BC audit must contain a warm_starts list.")
    selected: dict[tuple[str, int], Path] = {}
    provenance: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not bool(row.get("selected", True)):
            continue
        domain = str(row.get("domain", "")).lower()
        depth = int(row.get("transformer_layers", -1))
        key = (domain, depth)
        if key not in {(d, n) for d in DOMAINS for n in DEPTHS}:
            raise RuntimeError(f"Unexpected BC audit cell: {key}")
        if key in selected:
            raise RuntimeError(f"BC audit selects multiple warm starts for {key}")
        if row.get("warm_start_passed") is not True:
            raise RuntimeError(f"BC warm start did not pass its stabilization gate: {key}")
        checkpoint = Path(str(row.get("checkpoint", ""))).expanduser().resolve()
        digest = _verify_sidecar(checkpoint)
        recorded_digest = str(row.get("checkpoint_sha256") or digest)
        if recorded_digest != digest:
            raise RuntimeError(f"BC audit hash mismatch for {checkpoint}")
        try:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu")
        if checkpoint_payload.get("checkpoint_kind") != "behaviour_cloning_warm_start":
            raise RuntimeError(f"Not a BC warm-start checkpoint: {checkpoint}")
        architecture = dict(checkpoint_payload.get("policy_architecture") or {})
        config = dict(checkpoint_payload.get("config") or {})
        if architecture.get("policy_model", config.get("policy_model")) != "recurrent_transformer":
            raise RuntimeError(f"Warm start is not recurrent_transformer: {checkpoint}")
        if str(architecture.get("action_mode", config.get("action_mode"))) != "continuous":
            raise RuntimeError(f"Warm start is not continuous-action: {checkpoint}")
        actual_depth = int(architecture.get("transformer_layers", config.get("transformer_layers", -1)))
        if actual_depth != depth:
            raise RuntimeError(f"Warm-start depth mismatch: {actual_depth} != {depth} ({checkpoint})")
        expected_scene = "us-101" if domain == "us" else "japanese"
        configured_scene = str(config.get("scene") or expected_scene)
        if configured_scene != expected_scene:
            raise RuntimeError(
                f"Warm-start domain mismatch: scene={configured_scene!r}, expected={expected_scene!r}"
            )
        selected[key] = checkpoint
        provenance.append(
            {
                "domain": domain,
                "transformer_layers": depth,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            }
        )
    expected = {(domain, depth) for domain in DOMAINS for depth in DEPTHS}
    if set(selected) != expected:
        raise RuntimeError(f"BC audit warm-start coverage mismatch: {sorted(selected)} != {sorted(expected)}")
    return selected, {
        "path": str(audit_path),
        "sha256": _sha256(audit_path),
        "status": payload["status"],
        "terminal": True,
        "warm_starts": provenance,
    }


def canonical_cells() -> list[tuple[int, str, int, int]]:
    cells: list[tuple[int, str, int, int]] = []
    for domain in DOMAINS:
        for depth in DEPTHS:
            for seed in SEEDS:
                cells.append((len(cells), domain, depth, seed))
    assert [item[0] for item in cells] == list(range(12))
    assert [cells[index][1:] for index in CANARY_INDICES] == [
        ("us", 2, 0), ("us", 3, 0), ("japanese", 2, 0), ("japanese", 3, 0)
    ]
    return cells


def _validate_study_id(value: str) -> str:
    path = PurePosixPath(str(value).strip())
    if not str(path) or path.is_absolute() or ".." in path.parts:
        raise ValueError("study-id must be a non-empty relative path without '..'.")
    return str(path)


def _resource_geometry(args: argparse.Namespace, profile: str) -> dict[str, Any]:
    if profile == "canary":
        cpus = int(args.canary_cpus_per_task)
        memory = str(args.canary_memory_per_task)
        rollout_workers = int(args.canary_rollout_workers)
        evaluation_workers = int(args.canary_evaluation_workers)
    else:
        cpus = int(args.production_cpus_per_task)
        memory = str(args.production_memory_per_task)
        rollout_workers = int(args.production_rollout_workers)
        evaluation_workers = int(args.production_evaluation_workers)
    threads = int(args.worker_threads)
    if min(cpus, rollout_workers, evaluation_workers, threads) <= 0 or not memory:
        raise ValueError(f"Invalid {profile} resource geometry.")
    if rollout_workers * threads > cpus or evaluation_workers * threads > cpus:
        raise ValueError(f"{profile} worker geometry exceeds cpus_per_task={cpus}.")
    return {
        "cpus_per_task": cpus,
        "memory_per_task": memory,
        "num_rollout_workers": rollout_workers,
        "rollout_worker_threads": threads,
        "evaluation_num_workers": evaluation_workers,
        "evaluation_worker_threads": threads,
    }


def _simulator_arguments(profile: str) -> dict[str, Any]:
    if profile == "optimized":
        return {
            "road_query_mode": "spatial",
            "collision_check_mode": "broadphase",
            "record_replay_diagnostics": False,
            "sensor_road_edge_mode": "batched",
        }
    return {
        "road_query_mode": "legacy",
        "collision_check_mode": "legacy",
        "record_replay_diagnostics": True,
        "sensor_road_edge_mode": "per_vehicle",
    }


def _architecture_arguments(checkpoint: Path) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    architecture = dict(payload.get("policy_architecture") or {})
    config = dict(payload.get("config") or {})
    names = (
        "policy_model", "hidden_size", "transformer_layers", "transformer_heads",
        "transformer_dropout", "transformer_norm_first", "transformer_memory_tokens",
        "transformer_memory_context_length", "transformer_use_causal_attention",
        "continuous_action_dim",
    )
    return {
        name: architecture.get(name, config.get(name))
        for name in names
        if architecture.get(name, config.get(name)) is not None
    }


def _trial_row(
    *,
    method: str,
    stage: int,
    local_index: int,
    canonical_index: int,
    domain: str,
    depth: int,
    seed: int,
    arguments: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    return {
        "index": local_index,
        "trial_id": f"{method}_c{canonical_index:02d}_{domain}_d{depth}_s{seed}_stage{stage}",
        "method": method,
        "phase": f"final_stage{stage}",
        "seed": seed,
        "variant": str(arguments["algorithm_variant"]),
        "initialization": "bc" if stage == 1 else "resume",
        "collision_training": "final_curriculum",
        "module": (
            "scripts_gail.train_simple_ps_gail"
            if method == "gail"
            else "scripts_gail.train_simple_airl"
        ),
        "run_name": run_name,
        "arguments": arguments,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    if not args.run_root.is_absolute():
        raise ValueError("run-root must be an explicit absolute path.")
    study_id = _validate_study_id(args.study_id)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty manifest directory: {output_dir}")
    domain_data = {
        "us": (args.us_expert.resolve(), args.us_episode_root.resolve(), "us-101"),
        "japanese": (
            args.japanese_expert.resolve(), args.japanese_episode_root.resolve(), "japanese"
        ),
    }
    for domain, (expert, episodes, _scene) in domain_data.items():
        if not expert.is_dir() or not episodes.is_dir():
            raise FileNotFoundError(f"Missing {domain} expert/episode root: {expert}, {episodes}")
    recipes = {
        "gail": _load_recipe(args.gail_recipe.resolve(), "gail"),
        "airl": _load_recipe(args.airl_recipe.resolve(), "airl"),
    }
    warm_starts, bc_audit = _load_bc_warm_starts(args.bc_audit)
    source_lock = source_code_lock(args.source_repo, args.source_revision)
    resources = {
        "canary": _resource_geometry(args, "canary"),
        "production": _resource_geometry(args, "production"),
    }
    if int(args.evaluation_cache_limit) <= 0:
        raise ValueError("evaluation-cache-limit must be finite and positive.")
    cells = canonical_cells()
    written: dict[str, str] = {}
    for method in ("gail", "airl"):
        cell_records: list[dict[str, Any]] = []
        rows_by_stage: dict[int, dict[int, dict[str, Any]]] = {1: {}, 2: {}}
        for canonical_index, domain, depth, seed in cells:
            expert, episode_root, scene = domain_data[domain]
            warm_start = warm_starts[(domain, depth)]
            stem = f"c{canonical_index:02d}_{domain}_d{depth}_s{seed}"
            stage1_name = str(PurePosixPath(study_id) / method / stem / "stage1_rounds_0001_0600")
            stage2_name = str(PurePosixPath(study_id) / method / stem / "stage2_rounds_0601_0800")
            stage1_dir = (run_root / stage1_name).resolve()
            stage2_dir = (run_root / stage2_name).resolve()
            for target in (stage1_dir, stage2_dir):
                try:
                    target.relative_to(run_root)
                except ValueError as exc:
                    raise RuntimeError(f"Trial output escapes run root: {target}") from exc
                if target.exists():
                    raise FileExistsError(f"Refusing to reuse trial output: {target}")
            common = dict(recipes[method])
            common.update(_architecture_arguments(warm_start))
            common.update(
                {
                    "run_root": str(run_root),
                    "expert_data": str(expert),
                    "episode_root": str(episode_root),
                    "scene": scene,
                    "study_domain": domain,
                    "study_cell_index": canonical_index,
                    "seed": seed,
                    "action_mode": "continuous",
                    "policy_model": "recurrent_transformer",
                    "transformer_layers": depth,
                    "total_rounds": 800,
                    "controlled_vehicle_curriculum": True,
                    "initial_controlled_vehicles": 10,
                    "final_controlled_vehicles": 100,
                    "controlled_vehicle_curriculum_rounds": 800,
                    "controlled_vehicle_schedule": CONTROLLED_VEHICLE_SCHEDULE,
                    "rollout_target_agent_steps": 10_000,
                    "initial_rollout_target_agent_steps": 10_000,
                    "final_rollout_target_agent_steps": 40_000,
                    "rollout_target_agent_steps_curriculum_rounds": 800,
                    "rollout_target_agent_steps_schedule": ROLLOUT_TARGET_SCHEDULE,
                    "prebuilt_split": "train",
                    "validation_prebuilt_split": "val",
                    "test_prebuilt_split": "test",
                    "validation_every": 10,
                    "validation_episodes": 10,
                    "evaluate_initial_policy": True,
                    "save_best_checkpoint": True,
                    "checkpoint_every": 20,
                    "test_episodes": 30,
                    "save_checkpoint_video": False,
                    "rollout_cache_envs": True,
                    "rollout_max_cached_envs_per_worker": 2,
                    "rollout_profile": True,
                    "evaluation_cache_envs": True,
                    "evaluation_max_cached_envs_per_worker": int(args.evaluation_cache_limit),
                    "allow_legacy_model_only_resume": False,
                    "allow_unverified_resume_checkpoint": False,
                }
            )
            common.update(_simulator_arguments(args.simulator_profile))
            profile = "canary" if canonical_index in CANARY_INDICES else "production"
            geometry = resources[profile]
            common.update({key: value for key, value in geometry.items() if key != "cpus_per_task" and key != "memory_per_task"})
            stage1_args = dict(common)
            stage1_args.update(
                {
                    "run_name": stage1_name,
                    "initial_policy_checkpoint": str(warm_start),
                    "resume_checkpoint": "",
                    "stop_after_round": 600,
                    "expected_resume_round": 0,
                    "study_stage": 1,
                }
            )
            stage2_args = dict(common)
            stage2_args.update(
                {
                    "run_name": stage2_name,
                    "initial_policy_checkpoint": "",
                    "resume_checkpoint": str(stage1_dir / "resume_latest.pt"),
                    "stop_after_round": 0,
                    "expected_resume_round": 600,
                    "study_stage": 2,
                }
            )
            rows_by_stage[1][canonical_index] = _trial_row(
                method=method, stage=1, local_index=0, canonical_index=canonical_index,
                domain=domain, depth=depth, seed=seed, arguments=stage1_args, run_name=stage1_name,
            )
            rows_by_stage[2][canonical_index] = _trial_row(
                method=method, stage=2, local_index=0, canonical_index=canonical_index,
                domain=domain, depth=depth, seed=seed, arguments=stage2_args, run_name=stage2_name,
            )
            cell_records.append(
                {
                    "canonical_index": canonical_index,
                    "domain": domain,
                    "transformer_layers": depth,
                    "policy_seed": seed,
                    "warm_start": str(warm_start),
                    "stage1": {
                        "run_dir": str(stage1_dir),
                        "resume_checkpoint": str(stage1_dir / "resume_latest.pt"),
                        "best_checkpoint": str(stage1_dir / "best.pt"),
                        "rounds": [1, 600],
                    },
                    "stage2": {
                        "run_dir": str(stage2_dir),
                        "resume_checkpoint": str(stage2_dir / "resume_latest.pt"),
                        "best_checkpoint": str(stage2_dir / "best.pt"),
                        "final_checkpoint": str(stage2_dir / "final.pt"),
                        "rounds": [601, 800],
                    },
                    "official_best_checkpoint": str(stage2_dir / "best.pt"),
                    "official_final_checkpoint": str(stage2_dir / "final.pt"),
                    "interpretability_checkpoints": [
                        str(stage2_dir / "best.pt"), str(stage2_dir / "final.pt")
                    ],
                }
            )
        launch_files: dict[str, str] = {}
        selections = {
            "canary": list(CANARY_INDICES),
            "remaining": [index for index in range(12) if index not in CANARY_INDICES],
        }
        for selection, canonical_indices in selections.items():
            profile = "canary" if selection == "canary" else "production"
            for stage in (1, 2):
                rows: list[dict[str, Any]] = []
                for local_index, canonical_index in enumerate(canonical_indices):
                    row = dict(rows_by_stage[stage][canonical_index])
                    row["index"] = local_index
                    rows.append(row)
                name = f"{method}_stage{stage}_{selection}.json"
                path = output_dir / name
                payload = {
                    "schema_version": 3,
                    "manifest_kind": "gail_airl_final_stage",
                    "method": method,
                    "stage": stage,
                    "selection": selection,
                    "launch_profile": profile,
                    "simulator_profile": args.simulator_profile,
                    "canonical_indices": canonical_indices,
                    "trial_count": len(rows),
                    "run_root": str(run_root),
                    "resource_geometry": resources[profile],
                    "source_code": source_lock,
                    "bc_audit": bc_audit,
                    "trials": rows,
                }
                _atomic_json(path, payload)
                launch_files[f"stage{stage}_{selection}"] = str(path)
                written[name] = str(path)
        campaign_path = output_dir / f"{method}_final_12.json"
        campaign = {
            "schema_version": 3,
            "manifest_kind": "gail_airl_final_12",
            "method": method,
            "cell_count": 12,
            "axes": {
                "domains": list(DOMAINS),
                "transformer_layers": list(DEPTHS),
                "policy_seeds": list(SEEDS),
            },
            "canonical_indices": list(range(12)),
            "canary_indices": list(CANARY_INDICES),
            "remaining_indices": [index for index in range(12) if index not in CANARY_INDICES],
            "stage_boundaries": {"stage1": [1, 600], "stage2": [601, 800]},
            "simulator_profile": args.simulator_profile,
            "source_code": source_lock,
            "bc_audit": bc_audit,
            "resource_geometry": resources,
            "launch_manifests": launch_files,
            "checkpoint_retention": {
                "best": "lean inference/interpretability artifact; prior best is verified and carried to stage2",
                "periodic": "lean inference snapshots",
                "resume_latest": "single rotating exact optimizer/RNG/replay/archive continuation state",
                "final": "lean official round-800 inference/interpretability artifact",
            },
            "registry_contract": {
                "recursive_run_root_scans_supported": False,
                "rule": "Register only campaign-declared stage2 official_best_checkpoint and official_final_checkpoint paths.",
                "reason": "Stage1 checkpoints are continuation artifacts for the same canonical cell.",
            },
            "cells": cell_records,
        }
        _atomic_json(campaign_path, campaign)
        written[campaign_path.name] = str(campaign_path)
    return {
        "output_dir": str(output_dir),
        "simulator_profile": args.simulator_profile,
        "canary_indices": list(CANARY_INDICES),
        "files": written,
    }


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
