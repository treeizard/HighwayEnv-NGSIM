#!/usr/bin/env python3
"""Verify and index a recipe-defined domain/depth/seed BC study.

Every full-training model must have a summary, loadable checkpoint, hash
sidecar, collision-enabled locked-test record, and activation smoke
manifests. Reported metrics do not reject trained study models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", required=True)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finalize_study(policy_root: Path, smoke_root: Path, out: Path) -> dict[str, Any]:
    policy_root = policy_root.resolve()
    smoke_root = smoke_root.resolve()
    out = out.resolve()
    models: list[dict[str, Any]] = []
    checkpoint_count = 0
    metric_capability_passed_count = 0
    capability_passed_count = 0

    summary_paths = sorted(policy_root.glob("*/*/*/summary.json"))
    require(summary_paths, f"No BC study summaries found under {policy_root}.")
    for summary_path in summary_paths:
                run_dir = summary_path.parent
                relative = run_dir.relative_to(policy_root)
                checkpoint = run_dir / "best.pt"
                hash_path = run_dir / "best.pt.sha256"
                require(summary_path.is_file(), f"Missing required study summary: {summary_path}")
                summary = read_json(summary_path)
                domain = str(summary.get("domain"))
                layers = int(summary.get("transformer_layers", -1))
                seed = int(summary.get("seed", -1))
                require(domain in {"us", "japanese"}, f"Unknown domain in {summary_path}")
                require(summary.get("domain") == domain, f"Domain mismatch in {summary_path}")
                require(int(summary.get("transformer_layers", -1)) == layers, f"Depth mismatch in {summary_path}")
                require(int(summary.get("seed", -1)) == seed, f"Seed mismatch in {summary_path}")
                metric_capability = bool(summary.get("metric_capability_passed"))
                checkpoint_saved = bool(summary.get("checkpoint_saved"))
                require(
                    checkpoint_saved,
                    f"Full BC checkpoint was not saved for required model: {summary_path}",
                )
                evaluation = summary.get("held_out_evaluation") or {}
                require(
                    evaluation.get("prebuilt_split") == "test",
                    f"BC evaluation did not use the test split: {summary_path}",
                )
                require(
                    evaluation.get("collision_physics_enabled") is True,
                    f"BC locked test did not enable collision physics: {summary_path}",
                )
                require(
                    summary.get("final_test_qualification_passed") is True,
                    f"BC locked test qualification did not pass: {summary_path}",
                )
                evaluation_metrics = evaluation.get("metrics") or {}
                require(
                    "bc_eval/collision_episode_fraction" in evaluation_metrics,
                    f"BC evaluation did not record collision-only rate: {summary_path}",
                )
                require(
                    "bc_eval/collision_proxy_episode_fraction" in evaluation_metrics,
                    f"BC evaluation did not record collision-proxy rate: {summary_path}",
                )

                checkpoint_hash: str | None = None
                activation_manifests: list[str] = []
                for required_path in (checkpoint, hash_path):
                    require(required_path.is_file(), f"Missing full BC artifact: {required_path}")
                checkpoint_hash = sha256_file(checkpoint)
                require(summary.get("checkpoint_sha256") == checkpoint_hash, f"Summary hash mismatch: {checkpoint}")
                require(
                    hash_path.read_text(encoding="utf-8").split()[0] == checkpoint_hash,
                    f"Sidecar hash mismatch: {checkpoint}",
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                require(isinstance(payload, dict), f"Checkpoint is not a mapping: {checkpoint}")
                architecture = payload.get("policy_architecture") or {}
                config = payload.get("config") or {}
                require(
                    architecture.get("policy_model") == "recurrent_transformer",
                    f"Wrong policy model: {checkpoint}",
                )
                require(
                    int(architecture.get("transformer_layers", -1)) == layers,
                    f"Wrong checkpoint depth: {checkpoint}",
                )
                require(int(config.get("seed", -1)) == seed, f"Wrong checkpoint seed: {checkpoint}")
                require("policy_state_dict" in payload, f"Missing policy state: {checkpoint}")
                for layer in range(layers):
                    manifest = smoke_root / relative / f"residual_layer_{layer}_policy_token" / "manifest.json"
                    require(manifest.is_file(), f"Missing interpretation smoke manifest: {manifest}")
                    activation_manifests.append(str(manifest))
                checkpoint_count += 1

                if metric_capability:
                    metric_capability_passed_count += 1

                if bool(summary.get("capability_passed")):
                    capability_passed_count += 1

                models.append(
                    {
                        "domain": domain,
                        "transformer_layers": layers,
                        "policy_seed": seed,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "summary": str(summary_path),
                        "activation_manifests": activation_manifests,
                        "validation_skill": summary.get("validation_skill"),
                        "validation_mae": summary.get("validation_mae"),
                        "held_out_rollouts": summary.get("held_out_rollouts"),
                        "held_out_evaluation": evaluation,
                    }
                )

    require(
        checkpoint_count == len(models),
        f"Expected {len(models)} verified checkpoints, got {checkpoint_count}.",
    )
    result = {
        "schema_version": 1,
        "study": "bc_domain_depth_v1",
        "model_count": len(models),
        "checkpoint_count": checkpoint_count,
        "metric_capability_passed_count": metric_capability_passed_count,
        "metric_threshold_not_met_count": len(models) - metric_capability_passed_count,
        "capability_passed_count": capability_passed_count,
        "domains": ["us", "japanese"],
        "transformer_layers": [2, 3],
        "policy_seeds": [0, 1, 2],
        "models": models,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = finalize_study(Path(args.policy_root), Path(args.smoke_root), Path(args.out))
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "verified_runs": result["model_count"],
                "verified_checkpoints": result["checkpoint_count"],
                "capability_passed": result["capability_passed_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
