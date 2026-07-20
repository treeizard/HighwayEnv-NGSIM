#!/usr/bin/env python3
"""Build the terminal four-cell BC warm-start audit required by final GAIL/AIRL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED = {("us", 2), ("us", 3), ("japanese", 2), ("japanese", 3)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_mapping(value: str) -> tuple[tuple[str, int], Path]:
    try:
        identity, raw_path = value.split("=", 1)
        domain, raw_depth = identity.split(":", 1)
        key = (domain.strip().lower(), int(raw_depth))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Use DOMAIN:DEPTH=/absolute/best.pt") from exc
    if key not in EXPECTED:
        raise argparse.ArgumentTypeError(f"Unexpected warm-start identity: {key}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("Warm-start checkpoint paths must be absolute.")
    return key, path


def build(mappings: list[tuple[tuple[str, int], Path]]) -> dict[str, Any]:
    selected = dict(mappings)
    if len(selected) != len(mappings):
        raise RuntimeError("Duplicate BC warm-start mapping.")
    if set(selected) != EXPECTED:
        raise RuntimeError(f"Warm-start coverage mismatch: {sorted(selected)} != {sorted(EXPECTED)}")
    rows = []
    for domain, depth in sorted(EXPECTED):
        checkpoint = selected[(domain, depth)].resolve()
        sidecar = checkpoint.with_name(f"{checkpoint.name}.sha256")
        summary_path = checkpoint.with_name("summary.json")
        for required in (checkpoint, sidecar, summary_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        checkpoint_sha = _sha256(checkpoint)
        if sidecar.read_text(encoding="utf-8").split()[0] != checkpoint_sha:
            raise RuntimeError(f"BC warm-start sidecar mismatch: {checkpoint}")
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if (
            summary.get("checkpoint_purpose") != "warm_start"
            or summary.get("checkpoint_saved") is not True
            or summary.get("warm_start_passed") is not True
            or str(summary.get("domain")) != domain
            or int(summary.get("transformer_layers", -1)) != depth
        ):
            raise RuntimeError(f"BC warm-start summary gate failed: {summary_path}")
        warm_gate = dict(summary.get("warm_start_gate") or {})
        configured_epochs = int(
            warm_gate.get("configured_epochs", summary.get("configured_epochs", summary.get("epochs", 0)))
        )
        if not 1 <= configured_epochs <= 5:
            raise RuntimeError(f"BC warm start must use 1..5 epochs: {summary_path}")
        if int(warm_gate.get("maximum_epochs", 5)) > 5:
            raise RuntimeError(f"BC warm-start maximum epoch gate exceeds 5: {summary_path}")
        if summary.get("checkpoint_sha256") not in (None, checkpoint_sha):
            raise RuntimeError(f"BC summary checkpoint hash mismatch: {summary_path}")
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        architecture = dict(payload.get("policy_architecture") or {})
        config = dict(payload.get("config") or {})
        expected_scene = "us-101" if domain == "us" else "japanese"
        if (
            payload.get("checkpoint_kind") != "behaviour_cloning_warm_start"
            or architecture.get("policy_model", config.get("policy_model")) != "recurrent_transformer"
            or architecture.get("action_mode", config.get("action_mode")) != "continuous"
            or int(architecture.get("transformer_layers", config.get("transformer_layers", -1))) != depth
            or str(config.get("scene")) != expected_scene
            or "policy_state_dict" not in payload
        ):
            raise RuntimeError(f"BC warm-start checkpoint contract failed: {checkpoint}")
        rows.append(
            {
                "domain": domain,
                "transformer_layers": depth,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "warm_start_passed": True,
                "configured_epochs": configured_epochs,
                "selected": True,
            }
        )
    return {
        "schema_version": 1,
        "audit_kind": "gail_airl_bc_warm_start_selection",
        "status": "passed",
        "terminal": True,
        "warm_starts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warm-start", action="append", required=True, type=_parse_mapping,
        help="Repeat exactly four times: DOMAIN:DEPTH=/absolute/best.pt",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.warm_start)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite BC audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(output), "warm_starts": 4, "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
