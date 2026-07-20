#!/usr/bin/env python3
"""Atomically archive learning-qualified BC checkpoints with integrity metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def archive_bc_checkpoints(
    sources: list[Path],
    *,
    destination_root: Path,
    archive_id: str,
    label: str,
) -> dict[str, Any]:
    """Copy qualified cells into one checksummed, read-only archive directory."""
    destination_root = destination_root.resolve()
    final_dir = destination_root / archive_id
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint archive: {final_dir}")
    destination_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{archive_id}.staging.", dir=destination_root))
    records: list[dict[str, Any]] = []
    try:
        for source in sources:
            source = source.resolve()
            summary_path = source / "summary.json"
            checkpoint_path = source / "best.pt"
            summary = _read_json(summary_path)
            if not bool(summary.get("metric_capability_passed")):
                raise ValueError(f"Checkpoint is not learning-qualified: {checkpoint_path}")
            expected_digest = str(summary.get("checkpoint_sha256") or "")
            actual_digest = sha256_file(checkpoint_path)
            if not expected_digest or expected_digest != actual_digest:
                raise ValueError(f"Checkpoint digest mismatch: {checkpoint_path}")

            domain = str(summary["domain"])
            layers = int(summary["transformer_layers"])
            seed = int(summary["seed"])
            relative = Path(domain) / f"recurrent_transformer_{layers}layer" / f"policy_seed_{seed}"
            target = staging_dir / relative
            target.mkdir(parents=True, exist_ok=False)
            copied: dict[str, str] = {}
            for name in ("best.pt", "best.pt.sha256", "summary.json", "split_manifest.json", "metrics.jsonl"):
                source_file = source / name
                if not source_file.is_file():
                    if name in {"best.pt", "summary.json"}:
                        raise FileNotFoundError(source_file)
                    continue
                target_file = target / name
                shutil.copy2(source_file, target_file)
                copied[name] = sha256_file(target_file)
            if copied["best.pt"] != actual_digest:
                raise RuntimeError(f"Archived checkpoint verification failed: {target / 'best.pt'}")
            records.append(
                {
                    "domain": domain,
                    "transformer_layers": layers,
                    "seed": seed,
                    "source": str(source),
                    "archive_relative_path": str(relative),
                    "checkpoint_sha256": actual_digest,
                    "files": copied,
                }
            )

        records.sort(key=lambda row: (row["domain"], row["transformer_layers"], row["seed"]))
        manifest = {
            "schema_version": 1,
            "archive_id": archive_id,
            "label": label,
            "qualification": "metric_capability_passed",
            "checkpoint_count": len(records),
            "checkpoints": records,
        }
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_digest = sha256_file(manifest_path)
        (staging_dir / "manifest.json.sha256").write_text(
            f"{manifest_digest}  manifest.json\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, final_dir)
        for path in sorted(final_dir.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        final_dir.chmod(0o555)
        return {**manifest, "archive": str(final_dir), "manifest_sha256": manifest_digest}
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--archive-id", required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = archive_bc_checkpoints(
        list(args.source),
        destination_root=args.destination_root,
        archive_id=str(args.archive_id),
        label=str(args.label),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
