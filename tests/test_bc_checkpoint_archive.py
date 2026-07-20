from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts_gail.archive_bc_checkpoints import archive_bc_checkpoints


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_source(root: Path, *, qualified: bool = True) -> Path:
    root.mkdir(parents=True)
    checkpoint = root / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_digest = digest(checkpoint)
    (root / "best.pt.sha256").write_text(f"{checkpoint_digest}  best.pt\n")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "domain": "us",
                "transformer_layers": 2,
                "seed": 2,
                "metric_capability_passed": qualified,
                "checkpoint_sha256": checkpoint_digest,
            }
        )
        + "\n"
    )
    (root / "split_manifest.json").write_text("{}\n")
    return root


def test_archive_is_verified_atomic_read_only_and_overwrite_safe(tmp_path):
    source = make_source(tmp_path / "source")
    result = archive_bc_checkpoints(
        [source],
        destination_root=tmp_path / "archive",
        archive_id="study",
        label="test",
    )
    final = Path(result["archive"])
    copied = final / "us/recurrent_transformer_2layer/policy_seed_2/best.pt"
    assert copied.read_bytes() == b"checkpoint"
    assert copied.stat().st_mode & 0o222 == 0
    assert (final / "manifest.json.sha256").is_file()
    with pytest.raises(FileExistsError):
        archive_bc_checkpoints(
            [source],
            destination_root=tmp_path / "archive",
            archive_id="study",
            label="test",
        )


def test_archive_rejects_unqualified_checkpoint(tmp_path):
    source = make_source(tmp_path / "source", qualified=False)
    with pytest.raises(ValueError, match="not learning-qualified"):
        archive_bc_checkpoints(
            [source],
            destination_root=tmp_path / "archive",
            archive_id="study",
            label="test",
        )
