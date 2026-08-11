from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from policy.evaluation.checkpoint_archive import archive_bc_checkpoints


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_source(
    root: Path,
    *,
    qualified: bool = True,
    policy_qualified: bool | None = None,
    artifact_complete: bool | None = None,
) -> Path:
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
                "capability_passed": (
                    qualified if policy_qualified is None else policy_qualified
                ),
                "training_artifact_complete": (
                    qualified if artifact_complete is None else artifact_complete
                ),
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
    with pytest.raises(
        ValueError,
        match="does not satisfy metric_capability_passed",
    ):
        archive_bc_checkpoints(
            [source],
            destination_root=tmp_path / "archive",
            archive_id="study",
            label="test",
        )


def test_policy_archive_rejects_metric_only_checkpoint(tmp_path):
    source = make_source(
        tmp_path / "source",
        qualified=True,
        policy_qualified=False,
    )
    with pytest.raises(ValueError, match="does not satisfy capability_passed"):
        archive_bc_checkpoints(
            [source],
            destination_root=tmp_path / "archive",
            archive_id="study",
            label="test",
            qualification_field="capability_passed",
        )


def test_complete_artifact_archive_keeps_checkpoint_when_rollout_gate_fails(tmp_path):
    source = make_source(
        tmp_path / "source",
        qualified=True,
        policy_qualified=False,
        artifact_complete=True,
    )
    result = archive_bc_checkpoints(
        [source],
        destination_root=tmp_path / "archive",
        archive_id="study",
        label="BC interpretability reference",
        qualification_field="training_artifact_complete",
    )

    assert result["checkpoint_count"] == 1
    assert result["qualification"] == "training_artifact_complete"
