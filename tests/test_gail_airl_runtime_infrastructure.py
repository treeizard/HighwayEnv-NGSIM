from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts_gail.prepare_gail_airl_method_manifest import prepare
from scripts_gail.ps_gail.checkpoints import (
    atomic_torch_save,
    checkpoint_metadata,
    verify_checkpoint_sidecar,
)
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail import envs as training_envs
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.study import build_screening_trials, write_study_files
from scripts_gail.ps_gail.training import rollouts
from scripts_gail.ps_gail.validation import best_checkpoint_payload


class _DummyEnv:
    def __init__(self, identity: int):
        self.identity = identity
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_training_env_propagates_explicit_exact_fast_modes(monkeypatch):
    captured = {}
    monkeypatch.setattr(training_envs, "register_ngsim_env", lambda: None)
    monkeypatch.setattr(training_envs, "build_env_config", lambda **_kwargs: {"action": {}})

    def make(_env_id, **kwargs):
        captured.update(kwargs["config"])
        return object()

    monkeypatch.setattr(training_envs.gym, "make", make)
    training_envs.make_training_env(
        PSGAILConfig(
            action_mode="continuous",
            road_query_mode="spatial",
            collision_check_mode="broadphase",
            record_replay_diagnostics=False,
        )
    )
    assert captured["road_query_mode"] == "spatial"
    assert captured["collision_check_mode"] == "broadphase"
    assert captured["record_replay_diagnostics"] is False


def test_rollout_worker_reuses_policy_and_refreshes_weights():
    rollouts.clear_rollout_worker_caches()
    cfg = PSGAILConfig(
        policy_model="mlp",
        hidden_size=8,
        action_mode="continuous",
        continuous_action_dim=2,
    )
    source = make_actor_critic("mlp", 3, 8, action_mode="continuous", continuous_action_dim=2)
    first, first_hit = rollouts._cached_rollout_policy(cfg, source.state_dict(), 3, 3)
    with torch.no_grad():
        next(iter(source.parameters())).add_(1.0)
    second, second_hit = rollouts._cached_rollout_policy(cfg, source.state_dict(), 3, 3)

    assert not first_hit
    assert second_hit
    assert first is second
    for expected, actual in zip(source.parameters(), second.parameters()):
        torch.testing.assert_close(expected, actual)
    assert rollouts.rollout_worker_cache_stats()["policy_hits"] == 1
    rollouts.clear_rollout_worker_caches()


def test_rollout_worker_env_cache_is_structural_bounded_lru(monkeypatch, tmp_path):
    rollouts.clear_rollout_worker_caches()
    created: list[_DummyEnv] = []

    def make_env(_cfg):
        env = _DummyEnv(len(created))
        created.append(env)
        return env

    monkeypatch.setattr("scripts_gail.ps_gail.envs.make_training_env", make_env)
    cfg = PSGAILConfig(
        episode_root=str(tmp_path),
        rollout_max_cached_envs_per_worker=2,
        percentage_controlled_vehicles=10,
    )
    first, first_hit = rollouts._cached_rollout_env(cfg)
    again, again_hit = rollouts._cached_rollout_env(replace(cfg, seed=999, rollout_steps=999))
    second, second_hit = rollouts._cached_rollout_env(replace(cfg, percentage_controlled_vehicles=20))
    third, third_hit = rollouts._cached_rollout_env(replace(cfg, percentage_controlled_vehicles=30))

    assert not first_hit and again_hit
    assert first is again
    assert not second_hit and not third_hit
    assert first.closed
    assert not second.closed and not third.closed
    assert rollouts.rollout_worker_cache_stats() == {
        "policy_hits": 0,
        "policy_misses": 0,
        "env_hits": 1,
        "env_misses": 3,
        "policies": 0,
        "envs": 2,
    }
    rollouts.clear_rollout_worker_caches()
    assert second.closed and third.closed


def test_atomic_checkpoint_has_sha_and_canonical_study_cell(tmp_path):
    cfg = PSGAILConfig(scene="us-101", seed=2, transformer_layers=3)
    payload = {
        **checkpoint_metadata(cfg, method="gail", checkpoint_kind="gail_round"),
        "policy_state_dict": {"weight": torch.arange(3)},
    }
    payload = best_checkpoint_payload(
        payload,
        round_idx=7,
        validation_metrics={"validation/score": 1.0},
        validation_score=1.0,
        validation_cost=-1.0,
    )
    checkpoint = tmp_path / "best.pt"
    digest = atomic_torch_save(payload, checkpoint, refuse_overwrite=True)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert verify_checkpoint_sidecar(checkpoint) == digest
    assert loaded["checkpoint_kind"] == "gail_best"
    assert loaded["study_cell"] == {
        "method": "gail",
        "domain": "us",
        "transformer_layers": 3,
        "policy_seed": 2,
        "checkpoint_kind": "best",
    }
    assert len(loaded["normalized_config_hash"]) == 64
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        atomic_torch_save(payload, checkpoint, refuse_overwrite=True)


def test_method_manifest_filters_bounds_resources_and_refuses_outputs(tmp_path):
    source_dir = tmp_path / "source"
    trials = build_screening_trials(
        expert_data="expert",
        episode_root="episodes",
        bc_checkpoint="bc.pt",
    )
    source = write_study_files(str(source_dir), trials)["manifest"]
    output = tmp_path / "submission" / "trials.json"
    run_root = tmp_path / "runs"
    args = Namespace(
        source=type(output)(source),
        output=output,
        method="airl",
        selection="canary",
        canary_count=4,
        run_root=run_root,
        run_prefix="stamp/canary",
        study_domain="japanese",
        num_rollout_workers=16,
        rollout_worker_threads=2,
        evaluation_num_workers=8,
        evaluation_worker_threads=2,
        cpus_per_task=32,
    )
    result = prepare(args)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result["trial_count"] == 4
    assert payload["method"] == "airl"
    assert [row["index"] for row in payload["trials"]] == list(range(4))
    assert all(row["method"] == "airl" for row in payload["trials"])
    assert all(row["arguments"]["run_root"] == str(run_root.resolve()) for row in payload["trials"])
    assert all(row["arguments"]["num_rollout_workers"] == 16 for row in payload["trials"])
    assert all(row["arguments"]["rollout_worker_threads"] == 2 for row in payload["trials"])
    assert all(row["arguments"]["study_domain"] == "japanese" for row in payload["trials"])
    assert all(row["arguments"]["road_query_mode"] == "spatial" for row in payload["trials"])
    assert all(row["arguments"]["collision_check_mode"] == "broadphase" for row in payload["trials"])
    assert all(not row["arguments"]["record_replay_diagnostics"] for row in payload["trials"])
    with pytest.raises(FileExistsError, match="method manifest"):
        prepare(args)


def test_method_array_scripts_have_no_unbounded_embedded_array():
    repo = Path(__file__).resolve().parents[1]
    for relative in (
        "hpc/slurm/script_full_training/run_gail_airl_study_array.bash",
        "hpc/slurm/script_full_training/run_gail_airl_confirmation_array.bash",
        "hpc/slurm/script_full_training/run_gail_airl_method_array.bash",
    ):
        text = (repo / relative).read_text(encoding="utf-8")
        assert "#SBATCH --array" not in text
    submit = (repo / "hpc/slurm/script_full_training/submit_gail_airl_method_array.bash").read_text(
        encoding="utf-8"
    )
    assert "concurrency=2" in submit
    assert "concurrency=4" in submit
    assert '--cpus-per-task="${CPUS_PER_TASK}"' in submit


def test_submit_wrapper_dry_run_uses_profile_bounds_and_explicit_roots(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source_dir = tmp_path / "source"
    source = write_study_files(
        str(source_dir),
        build_screening_trials(
            expert_data="expert",
            episode_root="episodes",
            bc_checkpoint="bc.pt",
        ),
    )["manifest"]
    canonical = tmp_path / "canonical"
    env = os.environ.copy()
    env.update(
        {
            "REPODIR": str(repo),
            "SOURCE_MANIFEST": source,
            "METHOD": "gail",
            "LAUNCH_PROFILE": "canary",
            "RUN_STAMP": "teststamp",
            "DRY_RUN": "true",
            "PYTHON_BIN": sys.executable,
            "VFI_PROJECT_ROOT": str(canonical),
            "VFI_DATA_ROOT": str(canonical / "data"),
            "VFI_HIGHWAY_DATA_ROOT": str(canonical / "data" / "highway_env"),
            "VFI_CHECKPOINT_ROOT": str(canonical / "data" / "checkpoints"),
            "VFI_RESULTS_ROOT": str(canonical / "results"),
            "VFI_LOG_ROOT": str(canonical / "logs"),
            "VFI_ARTIFACT_ROOT": str(canonical / "artifacts"),
        }
    )
    wrapper = repo / "hpc/slurm/script_full_training/submit_gail_airl_method_array.bash"
    completed = subprocess.run(
        ["bash", str(wrapper)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "trials=4 array=0-3%2" in completed.stdout
    assert "--cpus-per-task=16" in completed.stdout
    assert "--mem=64G" in completed.stdout
    prepared = json.loads(
        (
            canonical
            / "results/runs/submissions/gail_airl_gail_canary_teststamp/trials.json"
        ).read_text(encoding="utf-8")
    )
    assert prepared["source_code"]["revision"]
    assert "scripts_gail/ps_gail/training/rollouts.py" in prepared["source_code"]["files_sha256"]

    env.update({"LAUNCH_PROFILE": "production", "RUN_STAMP": "prodstamp"})
    production = subprocess.run(
        ["bash", str(wrapper)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "trials=20 array=0-19%4" in production.stdout
    assert "--cpus-per-task=32" in production.stdout
    assert "--mem=128G" in production.stdout
