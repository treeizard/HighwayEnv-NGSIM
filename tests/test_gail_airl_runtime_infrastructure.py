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
from scripts_gail.build_gail_airl_final_manifests import build as build_final_manifests
from scripts_gail.promote_gail_airl_final_recipe import promote
from scripts_gail.build_bc_warm_start_audit import build as build_bc_warm_start_audit
from scripts_gail.ps_gail.checkpoints import (
    atomic_torch_save,
    checkpoint_metadata,
    exact_training_state,
    restore_exact_training_state,
    resume_config_hash,
    verify_checkpoint_sidecar,
    verify_resume_checkpoint,
)
from scripts_gail.ps_gail.config import PSGAILConfig
from scripts_gail.ps_gail import envs as training_envs
from scripts_gail.ps_gail.models import make_actor_critic
from scripts_gail.ps_gail.study import build_screening_trials, write_study_files
from scripts_gail.ps_gail.training import rollouts
from scripts_gail.ps_gail.training import evaluation
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
            sensor_road_edge_mode="batched",
            reuse_pre_reset_spaces=True,
        )
    )
    assert captured["road_query_mode"] == "spatial"
    assert captured["collision_check_mode"] == "broadphase"
    assert captured["record_replay_diagnostics"] is False
    assert captured["sensor_road_edge_mode"] == "batched"
    assert captured["reuse_pre_reset_spaces"] is True


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
    cfg = PSGAILConfig(
        scene="us-101", seed=2, transformer_layers=3, study_cell_index=5, study_stage=2
    )
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
        "canonical_index": 5,
        "campaign_stage": 2,
        "official": True,
    }
    assert loaded["identity_scope"] == "campaign_cell"
    assert len(loaded["normalized_config_hash"]) == 64
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        atomic_torch_save(payload, checkpoint, refuse_overwrite=True)


def test_noncampaign_checkpoint_has_explicit_trial_identity_only():
    metadata = checkpoint_metadata(
        PSGAILConfig(scene="japanese", seed=1, transformer_layers=2),
        method="airl",
        checkpoint_kind="airl_round",
    )
    assert metadata["identity_scope"] == "trial"
    assert "study_cell" not in metadata
    assert metadata["trial_identity"]["domain"] == "japanese"


def _deterministic_training_step(model, optimizer):
    python_scale = __import__("random").random()
    numpy_input = __import__("numpy").random.standard_normal((4, 3)).astype("float32")
    inputs = torch.from_numpy(numpy_input) + torch.rand(4, 3) * python_scale
    target = torch.rand(4, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    return loss.detach().clone()


def test_exact_training_state_interruption_resume_equivalence(tmp_path):
    import random
    import numpy as np

    cfg = PSGAILConfig(seed=9, total_rounds=3)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    uninterrupted = torch.nn.Linear(3, 2)
    uninterrupted_optimizer = torch.optim.Adam(uninterrupted.parameters(), lr=1.0e-2)
    _deterministic_training_step(uninterrupted, uninterrupted_optimizer)
    checkpoint_payload = {
        **checkpoint_metadata(cfg, method="gail", checkpoint_kind="gail_round"),
        "round": 1,
        "policy_state_dict": uninterrupted.state_dict(),
        "training_state": exact_training_state(
            completed_round=1,
            optimizers={"policy": uninterrupted_optimizer},
            trainer_state={"best_validation_round": 1, "replay": [np.arange(3)]},
        ),
    }
    checkpoint = tmp_path / "round_0001.pt"
    atomic_torch_save(checkpoint_payload, checkpoint)
    expected_loss = _deterministic_training_step(uninterrupted, uninterrupted_optimizer)
    expected_parameters = [parameter.detach().clone() for parameter in uninterrupted.parameters()]

    resumed = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1.0e-2)
    verify_resume_checkpoint(checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed.load_state_dict(loaded["policy_state_dict"])
    restored = restore_exact_training_state(
        loaded,
        optimizers={"policy": resumed_optimizer},
        expected_resume_config_hash=resume_config_hash(cfg),
    )
    actual_loss = _deterministic_training_step(resumed, resumed_optimizer)

    assert restored["start_round"] == 2
    assert restored["trainer_state"]["best_validation_round"] == 1
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for actual, expected in zip(resumed.parameters(), expected_parameters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
        study_domain="us",
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
    assert all(row["arguments"]["study_domain"] == "us" for row in payload["trials"])
    assert payload["simulator_profile"] == "legacy"
    assert all(row["arguments"]["road_query_mode"] == "legacy" for row in payload["trials"])
    assert all(row["arguments"]["collision_check_mode"] == "legacy" for row in payload["trials"])
    assert all(row["arguments"]["record_replay_diagnostics"] for row in payload["trials"])
    assert all(
        row["arguments"]["sensor_road_edge_mode"] == "per_vehicle"
        for row in payload["trials"]
    )
    assert all(not row["arguments"]["reuse_pre_reset_spaces"] for row in payload["trials"])
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
            "BLOCKING_JOB_IDS": "999999999",
            "ALLOW_TEST_SQUEUE_FIXTURE": "true",
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

    env.update(
        {
            "LAUNCH_PROFILE": "production",
            "RUN_STAMP": "prodstamp",
            "CPUS_PER_TASK": "32",
            "MEMORY_PER_TASK": "128G",
            "NUM_ROLLOUT_WORKERS": "16",
            "EVALUATION_NUM_WORKERS": "16",
        }
    )
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


class _CachedEvalEnv:
    def __init__(self, identity: str):
        self.identity = identity
        self.closed = False
        self.unwrapped = self
        self.config = {}

    def close(self):
        self.closed = True


def test_evaluation_env_cache_reports_miss_hit_and_enforces_lru(monkeypatch):
    evaluation.clear_evaluation_worker_caches()
    created = []

    def make_env(_cfg, *, split, episode_name):
        env = _CachedEvalEnv(f"{split}:{episode_name}")
        created.append(env)
        return env

    monkeypatch.setattr(evaluation, "_make_matched_eval_all_vehicle_env", make_env)
    cfg = PSGAILConfig(evaluation_max_cached_envs_per_worker=2)
    first, first_hit = evaluation._get_matched_eval_env(
        cfg, split="val", episode_name="a", all_vehicle=True
    )
    again, again_hit = evaluation._get_matched_eval_env(
        cfg, split="val", episode_name="a", all_vehicle=True
    )
    second, second_hit = evaluation._get_matched_eval_env(
        cfg, split="val", episode_name="b", all_vehicle=True
    )
    third, third_hit = evaluation._get_matched_eval_env(
        cfg, split="val", episode_name="c", all_vehicle=True
    )
    assert first is again
    assert (first_hit, again_hit, second_hit, third_hit) == (False, True, False, False)
    assert first.closed and not second.closed and not third.closed
    assert evaluation.evaluation_worker_cache_stats()["env_cache_size"] == 2
    evaluation.clear_evaluation_worker_caches()
    assert second.closed and third.closed


def test_all_matched_evaluation_constructors_propagate_simulator_modes(monkeypatch):
    captured = []
    monkeypatch.setattr(evaluation, "register_ngsim_env", lambda: None)
    monkeypatch.setattr(evaluation, "build_env_config", lambda **_kwargs: {})
    monkeypatch.setattr(training_envs, "observation_config", lambda _cfg: {})
    monkeypatch.setattr(
        evaluation.gym,
        "make",
        lambda _env_id, *, config: captured.append(dict(config)) or object(),
    )
    cfg = PSGAILConfig(
        action_mode="continuous",
        road_query_mode="spatial",
        collision_check_mode="broadphase",
        record_replay_diagnostics=False,
        sensor_road_edge_mode="batched",
        reuse_pre_reset_spaces=True,
    )
    evaluation._make_matched_eval_env(
        cfg, split="val", episode_name="episode", vehicle_id=1
    )
    evaluation._make_matched_eval_all_vehicle_env(
        cfg, split="val", episode_name="episode"
    )
    evaluation._make_matched_eval_selected_vehicle_env(
        cfg, split="val", episode_name="episode", vehicle_ids=(1, 2)
    )
    assert len(captured) == 3
    for env_cfg in captured:
        assert env_cfg["road_query_mode"] == "spatial"
        assert env_cfg["collision_check_mode"] == "broadphase"
        assert env_cfg["record_replay_diagnostics"] is False
        assert env_cfg["sensor_road_edge_mode"] == "batched"
        assert env_cfg["reuse_pre_reset_spaces"] is True


def _build_final_manifest_fixture(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    experts = {}
    episodes = {}
    for domain in ("us", "japanese"):
        experts[domain] = tmp_path / "expert" / domain
        episodes[domain] = tmp_path / "episodes" / domain
        experts[domain].mkdir(parents=True)
        episodes[domain].mkdir(parents=True)
    warm_rows = []
    for domain, scene in (("us", "us-101"), ("japanese", "japanese")):
        for depth in (2, 3):
            checkpoint = tmp_path / "warm" / domain / f"depth{depth}" / "best.pt"
            warm_sha = atomic_torch_save(
                {
                    "checkpoint_kind": "behaviour_cloning_warm_start",
                    "policy_state_dict": {"weight": torch.ones(1)},
                    "config": {
                        "scene": scene,
                        "policy_model": "recurrent_transformer",
                        "action_mode": "continuous",
                        "transformer_layers": depth,
                        "hidden_size": 32,
                        "transformer_heads": 4,
                        "transformer_dropout": 0.0,
                    },
                    "policy_architecture": {
                        "policy_model": "recurrent_transformer",
                        "action_mode": "continuous",
                        "transformer_layers": depth,
                        "hidden_size": 32,
                        "transformer_heads": 4,
                        "transformer_dropout": 0.0,
                    },
                },
                checkpoint,
            )
            checkpoint.with_name("summary.json").write_text(
                json.dumps(
                    {
                        "checkpoint_purpose": "warm_start",
                        "checkpoint_saved": True,
                        "warm_start_passed": True,
                        "domain": domain,
                        "transformer_layers": depth,
                        "checkpoint_sha256": warm_sha,
                        "warm_start_gate": {
                            "configured_epochs": 2,
                            "maximum_epochs": 5,
                        },
                    }
                )
            )
            warm_rows.append(
                {
                    "domain": domain,
                    "transformer_layers": depth,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": verify_checkpoint_sidecar(checkpoint),
                    "warm_start_passed": True,
                    "selected": True,
                }
            )
    bc_audit = tmp_path / "bc_audit.json"
    bc_audit.write_text(
        json.dumps({"status": "passed", "terminal": True, "warm_starts": warm_rows}),
        encoding="utf-8",
    )
    confirmation_manifest = tmp_path / "confirmation_manifest.json"
    confirmation_report = tmp_path / "confirmation_report.json"
    confirmation_trials = []
    groups = []
    for method in ("gail", "airl"):
        winner_ids = []
        for arm in ("winner", "current_baseline"):
            for seed in range(10, 15):
                trial_id = f"{method}_confirm_{arm}_s{seed}"
                confirmation_trials.append(
                    {
                        "trial_id": trial_id,
                        "method": method,
                        "phase": f"confirmation_{arm}",
                    }
                )
                if arm == "winner":
                    winner_ids.append(trial_id)
        groups.append(
            {
                "configuration_id": f"{method}-confirmed",
                "method": method,
                "phase": "confirmation_winner",
                "seed_gate_passed": True,
                "eligible_seeds": 5,
                "trial_ids": winner_ids,
                "mean_normalized_trajectory_error": 1.0,
                "representative": {
                    "eligible": True,
                    "method": method,
                    "arguments": {"algorithm_variant": f"{method}_bce"},
                },
            }
        )
    confirmation_manifest.write_text(json.dumps({"trial_count": 20, "trials": confirmation_trials}))
    confirmation_report.write_text(
        json.dumps({"completed": 20, "configuration_groups": groups})
    )
    gail_recipe = tmp_path / "gail_recipe.json"
    airl_recipe = tmp_path / "airl_recipe.json"
    gail_recipe.write_text(
        json.dumps(promote("gail", confirmation_report, confirmation_manifest))
    )
    airl_recipe.write_text(
        json.dumps(promote("airl", confirmation_report, confirmation_manifest))
    )
    output = tmp_path / "manifests"
    run_root = (tmp_path / "runs").resolve()
    result = build_final_manifests(
        Namespace(
            gail_recipe=gail_recipe,
            airl_recipe=airl_recipe,
            bc_audit=bc_audit,
            us_expert=experts["us"],
            japanese_expert=experts["japanese"],
            us_episode_root=episodes["us"],
            japanese_episode_root=episodes["japanese"],
            output_dir=output,
            run_root=run_root,
            study_id="final_fixture",
            source_repo=repo,
            source_revision="",
            simulator_profile="legacy",
            canary_cpus_per_task=16,
            canary_memory_per_task="64G",
            canary_rollout_workers=8,
            canary_evaluation_workers=8,
            production_cpus_per_task=32,
            production_memory_per_task="128G",
            production_rollout_workers=16,
            production_evaluation_workers=16,
            worker_threads=2,
            evaluation_cache_limit=4,
        )
    )
    return repo, output, run_root, bc_audit, result


def test_bc_warm_start_audit_builder_accepts_real_nested_gate_schema(tmp_path):
    _repo, _output, _run_root, hand_audit, _result = _build_final_manifest_fixture(tmp_path)
    rows = json.loads(hand_audit.read_text())["warm_starts"]
    mappings = [
        ((row["domain"], int(row["transformer_layers"])), Path(row["checkpoint"]))
        for row in rows
    ]
    audit = build_bc_warm_start_audit(mappings)
    assert audit["status"] == "passed"
    assert audit["terminal"] is True
    assert {row["configured_epochs"] for row in audit["warm_starts"]} == {2}


def test_final_manifest_is_exact_12_cell_two_stage_campaign(tmp_path):
    _repo, output, _run_root, _bc_audit, result = _build_final_manifest_fixture(tmp_path)
    assert result["canary_indices"] == [0, 3, 6, 9]
    for method in ("gail", "airl"):
        campaign = json.loads((output / f"{method}_final_12.json").read_text())
        assert campaign["cell_count"] == 12
        assert campaign["canonical_indices"] == list(range(12))
        assert campaign["canary_indices"] == [0, 3, 6, 9]
        assert campaign["registry_contract"]["recursive_run_root_scans_supported"] is False
        stage1 = json.loads((output / f"{method}_stage1_canary.json").read_text())
        stage2 = json.loads((output / f"{method}_stage2_canary.json").read_text())
        assert stage1["canonical_indices"] == stage2["canonical_indices"] == [0, 3, 6, 9]
        for row1, row2 in zip(stage1["trials"], stage2["trials"], strict=True):
            args1, args2 = row1["arguments"], row2["arguments"]
            assert args1["stop_after_round"] == 600
            assert args2["expected_resume_round"] == 600
            assert args2["resume_checkpoint"].endswith("stage1_rounds_0001_0600/resume_latest.pt")
            assert args1["transformer_layers"] == args2["transformer_layers"]
            assert args1["road_query_mode"] == args2["road_query_mode"] == "legacy"
            cfg_names = set(vars(PSGAILConfig()))
            cfg1 = PSGAILConfig(**{key: value for key, value in args1.items() if key in cfg_names})
            cfg2 = PSGAILConfig(**{key: value for key, value in args2.items() if key in cfg_names})
            assert resume_config_hash(cfg1) == resume_config_hash(cfg2)


def test_final_two_stage_wrapper_dry_run_records_freeze_and_dependency(tmp_path):
    repo, output, _run_root, bc_audit, _result = _build_final_manifest_fixture(tmp_path)
    canonical = tmp_path / "canonical"
    env = os.environ.copy()
    env.update(
        {
            "REPODIR": str(repo),
            "METHOD": "gail",
            "LAUNCH_PROFILE": "canary",
            "STAGE1_MANIFEST": str(output / "gail_stage1_canary.json"),
            "STAGE2_MANIFEST": str(output / "gail_stage2_canary.json"),
            "BC_AUDIT_JSON": str(bc_audit),
            "RUN_STAMP": "finaldryrun",
            "DRY_RUN": "true",
            "BLOCKING_JOB_IDS": "999999999",
            "ALLOW_TEST_SQUEUE_FIXTURE": "true",
            "PYTHON_BIN": sys.executable,
            "VFI_PROJECT_ROOT": str(canonical),
            "VFI_DATA_ROOT": str(canonical / "data"),
            "VFI_HIGHWAY_DATA_ROOT": str(canonical / "data/highway_env"),
            "VFI_CHECKPOINT_ROOT": str(canonical / "checkpoints"),
            "VFI_RESULTS_ROOT": str(canonical / "results"),
            "VFI_LOG_ROOT": str(canonical / "logs"),
            "VFI_ARTIFACT_ROOT": str(canonical / "artifacts"),
        }
    )
    wrapper = repo / "hpc/slurm/script_full_training/submit_gail_airl_final_two_stage.bash"
    completed = subprocess.run(
        ["bash", str(wrapper)], cwd=repo, env=env, check=True, capture_output=True, text=True
    )
    assert "array=0-3%2" in completed.stdout
    assert "aftercorr:DRYRUN_STAGE1" in completed.stdout
    assert "--signal=B:USR1@1800" in completed.stdout
    metadata = json.loads(
        (
            canonical
            / "results/runs/submissions/gail_airl_final_gail_canary_finaldryrun/submission_metadata.json"
        ).read_text()
    )
    assert metadata["blocking_jobs_checked"][0]["job_id"] == "999999999"
    assert metadata["dependency_mode"] == "aftercorr"


def test_checkpoint_retention_keeps_inference_payload_lean():
    from scripts_gail.train_simple_airl import airl_checkpoint_payload

    cfg = PSGAILConfig(
        action_mode="continuous", study_cell_index=0, study_stage=2, transformer_layers=2
    )
    lean = airl_checkpoint_payload(
        round_idx=800,
        policy=torch.nn.Linear(4, 2),
        reward_model=torch.nn.Linear(6, 1),
        expert_metadata={},
        cfg=cfg,
        round_cfg=cfg,
        checkpoint_kind="airl_final",
    )
    assert "training_state" not in lean
    import io

    buffer = io.BytesIO()
    torch.save(lean, buffer)
    assert buffer.tell() < 1_000_000
    assert PSGAILConfig().evaluation_max_cached_envs_per_worker == 4


def test_carried_best_is_reemitted_as_verified_stage2_official(tmp_path):
    from scripts_gail.train_simple_ps_gail import materialize_resume_best_checkpoint

    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    source_cfg = PSGAILConfig(
        scene="us-101", study_domain="us", study_cell_index=0, study_stage=1
    )
    source = stage1 / "best.pt"
    atomic_torch_save(
        {
            **checkpoint_metadata(source_cfg, method="gail", checkpoint_kind="gail_best"),
            "round": 500,
            "policy_state_dict": {"weight": torch.ones(1)},
            "training_state": {"large": "must be removed"},
            "validation_score": 2.0,
        },
        source,
    )
    resume = stage1 / "resume_latest.pt"
    atomic_torch_save({"round": 600}, resume)
    target_cfg = replace(source_cfg, study_stage=2)
    target = stage2 / "best.pt"
    assert materialize_resume_best_checkpoint(
        str(resume),
        str(target),
        best_validation_score=2.0,
        save_best_checkpoint=True,
        cfg=target_cfg,
        method="gail",
    )
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert verify_checkpoint_sidecar(target)
    assert loaded["study_cell"]["campaign_stage"] == 2
    assert loaded["study_cell"]["official"] is True
    assert "training_state" not in loaded
    assert loaded["carried_forward_from"]["checkpoint_sha256"] == verify_checkpoint_sidecar(source)
