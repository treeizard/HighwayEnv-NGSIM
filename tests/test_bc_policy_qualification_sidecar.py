from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts_gail import qualify_bc_policy_checkpoint as sidecar
from scripts_gail.ps_gail.config import PSGAILConfig


def test_development_mode_refuses_every_test_input_before_opening_data(
    monkeypatch,
    tmp_path,
):
    opened: list[Path] = []

    def forbidden_loader(*args, **kwargs):
        opened.append(Path(str(args[0])))
        raise AssertionError("No loader may run for a rejected test request.")

    monkeypatch.setattr(sidecar, "load_expert_transition_data", forbidden_loader)
    args = SimpleNamespace(
        mode="development",
        offline_test_root=tmp_path / "test",
        rollout_split=None,
        confirm_open_test=False,
    )
    with pytest.raises(ValueError, match="refuses --offline-test-root"):
        sidecar.validate_mode_contract(
            mode=args.mode,
            offline_test_root=args.offline_test_root,
            rollout_split=args.rollout_split,
            confirm_open_test=args.confirm_open_test,
        )
    assert opened == []

    with pytest.raises(ValueError, match="requires --rollout-split val"):
        sidecar.validate_mode_contract(
            mode="development",
            offline_test_root=None,
            rollout_split="test",
            confirm_open_test=False,
        )
    assert opened == []


def test_final_mode_is_disabled_before_any_test_path_access(tmp_path):
    forbidden_test = _ForbiddenTestPath()
    with pytest.raises(RuntimeError, match="Final mode is disabled"):
        sidecar.validate_mode_contract(
            mode="final",
            offline_test_root=forbidden_test,
            rollout_split="test",
            confirm_open_test=True,
        )
    assert forbidden_test.accesses == []


def test_threshold_protocol_keeps_defaults_diagnostic_and_final_fail_closed():
    diagnostic = sidecar.threshold_protocol_contract(
        mode="development",
        protocol_id="",
        thresholds_frozen=False,
    )
    assert diagnostic == {
        "protocol_id": None,
        "thresholds_frozen": False,
        "qualification_eligible": False,
        "status": "diagnostic_thresholds_unfrozen",
    }
    with pytest.raises(ValueError, match="Final mode requires"):
        sidecar.threshold_protocol_contract(
            mode="final",
            protocol_id="",
            thresholds_frozen=False,
        )
    with pytest.raises(ValueError, match="threshold-protocol-id"):
        sidecar.threshold_protocol_contract(
            mode="development",
            protocol_id="",
            thresholds_frozen=True,
        )
    frozen = sidecar.threshold_protocol_contract(
        mode="final",
        protocol_id="protocol-sha256:abc",
        thresholds_frozen=True,
    )
    assert frozen["qualification_eligible"] is True
    assert frozen["status"] == "prespecified_frozen"


def test_threshold_values_reject_shortcuts_and_nonfinite_values():
    valid = {
        "minimum_validation_skill": 0.1,
        "maximum_offline_mae": 0.35,
        "minimum_action_std_ratios": [0.25, 0.1],
        "minimum_action_correlations": [0.5, 0.2],
        "maximum_primary_crash_rate": 1.0,
        "maximum_primary_offroad_rate": 1.0,
        "minimum_horizon_coverage": 0.0,
        "maximum_expert_relative_crash_gap": 0.05,
        "maximum_expert_relative_offroad_gap": 0.05,
    }
    sidecar.validate_threshold_values(**valid)
    with pytest.raises(ValueError, match="cannot be negative"):
        sidecar.validate_threshold_values(
            **{**valid, "maximum_offline_mae": -1.0}
        )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        sidecar.validate_threshold_values(
            **{**valid, "maximum_expert_relative_crash_gap": 1.1}
        )
    with pytest.raises(ValueError, match="must be finite"):
        sidecar.validate_threshold_values(
            **{**valid, "minimum_validation_skill": float("nan")}
        )


def test_vacuous_closed_loop_thresholds_are_diagnostic_only():
    vacuous = sidecar.threshold_discrimination_contract(
        maximum_primary_crash_rate=1.0,
        maximum_primary_offroad_rate=1.0,
        minimum_horizon_coverage=0.0,
    )
    assert vacuous["passed"] is False
    assert set(vacuous["failures"]) == {
        "maximum_primary_crash_rate_is_vacuous",
        "maximum_primary_offroad_rate_is_vacuous",
        "minimum_horizon_coverage_is_vacuous",
    }
    discriminating = sidecar.threshold_discrimination_contract(
        maximum_primary_crash_rate=0.5,
        maximum_primary_offroad_rate=0.5,
        minimum_horizon_coverage=0.5,
    )
    assert discriminating["passed"] is True
    assert discriminating["failures"] == []


def _checkpoint_payload() -> dict[str, object]:
    return {
        "checkpoint_kind": "behaviour_cloning_best",
        "policy_state_dict": {"weight": torch.ones(1)},
        "config": {"policy_model": "recurrent_transformer"},
        "policy_architecture": {
            "policy_model": "recurrent_transformer",
            "obs_dim": 322,
            "hidden_size": 256,
            "action_mode": "continuous",
            "continuous_action_dim": 2,
        },
        "bc_stats": {
            "checkpoint_selection_rule": "validation_loss",
            "test_evaluation_mode": "deferred",
        },
    }


def _fully_declared_checkpoint_payload() -> dict[str, object]:
    behaviour = {
        "policy_model": "recurrent_transformer",
        "hidden_size": 256,
        "action_mode": "continuous",
        "continuous_action_dim": 2,
        "transformer_layers": 2,
        "transformer_heads": 4,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "policy_observation_standardization_clip": 0.0,
        "transformer_observation_tokenization": "dense_temporal",
        "policy_head_init_std": 0.01,
        "transformer_memory_tokens": 1,
        "transformer_memory_context_length": 32,
        "transformer_use_causal_attention": True,
    }
    payload = _checkpoint_payload()
    payload["config"] = {
        **vars(PSGAILConfig()),
        **behaviour,
    }
    payload["policy_architecture"] = {
        "obs_dim": 322,
        **behaviour,
    }
    return payload


def test_checkpoint_load_requires_explicit_and_stored_matching_hashes(tmp_path):
    checkpoint = tmp_path / "best.pt"
    torch.save(_checkpoint_payload(), checkpoint)
    digest = sidecar.sha256_file(checkpoint)
    (tmp_path / "best.pt.sha256").write_text(
        f"{digest}  best.pt\n",
        encoding="utf-8",
    )

    payload, record = sidecar.load_verified_checkpoint(
        checkpoint,
        expected_sha256=digest,
    )

    assert payload["checkpoint_kind"] == "behaviour_cloning_best"
    assert record["sha256"] == digest
    assert record["policy_model"] == "recurrent_transformer"
    assert record["policy_observation_standardization_clip"] == 5.0
    assert (
        record["policy_observation_standardization_clip_provenance"]
        == "legacy_missing_field_replayed_as_5"
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        sidecar.load_verified_checkpoint(
            checkpoint,
            expected_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("selection_rule", "test_mode", "message"),
    [
        ("test_loss", "deferred", "validation-only checkpoint"),
        ("validation_loss", "evaluate", "kept test evaluation deferred"),
    ],
)
def test_checkpoint_rejects_test_in_selection_history(
    tmp_path,
    selection_rule,
    test_mode,
    message,
):
    checkpoint = tmp_path / "best.pt"
    payload = _checkpoint_payload()
    payload["bc_stats"]["checkpoint_selection_rule"] = selection_rule
    payload["bc_stats"]["test_evaluation_mode"] = test_mode
    torch.save(payload, checkpoint)
    digest = sidecar.sha256_file(checkpoint)
    (tmp_path / "best.pt.sha256").write_text(
        f"{digest}  best.pt\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        sidecar.load_verified_checkpoint(
            checkpoint,
            expected_sha256=digest,
        )


@pytest.mark.parametrize(
    "selection_rule",
    ["validation_loss", "qualification_then_loss"],
)
def test_checkpoint_accepts_supported_validation_only_selection_rules(
    tmp_path,
    selection_rule,
):
    checkpoint = tmp_path / "best.pt"
    payload = _checkpoint_payload()
    payload["bc_stats"]["checkpoint_selection_rule"] = selection_rule
    torch.save(payload, checkpoint)
    digest = sidecar.sha256_file(checkpoint)
    (tmp_path / "best.pt.sha256").write_text(
        f"{digest}  best.pt\n",
        encoding="utf-8",
    )

    _payload, record = sidecar.load_verified_checkpoint(
        checkpoint,
        expected_sha256=digest,
    )

    assert record["checkpoint_selection_rule"] == selection_rule
    assert record["selection_data_role"] == "validation_only"


def test_full_deferred_qualification_then_loss_history_is_accepted():
    receipt = sidecar.validate_checkpoint_selection_contract(
        {
            "checkpoint_selection_rule": "qualification_then_loss",
            "checkpoint_selection_metric": (
                "weighted_mean_validation_action_mse"
            ),
            "test_evaluation_mode": "deferred",
            "offline_test_evaluated": False,
            "offline_test_status": "pending_deferred",
            "test_mse": None,
            "test_mae": None,
            "test_action_mse": None,
            "test_action_mae": None,
            "test_prediction_std": None,
            "test_target_std": None,
            "test_prediction_std_ratio": None,
            "test_prediction_target_correlation": None,
            "test_prediction_saturation_fraction": None,
            "test_samples": None,
            "test_windows": None,
            "held_out_learning_signal_passed": False,
            "held_out_metric_capability_passed": False,
            "held_out_learning_signal_gate": {
                "split": "test",
                "status": "pending_deferred",
                "passed": False,
                "actions": [],
            },
            "final_test_qualification_status": "pending_deferred",
            "final_test_qualification_passed": False,
            "selected_history_row": {
                "validation_mse": 0.1,
                "selection_eligible": True,
                "checkpoint_selection_rule": (
                    "qualification_then_loss"
                ),
            },
        }
    )

    assert receipt["selection_data_role"] == "validation_only"
    assert receipt["checkpoint_selection_rule"] == (
        "qualification_then_loss"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("test_mse", 0.1, "populated test outcomes"),
        ("offline_test_evaluated", True, "offline test split"),
        (
            "final_test_qualification_status",
            "evaluated",
            "pending_deferred",
        ),
    ],
)
def test_checkpoint_rejects_deferred_label_with_test_outcomes(
    tmp_path,
    field,
    value,
    message,
):
    checkpoint = tmp_path / "best.pt"
    payload = _checkpoint_payload()
    payload["bc_stats"][field] = value
    torch.save(payload, checkpoint)
    digest = sidecar.sha256_file(checkpoint)
    (tmp_path / "best.pt.sha256").write_text(
        f"{digest}  best.pt\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        sidecar.load_verified_checkpoint(
            checkpoint,
            expected_sha256=digest,
        )


def test_checkpoint_rejects_test_key_in_selected_history(tmp_path):
    checkpoint = tmp_path / "best.pt"
    payload = _checkpoint_payload()
    payload["bc_stats"]["selected_history_row"] = {
        "validation_mse": 0.1,
        "test_mse": 0.2,
    }
    torch.save(payload, checkpoint)
    digest = sidecar.sha256_file(checkpoint)
    (tmp_path / "best.pt.sha256").write_text(
        f"{digest}  best.pt\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test-informed.*history"):
        sidecar.load_verified_checkpoint(
            checkpoint,
            expected_sha256=digest,
        )


def test_evaluation_config_enables_collision_and_keeps_single_ego_primary(
    monkeypatch,
    tmp_path,
):
    payload = _checkpoint_payload()
    payload["config"] = {
        **vars(PSGAILConfig()),
        "policy_model": "recurrent_transformer",
        "action_mode": "continuous",
        "continuous_action_dim": 2,
        "hidden_size": 256,
        "transformer_layers": 2,
        "transformer_heads": 4,
        "enable_collision": False,
    }
    payload["policy_architecture"].update(
        {
            "transformer_layers": 2,
            "transformer_heads": 4,
        }
    )
    sentinel_policy = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        sidecar,
        "_make_policy_from_state_dict",
        lambda *args, **kwargs: sentinel_policy,
    )

    cfg, policy, obs_dim, action_dim = (
        sidecar.config_and_policy_from_checkpoint(
            payload,
            episode_root=tmp_path,
            rollout_split="val",
            device=torch.device("cpu"),
            evaluation_workers=1,
        )
    )

    assert policy is sentinel_policy
    assert (obs_dim, action_dim) == (322, 2)
    assert cfg.enable_collision is True
    assert cfg.evaluation_terminate_when_all_controlled_crashed is False
    assert cfg.validation_vehicle_mode == "single"
    assert cfg.validation_control_all_vehicles is False
    assert cfg.action_mode == "continuous"


def test_fully_declared_behaviour_contract_is_typed_equal():
    payload = _fully_declared_checkpoint_payload()

    resolved = sidecar.reconcile_checkpoint_behaviour_config(
        payload["config"],
        payload["policy_architecture"],
    )

    assert set(sidecar.CHECKPOINT_BEHAVIOUR_FIELD_TYPES) == {
        "policy_model",
        "hidden_size",
        "action_mode",
        "continuous_action_dim",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "transformer_norm_first",
        "transformer_observation_normalization",
        "policy_observation_standardization_clip",
        "transformer_observation_tokenization",
        "policy_head_init_std",
        "transformer_memory_tokens",
        "transformer_memory_context_length",
        "transformer_use_causal_attention",
    }
    for field, expected_type in (
        sidecar.CHECKPOINT_BEHAVIOUR_FIELD_TYPES.items()
    ):
        assert type(resolved[field]) is expected_type
        assert resolved[field] == payload["policy_architecture"][field]


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("policy_model", "recurrent_gru"),
        ("hidden_size", 128),
        ("action_mode", "discrete"),
        ("continuous_action_dim", 3),
        ("transformer_layers", 3),
        ("transformer_heads", 8),
        ("transformer_dropout", 0.1),
        ("transformer_norm_first", False),
        ("transformer_observation_normalization", False),
        ("policy_observation_standardization_clip", 5.0),
        ("transformer_observation_tokenization", "semantic"),
        ("policy_head_init_std", 0.02),
        ("transformer_memory_tokens", 2),
        ("transformer_memory_context_length", 16),
        ("transformer_use_causal_attention", False),
    ],
)
def test_every_duplicated_behaviour_mutation_fails_before_policy_construction(
    monkeypatch,
    tmp_path,
    field,
    mutation,
):
    payload = _fully_declared_checkpoint_payload()
    payload["policy_architecture"][field] = mutation
    policy_constructor_called = False

    def forbidden_policy_constructor(*args, **kwargs):
        nonlocal policy_constructor_called
        policy_constructor_called = True
        raise AssertionError("A mismatched checkpoint must not construct a policy.")

    monkeypatch.setattr(
        sidecar,
        "_make_policy_from_state_dict",
        forbidden_policy_constructor,
    )

    with pytest.raises(ValueError, match=field):
        sidecar.config_and_policy_from_checkpoint(
            payload,
            episode_root=tmp_path,
            rollout_split="val",
            device=torch.device("cpu"),
            evaluation_workers=1,
        )

    assert policy_constructor_called is False


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("policy_model", _StringSubclass("recurrent_transformer")),
        ("hidden_size", 256.0),
        ("transformer_dropout", 0),
        ("transformer_norm_first", 1),
    ],
)
def test_duplicate_behaviour_values_require_exact_primitive_types(
    field,
    mutation,
):
    payload = _fully_declared_checkpoint_payload()
    payload["policy_architecture"][field] = mutation

    with pytest.raises(ValueError, match=field):
        sidecar.reconcile_checkpoint_behaviour_config(
            payload["config"],
            payload["policy_architecture"],
        )


def test_verified_checkpoint_rejects_duplicate_mismatch_at_load_boundary(
    tmp_path,
):
    checkpoint = tmp_path / "best.pt"
    payload = _fully_declared_checkpoint_payload()
    payload["policy_architecture"][
        "transformer_memory_context_length"
    ] = 16
    torch.save(payload, checkpoint)
    digest = sidecar.sha256_file(checkpoint)
    (tmp_path / "best.pt.sha256").write_text(
        f"{digest}  best.pt\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="transformer_memory_context_length",
    ):
        sidecar.load_verified_checkpoint(
            checkpoint,
            expected_sha256=digest,
        )


def _identity(root: Path, *episode_ids: str) -> dict[str, object]:
    return {
        "root": str(root),
        "root_identity": sidecar._lexical_root_identity(root),
        "metadata_status": "synthetic_test_identity",
        "canonical_episode_ids": list(episode_ids),
    }


def _checkpoint_with_source_identities() -> dict[str, object]:
    return {
        "training_data_contract": {
            "source_root_integrity": {
                "sources": {
                    "train": {
                        "root": "/recorded/train",
                        "canonical_episode_ids": ["scene/train-episode"],
                    },
                    "validation": {
                        "root": "/recorded/validation",
                        "canonical_episode_ids": [
                            "scene/validation-episode"
                        ],
                    },
                }
            }
        }
    }


def test_split_identity_contract_accepts_exact_validation_alias():
    receipt = sidecar.validate_split_identity_contract(
        _checkpoint_with_source_identities(),
        supplied_validation=_identity(
            Path("/copied/validation"),
            "scene/validation-episode",
        ),
        supplied_test=_identity(
            Path("/supplied/test"),
            "scene/test-episode",
        ),
    )

    assert receipt["status"] == "passed"
    assert receipt["promotion_eligible"] is True
    validation_pair = receipt["pairwise"][
        "checkpoint_validation_vs_supplied_validation"
    ]
    assert validation_pair["relationship"] == (
        "equivalent_validation_episode_identity"
    )


def test_split_identity_contract_keeps_root_only_sources_diagnostic():
    checkpoint = {
        "expert_data": {
            "train": {"path": "/recorded/train"},
            "validation": {"path": "/recorded/validation"},
        }
    }
    receipt = sidecar.validate_split_identity_contract(
        checkpoint,
        supplied_validation={
            "root": "/supplied/validation",
            "root_identity": sidecar._lexical_root_identity(
                "/supplied/validation"
            ),
            "metadata_status": "manifest_not_available",
            "canonical_episode_ids": None,
        },
        supplied_test=None,
    )

    assert receipt["promotion_eligible"] is False
    assert receipt["status"] == (
        "diagnostic_incomplete_canonical_identity"
    )
    assert set(receipt["missing_required_canonical_identities"]) == {
        "checkpoint_train",
        "checkpoint_validation",
        "supplied_validation",
    }


@pytest.mark.parametrize(
    ("validation_identity", "test_identity", "message"),
    [
        (
            _identity(
                Path("/supplied/validation"),
                "scene/train-episode",
            ),
            None,
            "checkpoint_train/supplied_validation",
        ),
        (
            _identity(
                Path("/supplied/validation"),
                "scene/validation-episode",
            ),
            _identity(
                Path("/supplied/test"),
                "scene/validation-episode",
            ),
            "checkpoint_validation/supplied_test",
        ),
        (
            _identity(
                Path("/same/root"),
                "scene/validation-episode",
            ),
            _identity(
                Path("/same/root"),
                "scene/test-episode",
            ),
            "supplied_test/supplied_validation|supplied_validation/supplied_test",
        ),
    ],
)
def test_split_identity_contract_rejects_root_or_episode_overlap(
    validation_identity,
    test_identity,
    message,
):
    with pytest.raises(ValueError, match=message):
        sidecar.validate_split_identity_contract(
            _checkpoint_with_source_identities(),
            supplied_validation=validation_identity,
            supplied_test=test_identity,
        )


def test_root_manifest_identity_uses_only_synthetic_metadata(tmp_path):
    root = tmp_path / "validation"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "scene": "scene",
                "prebuilt_split": "val",
                "episodes": [
                    {"episode_name": "episode-a"},
                    {"episode_name": "episode-b"},
                ],
            }
        ),
        encoding="utf-8",
    )

    identity = sidecar.root_manifest_identity(
        root,
        expected_split="val",
    )

    assert identity["canonical_episode_ids"] == [
        "scene/episode-a",
        "scene/episode-b",
    ]
    with pytest.raises(ValueError, match="split mismatch"):
        sidecar.root_manifest_identity(
            root,
            expected_split="test",
        )


def test_expert_relative_gate_uses_policy_minus_same_scenario_expert():
    policy_cases = [
        {
            "episode_name": f"episode-{index}",
            "ego_vehicle_id": index,
            "reset_seed": 1000 + index,
            "first_collision_step": 10 if index < 3 else None,
            "first_offroad_step": 12 if index < 2 else None,
        }
        for index in range(24)
    ]
    expert_cases = [
        {
            **row,
            "first_collision_step": 10 if index < 2 else None,
            "first_offroad_step": 12 if index < 1 else None,
        }
        for index, row in enumerate(policy_cases)
    ]
    gate = sidecar.expert_relative_gate(
        {
            "policy/vehicle_crash_rate": 3 / 24,
            "policy/vehicle_offroad_rate": 2 / 24,
            "policy/vehicle_episodes": 24.0,
            "episodes": policy_cases,
        },
        {
            "expert/vehicle_crash_rate": 2 / 24,
            "expert/vehicle_offroad_rate": 1 / 24,
            "vehicle_episodes": 24,
            "episodes": expert_cases,
        },
        policy_prefix="policy",
        expert_prefix="expert",
        maximum_crash_rate_gap=0.20,
        maximum_offroad_rate_gap=0.20,
    )

    assert gate["observed"]["vehicle_crash_rate_gap"] == pytest.approx(1 / 24)
    assert gate["observed"]["vehicle_offroad_rate_gap"] == pytest.approx(1 / 24)
    assert gate["paired_outcomes"]["scenario_count"] == 24
    assert gate["passed"] is True


def test_expert_relative_gate_uses_paired_uncertainty_not_aggregate_tie():
    policy_cases = [
        {
            "episode_name": f"episode-{index}",
            "ego_vehicle_id": index,
            "reset_seed": 1000 + index,
            "first_collision_step": 10 if index == 0 else None,
            "first_offroad_step": None,
        }
        for index in range(24)
    ]
    expert_cases = [
        {
            **row,
            "first_collision_step": 10 if index == 1 else None,
        }
        for index, row in enumerate(policy_cases)
    ]
    gate = sidecar.expert_relative_gate(
        {
            "policy/vehicle_crash_rate": 1 / 24,
            "policy/vehicle_offroad_rate": 0.0,
            "policy/vehicle_episodes": 24.0,
            "episodes": policy_cases,
        },
        {
            "expert/vehicle_crash_rate": 1 / 24,
            "expert/vehicle_offroad_rate": 0.0,
            "evaluated_episodes": 24,
            "episodes": expert_cases,
        },
        policy_prefix="policy",
        expert_prefix="expert",
        maximum_crash_rate_gap=0.05,
        maximum_offroad_rate_gap=0.05,
    )

    assert gate["observed"]["vehicle_crash_rate_gap"] == pytest.approx(0.0)
    assert gate["checks"]["maximum_vehicle_crash_rate_gap"]
    assert not gate["checks"]["paired_crash_gap_upper"]
    assert not gate["passed"]


def test_expert_relative_gate_fails_closed_on_scenario_identity_mismatch():
    policy_case = {
        "episode_name": "episode-a",
        "ego_vehicle_id": 1,
        "reset_seed": 1001,
        "first_collision_step": None,
        "first_offroad_step": None,
    }
    expert_case = {**policy_case, "reset_seed": 1002}
    gate = sidecar.expert_relative_gate(
        {
            "policy/vehicle_crash_rate": 0.0,
            "policy/vehicle_offroad_rate": 0.0,
            "policy/vehicle_episodes": 1.0,
            "episodes": [policy_case],
        },
        {
            "expert/vehicle_crash_rate": 0.0,
            "expert/vehicle_offroad_rate": 0.0,
            "evaluated_episodes": 1,
            "episodes": [expert_case],
        },
        policy_prefix="policy",
        expert_prefix="expert",
        maximum_crash_rate_gap=0.05,
        maximum_offroad_rate_gap=0.05,
        minimum_paired_scenarios=2,
    )

    assert not gate["checks"]["paired_case_identities_match"]
    assert not gate["passed"]


class _ForbiddenTestPath:
    def __init__(self):
        self.accesses: list[str] = []

    def _forbid(self, operation: str):
        self.accesses.append(operation)
        raise AssertionError(
            f"Test path was accessed before validation: {operation}"
        )

    def resolve(self):
        return self._forbid("resolve")

    def is_dir(self):
        return self._forbid("is_dir")

    def __fspath__(self):
        return self._forbid("__fspath__")


def _synthetic_qualification_args(
    tmp_path,
    *,
    mode: str,
    offline_test_root,
) -> SimpleNamespace:
    validation_root = tmp_path / "validation"
    episode_root = tmp_path / "episodes"
    validation_root.mkdir(exist_ok=True)
    episode_root.mkdir(exist_ok=True)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic immutable checkpoint")
    return SimpleNamespace(
        mode=mode,
        offline_test_root=offline_test_root,
        rollout_split=None,
        confirm_open_test=mode == "final",
        threshold_protocol_id="synthetic-frozen-protocol",
        thresholds_frozen=True,
        episodes=2,
        all_vehicle_stress_episodes=0,
        max_offline_samples=10,
        evaluation_workers=1,
        receipt=tmp_path / f"{mode}-receipt.json",
        minimum_action_std_ratios="0.25,0.10",
        minimum_action_correlations="0.50,0.20",
        minimum_validation_skill=0.1,
        maximum_offline_mae=0.35,
        maximum_primary_crash_rate=0.5,
        maximum_primary_offroad_rate=0.5,
        minimum_horizon_coverage=0.5,
        maximum_expert_relative_crash_gap=0.05,
        maximum_expert_relative_offroad_gap=0.05,
        minimum_paired_scenarios=2,
        paired_bootstrap_replicates=500,
        paired_upper_confidence_quantile=0.95,
        offline_validation_root=validation_root,
        episode_root=episode_root,
        checkpoint=checkpoint,
        checkpoint_sha256=sidecar.sha256_file(checkpoint),
        device="cpu",
        offline_seed=7,
    )


def _patch_synthetic_qualification_dependencies(
    monkeypatch,
    args,
    *,
    validation_closed_loop_passed: bool,
) -> None:
    checkpoint_digest = sidecar.sha256_file(args.checkpoint)
    payload = {
        "bc_stats": {
            "validation_baseline_mse": 1.0,
        },
        "policy_output_action_contract": {},
        "policy_observation_contract": {},
    }
    monkeypatch.setattr(
        sidecar,
        "source_records",
        lambda: {"source": {"sha256": "a" * 64}},
    )
    monkeypatch.setattr(
        sidecar,
        "tree_hash_record",
        lambda root: {
            "path": str(Path(root)),
            "sha256_tree": "b" * 64,
        },
    )
    monkeypatch.setattr(
        sidecar,
        "load_verified_checkpoint",
        lambda *args, **kwargs: (
            payload,
            {
                "path": str(args),
                "sha256": checkpoint_digest,
            },
        ),
    )
    cfg = PSGAILConfig(
        scene="us-101",
        action_mode="continuous",
        continuous_action_dim=2,
        enable_collision=True,
        evaluation_terminate_when_all_controlled_crashed=False,
    )
    monkeypatch.setattr(
        sidecar,
        "config_and_policy_from_checkpoint",
        lambda *args, **kwargs: (
            cfg,
            torch.nn.Linear(1, 1),
            322,
            2,
        ),
    )
    monkeypatch.setattr(
        sidecar,
        "rollout_data_records",
        lambda **kwargs: {
            "split": kwargs["split"],
            "pair_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        sidecar,
        "evaluate_offline_root",
        lambda *args, **kwargs: {"passed": True},
    )

    def closed_loop(*args, **kwargs):
        split = kwargs["split"]
        passed = (
            validation_closed_loop_passed
            if split == "val"
            else True
        )
        scenarios = [(f"{split}-episode", 1)]
        return {
            "split": split,
            "scenario_count": 1,
            "scenarios": scenarios,
            "scenario_sha256": sidecar.canonical_sha256(scenarios),
            "policy_metrics": {},
            "expert_replay_metrics": {},
            "absolute_quality": {"passed": passed},
            "expert_relative_quality": {"passed": passed},
            "passed": passed,
            "all_vehicle_stress": {
                "status": "not_requested",
                "role": "diagnostic_non_promotion_gate",
            },
        }

    monkeypatch.setattr(
        sidecar,
        "evaluate_closed_loop_split",
        closed_loop,
    )


def test_invalid_checkpoint_cannot_access_final_test_path(
    monkeypatch,
    tmp_path,
):
    forbidden_test = _ForbiddenTestPath()
    args = _synthetic_qualification_args(
        tmp_path,
        mode="final",
        offline_test_root=forbidden_test,
    )
    monkeypatch.setattr(
        sidecar,
        "source_records",
        lambda: {"source": {"sha256": "a" * 64}},
    )
    monkeypatch.setattr(
        sidecar,
        "tree_hash_record",
        lambda root: {"path": str(root), "sha256_tree": "b" * 64},
    )
    monkeypatch.setattr(
        sidecar,
        "load_verified_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic invalid checkpoint")
        ),
    )

    with pytest.raises(RuntimeError, match="Final mode is disabled"):
        sidecar.run_qualification(args)

    assert forbidden_test.accesses == []


def test_failed_validation_cannot_access_final_test_path(
    monkeypatch,
    tmp_path,
):
    forbidden_test = _ForbiddenTestPath()
    args = _synthetic_qualification_args(
        tmp_path,
        mode="final",
        offline_test_root=forbidden_test,
    )
    _patch_synthetic_qualification_dependencies(
        monkeypatch,
        args,
        validation_closed_loop_passed=False,
    )

    with pytest.raises(RuntimeError, match="Final mode is disabled"):
        sidecar.run_qualification(args)

    assert forbidden_test.accesses == []


def test_vacuous_thresholds_cannot_access_final_test_path(
    monkeypatch,
    tmp_path,
):
    forbidden_test = _ForbiddenTestPath()
    args = _synthetic_qualification_args(
        tmp_path,
        mode="final",
        offline_test_root=forbidden_test,
    )
    args.maximum_primary_crash_rate = 1.0
    args.maximum_primary_offroad_rate = 1.0
    args.minimum_horizon_coverage = 0.0
    _patch_synthetic_qualification_dependencies(
        monkeypatch,
        args,
        validation_closed_loop_passed=True,
    )

    with pytest.raises(RuntimeError, match="Final mode is disabled"):
        sidecar.run_qualification(args)

    assert forbidden_test.accesses == []


def test_development_rejects_test_object_without_inspecting_it():
    forbidden_test = _ForbiddenTestPath()

    with pytest.raises(ValueError, match="refuses --offline-test-root"):
        sidecar.validate_mode_contract(
            mode="development",
            offline_test_root=forbidden_test,
            rollout_split=None,
            confirm_open_test=False,
        )

    assert forbidden_test.accesses == []


def test_final_candidate_cannot_claim_publication_qualification(
    monkeypatch,
    tmp_path,
):
    forbidden_test = _ForbiddenTestPath()
    args = _synthetic_qualification_args(
        tmp_path,
        mode="final",
        offline_test_root=forbidden_test,
    )

    with pytest.raises(RuntimeError, match="Final mode is disabled"):
        sidecar.run_qualification(args)

    assert forbidden_test.accesses == []


def test_manifestless_development_run_is_diagnostic_not_qualified(
    monkeypatch,
    tmp_path,
):
    args = _synthetic_qualification_args(
        tmp_path,
        mode="development",
        offline_test_root=None,
    )
    _patch_synthetic_qualification_dependencies(
        monkeypatch,
        args,
        validation_closed_loop_passed=True,
    )

    result = sidecar.run_qualification(args)

    identity = result["data"]["split_identity_contract"]
    assert identity["promotion_eligible"] is False
    assert result["qualification"]["validation_sidecar_passed"] is False
    assert result["qualification"]["test_access_preconditions"][
        "validation_identity_canonical"
    ] is False
    assert result["status"] == (
        "diagnostic_incomplete_canonical_identity_test_locked"
    )


def test_unfrozen_development_run_reports_metrics_but_cannot_pass(
    monkeypatch,
    tmp_path,
):
    validation_root = tmp_path / "validation"
    episode_root = tmp_path / "episodes"
    validation_root.mkdir()
    episode_root.mkdir()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"immutable checkpoint")
    checkpoint_digest = sidecar.sha256_file(checkpoint)
    receipt = tmp_path / "receipt.json"
    args = SimpleNamespace(
        mode="development",
        offline_test_root=None,
        rollout_split=None,
        confirm_open_test=False,
        threshold_protocol_id="",
        thresholds_frozen=False,
        episodes=2,
        all_vehicle_stress_episodes=0,
        max_offline_samples=10,
        evaluation_workers=1,
        receipt=receipt,
        minimum_action_std_ratios="0.25,0.10",
        minimum_action_correlations="0.50,0.20",
        minimum_validation_skill=0.1,
        maximum_offline_mae=0.35,
        maximum_primary_crash_rate=1.0,
        maximum_primary_offroad_rate=1.0,
        minimum_horizon_coverage=0.0,
        maximum_expert_relative_crash_gap=0.05,
        maximum_expert_relative_offroad_gap=0.05,
        minimum_paired_scenarios=2,
        paired_bootstrap_replicates=500,
        paired_upper_confidence_quantile=0.95,
        offline_validation_root=validation_root,
        episode_root=episode_root,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_digest,
        device="cpu",
        offline_seed=7,
    )
    payload = {
        "bc_stats": {
            "validation_baseline_mse": 1.0,
        },
        "policy_output_action_contract": {},
        "policy_observation_contract": {},
    }
    monkeypatch.setattr(
        sidecar,
        "source_records",
        lambda: {"source": {"sha256": "a" * 64}},
    )
    monkeypatch.setattr(
        sidecar,
        "tree_hash_record",
        lambda root: {
            "path": str(Path(root)),
            "sha256_tree": "b" * 64,
        },
    )
    monkeypatch.setattr(
        sidecar,
        "load_verified_checkpoint",
        lambda *args, **kwargs: (
            payload,
            {
                "path": str(checkpoint),
                "sha256": checkpoint_digest,
            },
        ),
    )
    cfg = PSGAILConfig(
        scene="us-101",
        action_mode="continuous",
        continuous_action_dim=2,
        enable_collision=True,
        evaluation_terminate_when_all_controlled_crashed=False,
    )
    monkeypatch.setattr(
        sidecar,
        "config_and_policy_from_checkpoint",
        lambda *args, **kwargs: (
            cfg,
            torch.nn.Linear(1, 1),
            322,
            2,
        ),
    )
    monkeypatch.setattr(
        sidecar,
        "rollout_data_records",
        lambda **kwargs: {"pair_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        sidecar,
        "evaluate_offline_root",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        sidecar,
        "_evaluation_scenarios",
        lambda *args, **kwargs: [("episode", 1), ("episode", 2)],
    )
    policy_metrics = {
        "validation_primary/vehicle_crash_rate": 0.0,
        "validation_primary/vehicle_offroad_rate": 0.0,
        "validation_primary/vehicle_episodes": 2.0,
        "validation_primary/horizon_coverage_20s": 1.0,
        "episodes": [
            {
                "episode_name": f"episode-{index}",
                "ego_vehicle_id": index + 1,
                "reset_seed": 1000 + index,
                "first_collision_step": None,
                "first_offroad_step": None,
            }
            for index in range(2)
        ],
    }
    monkeypatch.setattr(
        sidecar,
        "_evaluate_policy_matched_trajectories_impl",
        lambda *args, **kwargs: policy_metrics,
    )
    monkeypatch.setattr(
        sidecar,
        "evaluate_expert_replay_matched_single_vehicle_floor",
        lambda *args, **kwargs: {
            "validation_primary_expert/vehicle_crash_rate": 0.0,
            "validation_primary_expert/vehicle_offroad_rate": 0.0,
            "validation_primary_expert/horizon_coverage_20s": 1.0,
            "evaluated_episodes": 2,
            "episodes": [
                {
                    "episode_name": f"episode-{index}",
                    "ego_vehicle_id": index + 1,
                    "reset_seed": 1000 + index,
                    "first_collision_step": None,
                    "first_offroad_step": None,
                }
                for index in range(2)
            ],
        },
    )
    monkeypatch.setattr(
        sidecar,
        "clear_evaluation_worker_caches",
        lambda: None,
    )

    result = sidecar.run_qualification(args)

    assert result["primary_closed_loop"]["passed"] is True
    assert result["threshold_protocol"]["status"] == (
        "diagnostic_thresholds_unfrozen"
    )
    assert result["qualification"]["validation_sidecar_passed"] is False
    assert result["qualification"]["policy_realism_qualified"] is False
    assert result["status"] == "diagnostic_thresholds_unfrozen_test_locked"
    assert result["runtime_contract"][
        "evaluator_pre_env_action_clamping"
    ] is False
    assert result["runtime_contract"][
        "environment_continuous_action_clip_configured"
    ] is False
    assert "passed unchanged" in result["runtime_contract"][
        "environment_continuous_action_clip_role"
    ]


def test_receipt_is_fresh_hashed_and_never_overwritten(tmp_path):
    receipt = tmp_path / "qualification.json"
    payload = {"schema_version": 1, "status": "failed_closed"}

    written = sidecar.write_receipt(receipt, payload)

    expected = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert written["receipt_sha256"] == expected
    assert (
        receipt.with_name("qualification.json.sha256")
        .read_text(encoding="utf-8")
        .split()
        == [expected, "qualification.json"]
    )
    assert json.loads(receipt.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="overwrite"):
        sidecar.write_receipt(receipt, payload)


def test_standalone_expert_baseline_defaults_to_validation():
    source = (
        Path(sidecar.__file__).resolve().with_name(
            "audit_expert_replay_collision_baseline.py"
        )
    ).read_text(encoding="utf-8")

    assert (
        'choices=["train", "val", "test"], default="val"'
        in source
    )


def test_matched_single_ego_expert_floor_is_validation_only_and_pairable():
    source = (
        Path(sidecar.__file__).resolve().with_name(
            "audit_matched_single_ego_expert_floor.py"
        )
    ).read_text(encoding="utf-8")

    assert 'choices=["train", "val"], default="val"' in source
    assert '"test_split_opened": False' in source
    assert '"vehicle_mode": "single_requested_ego"' in source
    assert "evaluate_expert_replay_matched_single_vehicle_floor" in source
