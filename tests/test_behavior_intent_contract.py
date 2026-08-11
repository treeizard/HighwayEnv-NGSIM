from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from highway_env.imitation.behavior_intent import (
    BehaviorCommandSchedulerConfig,
    BehaviorFeasibility,
    BehaviorIntent,
    BehaviorLabelerConfig,
    ProspectiveBehaviorCommandScheduler,
    behavior_intent_contract,
    behavior_intent_one_hot,
    label_behavior_rows,
)

from policy.contracts.imitation import policy_observation_contract
from policy.contracts.observations import policy_observations_from_flat
from policy.contracts.training_config import PSGAILConfig
from policy.data.audit_behavior_intent_sidecars import (
    audit_root,
    independently_evaluate_support,
    reference_labels,
)
from policy.data.behavior_sidecars import (
    build_behavior_sidecar,
    load_behavior_sidecar,
    source_trajectory_ids,
)
from policy.data.build_behavior_intent_sidecars import evaluate_support
from policy.data.expert import load_expert_transition_data
from policy.evaluation.behavior_observability import (
    _features,
    _segment_shuffle,
    _vehicle_bootstrap_lower,
)
from policy.evaluation.checkpoints import canonical_tensor_state_sha256
from policy.evaluation.method_health import (
    build_gail,
    build_iq,
)
from policy.evaluation.qualify_behavior import (
    actor_parameter_delta,
    method_health_checks,
    primary_recipe_checks,
)
from policy.methods.bc import (
    SequenceWindow,
    class_uniform_behavior_windows,
    mirror_policy_observations,
    transition_indices_from_behavior_sampling_manifest,
    windows_from_behavior_sampling_manifest,
)
from policy.methods.gail.train import (
    match_expert_behavior_distribution,
    validate_primary_behavior_gail_recipe,
)


def test_stable_ids_and_326_field_path_free_contract():
    assert [int(value) for value in BehaviorIntent] == [0, 1, 2, 3]
    features = behavior_intent_one_hot(np.arange(4, dtype=np.int8))
    assert features.dtype == np.float32
    assert np.array_equal(features, np.eye(4, dtype=np.float32))

    contract = policy_observation_contract(
        lidar_cells=128,
        maximum_range=64.0,
        behavior_intent=behavior_intent_contract(),
    )
    assert contract["raw_observation_dim"] == 323
    assert contract["policy_observation_dim"] == 326
    assert contract["policy_observation_order"][-1] == "behavior_intent_one_hot"
    assert contract["behavior_intent_contract"]["unstandardized"] is True
    assert "route" in contract["behavior_intent_contract"]["forbidden_policy_inputs"]

    raw = np.zeros((4, 323), dtype=np.float32)
    projected = policy_observations_from_flat(
        raw,
        behavior_intent_features=features,
    )
    assert projected.shape == (4, 326)
    assert np.array_equal(projected[:, -4:], features)


def _support_groups(counts: tuple[int, int, int, int]):
    return {
        behavior: {
            (f"episode-{behavior}", vehicle)
            for vehicle in range(count)
        }
        for behavior, count in enumerate(counts)
    }


@pytest.mark.parametrize(
    ("profile", "counts", "canary_passed", "confirmatory_passed"),
    [
        ("confirmatory_per_behavior", (30, 2, 30, 30), True, False),
        ("canary_maneuver_family", (30, 2, 30, 30), True, False),
        ("canary_maneuver_family", (30, 0, 30, 30), False, False),
        ("confirmatory_per_behavior", (30, 24, 24, 30), True, True),
    ],
)
def test_versioned_support_profiles_are_independently_reproduced(
    profile,
    counts,
    canary_passed,
    confirmatory_passed,
):
    groups = _support_groups(counts)
    production = evaluate_support(
        groups,
        minimum_raw_vehicle_groups=24,
        support_profile=profile,
    )
    independent = independently_evaluate_support(
        groups,
        minimum_raw_vehicle_groups=24,
        support_profile=profile,
    )
    assert production == independent
    assert production["canary_support_passed"] is canary_passed
    assert production["confirmatory_support_passed"] is confirmatory_passed
    expected_selected = (
        confirmatory_passed
        if profile == "confirmatory_per_behavior"
        else canary_passed
    )
    assert production["support_passed"] is expected_selected


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((2, 3), dtype=np.float32),
        np.asarray([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        np.asarray([[1, 0, 0, 0], [np.nan, 0, 0, 1]], dtype=np.float32),
    ],
)
def test_behavior_projection_rejects_non_one_hot_rows(invalid):
    with pytest.raises(ValueError, match="Behavior-intent"):
        policy_observations_from_flat(
            np.zeros((2, 323), dtype=np.float32),
            behavior_intent_features=invalid,
        )


def test_lane_zero_is_valid_and_lane_change_window_is_frozen():
    timesteps = np.arange(50, dtype=np.int64)
    lane_ordinals = np.where(timesteps < 25, 0, 1)
    labels = label_behavior_rows(
        vehicle_ids=np.ones(50, dtype=np.int64),
        timesteps=timesteps,
        lane_group_ids=np.full(50, "road-a"),
        lane_ordinals=lane_ordinals,
        longitudinal_positions_m=timesteps.astype(float),
        lane_valid=np.ones(50, dtype=bool),
    )
    assert labels.label_valid.all()
    assert np.all(labels.behavior_ids[5:40] == BehaviorIntent.CHANGE_LEFT)
    assert np.all(labels.behavior_ids[:5] == BehaviorIntent.KEEP_LANE)
    assert np.all(labels.behavior_ids[40:] == BehaviorIntent.KEEP_LANE)
    assert labels.segment_ids[0] >= 0


def test_non_adjacent_jump_is_invalid_instead_of_forced_class():
    timesteps = np.arange(30, dtype=np.int64)
    labels = label_behavior_rows(
        vehicle_ids=np.ones(30, dtype=np.int64),
        timesteps=timesteps,
        lane_group_ids=np.full(30, "road-a"),
        lane_ordinals=np.where(timesteps < 15, 0, 2),
        longitudinal_positions_m=timesteps.astype(float),
        lane_valid=np.ones(30, dtype=bool),
    )
    invalid = labels.invalid_reasons == "non_adjacent_lane_jump"
    assert invalid.any()
    assert not labels.label_valid[invalid].any()
    assert np.all(labels.behavior_ids[invalid] == -1)


def test_conflicting_oscillation_stays_invalid_after_later_overlap():
    timesteps = np.arange(90, dtype=np.int64)
    lane_ordinals = np.zeros(90, dtype=np.int64)
    lane_ordinals[25:31] = 1
    lane_ordinals[45:] = 1
    labels = label_behavior_rows(
        vehicle_ids=np.ones(90, dtype=np.int64),
        timesteps=timesteps,
        lane_group_ids=np.full(90, "road-a"),
        lane_ordinals=lane_ordinals,
        longitudinal_positions_m=timesteps.astype(float),
        lane_valid=np.ones(90, dtype=bool),
    )
    conflicting = labels.invalid_reasons == "conflicting_overlap"
    assert np.any(conflicting)
    assert not np.any(labels.label_valid[conflicting])
    assert np.all(labels.behavior_ids[conflicting] == -1)

    reference = reference_labels(
        vehicle_ids=np.ones(90, dtype=np.int64),
        timesteps=timesteps,
        lane_group_ids=np.full(90, "road-a"),
        lane_ordinals=lane_ordinals,
        longitudinal_positions_m=timesteps.astype(float),
        lane_valid=np.ones(90, dtype=bool),
        active=np.ones(90, dtype=bool),
        config=BehaviorLabelerConfig(),
    )
    assert np.array_equal(labels.behavior_ids, reference["behavior_ids"])
    assert np.array_equal(labels.label_valid, reference["label_valid"])
    assert np.array_equal(labels.invalid_reasons, reference["invalid_reasons"])


def test_label_valid_requires_a_known_next_behavior_without_fabrication():
    timesteps = np.arange(60, dtype=np.int64)
    labels = label_behavior_rows(
        vehicle_ids=np.ones(60, dtype=np.int64),
        timesteps=timesteps,
        lane_group_ids=np.full(60, "road-a"),
        lane_ordinals=np.where(timesteps < 35, 0, 2),
        longitudinal_positions_m=timesteps.astype(float),
        lane_valid=np.ones(60, dtype=bool),
    )

    invalid_window = labels.invalid_reasons == "non_adjacent_lane_jump"
    predecessor = int(np.flatnonzero(invalid_window)[0]) - 1
    assert labels.invalid_reasons[predecessor] == (
        "invalid_or_missing_next_behavior"
    )
    assert not labels.label_valid[predecessor]
    assert np.all(
        np.isin(
            labels.next_behavior_ids[labels.label_valid],
            np.arange(4, dtype=np.int8),
        )
    )
    assert labels.label_valid[-1]
    assert labels.next_behavior_ids[-1] == labels.behavior_ids[-1]


def test_overtake_precedes_constituent_lane_change_and_is_translation_invariant():
    timesteps = np.tile(np.arange(20, dtype=np.int64), 2)
    vehicle_ids = np.repeat([1, 2], 20)
    lane_ordinals = np.concatenate(
        [
            np.where(np.arange(20) < 5, 0, 1),
            np.zeros(20, dtype=np.int64),
        ]
    )
    longitudinal = np.concatenate(
        [2.0 * np.arange(20), 10.0 + np.arange(20)]
    )

    def build(offset: float):
        return label_behavior_rows(
            vehicle_ids=vehicle_ids,
            timesteps=timesteps,
            lane_group_ids=np.full(40, "road-a"),
            lane_ordinals=lane_ordinals,
            longitudinal_positions_m=longitudinal + offset,
            lane_valid=np.ones(40, dtype=bool),
        )

    original = build(0.0)
    translated = build(10_000.0)
    ego = vehicle_ids == 1
    assert np.array_equal(original.behavior_ids, translated.behavior_ids)
    assert np.array_equal(original.label_valid, translated.label_valid)
    assert np.all(
        original.behavior_ids[np.flatnonzero(ego)[:14]]
        == BehaviorIntent.OVERTAKE
    )


@pytest.mark.parametrize("scenario", ["lane_change", "overtake"])
def test_independent_reference_labeler_matches_frozen_labeler(scenario):
    if scenario == "lane_change":
        timesteps = np.arange(50, dtype=np.int64)
        vehicle_ids = np.ones(50, dtype=np.int64)
        lane_ordinals = np.where(timesteps < 25, 0, 1)
        longitudinal = timesteps.astype(float)
    else:
        timesteps = np.tile(np.arange(20, dtype=np.int64), 2)
        vehicle_ids = np.repeat([1, 2], 20)
        lane_ordinals = np.concatenate(
            [
                np.where(np.arange(20) < 5, 0, 1),
                np.zeros(20, dtype=np.int64),
            ]
        )
        longitudinal = np.concatenate(
            [2.0 * np.arange(20), 10.0 + np.arange(20)]
        )
    inputs = {
        "vehicle_ids": vehicle_ids,
        "timesteps": timesteps,
        "lane_group_ids": np.full(len(timesteps), "road-a"),
        "lane_ordinals": lane_ordinals,
        "longitudinal_positions_m": longitudinal,
        "lane_valid": np.ones(len(timesteps), dtype=bool),
        "active": np.ones(len(timesteps), dtype=bool),
    }
    production = label_behavior_rows(**inputs)
    reference = reference_labels(
        **inputs,
        config=BehaviorLabelerConfig(),
    )
    assert np.array_equal(reference["behavior_ids"], production.behavior_ids)
    assert np.array_equal(
        reference["next_behavior_ids"],
        production.next_behavior_ids,
    )
    assert np.array_equal(reference["segment_ids"], production.segment_ids)
    assert np.array_equal(reference["label_valid"], production.label_valid)
    assert np.array_equal(
        reference["invalid_reasons"],
        production.invalid_reasons,
    )


def test_scheduler_holds_command_and_records_infeasible_attempts():
    scheduler = ProspectiveBehaviorCommandScheduler(
        BehaviorCommandSchedulerConfig(
            keep_lane_horizon_steps=3,
            lane_change_horizon_steps=3,
            overtake_horizon_steps=3,
        ),
        seed=4,
    )
    only_keep = BehaviorFeasibility(True, False, False, False)
    command = scheduler.command_for("ego", only_keep)
    assert command == BehaviorIntent.KEEP_LANE
    for _ in range(2):
        scheduler.advance(["ego"])
        assert scheduler.command_for("ego", only_keep) == command
    receipt = scheduler.receipt()
    assert receipt["action_teacher"] is False
    assert receipt["outcome_dependent_relabeling"] is False
    assert sum(receipt["infeasible_attempt_counts"]) > 0


def test_observability_negative_control_deranges_whole_segments():
    trajectories = np.repeat(["a", "b", "c", "d"], 3)
    segments = np.repeat(np.arange(4), 3)
    intent = np.repeat(np.eye(4, dtype=np.float32), 3, axis=0)
    shuffled = _segment_shuffle(
        intent,
        trajectories,
        segments,
        seed=17,
    )
    for group in range(4):
        rows = slice(group * 3, (group + 1) * 3)
        assert np.all(shuffled[rows] == shuffled[group * 3])
        assert not np.array_equal(shuffled[group * 3], intent[group * 3])


def test_observability_features_require_exact_dense_history_and_future():
    from types import SimpleNamespace

    timesteps = np.asarray([0, 1, 3, 4], dtype=np.int64)
    transitions = SimpleNamespace(
        policy_observations=np.arange(4 * 326, dtype=np.float32).reshape(4, 326),
        actions_continuous_env=np.column_stack(
            (np.zeros(4), np.arange(4))
        ).astype(np.float32),
        trajectory_ids=np.asarray(["trajectory-a"] * 4),
        source_file_names=np.asarray(["episode.npz"] * 4),
        vehicle_ids=np.ones(4, dtype=np.int64),
        timesteps=timesteps,
        trajectory_states=np.column_stack(
            (timesteps.astype(np.float32), np.zeros(4), np.ones(4))
        ),
        behavior_ids=np.zeros(4, dtype=np.int8),
        segment_ids=np.zeros(4, dtype=np.int64),
    )

    features = _features(transitions, history_steps=1)
    assert features["causal"].shape == (2, 322)
    assert features["oracle"].shape == (2, 322 + 4 + 2)
    assert np.array_equal(features["target"], np.asarray([0.0, 2.0]))
    assert np.array_equal(
        features["oracle"][:, -2:],
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_observability_bootstrap_requires_multiple_raw_vehicle_groups():
    with pytest.raises(ValueError, match="at least two"):
        _vehicle_bootstrap_lower(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray(["episode.npz:7", "episode.npz:7"], dtype=object),
            seed=1,
            replicates=5,
        )


def test_common_qualification_requires_pure_gail_recipe():
    config = PSGAILConfig(
        behavior_conditioning_enabled=True,
        behavior_label_sidecar="/labels",
        behavior_command_schedule="/schedule.json",
        behavior_sampling_manifest="/sampling.json",
        action_mode="continuous",
        continuous_action_dim=2,
        bc_pretrain_epochs=0,
        policy_bc_regularization_coef=0.0,
        policy_bc_regularization_final_coef=0.0,
        initial_action_std="",
    )
    payload = {"checkpoint_kind": "gail_final"}
    assert all(primary_recipe_checks("gail", payload, config).values())
    config.policy_bc_regularization_coef = 0.1
    assert not primary_recipe_checks(
        "gail",
        payload,
        config,
    )["gail_bc_regularization_zero"]


def test_common_qualification_actor_delta_is_finite_and_nonzero():
    initial = {
        "weight": torch.zeros((2, 3)),
        "bias": torch.zeros(2),
    }
    final = {
        "weight": torch.ones((2, 3)),
        "bias": torch.zeros(2),
    }
    assert actor_parameter_delta(initial, final) == pytest.approx(np.sqrt(6))
    with pytest.raises(ValueError, match="non-finite"):
        actor_parameter_delta(
            initial,
            {
                "weight": torch.full((2, 3), float("nan")),
                "bias": torch.zeros(2),
            },
        )


def test_iq_health_receipt_requires_joint_context_and_two_actions(tmp_path):
    receipt = build_iq(
        {
            "latest_iq_stats": {"q_abs_max": 2.0, "q_loss": 0.5},
            "best_selection": {"joint_updates": 20},
            "best_validation": {
                "validation_prediction_std_ratio": [0.5, 0.2]
            },
            "replay": {
                "behavior_ids_retained": True,
                "next_behavior_ids_retained": True,
                "recurrent_memory_update": "append_full_context",
                "memory_context_length": 32,
                "bc_coefficient": 0.0,
            },
            "test_split_opened": False,
        }
    )
    path = tmp_path / "iq_health.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    checks, _ = method_health_checks("iq_learn", path)
    assert all(checks.values())


def test_gail_health_receipt_requires_accepted_kl_and_optimizer_step(tmp_path):
    receipt = build_gail(
        {"policy_relative_l2_delta": 0.1},
        [
            {
                "discriminator/loss": 0.4,
                "reward/training_mean": 0.2,
                "policy/ppo_accepted_step_kl_max": 0.005,
                "policy/ppo_optimizer_steps": 2,
                "rollout/acceleration_action_std": 0.3,
                "rollout/steering_action_std": 0.2,
            }
        ],
    )
    path = tmp_path / "gail_health.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    checks, _ = method_health_checks("gail", path)
    assert all(checks.values())
    receipt["test_split_opened"] = True
    path.write_text(json.dumps(receipt), encoding="utf-8")
    checks, _ = method_health_checks("gail", path)
    assert checks["method_health_test_split_sealed"] is False


def test_mirror_swaps_left_and_right_command_without_touching_other_classes():
    values = np.zeros((4, 326), dtype=np.float32)
    values[:, 322:] = np.eye(4, dtype=np.float32)
    mirrored = mirror_policy_observations(values)
    assert np.array_equal(mirrored[0, 322:], values[0, 322:])
    assert np.array_equal(mirrored[1, 322:], values[2, 322:])
    assert np.array_equal(mirrored[2, 322:], values[1, 322:])
    assert np.array_equal(mirrored[3, 322:], values[3, 322:])


def test_sidecar_is_hash_bound_and_rejects_alignment_drift(tmp_path):
    source = tmp_path / "episode.npz"
    metadata = {"scene": "us-101", "episode_name": "episode-a"}
    vehicle_ids = np.ones(30, dtype=np.int64)
    timesteps = np.arange(30, dtype=np.int64)
    np.savez_compressed(
        source,
        vehicle_ids=vehicle_ids,
        timesteps=timesteps,
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )
    from policy.data.behavior_sidecars import sha256_file

    basis = tmp_path / "episode.behavior_basis.npz"
    basis_metadata = {
        "schema_version": 2,
        "sample_phase": "pre_action_t",
        "source_file": source.name,
        "source_sha256": sha256_file(source),
    }
    np.savez_compressed(
        basis,
        vehicle_ids=vehicle_ids,
        timesteps=timesteps,
        lane_group_ids=np.full(30, "road-a"),
        lane_ordinals=np.zeros(30, dtype=np.int16),
        longitudinal_positions_m=timesteps.astype(float),
        lane_valid=np.ones(30, dtype=bool),
        active=np.ones(30, dtype=bool),
        metadata_json=np.asarray(json.dumps(basis_metadata), dtype=np.str_),
    )
    output = tmp_path / "episode.behavior_intent.npz"
    build_behavior_sidecar(
        source_file=source,
        basis_file=basis,
        output_file=output,
        labeler_config=BehaviorLabelerConfig(),
    )
    trajectory_ids = source_trajectory_ids(
        source_file=source,
        source_metadata=metadata,
        vehicle_ids=vehicle_ids,
    )
    loaded = load_behavior_sidecar(
        output,
        source_file=source,
        source_metadata=metadata,
        trajectory_ids=trajectory_ids,
        vehicle_ids=vehicle_ids,
        timesteps=timesteps,
    )
    assert loaded.label_valid.all()
    audit = audit_root(
        expert_root=tmp_path,
        sidecar_root=tmp_path,
        output=tmp_path / "independent_audit.json",
        minimum_raw_vehicle_groups=0,
    )
    assert audit["status"] == "passed"
    assert audit["production_labeler_called"] is False
    assert audit["files"][source.name]["exact_reference_match"] is True
    drifted = vehicle_ids.copy()
    drifted[4] = 99
    with pytest.raises(ValueError, match="vehicle_ids alignment mismatch"):
        load_behavior_sidecar(
            output,
            source_file=source,
            source_metadata=metadata,
            trajectory_ids=trajectory_ids,
            vehicle_ids=drifted,
            timesteps=timesteps,
        )


def test_collector_behavior_basis_is_snapshotted_before_environment_step(
    monkeypatch,
):
    from policy.data import collect_expert as collector

    class FakeLane:
        @staticmethod
        def local_coordinates(position):
            return float(position[0]), float(position[1])

    class FakeNetwork:
        @staticmethod
        def get_closest_lane_index(position, _heading):
            return ("a", "b", 0 if float(position[0]) < 1.0 else 1)

        @staticmethod
        def get_lane(_lane_index):
            return FakeLane()

    vehicle = SimpleNamespace(
        vehicle_ID=7,
        position=np.asarray([0.0, 0.0]),
        heading=0.0,
        scene_collection_is_active=True,
        scene_collection_full_traj=[np.ones(4, dtype=np.float32)],
    )
    base = SimpleNamespace(
        steps=0,
        episode_name="timing-fixture",
        controlled_vehicles=[vehicle],
        road=SimpleNamespace(network=FakeNetwork(), vehicles=[vehicle]),
    )

    class FakeEnv:
        unwrapped = base

        @staticmethod
        def reset(seed):
            assert seed == 11
            return np.zeros((1, 323), dtype=np.float32), {}

        @staticmethod
        def step(_action):
            vehicle.position = np.asarray([2.0, 0.0])
            base.steps = 1
            return (
                np.ones((1, 323), dtype=np.float32),
                0.0,
                True,
                False,
                {},
            )

        @staticmethod
        def render():
            return None

    monkeypatch.setattr(
        collector,
        "flatten_agent_observations",
        lambda value: [np.asarray(value)[0]],
    )
    monkeypatch.setattr(collector, "idle_action_for_env", lambda *_args: 0)
    monkeypatch.setattr(
        collector,
        "continuous_expert_actions_from_info",
        lambda *_args, **_kwargs: np.asarray([[0.25, -0.5]], dtype=np.float32),
    )
    monkeypatch.setattr(
        collector,
        "simulated_trajectory_state_from_vehicle",
        lambda value: np.asarray(
            [value.position[0], value.position[1], 1.0], dtype=np.float32
        ),
    )
    monkeypatch.setattr(collector, "trajectory_row_is_active", lambda _row: True)
    monkeypatch.setattr(
        collector,
        "scene_snapshot_features",
        lambda *_args, **_kwargs: np.zeros(5, dtype=np.float32),
    )
    args = SimpleNamespace(
        max_samples_per_vehicle=1,
        trajectory_state_source="simulated",
        expert_control_mode="continuous",
        progress_update_interval=1,
        visualize_episode=False,
        visualize_episode_index=0,
        save_video=False,
        seed=11,
        max_steps_per_episode=1,
        max_episodes=1,
        disable_progress=True,
        scene_max_vehicles=1,
        cells=128,
        maximum_range=64.0,
        policy_frequency=10,
    )
    arrays, _metadata, _video, basis = collector.collect_expert_episode(
        FakeEnv(),
        args,
        episode_index=0,
    )
    assert arrays["trajectory_states"][0, 0] == 0.0
    assert arrays["provider_observation_flags"].tolist() == [-1]
    assert arrays["provider_action_source_flags"].tolist() == [-1]
    assert arrays["provider_lane_reconciliation_flags"].tolist() == [-1]
    assert basis["lane_ordinals"].tolist() == [0]
    assert basis["longitudinal_positions_m"].tolist() == [0.0]
    assert basis["lane_valid"].tolist() == [True]
    assert vehicle.position[0] == 2.0


def test_primary_sidecar_rejects_timing_ambiguous_basis(tmp_path):
    from policy.data.behavior_sidecars import sha256_file

    source = tmp_path / "episode.npz"
    np.savez_compressed(
        source,
        vehicle_ids=np.ones(5, dtype=np.int64),
        timesteps=np.arange(5, dtype=np.int64),
        metadata_json=np.asarray(
            json.dumps({"scene": "us-101", "episode_name": "fixture"}),
            dtype=object,
        ),
    )
    basis = tmp_path / "episode.behavior_basis.npz"
    np.savez_compressed(
        basis,
        vehicle_ids=np.ones(5, dtype=np.int64),
        timesteps=np.arange(5, dtype=np.int64),
        lane_group_ids=np.full(5, "road-a"),
        lane_ordinals=np.zeros(5, dtype=np.int16),
        longitudinal_positions_m=np.arange(5, dtype=float),
        lane_valid=np.ones(5, dtype=bool),
        active=np.ones(5, dtype=bool),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_file": source.name,
                    "source_sha256": sha256_file(source),
                }
            ),
            dtype=np.str_,
        ),
    )
    with pytest.raises(ValueError, match="schema v2"):
        build_behavior_sidecar(
            source_file=source,
            basis_file=basis,
            output_file=tmp_path / "episode.behavior_intent.npz",
        )


def test_behavior_config_requires_both_sidecar_and_schedule():
    from policy.data.behavior_sidecars import (
        validate_behavior_conditioning_config,
    )

    with pytest.raises(ValueError, match="requires behavior_label_sidecar"):
        validate_behavior_conditioning_config(
            PSGAILConfig(
                behavior_conditioning_enabled=True,
                behavior_label_sidecar="labels",
            )
        )


def test_expert_loader_carries_current_and_next_behavior_into_326_fields(
    tmp_path,
):
    expert_root = tmp_path / "expert"
    expert_root.mkdir()
    source = expert_root / "episode.npz"
    rows = 30
    vehicle_ids = np.ones(rows, dtype=np.int64)
    timesteps = np.arange(rows, dtype=np.int64)
    metadata = {
        "scene": "us-101",
        "episode_name": "episode-a",
        "schema_version": 3,
        "policy_observation_contract": policy_observation_contract(
            lidar_cells=128,
            maximum_range=64.0,
        ),
    }
    np.savez_compressed(
        source,
        observations=np.zeros((rows, 323), dtype=np.float32),
        next_observations=np.zeros((rows, 323), dtype=np.float32),
        trajectory_states=np.column_stack(
            (timesteps, np.zeros(rows), np.ones(rows))
        ).astype(np.float32),
        actions_continuous_env=np.zeros((rows, 2), dtype=np.float32),
        dones=timesteps == rows - 1,
        rewards=np.zeros(rows, dtype=np.float32),
        vehicle_ids=vehicle_ids,
        timesteps=timesteps,
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )
    from policy.data.behavior_sidecars import sha256_file

    basis = expert_root / "episode.behavior_basis.npz"
    np.savez_compressed(
        basis,
        vehicle_ids=vehicle_ids,
        timesteps=timesteps,
        lane_group_ids=np.full(rows, "road-a"),
        lane_ordinals=np.zeros(rows, dtype=np.int16),
        longitudinal_positions_m=timesteps.astype(float),
        lane_valid=np.ones(rows, dtype=bool),
        active=np.ones(rows, dtype=bool),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "schema_version": 2,
                    "sample_phase": "pre_action_t",
                    "source_file": source.name,
                    "source_sha256": sha256_file(source),
                }
            ),
            dtype=np.str_,
        ),
    )
    sidecar_root = tmp_path / "sidecars"
    sidecar_root.mkdir()
    build_behavior_sidecar(
        source_file=source,
        basis_file=basis,
        output_file=sidecar_root / "episode.behavior_intent.npz",
    )
    loaded = load_expert_transition_data(
        str(expert_root),
        max_samples=100,
        behavior_label_sidecar=str(sidecar_root),
    )
    assert loaded.policy_observations.shape == (rows, 326)
    assert loaded.next_policy_observations.shape == (rows, 326)
    assert np.array_equal(
        np.argmax(loaded.policy_observations[:, -4:], axis=1),
        loaded.behavior_ids,
    )
    assert np.array_equal(
        np.argmax(loaded.next_policy_observations[:, -4:], axis=1),
        loaded.next_behavior_ids,
    )
    assert loaded.metadata["behavior_event_coverage"] == 1.0
    assert loaded.source_file_names.tolist() == [source.name] * rows
    assert len(set(loaded.row_keys.tolist())) == rows
    assert json.loads(str(loaded.row_keys[4])) == [
        source.name,
        str(loaded.trajectory_ids[4]),
        1,
        4,
    ]
    original_sidecar = sidecar_root / "episode.behavior_intent.npz"
    mutated_root = tmp_path / "mutated_sidecars"
    mutated_root.mkdir()
    with np.load(original_sidecar, allow_pickle=False) as data:
        mutated = {name: np.asarray(data[name]) for name in data.files}
    mutated["label_valid"] = mutated["label_valid"].copy()
    mutated["behavior_ids"] = mutated["behavior_ids"].copy()
    mutated["segment_ids"] = mutated["segment_ids"].copy()
    mutated["invalid_reasons"] = mutated["invalid_reasons"].copy()
    mutated["label_valid"][10] = False
    mutated["behavior_ids"][10] = -1
    mutated["segment_ids"][10] = -1
    mutated["invalid_reasons"][10] = "synthetic_invalid_test_row"
    np.savez_compressed(
        mutated_root / original_sidecar.name,
        **mutated,
    )
    filtered = load_expert_transition_data(
        str(expert_root),
        max_samples=100,
        behavior_label_sidecar=str(mutated_root),
    )
    assert len(filtered.behavior_ids) == rows - 1
    assert filtered.metadata["behavior_event_coverage"] == pytest.approx(
        (rows - 1.0) / rows
    )


def test_class_uniform_sampler_is_deterministic_and_never_duplicates_windows():
    behavior_ids = np.repeat(np.arange(4, dtype=np.int8), [12, 8, 4, 4])
    segment_ids = np.concatenate(
        [
            np.repeat([0, 1, 2], 4),
            np.repeat([3, 4], 4),
            np.repeat([5], 4),
            np.repeat([6], 4),
        ]
    )
    trajectory_ids = np.asarray(
        [f"vehicle-{segment}" for segment in segment_ids],
        dtype=object,
    )
    windows = [
        SequenceWindow(
            trajectory_id=str(trajectory_ids[start]),
            context_indices=np.zeros((0,), dtype=np.int64),
            train_indices=np.arange(start, start + 4, dtype=np.int64),
        )
        for start in range(0, len(behavior_ids), 4)
    ]
    transitions = SimpleNamespace(
        trajectory_ids=trajectory_ids,
        behavior_ids=behavior_ids,
        segment_ids=segment_ids,
        timesteps=np.arange(len(behavior_ids), dtype=np.int64),
        source_file_names=np.full(
            len(behavior_ids),
            "episode.npz",
            dtype=object,
        ),
        row_keys=np.asarray(
            [f"row-{index}" for index in range(len(behavior_ids))],
            dtype=object,
        ),
    )
    selected_a, manifest_a = class_uniform_behavior_windows(
        transitions, windows, seed=9
    )
    selected_b, manifest_b = class_uniform_behavior_windows(
        transitions, windows, seed=9
    )
    assert manifest_a == manifest_b
    assert manifest_a["selected_windows_per_behavior"] == 1
    assert len(selected_a) == len(selected_b) == 4
    starts = [int(window.train_indices[0]) for window in selected_a]
    assert len(starts) == len(set(starts))
    resolved = windows_from_behavior_sampling_manifest(
        transitions,
        windows,
        manifest_a,
    )
    assert [
        value.train_indices.tolist() for value in resolved
    ] == [value.train_indices.tolist() for value in selected_a]
    selected_indices = transition_indices_from_behavior_sampling_manifest(
        transitions,
        manifest_a,
    )
    expected_keys = {
        key
        for row in manifest_a["rows"]
        for key in row["train_row_keys"]
    }
    assert {
        str(transitions.row_keys[index]) for index in selected_indices
    } == expected_keys
    tampered = json.loads(json.dumps(manifest_a))
    tampered["rows"][0]["train_row_keys"][0] = "missing-row"
    with pytest.raises(ValueError, match="digest mismatch"):
        windows_from_behavior_sampling_manifest(
            transitions,
            windows,
            tampered,
        )


def test_gail_expert_sampling_matches_generator_command_histogram():
    expert_ids = np.repeat(np.arange(4, dtype=np.int8), 3)
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    generator_ids = np.asarray([0, 0, 1, 3, 3, 3], dtype=np.int8)
    selected, receipt = match_expert_behavior_distribution(
        features,
        expert_ids,
        generator_ids,
        seed=4,
    )
    assert selected.shape == (len(generator_ids), 2)
    assert receipt["generator_counts"] == {
        "0": 2,
        "1": 1,
        "2": 0,
        "3": 3,
    }


def test_primary_gail_requires_zero_bc_terms():
    base = PSGAILConfig(
        behavior_conditioning_enabled=True,
        behavior_label_sidecar="labels",
        behavior_command_schedule="schedule.json",
        behavior_sampling_manifest="sampling.json",
        initial_policy_checkpoint="shared.pt",
        action_mode="continuous",
        bc_pretrain_epochs=0,
        policy_bc_regularization_coef=0.0,
        policy_bc_regularization_final_coef=0.0,
        policy_bc_regularization_decay_rounds=0,
        test_episodes=0,
    )
    validate_primary_behavior_gail_recipe(base)
    base.bc_pretrain_epochs = 1
    with pytest.raises(ValueError, match="exactly zero"):
        validate_primary_behavior_gail_recipe(base)
    base.bc_pretrain_epochs = 0
    base.initial_action_std = "0.1,0.1"
    with pytest.raises(ValueError, match="initial_action_std"):
        validate_primary_behavior_gail_recipe(base)


def test_shared_actor_schema_v4_hashes_log_std_and_frozen_intent_normalizer(
    tmp_path,
):
    from policy.models.build_shared_random_actor import (
        build_shared_random_actor,
    )

    normalizer = tmp_path / "normalizer.npz"
    np.savez_compressed(
        normalizer,
        sensor_mean=np.zeros(322, dtype=np.float32),
        sensor_std=np.ones(322, dtype=np.float32),
    )
    output = tmp_path / "shared.pt"
    receipt = build_shared_random_actor(
        output=output,
        depth=2,
        seed=0,
        normalizer=normalizer,
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["checkpoint_schema_version"] == 4
    assert payload["initial_log_std"] == [-2.5, -2.5]
    assert payload["actor_state_sha256"] == receipt["actor_state_sha256"]
    assert payload["actor_state_sha256"] == canonical_tensor_state_sha256(
        payload["policy_state_dict"]
    )
    assert torch.equal(
        payload["policy_state_dict"]["observation_normalizer_mean"][-4:],
        torch.zeros(4),
    )
    assert torch.equal(
        payload["policy_state_dict"]["observation_normalizer_std"][-4:],
        torch.ones(4),
    )
    changed = dict(payload["policy_state_dict"])
    changed["log_std"] = changed["log_std"].clone()
    changed["log_std"][0] += 0.5
    assert canonical_tensor_state_sha256(changed) != payload[
        "actor_state_sha256"
    ]


def test_primary_iq_rejects_trained_bc_checkpoint_kind(tmp_path):
    from policy.methods.iq_learn_train import (
        load_initial_policy_checkpoint,
    )

    checkpoint = tmp_path / "bc.pt"
    torch.save(
        {
            "checkpoint_kind": "behaviour_cloning_best",
            "policy_state_dict": {},
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_name(f"{checkpoint.name}.sha256").write_text(
        f"{digest}  {checkpoint.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="shared_random_actor"):
        load_initial_policy_checkpoint(
            torch.nn.Linear(1, 1),
            checkpoint,
            obs_dim=326,
            action_dim=2,
            cfg=PSGAILConfig(),
            required_checkpoint_kind="shared_random_actor",
        )
