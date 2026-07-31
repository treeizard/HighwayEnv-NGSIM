import json
import math
import hashlib
from pathlib import Path

import pytest

from scripts_gail.run_bc_domain_depth_matrix import (
    read_locked_recipe,
    validate_active_causal_motif_expert_paths,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs" / "bc_gail_causal_motif_v7_seed0_draft.json"


def test_causal_motif_draft_has_only_depth_as_architecture_axis():
    payload = json.loads(RECIPE.read_text(encoding="utf-8"))
    architecture = payload["architecture"]
    assert architecture == {
        "policy_model": "recurrent_transformer",
        "depths": [2, 3],
        "policy_seeds": [0],
        "hidden_size": 256,
        "transformer_heads": 4,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "policy_observation_standardization_clip": 0.0,
        "transformer_observation_tokenization": "dense_temporal",
        "policy_head_init_std": 0.01,
        "memory_tokens": 1,
        "memory_context_length": 32,
        "only_permitted_architecture_axis": "depth",
    }
    assert payload["data"]["continuous_action_columns"] == [
        "acceleration_norm",
        "steering_norm",
    ]
    assert payload["data"]["continuous_action_scales"] == [
        5.0,
        math.pi / 4.0,
    ]
    assert payload["data"]["collection_id"] is None
    assert payload["data"]["training_collection"] is None


def test_causal_motif_draft_fails_closed_until_contract_decision():
    payload = json.loads(RECIPE.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_pending_shared_input_target_contract"
    assert payload["data"]["shared_input_target_contract"] is None
    assert payload["data"]["shared_input_target_contract_decision_receipt"] is None
    assert payload["activation_requirements"][
        "audited_training_collection_receipt"
    ]
    assert payload["data"]["policy_observation_dim"] is None
    with pytest.raises(ValueError, match="locked supported schema"):
        read_locked_recipe(RECIPE)


def _write_locked_recipe_and_audit(tmp_path: Path) -> tuple[Path, Path]:
    payload = json.loads(RECIPE.read_text(encoding="utf-8"))
    canonical_root = tmp_path / "collection"
    for domain in ("us", "japanese"):
        for split in ("train", "val"):
            (canonical_root / domain / split).mkdir(parents=True)
    audit_path = tmp_path / "collection_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "collection_root": str(canonical_root.resolve()),
                "test_data_status": "not_opened",
                "total_row_count": 1000,
                "total_episode_count": 24,
            }
        )
        + "\n"
    )
    contract_results_path = tmp_path / "contract_results.json"
    contract_results_path.write_text(
        json.dumps(
            {
                "canonical_collection_view": str(canonical_root.resolve()),
                "test_data_status": "sealed_not_opened",
                "rows_modified": 0,
                "rows_excluded": 0,
            }
        )
        + "\n"
    )
    payload["status"] = "locked"
    payload["data"]["collection_id"] = "bounded_recollection_v3"
    payload["data"]["training_collection"] = {
        "canonical_root": str(canonical_root),
        "audit_artifact": {
            "path": str(audit_path),
            "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        },
        "contract_results_artifact": {
            "path": str(contract_results_path),
            "sha256": hashlib.sha256(
                contract_results_path.read_bytes()
            ).hexdigest(),
        },
    }
    recipe_path = tmp_path / "locked_recipe.json"
    recipe_path.write_text(json.dumps(payload) + "\n")
    return recipe_path, canonical_root


def test_active_runtime_revalidates_architecture_audit_and_exact_split_roots(
    tmp_path,
):
    recipe_path, canonical_root = _write_locked_recipe_and_audit(tmp_path)
    recipe = read_locked_recipe(recipe_path)
    exact_paths = {
        domain: {
            "train": canonical_root / domain / "train",
            "validation": canonical_root / domain / "val",
        }
        for domain in ("us", "japanese")
    }
    validate_active_causal_motif_expert_paths(recipe, exact_paths)

    changed = json.loads(recipe_path.read_text())
    changed["architecture"]["hidden_size"] = 128
    recipe_path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="architecture mismatch"):
        read_locked_recipe(recipe_path)


def test_active_runtime_rejects_cli_roots_outside_audited_collection(tmp_path):
    recipe_path, canonical_root = _write_locked_recipe_and_audit(tmp_path)
    recipe = read_locked_recipe(recipe_path)
    paths = {
        domain: {
            "train": canonical_root / domain / "train",
            "validation": canonical_root / domain / "val",
        }
        for domain in ("us", "japanese")
    }
    paths["us"]["validation"] = tmp_path / "other_validation"
    paths["us"]["validation"].mkdir()
    with pytest.raises(ValueError, match="differ from the exact audited"):
        validate_active_causal_motif_expert_paths(recipe, paths)
