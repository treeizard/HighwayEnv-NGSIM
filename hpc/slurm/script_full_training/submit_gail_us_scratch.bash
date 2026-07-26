#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
REPODIR="$(cd "${REPODIR}" && pwd)"
: "${VFI_PROJECT_ROOT:?Export the canonical validation_first_interpretability project root}"

VFI_PROJECT_ROOT="$(cd "${VFI_PROJECT_ROOT}" && pwd)"
CAMPAIGN_ID="${CAMPAIGN_ID:-gail_us_scratch_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/bt60/ytao0016/conda/envs/ngsim_env/bin/python}"
RUN_CPU_TESTS="${RUN_CPU_TESTS:-true}"
DRY_RUN="${DRY_RUN:-false}"
COLLECTION_ID="${COLLECTION_ID:-domain_matched_accel5_v2}"
EXPERT_DATA="${EXPERT_DATA:-${VFI_PROJECT_ROOT}/data/expert/${COLLECTION_ID}/us/train}"
POLICY_RECIPE="${POLICY_RECIPE:-${REPODIR}/configs/bc_gail_aligned_accel5_v4.json}"
UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"

case "${CAMPAIGN_ID}" in
    *[!A-Za-z0-9._-]*|'') echo "Unsafe CAMPAIGN_ID: ${CAMPAIGN_ID}" >&2; exit 2 ;;
esac
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ngsim_env Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
EXPERT_DATA="$(realpath "${EXPERT_DATA}")"
if [ ! -s "${EXPERT_DATA}/manifest.json" ] \
    || [ ! -s "${EXPERT_DATA}/action_contract_validation.json" ]; then
    echo "Explicit matched expert contracts are unavailable: ${EXPERT_DATA}" >&2
    exit 2
fi
POLICY_RECIPE="$(realpath "${POLICY_RECIPE}")"
if [ ! -s "${POLICY_RECIPE}" ]; then
    echo "Shared BC/GAIL policy recipe is unavailable: ${POLICY_RECIPE}" >&2
    exit 2
fi
if [ -n "${UPSTREAM_DEPENDENCY}" ]; then
    case "${UPSTREAM_DEPENDENCY}" in
        afterok:[0-9]*) ;;
        *) echo "UPSTREAM_DEPENDENCY must be empty or afterok:<job_id>." >&2; exit 2 ;;
    esac
fi

submission_dir="${VFI_PROJECT_ROOT}/results/runs/submissions/${CAMPAIGN_ID}"
run_root="${VFI_PROJECT_ROOT}/results/runs/policies/gail_airl/${CAMPAIGN_ID}"
slurm_log_root="${VFI_PROJECT_ROOT}/logs/slurm/${CAMPAIGN_ID}"
if [ -e "${submission_dir}" ] || [ -e "${run_root}" ]; then
    echo "Refusing to reuse campaign output: ${CAMPAIGN_ID}" >&2
    exit 3
fi

temporary_root="$(mktemp -d)"
cleanup() {
    if [ -n "${temporary_root:-}" ] && [ -d "${temporary_root}" ]; then
        rm -rf -- "${temporary_root}"
    fi
}
trap cleanup EXIT
manifest="${temporary_root}/pilot_manifest.json"
cd "${REPODIR}"
"${PYTHON_BIN}" -m scripts_gail.build_gail_airl_us_pilot \
    --repo "${REPODIR}" --project-root "${VFI_PROJECT_ROOT}" \
    --run-root "${run_root}" --campaign-id "${CAMPAIGN_ID}" \
    --gail-only --no-bc-initialization --num-rollout-workers 16 \
    --policy-recipe "${POLICY_RECIPE}" --shared-policy-seed 0 \
    --expert-data "${EXPERT_DATA}" --require-explicit-data-contracts \
    --output "${manifest}"

"${PYTHON_BIN}" - "${manifest}" <<'PY'
import sys
from pathlib import Path

from scripts_gail.ps_gail.pilot import load_manifest, select_trial

manifest = load_manifest(Path(sys.argv[1]))
scope = manifest["scope"]
if scope["methods"] != ["gail"] or scope["trial_count"] != 2:
    raise SystemExit("Scratch submission must contain exactly two GAIL trials")
if scope["policy_initialization"] != "random_seeded":
    raise SystemExit("Scratch submission is not random-seeded")
if scope["uses_bc_initialization"] is not False or manifest["initializers"]:
    raise SystemExit("Scratch submission contains BC initialization metadata")
if manifest["data"].get("explicit_contracts_required") is not True:
    raise SystemExit("Scratch submission does not require explicit data contracts")
if scope.get("depth_seed_pairs") != [[2, 0], [3, 0]]:
    raise SystemExit("Depth comparison does not use one paired policy seed")
for depth in (2, 3):
    args = select_trial(manifest, method="gail", depth=depth)["arguments"]
    if "initial_policy_checkpoint" in args or "resume_checkpoint" in args:
        raise SystemExit(f"Depth {depth} contains a checkpoint argument")
    expected = {
        "bc_pretrain_epochs": 0,
        "policy_bc_regularization_coef": 0.0,
        "policy_bc_regularization_final_coef": 0.0,
        "policy_bc_regularization_decay_rounds": 0,
        "num_rollout_workers": 16,
        "rollout_worker_threads": 2,
        "evaluation_num_workers": 16,
        "evaluation_worker_threads": 2,
        "vehicle_increase_soft_collision_rounds": 5,
        "collision_proxy_penalty_coef": 1.0,
        "require_explicit_data_contracts": True,
        "validation_require_exact_horizon": False,
        "validation_min_horizon_coverage": 0.0,
        "policy_model": "recurrent_transformer",
        "hidden_size": 256,
        "transformer_heads": 4,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "transformer_observation_tokenization": "dense_temporal",
        "policy_head_init_std": 0.01,
        "transformer_memory_tokens": 1,
        "transformer_memory_context_length": 32,
    }
    actual = {name: args.get(name) for name in expected}
    if actual != expected:
        raise SystemExit(f"Depth {depth} configuration mismatch: {actual}")
print("manifest_preflight=scratch_no_bc_shared_actor_16x2_passed")
PY

if [ "${RUN_CPU_TESTS}" = true ]; then
    "${PYTHON_BIN}" -m pytest -q \
        tests/test_gail_airl_runtime_infrastructure.py \
        tests/test_ps_gail_training_logic.py
fi

"${PYTHON_BIN}" - <<'PY'
import wandb
if not getattr(wandb.Api(), "api_key", None):
    raise SystemExit("W&B online mode requested but no API key is configured")
PY

mkdir -p "${submission_dir}" "${slurm_log_root}"
mv "${manifest}" "${submission_dir}/pilot_manifest.json"
manifest="${submission_dir}/pilot_manifest.json"
runner="${REPODIR}/hpc/slurm/script_full_training/run_gail_us_scratch_depth.bash"
common_export="ALL,REPODIR=${REPODIR},PILOT_MANIFEST=${manifest},PILOT_SLURM_LOG_ROOT=${slurm_log_root}"
common_export="${common_export},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT},VFI_CONDA_ENV=ngsim_env,PYTHON_BIN=${PYTHON_BIN}"
common_export="${common_export},NGSIM_ACCELERATION_LIMIT_MPS2=5.0"
dependency_args=()
if [ -n "${UPSTREAM_DEPENDENCY}" ]; then
    dependency_args=(--dependency="${UPSTREAM_DEPENDENCY}")
fi

if [ "${DRY_RUN}" = true ]; then
    for depth in 2 3; do
        command=(sbatch --parsable --chdir="${REPODIR}" \
            --job-name="gail_us_scratch_d${depth}" "${dependency_args[@]}" \
            --export="${common_export},DEPTH=${depth}" \
            --output="${slurm_log_root}/gail_depth${depth}_%j.out" \
            --error="${slurm_log_root}/gail_depth${depth}_%j.err" "${runner}")
        printf '%q ' "${command[@]}"; printf '\n'
    done
    exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable." >&2
    exit 127
fi

sbatch --test-only --chdir="${REPODIR}" --job-name=gail_us_scratch_d2 \
    --export="${common_export},DEPTH=2" "${runner}"
sbatch --test-only --chdir="${REPODIR}" --job-name=gail_us_scratch_d3 \
    --export="${common_export},DEPTH=3" "${runner}"

submitted_jobs=()
for depth in 2 3; do
    set +e
    output="$(sbatch --parsable --chdir="${REPODIR}" \
        --job-name="gail_us_scratch_d${depth}" "${dependency_args[@]}" \
        --export="${common_export},DEPTH=${depth}" \
        --output="${slurm_log_root}/gail_depth${depth}_%j.out" \
        --error="${slurm_log_root}/gail_depth${depth}_%j.err" "${runner}" 2>&1)"
    status=$?
    set -e
    if [ "${status}" -ne 0 ]; then
        if [ "${#submitted_jobs[@]}" -gt 0 ]; then
            scancel "${submitted_jobs[@]}" || true
        fi
        echo "Depth-${depth} submission failed; cancelled sibling jobs: ${output}" >&2
        exit "${status}"
    fi
    submitted_jobs+=("${output%%;*}")
done
depth2_job="${submitted_jobs[0]}"
depth3_job="${submitted_jobs[1]}"

export CAMPAIGN_ID manifest run_root slurm_log_root depth2_job depth3_job
export EXPERT_DATA POLICY_RECIPE UPSTREAM_DEPENDENCY
"${PYTHON_BIN}" - "${submission_dir}/submission_metadata.json" <<'PY'
from datetime import datetime, timezone
import json
import os
import sys

payload = {
    "schema_version": 1,
    "submitted_utc": datetime.now(timezone.utc).isoformat(),
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "manifest": os.environ["manifest"],
    "run_root": os.environ["run_root"],
    "slurm_log_root": os.environ["slurm_log_root"],
    "policy_initialization": "random_seeded",
    "expert_data": os.environ["EXPERT_DATA"],
    "policy_recipe": os.environ["POLICY_RECIPE"],
    "architecture_contract_id": "shared_dense_temporal_recurrent_transformer_v1",
    "explicit_data_contracts_required": True,
    "continuous_acceleration_range_mps2": [-5.0, 5.0],
    "bc_initialization": False,
    "bc_pretraining_epochs": 0,
    "bc_regularization": 0.0,
    "vehicle_increase_collision_gate": {
        "soft_rounds": 5,
        "destructive_collision_physics": False,
        "terminate_on_collision": False,
        "collision_proxy_penalty_coef": 1.0,
    },
    "worker_geometry": {
        "allocated_cpus": 32,
        "rollout_workers": 16,
        "rollout_worker_threads": 2,
        "evaluation_workers": 16,
        "evaluation_worker_threads": 2,
    },
    "jobs": {
        "gail_depth2": os.environ["depth2_job"],
        "gail_depth3": os.environ["depth3_job"],
    },
    "upstream_dependency": os.environ.get("UPSTREAM_DEPENDENCY") or None,
    "gpus_per_job": 1,
    "concurrent_depths": True,
    "depth_jobs_are_independent": True,
    "paired_policy_seed": 0,
    "arrays": False,
    "requeue": False,
    "automatic_retry": False,
    "replaces_failed_jobs": ["58473532", "58473533"],
}
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
echo "submitted_depth2_job=${depth2_job}"
echo "submitted_depth3_job=${depth3_job}"
echo "shared_upstream_dependency=${UPSTREAM_DEPENDENCY:-none}"
echo "manifest=${manifest}"
