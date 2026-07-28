#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
REPODIR="$(cd "${REPODIR}" && pwd)"
: "${VFI_PROJECT_ROOT:?Export the canonical validation_first_interpretability project root}"

VFI_PROJECT_ROOT="$(cd "${VFI_PROJECT_ROOT}" && pwd)"
CAMPAIGN_ID="${CAMPAIGN_ID:-gail_us_realistic_wgan_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/bt60/ytao0016/conda/envs/ngsim_env/bin/python}"
RUN_CPU_TESTS="${RUN_CPU_TESTS:-true}"
DRY_RUN="${DRY_RUN:-false}"
COLLECTION_ID="${COLLECTION_ID:-domain_matched_accel5_v2}"
EXPERT_DATA="${EXPERT_DATA:-${VFI_PROJECT_ROOT}/data/expert/${COLLECTION_ID}/us/train}"
BC_ROOT="${BC_ROOT:-${VFI_PROJECT_ROOT}/results/runs/policies/bc/gail_aligned_accel5_58539772/us}"
POLICY_RECIPE="${POLICY_RECIPE:-${REPODIR}/configs/bc_gail_aligned_accel5_v4.json}"
UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"
GAIL_DEPTHS="${GAIL_DEPTHS:-2 3}"

case "${CAMPAIGN_ID}" in
    *[!A-Za-z0-9._-]*|'') echo "Unsafe CAMPAIGN_ID: ${CAMPAIGN_ID}" >&2; exit 2 ;;
esac
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ngsim_env Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [ "${DRY_RUN}" != true ] \
    && [ -n "$(git -C "${REPODIR}" status --porcelain --untracked-files=all)" ]; then
    echo "Official realistic-WGAN submission requires a clean checkout: ${REPODIR}" >&2
    exit 3
fi
EXPERT_DATA="$(realpath "${EXPERT_DATA}")"
BC_ROOT="$(realpath "${BC_ROOT}")"
POLICY_RECIPE="$(realpath "${POLICY_RECIPE}")"
if [ ! -s "${EXPERT_DATA}/manifest.json" ] \
    || [ ! -s "${EXPERT_DATA}/action_contract_validation.json" ]; then
    echo "Explicit matched expert contracts are unavailable: ${EXPERT_DATA}" >&2
    exit 2
fi
if [ ! -s "${POLICY_RECIPE}" ]; then
    echo "Shared BC/GAIL policy recipe is unavailable: ${POLICY_RECIPE}" >&2
    exit 2
fi
for depth in 2 3; do
    checkpoint="${BC_ROOT}/recurrent_transformer_${depth}layer/policy_seed_0/best.pt"
    if [ ! -s "${checkpoint}" ] || [ ! -s "${checkpoint}.sha256" ]; then
        echo "BC initializer is unavailable: ${checkpoint}" >&2
        exit 2
    fi
done
if [ -n "${UPSTREAM_DEPENDENCY}" ]; then
    case "${UPSTREAM_DEPENDENCY}" in
        afterok:[0-9]*) ;;
        *) echo "UPSTREAM_DEPENDENCY must be empty or afterok:<job_id>." >&2; exit 2 ;;
    esac
fi
read -r -a requested_depths <<< "${GAIL_DEPTHS}"
if [ "${#requested_depths[@]}" -eq 0 ]; then
    echo "GAIL_DEPTHS must select depth 2, depth 3, or both." >&2
    exit 2
fi
seen_depth2=false
seen_depth3=false
for depth in "${requested_depths[@]}"; do
    case "${depth}" in
        2)
            if [ "${seen_depth2}" = true ]; then
                echo "GAIL_DEPTHS contains duplicate depth 2." >&2
                exit 2
            fi
            seen_depth2=true
            ;;
        3)
            if [ "${seen_depth3}" = true ]; then
                echo "GAIL_DEPTHS contains duplicate depth 3." >&2
                exit 2
            fi
            seen_depth3=true
            ;;
        *)
            echo "GAIL_DEPTHS contains unsupported depth: ${depth}" >&2
            exit 2
            ;;
    esac
done

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
    --gail-only --gail-training-profile realistic_wgan_v1 \
    --num-rollout-workers 16 --shared-policy-seed 0 \
    --expert-data "${EXPERT_DATA}" --bc-root "${BC_ROOT}" \
    --policy-recipe "${POLICY_RECIPE}" --require-explicit-data-contracts \
    --output "${manifest}"

"${PYTHON_BIN}" - "${manifest}" <<'PY'
import sys
from pathlib import Path

from scripts_gail.ps_gail.pilot import load_manifest, select_trial

manifest = load_manifest(Path(sys.argv[1]))
scope = dict(manifest["scope"])
if scope.get("methods") != ["gail"] or scope.get("trial_count") != 2:
    raise SystemExit("Realistic WGAN manifest must contain exactly two GAIL trials")
if scope.get("policy_initialization") != "verified_matched_bc_checkpoint":
    raise SystemExit("Realistic WGAN manifest is not verified BC-initialized")
if scope.get("bc_initializer_qualification") != "matched_training_artifact_v1":
    raise SystemExit("Realistic WGAN manifest has the wrong BC initializer contract")
if scope.get("uses_bc_initialization") is not True:
    raise SystemExit("Realistic WGAN manifest disables BC initialization")
if scope.get("gail_training_profile") != "realistic_wgan_v1":
    raise SystemExit("Realistic WGAN manifest has the wrong training profile")
if scope.get("depth_seed_pairs") != [[2, 0], [3, 0]]:
    raise SystemExit("Depth comparison does not use the paired seed-0 BC policies")
if manifest["data"].get("explicit_contracts_required") is not True:
    raise SystemExit("Realistic WGAN manifest does not require explicit data contracts")

expected = {
    "algorithm_variant": "gail_wgan_gp",
    "discriminator_input": "action",
    "wgan_gp_lambda": 2.0,
    "normalize_discriminator_features": True,
    "disc_updates_per_round": 2,
    "discriminator_replay_rounds": 3,
    "discriminator_replay_max_samples": 120000,
    "terminate_when_all_controlled_crashed": False,
    "rollout_fixed_horizon": True,
    "evaluation_terminate_when_all_controlled_crashed": False,
    "normalize_gail_reward": True,
    "allow_wgan_reward_normalization": True,
    "learning_rate": 1.0e-5,
    "entropy_coef": 5.0e-4,
    "policy_bc_regularization_coef": 0.02,
    "policy_bc_regularization_final_coef": 0.0,
    "policy_bc_regularization_decay_rounds": 50,
    "initial_action_std": "0.10,0.05",
    "minimum_action_std": "0.02,0.01",
    "maximum_action_std": "0.30,0.15",
    "total_rounds": 800,
    "final_controlled_vehicles": 100.0,
    "full_load_selection_start_round": 701,
    "num_rollout_workers": 16,
    "rollout_worker_threads": 2,
    "evaluation_num_workers": 16,
    "evaluation_worker_threads": 2,
    "require_explicit_data_contracts": True,
}
for depth in (2, 3):
    args = select_trial(manifest, method="gail", depth=depth)["arguments"]
    actual = {name: args.get(name) for name in expected}
    if actual != expected:
        raise SystemExit(
            f"Depth-{depth} realistic WGAN configuration mismatch: {actual}"
        )
    if not args.get("initial_policy_checkpoint") or args.get("resume_checkpoint"):
        raise SystemExit(f"Depth-{depth} does not contain exactly one BC initializer")
print("manifest_preflight=realistic_wgan_bc_init_100_vehicle_passed")
PY

if [ "${RUN_CPU_TESTS}" = true ]; then
    "${PYTHON_BIN}" -m pytest -q \
        tests/test_gail_airl_runtime_infrastructure.py \
        tests/test_ps_gail_training_logic.py
fi

runner="${REPODIR}/hpc/slurm/script_full_training/run_gail_us_scratch_depth.bash"
common_export="ALL,REPODIR=${REPODIR},PILOT_MANIFEST=${manifest},PILOT_SLURM_LOG_ROOT=${slurm_log_root}"
common_export="${common_export},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT},VFI_CONDA_ENV=ngsim_env,PYTHON_BIN=${PYTHON_BIN}"
common_export="${common_export},NGSIM_ACCELERATION_LIMIT_MPS2=5.0,GAIL_RUN_PROFILE=realistic_wgan_v1"
dependency_args=()
if [ -n "${UPSTREAM_DEPENDENCY}" ]; then
    dependency_args=(--dependency="${UPSTREAM_DEPENDENCY}")
fi

if [ "${DRY_RUN}" = true ]; then
    for depth in "${requested_depths[@]}"; do
        command=(sbatch --parsable --chdir="${REPODIR}" \
            --job-name="gail_us_wgan_d${depth}" "${dependency_args[@]}" \
            --export="${common_export},DEPTH=${depth}" \
            --output="${slurm_log_root}/gail_wgan_depth${depth}_%j.out" \
            --error="${slurm_log_root}/gail_wgan_depth${depth}_%j.err" "${runner}")
        printf '%q ' "${command[@]}"; printf '\n'
    done
    exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable." >&2
    exit 127
fi
"${PYTHON_BIN}" - <<'PY'
import wandb
if not getattr(wandb.Api(), "api_key", None):
    raise SystemExit("W&B online mode requested but no API key is configured")
PY

mkdir -p "${submission_dir}" "${slurm_log_root}"
mv "${manifest}" "${submission_dir}/pilot_manifest.json"
manifest="${submission_dir}/pilot_manifest.json"
common_export="ALL,REPODIR=${REPODIR},PILOT_MANIFEST=${manifest},PILOT_SLURM_LOG_ROOT=${slurm_log_root}"
common_export="${common_export},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT},VFI_CONDA_ENV=ngsim_env,PYTHON_BIN=${PYTHON_BIN}"
common_export="${common_export},NGSIM_ACCELERATION_LIMIT_MPS2=5.0,GAIL_RUN_PROFILE=realistic_wgan_v1"

for depth in "${requested_depths[@]}"; do
    sbatch --test-only --chdir="${REPODIR}" --job-name="gail_us_wgan_d${depth}" \
        --export="${common_export},DEPTH=${depth}" "${runner}"
done

submitted_jobs=()
for depth in "${requested_depths[@]}"; do
    set +e
    output="$(sbatch --parsable --chdir="${REPODIR}" \
        --job-name="gail_us_wgan_d${depth}" "${dependency_args[@]}" \
        --export="${common_export},DEPTH=${depth}" \
        --output="${slurm_log_root}/gail_wgan_depth${depth}_%j.out" \
        --error="${slurm_log_root}/gail_wgan_depth${depth}_%j.err" "${runner}" 2>&1)"
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

submitted_depth_pairs=""
for index in "${!requested_depths[@]}"; do
    submitted_depth_pairs+="${requested_depths[$index]}:${submitted_jobs[$index]} "
done
submitted_depth_pairs="${submitted_depth_pairs% }"
export CAMPAIGN_ID manifest run_root slurm_log_root submitted_depth_pairs
export EXPERT_DATA BC_ROOT POLICY_RECIPE UPSTREAM_DEPENDENCY
"${PYTHON_BIN}" - "${submission_dir}/submission_metadata.json" <<'PY'
from datetime import datetime, timezone
import json
import os
import sys

jobs = {
    f"gail_depth{depth}": job_id
    for item in os.environ["submitted_depth_pairs"].split()
    for depth, job_id in [item.split(":", 1)]
}
payload = {
    "schema_version": 1,
    "submitted_utc": datetime.now(timezone.utc).isoformat(),
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "training_profile": "realistic_wgan_v1",
    "manifest": os.environ["manifest"],
    "run_root": os.environ["run_root"],
    "slurm_log_root": os.environ["slurm_log_root"],
    "expert_data": os.environ["EXPERT_DATA"],
    "bc_root": os.environ["BC_ROOT"],
    "policy_recipe": os.environ["POLICY_RECIPE"],
    "algorithm_variant": "gail_wgan_gp",
    "policy_initialization": "verified_matched_bc_checkpoint",
    "bc_initializer_qualification": "matched_training_artifact_v1",
    "full_load_selection_start_round": 701,
    "jobs": jobs,
    "upstream_dependency": os.environ.get("UPSTREAM_DEPENDENCY") or None,
    "gpus_per_job": 1,
    "concurrent_depths": len(jobs) > 1,
    "arrays": False,
    "requeue": False,
    "automatic_retry": False,
}
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
for index in "${!requested_depths[@]}"; do
    echo "submitted_depth${requested_depths[$index]}_job=${submitted_jobs[$index]}"
done
echo "manifest=${manifest}"
