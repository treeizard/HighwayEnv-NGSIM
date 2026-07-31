#!/bin/bash
set -euo pipefail

# Recover failed jobs 58507057, 58508548 (superseding 58508523), 58508531,
# and 58508532 from one immutable deployment snapshot.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPODIR="${SOURCE_REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SOURCE_REPODIR="$(cd "${SOURCE_REPODIR}" && pwd)"
: "${VFI_PROJECT_ROOT:?Export the validation_first_interpretability_dev project root}"
VFI_PROJECT_ROOT="$(cd "${VFI_PROJECT_ROOT}" && pwd)"

CAMPAIGN_ID="${CAMPAIGN_ID:-failed_policy_recovery_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/bt60/ytao0016/conda/envs/ngsim_env/bin/python}"
DATA_PROJECT_ROOT="${DATA_PROJECT_ROOT:-/fs04/bt60/ytao0016/validation_first_interpretability}"
RUN_TESTS="${RUN_TESTS:-true}"
DRY_RUN="${DRY_RUN:-false}"
SUBMIT_BC="${SUBMIT_BC:-true}"
DEPTH2_CHECKPOINT="${DEPTH2_CHECKPOINT:-${VFI_PROJECT_ROOT}/results/runs/policies/gail_airl/gail_us_aligned_a5_20260724T054734Z/gail/recurrent_transformer_2layer_seed_0/resume_latest.pt}"
DEPTH3_CHECKPOINT="${DEPTH3_CHECKPOINT:-${VFI_PROJECT_ROOT}/results/runs/policies/gail_airl/gail_us_aligned_a5_20260724T054734Z/gail/recurrent_transformer_3layer_seed_1/resume_latest.pt}"
FAILED_GAIL_MANIFEST="${FAILED_GAIL_MANIFEST:-${VFI_PROJECT_ROOT}/results/runs/submissions/gail_us_aligned_a5_20260724T054734Z/pilot_manifest.json}"

case "${CAMPAIGN_ID}" in
    *[!A-Za-z0-9._-]*|'') echo "Unsafe CAMPAIGN_ID: ${CAMPAIGN_ID}" >&2; exit 2 ;;
esac
for required in "${PYTHON_BIN}" \
    "${FAILED_GAIL_MANIFEST}" \
    "${DEPTH2_CHECKPOINT}" "${DEPTH2_CHECKPOINT}.sha256" \
    "${DEPTH3_CHECKPOINT}" "${DEPTH3_CHECKPOINT}.sha256"; do
    test -s "${required}"
done
deployment_root="${VFI_PROJECT_ROOT}/deployments/${CAMPAIGN_ID}"
repodir="${deployment_root}/HighwayEnv-NGSIM"
submission_dir="${VFI_PROJECT_ROOT}/results/runs/submissions/${CAMPAIGN_ID}"
run_root="${VFI_PROJECT_ROOT}/results/runs/policies/gail_airl/${CAMPAIGN_ID}"
slurm_log_root="${VFI_PROJECT_ROOT}/logs/slurm/${CAMPAIGN_ID}"
audit_root="${VFI_PROJECT_ROOT}/results/audits/${CAMPAIGN_ID}"
for target in "${deployment_root}" "${submission_dir}" "${run_root}" \
    "${slurm_log_root}" "${audit_root}"; do
    if [ -e "${target}" ]; then
        echo "Refusing to reuse recovery target: ${target}" >&2
        exit 3
    fi
done

mkdir -p "${deployment_root}" "${submission_dir}" "${slurm_log_root}" "${audit_root}"
git clone --quiet --no-hardlinks "${SOURCE_REPODIR}" "${repodir}"
rsync -a --exclude='.git' --exclude='__pycache__/' --exclude='.pytest_cache/' \
    "${SOURCE_REPODIR}/" "${repodir}/"

if [ "${RUN_TESTS}" = true ]; then
    (
        cd "${repodir}"
        PYTHONPATH="${repodir}" "${PYTHON_BIN}" -m pytest -q \
            tests/test_recurrent_bc.py \
            tests/test_bc_load_once_matrix.py \
            tests/test_expert_observation_validation.py \
            tests/test_gail_airl_runtime_infrastructure.py \
            tests/test_gail_airl_study.py \
            tests/test_ps_gail_training_logic.py
    )
fi

expert_data="${DATA_PROJECT_ROOT}/data/expert/domain_matched_accel5_v2/us/train"
base_manifest="${submission_dir}/base_scratch_manifest.json"
manifest="${submission_dir}/pilot_manifest.json"
(
    cd "${repodir}"
    PYTHONPATH="${repodir}" "${PYTHON_BIN}" -m scripts_gail.build_gail_airl_us_pilot \
        --repo "${repodir}" \
        --project-root "${DATA_PROJECT_ROOT}" \
        --run-root "${run_root}" \
        --campaign-id "${CAMPAIGN_ID}" \
        --gail-only --no-bc-initialization --num-rollout-workers 16 \
        --expert-data "${expert_data}" --require-explicit-data-contracts \
        --output "${base_manifest}"
    PYTHONPATH="${repodir}" "${PYTHON_BIN}" \
        -m scripts_gail.build_gail_us_scratch_recovery_manifest \
        --base-manifest "${base_manifest}" \
        --failed-manifest "${FAILED_GAIL_MANIFEST}" \
        --depth2-checkpoint "${DEPTH2_CHECKPOINT}" \
        --depth3-checkpoint "${DEPTH3_CHECKPOINT}" \
        --output "${manifest}"
    PYTHONPATH="${repodir}" "${PYTHON_BIN}" \
        -m scripts_gail.run_gail_airl_us_pilot_trial \
        --manifest "${manifest}" --repo "${repodir}" \
        --method gail --depth 2 --python "${PYTHON_BIN}" \
        --verify-only --include-large-data
    PYTHONPATH="${repodir}" "${PYTHON_BIN}" \
        -m scripts_gail.run_gail_airl_us_pilot_trial \
        --manifest "${manifest}" --repo "${repodir}" \
        --method gail --depth 3 --python "${PYTHON_BIN}" \
        --verify-only --include-large-data
)

recipe="${repodir}/configs/bc_gail_aligned_accel5_v5.json"
audit_runner="${repodir}/hpc/slurm/script_data_collection/audit_domain_matched_expert_accel5.bash"
bc_runner="${repodir}/hpc/slurm/script_full_training/run_bc_gail_aligned_accel5.bash"
gail_runner="${repodir}/hpc/slurm/script_full_training/run_gail_us_scratch_depth.bash"
audit_out="${audit_root}/collection_contract_audit_train_val.json"
audit_script="${repodir}/scripts_gail/audit_domain_matched_expert.py"

audit_export="ALL,REPODIR=${repodir},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT}"
audit_export="${audit_export},VFI_CONDA_ENV=ngsim_env,COLLECTION_ID=domain_matched_accel5_v2"
audit_export="${audit_export},AUDIT_OUT=${audit_out}"
audit_export="${audit_export},AUDIT_SPLITS=train:val"
audit_export="${audit_export},AUDIT_EXPECTED_SCRIPT_SHA256=$(sha256sum "${audit_script}" | awk '{print $1}')"
audit_export="${audit_export},AUDIT_EXPECTED_RUNNER_SHA256=$(sha256sum "${audit_runner}" | awk '{print $1}')"

bc_export="ALL,REPODIR=${repodir},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT}"
bc_export="${bc_export},VFI_CONDA_ENV=ngsim_env,COLLECTION_ID=domain_matched_accel5_v2"
bc_export="${bc_export},BC_LOCKED_RECIPE=${recipe},BC_AUDIT_PATH=${audit_out}"
bc_export="${bc_export},BC_EXPECTED_RECIPE_SHA256=$(sha256sum "${recipe}" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_RUNNER_SHA256=$(sha256sum "${bc_runner}" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_MATRIX_SHA256=$(sha256sum "${repodir}/scripts_gail/run_bc_domain_depth_matrix.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_TRAINER_SHA256=$(sha256sum "${repodir}/scripts_gail/train_recurrent_bc_policy.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_RECURRENT_BC_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/recurrent_bc.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_MODELS_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/models.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_CONTRACTS_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/contracts.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_DATA_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/data.py" | awk '{print $1}')"

gail_export="ALL,REPODIR=${repodir},PILOT_MANIFEST=${manifest}"
gail_export="${gail_export},PILOT_SLURM_LOG_ROOT=${slurm_log_root}"
gail_export="${gail_export},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT},VFI_CONDA_ENV=ngsim_env"
gail_export="${gail_export},PYTHON_BIN=${PYTHON_BIN},NGSIM_ACCELERATION_LIMIT_MPS2=5.0"
gail_export="${gail_export},RECOVERY_RESUME=true"

if [ "${DRY_RUN}" = true ]; then
    echo "deployment=${repodir}"
    echo "manifest=${manifest}"
    echo "audit_runner=${audit_runner}"
    echo "bc_runner=${bc_runner}"
    echo "gail_runner=${gail_runner}"
    exit 0
fi

test_runners=("${audit_runner}" "${gail_runner}")
if [ "${SUBMIT_BC}" = true ]; then
    test_runners+=("${bc_runner}")
fi
for runner in "${test_runners[@]}"; do
    sbatch --test-only --chdir="${repodir}" "${runner}"
done

audit_job="$(sbatch --parsable --chdir="${repodir}" \
    --export="${audit_export}" \
    --output="${slurm_log_root}/audit_expert_a5_%j.out" \
    --error="${slurm_log_root}/audit_expert_a5_%j.err" \
    "${audit_runner}")"
audit_job="${audit_job%%;*}"

bc_job=""
if [ "${SUBMIT_BC}" = true ]; then
    bc_job="$(sbatch --parsable --chdir="${repodir}" \
        --dependency="afterok:${audit_job}" \
        --export="${bc_export}" \
        --output="${slurm_log_root}/bc_gail_a5_%j.out" \
        --error="${slurm_log_root}/bc_gail_a5_%j.err" \
        "${bc_runner}")"
    bc_job="${bc_job%%;*}"
fi

depth2_job="$(sbatch --parsable --chdir="${repodir}" \
    --job-name=gail_us_scratch_d2 \
    --dependency="afterok:${audit_job}" \
    --export="${gail_export},DEPTH=2" \
    --output="${slurm_log_root}/gail_depth2_%j.out" \
    --error="${slurm_log_root}/gail_depth2_%j.err" \
    "${gail_runner}")"
depth2_job="${depth2_job%%;*}"

depth3_job="$(sbatch --parsable --chdir="${repodir}" \
    --job-name=gail_us_scratch_d3 \
    --dependency="afterok:${audit_job}" \
    --export="${gail_export},DEPTH=3" \
    --output="${slurm_log_root}/gail_depth3_%j.out" \
    --error="${slurm_log_root}/gail_depth3_%j.err" \
    "${gail_runner}")"
depth3_job="${depth3_job%%;*}"

export CAMPAIGN_ID repodir manifest base_manifest run_root slurm_log_root audit_out
export audit_job bc_job depth2_job depth3_job
"${PYTHON_BIN}" - "${submission_dir}/submission_metadata.json" <<'PY'
from datetime import datetime, timezone
import json
import os
import sys

payload = {
    "schema_version": 1,
    "submitted_utc": datetime.now(timezone.utc).isoformat(),
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "deployment": os.environ["repodir"],
    "base_manifest": os.environ["base_manifest"],
    "recovery_manifest": os.environ["manifest"],
    "run_root": os.environ["run_root"],
    "slurm_log_root": os.environ["slurm_log_root"],
    "audit_output": os.environ["audit_out"],
    "historical_bc_promotion_precondition": "removed_circular_precondition",
    "bc_scientific_qualification": "pending_prospective_validation",
    "continuous_acceleration_range_mps2": [-5.0, 5.0],
    "replaces_failed_jobs": [
        "58507057",
        "58508523",
        "58508548",
        "58508531",
        "58508532",
    ],
    "distinct_replacement_jobs": {
        "expert_audit": os.environ["audit_job"],
        "bc_matrix": os.environ["bc_job"] or None,
        "gail_depth2_exact_resume": os.environ["depth2_job"],
        "gail_depth3_exact_resume": os.environ["depth3_job"],
    },
    "dependencies": {
        "bc_matrix": (
            f"afterok:{os.environ['audit_job']}"
            if os.environ["bc_job"]
            else None
        ),
        "gail_depth2_exact_resume": f"afterok:{os.environ['audit_job']}",
        "gail_depth3_exact_resume": f"afterok:{os.environ['audit_job']}",
    },
    "automatic_retry": False,
    "arrays": False,
}
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "submitted_audit_job=${audit_job}"
echo "submitted_bc_job=${bc_job}"
echo "submitted_gail_depth2_job=${depth2_job}"
echo "submitted_gail_depth3_job=${depth3_job}"
echo "deployment=${repodir}"
echo "manifest=${manifest}"
