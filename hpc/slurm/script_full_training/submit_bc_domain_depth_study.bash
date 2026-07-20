#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

mkdir -p "${VFI_LOG_ROOT}/slurm" "${VFI_RESULTS_ROOT}/runs/submissions"
RUNNER="${REPODIR}/hpc/slurm/script_full_training/run_bc_domain_depth_study.bash"

command -v sbatch >/dev/null 2>&1
test -f "${RUNNER}"
test -s "${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982/manifest.json"
test -s "${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train/manifest.json"
LOCKED_RECIPE="${BC_LOCKED_RECIPE:-${VFI_PROJECT_ROOT}/configs/bc_recovery_recipe.json}"
test -s "${LOCKED_RECIPE}"

RUNNER_SHA256="$(sha256sum "${RUNNER}" | awk '{print $1}')"
RECIPE_SHA256="$(sha256sum "${LOCKED_RECIPE}" | awk '{print $1}')"
MATRIX_SHA256="$(sha256sum "${REPODIR}/scripts_gail/run_bc_domain_depth_matrix.py" | awk '{print $1}')"
TRAINER_SHA256="$(sha256sum "${REPODIR}/scripts_gail/train_recurrent_bc_policy.py" | awk '{print $1}')"
RECURRENT_BC_SHA256="$(sha256sum "${REPODIR}/scripts_gail/ps_gail/recurrent_bc.py" | awk '{print $1}')"
MODELS_SHA256="$(sha256sum "${REPODIR}/scripts_gail/ps_gail/models.py" | awk '{print $1}')"

STUDY_JOB_ID="$(
    sbatch --parsable \
        --chdir="${VFI_PROJECT_ROOT}" \
        --export="ALL,REPODIR=${REPODIR},BC_PRODUCTION_SUBMISSION=1,BC_LOCKED_RECIPE=${LOCKED_RECIPE},BC_EXPECTED_RECIPE_SHA256=${RECIPE_SHA256},BC_EXPECTED_MATRIX_SHA256=${MATRIX_SHA256},BC_EXPECTED_TRAINER_SHA256=${TRAINER_SHA256},BC_EXPECTED_RECURRENT_BC_SHA256=${RECURRENT_BC_SHA256},BC_EXPECTED_MODELS_SHA256=${MODELS_SHA256}" \
        "${RUNNER}"
)"
STUDY_JOB_ID="${STUDY_JOB_ID%%;*}"
POLICY_ROOT="${VFI_RESULTS_ROOT}/runs/policies/bc/domain_depth_recovered_${STUDY_JOB_ID}"
ACTIVATION_ROOT="${VFI_RESULTS_ROOT}/test_runs/bc_interpretability_smoke/domain_depth_recovered_${STUDY_JOB_ID}"
REGISTRY_ROOT="${VFI_RESULTS_ROOT}/runs/policy_registry/bc_domain_depth_recovered_${STUDY_JOB_ID}"
CHECKPOINT_ARCHIVE="${VFI_CHECKPOINT_ROOT}/bc/autoregressive_policy_comparison/recovered_${STUDY_JOB_ID}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_DIR="${VFI_RESULTS_ROOT}/runs/submissions/bc_domain_depth_${STAMP}"
mkdir -p "${SUBMISSION_DIR}"
SUBMISSION_DIR="${SUBMISSION_DIR}" STUDY_JOB_ID="${STUDY_JOB_ID}" POLICY_ROOT="${POLICY_ROOT}" ACTIVATION_ROOT="${ACTIVATION_ROOT}" REGISTRY_ROOT="${REGISTRY_ROOT}" CHECKPOINT_ARCHIVE="${CHECKPOINT_ARCHIVE}" LOCKED_RECIPE="${LOCKED_RECIPE}" RUNNER_SHA256="${RUNNER_SHA256}" RECIPE_SHA256="${RECIPE_SHA256}" MATRIX_SHA256="${MATRIX_SHA256}" TRAINER_SHA256="${TRAINER_SHA256}" RECURRENT_BC_SHA256="${RECURRENT_BC_SHA256}" MODELS_SHA256="${MODELS_SHA256}" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["SUBMISSION_DIR"])
payload = {
    "study": "bc_domain_depth_load_once_v2",
    "job_id": os.environ["STUDY_JOB_ID"],
    "execution": "single_process_serial_models_load_each_domain_once",
    "stages": ["gpu_check", "load_once_training", "checkpoint_archive", "activation_smoke", "finalization"],
    "locked_recipe": os.environ["LOCKED_RECIPE"],
    "policy_root": os.environ["POLICY_ROOT"],
    "activation_root": os.environ["ACTIVATION_ROOT"],
    "registry_root": os.environ["REGISTRY_ROOT"],
    "checkpoint_archive": os.environ["CHECKPOINT_ARCHIVE"],
    "domains": ["us", "japanese"],
    "transformer_layers": [2, 3],
    "policy_seeds": [0, 1, 2],
    "model_count": 12,
    "source_sha256": {
        "runner": os.environ["RUNNER_SHA256"],
        "recipe": os.environ["RECIPE_SHA256"],
        "matrix": os.environ["MATRIX_SHA256"],
        "trainer": os.environ["TRAINER_SHA256"],
        "recurrent_bc": os.environ["RECURRENT_BC_SHA256"],
        "models": os.environ["MODELS_SHA256"],
    },
}
(out / "submission.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Full BC load-once serial 12-model job: ${STUDY_JOB_ID}"
echo "Submission manifest: ${SUBMISSION_DIR}/submission.json"
