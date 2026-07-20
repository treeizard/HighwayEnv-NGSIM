#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

mkdir -p "${VFI_LOG_ROOT}/slurm" "${VFI_RESULTS_ROOT}/runs/submissions"
PILOT_RUNNER="${REPODIR}/hpc/slurm/script_full_training/run_iq_convergence_pilot.bash"
STUDY_RUNNER="${REPODIR}/hpc/slurm/script_full_training/run_iq_domain_depth_study.bash"
BC_JOB_ID="${IQ_BC_JOB_ID:-58391443}"
BC_ROOT="${IQ_BC_ROOT:-${VFI_RESULTS_ROOT}/runs/policies/bc/domain_depth_recovered_${BC_JOB_ID}}"

command -v sbatch >/dev/null 2>&1
test -f "${PILOT_RUNNER}"
test -f "${STUDY_RUNNER}"
test -s "${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982/manifest.json"
test -s "${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train/manifest.json"

PILOT_SHA256="$(sha256sum "${REPODIR}/scripts_gail/run_iq_convergence_pilot.py" | awk '{print $1}')"
MATRIX_SHA256="$(sha256sum "${REPODIR}/scripts_gail/run_iq_domain_depth_matrix.py" | awk '{print $1}')"
STUDY_SHA256="$(sha256sum "${REPODIR}/scripts_gail/iq_study.py" | awk '{print $1}')"
TRAINER_SHA256="$(sha256sum "${REPODIR}/scripts_gail/train_recurrent_iq_learn.py" | awk '{print $1}')"
CORE_SHA256="$(sha256sum "${REPODIR}/scripts_gail/ps_gail/recurrent_iq.py" | awk '{print $1}')"
MODELS_SHA256="$(sha256sum "${REPODIR}/scripts_gail/ps_gail/models.py" | awk '{print $1}')"
RECURRENT_BC_SHA256="$(sha256sum "${REPODIR}/scripts_gail/ps_gail/recurrent_bc.py" | awk '{print $1}')"

PILOT_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${BC_JOB_ID}" --chdir="${VFI_PROJECT_ROOT}" \
        --export="ALL,REPODIR=${REPODIR},IQ_BC_ROOT=${BC_ROOT},IQ_EXPECTED_PILOT_SHA256=${PILOT_SHA256},IQ_EXPECTED_STUDY_SHA256=${STUDY_SHA256},IQ_EXPECTED_TRAINER_SHA256=${TRAINER_SHA256},IQ_EXPECTED_CORE_SHA256=${CORE_SHA256},IQ_EXPECTED_MODELS_SHA256=${MODELS_SHA256},IQ_EXPECTED_RECURRENT_BC_SHA256=${RECURRENT_BC_SHA256}" \
        "${PILOT_RUNNER}"
)"
PILOT_JOB_ID="${PILOT_JOB_ID%%;*}"
LOCKED_RECIPE="${VFI_RESULTS_ROOT}/runs/policies/iq/pilot_${PILOT_JOB_ID}/locked_recipe.json"

STUDY_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${PILOT_JOB_ID}" --chdir="${VFI_PROJECT_ROOT}" \
        --export="ALL,REPODIR=${REPODIR},IQ_BC_ROOT=${BC_ROOT},IQ_LOCKED_RECIPE=${LOCKED_RECIPE},IQ_EXPECTED_MATRIX_SHA256=${MATRIX_SHA256},IQ_EXPECTED_STUDY_SHA256=${STUDY_SHA256},IQ_EXPECTED_TRAINER_SHA256=${TRAINER_SHA256},IQ_EXPECTED_CORE_SHA256=${CORE_SHA256},IQ_EXPECTED_MODELS_SHA256=${MODELS_SHA256},IQ_EXPECTED_RECURRENT_BC_SHA256=${RECURRENT_BC_SHA256}" \
        "${STUDY_RUNNER}"
)"
STUDY_JOB_ID="${STUDY_JOB_ID%%;*}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_DIR="${VFI_RESULTS_ROOT}/runs/submissions/iq_domain_depth_${STAMP}"
mkdir -p "${SUBMISSION_DIR}"
SUBMISSION_DIR="${SUBMISSION_DIR}" BC_JOB_ID="${BC_JOB_ID}" BC_ROOT="${BC_ROOT}" PILOT_JOB_ID="${PILOT_JOB_ID}" STUDY_JOB_ID="${STUDY_JOB_ID}" LOCKED_RECIPE="${LOCKED_RECIPE}" PILOT_SHA256="${PILOT_SHA256}" MATRIX_SHA256="${MATRIX_SHA256}" STUDY_SHA256="${STUDY_SHA256}" TRAINER_SHA256="${TRAINER_SHA256}" CORE_SHA256="${CORE_SHA256}" MODELS_SHA256="${MODELS_SHA256}" RECURRENT_BC_SHA256="${RECURRENT_BC_SHA256}" python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "study": "online_recurrent_iq_domain_depth_v1",
    "execution": "cross_domain_pilot_then_confirmation_first_serial_12_cells",
    "bc_dependency_job_id": os.environ["BC_JOB_ID"],
    "bc_root": os.environ["BC_ROOT"],
    "pilot_job_id": os.environ["PILOT_JOB_ID"],
    "production_job_id": os.environ["STUDY_JOB_ID"],
    "locked_recipe": os.environ["LOCKED_RECIPE"],
    "domains": ["us", "japanese"], "transformer_layers": [2, 3],
    "policy_seeds": [0, 1, 2], "model_count": 12,
    "dependencies": [
        f"pilot afterok:{os.environ['BC_JOB_ID']}",
        f"production afterok:{os.environ['PILOT_JOB_ID']}",
    ],
    "source_sha256": {
        "pilot": os.environ["PILOT_SHA256"], "matrix": os.environ["MATRIX_SHA256"],
        "study": os.environ["STUDY_SHA256"], "trainer": os.environ["TRAINER_SHA256"],
        "recurrent_iq": os.environ["CORE_SHA256"], "models": os.environ["MODELS_SHA256"],
        "recurrent_bc": os.environ["RECURRENT_BC_SHA256"],
    },
}
out = Path(os.environ["SUBMISSION_DIR"])
(out / "submission.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "IQ convergence pilot job: ${PILOT_JOB_ID} (afterok:${BC_JOB_ID})"
echo "IQ serial 12-checkpoint job: ${STUDY_JOB_ID} (afterok:${PILOT_JOB_ID})"
echo "Submission manifest: ${SUBMISSION_DIR}/submission.json"
