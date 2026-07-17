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

STUDY_JOB_ID="$(
    sbatch --parsable \
        --chdir="${VFI_PROJECT_ROOT}" \
        --export="ALL,REPODIR=${REPODIR}" \
        "${RUNNER}"
)"
STUDY_JOB_ID="${STUDY_JOB_ID%%;*}"
POLICY_ROOT="${VFI_RESULTS_ROOT}/runs/policies/bc/domain_depth_${STUDY_JOB_ID}"
ACTIVATION_ROOT="${VFI_RESULTS_ROOT}/test_runs/bc_interpretability_smoke/domain_depth_${STUDY_JOB_ID}"
REGISTRY_ROOT="${VFI_RESULTS_ROOT}/runs/policy_registry/bc_domain_depth_${STUDY_JOB_ID}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_DIR="${VFI_RESULTS_ROOT}/runs/submissions/bc_domain_depth_${STAMP}"
mkdir -p "${SUBMISSION_DIR}"
SUBMISSION_DIR="${SUBMISSION_DIR}" STUDY_JOB_ID="${STUDY_JOB_ID}" POLICY_ROOT="${POLICY_ROOT}" ACTIVATION_ROOT="${ACTIVATION_ROOT}" REGISTRY_ROOT="${REGISTRY_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["SUBMISSION_DIR"])
payload = {
    "study": "bc_domain_depth_v1",
    "job_id": os.environ["STUDY_JOB_ID"],
    "execution": "single_job_smoke_first_serial",
    "stages": ["gpu_smoke", "full_bc_training", "checkpoint_validation", "activation_smoke", "finalization"],
    "policy_root": os.environ["POLICY_ROOT"],
    "activation_root": os.environ["ACTIVATION_ROOT"],
    "registry_root": os.environ["REGISTRY_ROOT"],
    "domains": ["us", "japanese"],
    "transformer_layers": [2, 3],
    "policy_seeds": [0, 1, 2],
    "model_count": 12,
}
(out / "submission.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Full BC smoke-first serial 12-model job: ${STUDY_JOB_ID}"
echo "Submission manifest: ${SUBMISSION_DIR}/submission.json"
