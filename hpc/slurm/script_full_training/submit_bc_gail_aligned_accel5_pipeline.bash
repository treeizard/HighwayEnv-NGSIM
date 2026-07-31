#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
REPODIR="$(cd -- "${REPODIR}" && pwd)"
export REPODIR
source "${REPODIR}/hpc/slurm/project_env.bash"

COLLECTION_RUNNER="${REPODIR}/hpc/slurm/script_data_collection/collect_domain_matched_expert_accel5_array.bash"
AUDIT_RUNNER="${REPODIR}/hpc/slurm/script_data_collection/audit_domain_matched_expert_accel5.bash"
BC_RUNNER="${REPODIR}/hpc/slurm/script_full_training/run_bc_gail_aligned_accel5.bash"
LOCKED_RECIPE="${REPODIR}/configs/bc_gail_aligned_accel5_v1.json"
COLLECTION_ID="${COLLECTION_ID:-domain_matched_accel5_v2}"
COLLECTION_ROOT="${VFI_DATA_ROOT}/expert/${COLLECTION_ID}"

command -v sbatch >/dev/null 2>&1
for required in \
    "${COLLECTION_RUNNER}" \
    "${AUDIT_RUNNER}" \
    "${BC_RUNNER}" \
    "${LOCKED_RECIPE}"; do
    test -s "${required}"
done
if [ -e "${COLLECTION_ROOT}" ]; then
    echo "Refusing to reuse expert collection root: ${COLLECTION_ROOT}" >&2
    exit 2
fi
mkdir -p \
    "${VFI_LOG_ROOT}/slurm" \
    "${VFI_RESULTS_ROOT}/runs/submissions"

COLLECTOR_SHA256="$(sha256sum "${REPODIR}/scripts_gail/build_ps_traj_expert_discrete.py" | awk '{print $1}')"
CONSTANTS_SHA256="$(sha256sum "${REPODIR}/highway_env/ngsim_utils/core/constants.py" | awk '{print $1}')"
CONTRACTS_SHA256="$(sha256sum "${REPODIR}/scripts_gail/ps_gail/contracts.py" | awk '{print $1}')"
REPLAY_SHA256="$(sha256sum "${REPODIR}/highway_env/ngsim_utils/vehicles/replay.py" | awk '{print $1}')"
TRAJECTORY_GEN_SHA256="$(sha256sum "${REPODIR}/highway_env/ngsim_utils/data/trajectory_gen.py" | awk '{print $1}')"
NGSIM_ENV_SHA256="$(sha256sum "${REPODIR}/highway_env/envs/ngsim_env.py" | awk '{print $1}')"
LIDAR_SHA256="$(sha256sum "${REPODIR}/highway_env/envs/common/observations/lidar.py" | awk '{print $1}')"
AUDIT_SCRIPT_SHA256="$(sha256sum "${REPODIR}/scripts_gail/audit_domain_matched_expert.py" | awk '{print $1}')"
RECIPE_SHA256="$(sha256sum "${LOCKED_RECIPE}" | awk '{print $1}')"
BC_RUNNER_SHA256="$(sha256sum "${BC_RUNNER}" | awk '{print $1}')"
MATRIX_SHA256="$(sha256sum "${REPODIR}/scripts_gail/run_bc_domain_depth_matrix.py" | awk '{print $1}')"
TRAINER_SHA256="$(sha256sum "${REPODIR}/scripts_gail/train_recurrent_bc_policy.py" | awk '{print $1}')"
RECURRENT_BC_SHA256="$(sha256sum "${REPODIR}/scripts_gail/ps_gail/recurrent_bc.py" | awk '{print $1}')"
DATA_SHA256="$(sha256sum "${REPODIR}/scripts_gail/ps_gail/data.py" | awk '{print $1}')"

COLLECTION_JOB_ID="$(
    sbatch --parsable \
        --array=0,1,3,4 \
        --chdir="${VFI_PROJECT_ROOT}" \
        --export="ALL,REPODIR=${REPODIR},COLLECTION_ID=${COLLECTION_ID},EXPERT_ACCELERATION_LIMIT_MPS2=5.0,COLLECTION_EXPECTED_COLLECTOR_SHA256=${COLLECTOR_SHA256},COLLECTION_EXPECTED_CONSTANTS_SHA256=${CONSTANTS_SHA256},COLLECTION_EXPECTED_CONTRACTS_SHA256=${CONTRACTS_SHA256},COLLECTION_EXPECTED_REPLAY_SHA256=${REPLAY_SHA256},COLLECTION_EXPECTED_TRAJECTORY_GEN_SHA256=${TRAJECTORY_GEN_SHA256},COLLECTION_EXPECTED_NGSIM_ENV_SHA256=${NGSIM_ENV_SHA256},COLLECTION_EXPECTED_LIDAR_SHA256=${LIDAR_SHA256}" \
        "${COLLECTION_RUNNER}"
)"
COLLECTION_JOB_ID="${COLLECTION_JOB_ID%%;*}"

AUDIT_JOB_ID="$(
    sbatch --parsable \
        --dependency="afterok:${COLLECTION_JOB_ID}" \
        --chdir="${VFI_PROJECT_ROOT}" \
        --export="ALL,REPODIR=${REPODIR},COLLECTION_ID=${COLLECTION_ID},EXPERT_ACCELERATION_LIMIT_MPS2=5.0,AUDIT_SPLITS=train:val,AUDIT_EXPECTED_SCRIPT_SHA256=${AUDIT_SCRIPT_SHA256}" \
        "${AUDIT_RUNNER}"
)"
AUDIT_JOB_ID="${AUDIT_JOB_ID%%;*}"

BC_JOB_ID="$(
    sbatch --parsable \
        --dependency="afterok:${AUDIT_JOB_ID}" \
        --chdir="${VFI_PROJECT_ROOT}" \
        --export="ALL,REPODIR=${REPODIR},COLLECTION_ID=${COLLECTION_ID},BC_LOCKED_RECIPE=${LOCKED_RECIPE},BC_EXPECTED_RECIPE_SHA256=${RECIPE_SHA256},BC_EXPECTED_RUNNER_SHA256=${BC_RUNNER_SHA256},BC_EXPECTED_MATRIX_SHA256=${MATRIX_SHA256},BC_EXPECTED_TRAINER_SHA256=${TRAINER_SHA256},BC_EXPECTED_RECURRENT_BC_SHA256=${RECURRENT_BC_SHA256},BC_EXPECTED_CONTRACTS_SHA256=${CONTRACTS_SHA256},BC_EXPECTED_DATA_SHA256=${DATA_SHA256}" \
        "${BC_RUNNER}"
)"
BC_JOB_ID="${BC_JOB_ID%%;*}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_DIR="${VFI_RESULTS_ROOT}/runs/submissions/bc_gail_aligned_accel5_${STAMP}"
mkdir -p "${SUBMISSION_DIR}"
SUBMISSION_JSON="${SUBMISSION_DIR}/submission.json"
SUBMISSION_JSON="${SUBMISSION_JSON}" \
COLLECTION_JOB_ID="${COLLECTION_JOB_ID}" \
AUDIT_JOB_ID="${AUDIT_JOB_ID}" \
BC_JOB_ID="${BC_JOB_ID}" \
COLLECTION_ROOT="${COLLECTION_ROOT}" \
LOCKED_RECIPE="${LOCKED_RECIPE}" \
RECIPE_SHA256="${RECIPE_SHA256}" \
BC_RUNNER_SHA256="${BC_RUNNER_SHA256}" \
COLLECTOR_SHA256="${COLLECTOR_SHA256}" \
CONSTANTS_SHA256="${CONSTANTS_SHA256}" \
CONTRACTS_SHA256="${CONTRACTS_SHA256}" \
AUDIT_SCRIPT_SHA256="${AUDIT_SCRIPT_SHA256}" \
MATRIX_SHA256="${MATRIX_SHA256}" \
TRAINER_SHA256="${TRAINER_SHA256}" \
RECURRENT_BC_SHA256="${RECURRENT_BC_SHA256}" \
DATA_SHA256="${DATA_SHA256}" \
python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": 1,
    "study": "bc_gail_aligned_accel5_v1",
    "status": "submitted",
    "dependency_chain": [
        {"stage": "collect_six_domain_splits", "job_id": os.environ["COLLECTION_JOB_ID"]},
        {"stage": "audit_action_and_sensor_contracts", "job_id": os.environ["AUDIT_JOB_ID"]},
        {"stage": "train_and_evaluate_twelve_bc_models", "job_id": os.environ["BC_JOB_ID"]},
    ],
    "collection_root": os.environ["COLLECTION_ROOT"],
    "locked_recipe": os.environ["LOCKED_RECIPE"],
    "policy_root": (
        f"{os.environ.get('VFI_RESULTS_ROOT', '')}/runs/policies/bc/"
        f"gail_aligned_accel5_{os.environ['BC_JOB_ID']}"
    ),
    "checkpoint_archive": (
        f"{os.environ.get('VFI_CHECKPOINT_ROOT', '')}/bc/validated_comparisons/"
        f"gail_aligned_accel5_{os.environ['BC_JOB_ID']}"
    ),
    "source_sha256": {
        "recipe": os.environ["RECIPE_SHA256"],
        "bc_runner": os.environ["BC_RUNNER_SHA256"],
        "collector": os.environ["COLLECTOR_SHA256"],
        "constants": os.environ["CONSTANTS_SHA256"],
        "contracts": os.environ["CONTRACTS_SHA256"],
        "audit": os.environ["AUDIT_SCRIPT_SHA256"],
        "matrix": os.environ["MATRIX_SHA256"],
        "trainer": os.environ["TRAINER_SHA256"],
        "recurrent_bc": os.environ["RECURRENT_BC_SHA256"],
        "data": os.environ["DATA_SHA256"],
    },
}
path = Path(os.environ["SUBMISSION_JSON"])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Expert collection array: ${COLLECTION_JOB_ID}"
echo "Contract audit: ${AUDIT_JOB_ID} (afterok:${COLLECTION_JOB_ID})"
echo "Aligned BC matrix: ${BC_JOB_ID} (afterok:${AUDIT_JOB_ID})"
echo "Submission manifest: ${SUBMISSION_JSON}"
