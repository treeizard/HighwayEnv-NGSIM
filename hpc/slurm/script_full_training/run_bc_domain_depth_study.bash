#!/bin/bash
#SBATCH --job-name=bc_domain_study
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/bc_domain_study_%j.out
#SBATCH --error=logs/slurm/bc_domain_study_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

assert_sha256() {
    local expected="$1"
    local path="$2"
    if [ -n "${expected}" ]; then
        local actual
        actual="$(sha256sum "${path}" | awk '{print $1}')"
        if [ "${actual}" != "${expected}" ]; then
            echo "Submission-locked source changed: ${path}" >&2
            echo "expected ${expected}, got ${actual}" >&2
            exit 2
        fi
    fi
}

assert_sha256 "${BC_EXPECTED_MATRIX_SHA256:-}" "${REPODIR}/scripts_gail/run_bc_domain_depth_matrix.py"
assert_sha256 "${BC_EXPECTED_TRAINER_SHA256:-}" "${REPODIR}/scripts_gail/train_recurrent_bc_policy.py"
assert_sha256 "${BC_EXPECTED_RECURRENT_BC_SHA256:-}" "${REPODIR}/scripts_gail/ps_gail/recurrent_bc.py"
assert_sha256 "${BC_EXPECTED_MODELS_SHA256:-}" "${REPODIR}/scripts_gail/ps_gail/models.py"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_bc_study_${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

COLLECTION_ROOT="${VFI_DATA_ROOT}/expert/${COLLECTION_ID:-domain_matched_accel5_v2}"
US_TRAIN_EXPERT="${COLLECTION_ROOT}/us/train"
US_VALIDATION_EXPERT="${COLLECTION_ROOT}/us/val"
JAPANESE_TRAIN_EXPERT="${COLLECTION_ROOT}/japanese/train"
JAPANESE_VALIDATION_EXPERT="${COLLECTION_ROOT}/japanese/val"
LOCKED_RECIPE="${BC_LOCKED_RECIPE:-${VFI_PROJECT_ROOT}/configs/bc_gail_aligned_accel5_v5.json}"
for required in \
    "${US_TRAIN_EXPERT}/manifest.json" \
    "${US_VALIDATION_EXPERT}/manifest.json" \
    "${JAPANESE_TRAIN_EXPERT}/manifest.json" \
    "${JAPANESE_VALIDATION_EXPERT}/manifest.json" \
    "${LOCKED_RECIPE}"; do
    test -s "${required}"
done
assert_sha256 "${BC_EXPECTED_RECIPE_SHA256:-}" "${LOCKED_RECIPE}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the BC study environment")
print(f"BC matrix device: {torch.cuda.get_device_name(0)}")
PY

if [ "${BC_PRODUCTION_SUBMISSION:-0}" = "1" ]; then
    STUDY_RUN_ID="${SLURM_JOB_ID:?Production BC workflow requires a Slurm job id}"
    POLICY_ROOT="${VFI_RESULTS_ROOT}/runs/policies/bc/domain_depth_recovered_${STUDY_RUN_ID}"
    ACTIVATION_ROOT="${VFI_RESULTS_ROOT}/test_runs/bc_interpretability_smoke/domain_depth_recovered_${STUDY_RUN_ID}"
    CHECKPOINT_ARCHIVE_ROOT="${VFI_CHECKPOINT_ROOT}/bc/autoregressive_policy_comparison"
    REGISTRY_ROOT="${VFI_RESULTS_ROOT}/runs/policy_registry/bc_domain_depth_recovered_${STUDY_RUN_ID}"
    MODEL_LIMIT=0
else
    STUDY_RUN_ID="${BC_STUDY_RUN_ID:-${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}}"
    POLICY_ROOT="${BC_STUDY_POLICY_ROOT:-${VFI_RESULTS_ROOT}/runs/policies/bc/domain_depth_recovered_${STUDY_RUN_ID}}"
    ACTIVATION_ROOT="${BC_STUDY_ACTIVATION_ROOT:-${VFI_RESULTS_ROOT}/test_runs/bc_interpretability_smoke/domain_depth_recovered_${STUDY_RUN_ID}}"
    CHECKPOINT_ARCHIVE_ROOT="${BC_CHECKPOINT_ARCHIVE_ROOT:-${VFI_CHECKPOINT_ROOT}/bc/autoregressive_policy_comparison}"
    REGISTRY_ROOT="${BC_STUDY_REGISTRY_ROOT:-${VFI_RESULTS_ROOT}/runs/policy_registry/bc_domain_depth_recovered_${STUDY_RUN_ID}}"
    MODEL_LIMIT="${BC_STUDY_MODEL_LIMIT:-0}"
fi

cd "${REPODIR}"
python -m scripts_gail.run_bc_domain_depth_matrix \
    --recipe "${LOCKED_RECIPE}" \
    --us-train-expert "${US_TRAIN_EXPERT}" \
    --us-validation-expert "${US_VALIDATION_EXPERT}" \
    --japanese-train-expert "${JAPANESE_TRAIN_EXPERT}" \
    --japanese-validation-expert "${JAPANESE_VALIDATION_EXPERT}" \
    --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
    --policy-root "${POLICY_ROOT}" \
    --checkpoint-archive-root "${CHECKPOINT_ARCHIVE_ROOT}" \
    --study-id "recovered_${STUDY_RUN_ID}" \
    --model-limit "${MODEL_LIMIT}" \
    --device cuda

MATRIX_MANIFEST="${POLICY_ROOT}/matrix_manifest.json"
MATRIX_MANIFEST="${MATRIX_MANIFEST}" MODEL_LIMIT="${MODEL_LIMIT}" python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["MATRIX_MANIFEST"]).read_text(encoding="utf-8"))
expected_loads = {
    "us": {"train": 1, "validation": 1, "test": 0},
    "japanese": {"train": 1, "validation": 1, "test": 0},
}
if manifest.get("loader_calls") != expected_loads:
    raise SystemExit(f"Explicit sources were not loaded exactly once: {manifest.get('loader_calls')}")
requested_limit = int(os.environ["MODEL_LIMIT"])
expected_models = (
    requested_limit
    if requested_limit > 0
    else 2
    * len(manifest.get("recipe_depths", []))
    * len(manifest.get("recipe_policy_seeds", []))
)
if int(manifest.get("model_count", -1)) != expected_models:
    raise SystemExit("Matrix model count does not match the requested limit")
if int(manifest.get("training_artifact_complete_count", -1)) != expected_models:
    raise SystemExit(
        "The validation matrix did not produce every requested artifact."
    )
PY

echo "Completed validation-only BC matrix: ${MATRIX_MANIFEST}"
echo "Locked test, expert-relative qualification, activation collection, and finalization remain pending."
