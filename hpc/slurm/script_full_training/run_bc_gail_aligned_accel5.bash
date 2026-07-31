#!/bin/bash
#SBATCH --job-name=bc_gail_a5
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/bc_gail_a5_%j.out
#SBATCH --error=logs/slurm/bc_gail_a5_%j.err

set -euo pipefail

: "${REPODIR:?Submit this script with the absolute component REPODIR exported}"
REPODIR="$(cd -- "${REPODIR}" && pwd)"
export REPODIR
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

assert_sha256 "${BC_EXPECTED_RECIPE_SHA256:-}" "${BC_LOCKED_RECIPE}"
assert_sha256 "${BC_EXPECTED_RUNNER_SHA256:-}" \
    "${REPODIR}/hpc/slurm/script_full_training/run_bc_gail_aligned_accel5.bash"
assert_sha256 "${BC_EXPECTED_MATRIX_SHA256:-}" \
    "${REPODIR}/scripts_gail/run_bc_domain_depth_matrix.py"
assert_sha256 "${BC_EXPECTED_TRAINER_SHA256:-}" \
    "${REPODIR}/scripts_gail/train_recurrent_bc_policy.py"
assert_sha256 "${BC_EXPECTED_RECURRENT_BC_SHA256:-}" \
    "${REPODIR}/scripts_gail/ps_gail/recurrent_bc.py"
assert_sha256 "${BC_EXPECTED_MODELS_SHA256:-}" \
    "${REPODIR}/scripts_gail/ps_gail/models.py"
assert_sha256 "${BC_EXPECTED_CONTRACTS_SHA256:-}" \
    "${REPODIR}/scripts_gail/ps_gail/contracts.py"
assert_sha256 "${BC_EXPECTED_DATA_SHA256:-}" \
    "${REPODIR}/scripts_gail/ps_gail/data.py"
assert_sha256 "${BC_EXPECTED_CHECKPOINTS_SHA256:-}" \
    "${REPODIR}/scripts_gail/ps_gail/checkpoints.py"
assert_sha256 "${BC_EXPECTED_ARCHIVE_SHA256:-}" \
    "${REPODIR}/scripts_gail/archive_bc_checkpoints.py"

export NGSIM_ACCELERATION_LIMIT_MPS2=5.0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPODIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_bc_gail_a5_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COLLECTION_ID="${COLLECTION_ID:-domain_matched_accel5_v2}"
COLLECTION_ROOT="${VFI_DATA_ROOT}/expert/${COLLECTION_ID}"
AUDIT_PATH="${BC_AUDIT_PATH:-${COLLECTION_ROOT}/collection_contract_audit_train_val.json}"
US_TRAIN_EXPERT="${COLLECTION_ROOT}/us/train"
US_VALIDATION_EXPERT="${COLLECTION_ROOT}/us/val"
JAPANESE_TRAIN_EXPERT="${COLLECTION_ROOT}/japanese/train"
JAPANESE_VALIDATION_EXPERT="${COLLECTION_ROOT}/japanese/val"
STUDY_RUN_ID="${SLURM_JOB_ID:?BC comparison requires a Slurm job id}"
POLICY_ROOT="${VFI_RESULTS_ROOT}/runs/policies/bc/gail_aligned_accel5_${STUDY_RUN_ID}"
CHECKPOINT_ARCHIVE_ROOT="${VFI_CHECKPOINT_ROOT}/bc/validated_comparisons"

for required in \
    "${AUDIT_PATH}" \
    "${US_TRAIN_EXPERT}/manifest.json" \
    "${US_VALIDATION_EXPERT}/manifest.json" \
    "${JAPANESE_TRAIN_EXPERT}/manifest.json" \
    "${JAPANESE_VALIDATION_EXPERT}/manifest.json" \
    "${BC_LOCKED_RECIPE}"; do
    test -s "${required}"
done
AUDIT_PATH="${AUDIT_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["AUDIT_PATH"]).read_text(encoding="utf-8"))
if (
    payload.get("status") != "passed"
    or payload.get("audited_splits") != ["train", "val"]
    or payload.get("test_data_status") != "not_opened"
    or int(payload.get("domain_split_count", 0)) != 4
):
    raise SystemExit(
        "Expert collection did not pass the explicit train/validation-only "
        f"four-cell audit: {payload}"
    )
scales = payload.get("continuous_action_contract", {}).get("scales")
if scales is None or abs(float(scales[0]) - 5.0) > 1e-8:
    raise SystemExit(f"Expert collection has the wrong acceleration scale: {scales}")
PY

mkdir -p \
    "${VFI_LOG_ROOT}/slurm" \
    "${PYTHONPYCACHEPREFIX}" \
    "${MPLCONFIGDIR}" \
    "${CHECKPOINT_ARCHIVE_ROOT}"
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

python - <<'PY'
import torch
from highway_env.ngsim_utils.core.constants import ACCELERATION_RANGE

if ACCELERATION_RANGE != (-5.0, 5.0):
    raise SystemExit(f"Wrong runtime acceleration contract: {ACCELERATION_RANGE}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the aligned BC job")
print(f"Aligned BC device: {torch.cuda.get_device_name(0)}")
PY

cd "${REPODIR}"
MODEL_COUNT="$(
    python - "${BC_LOCKED_RECIPE}" <<'PY'
import json
import sys
from pathlib import Path

recipe = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
depths = recipe["architecture"]["depths"]
seeds = recipe["architecture"]["policy_seeds"]
print(2 * len(depths) * len(seeds))
PY
)"
export MODEL_COUNT
python -m scripts_gail.run_bc_domain_depth_matrix \
    --recipe "${BC_LOCKED_RECIPE}" \
    --us-train-expert "${US_TRAIN_EXPERT}" \
    --us-validation-expert "${US_VALIDATION_EXPERT}" \
    --japanese-train-expert "${JAPANESE_TRAIN_EXPERT}" \
    --japanese-validation-expert "${JAPANESE_VALIDATION_EXPERT}" \
    --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
    --policy-root "${POLICY_ROOT}" \
    --checkpoint-archive-root "${CHECKPOINT_ARCHIVE_ROOT}" \
    --study-id "gail_aligned_accel5_${STUDY_RUN_ID}" \
    --model-limit "${MODEL_COUNT}" \
    --priority-cells-first \
    --device cuda

MATRIX_MANIFEST="${POLICY_ROOT}/matrix_manifest.json"
MATRIX_MANIFEST="${MATRIX_MANIFEST}" python - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["MATRIX_MANIFEST"]).read_text(encoding="utf-8"))
expected_loader_calls = {
    "us": {"train": 1, "validation": 1, "test": 0},
    "japanese": {"train": 1, "validation": 1, "test": 0},
}
if payload.get("loader_calls") != expected_loader_calls:
    raise SystemExit(f"Unexpected expert loader calls: {payload.get('loader_calls')}")
if payload.get("test_source_status") != "pending_deferred_not_accepted":
    raise SystemExit(f"Test source was not sealed: {payload.get('test_source_status')}")
expected = int(os.environ["MODEL_COUNT"])
if int(payload.get("model_count", -1)) != expected:
    raise SystemExit(f"The aligned BC job did not train all {expected} requested cells")
if int(payload.get("training_artifact_complete_count", -1)) != expected:
    raise SystemExit(
        "Not all aligned BC cells produced complete finite checkpoints. Offline "
        "and closed-loop diagnostics are preserved, but the requested artifact "
        "matrix is incomplete."
    )
archive = payload.get("checkpoint_archive") or {}
if int(archive.get("checkpoint_count", -1)) != expected:
    raise SystemExit(
        f"Only {archive.get('checkpoint_count')}/{expected} complete BC "
        "checkpoints were archived"
    )
print(json.dumps({
    "status": "artifact_matrix_completed",
    "scientific_qualification_status": "pending_locked_test_and_expert_replay",
    "interpretability_qualification_passed": False,
    "policy_root": str(Path(os.environ["MATRIX_MANIFEST"]).parent),
    "training_artifact_complete_count": expected,
    "interpretability_baseline_eligible_count": payload[
        "interpretability_baseline_eligible_count"
    ],
    "metric_capability_passed_count": payload["metric_capability_passed_count"],
    "closed_loop_quality_passed_count": payload[
        "closed_loop_quality_passed_count"
    ],
    "validation_candidate_selected_count": payload[
        "validation_candidate_selected_count"
    ],
    "validation_candidate_selections": payload[
        "validation_candidate_selections"
    ],
    "development_fallback_selected_count": payload[
        "development_fallback_selected_count"
    ],
    "closed_loop_metrics_role": "qualification_non_terminal_for_matrix_execution",
    "checkpoint_archive": archive.get("archive"),
    "matrix_manifest": os.environ["MATRIX_MANIFEST"],
}, indent=2))
PY

echo "Aligned BC matrix: ${MATRIX_MANIFEST}"
echo "Complete BC reference archive: ${CHECKPOINT_ARCHIVE_ROOT}/gail_aligned_accel5_${STUDY_RUN_ID}"
