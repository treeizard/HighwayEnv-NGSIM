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

US_EXPERT="${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982"
JAPANESE_EXPERT="${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train"
LOCKED_RECIPE="${BC_LOCKED_RECIPE:-${VFI_PROJECT_ROOT}/configs/bc_recovery_recipe.json}"
for required in "${US_EXPERT}/manifest.json" "${JAPANESE_EXPERT}/manifest.json" "${LOCKED_RECIPE}"; do
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
    MODEL_LIMIT=12
else
    STUDY_RUN_ID="${BC_STUDY_RUN_ID:-${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}}"
    POLICY_ROOT="${BC_STUDY_POLICY_ROOT:-${VFI_RESULTS_ROOT}/runs/policies/bc/domain_depth_recovered_${STUDY_RUN_ID}}"
    ACTIVATION_ROOT="${BC_STUDY_ACTIVATION_ROOT:-${VFI_RESULTS_ROOT}/test_runs/bc_interpretability_smoke/domain_depth_recovered_${STUDY_RUN_ID}}"
    CHECKPOINT_ARCHIVE_ROOT="${BC_CHECKPOINT_ARCHIVE_ROOT:-${VFI_CHECKPOINT_ROOT}/bc/autoregressive_policy_comparison}"
    REGISTRY_ROOT="${BC_STUDY_REGISTRY_ROOT:-${VFI_RESULTS_ROOT}/runs/policy_registry/bc_domain_depth_recovered_${STUDY_RUN_ID}}"
    MODEL_LIMIT="${BC_STUDY_MODEL_LIMIT:-12}"
fi

cd "${REPODIR}"
python -m scripts_gail.run_bc_domain_depth_matrix \
    --recipe "${LOCKED_RECIPE}" \
    --us-expert "${US_EXPERT}" \
    --japanese-expert "${JAPANESE_EXPERT}" \
    --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
    --policy-root "${POLICY_ROOT}" \
    --checkpoint-archive-root "${CHECKPOINT_ARCHIVE_ROOT}" \
    --study-id "recovered_${STUDY_RUN_ID}" \
    --model-limit "${MODEL_LIMIT}" \
    --require-confirmation \
    --device cuda

MATRIX_MANIFEST="${POLICY_ROOT}/matrix_manifest.json"
MATRIX_MANIFEST="${MATRIX_MANIFEST}" MODEL_LIMIT="${MODEL_LIMIT}" python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["MATRIX_MANIFEST"]).read_text(encoding="utf-8"))
if manifest.get("loader_calls") != {"us": 1, "japanese": 1}:
    raise SystemExit(f"Expert data was not loaded exactly once per domain: {manifest.get('loader_calls')}")
if int(manifest.get("model_count", -1)) != int(os.environ["MODEL_LIMIT"]):
    raise SystemExit("Matrix model count does not match the requested limit")
passed = int(manifest.get("metric_capability_passed_count", -1))
if passed != int(os.environ["MODEL_LIMIT"]):
    raise SystemExit(
        f"Only {passed}/{os.environ['MODEL_LIMIT']} BC cells passed the locked learning gate; "
        "preserving diagnostics but refusing activation collection and study promotion."
    )
PY

if [ "${MODEL_LIMIT}" -lt 12 ]; then
    echo "Completed requested load-once BC pilot cells: ${MODEL_LIMIT}"
    exit 0
fi

run_activation_check() {
    local checkpoint="$1"
    local expert_data="$2"
    local output_dir="$3"
    local layers="$4"

    cd "${VFI_PROJECT_ROOT}"
    python -m interpretability.sae.cli.inspect_checkpoint \
        --checkpoint "${checkpoint}" \
        --device cpu
    python -m interpretability.sae.cli.collect_activations \
        --checkpoint "${checkpoint}" \
        --expert-data "${expert_data}" \
        --out "${output_dir}" \
        --signal-target residual_policy_tokens \
        --layers "${layers}" \
        --split all \
        --max-transitions 64 \
        --max-files 2 \
        --max-vehicles-per-file 1 \
        --shard-size 64 \
        --device cuda
    for layer in ${layers//,/ }; do
        test -s "${output_dir}/residual_layer_${layer}_policy_token/manifest.json"
    done
}

for domain in us japanese; do
    if [ "${domain}" = us ]; then
        expert_data="${US_EXPERT}"
    else
        expert_data="${JAPANESE_EXPERT}"
    fi
    for depth in 2 3; do
        if [ "${depth}" -eq 2 ]; then
            capture_layers="0,1"
        else
            capture_layers="0,1,2"
        fi
        for seed in 0 1 2; do
            relative="${domain}/recurrent_transformer_${depth}layer/policy_seed_${seed}"
            run_activation_check \
                "${POLICY_ROOT}/${relative}/best.pt" \
                "${expert_data}" \
                "${ACTIVATION_ROOT}/${relative}" \
                "${capture_layers}"
        done
    done
done

STUDY_MANIFEST="${POLICY_ROOT}/study_manifest.json"
cd "${REPODIR}"
python -m scripts_gail.finalize_bc_domain_depth_study \
    --policy-root "${POLICY_ROOT}" \
    --smoke-root "${ACTIVATION_ROOT}" \
    --out "${STUDY_MANIFEST}"

cd "${VFI_PROJECT_ROOT}"
python -m workflows.analysis.build_policy_checkpoint_registry \
    --policy-root "${POLICY_ROOT}" \
    --sae-root "${VFI_RESULTS_ROOT}/runs/sae" \
    --out "${REGISTRY_ROOT}"

echo "Completed load-once serial BC study: ${STUDY_MANIFEST}"
echo "Qualified checkpoint archive: ${CHECKPOINT_ARCHIVE_ROOT}/recovered_${STUDY_RUN_ID}"
