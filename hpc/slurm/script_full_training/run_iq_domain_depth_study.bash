#!/bin/bash
#SBATCH --job-name=iq_domain_study
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/iq_domain_study_%j.out
#SBATCH --error=logs/slurm/iq_domain_study_%j.err
#SBATCH --requeue

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
            exit 2
        fi
    fi
}

assert_sha256 "${IQ_EXPECTED_MATRIX_SHA256:-}" "${REPODIR}/scripts_gail/run_iq_domain_depth_matrix.py"
assert_sha256 "${IQ_EXPECTED_STUDY_SHA256:-}" "${REPODIR}/scripts_gail/iq_study.py"
assert_sha256 "${IQ_EXPECTED_TRAINER_SHA256:-}" "${REPODIR}/scripts_gail/train_recurrent_iq_learn.py"
assert_sha256 "${IQ_EXPECTED_CORE_SHA256:-}" "${REPODIR}/scripts_gail/ps_gail/recurrent_iq.py"
assert_sha256 "${IQ_EXPECTED_MODELS_SHA256:-}" "${REPODIR}/scripts_gail/ps_gail/models.py"
assert_sha256 "${IQ_EXPECTED_RECURRENT_BC_SHA256:-}" "${REPODIR}/scripts_gail/ps_gail/recurrent_bc.py"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_iq_study_${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

BC_ROOT="${IQ_BC_ROOT:?IQ_BC_ROOT must identify the completed recovered BC matrix}"
LOCKED_RECIPE="${IQ_LOCKED_RECIPE:?IQ_LOCKED_RECIPE must identify the pilot-selected recipe}"
US_EXPERT="${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982"
JAPANESE_EXPERT="${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train"
STUDY_RUN_ID="${SLURM_JOB_ID:?Production IQ workflow requires a Slurm job id}"
POLICY_ROOT="${VFI_RESULTS_ROOT}/runs/policies/iq/domain_depth_${STUDY_RUN_ID}"
ACTIVATION_ROOT="${VFI_RESULTS_ROOT}/test_runs/iq_interpretability_smoke/domain_depth_${STUDY_RUN_ID}"
REGISTRY_ROOT="${VFI_RESULTS_ROOT}/runs/policy_registry/iq_domain_depth_${STUDY_RUN_ID}"

for required in "${BC_ROOT}/matrix_manifest.json" "${LOCKED_RECIPE}" "${US_EXPERT}/manifest.json" "${JAPANESE_EXPERT}/manifest.json"; do
    test -s "${required}"
done

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the IQ production environment")
print(f"IQ matrix device: {torch.cuda.get_device_name(0)}")
PY

cd "${REPODIR}"
python -m scripts_gail.run_iq_domain_depth_matrix \
    --recipe "${LOCKED_RECIPE}" \
    --bc-root "${BC_ROOT}" \
    --us-expert "${US_EXPERT}" \
    --japanese-expert "${JAPANESE_EXPERT}" \
    --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
    --policy-root "${POLICY_ROOT}" \
    --study-id "iq_${STUDY_RUN_ID}" \
    --model-limit 12 \
    --resume \
    --device cuda

run_activation_check() {
    local checkpoint="$1"
    local expert_data="$2"
    local output_dir="$3"
    local layers="$4"
    cd "${VFI_PROJECT_ROOT}"
    python -m interpretability.sae.cli.inspect_checkpoint --checkpoint "${checkpoint}" --device cpu
    python -m interpretability.sae.cli.collect_activations \
        --checkpoint "${checkpoint}" --expert-data "${expert_data}" --out "${output_dir}" \
        --signal-target residual_policy_tokens --layers "${layers}" --split all \
        --max-transitions 64 --max-files 2 --max-vehicles-per-file 1 --shard-size 64 --device cuda
    for layer in ${layers//,/ }; do
        test -s "${output_dir}/residual_layer_${layer}_policy_token/manifest.json"
    done
}

for domain in us japanese; do
    if [ "${domain}" = us ]; then expert_data="${US_EXPERT}"; else expert_data="${JAPANESE_EXPERT}"; fi
    for depth in 2 3; do
        if [ "${depth}" -eq 2 ]; then capture_layers="0,1"; else capture_layers="0,1,2"; fi
        for seed in 0 1 2; do
            relative="${domain}/recurrent_transformer_${depth}layer/policy_seed_${seed}"
            run_activation_check "${POLICY_ROOT}/${relative}/best.pt" "${expert_data}" "${ACTIVATION_ROOT}/${relative}" "${capture_layers}"
        done
    done
done

cd "${VFI_PROJECT_ROOT}"
python -m workflows.analysis.build_policy_checkpoint_registry \
    --policy-root "${POLICY_ROOT}" --sae-root "${VFI_RESULTS_ROOT}/runs/sae" --out "${REGISTRY_ROOT}"
echo "Completed recurrent IQ-Learn 12-cell study: ${POLICY_ROOT}/matrix_manifest.json"
