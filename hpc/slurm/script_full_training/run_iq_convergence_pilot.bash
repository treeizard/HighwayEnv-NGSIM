#!/bin/bash
#SBATCH --job-name=iq_convergence_pilot
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/iq_convergence_pilot_%j.out
#SBATCH --error=logs/slurm/iq_convergence_pilot_%j.err

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
        test "${actual}" = "${expected}"
    fi
}

assert_sha256 "${IQ_EXPECTED_PILOT_SHA256:-}" "${REPODIR}/scripts_gail/run_iq_convergence_pilot.py"
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
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_iq_pilot_${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

BC_ROOT="${IQ_BC_ROOT:?IQ_BC_ROOT must identify the completed recovered BC matrix}"
PILOT_ROOT="${IQ_PILOT_ROOT:-${VFI_RESULTS_ROOT}/test_runs/iq_convergence_pilot/${SLURM_JOB_ID:-local}}"
LOCKED_RECIPE="${IQ_LOCKED_RECIPE_OUT:-${VFI_RESULTS_ROOT}/runs/policies/iq/pilot_${SLURM_JOB_ID:-local}/locked_recipe.json}"
US_EXPERT="${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982"
JAPANESE_EXPERT="${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train"
test -s "${BC_ROOT}/matrix_manifest.json"
test -s "${US_EXPERT}/manifest.json"
test -s "${JAPANESE_EXPERT}/manifest.json"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the IQ pilot environment")
print(f"IQ pilot device: {torch.cuda.get_device_name(0)}")
PY

cd "${REPODIR}"
python -m scripts_gail.run_iq_convergence_pilot \
    --bc-root "${BC_ROOT}" \
    --us-expert "${US_EXPERT}" \
    --japanese-expert "${JAPANESE_EXPERT}" \
    --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
    --out-root "${PILOT_ROOT}" \
    --locked-recipe-out "${LOCKED_RECIPE}" \
    --device cuda

test -s "${LOCKED_RECIPE}"
echo "Locked IQ recipe: ${LOCKED_RECIPE}"
