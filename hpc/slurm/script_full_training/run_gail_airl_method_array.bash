#!/bin/bash
#SBATCH --job-name=gail_airl_method
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

: "${REPODIR:?Set REPODIR to the immutable isolated HighwayEnv-NGSIM checkout}"
: "${METHOD:?Set METHOD to gail or airl}"
: "${STUDY_MANIFEST:?Set STUDY_MANIFEST to a prepared method-specific manifest}"
: "${VFI_PROJECT_ROOT:?Set VFI_PROJECT_ROOT explicitly to the canonical project root}"
: "${VFI_DATA_ROOT:?Set VFI_DATA_ROOT explicitly}"
: "${VFI_HIGHWAY_DATA_ROOT:?Set VFI_HIGHWAY_DATA_ROOT explicitly}"
: "${VFI_CHECKPOINT_ROOT:?Set VFI_CHECKPOINT_ROOT explicitly}"
: "${VFI_RESULTS_ROOT:?Set VFI_RESULTS_ROOT explicitly}"
: "${VFI_LOG_ROOT:?Set VFI_LOG_ROOT explicitly}"
: "${VFI_ARTIFACT_ROOT:?Set VFI_ARTIFACT_ROOT explicitly}"
: "${SLURM_ARRAY_TASK_ID:?Submit this runner with an explicit bounded --array}"

case "${METHOD}" in
    gail|airl) ;;
    *) echo "METHOD must be gail or airl, got: ${METHOD}" >&2; exit 2 ;;
esac

REPODIR="$(cd "${REPODIR}" && pwd)"
STUDY_MANIFEST="$(realpath "${STUDY_MANIFEST}")"
source "${REPODIR}/hpc/slurm/project_env.bash"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_gail_airl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the GAIL/AIRL array environment")
print(f"GAIL/AIRL array device: {torch.cuda.get_device_name(0)}")
PY

cd "${REPODIR}"
"${PYTHON_BIN}" -m scripts_gail.run_gail_airl_study_trial \
    --manifest "${STUDY_MANIFEST}" \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --expected-method "${METHOD}" \
    --python "${PYTHON_BIN}"
