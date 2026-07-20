#!/bin/bash
#SBATCH --job-name=finalize_bc_study
#SBATCH --account=bt60
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/finalize_bc_study_%j.out
#SBATCH --error=logs/slurm/finalize_bc_study_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

mkdir -p "${PYTHONPYCACHEPREFIX}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

POLICY_ROOT="${VFI_RESULTS_ROOT}/runs/policies/bc"
SMOKE_ROOT="${VFI_RESULTS_ROOT}/test_runs/bc_interpretability_smoke"
STUDY_MANIFEST="${POLICY_ROOT}/study_manifest.json"

cd "${REPODIR}"
python -m scripts_gail.finalize_bc_domain_depth_study \
    --policy-root "${POLICY_ROOT}" \
    --smoke-root "${SMOKE_ROOT}" \
    --out "${STUDY_MANIFEST}"

cd "${VFI_PROJECT_ROOT}"
python -m workflows.analysis.build_policy_checkpoint_registry \
    --policy-root "${POLICY_ROOT}" \
    --sae-root "${VFI_RESULTS_ROOT}/runs/sae" \
    --out "${VFI_RESULTS_ROOT}/runs/policy_registry/bc_domain_depth_v1"

echo "Verified study manifest: ${STUDY_MANIFEST}"
