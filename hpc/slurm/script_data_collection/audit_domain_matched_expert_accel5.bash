#!/bin/bash
#SBATCH --job-name=audit_expert_a5
#SBATCH --account=bt60
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/audit_expert_a5_%j.out
#SBATCH --error=logs/slurm/audit_expert_a5_%j.err

set -euo pipefail

: "${REPODIR:?Submit this script with the absolute component REPODIR exported}"
REPODIR="$(cd -- "${REPODIR}" && pwd)"
export REPODIR
source "${REPODIR}/hpc/slurm/project_env.bash"

export NGSIM_ACCELERATION_LIMIT_MPS2="${EXPERT_ACCELERATION_LIMIT_MPS2:-5.0}"
export PYTHONUNBUFFERED=1
COLLECTION_ID="${COLLECTION_ID:-domain_matched_accel5_v2}"
COLLECTION_ROOT="${VFI_DATA_ROOT}/expert/${COLLECTION_ID}"
AUDIT_OUT="${AUDIT_OUT:-${COLLECTION_ROOT}/collection_contract_audit.json}"

if [ "${NGSIM_ACCELERATION_LIMIT_MPS2}" != "5.0" ]; then
    echo "Corrected comparison collection must use 5.0 m/s^2." >&2
    exit 2
fi
if [ -n "${AUDIT_EXPECTED_SCRIPT_SHA256:-}" ]; then
    actual="$(sha256sum "${REPODIR}/scripts_gail/audit_domain_matched_expert.py" | awk '{print $1}')"
    test "${actual}" = "${AUDIT_EXPECTED_SCRIPT_SHA256}"
fi
if [ -n "${AUDIT_EXPECTED_RUNNER_SHA256:-}" ]; then
    actual="$(sha256sum "${REPODIR}/hpc/slurm/script_data_collection/audit_domain_matched_expert_accel5.bash" | awk '{print $1}')"
    test "${actual}" = "${AUDIT_EXPECTED_RUNNER_SHA256}"
fi

mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}"
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

cd "${REPODIR}"
python -m scripts_gail.audit_domain_matched_expert \
    --collection-root "${COLLECTION_ROOT}" \
    --out "${AUDIT_OUT}"

echo "Passed six-cell expert contract audit: ${AUDIT_OUT}"
