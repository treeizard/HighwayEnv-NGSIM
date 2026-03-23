#!/bin/bash
#SBATCH --job-name=ngsim_expert_collect
#SBATCH --account=bt60
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ngsim_collect_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ngsim_collect_%j.err

set -euo pipefail

export REPODIR=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

cd "${REPODIR}"

mkdir -p "${REPODIR}/logs"
mkdir -p "${REPODIR}/expert_data"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

python "${REPODIR}/scripts_ngsim/build_expert_data_discrete.py" \
    --episodes 10 \
    --out "${REPODIR}/expert_data/ngsim_expert_discrete_${SLURM_JOB_ID}.npz"