#!/bin/bash
#SBATCH --job-name=ps_traj_expert_collect
#SBATCH --account=bt60
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_traj_collect_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_traj_collect_%j.err

set -euo pipefail

export REPODIR=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

cd "${REPODIR}"

mkdir -p "${REPODIR}/logs"
mkdir -p "${REPODIR}/expert_data"
mkdir -p "${REPODIR}/expert_data/videos"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

python "${REPODIR}/scripts_gail/build_ps_traj_expert_discrete.py" \
    --scene us-101 \
    --prebuilt-split train \
    --max-episodes 10 \
    --max-steps-per-episode 200 \
    --control-all-vehicles \
    --expert-control-mode continuous \
    --trajectory-state-source simulated \
    --no-allow-idm \
    --out "${REPODIR}/expert_data/ngsim_ps_traj_expert_discrete_${SLURM_JOB_ID}" \
    --video-dir "${REPODIR}/expert_data/videos/${SLURM_JOB_ID}" \
    --save-video
