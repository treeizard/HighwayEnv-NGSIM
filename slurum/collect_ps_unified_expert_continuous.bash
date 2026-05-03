#!/bin/bash
#SBATCH --job-name=ps_unified_expert
#SBATCH --account=bt60
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_unified_collect_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_unified_collect_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

COLLECTION_WORKER_THREADS="${COLLECTION_WORKER_THREADS:-2}"
if [ "${COLLECTION_WORKER_THREADS}" -lt 1 ]; then
    COLLECTION_WORKER_THREADS=1
fi
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-4}"
COLLECTION_WORKERS="${COLLECTION_WORKERS:-$((SLURM_CPUS / COLLECTION_WORKER_THREADS))}"
if [ "${COLLECTION_WORKERS}" -lt 1 ]; then
    COLLECTION_WORKERS=1
fi

export OMP_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export MKL_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export OPENBLAS_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export NUMEXPR_NUM_THREADS="${COLLECTION_WORKER_THREADS}"

cd "${REPODIR}"

mkdir -p "${REPODIR}/logs"
mkdir -p "${REPODIR}/expert_data"
mkdir -p "${REPODIR}/expert_data/videos"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

OUT="${OUT:-${REPODIR}/expert_data/ngsim_ps_unified_expert_continuous_${SLURM_JOB_ID}}"
VIDEO_DIR="${VIDEO_DIR:-${REPODIR}/expert_data/videos/${SLURM_JOB_ID}}"
MAX_EPISODES="${MAX_EPISODES:-0}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-200}"
MAX_SAMPLES_PER_VEHICLE="${MAX_SAMPLES_PER_VEHICLE:-200}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Collection workers: ${COLLECTION_WORKERS}"
echo "Collection worker threads: ${COLLECTION_WORKER_THREADS}"

if [ "${SAVE_VIDEO}" = "true" ]; then
    VIDEO_ARG="--save-video"
else
    VIDEO_ARG=""
fi

python "${REPODIR}/scripts_gail/build_ps_traj_expert_discrete.py" \
    --scene us-101 \
    --prebuilt-split train \
    --max-episodes "${MAX_EPISODES}" \
    --collect-all-split-episodes \
    --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}" \
    --max-samples-per-vehicle "${MAX_SAMPLES_PER_VEHICLE}" \
    --num-collection-workers "${COLLECTION_WORKERS}" \
    --collection-worker-threads "${COLLECTION_WORKER_THREADS}" \
    --control-all-vehicles \
    --expert-control-mode continuous \
    --trajectory-state-source simulated \
    --no-allow-idm \
    --out "${OUT}" \
    --video-dir "${VIDEO_DIR}" \
    ${VIDEO_ARG}
