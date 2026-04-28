#!/bin/bash
#SBATCH --job-name=ps_gail_diag
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_diag_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_diag_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export MPLCONFIGDIR="${REPODIR}/logs/matplotlib_${SLURM_JOB_ID}"

cd "${REPODIR}"
mkdir -p "${REPODIR}/logs" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_traj_expert_discrete_54902119}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-8192}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-1}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
PERCENTAGE_CONTROLLED_VEHICLES="${PERCENTAGE_CONTROLLED_VEHICLES:-0.20}"
TRAJECTORY_FRAME="${TRAJECTORY_FRAME:-relative}"
SCENE_MAX_VEHICLES="${SCENE_MAX_VEHICLES:-64}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-1}"
ENABLE_SCENE_DIAG="${ENABLE_SCENE_DIAG:-false}"
ENABLE_SEQUENCE_DIAG="${ENABLE_SEQUENCE_DIAG:-true}"

SCENE_ARG=""
if [ "${ENABLE_SCENE_DIAG}" = "true" ]; then
    SCENE_ARG="--enable-scene-discriminator"
fi

SEQUENCE_ARG=""
if [ "${ENABLE_SEQUENCE_DIAG}" = "true" ]; then
    SEQUENCE_ARG="--enable-sequence-discriminator"
fi

python "${REPODIR}/scripts_gail/diagnose_gail_convergence.py" \
    --expert-data "${EXPERT_DATA}" \
    --scene us-101 \
    --episode-root "${REPODIR}/highway_env/data/processed_20s" \
    --prebuilt-split train \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    --rollout-episodes "${ROLLOUT_EPISODES}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --percentage-controlled-vehicles "${PERCENTAGE_CONTROLLED_VEHICLES}" \
    --trajectory-frame "${TRAJECTORY_FRAME}" \
    --scene-max-vehicles "${SCENE_MAX_VEHICLES}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --sequence-stride "${SEQUENCE_STRIDE}" \
    --device cuda \
    ${SCENE_ARG} \
    ${SEQUENCE_ARG}
