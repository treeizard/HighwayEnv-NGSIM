#!/bin/bash
#SBATCH --job-name=ps_gail_dual_train_check
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_dual_train_check_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_dual_train_check_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

ROLLOUT_WORKER_THREADS="${ROLLOUT_WORKER_THREADS:-2}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-$((SLURM_CPUS_PER_TASK / ROLLOUT_WORKER_THREADS))}"
if [ "${ROLLOUT_WORKERS}" -lt 1 ]; then
    ROLLOUT_WORKERS=1
fi

export OMP_NUM_THREADS="${ROLLOUT_WORKER_THREADS}"
export MKL_NUM_THREADS="${ROLLOUT_WORKER_THREADS}"
export OPENBLAS_NUM_THREADS="${ROLLOUT_WORKER_THREADS}"
export NUMEXPR_NUM_THREADS="${ROLLOUT_WORKER_THREADS}"
export MPLCONFIGDIR="${REPODIR}/logs/matplotlib_${SLURM_JOB_ID}"

cd "${REPODIR}"
mkdir -p "${REPODIR}/logs" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_traj_expert_dual_disc_check}"
RUN_NAME="${RUN_NAME:-ps_gail_dual_disc_check_${SLURM_JOB_ID}}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-1}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
ROLLOUT_MIN_EPISODES="${ROLLOUT_MIN_EPISODES:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-4096}"
SCENE_MAX_VEHICLES="${SCENE_MAX_VEHICLES:-64}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-1}"

python "${REPODIR}/scripts_gail/train_simple_ps_gail.py" \
    --expert-data "${EXPERT_DATA}" \
    --scene us-101 \
    --episode-root "${REPODIR}/highway_env/data/processed_20s" \
    --prebuilt-split train \
    --no-control-all-vehicles \
    --percentage-controlled-vehicles 0.20 \
    --enable-collision \
    --terminate-when-all-controlled-crashed \
    --allow-idm \
    --rollout-full-episodes \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --rollout-min-episodes "${ROLLOUT_MIN_EPISODES}" \
    --rollout-max-episode-steps 0 \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --num-rollout-workers "${ROLLOUT_WORKERS}" \
    --rollout-worker-threads "${ROLLOUT_WORKER_THREADS}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    --enable-scene-discriminator \
    --enable-sequence-discriminator \
    --scene-max-vehicles "${SCENE_MAX_VEHICLES}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --sequence-stride "${SEQUENCE_STRIDE}" \
    --scene-reward-coef 1.0 \
    --sequence-reward-coef 1.0 \
    --hidden-size 128 \
    --batch-size 512 \
    --disc-batch-size 512 \
    --disc-updates-per-round 1 \
    --ppo-epochs 1 \
    --device cuda \
    --no-save-checkpoint-video \
    --wandb-mode disabled \
    --run-name "${RUN_NAME}"
