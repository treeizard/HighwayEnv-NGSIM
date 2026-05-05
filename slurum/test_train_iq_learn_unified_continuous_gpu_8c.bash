#!/bin/bash
#SBATCH --job-name=iqlearn_unified_test
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/iqlearn_unified_test_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/iqlearn_unified_test_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

# IQ-Learn can train mostly from expert transitions, so it needs far fewer CPU
# rollout workers than PS-GAIL/AIRL for a first test. Keep one GPU for batched
# critic/policy updates.
WORKER_THREADS="${WORKER_THREADS:-2}"
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-8}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-2}"
if [ "${ROLLOUT_WORKERS}" -gt "$((SLURM_CPUS / WORKER_THREADS))" ]; then
    ROLLOUT_WORKERS="$((SLURM_CPUS / WORKER_THREADS))"
fi
if [ "${ROLLOUT_WORKERS}" -lt 1 ]; then
    ROLLOUT_WORKERS=1
fi

export OMP_NUM_THREADS="${WORKER_THREADS}"
export MKL_NUM_THREADS="${WORKER_THREADS}"
export OPENBLAS_NUM_THREADS="${WORKER_THREADS}"
export NUMEXPR_NUM_THREADS="${WORKER_THREADS}"
export MPLCONFIGDIR="${REPODIR}/logs/matplotlib_${SLURM_JOB_ID}"

cd "${REPODIR}"

mkdir -p "${REPODIR}/logs"
mkdir -p "${REPODIR}/logs/iq_learn"
mkdir -p "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_unified_expert_continuous_55145982}"
IQ_TRAIN_SCRIPT="${IQ_TRAIN_SCRIPT:-${REPODIR}/scripts_gail/train_simple_iq_learn.py}"
RUN_NAME="${RUN_NAME:-iqlearn_unified_continuous_test_${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"

TOTAL_UPDATES="${TOTAL_UPDATES:-20000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-100000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
REPLAY_SIZE="${REPLAY_SIZE:-200000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
SAVE_CHECKPOINT_VIDEO="${SAVE_CHECKPOINT_VIDEO:-true}"
CHECKPOINT_VIDEO_STEPS="${CHECKPOINT_VIDEO_STEPS:-120}"
if [ "${SAVE_CHECKPOINT_VIDEO}" = "true" ]; then
    CHECKPOINT_VIDEO_ARG="--save-checkpoint-video"
else
    CHECKPOINT_VIDEO_ARG="--no-save-checkpoint-video"
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Expert data: ${EXPERT_DATA}"
echo "IQ-Learn trainer: ${IQ_TRAIN_SCRIPT}"
echo "CPUs per task: ${SLURM_CPUS}"
echo "Eval rollout workers: ${ROLLOUT_WORKERS}"
echo "Worker threads: ${WORKER_THREADS}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python - <<'PY'
import os
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
wandb_mode = os.environ.get("WANDB_MODE", "online").lower()
print("wandb mode:", wandb_mode)
if wandb_mode != "disabled":
    import numpy as np
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    import wandb
    print("wandb:", wandb.__version__)
save_video = os.environ.get("SAVE_CHECKPOINT_VIDEO", "true").lower()
print("save checkpoint video:", save_video)
if save_video == "true":
    import imageio
    import imageio.v2 as imageio_v2
    print("imageio:", imageio.__version__)
PY

if [ ! -f "${IQ_TRAIN_SCRIPT}" ]; then
    echo "IQ-Learn trainer not found: ${IQ_TRAIN_SCRIPT}" >&2
    echo "Add scripts_gail/train_simple_iq_learn.py or submit with IQ_TRAIN_SCRIPT=/path/to/trainer.py." >&2
    exit 2
fi

python "${IQ_TRAIN_SCRIPT}" \
    --expert-data "${EXPERT_DATA}" \
    --scene us-101 \
    --action-mode continuous \
    --episode-root "${REPODIR}/highway_env/data/processed_20s" \
    --prebuilt-split train \
    --enable-collision \
    --allow-idm \
    --device cuda \
    --total-updates "${TOTAL_UPDATES}" \
    --eval-every "${EVAL_EVERY}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --num-rollout-workers "${ROLLOUT_WORKERS}" \
    --rollout-worker-threads "${WORKER_THREADS}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --replay-size "${REPLAY_SIZE}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${CHECKPOINT_VIDEO_ARG}" \
    --checkpoint-video-steps "${CHECKPOINT_VIDEO_STEPS}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project highwayenv-iq-learn \
    --wandb-tags iq-learn,continuous,unified-expert,test,gpu,8c \
    --run-name "${RUN_NAME}"
