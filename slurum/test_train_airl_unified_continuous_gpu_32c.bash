#!/bin/bash
#SBATCH --job-name=ps_airl_gpu_32c
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_airl_gpu_32c_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_airl_gpu_32c_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

# PS-AIRL uses the AIRL reward model with the same paper-style controlled-vehicle
# schedule as PS-GAIL. Rollouts run in CPU workers while PyTorch updates use GPU.
ROLLOUT_WORKER_THREADS="${ROLLOUT_WORKER_THREADS:-2}"
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-32}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-$((SLURM_CPUS / ROLLOUT_WORKER_THREADS))}"
if [ "${ROLLOUT_WORKERS}" -lt 1 ]; then
    ROLLOUT_WORKERS=1
fi
REQUESTED_ROLLOUT_CPUS=$((ROLLOUT_WORKERS * ROLLOUT_WORKER_THREADS))
if [ "${REQUESTED_ROLLOUT_CPUS}" -gt "${SLURM_CPUS}" ]; then
    echo "Requested rollout CPU use (${REQUESTED_ROLLOUT_CPUS}) exceeds SLURM_CPUS_PER_TASK (${SLURM_CPUS})." >&2
    exit 1
fi

export OMP_NUM_THREADS="${ROLLOUT_WORKER_THREADS}"
export MKL_NUM_THREADS="${ROLLOUT_WORKER_THREADS}"
export OPENBLAS_NUM_THREADS="${ROLLOUT_WORKER_THREADS}"
export NUMEXPR_NUM_THREADS="${ROLLOUT_WORKER_THREADS}"
export MPLCONFIGDIR="${REPODIR}/logs/matplotlib_${SLURM_JOB_ID}"

cd "${REPODIR}"

mkdir -p "${REPODIR}/logs"
mkdir -p "${REPODIR}/logs/airl"
mkdir -p "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_unified_expert_continuous_55145982}"
AIRL_TRAIN_SCRIPT="${AIRL_TRAIN_SCRIPT:-${REPODIR}/scripts_gail/train_simple_airl.py}"
RUN_NAME="${RUN_NAME:-ps_airl_unified_continuous_gpu_32c_${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"

# Training defaults: full-episode AIRL rounds. With --rollout-full-episodes,
# active rollout workers are capped by ROLLOUT_MIN_EPISODES.
TOTAL_ROUNDS="${TOTAL_ROUNDS:-800}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
ROLLOUT_MIN_EPISODES="${ROLLOUT_MIN_EPISODES:-${ROLLOUT_WORKERS}}"
ROLLOUT_MAX_EPISODE_STEPS="${ROLLOUT_MAX_EPISODE_STEPS:-0}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-100000}"
TRAJECTORY_FRAME="${TRAJECTORY_FRAME:-relative}"
INITIAL_CONTROLLED_VEHICLE_FRACTION="${INITIAL_CONTROLLED_VEHICLE_FRACTION:-0.05}"
FINAL_CONTROLLED_VEHICLE_FRACTION="${FINAL_CONTROLLED_VEHICLE_FRACTION:-1.0}"
CONTROLLED_VEHICLE_CURRICULUM_ROUNDS="${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
CONTROLLED_VEHICLE_CURRICULUM="${CONTROLLED_VEHICLE_CURRICULUM:-true}"
PAPER_STYLE_TRAINING="${PAPER_STYLE_TRAINING:-true}"
PAPER_PHASE1_ROUNDS="${PAPER_PHASE1_ROUNDS:-500}"
PAPER_PHASE2_ROUNDS="${PAPER_PHASE2_ROUNDS:-300}"
PAPER_PHASE1_GAMMA="${PAPER_PHASE1_GAMMA:-0.95}"
PAPER_PHASE2_GAMMA="${PAPER_PHASE2_GAMMA:-0.99}"
PAPER_PHASE1_AGENT_STEPS="${PAPER_PHASE1_AGENT_STEPS:-10000}"
PAPER_PHASE2_AGENT_STEPS="${PAPER_PHASE2_AGENT_STEPS:-40000}"
PAPER_INITIAL_AGENT_COUNT="${PAPER_INITIAL_AGENT_COUNT:-10}"
PAPER_AGENT_INCREMENT="${PAPER_AGENT_INCREMENT:-10}"
PAPER_AGENT_INCREMENT_INTERVAL="${PAPER_AGENT_INCREMENT_INTERVAL:-100}"
PAPER_PHASE2_INITIAL_AGENT_COUNT="${PAPER_PHASE2_INITIAL_AGENT_COUNT:-50}"
PAPER_PHASE2_AGENT_COUNT="${PAPER_PHASE2_AGENT_COUNT:-100}"
PAPER_PHASE2_AGENT_RAMP_ROUNDS="${PAPER_PHASE2_AGENT_RAMP_ROUNDS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-4096}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
DISC_LEARNING_RATE="${DISC_LEARNING_RATE:-5e-5}"
DISC_UPDATES_PER_ROUND="${DISC_UPDATES_PER_ROUND:-2}"
DISC_EXPERT_LABEL="${DISC_EXPERT_LABEL:-0.9}"
DISC_GENERATOR_LABEL="${DISC_GENERATOR_LABEL:-0.1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
SAVE_CHECKPOINT_VIDEO="${SAVE_CHECKPOINT_VIDEO:-true}"
CHECKPOINT_VIDEO_STEPS="${CHECKPOINT_VIDEO_STEPS:-200}"
if [ "${PAPER_STYLE_TRAINING}" = "true" ]; then
    PAPER_STYLE_ARG="--paper-style-training"
else
    PAPER_STYLE_ARG="--no-paper-style-training"
fi
if [ "${CONTROLLED_VEHICLE_CURRICULUM}" = "true" ] && [ "${PAPER_STYLE_TRAINING}" != "true" ]; then
    CONTROLLED_VEHICLE_CURRICULUM_ARG="--controlled-vehicle-curriculum"
else
    CONTROLLED_VEHICLE_CURRICULUM_ARG="--no-controlled-vehicle-curriculum"
fi
if [ "${SAVE_CHECKPOINT_VIDEO}" = "true" ]; then
    CHECKPOINT_VIDEO_ARG="--save-checkpoint-video"
else
    CHECKPOINT_VIDEO_ARG="--no-save-checkpoint-video"
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Expert data: ${EXPERT_DATA}"
echo "AIRL trainer: ${AIRL_TRAIN_SCRIPT}"
echo "Total rounds: ${TOTAL_ROUNDS}"
echo "Rollout steps: ${ROLLOUT_STEPS}"
echo "Rollout min episodes: ${ROLLOUT_MIN_EPISODES}"
echo "CPUs per task: ${SLURM_CPUS}"
echo "Rollout workers: ${ROLLOUT_WORKERS}"
echo "Rollout worker threads: ${ROLLOUT_WORKER_THREADS}"
echo "Requested rollout CPUs: ${REQUESTED_ROLLOUT_CPUS}"
echo "Paper-style training: ${PAPER_STYLE_TRAINING}"
echo "Paper schedule: phase1=${PAPER_PHASE1_ROUNDS} rounds, agents ${PAPER_INITIAL_AGENT_COUNT}+${PAPER_AGENT_INCREMENT}/${PAPER_AGENT_INCREMENT_INTERVAL} rounds; phase2=${PAPER_PHASE2_ROUNDS} rounds, ramp ${PAPER_PHASE2_INITIAL_AGENT_COUNT}->${PAPER_PHASE2_AGENT_COUNT}/${PAPER_PHASE2_AGENT_RAMP_ROUNDS} rounds"
echo "Controlled-vehicle curriculum: ${CONTROLLED_VEHICLE_CURRICULUM_ARG}"
echo "AIRL reward lr: ${DISC_LEARNING_RATE}"
echo "AIRL reward updates per round: ${DISC_UPDATES_PER_ROUND}"
echo "AIRL reward labels: expert=${DISC_EXPERT_LABEL}, generator=${DISC_GENERATOR_LABEL}"
echo "Checkpoint every: ${CHECKPOINT_EVERY}"
echo "Save checkpoint video: ${SAVE_CHECKPOINT_VIDEO}"
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

if [ ! -f "${AIRL_TRAIN_SCRIPT}" ]; then
    echo "AIRL trainer not found: ${AIRL_TRAIN_SCRIPT}" >&2
    echo "Add scripts_gail/train_simple_airl.py or submit with AIRL_TRAIN_SCRIPT=/path/to/trainer.py." >&2
    exit 2
fi

python "${AIRL_TRAIN_SCRIPT}" \
    --expert-data "${EXPERT_DATA}" \
    --scene us-101 \
    --action-mode continuous \
    --episode-root "${REPODIR}/highway_env/data/processed_20s" \
    --prebuilt-split train \
    --no-control-all-vehicles \
    "${CONTROLLED_VEHICLE_CURRICULUM_ARG}" \
    --initial-controlled-vehicle-fraction "${INITIAL_CONTROLLED_VEHICLE_FRACTION}" \
    --final-controlled-vehicle-fraction "${FINAL_CONTROLLED_VEHICLE_FRACTION}" \
    --controlled-vehicle-curriculum-rounds "${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS}" \
    "${PAPER_STYLE_ARG}" \
    --paper-phase1-rounds "${PAPER_PHASE1_ROUNDS}" \
    --paper-phase2-rounds "${PAPER_PHASE2_ROUNDS}" \
    --paper-phase1-gamma "${PAPER_PHASE1_GAMMA}" \
    --paper-phase2-gamma "${PAPER_PHASE2_GAMMA}" \
    --paper-phase1-agent-steps "${PAPER_PHASE1_AGENT_STEPS}" \
    --paper-phase2-agent-steps "${PAPER_PHASE2_AGENT_STEPS}" \
    --paper-initial-agent-count "${PAPER_INITIAL_AGENT_COUNT}" \
    --paper-agent-increment "${PAPER_AGENT_INCREMENT}" \
    --paper-agent-increment-interval "${PAPER_AGENT_INCREMENT_INTERVAL}" \
    --paper-phase2-initial-agent-count "${PAPER_PHASE2_INITIAL_AGENT_COUNT}" \
    --paper-phase2-agent-count "${PAPER_PHASE2_AGENT_COUNT}" \
    --paper-phase2-agent-ramp-rounds "${PAPER_PHASE2_AGENT_RAMP_ROUNDS}" \
    --enable-collision \
    --allow-idm \
    --device cuda \
    --total-rounds "${TOTAL_ROUNDS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --rollout-min-episodes "${ROLLOUT_MIN_EPISODES}" \
    --rollout-full-episodes \
    --rollout-max-episode-steps "${ROLLOUT_MAX_EPISODE_STEPS}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --trajectory-frame "${TRAJECTORY_FRAME}" \
    --num-rollout-workers "${ROLLOUT_WORKERS}" \
    --rollout-worker-threads "${ROLLOUT_WORKER_THREADS}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --reward-batch-size "${REWARD_BATCH_SIZE}" \
    --discriminator-loss airl_bce \
    --disc-learning-rate "${DISC_LEARNING_RATE}" \
    --disc-updates-per-round "${DISC_UPDATES_PER_ROUND}" \
    --disc-expert-label "${DISC_EXPERT_LABEL}" \
    --disc-generator-label "${DISC_GENERATOR_LABEL}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${CHECKPOINT_VIDEO_ARG}" \
    --checkpoint-video-steps "${CHECKPOINT_VIDEO_STEPS}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project highwayenv-ps-gail \
    --wandb-tags ps-airl,airl,continuous,unified-expert,paper-style,gpu,32c,disc-updates-${DISC_UPDATES_PER_ROUND} \
    --run-name "${RUN_NAME}"
