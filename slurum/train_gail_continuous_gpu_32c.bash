#!/bin/bash
#SBATCH --job-name=ps_gail_cont_gpu_32c
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_cont_gpu_32c_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_cont_gpu_32c_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

# Rollouts run in multiple CPU worker processes while PyTorch updates run on
# the allocated GPU. Give each rollout worker a small native thread pool so the
# total CPU use stays near SLURM_CPUS_PER_TASK.
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-32}"
ROLLOUT_WORKER_THREADS="${ROLLOUT_WORKER_THREADS:-2}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
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
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPODIR}"

mkdir -p "${REPODIR}/logs"
mkdir -p "${REPODIR}/logs/simple_ps_gail"
mkdir -p "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CPUs per task: ${SLURM_CPUS}"
echo "Rollout workers: ${ROLLOUT_WORKERS}"
echo "Rollout worker threads: ${ROLLOUT_WORKER_THREADS}"
echo "Requested rollout CPUs: ${REQUESTED_ROLLOUT_CPUS}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

RUN_NAME="${RUN_NAME:-ps_gail_continuous_gpu_32c_${SLURM_JOB_ID}}"
ACTION_MODE="${ACTION_MODE:-continuous}"
EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_unified_expert_continuous_55145982}"
WANDB_MODE="${WANDB_MODE:-online}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-1100}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
ROLLOUT_MIN_EPISODES="${ROLLOUT_MIN_EPISODES:-2}"
ROLLOUT_TARGET_AWARE_EPISODES="${ROLLOUT_TARGET_AWARE_EPISODES:-true}"
ROLLOUT_TARGET_MIN_EPISODES="${ROLLOUT_TARGET_MIN_EPISODES:-${ROLLOUT_WORKERS}}"
ROLLOUT_TARGET_EPISODE_SAFETY_FACTOR="${ROLLOUT_TARGET_EPISODE_SAFETY_FACTOR:-1.25}"
ROLLOUT_TRAINING_SUBSAMPLE="${ROLLOUT_TRAINING_SUBSAMPLE:-true}"
ROLLOUT_TRAINING_AGENT_STEPS="${ROLLOUT_TRAINING_AGENT_STEPS:-0}"
ROLLOUT_MAX_EPISODE_STEPS="${ROLLOUT_MAX_EPISODE_STEPS:-0}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
MAX_EPISODE_STEPS_SCHEDULE="${MAX_EPISODE_STEPS_SCHEDULE:-}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-100000}"
TRAJECTORY_FRAME="${TRAJECTORY_FRAME:-relative}"
INITIAL_CONTROLLED_VEHICLE_FRACTION="${INITIAL_CONTROLLED_VEHICLE_FRACTION:-0.05}"
FINAL_CONTROLLED_VEHICLE_FRACTION="${FINAL_CONTROLLED_VEHICLE_FRACTION:-1.0}"
INITIAL_CONTROLLED_VEHICLES="${INITIAL_CONTROLLED_VEHICLES:-${INITIAL_CONTROLLED_VEHICLE_FRACTION}}"
FINAL_CONTROLLED_VEHICLES="${FINAL_CONTROLLED_VEHICLES:-${FINAL_CONTROLLED_VEHICLE_FRACTION}}"
CONTROLLED_VEHICLE_CURRICULUM_ROUNDS="${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
CONTROLLED_VEHICLE_INCREMENT_ROUNDS="${CONTROLLED_VEHICLE_INCREMENT_ROUNDS:-0}"
CONTROLLED_VEHICLE_SCHEDULE="${CONTROLLED_VEHICLE_SCHEDULE:-0:100:10:20;100:200:20:30;200:300:30:40;300:400:40:50;400:500:50:60;500:600:60:70;600:700:70:80;700:800:80:90;800:900:90:100;900:1100:100:100}"
CONTROLLED_VEHICLE_CURRICULUM="${CONTROLLED_VEHICLE_CURRICULUM:-true}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-0}"
WARMUP_LEARNING_RATE="${WARMUP_LEARNING_RATE:-1e-4}"
WARMUP_DISC_LEARNING_RATE="${WARMUP_DISC_LEARNING_RATE:-2e-5}"
WARMUP_ENTROPY_COEF="${WARMUP_ENTROPY_COEF:-0.03}"
WARMUP_CLIP_RANGE="${WARMUP_CLIP_RANGE:-0.10}"
WARMUP_DISC_UPDATES_PER_ROUND="${WARMUP_DISC_UPDATES_PER_ROUND:-0}"
WARMUP_GAIL_REWARD_CLIP="${WARMUP_GAIL_REWARD_CLIP:-2.0}"
WARMUP_FINAL_REWARD_CLIP="${WARMUP_FINAL_REWARD_CLIP:-5.0}"
VEHICLE_INCREASE_WARMUP_ROUNDS="${VEHICLE_INCREASE_WARMUP_ROUNDS:-0}"
ROLLOUT_TARGET_AGENT_STEPS="${ROLLOUT_TARGET_AGENT_STEPS:-0}"
INITIAL_ROLLOUT_TARGET_AGENT_STEPS="${INITIAL_ROLLOUT_TARGET_AGENT_STEPS:-10000}"
FINAL_ROLLOUT_TARGET_AGENT_STEPS="${FINAL_ROLLOUT_TARGET_AGENT_STEPS:-40000}"
ROLLOUT_TARGET_AGENT_STEPS_CURRICULUM_ROUNDS="${ROLLOUT_TARGET_AGENT_STEPS_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE="${ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE:-0:900:10000:10000;900:1100:40000:40000}"
INITIAL_GAMMA="${INITIAL_GAMMA:-0.95}"
FINAL_GAMMA="${FINAL_GAMMA:-0.99}"
GAMMA_CURRICULUM_ROUNDS="${GAMMA_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
GAMMA_SCHEDULE="${GAMMA_SCHEDULE:-0:900:0.95:0.95;900:1100:0.99:0.99}"
DISC_EXPERT_LABEL="${DISC_EXPERT_LABEL:-0.8}"
DISC_GENERATOR_LABEL="${DISC_GENERATOR_LABEL:-0.2}"
DISC_LEARNING_RATE="${DISC_LEARNING_RATE:-5e-5}"
DISC_UPDATES_PER_ROUND="${DISC_UPDATES_PER_ROUND:-2}"
BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-0}"
BC_PRETRAIN_LEARNING_RATE="${BC_PRETRAIN_LEARNING_RATE:-3e-4}"
BC_PRETRAIN_BATCH_SIZE="${BC_PRETRAIN_BATCH_SIZE:-4096}"
BC_PRETRAIN_MICRO_BATCH_SIZE="${BC_PRETRAIN_MICRO_BATCH_SIZE:-128}"
BC_PRETRAIN_VALIDATION_FRACTION="${BC_PRETRAIN_VALIDATION_FRACTION:-0.1}"
BC_PRETRAIN_EVAL_EPISODES="${BC_PRETRAIN_EVAL_EPISODES:-4}"
BC_PRETRAIN_MIN_MEAN_EPISODE_LENGTH="${BC_PRETRAIN_MIN_MEAN_EPISODE_LENGTH:-0}"
BC_PRETRAIN_ABORT_ON_FAILED_EVAL="${BC_PRETRAIN_ABORT_ON_FAILED_EVAL:-false}"
POLICY_BC_REGULARIZATION_COEF="${POLICY_BC_REGULARIZATION_COEF:-0.0}"
POLICY_BC_REGULARIZATION_FINAL_COEF="${POLICY_BC_REGULARIZATION_FINAL_COEF:-0.0}"
POLICY_BC_REGULARIZATION_DECAY_ROUNDS="${POLICY_BC_REGULARIZATION_DECAY_ROUNDS:-${TOTAL_ROUNDS}}"
COLLISION_PENALTY="${COLLISION_PENALTY:-2.0}"
OFFROAD_PENALTY="${OFFROAD_PENALTY:-2.0}"
GAIL_REWARD_CLIP="${GAIL_REWARD_CLIP:-5.0}"
FINAL_REWARD_CLIP="${FINAL_REWARD_CLIP:-10.0}"
SAVE_CHECKPOINT_VIDEO="${SAVE_CHECKPOINT_VIDEO:-true}"
CHECKPOINT_VIDEO_EVERY="${CHECKPOINT_VIDEO_EVERY:-50}"
CHECKPOINT_VIDEO_STEPS="${CHECKPOINT_VIDEO_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
DISC_BATCH_SIZE="${DISC_BATCH_SIZE:-4096}"
POLICY_MODEL="${POLICY_MODEL:-transformer}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
DISCRIMINATOR_HIDDEN_SIZES="${DISCRIMINATOR_HIDDEN_SIZES:-256,256,128}"
DISCRIMINATOR_DROPOUT="${DISCRIMINATOR_DROPOUT:-0.1}"
DISCRIMINATOR_SPECTRAL_NORM="${DISCRIMINATOR_SPECTRAL_NORM:-true}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.1}"
CENTRALIZED_CRITIC="${CENTRALIZED_CRITIC:-true}"
CENTRAL_CRITIC_MAX_VEHICLES="${CENTRAL_CRITIC_MAX_VEHICLES:-128}"
CENTRAL_CRITIC_INCLUDE_LOCAL_OBS="${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS:-false}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
TERMINATE_WHEN_ALL_CONTROLLED_CRASHED="${TERMINATE_WHEN_ALL_CONTROLLED_CRASHED:-true}"
export EXPERT_DATA
export INITIAL_CONTROLLED_VEHICLES
if [ "${TERMINATE_WHEN_ALL_CONTROLLED_CRASHED}" = "true" ]; then
    TERMINATION_ARG="--terminate-when-all-controlled-crashed"
else
    TERMINATION_ARG="--no-terminate-when-all-controlled-crashed"
fi
if [ "${SAVE_CHECKPOINT_VIDEO}" = "true" ]; then
    CHECKPOINT_VIDEO_ARG="--save-checkpoint-video"
else
    CHECKPOINT_VIDEO_ARG="--no-save-checkpoint-video"
fi
if [ "${BC_PRETRAIN_ABORT_ON_FAILED_EVAL}" = "true" ]; then
    BC_PRETRAIN_ABORT_ARG="--bc-pretrain-abort-on-failed-eval"
else
    BC_PRETRAIN_ABORT_ARG="--no-bc-pretrain-abort-on-failed-eval"
fi
if [ "${CONTROLLED_VEHICLE_CURRICULUM}" = "true" ]; then
    CONTROLLED_VEHICLE_CURRICULUM_ARG="--controlled-vehicle-curriculum"
else
    CONTROLLED_VEHICLE_CURRICULUM_ARG="--no-controlled-vehicle-curriculum"
fi
if [ "${ROLLOUT_TARGET_AWARE_EPISODES}" = "true" ]; then
    ROLLOUT_TARGET_AWARE_ARG="--rollout-target-aware-episodes"
else
    ROLLOUT_TARGET_AWARE_ARG="--no-rollout-target-aware-episodes"
fi
if [ "${ROLLOUT_TRAINING_SUBSAMPLE}" = "true" ]; then
    ROLLOUT_TRAINING_SUBSAMPLE_ARG="--rollout-training-subsample"
else
    ROLLOUT_TRAINING_SUBSAMPLE_ARG="--no-rollout-training-subsample"
fi
if [ "${DISCRIMINATOR_SPECTRAL_NORM}" = "true" ]; then
    DISCRIMINATOR_SPECTRAL_NORM_ARG="--discriminator-spectral-norm"
else
    DISCRIMINATOR_SPECTRAL_NORM_ARG="--no-discriminator-spectral-norm"
fi
if [ "${CENTRALIZED_CRITIC}" = "true" ]; then
    CENTRALIZED_CRITIC_ARG="--centralized-critic"
else
    CENTRALIZED_CRITIC_ARG="--no-centralized-critic"
fi
if [ "${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS}" = "true" ]; then
    CENTRAL_CRITIC_INCLUDE_LOCAL_OBS_ARG="--central-critic-include-local-obs"
else
    CENTRAL_CRITIC_INCLUDE_LOCAL_OBS_ARG="--no-central-critic-include-local-obs"
fi

echo "Expert data: ${EXPERT_DATA}"
echo "Total rounds: ${TOTAL_ROUNDS}"
echo "Rollout target-aware episodes: ${ROLLOUT_TARGET_AWARE_EPISODES} min=${ROLLOUT_TARGET_MIN_EPISODES} safety=${ROLLOUT_TARGET_EPISODE_SAFETY_FACTOR}"
echo "Rollout training subsample: ${ROLLOUT_TRAINING_SUBSAMPLE} cap=${ROLLOUT_TRAINING_AGENT_STEPS:-0} (0 follows rollout target)"
echo "BC pretrain epochs: ${BC_PRETRAIN_EPOCHS}"
echo "Controlled-vehicle curriculum: ${CONTROLLED_VEHICLE_CURRICULUM_ARG} ${INITIAL_CONTROLLED_VEHICLES}->${FINAL_CONTROLLED_VEHICLES}/${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS} rounds increment_rounds=${CONTROLLED_VEHICLE_INCREMENT_ROUNDS}"
echo "Controlled-vehicle piecewise schedule: ${CONTROLLED_VEHICLE_SCHEDULE}"
echo "Warmup: rounds=${WARMUP_ROUNDS} policy_lr=${WARMUP_LEARNING_RATE}->default disc_lr=${WARMUP_DISC_LEARNING_RATE}->default entropy=${WARMUP_ENTROPY_COEF}->default clip=${WARMUP_CLIP_RANGE}->default"
echo "Vehicle-increase warmup rounds: ${VEHICLE_INCREASE_WARMUP_ROUNDS}"
echo "Policy BC regularization: ${POLICY_BC_REGULARIZATION_COEF}->${POLICY_BC_REGULARIZATION_FINAL_COEF}/${POLICY_BC_REGULARIZATION_DECAY_ROUNDS} rounds"
echo "Rollout target agent steps: base=${ROLLOUT_TARGET_AGENT_STEPS} curriculum=${INITIAL_ROLLOUT_TARGET_AGENT_STEPS}->${FINAL_ROLLOUT_TARGET_AGENT_STEPS}/${ROLLOUT_TARGET_AGENT_STEPS_CURRICULUM_ROUNDS} rounds"
echo "Rollout target agent steps schedule: ${ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE}"
echo "Gamma curriculum: ${INITIAL_GAMMA}->${FINAL_GAMMA}/${GAMMA_CURRICULUM_ROUNDS} rounds"
echo "Gamma schedule: ${GAMMA_SCHEDULE}"
echo "Max episode steps schedule: ${MAX_EPISODE_STEPS_SCHEDULE}"
echo "Policy model: ${POLICY_MODEL}"
echo "Centralized critic: ${CENTRALIZED_CRITIC} max_vehicles=${CENTRAL_CRITIC_MAX_VEHICLES} include_local_obs=${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS}"
echo "Discriminator architecture: ${DISCRIMINATOR_HIDDEN_SIZES} dropout=${DISCRIMINATOR_DROPOUT}"
echo "Discriminator spectral norm: ${DISCRIMINATOR_SPECTRAL_NORM}"
echo "Discriminator learning rate: ${DISC_LEARNING_RATE}"
echo "Discriminator updates per round: ${DISC_UPDATES_PER_ROUND}"
echo "Checkpoint every: ${CHECKPOINT_EVERY}"
echo "Save checkpoint video: ${SAVE_CHECKPOINT_VIDEO}"
echo "Checkpoint video every: ${CHECKPOINT_VIDEO_EVERY}"
echo "BC pretrain batch size: ${BC_PRETRAIN_BATCH_SIZE}"
echo "BC pretrain micro-batch size: ${BC_PRETRAIN_MICRO_BATCH_SIZE}"
echo "PyTorch CUDA alloc conf: ${PYTORCH_CUDA_ALLOC_CONF}"

python - <<'PY'
import os
import numpy as np

root = os.environ["EXPERT_DATA"]
if not os.path.exists(root):
    raise SystemExit(f"EXPERT_DATA does not exist: {root}")
files = sorted(
    os.path.join(root, name) for name in os.listdir(root) if name.endswith(".npz")
) if os.path.isdir(root) else [root]
if not files:
    raise SystemExit(f"No expert .npz files found under {root}")
with np.load(files[0], allow_pickle=True) as data:
    if "actions_continuous_env" not in data.files:
        raise SystemExit(
            f"{files[0]} is missing actions_continuous_env. "
            "Continuous action-conditioned PS-GAIL requires unified continuous expert data."
        )
    actions = np.asarray(data["actions_continuous_env"])
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise SystemExit(
            f"{files[0]} actions_continuous_env must have shape [N, 2], got {actions.shape}."
        )
    print("continuous expert preflight ok:", files[0], "samples", len(actions))
PY

python - <<'PY'
import os

import gymnasium as gym

from highway_env.imitation.expert_dataset import ENV_ID, register_ngsim_env

episode_name = "t1118846979700"
requested = float(os.environ["INITIAL_CONTROLLED_VEHICLES"])
register_ngsim_env()
cfg = {
    "scene": "us-101",
    "episode_root": os.path.join(os.environ["REPODIR"], "highway_env/data/processed_20s"),
    "prebuilt_split": "train",
    "simulation_period": {"episode_name": episode_name},
    "percentage_controlled_vehicles": requested,
    "control_all_vehicles": False,
    "show_trajectories": False,
    "max_episode_steps": 5,
    "observation": {
        "type": "LidarObservation",
        "cells": 16,
        "maximum_range": 64,
        "normalize": True,
    },
    "action": {"type": "ContinuousAction"},
    "action_mode": "continuous",
}
env = gym.make(ENV_ID, config=cfg)
try:
    env.reset(seed=0)
    selected = len(env.unwrapped.ego_ids)
finally:
    env.close()
if selected < 1:
    raise SystemExit(f"selected no ego vehicles from request {requested:g}")
if requested >= 1.0 and float(requested).is_integer() and selected > int(requested):
    raise SystemExit(
        f"selected {selected} ego vehicles from absolute-count request {int(requested)}"
    )
print(
    "sparse episode preflight ok:",
    episode_name,
    "requested",
    f"{requested:g}",
    "selected",
    selected,
)
PY

python "${REPODIR}/scripts_gail/train_simple_ps_gail.py" \
    --expert-data "${EXPERT_DATA}" \
    --scene us-101 \
    --action-mode "${ACTION_MODE}" \
    --discriminator-input action \
    --episode-root "${REPODIR}/highway_env/data/processed_20s" \
    --prebuilt-split train \
    --no-control-all-vehicles \
    "${CONTROLLED_VEHICLE_CURRICULUM_ARG}" \
    --initial-controlled-vehicles "${INITIAL_CONTROLLED_VEHICLES}" \
    --final-controlled-vehicles "${FINAL_CONTROLLED_VEHICLES}" \
    --controlled-vehicle-curriculum-rounds "${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS}" \
    --controlled-vehicle-increment-rounds "${CONTROLLED_VEHICLE_INCREMENT_ROUNDS}" \
    --controlled-vehicle-schedule "${CONTROLLED_VEHICLE_SCHEDULE}" \
    --warmup-rounds "${WARMUP_ROUNDS}" \
    --warmup-learning-rate "${WARMUP_LEARNING_RATE}" \
    --warmup-disc-learning-rate "${WARMUP_DISC_LEARNING_RATE}" \
    --warmup-entropy-coef "${WARMUP_ENTROPY_COEF}" \
    --warmup-clip-range "${WARMUP_CLIP_RANGE}" \
    --warmup-disc-updates-per-round "${WARMUP_DISC_UPDATES_PER_ROUND}" \
    --warmup-gail-reward-clip "${WARMUP_GAIL_REWARD_CLIP}" \
    --warmup-final-reward-clip "${WARMUP_FINAL_REWARD_CLIP}" \
    --vehicle-increase-warmup-rounds "${VEHICLE_INCREASE_WARMUP_ROUNDS}" \
    --enable-collision \
    --normalize-gail-reward \
    --gail-reward-clip "${GAIL_REWARD_CLIP}" \
    --collision-penalty "${COLLISION_PENALTY}" \
    --offroad-penalty "${OFFROAD_PENALTY}" \
    --final-reward-clip "${FINAL_REWARD_CLIP}" \
    --disc-expert-label "${DISC_EXPERT_LABEL}" \
    --disc-generator-label "${DISC_GENERATOR_LABEL}" \
    --disc-learning-rate "${DISC_LEARNING_RATE}" \
    --bc-pretrain-epochs "${BC_PRETRAIN_EPOCHS}" \
    --bc-pretrain-learning-rate "${BC_PRETRAIN_LEARNING_RATE}" \
    --bc-pretrain-batch-size "${BC_PRETRAIN_BATCH_SIZE}" \
    --bc-pretrain-micro-batch-size "${BC_PRETRAIN_MICRO_BATCH_SIZE}" \
    --bc-pretrain-validation-fraction "${BC_PRETRAIN_VALIDATION_FRACTION}" \
    --bc-pretrain-eval-episodes "${BC_PRETRAIN_EVAL_EPISODES}" \
    --bc-pretrain-min-mean-episode-length "${BC_PRETRAIN_MIN_MEAN_EPISODE_LENGTH}" \
    "${BC_PRETRAIN_ABORT_ARG}" \
    --policy-bc-regularization-coef "${POLICY_BC_REGULARIZATION_COEF}" \
    --policy-bc-regularization-final-coef "${POLICY_BC_REGULARIZATION_FINAL_COEF}" \
    --policy-bc-regularization-decay-rounds "${POLICY_BC_REGULARIZATION_DECAY_ROUNDS}" \
    --rollout-target-agent-steps "${ROLLOUT_TARGET_AGENT_STEPS}" \
    --initial-rollout-target-agent-steps "${INITIAL_ROLLOUT_TARGET_AGENT_STEPS}" \
    --final-rollout-target-agent-steps "${FINAL_ROLLOUT_TARGET_AGENT_STEPS}" \
    --rollout-target-agent-steps-curriculum-rounds "${ROLLOUT_TARGET_AGENT_STEPS_CURRICULUM_ROUNDS}" \
    --rollout-target-agent-steps-schedule "${ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE}" \
    --initial-gamma "${INITIAL_GAMMA}" \
    --final-gamma "${FINAL_GAMMA}" \
    --gamma-curriculum-rounds "${GAMMA_CURRICULUM_ROUNDS}" \
    --gamma-schedule "${GAMMA_SCHEDULE}" \
    "${TERMINATION_ARG}" \
    --allow-idm \
    --device cuda \
    --total-rounds "${TOTAL_ROUNDS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --rollout-min-episodes "${ROLLOUT_MIN_EPISODES}" \
    --rollout-full-episodes \
    "${ROLLOUT_TARGET_AWARE_ARG}" \
    --rollout-target-min-episodes "${ROLLOUT_TARGET_MIN_EPISODES}" \
    --rollout-target-episode-safety-factor "${ROLLOUT_TARGET_EPISODE_SAFETY_FACTOR}" \
    "${ROLLOUT_TRAINING_SUBSAMPLE_ARG}" \
    --rollout-training-agent-steps "${ROLLOUT_TRAINING_AGENT_STEPS}" \
    --rollout-max-episode-steps "${ROLLOUT_MAX_EPISODE_STEPS}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --max-episode-steps-schedule "${MAX_EPISODE_STEPS_SCHEDULE}" \
    --trajectory-frame "${TRAJECTORY_FRAME}" \
    --num-rollout-workers "${ROLLOUT_WORKERS}" \
    --rollout-worker-threads "${ROLLOUT_WORKER_THREADS}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    --policy-model "${POLICY_MODEL}" \
    --hidden-size "${HIDDEN_SIZE}" \
    "${CENTRALIZED_CRITIC_ARG}" \
    --central-critic-max-vehicles "${CENTRAL_CRITIC_MAX_VEHICLES}" \
    "${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS_ARG}" \
    --discriminator-hidden-sizes "${DISCRIMINATOR_HIDDEN_SIZES}" \
    --discriminator-dropout "${DISCRIMINATOR_DROPOUT}" \
    "${DISCRIMINATOR_SPECTRAL_NORM_ARG}" \
    --transformer-layers "${TRANSFORMER_LAYERS}" \
    --transformer-heads "${TRANSFORMER_HEADS}" \
    --transformer-dropout "${TRANSFORMER_DROPOUT}" \
    --batch-size "${BATCH_SIZE}" \
    --disc-batch-size "${DISC_BATCH_SIZE}" \
    --disc-updates-per-round "${DISC_UPDATES_PER_ROUND}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${CHECKPOINT_VIDEO_ARG}" \
    --checkpoint-video-every "${CHECKPOINT_VIDEO_EVERY}" \
    --checkpoint-video-steps "${CHECKPOINT_VIDEO_STEPS}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project highwayenv-ps-gail \
    --wandb-tags ps-gail,continuous,collision,idm,gpu,32c \
    --run-name "${RUN_NAME}"
