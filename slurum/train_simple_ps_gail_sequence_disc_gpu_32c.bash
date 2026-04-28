#!/bin/bash
#SBATCH --job-name=ps_gail_seq_32c
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_seq_32c_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_seq_32c_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Section 1: cluster/runtime setup
# ---------------------------------------------------------------------------
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
mkdir -p "${REPODIR}/logs" "${REPODIR}/logs/simple_ps_gail" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Rollout workers: ${ROLLOUT_WORKERS}"
echo "Rollout worker threads: ${ROLLOUT_WORKER_THREADS}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# ---------------------------------------------------------------------------
# Section 2: autoregressive sequence-discriminator training configuration and launch
# ---------------------------------------------------------------------------
# This mode only requires the standard per-vehicle expert arrays:
# observations, trajectory_states, vehicle_ids, and timesteps.
EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_traj_expert_discrete_54902119}"
RUN_NAME="${RUN_NAME:-ps_gail_sequence_disc_32c_${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-200}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
ROLLOUT_MIN_EPISODES="${ROLLOUT_MIN_EPISODES:-4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-100000}"
TRAJECTORY_FRAME="${TRAJECTORY_FRAME:-relative}"
INITIAL_CONTROLLED_VEHICLE_FRACTION="${INITIAL_CONTROLLED_VEHICLE_FRACTION:-0.20}"
FINAL_CONTROLLED_VEHICLE_FRACTION="${FINAL_CONTROLLED_VEHICLE_FRACTION:-1.0}"
CONTROLLED_VEHICLE_CURRICULUM_ROUNDS="${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-1}"
SEQUENCE_REWARD_COEF="${SEQUENCE_REWARD_COEF:-1.0}"
DISC_EXPERT_LABEL="${DISC_EXPERT_LABEL:-0.9}"
DISC_GENERATOR_LABEL="${DISC_GENERATOR_LABEL:-0.1}"
COLLISION_PENALTY="${COLLISION_PENALTY:-2.0}"
OFFROAD_PENALTY="${OFFROAD_PENALTY:-2.0}"
GAIL_REWARD_CLIP="${GAIL_REWARD_CLIP:-5.0}"
FINAL_REWARD_CLIP="${FINAL_REWARD_CLIP:-10.0}"
SAVE_CHECKPOINT_VIDEO="${SAVE_CHECKPOINT_VIDEO:-false}"
CHECKPOINT_VIDEO_STEPS="${CHECKPOINT_VIDEO_STEPS:-120}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
DISC_BATCH_SIZE="${DISC_BATCH_SIZE:-4096}"
DISC_UPDATES_PER_ROUND="${DISC_UPDATES_PER_ROUND:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
TERMINATE_WHEN_ALL_CONTROLLED_CRASHED="${TERMINATE_WHEN_ALL_CONTROLLED_CRASHED:-true}"

python - <<'PY'
import os
import numpy as np

root = os.environ["EXPERT_DATA"]
if not os.path.exists(root):
    raise SystemExit(f"EXPERT_DATA does not exist: {root}")
files = []
if os.path.isdir(root):
    files = sorted(os.path.join(root, name) for name in os.listdir(root) if name.endswith(".npz"))
else:
    files = [root]
if not files:
    raise SystemExit(f"No expert .npz files found under {root}")
with np.load(files[0], allow_pickle=True) as data:
    missing = {"observations", "trajectory_states", "vehicle_ids", "timesteps"}.difference(data.files)
    if missing:
        raise SystemExit(f"{files[0]} is missing {sorted(missing)} for sequence-discriminator training.")
    print("sequence expert preflight ok:", files[0], "samples", len(data["observations"]))
PY

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

python "${REPODIR}/scripts_gail/train_simple_ps_gail.py" \
    --expert-data "${EXPERT_DATA}" \
    --scene us-101 \
    --episode-root "${REPODIR}/highway_env/data/processed_20s" \
    --prebuilt-split train \
    --no-control-all-vehicles \
    --controlled-vehicle-curriculum \
    --initial-controlled-vehicle-fraction "${INITIAL_CONTROLLED_VEHICLE_FRACTION}" \
    --final-controlled-vehicle-fraction "${FINAL_CONTROLLED_VEHICLE_FRACTION}" \
    --controlled-vehicle-curriculum-rounds "${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS}" \
    --enable-collision \
    --normalize-gail-reward \
    --gail-reward-clip "${GAIL_REWARD_CLIP}" \
    --collision-penalty "${COLLISION_PENALTY}" \
    --offroad-penalty "${OFFROAD_PENALTY}" \
    --final-reward-clip "${FINAL_REWARD_CLIP}" \
    --disc-expert-label "${DISC_EXPERT_LABEL}" \
    --disc-generator-label "${DISC_GENERATOR_LABEL}" \
    "${TERMINATION_ARG}" \
    --allow-idm \
    --device cuda \
    --total-rounds "${TOTAL_ROUNDS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --rollout-min-episodes "${ROLLOUT_MIN_EPISODES}" \
    --rollout-full-episodes \
    --rollout-max-episode-steps 0 \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --trajectory-frame "${TRAJECTORY_FRAME}" \
    --num-rollout-workers "${ROLLOUT_WORKERS}" \
    --rollout-worker-threads "${ROLLOUT_WORKER_THREADS}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    --enable-sequence-discriminator \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --sequence-stride "${SEQUENCE_STRIDE}" \
    --sequence-reward-coef "${SEQUENCE_REWARD_COEF}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --disc-batch-size "${DISC_BATCH_SIZE}" \
    --disc-updates-per-round "${DISC_UPDATES_PER_ROUND}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${CHECKPOINT_VIDEO_ARG}" \
    --checkpoint-video-steps "${CHECKPOINT_VIDEO_STEPS}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project highwayenv-ps-gail \
    --wandb-tags ps-gail,sequence-discriminator,collision,idm,gpu,32c \
    --run-name "${RUN_NAME}"
