#!/bin/bash
#SBATCH --job-name=ps_gail_seq_tf_64c
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_seq_tf_64c_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_seq_tf_64c_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

ROLLOUT_WORKER_THREADS="${ROLLOUT_WORKER_THREADS:-2}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-$((SLURM_CPUS_PER_TASK / ROLLOUT_WORKER_THREADS))}"
CGAIL_K="${CGAIL_K:-0.0}"
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

export EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_traj_expert_discrete_54902119}"
POLICY_MODEL="${POLICY_MODEL:-transformer}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.1}"
DISCRIMINATOR_LOSS="${DISCRIMINATOR_LOSS:-wgan_gp}"
WGAN_GP_LAMBDA="${WGAN_GP_LAMBDA:-2.0}"
WGAN_REWARD_CENTER="${WGAN_REWARD_CENTER:-true}"
WGAN_REWARD_CLIP="${WGAN_REWARD_CLIP:-2.0}"
WGAN_REWARD_SCALE="${WGAN_REWARD_SCALE:-1.0}"
SEQUENCE_FEATURE_MODE="${SEQUENCE_FEATURE_MODE:-local_deltas}"
NORMALIZE_DISCRIMINATOR_FEATURES="${NORMALIZE_DISCRIMINATOR_FEATURES:-true}"
DISCRIMINATOR_FEATURE_CLIP="${DISCRIMINATOR_FEATURE_CLIP:-10.0}"
ENABLE_ACTION_MASKING="${ENABLE_ACTION_MASKING:-true}"
ALLOW_IDM="${ALLOW_IDM:-true}"
RUN_NAME="${RUN_NAME:-ps_gail_seq_tf_${DISCRIMINATOR_LOSS}_${SEQUENCE_FEATURE_MODE}_mask${ENABLE_ACTION_MASKING}_300r_200h_64c_${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-300}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
ROLLOUT_MIN_EPISODES="${ROLLOUT_MIN_EPISODES:-${ROLLOUT_WORKERS}}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-100000}"
TRAJECTORY_FRAME="${TRAJECTORY_FRAME:-relative}"
INITIAL_CONTROLLED_VEHICLE_FRACTION="${INITIAL_CONTROLLED_VEHICLE_FRACTION:-0.15}"
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
SAVE_CHECKPOINT_VIDEO="${SAVE_CHECKPOINT_VIDEO:-true}"
CHECKPOINT_VIDEO_STEPS="${CHECKPOINT_VIDEO_STEPS:-120}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DISC_BATCH_SIZE="${DISC_BATCH_SIZE:-8192}"
DISC_UPDATES_PER_ROUND="${DISC_UPDATES_PER_ROUND:-2}"
PPO_EPOCHS="${PPO_EPOCHS:-6}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
TERMINATE_WHEN_ALL_CONTROLLED_CRASHED="${TERMINATE_WHEN_ALL_CONTROLLED_CRASHED:-true}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Rollout workers: ${ROLLOUT_WORKERS}"
echo "Rollout worker threads: ${ROLLOUT_WORKER_THREADS}"
echo "Policy model: ${POLICY_MODEL}"
echo "Transformer layers/heads/dropout: ${TRANSFORMER_LAYERS}/${TRANSFORMER_HEADS}/${TRANSFORMER_DROPOUT}"
echo "C-GAIL discriminator k (BCE only): ${CGAIL_K}"
echo "Discriminator loss: ${DISCRIMINATOR_LOSS}"
echo "WGAN-GP lambda: ${WGAN_GP_LAMBDA}"
echo "WGAN reward center: ${WGAN_REWARD_CENTER}"
echo "WGAN reward clip: ${WGAN_REWARD_CLIP}"
echo "WGAN reward scale: ${WGAN_REWARD_SCALE}"
echo "Discriminator updates per round: ${DISC_UPDATES_PER_ROUND}"
echo "Sequence feature mode: ${SEQUENCE_FEATURE_MODE}"
echo "Normalize discriminator features: ${NORMALIZE_DISCRIMINATOR_FEATURES}"
echo "Discriminator feature clip: ${DISCRIMINATOR_FEATURE_CLIP}"
echo "Action masking: ${ENABLE_ACTION_MASKING}"
echo "Allow IDM: ${ALLOW_IDM}"
echo "Total rounds: ${TOTAL_ROUNDS}"
echo "Rollout steps: ${ROLLOUT_STEPS}"
echo "Rollout min episodes: ${ROLLOUT_MIN_EPISODES}"
echo "Policy batch size: ${BATCH_SIZE}"
echo "Discriminator batch size: ${DISC_BATCH_SIZE}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

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
if [ "${NORMALIZE_DISCRIMINATOR_FEATURES}" = "true" ]; then
    DISC_FEATURE_NORM_ARG="--normalize-discriminator-features"
else
    DISC_FEATURE_NORM_ARG="--no-normalize-discriminator-features"
fi
if [ "${ENABLE_ACTION_MASKING}" = "true" ]; then
    ACTION_MASKING_ARG="--enable-action-masking"
else
    ACTION_MASKING_ARG="--no-enable-action-masking"
fi
if [ "${ALLOW_IDM}" = "true" ]; then
    ALLOW_IDM_ARG="--allow-idm"
else
    ALLOW_IDM_ARG="--no-allow-idm"
fi
if [ "${WGAN_REWARD_CENTER}" = "true" ]; then
    WGAN_REWARD_CENTER_ARG="--wgan-reward-center"
else
    WGAN_REWARD_CENTER_ARG="--no-wgan-reward-center"
fi

python -u "${REPODIR}/scripts_gail/train_simple_ps_gail.py" \
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
    --discriminator-loss "${DISCRIMINATOR_LOSS}" \
    --wgan-gp-lambda "${WGAN_GP_LAMBDA}" \
    "${WGAN_REWARD_CENTER_ARG}" \
    --wgan-reward-clip "${WGAN_REWARD_CLIP}" \
    --wgan-reward-scale "${WGAN_REWARD_SCALE}" \
    "${DISC_FEATURE_NORM_ARG}" \
    --discriminator-feature-clip "${DISCRIMINATOR_FEATURE_CLIP}" \
    "${ACTION_MASKING_ARG}" \
    --cgail-k "${CGAIL_K}" \
    "${TERMINATION_ARG}" \
    "${ALLOW_IDM_ARG}" \
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
    --policy-model "${POLICY_MODEL}" \
    --transformer-layers "${TRANSFORMER_LAYERS}" \
    --transformer-heads "${TRANSFORMER_HEADS}" \
    --transformer-dropout "${TRANSFORMER_DROPOUT}" \
    --enable-sequence-discriminator \
    --sequence-feature-mode "${SEQUENCE_FEATURE_MODE}" \
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
    --wandb-tags ps-gail,sequence-discriminator,transformer-policy,wgan-gp,action-masking,local-deltas,collision,allow-idm-${ALLOW_IDM},gpu,64c,long-run \
    --run-name "${RUN_NAME}"
