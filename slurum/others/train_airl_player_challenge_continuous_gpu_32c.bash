#!/bin/bash
#SBATCH --job-name=ps_airl_challenge_32c
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_airl_challenge_32c_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_airl_challenge_32c_%j.err

set -euo pipefail

bool_arg() {
    local name="$1"
    local value="$2"
    if [ "${value}" = "true" ]; then
        printf '%s\n' "--${name}"
    else
        printf '%s\n' "--no-${name}"
    fi
}

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

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
export MPLCONFIGDIR="${REPODIR}/logs/matplotlib_${SLURM_JOB_ID:-manual}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPODIR}"
mkdir -p "${REPODIR}/logs/airl" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

RUN_NAME="${RUN_NAME:-ps_airl_player_challenge_continuous_gpu_32c_${SLURM_JOB_ID:-manual}}"
AIRL_TRAIN_SCRIPT="${AIRL_TRAIN_SCRIPT:-${REPODIR}/scripts_gail/train_simple_airl.py}"
ACTION_MODE="${ACTION_MODE:-continuous}"
SCENE="${SCENE:-us-101}"
EPISODE_ROOT="${EPISODE_ROOT:-${REPODIR}/highway_env/data/processed_20s}"
EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_unified_expert_continuous_55145982}"
PREBUILT_SPLIT="${PREBUILT_SPLIT:-train}"
WANDB_MODE="${WANDB_MODE:-online}"

TOTAL_ROUNDS="${TOTAL_ROUNDS:-800}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
ROLLOUT_MIN_EPISODES="${ROLLOUT_MIN_EPISODES:-2}"
ROLLOUT_TARGET_MIN_EPISODES="${ROLLOUT_TARGET_MIN_EPISODES:-${ROLLOUT_WORKERS}}"
ROLLOUT_TARGET_EPISODE_SAFETY_FACTOR="${ROLLOUT_TARGET_EPISODE_SAFETY_FACTOR:-1.25}"
ROLLOUT_TRAINING_AGENT_STEPS="${ROLLOUT_TRAINING_AGENT_STEPS:-0}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-100000}"
TRAJECTORY_FRAME="${TRAJECTORY_FRAME:-relative}"

CONTROLLED_VEHICLE_CURRICULUM="${CONTROLLED_VEHICLE_CURRICULUM:-true}"
# CONTROLLED_VEHICLE_SCHEDULE below is authoritative; these are fallback
# endpoints only if the schedule is explicitly removed.
INITIAL_CONTROLLED_VEHICLES="${INITIAL_CONTROLLED_VEHICLES:-10}"
FINAL_CONTROLLED_VEHICLES="${FINAL_CONTROLLED_VEHICLES:-100}"
CONTROLLED_VEHICLE_CURRICULUM_ROUNDS="${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
CONTROLLED_VEHICLE_INCREMENT_ROUNDS="${CONTROLLED_VEHICLE_INCREMENT_ROUNDS:-0}"
CONTROLLED_VEHICLE_SCHEDULE="${CONTROLLED_VEHICLE_SCHEDULE:-0:120:10:10;120:240:20:20;240:360:30:30;360:480:40:40;480:600:50:50;600:800:100:100}"

PSRO_LITE="${PSRO_LITE:-true}"
PSRO_ARCHIVE_EVERY="${PSRO_ARCHIVE_EVERY:-20}"
PSRO_ARCHIVE_SIZE="${PSRO_ARCHIVE_SIZE:-5}"
PSRO_MIXTURE_AFTER_JUMP_ROUNDS="${PSRO_MIXTURE_AFTER_JUMP_ROUNDS:-20}"
PSRO_CURRENT_POLICY_FRACTION="${PSRO_CURRENT_POLICY_FRACTION:-0.65}"

DISCRIMINATOR_LOSS="${DISCRIMINATOR_LOSS:-wgan_gp}"
DISC_LEARNING_RATE="${DISC_LEARNING_RATE:-0.0004}"
DISC_UPDATES_PER_ROUND="${DISC_UPDATES_PER_ROUND:-2}"
DISC_BATCH_SIZE="${DISC_BATCH_SIZE:-4096}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-${DISC_BATCH_SIZE}}"
AIRL_LOG_PROB_BATCH_SIZE="${AIRL_LOG_PROB_BATCH_SIZE:-512}"
DISCRIMINATOR_REPLAY_ROUNDS="${DISCRIMINATOR_REPLAY_ROUNDS:-3}"
DISCRIMINATOR_REPLAY_MAX_SAMPLES="${DISCRIMINATOR_REPLAY_MAX_SAMPLES:-120000}"
DISCRIMINATOR_HIDDEN_SIZES="${DISCRIMINATOR_HIDDEN_SIZES:-256,256,128}"
DISCRIMINATOR_DROPOUT="${DISCRIMINATOR_DROPOUT:-0.0}"
DISCRIMINATOR_SPECTRAL_NORM="${DISCRIMINATOR_SPECTRAL_NORM:-true}"
DISC_EXPERT_LABEL="${DISC_EXPERT_LABEL:-0.8}"
DISC_GENERATOR_LABEL="${DISC_GENERATOR_LABEL:-0.2}"
WGAN_GP_LAMBDA="${WGAN_GP_LAMBDA:-2.0}"

LEARNING_RATE="${LEARNING_RATE:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
ENTROPY_COEF="${ENTROPY_COEF:-0.015}"
CLIP_RANGE="${CLIP_RANGE:-0.20}"
GAIL_REWARD_CLIP="${GAIL_REWARD_CLIP:-5.0}"
FINAL_REWARD_CLIP="${FINAL_REWARD_CLIP:-10.0}"
COLLISION_PENALTY="${COLLISION_PENALTY:-2.0}"
OFFROAD_PENALTY="${OFFROAD_PENALTY:-2.0}"
NORMALIZE_GAIL_REWARD="${NORMALIZE_GAIL_REWARD:-true}"
ALLOW_WGAN_REWARD_NORMALIZATION="${ALLOW_WGAN_REWARD_NORMALIZATION:-true}"
WGAN_REWARD_CENTER="${WGAN_REWARD_CENTER:-false}"
WGAN_REWARD_CLIP="${WGAN_REWARD_CLIP:-0.0}"
WGAN_REWARD_SCALE="${WGAN_REWARD_SCALE:-1.0}"
WGAN_REWARD_NORM_MIN_STD="${WGAN_REWARD_NORM_MIN_STD:-0.001}"
WGAN_REWARD_NORM_CLIP="${WGAN_REWARD_NORM_CLIP:-5.0}"
AIRL_POLICY_REWARD_MODE="${AIRL_POLICY_REWARD_MODE:-discriminator}"

POLICY_MODEL="${POLICY_MODEL:-transformer}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.1}"
TRANSFORMER_TEMPORAL_MODULE="${TRANSFORMER_TEMPORAL_MODULE:-true}"
TRANSFORMER_TEMPORAL_KERNEL_SIZE="${TRANSFORMER_TEMPORAL_KERNEL_SIZE:-5}"
TRANSFORMER_TEMPORAL_LAYERS="${TRANSFORMER_TEMPORAL_LAYERS:-1}"
CENTRALIZED_CRITIC="${CENTRALIZED_CRITIC:-true}"
CENTRAL_CRITIC_MAX_VEHICLES="${CENTRAL_CRITIC_MAX_VEHICLES:-128}"
CENTRAL_CRITIC_INCLUDE_LOCAL_OBS="${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS:-false}"
CENTRAL_CRITIC_POOLING="${CENTRAL_CRITIC_POOLING:-attention}"
CENTRAL_CRITIC_ATTENTION_HEADS="${CENTRAL_CRITIC_ATTENTION_HEADS:-4}"

ENABLE_PLAYER_CHALLENGE_REWARD="${ENABLE_PLAYER_CHALLENGE_REWARD:-true}"
CHALLENGE_REWARD_COEF="${CHALLENGE_REWARD_COEF:-0.2}"
CHALLENGE_REWARD_CLIP="${CHALLENGE_REWARD_CLIP:-0.25}"
CHALLENGE_MAX_PRIMARY_REWARD_FRACTION="${CHALLENGE_MAX_PRIMARY_REWARD_FRACTION:-0.10}"
CHALLENGE_TTC_TARGET="${CHALLENGE_TTC_TARGET:-0.0}"
CHALLENGE_TTC_MARGIN="${CHALLENGE_TTC_MARGIN:-0.75}"
CHALLENGE_TTC_FLOOR="${CHALLENGE_TTC_FLOOR:-0.0}"
CHALLENGE_GAP_TARGET="${CHALLENGE_GAP_TARGET:-0.0}"
CHALLENGE_GAP_FLOOR="${CHALLENGE_GAP_FLOOR:-0.0}"
CHALLENGE_TTC_WEIGHT="${CHALLENGE_TTC_WEIGHT:-0.6}"
CHALLENGE_GAP_WEIGHT="${CHALLENGE_GAP_WEIGHT:-0.4}"
CHALLENGE_CRASH_WEIGHT="${CHALLENGE_CRASH_WEIGHT:-4.0}"
CHALLENGE_OFFROAD_WEIGHT="${CHALLENGE_OFFROAD_WEIGHT:-2.0}"
CHALLENGE_RISK_EMA_BETA="${CHALLENGE_RISK_EMA_BETA:-0.95}"
CHALLENGE_EXPERT_LIKE_QUANTILE="${CHALLENGE_EXPERT_LIKE_QUANTILE:-0.25}"

CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
SAVE_CHECKPOINT_VIDEO="${SAVE_CHECKPOINT_VIDEO:-true}"
CHECKPOINT_VIDEO_EVERY="${CHECKPOINT_VIDEO_EVERY:-50}"
CHECKPOINT_VIDEO_STEPS="${CHECKPOINT_VIDEO_STEPS:-200}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Scene: ${SCENE}"
echo "Expert data: ${EXPERT_DATA}"
echo "AIRL train script: ${AIRL_TRAIN_SCRIPT}"
echo "Rollout workers: ${ROLLOUT_WORKERS} threads=${ROLLOUT_WORKER_THREADS}"
echo "PSRO-lite: ${PSRO_LITE} archive_every=${PSRO_ARCHIVE_EVERY} archive_size=${PSRO_ARCHIVE_SIZE}"
echo "AIRL hard selector: not used by AIRL reward-model training"
echo "Challenge reward: ${ENABLE_PLAYER_CHALLENGE_REWARD} coef=${CHALLENGE_REWARD_COEF} cap_fraction=${CHALLENGE_MAX_PRIMARY_REWARD_FRACTION} abs_clip=${CHALLENGE_REWARD_CLIP}"
echo "Challenge targets: ttc=${CHALLENGE_TTC_TARGET} gap=${CHALLENGE_GAP_TARGET} (0 derives from ${SCENE} IDM parameters)"
echo "AIRL policy reward mode: ${AIRL_POLICY_REWARD_MODE}"
nvidia-smi || true

if [ ! -f "${AIRL_TRAIN_SCRIPT}" ]; then
    echo "AIRL train script not found: ${AIRL_TRAIN_SCRIPT}" >&2
    exit 2
fi

python "${AIRL_TRAIN_SCRIPT}" \
    --expert-data "${EXPERT_DATA}" \
    --scene "${SCENE}" \
    --episode-root "${EPISODE_ROOT}" \
    --prebuilt-split "${PREBUILT_SPLIT}" \
    --run-name "${RUN_NAME}" \
    --action-mode "${ACTION_MODE}" \
    --no-control-all-vehicles \
    "$(bool_arg controlled-vehicle-curriculum "${CONTROLLED_VEHICLE_CURRICULUM}")" \
    --initial-controlled-vehicles "${INITIAL_CONTROLLED_VEHICLES}" \
    --final-controlled-vehicles "${FINAL_CONTROLLED_VEHICLES}" \
    --controlled-vehicle-curriculum-rounds "${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS}" \
    --controlled-vehicle-increment-rounds "${CONTROLLED_VEHICLE_INCREMENT_ROUNDS}" \
    --controlled-vehicle-schedule "${CONTROLLED_VEHICLE_SCHEDULE}" \
    --enable-collision \
    --terminate-when-all-controlled-crashed \
    --allow-idm \
    --device cuda \
    --total-rounds "${TOTAL_ROUNDS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --rollout-min-episodes "${ROLLOUT_MIN_EPISODES}" \
    --rollout-full-episodes \
    --rollout-target-aware-episodes \
    --rollout-target-min-episodes "${ROLLOUT_TARGET_MIN_EPISODES}" \
    --rollout-target-episode-safety-factor "${ROLLOUT_TARGET_EPISODE_SAFETY_FACTOR}" \
    --rollout-training-subsample \
    --rollout-training-agent-steps "${ROLLOUT_TRAINING_AGENT_STEPS}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --trajectory-frame "${TRAJECTORY_FRAME}" \
    --num-rollout-workers "${ROLLOUT_WORKERS}" \
    --rollout-worker-threads "${ROLLOUT_WORKER_THREADS}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    "$(bool_arg psro-lite "${PSRO_LITE}")" \
    --psro-archive-every "${PSRO_ARCHIVE_EVERY}" \
    --psro-archive-size "${PSRO_ARCHIVE_SIZE}" \
    --psro-mixture-after-jump-rounds "${PSRO_MIXTURE_AFTER_JUMP_ROUNDS}" \
    --psro-current-policy-fraction "${PSRO_CURRENT_POLICY_FRACTION}" \
    --policy-model "${POLICY_MODEL}" \
    --hidden-size "${HIDDEN_SIZE}" \
    "$(bool_arg centralized-critic "${CENTRALIZED_CRITIC}")" \
    --central-critic-max-vehicles "${CENTRAL_CRITIC_MAX_VEHICLES}" \
    "$(bool_arg central-critic-include-local-obs "${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS}")" \
    --central-critic-pooling "${CENTRAL_CRITIC_POOLING}" \
    --central-critic-attention-heads "${CENTRAL_CRITIC_ATTENTION_HEADS}" \
    --transformer-layers "${TRANSFORMER_LAYERS}" \
    --transformer-heads "${TRANSFORMER_HEADS}" \
    --transformer-dropout "${TRANSFORMER_DROPOUT}" \
    "$(bool_arg transformer-temporal-module "${TRANSFORMER_TEMPORAL_MODULE}")" \
    --transformer-temporal-kernel-size "${TRANSFORMER_TEMPORAL_KERNEL_SIZE}" \
    --transformer-temporal-layers "${TRANSFORMER_TEMPORAL_LAYERS}" \
    --learning-rate "${LEARNING_RATE}" \
    --batch-size "${BATCH_SIZE}" \
    --entropy-coef "${ENTROPY_COEF}" \
    --clip-range "${CLIP_RANGE}" \
    --discriminator-loss "${DISCRIMINATOR_LOSS}" \
    --disc-expert-label "${DISC_EXPERT_LABEL}" \
    --disc-generator-label "${DISC_GENERATOR_LABEL}" \
    --disc-learning-rate "${DISC_LEARNING_RATE}" \
    --disc-updates-per-round "${DISC_UPDATES_PER_ROUND}" \
    --disc-batch-size "${DISC_BATCH_SIZE}" \
    --reward-batch-size "${REWARD_BATCH_SIZE}" \
    --airl-log-prob-batch-size "${AIRL_LOG_PROB_BATCH_SIZE}" \
    --wgan-gp-lambda "${WGAN_GP_LAMBDA}" \
    --discriminator-replay-rounds "${DISCRIMINATOR_REPLAY_ROUNDS}" \
    --discriminator-replay-max-samples "${DISCRIMINATOR_REPLAY_MAX_SAMPLES}" \
    --discriminator-hidden-sizes "${DISCRIMINATOR_HIDDEN_SIZES}" \
    --discriminator-dropout "${DISCRIMINATOR_DROPOUT}" \
    "$(bool_arg discriminator-spectral-norm "${DISCRIMINATOR_SPECTRAL_NORM}")" \
    "$(bool_arg normalize-gail-reward "${NORMALIZE_GAIL_REWARD}")" \
    "$(bool_arg allow-wgan-reward-normalization "${ALLOW_WGAN_REWARD_NORMALIZATION}")" \
    "$(bool_arg wgan-reward-center "${WGAN_REWARD_CENTER}")" \
    --wgan-reward-clip "${WGAN_REWARD_CLIP}" \
    --wgan-reward-scale "${WGAN_REWARD_SCALE}" \
    --wgan-reward-norm-min-std "${WGAN_REWARD_NORM_MIN_STD}" \
    --wgan-reward-norm-clip "${WGAN_REWARD_NORM_CLIP}" \
    --airl-policy-reward-mode "${AIRL_POLICY_REWARD_MODE}" \
    --gail-reward-clip "${GAIL_REWARD_CLIP}" \
    --final-reward-clip "${FINAL_REWARD_CLIP}" \
    --collision-penalty "${COLLISION_PENALTY}" \
    --offroad-penalty "${OFFROAD_PENALTY}" \
    --no-enable-hard-example-selection \
    "$(bool_arg enable-player-challenge-reward "${ENABLE_PLAYER_CHALLENGE_REWARD}")" \
    --challenge-reward-coef "${CHALLENGE_REWARD_COEF}" \
    --challenge-reward-clip "${CHALLENGE_REWARD_CLIP}" \
    --challenge-max-primary-reward-fraction "${CHALLENGE_MAX_PRIMARY_REWARD_FRACTION}" \
    --challenge-ttc-target "${CHALLENGE_TTC_TARGET}" \
    --challenge-ttc-margin "${CHALLENGE_TTC_MARGIN}" \
    --challenge-ttc-floor "${CHALLENGE_TTC_FLOOR}" \
    --challenge-gap-target "${CHALLENGE_GAP_TARGET}" \
    --challenge-gap-floor "${CHALLENGE_GAP_FLOOR}" \
    --challenge-ttc-weight "${CHALLENGE_TTC_WEIGHT}" \
    --challenge-gap-weight "${CHALLENGE_GAP_WEIGHT}" \
    --challenge-crash-weight "${CHALLENGE_CRASH_WEIGHT}" \
    --challenge-offroad-weight "${CHALLENGE_OFFROAD_WEIGHT}" \
    --challenge-risk-ema-beta "${CHALLENGE_RISK_EMA_BETA}" \
    --challenge-expert-like-quantile "${CHALLENGE_EXPERT_LIKE_QUANTILE}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "$(bool_arg save-checkpoint-video "${SAVE_CHECKPOINT_VIDEO}")" \
    --checkpoint-video-every "${CHECKPOINT_VIDEO_EVERY}" \
    --checkpoint-video-steps "${CHECKPOINT_VIDEO_STEPS}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project highwayenv-ps-gail \
    --wandb-tags "ps-airl,airl,continuous,player-challenge,psro-lite-${PSRO_LITE},scene-${SCENE}"
