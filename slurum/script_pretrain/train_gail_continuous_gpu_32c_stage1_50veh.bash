#!/bin/bash
#SBATCH --job-name=ps_gail_stage1_50veh
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_stage1_50veh_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_stage1_50veh_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

# Rollouts run in multiple CPU worker processes while PyTorch updates run on
# the allocated GPU. Give each rollout worker a small native thread pool so the
# total CPU use stays near SLURM_CPUS_PER_TASK.
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-32}"
ROLLOUT_WORKER_THREADS="${ROLLOUT_WORKER_THREADS:-2}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
EVALUATION_WORKER_THREADS="${EVALUATION_WORKER_THREADS:-2}"
EVALUATION_WORKERS="${EVALUATION_WORKERS:-$((SLURM_CPUS / EVALUATION_WORKER_THREADS))}"
if [ "${ROLLOUT_WORKERS}" -lt 1 ]; then
    ROLLOUT_WORKERS=1
fi
if [ "${EVALUATION_WORKERS}" -lt 1 ]; then
    EVALUATION_WORKERS=1
fi
REQUESTED_ROLLOUT_CPUS=$((ROLLOUT_WORKERS * ROLLOUT_WORKER_THREADS))
if [ "${REQUESTED_ROLLOUT_CPUS}" -gt "${SLURM_CPUS}" ]; then
    echo "Requested rollout CPU use (${REQUESTED_ROLLOUT_CPUS}) exceeds SLURM_CPUS_PER_TASK (${SLURM_CPUS})." >&2
    exit 1
fi
REQUESTED_EVALUATION_CPUS=$((EVALUATION_WORKERS * EVALUATION_WORKER_THREADS))
if [ "${REQUESTED_EVALUATION_CPUS}" -gt "${SLURM_CPUS}" ]; then
    echo "Requested evaluation CPU use (${REQUESTED_EVALUATION_CPUS}) exceeds SLURM_CPUS_PER_TASK (${SLURM_CPUS})." >&2
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
echo "Evaluation workers: ${EVALUATION_WORKERS}"
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

RUN_NAME="${RUN_NAME:-ps_gail_stage1_50veh_${SLURM_JOB_ID}}"
ACTION_MODE="${ACTION_MODE:-continuous}"
SCENE="${SCENE:-us-101}"
EPISODE_ROOT="${EPISODE_ROOT:-${REPODIR}/highway_env/data/processed_20s}"
PREBUILT_SPLIT="${PREBUILT_SPLIT:-train}"
EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_unified_expert_continuous_55145982}"
WANDB_MODE="${WANDB_MODE:-online}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-600}"
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
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-0}"
TRAJECTORY_FRAME="${TRAJECTORY_FRAME:-relative}"
# Piecewise schedules below are the source of truth for stage one. The older
# fraction defaults are intentionally commented out because they are superseded
# whenever CONTROLLED_VEHICLE_SCHEDULE is non-empty.
# INITIAL_CONTROLLED_VEHICLE_FRACTION="${INITIAL_CONTROLLED_VEHICLE_FRACTION:-0.05}"
# FINAL_CONTROLLED_VEHICLE_FRACTION="${FINAL_CONTROLLED_VEHICLE_FRACTION:-1.0}"
INITIAL_CONTROLLED_VEHICLES="${INITIAL_CONTROLLED_VEHICLES:-10}"
FINAL_CONTROLLED_VEHICLES="${FINAL_CONTROLLED_VEHICLES:-50}"
CONTROLLED_VEHICLE_CURRICULUM_ROUNDS="${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
CONTROLLED_VEHICLE_INCREMENT_ROUNDS="${CONTROLLED_VEHICLE_INCREMENT_ROUNDS:-0}"
CONTROLLED_VEHICLE_SCHEDULE_PROFILE="${CONTROLLED_VEHICLE_SCHEDULE_PROFILE:-sudden}"
CONTROLLED_VEHICLE_SCHEDULE_GRADUAL="${CONTROLLED_VEHICLE_SCHEDULE_GRADUAL:-0:100:10:20;100:200:20:30;200:300:30:40;300:600:40:50}"
CONTROLLED_VEHICLE_SCHEDULE_SUDDEN="${CONTROLLED_VEHICLE_SCHEDULE_SUDDEN:-0:120:10:10;120:240:20:20;240:360:30:30;360:480:40:40;480:600:50:50}"
if [ -z "${CONTROLLED_VEHICLE_SCHEDULE:-}" ]; then
    case "${CONTROLLED_VEHICLE_SCHEDULE_PROFILE}" in
        gradual)
            CONTROLLED_VEHICLE_SCHEDULE="${CONTROLLED_VEHICLE_SCHEDULE_GRADUAL}"
            ;;
        sudden)
            CONTROLLED_VEHICLE_SCHEDULE="${CONTROLLED_VEHICLE_SCHEDULE_SUDDEN}"
            ;;
        *)
            echo "Unsupported CONTROLLED_VEHICLE_SCHEDULE_PROFILE=${CONTROLLED_VEHICLE_SCHEDULE_PROFILE}. Use gradual or sudden." >&2
            exit 2
            ;;
    esac
fi
CONTROLLED_VEHICLE_CURRICULUM="${CONTROLLED_VEHICLE_CURRICULUM:-true}"
# Warmup is disabled for this stage; keep the knobs below only for reference.
WARMUP_ROUNDS=0
WARMUP_LEARNING_RATE="${WARMUP_LEARNING_RATE:-0}"
WARMUP_DISC_LEARNING_RATE="${WARMUP_DISC_LEARNING_RATE:-0}"
WARMUP_ENTROPY_COEF="${WARMUP_ENTROPY_COEF:-0.0015}"
WARMUP_CLIP_RANGE="${WARMUP_CLIP_RANGE:-0.10}"
WARMUP_DISC_UPDATES_PER_ROUND="${WARMUP_DISC_UPDATES_PER_ROUND:-0}"
WARMUP_GAIL_REWARD_CLIP="${WARMUP_GAIL_REWARD_CLIP:-2.0}"
WARMUP_FINAL_REWARD_CLIP="${WARMUP_FINAL_REWARD_CLIP:-5.0}"
VEHICLE_INCREASE_WARMUP_ROUNDS=0
ROLLOUT_TARGET_AGENT_STEPS="${ROLLOUT_TARGET_AGENT_STEPS:-0}"
INITIAL_ROLLOUT_TARGET_AGENT_STEPS="${INITIAL_ROLLOUT_TARGET_AGENT_STEPS:-10000}"
FINAL_ROLLOUT_TARGET_AGENT_STEPS="${FINAL_ROLLOUT_TARGET_AGENT_STEPS:-40000}"
ROLLOUT_TARGET_AGENT_STEPS_CURRICULUM_ROUNDS="${ROLLOUT_TARGET_AGENT_STEPS_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE="${ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE:-0:500:10000:10000;500:600:10000:20000;600:700:20000:30000;700:800:30000:40000}"
PSRO_LITE=true
PSRO_ARCHIVE_EVERY="${PSRO_ARCHIVE_EVERY:-20}"
PSRO_ARCHIVE_SIZE="${PSRO_ARCHIVE_SIZE:-5}"
PSRO_MIXTURE_AFTER_JUMP_ROUNDS="${PSRO_MIXTURE_AFTER_JUMP_ROUNDS:-20}"
PSRO_CURRENT_POLICY_FRACTION="${PSRO_CURRENT_POLICY_FRACTION:-0.65}"
INITIAL_GAMMA="${INITIAL_GAMMA:-0.95}"
FINAL_GAMMA="${FINAL_GAMMA:-0.99}"
GAMMA_CURRICULUM_ROUNDS="${GAMMA_CURRICULUM_ROUNDS:-${TOTAL_ROUNDS}}"
GAMMA_SCHEDULE="${GAMMA_SCHEDULE:-0:500:0.95:0.95;500:600:0.95:0.99;600:800:0.99:0.99}"
DISC_EXPERT_LABEL="${DISC_EXPERT_LABEL:-0.8}"
DISC_GENERATOR_LABEL="${DISC_GENERATOR_LABEL:-0.2}"
DISCRIMINATOR_LOSS="${DISCRIMINATOR_LOSS:-wgan_gp}"
DISCRIMINATOR_INPUT="${DISCRIMINATOR_INPUT:-action}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
LEARNING_RATE_SCHEDULE="${LEARNING_RATE_SCHEDULE:-}"
DISC_LEARNING_RATE="${DISC_LEARNING_RATE:-0.0004}"
DISC_LEARNING_RATE_SCHEDULE="${DISC_LEARNING_RATE_SCHEDULE:-}"
DISC_UPDATES_PER_ROUND="${DISC_UPDATES_PER_ROUND:-2}"
DISC_UPDATES_PER_ROUND_SCHEDULE="${DISC_UPDATES_PER_ROUND_SCHEDULE:-}"
DISCRIMINATOR_REPLAY_ROUNDS="${DISCRIMINATOR_REPLAY_ROUNDS:-3}"
DISCRIMINATOR_REPLAY_MAX_SAMPLES="${DISCRIMINATOR_REPLAY_MAX_SAMPLES:-120000}"
COLLISION_PENALTY="${COLLISION_PENALTY:-2.0}"
OFFROAD_PENALTY="${OFFROAD_PENALTY:-2.0}"
if [ -z "${NORMALIZE_GAIL_REWARD:-}" ]; then
    NORMALIZE_GAIL_REWARD="true"
fi
ALLOW_WGAN_REWARD_NORMALIZATION="${ALLOW_WGAN_REWARD_NORMALIZATION:-true}"
WGAN_REWARD_CENTER="${WGAN_REWARD_CENTER:-false}"
WGAN_REWARD_CLIP="${WGAN_REWARD_CLIP:-0.0}"
WGAN_REWARD_SCALE="${WGAN_REWARD_SCALE:-1.0}"
WGAN_REWARD_NORM_MIN_STD="${WGAN_REWARD_NORM_MIN_STD:-0.001}"
WGAN_REWARD_NORM_CLIP="${WGAN_REWARD_NORM_CLIP:-5.0}"
GAIL_REWARD_CLIP="${GAIL_REWARD_CLIP:-5.0}"
FINAL_REWARD_CLIP="${FINAL_REWARD_CLIP:-10.0}"
SAVE_CHECKPOINT_VIDEO="${SAVE_CHECKPOINT_VIDEO:-true}"
CHECKPOINT_VIDEO_EVERY="${CHECKPOINT_VIDEO_EVERY:-50}"
CHECKPOINT_VIDEO_STEPS="${CHECKPOINT_VIDEO_STEPS:-200}"
VALIDATION_EVERY="${VALIDATION_EVERY:-20}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-2}"
VALIDATION_PREBUILT_SPLIT="${VALIDATION_PREBUILT_SPLIT:-val}"
VALIDATION_VEHICLE_MODE="${VALIDATION_VEHICLE_MODE:-training_count}"
VALIDATION_STRESS_EVERY="${VALIDATION_STRESS_EVERY:-100}"
VALIDATION_STRESS_EPISODES="${VALIDATION_STRESS_EPISODES:-2}"
VALIDATION_STRESS_VEHICLE_MODE="${VALIDATION_STRESS_VEHICLE_MODE:-all}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"
VALIDATION_MIN_DELTA="${VALIDATION_MIN_DELTA:-0.0}"
VALIDATION_SCORE_HORIZON_SECONDS="${VALIDATION_SCORE_HORIZON_SECONDS:-20}"
VALIDATION_SCORE_POSITION_WEIGHT="${VALIDATION_SCORE_POSITION_WEIGHT:-1.0}"
VALIDATION_SCORE_SPEED_WEIGHT="${VALIDATION_SCORE_SPEED_WEIGHT:-0.5}"
VALIDATION_SCORE_LANE_OFFSET_WEIGHT="${VALIDATION_SCORE_LANE_OFFSET_WEIGHT:-2.0}"
VALIDATION_SCORE_CRASH_WEIGHT="${VALIDATION_SCORE_CRASH_WEIGHT:-25.0}"
VALIDATION_SCORE_OFFROAD_WEIGHT="${VALIDATION_SCORE_OFFROAD_WEIGHT:-25.0}"
VALIDATION_SCORE_HARD_BRAKE_WEIGHT="${VALIDATION_SCORE_HARD_BRAKE_WEIGHT:-2.0}"
TEST_EPISODES="${TEST_EPISODES:-4}"
TEST_PREBUILT_SPLIT="${TEST_PREBUILT_SPLIT:-test}"
EVALUATION_HORIZONS_SECONDS="${EVALUATION_HORIZONS_SECONDS:-1,5,10,20}"
HARD_BRAKE_ACCEL_THRESHOLD="${HARD_BRAKE_ACCEL_THRESHOLD:--3.0}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
PPO_EPOCHS="${PPO_EPOCHS:-6}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0015}"
ENTROPY_COEF_SCHEDULE="${ENTROPY_COEF_SCHEDULE:-}"
CLIP_RANGE="${CLIP_RANGE:-0.20}"
CLIP_RANGE_SCHEDULE="${CLIP_RANGE_SCHEDULE:-0:100:0.10:0.20;100:500:0.20:0.20;500:600:0.10:0.20;600:800:0.20:0.20}"
DISC_BATCH_SIZE="${DISC_BATCH_SIZE:-4096}"
POLICY_MODEL="${POLICY_MODEL:-recurrent_transformer}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
DISCRIMINATOR_HIDDEN_SIZES="${DISCRIMINATOR_HIDDEN_SIZES:-256,256,128}"
DISCRIMINATOR_DROPOUT="${DISCRIMINATOR_DROPOUT:-0.0}"
DISCRIMINATOR_SPECTRAL_NORM="${DISCRIMINATOR_SPECTRAL_NORM:-true}"
ENABLE_SEQUENCE_DISCRIMINATOR="${ENABLE_SEQUENCE_DISCRIMINATOR:-false}"
SEQUENCE_FEATURE_MODE="${SEQUENCE_FEATURE_MODE:-local_deltas}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-1}"
SEQUENCE_REWARD_COEF="${SEQUENCE_REWARD_COEF:-1.0}"
SEQUENCE_REWARD_ASSIGNMENT="${SEQUENCE_REWARD_ASSIGNMENT:-mean}"
ENABLE_VENDI_DIAGNOSTICS="${ENABLE_VENDI_DIAGNOSTICS:-false}"
VENDI_MAX_WINDOWS="${VENDI_MAX_WINDOWS:-2048}"
VENDI_RBF_SIGMA="${VENDI_RBF_SIGMA:-0.0}"
VENDI_SAFE_ONLY="${VENDI_SAFE_ONLY:-true}"
VENDI_SEED="${VENDI_SEED:-0}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-3}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.1}"
TRANSFORMER_MEMORY_TOKENS="${TRANSFORMER_MEMORY_TOKENS:-8}"
TRANSFORMER_MEMORY_CONTEXT_LENGTH="${TRANSFORMER_MEMORY_CONTEXT_LENGTH:-32}"
TRANSFORMER_RECURRENT_SEQUENCE_LENGTH="${TRANSFORMER_RECURRENT_SEQUENCE_LENGTH:-32}"
TRANSFORMER_RECURRENT_SEQUENCES_PER_BATCH="${TRANSFORMER_RECURRENT_SEQUENCES_PER_BATCH:-32}"
TRANSFORMER_RECURRENT_MICRO_BATCH_SEQUENCES="${TRANSFORMER_RECURRENT_MICRO_BATCH_SEQUENCES:-8}"
TRANSFORMER_MEMORY_STORAGE_DTYPE="${TRANSFORMER_MEMORY_STORAGE_DTYPE:-float16}"
TRANSFORMER_USE_CAUSAL_ATTENTION="${TRANSFORMER_USE_CAUSAL_ATTENTION:-true}"
CENTRALIZED_CRITIC="${CENTRALIZED_CRITIC:-true}"
CENTRAL_CRITIC_MAX_VEHICLES="${CENTRAL_CRITIC_MAX_VEHICLES:-128}"
CENTRAL_CRITIC_INCLUDE_LOCAL_OBS="${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS:-false}"
CENTRAL_CRITIC_POOLING="${CENTRAL_CRITIC_POOLING:-attention}"
CENTRAL_CRITIC_ATTENTION_HEADS="${CENTRAL_CRITIC_ATTENTION_HEADS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
TERMINATE_WHEN_ALL_CONTROLLED_CRASHED="${TERMINATE_WHEN_ALL_CONTROLLED_CRASHED:-true}"
export EXPERT_DATA
export SCENE
export EPISODE_ROOT
export PREBUILT_SPLIT
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
if [ "${SAVE_BEST_CHECKPOINT}" = "true" ]; then
    SAVE_BEST_CHECKPOINT_ARG="--save-best-checkpoint"
else
    SAVE_BEST_CHECKPOINT_ARG="--no-save-best-checkpoint"
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
if [ "${PSRO_LITE}" = "true" ]; then
    PSRO_LITE_ARG="--psro-lite"
else
    PSRO_LITE_ARG="--no-psro-lite"
fi
if [ "${DISCRIMINATOR_SPECTRAL_NORM}" = "true" ]; then
    DISCRIMINATOR_SPECTRAL_NORM_ARG="--discriminator-spectral-norm"
else
    DISCRIMINATOR_SPECTRAL_NORM_ARG="--no-discriminator-spectral-norm"
fi
if [ "${ENABLE_SEQUENCE_DISCRIMINATOR}" = "true" ]; then
    SEQUENCE_DISCRIMINATOR_ARG="--enable-sequence-discriminator"
else
    SEQUENCE_DISCRIMINATOR_ARG="--no-enable-sequence-discriminator"
fi
if [ "${ENABLE_VENDI_DIAGNOSTICS}" = "true" ]; then
    VENDI_DIAGNOSTICS_ARG="--enable-vendi-diagnostics"
else
    VENDI_DIAGNOSTICS_ARG="--no-enable-vendi-diagnostics"
fi
if [ "${VENDI_SAFE_ONLY}" = "true" ]; then
    VENDI_SAFE_ONLY_ARG="--vendi-safe-only"
else
    VENDI_SAFE_ONLY_ARG="--no-vendi-safe-only"
fi
if [ "${NORMALIZE_GAIL_REWARD}" = "true" ]; then
    NORMALIZE_GAIL_REWARD_ARG="--normalize-gail-reward"
else
    NORMALIZE_GAIL_REWARD_ARG="--no-normalize-gail-reward"
fi
if [ "${ALLOW_WGAN_REWARD_NORMALIZATION}" = "true" ]; then
    ALLOW_WGAN_REWARD_NORMALIZATION_ARG="--allow-wgan-reward-normalization"
else
    ALLOW_WGAN_REWARD_NORMALIZATION_ARG="--no-allow-wgan-reward-normalization"
fi
if [ "${WGAN_REWARD_CENTER}" = "true" ]; then
    WGAN_REWARD_CENTER_ARG="--wgan-reward-center"
else
    WGAN_REWARD_CENTER_ARG="--no-wgan-reward-center"
fi
if [ "${TRANSFORMER_USE_CAUSAL_ATTENTION}" = "true" ]; then
    TRANSFORMER_USE_CAUSAL_ATTENTION_ARG="--transformer-use-causal-attention"
else
    TRANSFORMER_USE_CAUSAL_ATTENTION_ARG="--no-transformer-use-causal-attention"
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
echo "Scene: ${SCENE}"
echo "Episode root: ${EPISODE_ROOT}"
echo "Prebuilt split: ${PREBUILT_SPLIT}"
echo "Max expert samples: ${MAX_EXPERT_SAMPLES} (0 means full dataset)"
echo "Total rounds: ${TOTAL_ROUNDS}"
echo "Rollout target-aware episodes: ${ROLLOUT_TARGET_AWARE_EPISODES} min=${ROLLOUT_TARGET_MIN_EPISODES} safety=${ROLLOUT_TARGET_EPISODE_SAFETY_FACTOR}"
echo "Rollout training subsample: ${ROLLOUT_TRAINING_SUBSAMPLE} cap=${ROLLOUT_TRAINING_AGENT_STEPS:-0} (0 follows rollout target)"
echo "Controlled-vehicle schedule endpoints: ${INITIAL_CONTROLLED_VEHICLES}->${FINAL_CONTROLLED_VEHICLES}/${CONTROLLED_VEHICLE_CURRICULUM_ROUNDS} rounds increment_rounds=${CONTROLLED_VEHICLE_INCREMENT_ROUNDS}"
echo "Controlled-vehicle schedule profile: ${CONTROLLED_VEHICLE_SCHEDULE_PROFILE}"
echo "Controlled-vehicle piecewise schedule: ${CONTROLLED_VEHICLE_SCHEDULE}"
echo "Warmup: rounds=${WARMUP_ROUNDS} policy_lr=${WARMUP_LEARNING_RATE}->default disc_lr=${WARMUP_DISC_LEARNING_RATE}->default entropy=${WARMUP_ENTROPY_COEF}->default clip=${WARMUP_CLIP_RANGE}->default"
echo "Policy LR schedule: ${LEARNING_RATE_SCHEDULE}"
echo "Discriminator LR schedule: ${DISC_LEARNING_RATE_SCHEDULE}"
echo "Entropy schedule: ${ENTROPY_COEF_SCHEDULE}"
echo "PPO clip schedule: ${CLIP_RANGE_SCHEDULE}"
echo "PPO epochs: ${PPO_EPOCHS}"
echo "Discriminator updates schedule: ${DISC_UPDATES_PER_ROUND_SCHEDULE}"
echo "Discriminator input: ${DISCRIMINATOR_INPUT}"
echo "Sequence discriminator: ${ENABLE_SEQUENCE_DISCRIMINATOR} mode=${SEQUENCE_FEATURE_MODE} length=${SEQUENCE_LENGTH} stride=${SEQUENCE_STRIDE} coef=${SEQUENCE_REWARD_COEF} assignment=${SEQUENCE_REWARD_ASSIGNMENT}"
echo "Entropy coef: ${ENTROPY_COEF}"
echo "Vehicle-increase warmup rounds: ${VEHICLE_INCREASE_WARMUP_ROUNDS}"
echo "Rollout target agent steps: base=${ROLLOUT_TARGET_AGENT_STEPS} curriculum=${INITIAL_ROLLOUT_TARGET_AGENT_STEPS}->${FINAL_ROLLOUT_TARGET_AGENT_STEPS}/${ROLLOUT_TARGET_AGENT_STEPS_CURRICULUM_ROUNDS} rounds"
echo "Rollout target agent steps schedule: ${ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE}"
echo "PSRO-lite: ${PSRO_LITE} archive_every=${PSRO_ARCHIVE_EVERY} archive_size=${PSRO_ARCHIVE_SIZE} after_jump_rounds=${PSRO_MIXTURE_AFTER_JUMP_ROUNDS} current_fraction=${PSRO_CURRENT_POLICY_FRACTION}"
echo "Gamma curriculum: ${INITIAL_GAMMA}->${FINAL_GAMMA}/${GAMMA_CURRICULUM_ROUNDS} rounds"
echo "Gamma schedule: ${GAMMA_SCHEDULE}"
echo "Max episode steps schedule: ${MAX_EPISODE_STEPS_SCHEDULE}"
echo "Policy model: ${POLICY_MODEL}"
echo "Centralized critic: ${CENTRALIZED_CRITIC} max_vehicles=${CENTRAL_CRITIC_MAX_VEHICLES} include_local_obs=${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS} pooling=${CENTRAL_CRITIC_POOLING} attention_heads=${CENTRAL_CRITIC_ATTENTION_HEADS}"
echo "Discriminator architecture: ${DISCRIMINATOR_HIDDEN_SIZES} dropout=${DISCRIMINATOR_DROPOUT}"
echo "Discriminator spectral norm: ${DISCRIMINATOR_SPECTRAL_NORM}"
echo "Discriminator loss: ${DISCRIMINATOR_LOSS}"
echo "Discriminator learning rate: ${DISC_LEARNING_RATE}"
echo "Discriminator updates per round: ${DISC_UPDATES_PER_ROUND}"
echo "Discriminator replay: rounds=${DISCRIMINATOR_REPLAY_ROUNDS} max_samples=${DISCRIMINATOR_REPLAY_MAX_SAMPLES}"
echo "Normalize GAIL reward: ${NORMALIZE_GAIL_REWARD}"
echo "Allow WGAN reward normalization: ${ALLOW_WGAN_REWARD_NORMALIZATION}"
echo "WGAN reward shaping: center=${WGAN_REWARD_CENTER} clip=${WGAN_REWARD_CLIP} scale=${WGAN_REWARD_SCALE} norm_min_std=${WGAN_REWARD_NORM_MIN_STD} norm_clip=${WGAN_REWARD_NORM_CLIP}"
echo "Transformer: layers=${TRANSFORMER_LAYERS} heads=${TRANSFORMER_HEADS} dropout=${TRANSFORMER_DROPOUT}"
echo "Recurrent transformer: memory_tokens=${TRANSFORMER_MEMORY_TOKENS} context=${TRANSFORMER_MEMORY_CONTEXT_LENGTH} sequence_length=${TRANSFORMER_RECURRENT_SEQUENCE_LENGTH} sequences_per_batch=${TRANSFORMER_RECURRENT_SEQUENCES_PER_BATCH} micro_sequences=${TRANSFORMER_RECURRENT_MICRO_BATCH_SEQUENCES} storage=${TRANSFORMER_MEMORY_STORAGE_DTYPE} causal=${TRANSFORMER_USE_CAUSAL_ATTENTION}"
echo "Checkpoint every: ${CHECKPOINT_EVERY}"
echo "Save checkpoint video: ${SAVE_CHECKPOINT_VIDEO}"
echo "Checkpoint video every: ${CHECKPOINT_VIDEO_EVERY}"
echo "Validation: every=${VALIDATION_EVERY} episodes=${VALIDATION_EPISODES} split=${VALIDATION_PREBUILT_SPLIT} mode=${VALIDATION_VEHICLE_MODE} horizons=${EVALUATION_HORIZONS_SECONDS}"
echo "Validation stress: every=${VALIDATION_STRESS_EVERY} episodes=${VALIDATION_STRESS_EPISODES} mode=${VALIDATION_STRESS_VEHICLE_MODE}"
echo "Best checkpoint: save=${SAVE_BEST_CHECKPOINT} min_delta=${VALIDATION_MIN_DELTA} score_horizon=${VALIDATION_SCORE_HORIZON_SECONDS}s weights=pos:${VALIDATION_SCORE_POSITION_WEIGHT},speed:${VALIDATION_SCORE_SPEED_WEIGHT},lane:${VALIDATION_SCORE_LANE_OFFSET_WEIGHT},crash:${VALIDATION_SCORE_CRASH_WEIGHT},offroad:${VALIDATION_SCORE_OFFROAD_WEIGHT},hard_brake:${VALIDATION_SCORE_HARD_BRAKE_WEIGHT}"
echo "Test: episodes=${TEST_EPISODES} split=${TEST_PREBUILT_SPLIT} hard_brake_threshold=${HARD_BRAKE_ACCEL_THRESHOLD}"
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
    "scene": os.environ["SCENE"],
    "episode_root": os.environ["EPISODE_ROOT"],
    "prebuilt_split": os.environ["PREBUILT_SPLIT"],
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
    --scene "${SCENE}" \
    --action-mode "${ACTION_MODE}" \
    --discriminator-input "${DISCRIMINATOR_INPUT}" \
    --episode-root "${EPISODE_ROOT}" \
    --prebuilt-split "${PREBUILT_SPLIT}" \
    --validation-every "${VALIDATION_EVERY}" \
    --validation-episodes "${VALIDATION_EPISODES}" \
    --validation-prebuilt-split "${VALIDATION_PREBUILT_SPLIT}" \
    --validation-vehicle-mode "${VALIDATION_VEHICLE_MODE}" \
    --validation-stress-every "${VALIDATION_STRESS_EVERY}" \
    --validation-stress-episodes "${VALIDATION_STRESS_EPISODES}" \
    --validation-stress-vehicle-mode "${VALIDATION_STRESS_VEHICLE_MODE}" \
    "${SAVE_BEST_CHECKPOINT_ARG}" \
    --validation-min-delta "${VALIDATION_MIN_DELTA}" \
    --validation-score-horizon-seconds "${VALIDATION_SCORE_HORIZON_SECONDS}" \
    --validation-score-position-weight "${VALIDATION_SCORE_POSITION_WEIGHT}" \
    --validation-score-speed-weight "${VALIDATION_SCORE_SPEED_WEIGHT}" \
    --validation-score-lane-offset-weight "${VALIDATION_SCORE_LANE_OFFSET_WEIGHT}" \
    --validation-score-crash-weight "${VALIDATION_SCORE_CRASH_WEIGHT}" \
    --validation-score-offroad-weight "${VALIDATION_SCORE_OFFROAD_WEIGHT}" \
    --validation-score-hard-brake-weight "${VALIDATION_SCORE_HARD_BRAKE_WEIGHT}" \
    --test-episodes "${TEST_EPISODES}" \
    --test-prebuilt-split "${TEST_PREBUILT_SPLIT}" \
    --evaluation-horizons-seconds "${EVALUATION_HORIZONS_SECONDS}" \
    --hard-brake-accel-threshold "${HARD_BRAKE_ACCEL_THRESHOLD}" \
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
    "${NORMALIZE_GAIL_REWARD_ARG}" \
    "${ALLOW_WGAN_REWARD_NORMALIZATION_ARG}" \
    "${WGAN_REWARD_CENTER_ARG}" \
    --wgan-reward-clip "${WGAN_REWARD_CLIP}" \
    --wgan-reward-scale "${WGAN_REWARD_SCALE}" \
    --wgan-reward-norm-min-std "${WGAN_REWARD_NORM_MIN_STD}" \
    --wgan-reward-norm-clip "${WGAN_REWARD_NORM_CLIP}" \
    --gail-reward-clip "${GAIL_REWARD_CLIP}" \
    --collision-penalty "${COLLISION_PENALTY}" \
    --offroad-penalty "${OFFROAD_PENALTY}" \
    --final-reward-clip "${FINAL_REWARD_CLIP}" \
    --disc-expert-label "${DISC_EXPERT_LABEL}" \
    --disc-generator-label "${DISC_GENERATOR_LABEL}" \
    --discriminator-loss "${DISCRIMINATOR_LOSS}" \
    --disc-learning-rate "${DISC_LEARNING_RATE}" \
    --disc-learning-rate-schedule "${DISC_LEARNING_RATE_SCHEDULE}" \
    --discriminator-replay-rounds "${DISCRIMINATOR_REPLAY_ROUNDS}" \
    --discriminator-replay-max-samples "${DISCRIMINATOR_REPLAY_MAX_SAMPLES}" \
    --rollout-target-agent-steps "${ROLLOUT_TARGET_AGENT_STEPS}" \
    --initial-rollout-target-agent-steps "${INITIAL_ROLLOUT_TARGET_AGENT_STEPS}" \
    --final-rollout-target-agent-steps "${FINAL_ROLLOUT_TARGET_AGENT_STEPS}" \
    --rollout-target-agent-steps-curriculum-rounds "${ROLLOUT_TARGET_AGENT_STEPS_CURRICULUM_ROUNDS}" \
    --rollout-target-agent-steps-schedule "${ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE}" \
    "${PSRO_LITE_ARG}" \
    --psro-archive-every "${PSRO_ARCHIVE_EVERY}" \
    --psro-archive-size "${PSRO_ARCHIVE_SIZE}" \
    --psro-mixture-after-jump-rounds "${PSRO_MIXTURE_AFTER_JUMP_ROUNDS}" \
    --psro-current-policy-fraction "${PSRO_CURRENT_POLICY_FRACTION}" \
    --initial-gamma "${INITIAL_GAMMA}" \
    --final-gamma "${FINAL_GAMMA}" \
    --gamma-curriculum-rounds "${GAMMA_CURRICULUM_ROUNDS}" \
    --gamma-schedule "${GAMMA_SCHEDULE}" \
    "${TERMINATION_ARG}" \
    --allow-idm \
    --device cuda \
    --learning-rate "${LEARNING_RATE}" \
    --learning-rate-schedule "${LEARNING_RATE_SCHEDULE}" \
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
    --evaluation-num-workers "${EVALUATION_WORKERS}" \
    --evaluation-worker-threads "${EVALUATION_WORKER_THREADS}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    --policy-model "${POLICY_MODEL}" \
    --hidden-size "${HIDDEN_SIZE}" \
    "${CENTRALIZED_CRITIC_ARG}" \
    --central-critic-max-vehicles "${CENTRAL_CRITIC_MAX_VEHICLES}" \
    "${CENTRAL_CRITIC_INCLUDE_LOCAL_OBS_ARG}" \
    --central-critic-pooling "${CENTRAL_CRITIC_POOLING}" \
    --central-critic-attention-heads "${CENTRAL_CRITIC_ATTENTION_HEADS}" \
    --discriminator-hidden-sizes "${DISCRIMINATOR_HIDDEN_SIZES}" \
    --discriminator-dropout "${DISCRIMINATOR_DROPOUT}" \
    "${DISCRIMINATOR_SPECTRAL_NORM_ARG}" \
    "${SEQUENCE_DISCRIMINATOR_ARG}" \
    --sequence-feature-mode "${SEQUENCE_FEATURE_MODE}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --sequence-stride "${SEQUENCE_STRIDE}" \
    --sequence-reward-coef "${SEQUENCE_REWARD_COEF}" \
    --sequence-reward-assignment "${SEQUENCE_REWARD_ASSIGNMENT}" \
    "${VENDI_DIAGNOSTICS_ARG}" \
    --vendi-max-windows "${VENDI_MAX_WINDOWS}" \
    --vendi-rbf-sigma "${VENDI_RBF_SIGMA}" \
    "${VENDI_SAFE_ONLY_ARG}" \
    --vendi-seed "${VENDI_SEED}" \
    --transformer-layers "${TRANSFORMER_LAYERS}" \
    --transformer-heads "${TRANSFORMER_HEADS}" \
    --transformer-dropout "${TRANSFORMER_DROPOUT}" \
    --transformer-memory-tokens "${TRANSFORMER_MEMORY_TOKENS}" \
    --transformer-memory-context-length "${TRANSFORMER_MEMORY_CONTEXT_LENGTH}" \
    --transformer-recurrent-sequence-length "${TRANSFORMER_RECURRENT_SEQUENCE_LENGTH}" \
    --transformer-recurrent-sequences-per-batch "${TRANSFORMER_RECURRENT_SEQUENCES_PER_BATCH}" \
    --transformer-recurrent-micro-batch-sequences "${TRANSFORMER_RECURRENT_MICRO_BATCH_SEQUENCES}" \
    --transformer-memory-storage-dtype "${TRANSFORMER_MEMORY_STORAGE_DTYPE}" \
    "${TRANSFORMER_USE_CAUSAL_ATTENTION_ARG}" \
    --batch-size "${BATCH_SIZE}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --entropy-coef "${ENTROPY_COEF}" \
    --entropy-coef-schedule "${ENTROPY_COEF_SCHEDULE}" \
    --clip-range "${CLIP_RANGE}" \
    --clip-range-schedule "${CLIP_RANGE_SCHEDULE}" \
    --disc-batch-size "${DISC_BATCH_SIZE}" \
    --disc-updates-per-round "${DISC_UPDATES_PER_ROUND}" \
    --disc-updates-per-round-schedule "${DISC_UPDATES_PER_ROUND_SCHEDULE}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${CHECKPOINT_VIDEO_ARG}" \
    --checkpoint-video-every "${CHECKPOINT_VIDEO_EVERY}" \
    --checkpoint-video-steps "${CHECKPOINT_VIDEO_STEPS}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project highwayenv-ps-gail \
    --wandb-tags ps-gail,continuous,collision,idm,gpu,32c,vehicle-${CONTROLLED_VEHICLE_SCHEDULE_PROFILE},psro-lite-${PSRO_LITE} \
    --run-name "${RUN_NAME}"
