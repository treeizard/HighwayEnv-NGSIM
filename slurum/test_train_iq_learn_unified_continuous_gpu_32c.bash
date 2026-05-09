#!/bin/bash
#SBATCH --job-name=iqlearn_full_conv
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/iqlearn_full_conv_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/iqlearn_full_conv_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"

# IQ-Learn can train mostly from expert transitions, so it needs far fewer CPU
# rollout workers than PS-GAIL/AIRL for a first test. Keep one GPU for batched
# critic/policy updates.
WORKER_THREADS="${WORKER_THREADS:-2}"
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-32}"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
RUN_NAME="${RUN_NAME:-iqlearn_unified_continuous_full_conv_${SLURM_JOB_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"

TOTAL_UPDATES="${TOTAL_UPDATES:-150000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-6}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
MAX_EXPERT_SAMPLES="${MAX_EXPERT_SAMPLES:-200000}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
REPLAY_SIZE="${REPLAY_SIZE:-200000}"
REPLAY_DEVICE="${REPLAY_DEVICE:-cuda}"
PIN_CPU_REPLAY="${PIN_CPU_REPLAY:-true}"
TORCH_MATMUL_PRECISION="${TORCH_MATMUL_PRECISION:-high}"
LEARNING_RATE="${LEARNING_RATE:-7.5e-5}"
DISC_LEARNING_RATE="${DISC_LEARNING_RATE:-7.5e-5}"
GAMMA="${GAMMA:-0.95}"
IQ_ALPHA="${IQ_ALPHA:-0.05}"
TARGET_TAU="${TARGET_TAU:-0.002}"
BC_COEF="${BC_COEF:-0.35}"
BC_WARMUP_COEF="${BC_WARMUP_COEF:-2.0}"
BC_WARMUP_UPDATES="${BC_WARMUP_UPDATES:-10000}"
POLICY_BC_ONLY_UPDATES="${POLICY_BC_ONLY_UPDATES:-2500}"
Q_L2_COEF="${Q_L2_COEF:-0.002}"
Q_POLICY_REG_COEF="${Q_POLICY_REG_COEF:-0.002}"
CONSERVATIVE_Q_COEF="${CONSERVATIVE_Q_COEF:-0.08}"
TARGET_VALUE_CLIP="${TARGET_VALUE_CLIP:-20.0}"
POLICY_Q_CLIP="${POLICY_Q_CLIP:-20.0}"
MAX_Q_ABS="${MAX_Q_ABS:-80.0}"
LOG_STD_MIN="${LOG_STD_MIN:--5.0}"
LOG_STD_MAX="${LOG_STD_MAX:-0.5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
SAVE_CHECKPOINT_VIDEO="${SAVE_CHECKPOINT_VIDEO:-true}"
CHECKPOINT_VIDEO_STEPS="${CHECKPOINT_VIDEO_STEPS:-200}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"
ABORT_ON_NONFINITE="${ABORT_ON_NONFINITE:-true}"
ABORT_ON_STALLED_CONVERGENCE="${ABORT_ON_STALLED_CONVERGENCE:-false}"
CONVERGENCE_PATIENCE_EVALS="${CONVERGENCE_PATIENCE_EVALS:-12}"
MIN_EVALS_BEFORE_STALL="${MIN_EVALS_BEFORE_STALL:-6}"
CONVERGENCE_MIN_DELTA="${CONVERGENCE_MIN_DELTA:-1e-3}"
CONVERGENCE_CRASH_PENALTY="${CONVERGENCE_CRASH_PENALTY:-25.0}"
BC_SCORE_WEIGHT="${BC_SCORE_WEIGHT:-1.0}"
TARGET_EVAL_MEAN_LENGTH="${TARGET_EVAL_MEAN_LENGTH:-195.0}"
TARGET_BC_LOSS="${TARGET_BC_LOSS:-0.02}"
MAX_EVAL_CRASHES="${MAX_EVAL_CRASHES:-0.0}"
export WANDB_MODE
export SAVE_CHECKPOINT_VIDEO
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
if [ "${ABORT_ON_NONFINITE}" = "true" ]; then
    ABORT_ON_NONFINITE_ARG="--abort-on-nonfinite"
else
    ABORT_ON_NONFINITE_ARG="--no-abort-on-nonfinite"
fi
if [ "${ABORT_ON_STALLED_CONVERGENCE}" = "true" ]; then
    ABORT_ON_STALLED_CONVERGENCE_ARG="--abort-on-stalled-convergence"
else
    ABORT_ON_STALLED_CONVERGENCE_ARG="--no-abort-on-stalled-convergence"
fi
if [ "${PIN_CPU_REPLAY}" = "true" ]; then
    PIN_CPU_REPLAY_ARG="--pin-cpu-replay"
else
    PIN_CPU_REPLAY_ARG="--no-pin-cpu-replay"
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Expert data: ${EXPERT_DATA}"
echo "IQ-Learn trainer: ${IQ_TRAIN_SCRIPT}"
echo "Total updates: ${TOTAL_UPDATES}"
echo "Eval every: ${EVAL_EVERY}"
echo "Eval episodes: ${EVAL_EPISODES}"
echo "CPUs per task: ${SLURM_CPUS}"
echo "Eval rollout workers: ${ROLLOUT_WORKERS}"
echo "Worker threads: ${WORKER_THREADS}"
echo "GPU throughput: batch_size=${BATCH_SIZE} replay_device=${REPLAY_DEVICE} matmul_precision=${TORCH_MATMUL_PRECISION}"
echo "Learning rates: policy=${LEARNING_RATE} q=${DISC_LEARNING_RATE}"
echo "IQ stability: gamma=${GAMMA} alpha=${IQ_ALPHA} tau=${TARGET_TAU} bc=${BC_COEF} warmup_bc=${BC_WARMUP_COEF}"
echo "Q stability: q_l2=${Q_L2_COEF} policy_q_reg=${Q_POLICY_REG_COEF} conservative=${CONSERVATIVE_Q_COEF} target_clip=${TARGET_VALUE_CLIP} policy_q_clip=${POLICY_Q_CLIP} max_q_abs=${MAX_Q_ABS}"
echo "Convergence target: eval_len>=${TARGET_EVAL_MEAN_LENGTH} bc_loss<=${TARGET_BC_LOSS} eval_crashes<=${MAX_EVAL_CRASHES}"
echo "Checkpointing: every=${CHECKPOINT_EVERY} save_video=${SAVE_CHECKPOINT_VIDEO} video_steps=${CHECKPOINT_VIDEO_STEPS} save_best=${SAVE_BEST_CHECKPOINT}"
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
    --eval-episodes "${EVAL_EPISODES}" \
    --replay-device "${REPLAY_DEVICE}" \
    "${PIN_CPU_REPLAY_ARG}" \
    --torch-matmul-precision "${TORCH_MATMUL_PRECISION}" \
    --learning-rate "${LEARNING_RATE}" \
    --disc-learning-rate "${DISC_LEARNING_RATE}" \
    --gamma "${GAMMA}" \
    --iq-alpha "${IQ_ALPHA}" \
    --target-tau "${TARGET_TAU}" \
    --bc-coef "${BC_COEF}" \
    --bc-warmup-coef "${BC_WARMUP_COEF}" \
    --bc-warmup-updates "${BC_WARMUP_UPDATES}" \
    --policy-bc-only-updates "${POLICY_BC_ONLY_UPDATES}" \
    --q-l2-coef "${Q_L2_COEF}" \
    --q-policy-reg-coef "${Q_POLICY_REG_COEF}" \
    --conservative-q-coef "${CONSERVATIVE_Q_COEF}" \
    --target-value-clip "${TARGET_VALUE_CLIP}" \
    --policy-q-clip "${POLICY_Q_CLIP}" \
    --max-q-abs "${MAX_Q_ABS}" \
    --log-std-min "${LOG_STD_MIN}" \
    --log-std-max "${LOG_STD_MAX}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --num-rollout-workers "${ROLLOUT_WORKERS}" \
    --rollout-worker-threads "${WORKER_THREADS}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --replay-size "${REPLAY_SIZE}" \
    "${SAVE_BEST_CHECKPOINT_ARG}" \
    "${ABORT_ON_NONFINITE_ARG}" \
    "${ABORT_ON_STALLED_CONVERGENCE_ARG}" \
    --convergence-patience-evals "${CONVERGENCE_PATIENCE_EVALS}" \
    --min-evals-before-stall "${MIN_EVALS_BEFORE_STALL}" \
    --convergence-min-delta "${CONVERGENCE_MIN_DELTA}" \
    --convergence-crash-penalty "${CONVERGENCE_CRASH_PENALTY}" \
    --bc-score-weight "${BC_SCORE_WEIGHT}" \
    --target-eval-mean-length "${TARGET_EVAL_MEAN_LENGTH}" \
    --target-bc-loss "${TARGET_BC_LOSS}" \
    --max-eval-crashes "${MAX_EVAL_CRASHES}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    "${CHECKPOINT_VIDEO_ARG}" \
    --checkpoint-video-steps "${CHECKPOINT_VIDEO_STEPS}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project highwayenv-ps-gail \
    --wandb-tags iq-learn,continuous,unified-expert,full-convergence,periodic-video,gpu,32c \
    --run-name "${RUN_NAME}"
