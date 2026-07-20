#!/bin/bash

set -euo pipefail

MODE="${1:-smoke}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXPERT_DATA="${IQ_EXPERT_DATA:-${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982}"
EPISODE_ROOT="${IQ_EPISODE_ROOT:-${VFI_HIGHWAY_DATA_ROOT}/processed_20s}"
if [ -z "${IQ_INITIAL_POLICY_CHECKPOINT:-}" ]; then
    echo "All recurrent IQ modes require IQ_INITIAL_POLICY_CHECKPOINT pointing to a metric-qualified matched BC checkpoint" >&2
    exit 2
fi
IQ_DOMAIN="${IQ_DOMAIN:-us}"
if [ "${IQ_DOMAIN}" = us ]; then IQ_SCENE="us-101"; else IQ_SCENE="japanese"; fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMMON_ARGS=(
    --expert-data "${EXPERT_DATA}"
    --domain "${IQ_DOMAIN}"
    --scene "${IQ_SCENE}"
    --episode-root "${EPISODE_ROOT}"
    --initial-policy-checkpoint "${IQ_INITIAL_POLICY_CHECKPOINT}"
    --device cuda
    --transformer-layers "${IQ_TRANSFORMER_LAYERS:-2}"
    --transformer-heads "${IQ_TRANSFORMER_HEADS:-4}"
    --gamma "${IQ_GAMMA:-0.95}"
    --entropy-temperature "${IQ_ENTROPY_TEMPERATURE:-0.001}"
    --chi2-alpha "${IQ_CHI2_ALPHA:-0.5}"
    --target-tau "${IQ_TARGET_TAU:-0.005}"
    --max-q-abs "${IQ_MAX_Q_ABS:-50}"
    --target-q-clip "${IQ_TARGET_Q_CLIP:-20}"
    --transformer-dropout 0
    --memory-tokens 8
    --memory-context-length 32
    --training-context-length 8
    --sequence-length 8
    --validation-sequence-length 32
)

case "${MODE}" in
    smoke)
        OUT_DIR="${IQ_OUT_DIR:-${VFI_RESULTS_ROOT}/test_runs/recurrent_iq_smoke/launcher_${STAMP}}"
        MODE_ARGS=(
            --out-dir "${OUT_DIR}"
            --max-expert-samples 300000
            --hidden-size 256
            --sequences-per-update 2
            --micro-batch-sequences 2
            --updates 1
            --eval-every 1
            --q-only-updates 0
            --minimum-joint-updates 1
            --initial-policy-replay 32
            --collect-steps 32
            --collect-every 64
            --evaluation-episodes 1
            --min-rollout-steps 0
            --max-crash-fraction 1
            --max-offroad-fraction 1
            --max-collision-proxy-fraction 1
            --capability-failure-mode report
        )
        ;;
    stability)
        OUT_DIR="${IQ_OUT_DIR:-${VFI_RESULTS_ROOT}/test_runs/recurrent_iq_stability/run_${STAMP}}"
        MODE_ARGS=(
            --out-dir "${OUT_DIR}"
            --max-expert-samples 300000
            --hidden-size 256
            --sequences-per-update 4
            --micro-batch-sequences 16
            --updates 120
            --minimum-joint-updates 100
            --eval-every 40
            --policy-learning-rate "${IQ_POLICY_LEARNING_RATE:-0.000003}"
            --q-learning-rate "${IQ_Q_LEARNING_RATE:-0.00003}"
            --bc-coef "${IQ_BC_COEF:-30}"
            --q-only-updates "${IQ_Q_ONLY_UPDATES:-20}"
            --initial-policy-replay 512
            --collect-steps 256
            --evaluation-episodes 3
            --capability-failure-mode report
        )
        ;;
    production)
        OUT_DIR="${IQ_OUT_DIR:-${VFI_RESULTS_ROOT}/runs/policies/iq/recurrent_transformer_${STAMP}}"
        MODE_ARGS=(
            --out-dir "${OUT_DIR}"
            --max-expert-samples "${IQ_MAX_EXPERT_SAMPLES:-300000}"
            --hidden-size "${IQ_HIDDEN_SIZE:-256}"
            --sequences-per-update "${IQ_SEQUENCES_PER_UPDATE:-4}"
            --micro-batch-sequences "${IQ_MICRO_BATCH_SEQUENCES:-16}"
            --updates "${IQ_UPDATES:-2000}"
            --minimum-joint-updates "${IQ_MINIMUM_JOINT_UPDATES:-800}"
            --eval-every "${IQ_EVAL_EVERY:-200}"
            --policy-learning-rate "${IQ_POLICY_LEARNING_RATE:-0.000003}"
            --q-learning-rate "${IQ_Q_LEARNING_RATE:-0.00003}"
            --bc-coef "${IQ_BC_COEF:-30}"
            --q-only-updates "${IQ_Q_ONLY_UPDATES:-200}"
            --initial-policy-replay "${IQ_INITIAL_POLICY_REPLAY:-1024}"
            --collect-steps "${IQ_COLLECT_STEPS:-512}"
            --evaluation-episodes "${IQ_EVALUATION_EPISODES:-3}"
        )
        ;;
    *)
        echo "Usage: $0 {smoke|stability|production}" >&2
        exit 2
        ;;
esac

cd "${REPODIR}"
nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1s/^/IQ-Learn CUDA device: /p'
python -m scripts_gail.train_recurrent_iq_learn "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}"
echo "Recurrent IQ-Learn ${MODE} run complete: ${OUT_DIR}"
echo "TensorBoard: tensorboard --logdir ${OUT_DIR}/tensorboard --port ${IQ_TENSORBOARD_PORT:-6007}"
