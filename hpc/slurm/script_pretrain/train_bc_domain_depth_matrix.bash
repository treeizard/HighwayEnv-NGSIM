#!/bin/bash
#SBATCH --job-name=bc_domain_depth
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --array=0-11%3
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/bc_domain_depth_%A_%a.out
#SBATCH --error=logs/slurm/bc_domain_depth_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

DOMAINS=(us japanese)
LAYERS=(2 3)
SEEDS=(0 1 2)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge 12 ]; then
    echo "Invalid BC model array task: ${TASK_ID}" >&2
    exit 2
fi
DOMAIN_INDEX=$((TASK_ID / 6))
WITHIN_DOMAIN=$((TASK_ID % 6))
LAYER_INDEX=$((WITHIN_DOMAIN / 3))
SEED_INDEX=$((WITHIN_DOMAIN % 3))
DOMAIN="${DOMAINS[${DOMAIN_INDEX}]}"
TRANSFORMER_LAYERS="${LAYERS[${LAYER_INDEX}]}"
POLICY_SEED="${SEEDS[${SEED_INDEX}]}"

if [ "${DOMAIN}" = "us" ]; then
    SCENE="us-101"
    EXPERT_DATA="${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982"
else
    SCENE="japanese"
    EXPERT_DATA="${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train"
fi

STUDY_ID="${BC_STUDY_ID:?BC_STUDY_ID must be exported by the submission script}"
POLICY_ROOT="${BC_STUDY_POLICY_ROOT:-${VFI_RESULTS_ROOT}/runs/policies/bc/${STUDY_ID}}"
ACTIVATION_ROOT="${BC_STUDY_ACTIVATION_ROOT:-${VFI_RESULTS_ROOT}/test_runs/bc_interpretability_smoke/${STUDY_ID}}"
RELATIVE="${DOMAIN}/recurrent_transformer_${TRANSFORMER_LAYERS}layer/policy_seed_${POLICY_SEED}"
RUN_DIR="${POLICY_ROOT}/${RELATIVE}"
SMOKE_DIR="${ACTIVATION_ROOT}/${RELATIVE}"
if [ "${TRANSFORMER_LAYERS}" -eq 2 ]; then
    CAPTURE_LAYERS="0,1"
else
    CAPTURE_LAYERS="0,1,2"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPODIR}"
mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}" "$(dirname "${RUN_DIR}")" "${SMOKE_DIR}"

if [ -e "${RUN_DIR}/summary.json" ] || [ -e "${RUN_DIR}/best.pt" ]; then
    echo "Refusing to overwrite an existing full BC run: ${RUN_DIR}" >&2
    exit 3
fi

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

echo "BC matrix task=${TASK_ID} domain=${DOMAIN} scene=${SCENE} layers=${TRANSFORMER_LAYERS} seed=${POLICY_SEED}"
echo "Expert data: ${EXPERT_DATA}"
echo "Run directory: ${RUN_DIR}"
nvidia-smi

python -m scripts_gail.train_recurrent_bc_policy \
    --expert-data "${EXPERT_DATA}" \
    --out-dir "${RUN_DIR}" \
    --domain "${DOMAIN}" \
    --scene "${SCENE}" \
    --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
    --prebuilt-split train \
    --seed "${POLICY_SEED}" \
    --data-seed "${DATA_SEED:-20260716}" \
    --split-seed "${SPLIT_SEED:-20260716}" \
    --max-expert-samples "${MAX_EXPERT_SAMPLES:-300000}" \
    --epochs "${BC_EPOCHS:-50}" \
    --learning-rate "${BC_LEARNING_RATE:-0.0003}" \
    --weight-decay "${BC_WEIGHT_DECAY:-0.00001}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-10}" \
    --min-validation-skill "${MIN_VALIDATION_SKILL:-0.05}" \
    --max-validation-mae "${MAX_VALIDATION_MAE:-0.35}" \
    --hidden-size "${HIDDEN_SIZE:-256}" \
    --transformer-layers "${TRANSFORMER_LAYERS}" \
    --transformer-heads "${TRANSFORMER_HEADS:-4}" \
    --transformer-dropout "${TRANSFORMER_DROPOUT:-0.1}" \
    --memory-tokens "${MEMORY_TOKENS:-8}" \
    --memory-context-length "${MEMORY_CONTEXT_LENGTH:-32}" \
    --sequence-length "${SEQUENCE_LENGTH:-32}" \
    --sequences-per-batch "${SEQUENCES_PER_BATCH:-16}" \
    --micro-batch-sequences "${MICRO_BATCH_SEQUENCES:-16}" \
    --no-render-video \
    --evaluation-episodes "${EVALUATION_EPISODES:-3}" \
    --min-rollout-steps "${MIN_ROLLOUT_STEPS:-100}" \
    --max-crash-fraction "${MAX_CRASH_FRACTION:-0.34}" \
    --max-offroad-fraction "${MAX_OFFROAD_FRACTION:-0.34}" \
    --capability-failure-mode report \
    --device cuda

CHECKPOINT="${RUN_DIR}/best.pt"
test -s "${CHECKPOINT}"
test -s "${CHECKPOINT}.sha256"
cd "${VFI_PROJECT_ROOT}"
python -m interpretability.sae.cli.inspect_checkpoint --checkpoint "${CHECKPOINT}" --device cpu
python -m interpretability.sae.cli.collect_activations \
    --checkpoint "${CHECKPOINT}" \
    --expert-data "${EXPERT_DATA}" \
    --out "${SMOKE_DIR}" \
    --signal-target residual_policy_tokens \
    --layers "${CAPTURE_LAYERS}" \
    --split all \
    --max-transitions 64 \
    --max-files 2 \
    --max-vehicles-per-file 1 \
    --shard-size 64 \
    --device cuda

for layer in ${CAPTURE_LAYERS//,/ }; do
    test -f "${SMOKE_DIR}/residual_layer_${layer}_policy_token/manifest.json"
done
echo "Interpretability smoke passed for ${CHECKPOINT}"
