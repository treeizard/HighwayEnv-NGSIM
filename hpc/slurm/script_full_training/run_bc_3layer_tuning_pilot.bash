#!/bin/bash
#SBATCH --job-name=bc_3layer_tune
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/bc_3layer_tune_%j.out
#SBATCH --error=logs/slurm/bc_3layer_tune_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_bc_3layer_tuning_${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the three-layer BC tuning environment")
print(f"Three-layer tuning device: {torch.cuda.get_device_name(0)}")
PY

US_EXPERT="${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982"
JAPANESE_EXPERT="${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train"
test -s "${US_EXPERT}/manifest.json"
test -s "${JAPANESE_EXPERT}/manifest.json"

TUNING_RUN_ID="${BC_3L_TUNING_RUN_ID:-${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}}"
TUNING_ROOT="${BC_3L_TUNING_ROOT:-${VFI_RESULTS_ROOT}/runs/policies/bc/three_layer_tuning_${TUNING_RUN_ID}}"
SCREEN_ROOT="${TUNING_ROOT}/screen"
CONFIRMATION_ROOT="${TUNING_ROOT}/confirmation"
mkdir -p "${SCREEN_ROOT}" "${CONFIRMATION_ROOT}"

CANDIDATE_IDS=(
    lr3em4_clip1_drop01
    lr3em4_clip5_drop01
    lr3em4_clip1_drop0
    lr1em3_clip1_drop0
    lr1em4_clip1_drop0
)
CANDIDATE_LRS=(0.0003 0.0003 0.0003 0.001 0.0001)
CANDIDATE_CLIPS=(1.0 5.0 1.0 1.0 1.0)
CANDIDATE_DROPOUTS=(0.1 0.1 0.0 0.0 0.0)
CANDIDATE_LIMIT="${BC_3L_TUNING_CANDIDATE_LIMIT:-${#CANDIDATE_IDS[@]}}"
if [ "${CANDIDATE_LIMIT}" -lt 1 ] || [ "${CANDIDATE_LIMIT}" -gt "${#CANDIDATE_IDS[@]}" ]; then
    echo "BC_3L_TUNING_CANDIDATE_LIMIT must be between 1 and ${#CANDIDATE_IDS[@]}; got ${CANDIDATE_LIMIT}" >&2
    exit 4
fi
ACTIVE_CANDIDATE_IDS=("${CANDIDATE_IDS[@]:0:${CANDIDATE_LIMIT}}")

train_trial() {
    local out_dir="$1"
    local domain="$2"
    local seed="$3"
    local learning_rate="$4"
    local max_grad_norm="$5"
    local dropout="$6"
    local epochs="$7"
    local scene
    local expert_data

    if [ "${domain}" = "us" ]; then
        scene="us-101"
        expert_data="${US_EXPERT}"
    elif [ "${domain}" = "japanese" ]; then
        scene="japanese"
        expert_data="${JAPANESE_EXPERT}"
    else
        echo "Unsupported tuning domain: ${domain}" >&2
        return 2
    fi
    if [ -e "${out_dir}/best.pt" ]; then
        echo "Refusing to overwrite a tuning checkpoint: ${out_dir}/best.pt" >&2
        return 3
    fi

    cd "${REPODIR}"
    python -m scripts_gail.train_recurrent_bc_policy \
        --expert-data "${expert_data}" \
        --out-dir "${out_dir}" \
        --domain "${domain}" \
        --scene "${scene}" \
        --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
        --prebuilt-split train \
        --seed "${seed}" \
        --data-seed "${DATA_SEED:-20260716}" \
        --split-seed "${SPLIT_SEED:-20260716}" \
        --max-expert-samples "${MAX_EXPERT_SAMPLES:-300000}" \
        --epochs "${epochs}" \
        --learning-rate "${learning_rate}" \
        --weight-decay "${BC_WEIGHT_DECAY:-0.00001}" \
        --max-grad-norm "${max_grad_norm}" \
        --early-stopping-patience 0 \
        --min-validation-skill "${MIN_VALIDATION_SKILL:-0.10}" \
        --max-validation-mae "${MAX_VALIDATION_MAE:-0.35}" \
        --learning-action-index 0 \
        --min-learning-action-std-ratio "${MIN_LEARNING_ACTION_STD_RATIO:-0.25}" \
        --min-learning-action-correlation "${MIN_LEARNING_ACTION_CORRELATION:-0.50}" \
        --hidden-size 256 \
        --transformer-layers 3 \
        --transformer-heads 4 \
        --transformer-dropout "${dropout}" \
        --memory-tokens 8 \
        --memory-context-length 32 \
        --sequence-length 32 \
        --sequences-per-batch 16 \
        --micro-batch-sequences 16 \
        --evaluation-episodes 1 \
        --evaluation-split test \
        --no-evaluation-enable-collision \
        --min-rollout-steps 0 \
        --max-crash-fraction 1 \
        --max-offroad-fraction 1 \
        --no-render-video \
        --capability-failure-mode report \
        --device cuda
}

for index in "${!ACTIVE_CANDIDATE_IDS[@]}"; do
    candidate_id="${CANDIDATE_IDS[$index]}"
    echo "Screening three-layer candidate=${candidate_id} domain=japanese seed=0"
    train_trial \
        "${SCREEN_ROOT}/${candidate_id}" \
        japanese \
        0 \
        "${CANDIDATE_LRS[$index]}" \
        "${CANDIDATE_CLIPS[$index]}" \
        "${CANDIDATE_DROPOUTS[$index]}" \
        "${SCREEN_EPOCHS:-30}"
done

selection_args=()
for candidate_id in "${ACTIVE_CANDIDATE_IDS[@]}"; do
    selection_args+=(--candidate "${candidate_id}")
done
selected_candidate="$(
    cd "${REPODIR}"
    python -m scripts_gail.select_bc_3layer_tuning \
        --screen-root "${SCREEN_ROOT}" \
        "${selection_args[@]}" \
        --out "${TUNING_ROOT}/screen_selection.json" \
        --print-selected
)"

if [ "${selected_candidate}" = "NONE" ]; then
    echo "No three-layer candidate passed the learning gates; no checkpoint was promoted."
    cp "${TUNING_ROOT}/screen_selection.json" "${TUNING_ROOT}/tuning_manifest.json"
    exit 0
fi

if [ "${BC_3L_TUNING_SCREEN_ONLY:-0}" = "1" ]; then
    echo "Screen-only three-layer tuning run completed; confirmation was intentionally skipped."
    cp "${TUNING_ROOT}/screen_selection.json" "${TUNING_ROOT}/tuning_manifest.json"
    exit 0
fi

selected_summary="${SCREEN_ROOT}/${selected_candidate}/summary.json"
selected_learning_rate="$(jq -r '.learning_rate' "${selected_summary}")"
selected_max_grad_norm="$(jq -r '.max_grad_norm' "${selected_summary}")"
selected_dropout="$(jq -r '.transformer_dropout' "${selected_summary}")"
echo "Selected candidate=${selected_candidate}; confirming across domains and previously collapsed seeds."

train_trial \
    "${CONFIRMATION_ROOT}/us_seed_0" \
    us \
    0 \
    "${selected_learning_rate}" \
    "${selected_max_grad_norm}" \
    "${selected_dropout}" \
    "${CONFIRM_EPOCHS:-50}"
train_trial \
    "${CONFIRMATION_ROOT}/japanese_seed_1" \
    japanese \
    1 \
    "${selected_learning_rate}" \
    "${selected_max_grad_norm}" \
    "${selected_dropout}" \
    "${CONFIRM_EPOCHS:-50}"

cd "${REPODIR}"
python -m scripts_gail.select_bc_3layer_tuning \
    --screen-root "${SCREEN_ROOT}" \
    "${selection_args[@]}" \
    --confirmation-root "${CONFIRMATION_ROOT}" \
    --confirmation us_seed_0 \
    --confirmation japanese_seed_1 \
    --out "${TUNING_ROOT}/tuning_manifest.json"

echo "Completed three-layer BC tuning pilot: ${TUNING_ROOT}/tuning_manifest.json"
