#!/bin/bash
#SBATCH --job-name=bc_domain_study
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/bc_domain_study_%j.out
#SBATCH --error=logs/slurm/bc_domain_study_%j.err

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
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_bc_study_${SLURM_JOB_ID:-local}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

US_EXPERT="${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982"
JAPANESE_EXPERT="${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train"
for expert_data in "${US_EXPERT}" "${JAPANESE_EXPERT}"; do
    if [ ! -f "${expert_data}/manifest.json" ]; then
        echo "Missing completed expert dataset: ${expert_data}/manifest.json" >&2
        exit 2
    fi
done

if ! python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the BC study environment")
print(f"CUDA smoke device: {torch.cuda.get_device_name(0)}")
PY
then
    echo "GPU prerequisite check failed" >&2
    exit 3
fi

run_activation_check() {
    local checkpoint="$1"
    local expert_data="$2"
    local output_dir="$3"
    local layers="$4"
    local max_transitions="$5"

    cd "${VFI_PROJECT_ROOT}"
    python -m interpretability.sae.cli.inspect_checkpoint \
        --checkpoint "${checkpoint}" \
        --device cpu
    python -m interpretability.sae.cli.collect_activations \
        --checkpoint "${checkpoint}" \
        --expert-data "${expert_data}" \
        --out "${output_dir}" \
        --signal-target residual_policy_tokens \
        --layers "${layers}" \
        --split all \
        --max-transitions "${max_transitions}" \
        --max-files 2 \
        --max-vehicles-per-file 1 \
        --shard-size "${max_transitions}" \
        --device cuda
    for layer in ${layers//,/ }; do
        test -s "${output_dir}/residual_layer_${layer}_policy_token/manifest.json"
    done
}

# Run a small but real train -> save -> reload -> activation-capture path first.
SMOKE_RUN_ID="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_ROOT="${BC_SMOKE_ROOT:-${VFI_RESULTS_ROOT}/test_runs/bc_pipeline_smoke/${SMOKE_RUN_ID}}"
SMOKE_POLICY_DIR="${SMOKE_ROOT}/policy"
SMOKE_ACTIVATION_DIR="${SMOKE_ROOT}/activations"
if [ -e "${SMOKE_POLICY_DIR}/best.pt" ]; then
    echo "Refusing to overwrite an existing smoke checkpoint: ${SMOKE_POLICY_DIR}/best.pt" >&2
    exit 4
fi

cd "${REPODIR}"
python -m scripts_gail.train_recurrent_bc_policy \
    --expert-data "${US_EXPERT}" \
    --out-dir "${SMOKE_POLICY_DIR}" \
    --domain us \
    --scene us-101 \
    --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
    --prebuilt-split train \
    --seed 0 \
    --data-seed 20260716 \
    --split-seed 20260716 \
    --max-expert-samples "${BC_SMOKE_MAX_EXPERT_SAMPLES:-4096}" \
    --epochs "${BC_SMOKE_EPOCHS:-1}" \
    --early-stopping-patience 0 \
    --min-validation-skill -1000000 \
    --max-validation-mae 1000000 \
    --hidden-size 32 \
    --transformer-layers 2 \
    --transformer-heads 4 \
    --transformer-dropout 0.0 \
    --memory-tokens 2 \
    --memory-context-length 8 \
    --sequence-length 8 \
    --sequences-per-batch 8 \
    --micro-batch-sequences 4 \
    --evaluation-episodes 1 \
    --evaluation-split test \
    --no-evaluation-enable-collision \
    --min-rollout-steps 1000000 \
    --max-crash-fraction 1 \
    --max-offroad-fraction 1 \
    --no-render-video \
    --capability-failure-mode report \
    --device cuda

test -s "${SMOKE_POLICY_DIR}/best.pt"
test -s "${SMOKE_POLICY_DIR}/best.pt.sha256"
SMOKE_SUMMARY="${SMOKE_POLICY_DIR}/summary.json" python - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["SMOKE_SUMMARY"]).read_text(encoding="utf-8"))
if not summary.get("metric_capability_passed"):
    raise SystemExit("Smoke model did not pass its permissive offline metric gate")
if not summary.get("checkpoint_saved"):
    raise SystemExit("Smoke checkpoint was not saved")
evaluation = summary.get("held_out_evaluation") or {}
if evaluation.get("prebuilt_split") != "test":
    raise SystemExit("Smoke evaluation did not use the test split")
if evaluation.get("collision_physics_enabled") is not False:
    raise SystemExit("Smoke evaluation did not disable collision physics")
metrics = evaluation.get("metrics") or {}
if "bc_eval/collision_episode_fraction" not in metrics:
    raise SystemExit("Smoke evaluation did not record collision-only rate")
if "bc_eval/collision_proxy_episode_fraction" not in metrics:
    raise SystemExit("Smoke evaluation did not record collision-proxy rate")
PY
run_activation_check "${SMOKE_POLICY_DIR}/best.pt" "${US_EXPERT}" "${SMOKE_ACTIVATION_DIR}" "0,1" 16
echo "End-to-end GPU smoke test passed: ${SMOKE_ROOT}"

if [ "${BC_STUDY_SMOKE_ONLY:-0}" = "1" ]; then
    exit 0
fi

STUDY_RUN_ID="${BC_STUDY_RUN_ID:-${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}}"
POLICY_ROOT="${BC_STUDY_POLICY_ROOT:-${VFI_RESULTS_ROOT}/runs/policies/bc/domain_depth_${STUDY_RUN_ID}}"
ACTIVATION_ROOT="${BC_STUDY_ACTIVATION_ROOT:-${VFI_RESULTS_ROOT}/test_runs/bc_interpretability_smoke/domain_depth_${STUDY_RUN_ID}}"
DOMAINS=(us japanese)
LAYERS=(2 3)
SEEDS=(0 1 2)
MODEL_LIMIT="${BC_STUDY_MODEL_LIMIT:-12}"
if [ "${MODEL_LIMIT}" -lt 1 ] || [ "${MODEL_LIMIT}" -gt 12 ]; then
    echo "BC_STUDY_MODEL_LIMIT must be between 1 and 12; got ${MODEL_LIMIT}" >&2
    exit 6
fi
completed_models=0

for domain in "${DOMAINS[@]}"; do
    if [ "${domain}" = "us" ]; then
        scene="us-101"
        expert_data="${US_EXPERT}"
    else
        scene="japanese"
        expert_data="${JAPANESE_EXPERT}"
    fi
    for transformer_layers in "${LAYERS[@]}"; do
        if [ "${transformer_layers}" -eq 2 ]; then
            capture_layers="0,1"
        else
            capture_layers="0,1,2"
        fi
        for policy_seed in "${SEEDS[@]}"; do
            relative="${domain}/recurrent_transformer_${transformer_layers}layer/policy_seed_${policy_seed}"
            run_dir="${POLICY_ROOT}/${relative}"
            activation_dir="${ACTIVATION_ROOT}/${relative}"
            if [ -e "${run_dir}/best.pt" ]; then
                echo "Refusing to overwrite an existing study checkpoint: ${run_dir}/best.pt" >&2
                exit 5
            fi

            echo "Training domain=${domain} layers=${transformer_layers} seed=${policy_seed}"
            cd "${REPODIR}"
            python -m scripts_gail.train_recurrent_bc_policy \
                --expert-data "${expert_data}" \
                --out-dir "${run_dir}" \
                --domain "${domain}" \
                --scene "${scene}" \
                --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
                --prebuilt-split train \
                --seed "${policy_seed}" \
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
                --transformer-layers "${transformer_layers}" \
                --transformer-heads "${TRANSFORMER_HEADS:-4}" \
                --transformer-dropout "${TRANSFORMER_DROPOUT:-0.1}" \
                --memory-tokens "${MEMORY_TOKENS:-8}" \
                --memory-context-length "${MEMORY_CONTEXT_LENGTH:-32}" \
                --sequence-length "${SEQUENCE_LENGTH:-32}" \
                --sequences-per-batch "${SEQUENCES_PER_BATCH:-16}" \
                --micro-batch-sequences "${MICRO_BATCH_SEQUENCES:-16}" \
                --evaluation-episodes "${EVALUATION_EPISODES:-3}" \
                --evaluation-split test \
                --no-evaluation-enable-collision \
                --min-rollout-steps "${MIN_ROLLOUT_STEPS:-100}" \
                --max-crash-fraction "${MAX_CRASH_FRACTION:-0.34}" \
                --max-offroad-fraction "${MAX_OFFROAD_FRACTION:-0.34}" \
                --no-render-video \
                --capability-failure-mode report \
                --device cuda

            test -s "${run_dir}/summary.json"
            test -s "${run_dir}/best.pt"
            test -s "${run_dir}/best.pt.sha256"
            BC_SUMMARY="${run_dir}/summary.json" python - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["BC_SUMMARY"]).read_text(encoding="utf-8"))
evaluation = summary.get("held_out_evaluation") or {}
if evaluation.get("prebuilt_split") != "test":
    raise SystemExit("BC evaluation did not use the test split")
if evaluation.get("collision_physics_enabled") is not False:
    raise SystemExit("BC evaluation did not disable collision physics")
metrics = evaluation.get("metrics") or {}
if "bc_eval/collision_episode_fraction" not in metrics:
    raise SystemExit("BC evaluation did not record collision-only rate")
if "bc_eval/collision_proxy_episode_fraction" not in metrics:
    raise SystemExit("BC evaluation did not record collision-proxy rate")
PY
            run_activation_check \
                "${run_dir}/best.pt" \
                "${expert_data}" \
                "${activation_dir}" \
                "${capture_layers}" \
                64
            completed_models=$((completed_models + 1))
            if [ "${completed_models}" -eq "${MODEL_LIMIT}" ]; then
                if [ "${MODEL_LIMIT}" -lt 12 ]; then
                    echo "Completed requested BC pilot models: ${MODEL_LIMIT}"
                    exit 0
                fi
            fi
        done
    done
done

STUDY_MANIFEST="${POLICY_ROOT}/study_manifest.json"
cd "${REPODIR}"
python -m scripts_gail.finalize_bc_domain_depth_study \
    --policy-root "${POLICY_ROOT}" \
    --smoke-root "${ACTIVATION_ROOT}" \
    --out "${STUDY_MANIFEST}"

cd "${VFI_PROJECT_ROOT}"
REGISTRY_ROOT="${BC_STUDY_REGISTRY_ROOT:-${VFI_RESULTS_ROOT}/runs/policy_registry/bc_domain_depth_${STUDY_RUN_ID}}"
python -m workflows.analysis.build_policy_checkpoint_registry \
    --policy-root "${POLICY_ROOT}" \
    --sae-root "${VFI_RESULTS_ROOT}/runs/sae" \
    --out "${REGISTRY_ROOT}"

echo "Completed single-job BC study: ${STUDY_MANIFEST}"
