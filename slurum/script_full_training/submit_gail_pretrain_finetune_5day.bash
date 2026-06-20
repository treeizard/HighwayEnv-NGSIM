#!/bin/bash
set -euo pipefail

# Submit the full continuous-control PS-GAIL pipeline:
#   stage 1: 50-vehicle pretrain
#   stage 2: 100-vehicle finetune, dependent on successful stage 1 completion
#
# Each submitted job requests a 5-day SLURM wall time by default.
#
# Usage:
#   bash slurum/script_full_training/submit_gail_pretrain_finetune_5day.bash
#
# Useful overrides:
#   RUN_STAMP=my_gail WANDB_MODE=offline \
#   bash slurum/script_full_training/submit_gail_pretrain_finetune_5day.bash

REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
JOB_TIME="${JOB_TIME:-5-00:00:00}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
SCENE="${SCENE:-us-101}"
CKPT_ROOT="${CKPT_ROOT:-${REPODIR}/ckpt}"
CKPT_DATASET="${CKPT_DATASET:-}"
EXPERT_DATA="${EXPERT_DATA:-${REPODIR}/expert_data/ngsim_ps_unified_expert_continuous_55145982}"
EPISODE_ROOT="${EPISODE_ROOT:-${REPODIR}/highway_env/data/processed_20s}"
PREBUILT_SPLIT="${PREBUILT_SPLIT:-train}"

PRETRAIN_SCRIPT="${PRETRAIN_SCRIPT:-${REPODIR}/slurum/script_pretrain/train_gail_continuous_gpu_32c_stage1_50veh.bash}"
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-${REPODIR}/slurum/script_finetune/train_gail_continuous_gpu_32c_stage2_100veh.bash}"

PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-full_${RUN_STAMP}_gail_stage1_50veh}"
FINETUNE_RUN_NAME="${FINETUNE_RUN_NAME:-full_${RUN_STAMP}_gail_stage2_100veh}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${REPODIR}/logs/simple_ps_gail/${PRETRAIN_RUN_NAME}/final.pt}"

# Stage two resumes a 50-vehicle policy. Ramp during the first 100 rounds, then
# spend at least 100 rounds training at the full 100 vehicles and 40k agent steps.
FINETUNE_TOTAL_ROUNDS="${FINETUNE_TOTAL_ROUNDS:-200}"
FINETUNE_INITIAL_CONTROLLED_VEHICLES="${FINETUNE_INITIAL_CONTROLLED_VEHICLES:-50}"
FINETUNE_FINAL_CONTROLLED_VEHICLES="${FINETUNE_FINAL_CONTROLLED_VEHICLES:-100}"
FINETUNE_CONTROLLED_VEHICLE_CURRICULUM_ROUNDS="${FINETUNE_CONTROLLED_VEHICLE_CURRICULUM_ROUNDS:-${FINETUNE_TOTAL_ROUNDS}}"
FINETUNE_CONTROLLED_VEHICLE_SCHEDULE="${FINETUNE_CONTROLLED_VEHICLE_SCHEDULE:-0:40:50:70;40:80:70:90;80:100:90:100;100:200:100:100}"
FINETUNE_ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE="${FINETUNE_ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE:-0:40:10000:18000;40:80:18000:30000;80:100:30000:40000;100:200:40000:40000}"
FINETUNE_GAMMA_SCHEDULE="${FINETUNE_GAMMA_SCHEDULE:-0:50:0.95:0.97;50:100:0.97:0.99;100:200:0.99:0.99}"
FINETUNE_ENTROPY_COEF="${FINETUNE_ENTROPY_COEF:-0.001}"
FINETUNE_WARMUP_ENTROPY_COEF="${FINETUNE_WARMUP_ENTROPY_COEF:-0.001}"
FINETUNE_ENTROPY_COEF_SCHEDULE="${FINETUNE_ENTROPY_COEF_SCHEDULE:-0:50:0.001:0.0005;50:100:0.0005:0.0001;100:200:0.0001:0.0001}"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found on PATH. Run this script on the SLURM login node." >&2
    exit 127
fi

for script in "${PRETRAIN_SCRIPT}" "${FINETUNE_SCRIPT}"; do
    if [ ! -f "${script}" ]; then
        echo "Missing script: ${script}" >&2
        exit 2
    fi
done

submit_job() {
    local script="$1"
    local dependency="$2"
    local export_vars="$3"
    local args=("--parsable" "--time=${JOB_TIME}" "--export=ALL,${export_vars}")
    local raw_job_id
    if [ -n "${dependency}" ]; then
        args+=("--dependency=${dependency}")
    fi
    raw_job_id="$(sbatch "${args[@]}" "${script}")"
    echo "${raw_job_id%%;*}"
}

common_exports="REPODIR=${REPODIR},WANDB_MODE=${WANDB_MODE},SCENE=${SCENE},EXPERT_DATA=${EXPERT_DATA},EPISODE_ROOT=${EPISODE_ROOT},PREBUILT_SPLIT=${PREBUILT_SPLIT},CKPT_ROOT=${CKPT_ROOT},CKPT_DATASET=${CKPT_DATASET}"
finetune_exports="${common_exports},RUN_NAME=${FINETUNE_RUN_NAME},RESUME_CHECKPOINT=${PRETRAIN_CHECKPOINT},ALLOW_NON_BEST_RESUME=true"
finetune_exports="${finetune_exports},TOTAL_ROUNDS=${FINETUNE_TOTAL_ROUNDS}"
finetune_exports="${finetune_exports},INITIAL_CONTROLLED_VEHICLES=${FINETUNE_INITIAL_CONTROLLED_VEHICLES},FINAL_CONTROLLED_VEHICLES=${FINETUNE_FINAL_CONTROLLED_VEHICLES},CONTROLLED_VEHICLE_CURRICULUM_ROUNDS=${FINETUNE_CONTROLLED_VEHICLE_CURRICULUM_ROUNDS},CONTROLLED_VEHICLE_SCHEDULE=${FINETUNE_CONTROLLED_VEHICLE_SCHEDULE}"
finetune_exports="${finetune_exports},ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE=${FINETUNE_ROLLOUT_TARGET_AGENT_STEPS_SCHEDULE},GAMMA_SCHEDULE=${FINETUNE_GAMMA_SCHEDULE}"
finetune_exports="${finetune_exports},ENTROPY_COEF=${FINETUNE_ENTROPY_COEF},WARMUP_ENTROPY_COEF=${FINETUNE_WARMUP_ENTROPY_COEF},ENTROPY_COEF_SCHEDULE=${FINETUNE_ENTROPY_COEF_SCHEDULE}"

echo "Submitting full PS-GAIL training"
echo "  REPODIR=${REPODIR}"
echo "  RUN_STAMP=${RUN_STAMP}"
echo "  JOB_TIME=${JOB_TIME}"
echo "  EXPERT_DATA=${EXPERT_DATA}"

pretrain_job="$(
    submit_job \
        "${PRETRAIN_SCRIPT}" \
        "" \
        "${common_exports},RUN_NAME=${PRETRAIN_RUN_NAME}"
)"

finetune_job="$(
    submit_job \
        "${FINETUNE_SCRIPT}" \
        "afterok:${pretrain_job}" \
        "${finetune_exports}"
)"

echo "Submitted jobs:"
echo "  GAIL pretrain : ${pretrain_job} (${PRETRAIN_RUN_NAME})"
echo "  GAIL finetune : ${finetune_job} (${FINETUNE_RUN_NAME}) afterok:${pretrain_job}"
echo "  GAIL resume   : ${PRETRAIN_CHECKPOINT}"
