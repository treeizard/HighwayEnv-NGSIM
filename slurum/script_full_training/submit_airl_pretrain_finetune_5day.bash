#!/bin/bash
set -euo pipefail

# Submit the full continuous-control PS-AIRL pipeline:
#   stage 1: 50-vehicle pretrain
#   stage 2: 100-vehicle finetune, dependent on successful stage 1 completion
#
# Each submitted job requests a 5-day SLURM wall time by default.
#
# Usage:
#   bash slurum/script_full_training/submit_airl_pretrain_finetune_5day.bash
#
# Useful overrides:
#   RUN_STAMP=my_airl WANDB_MODE=offline \
#   bash slurum/script_full_training/submit_airl_pretrain_finetune_5day.bash

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

PRETRAIN_SCRIPT="${PRETRAIN_SCRIPT:-${REPODIR}/slurum/script_pretrain/train_airl_continuous_gpu_32c_stage1_50veh.bash}"
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-${REPODIR}/slurum/script_finetune/train_airl_continuous_gpu_32c_stage2_100veh.bash}"

PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-full_${RUN_STAMP}_airl_stage1_50veh}"
FINETUNE_RUN_NAME="${FINETUNE_RUN_NAME:-full_${RUN_STAMP}_airl_stage2_100veh}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${REPODIR}/logs/airl/${PRETRAIN_RUN_NAME}/final.pt}"

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

echo "Submitting full PS-AIRL training"
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
        "${common_exports},RUN_NAME=${FINETUNE_RUN_NAME},RESUME_CHECKPOINT=${PRETRAIN_CHECKPOINT},ALLOW_NON_BEST_RESUME=true"
)"

echo "Submitted jobs:"
echo "  AIRL pretrain : ${pretrain_job} (${PRETRAIN_RUN_NAME})"
echo "  AIRL finetune : ${finetune_job} (${FINETUNE_RUN_NAME}) afterok:${pretrain_job}"
echo "  AIRL resume   : ${PRETRAIN_CHECKPOINT}"
