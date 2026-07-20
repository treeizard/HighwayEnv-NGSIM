#!/bin/bash
#SBATCH --job-name=gail_airl_confirm
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-19
#SBATCH --output=logs/slurm/gail_airl_confirm_%A_%a.out
#SBATCH --error=logs/slurm/gail_airl_confirm_%A_%a.err

set -euo pipefail

REPODIR="${REPODIR:-$(pwd)}"
STUDY_MANIFEST="${STUDY_MANIFEST:?Set STUDY_MANIFEST to confirmation trials.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPODIR}"
"${PYTHON_BIN}" -m scripts_gail.run_gail_airl_study_trial \
    --manifest "${STUDY_MANIFEST}" \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --python "${PYTHON_BIN}"
