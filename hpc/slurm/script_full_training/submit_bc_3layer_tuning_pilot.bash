#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

mkdir -p "${VFI_LOG_ROOT}/slurm" "${VFI_RESULTS_ROOT}/runs/submissions"
RUNNER="${REPODIR}/hpc/slurm/script_full_training/run_bc_3layer_tuning_pilot.bash"

command -v sbatch >/dev/null 2>&1
test -f "${RUNNER}"
test -s "${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982/manifest.json"
test -s "${VFI_DATA_ROOT}/expert/japanese/continuous_v1/train/manifest.json"

TUNING_JOB_ID="$(
    sbatch --parsable \
        --chdir="${VFI_PROJECT_ROOT}" \
        --export="ALL,REPODIR=${REPODIR}" \
        "${RUNNER}"
)"
TUNING_JOB_ID="${TUNING_JOB_ID%%;*}"
TUNING_ROOT="${VFI_RESULTS_ROOT}/runs/policies/bc/three_layer_tuning_${TUNING_JOB_ID}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_DIR="${VFI_RESULTS_ROOT}/runs/submissions/bc_3layer_tuning_${STAMP}"
mkdir -p "${SUBMISSION_DIR}"
SUBMISSION_DIR="${SUBMISSION_DIR}" TUNING_JOB_ID="${TUNING_JOB_ID}" TUNING_ROOT="${TUNING_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["SUBMISSION_DIR"])
payload = {
    "study": "bc_3layer_optimization_recovery_v1",
    "job_id": os.environ["TUNING_JOB_ID"],
    "execution": "single_job_serial_screen_then_confirmation",
    "tuning_root": os.environ["TUNING_ROOT"],
    "screen_cell": {"domain": "japanese", "seed": 0, "transformer_layers": 3},
    "candidates": [
        {"id": "lr3em4_clip1_drop01", "learning_rate": 3e-4, "max_grad_norm": 1.0, "dropout": 0.1},
        {"id": "lr3em4_clip5_drop01", "learning_rate": 3e-4, "max_grad_norm": 5.0, "dropout": 0.1},
        {"id": "lr3em4_clip1_drop0", "learning_rate": 3e-4, "max_grad_norm": 1.0, "dropout": 0.0},
        {"id": "lr1em3_clip1_drop0", "learning_rate": 1e-3, "max_grad_norm": 1.0, "dropout": 0.0},
        {"id": "lr1em4_clip1_drop0", "learning_rate": 1e-4, "max_grad_norm": 1.0, "dropout": 0.0},
    ],
    "learning_gate": {
        "minimum_validation_skill": 0.10,
        "learning_action_index": 0,
        "minimum_prediction_std_ratio": 0.25,
        "minimum_prediction_target_correlation": 0.50,
    },
    "confirmation_cells": [
        {"domain": "us", "seed": 0, "transformer_layers": 3},
        {"domain": "japanese", "seed": 1, "transformer_layers": 3},
    ],
}
(out / "submission.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Three-layer BC tuning pilot job: ${TUNING_JOB_ID}"
echo "Submission manifest: ${SUBMISSION_DIR}/submission.json"
