#!/bin/bash
#SBATCH --job-name=jp_expert_actions
#SBATCH --account=bt60
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/slurm/jp_expert_actions_%A_%a.out
#SBATCH --error=logs/slurm/jp_expert_actions_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

SPLITS=(train val test)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#SPLITS[@]}" ]; then
    echo "Invalid Japanese expert array task: ${TASK_ID}" >&2
    exit 2
fi
SPLIT="${SPLITS[${TASK_ID}]}"
OUT="${VFI_DATA_ROOT}/expert/japanese/continuous_v1/${SPLIT}"
COLLECTION_WORKER_THREADS="${COLLECTION_WORKER_THREADS:-2}"
COLLECTION_WORKERS="${COLLECTION_WORKERS:-$((SLURM_CPUS_PER_TASK / COLLECTION_WORKER_THREADS))}"

export OMP_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export MKL_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export OPENBLAS_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export NUMEXPR_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"

cd "${REPODIR}"
mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}" "${VFI_DATA_ROOT}/expert/japanese/continuous_v1"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

if [ -d "${OUT}" ] && [ ! -f "${OUT}/manifest.json" ]; then
    echo "Refusing to mix with incomplete output directory: ${OUT}" >&2
    exit 3
fi

if [ ! -f "${OUT}/manifest.json" ]; then
    python -m scripts_gail.build_ps_traj_expert_discrete \
        --scene japanese \
        --episode-root "${VFI_HIGHWAY_DATA_ROOT}/processed_20s" \
        --prebuilt-split "${SPLIT}" \
        --max-episodes 0 \
        --collect-all-split-episodes \
        --max-steps-per-episode 200 \
        --max-samples-per-vehicle 200 \
        --num-collection-workers "${COLLECTION_WORKERS}" \
        --collection-worker-threads "${COLLECTION_WORKER_THREADS}" \
        --control-all-vehicles \
        --expert-control-mode continuous \
        --trajectory-state-source simulated \
        --no-allow-idm \
        --disable-progress \
        --out "${OUT}"
else
    echo "Reusing completed Japanese expert dataset: ${OUT}"
fi

EXPERT_DATA_PATH="${OUT}" python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

from scripts_gail.ps_gail.data import ACTION_CONTINUOUS_ENV_COLUMNS, load_expert_transition_data

path = Path(os.environ["EXPERT_DATA_PATH"])
transitions = load_expert_transition_data(str(path), max_samples=0)
actions = np.asarray(transitions.actions_continuous_env, dtype=np.float32)
if tuple(transitions.metadata["actions_continuous_env_columns"]) != tuple(ACTION_CONTINUOUS_ENV_COLUMNS):
    raise RuntimeError(f"Unexpected action columns: {transitions.metadata['actions_continuous_env_columns']}")
if actions.ndim != 2 or actions.shape[1] != 2:
    raise RuntimeError(f"Expected continuous actions [N, 2], got {actions.shape}")
if not np.isfinite(actions).all():
    raise RuntimeError("Japanese expert actions contain non-finite values.")
if float(np.max(np.abs(actions))) > 1.000001:
    raise RuntimeError("Japanese normalized expert actions exceed [-1, 1].")
summary = {
    "valid": True,
    "source": str(path.resolve()),
    "samples": int(actions.shape[0]),
    "trajectories": int(len(set(map(str, transitions.trajectory_ids.tolist())))),
    "action_columns": list(ACTION_CONTINUOUS_ENV_COLUMNS),
    "action_mean": actions.mean(axis=0).tolist(),
    "action_std": actions.std(axis=0).tolist(),
    "action_min": actions.min(axis=0).tolist(),
    "action_max": actions.max(axis=0).tolist(),
}
(path / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
