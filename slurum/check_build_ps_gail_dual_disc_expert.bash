#!/bin/bash
#SBATCH --job-name=ps_gail_dual_expert_check
#SBATCH --account=bt60
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_dual_expert_check_%j.out
#SBATCH --error=/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM/logs/ps_gail_dual_expert_check_%j.err

set -euo pipefail

export REPODIR="${REPODIR:-/home/ytao0016/bt60/ytao0016/HighwayEnv-NGSIM}"
export PYTHONPATH="${REPODIR}:${PYTHONPATH:-}"
COLLECTION_WORKER_THREADS="${COLLECTION_WORKER_THREADS:-2}"
if [ "${COLLECTION_WORKER_THREADS}" -lt 1 ]; then
    COLLECTION_WORKER_THREADS=1
fi
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-4}"
COLLECTION_WORKERS="${COLLECTION_WORKERS:-$((SLURM_CPUS / COLLECTION_WORKER_THREADS))}"
if [ "${COLLECTION_WORKERS}" -lt 1 ]; then
    COLLECTION_WORKERS=1
fi

export OMP_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export MKL_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export OPENBLAS_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export NUMEXPR_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export MPLCONFIGDIR="${REPODIR}/logs/matplotlib_${SLURM_JOB_ID}"

cd "${REPODIR}"
mkdir -p "${REPODIR}/logs" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngsim_env

OUT="${OUT:-${REPODIR}/expert_data/ngsim_ps_traj_expert_dual_disc_check}"
export OUT
MAX_EPISODES="${MAX_EPISODES:-2}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-200}"
SCENE_MAX_VEHICLES="${SCENE_MAX_VEHICLES:-64}"

echo "Collection workers: ${COLLECTION_WORKERS}"
echo "Collection worker threads: ${COLLECTION_WORKER_THREADS}"

python "${REPODIR}/scripts_gail/build_ps_traj_expert_discrete.py" \
    --out "${OUT}" \
    --scene us-101 \
    --episode-root "${REPODIR}/highway_env/data/processed_20s" \
    --prebuilt-split train \
    --control-all-vehicles \
    --max-episodes "${MAX_EPISODES}" \
    --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}" \
    --max-samples-per-vehicle "${MAX_STEPS_PER_EPISODE}" \
    --num-collection-workers "${COLLECTION_WORKERS}" \
    --collection-worker-threads "${COLLECTION_WORKER_THREADS}" \
    --max-episode-steps "${MAX_STEPS_PER_EPISODE}" \
    --scene-max-vehicles "${SCENE_MAX_VEHICLES}" \
    --expert-control-mode continuous \
    --trajectory-state-source simulated \
    --allow-idm

python - <<'PY'
import os
import numpy as np

root = os.environ["OUT"]
files = sorted(
    os.path.join(root, name)
    for name in os.listdir(root)
    if name.endswith(".npz")
)
if not files:
    raise SystemExit(f"No expert npz files were written under {root}")
with np.load(files[0], allow_pickle=True) as data:
    required = {"observations", "trajectory_states", "vehicle_ids", "timesteps", "scene_features", "scene_timesteps"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise SystemExit(f"{files[0]} is missing required arrays: {missing}")
    if data["scene_features"].ndim != 2:
        raise SystemExit(f"scene_features must be rank-2, got {data['scene_features'].shape}")
    print("dual-disc expert check ok:", files[0], "scene_features", data["scene_features"].shape)
PY
