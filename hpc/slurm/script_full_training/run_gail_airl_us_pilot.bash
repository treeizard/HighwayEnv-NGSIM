#!/bin/bash
#SBATCH --job-name=gail_airl_us_pilot
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:2
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@1800
#SBATCH --no-requeue

set -euo pipefail

: "${REPODIR:?Set REPODIR to the locked HighwayEnv-NGSIM checkout}"
: "${PILOT_MANIFEST:?Set PILOT_MANIFEST to the locked four-trial manifest}"
: "${METHOD:?Set METHOD to gail or airl}"
: "${PILOT_SLURM_LOG_ROOT:?Set PILOT_SLURM_LOG_ROOT explicitly}"

case "${METHOD}" in
    gail|airl) ;;
    *) echo "METHOD must be gail or airl, got ${METHOD}" >&2; exit 2 ;;
esac
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "The paired pilot runner refuses Slurm arrays." >&2
    exit 2
fi

REPODIR="$(cd "${REPODIR}" && pwd)"
PILOT_MANIFEST="$(realpath "${PILOT_MANIFEST}")"
mkdir -p "${PILOT_SLURM_LOG_ROOT}"
source "${REPODIR}/hpc/slurm/project_env.bash"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPODIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NGSIM_ACCELERATION_LIMIT_MPS2="${NGSIM_ACCELERATION_LIMIT_MPS2:-5.0}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_gail_airl_us_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

cd "${REPODIR}"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import torch
import wandb
import highway_env
import scripts_gail

repo = Path(os.environ["REPODIR"]).resolve()
for module in (highway_env, scripts_gail):
    Path(module.__file__).resolve().relative_to(repo)
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected exactly two allocated GPUs, found {torch.cuda.device_count()}")
if not getattr(wandb.Api(), "api_key", None):
    raise SystemExit("W&B online mode requested but no API key is configured")
print("allocated_gpus=" + ",".join(torch.cuda.get_device_name(i) for i in range(2)))
PY

# One full data-integrity pass per method job, then a second depth/output check.
"${PYTHON_BIN}" -m scripts_gail.run_gail_airl_us_pilot_trial \
    --manifest "${PILOT_MANIFEST}" --repo "${REPODIR}" \
    --method "${METHOD}" --depth 2 --python "${PYTHON_BIN}" \
    --verify-only --include-large-data
"${PYTHON_BIN}" -m scripts_gail.run_gail_airl_us_pilot_trial \
    --manifest "${PILOT_MANIFEST}" --repo "${REPODIR}" \
    --method "${METHOD}" --depth 3 --python "${PYTHON_BIN}" --verify-only

declare -a child_pids=()
declare -a child_depths=(2 3)
declare -A child_done=()
runtime_pid=""

signal_children() {
    local signal_name="${1:-USR1}"
    local pid
    for pid in "${child_pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -s "${signal_name}" "${pid}" 2>/dev/null || true
        fi
    done
}

graceful_batch_stop() {
    trap - USR1 TERM INT
    echo "Batch signal received; requesting durable checkpoints from both ${METHOD} depths." >&2
    signal_children USR1
    if [ -n "${runtime_pid}" ] && kill -0 "${runtime_pid}" 2>/dev/null; then
        kill -TERM "${runtime_pid}" 2>/dev/null || true
    fi
    wait || true
    exit 99
}
trap graceful_batch_stop USR1 TERM INT

for depth in "${child_depths[@]}"; do
    trial_log="${PILOT_SLURM_LOG_ROOT}/${METHOD}_depth${depth}_${SLURM_JOB_ID}.log"
    srun --exclusive --exact --ntasks=1 --cpus-per-task=16 --mem=64G \
        --gres=gpu:L40S:1 --unbuffered \
        --output="${trial_log}" --error="${trial_log}" \
        "${PYTHON_BIN}" -m scripts_gail.run_gail_airl_us_pilot_trial \
        --manifest "${PILOT_MANIFEST}" --repo "${REPODIR}" \
        --method "${METHOD}" --depth "${depth}" --python "${PYTHON_BIN}" &
    child_pids+=("$!")
done

"${PYTHON_BIN}" -m scripts_gail.monitor_gail_airl_us_pilot_runtime \
    --manifest "${PILOT_MANIFEST}" --method "${METHOD}" &
runtime_pid="$!"
runtime_done=false
overall_status=0

while true; do
    if [ "${runtime_done}" = false ] && ! kill -0 "${runtime_pid}" 2>/dev/null; then
        set +e
        wait "${runtime_pid}"
        runtime_status=$?
        set -e
        runtime_done=true
        if [ "${runtime_status}" -ne 0 ]; then
            echo "Runtime projection gate failed with status ${runtime_status}; stopping both depths." >&2
            overall_status="${runtime_status}"
            signal_children USR1
        fi
    fi

    all_children_done=true
    for index in "${!child_pids[@]}"; do
        pid="${child_pids[${index}]}"
        if [ "${child_done[${pid}]:-false}" = true ]; then
            continue
        fi
        all_children_done=false
        if ! kill -0 "${pid}" 2>/dev/null; then
            set +e
            wait "${pid}"
            child_status=$?
            set -e
            child_done["${pid}"]=true
            echo "${METHOD} depth ${child_depths[${index}]} exited with ${child_status}."
            if [ "${child_status}" -ne 0 ] && [ "${overall_status}" -eq 0 ]; then
                overall_status="${child_status}"
                signal_children USR1
            fi
        fi
    done

    if [ "${all_children_done}" = true ]; then
        break
    fi
    sleep 10
done

if [ "${runtime_done}" = false ]; then
    set +e
    wait "${runtime_pid}"
    runtime_status=$?
    set -e
    if [ "${runtime_status}" -ne 0 ] && [ "${overall_status}" -eq 0 ]; then
        overall_status="${runtime_status}"
    fi
fi
exit "${overall_status}"
