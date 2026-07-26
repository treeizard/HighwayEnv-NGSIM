#!/bin/bash
#SBATCH --job-name=gail_us_scratch
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@1800
#SBATCH --no-requeue

set -euo pipefail

: "${REPODIR:?Set REPODIR to the hash-locked HighwayEnv-NGSIM checkout}"
: "${PILOT_MANIFEST:?Set PILOT_MANIFEST to the scratch GAIL manifest}"
: "${DEPTH:?Set DEPTH to 2 or 3}"
: "${PILOT_SLURM_LOG_ROOT:?Set PILOT_SLURM_LOG_ROOT explicitly}"
RECOVERY_RESUME="${RECOVERY_RESUME:-false}"

case "${DEPTH}" in 2|3) ;; *) echo "DEPTH must be 2 or 3" >&2; exit 2 ;; esac
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Scratch GAIL depth runner refuses Slurm arrays." >&2
    exit 2
fi

REPODIR="$(cd "${REPODIR}" && pwd)"
PILOT_MANIFEST="$(realpath "${PILOT_MANIFEST}")"
mkdir -p "${PILOT_SLURM_LOG_ROOT}"
source "${REPODIR}/hpc/slurm/project_env.bash"

export NGSIM_ACCELERATION_LIMIT_MPS2="${NGSIM_ACCELERATION_LIMIT_MPS2:-5.0}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPODIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_gail_us_scratch_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

cd "${REPODIR}"
"${PYTHON_BIN}" - "${PILOT_MANIFEST}" "${DEPTH}" <<'PY'
import os
import sys
from pathlib import Path

import torch
import wandb

from scripts_gail.ps_gail.checkpoints import policy_architecture_contract
from scripts_gail.ps_gail.pilot import load_manifest, select_trial
from highway_env.ngsim_utils.core.constants import ACCELERATION_RANGE

manifest = load_manifest(Path(sys.argv[1]))
trial = select_trial(manifest, method="gail", depth=int(sys.argv[2]))
scope = dict(manifest["scope"])
data = dict(manifest.get("data") or {})
args = dict(trial["arguments"])
if ACCELERATION_RANGE != (-5.0, 5.0):
    raise SystemExit(f"Expected the aligned ±5 m/s² action contract, got {ACCELERATION_RANGE}")
if data.get("explicit_contracts_required") is not True:
    raise SystemExit("Scratch GAIL manifest does not require explicit data contracts")
if args.get("require_explicit_data_contracts") is not True:
    raise SystemExit("Scratch GAIL trial does not enforce explicit data contracts")
if args.get("validation_require_exact_horizon") is not False:
    raise SystemExit("Scratch GAIL must use a finite score for sub-20-second early policies")
if float(args.get("validation_min_horizon_coverage", -1.0)) != 0.0:
    raise SystemExit("Scratch GAIL optimization must not gate early policies on 20-second coverage")
recovery_resume = os.environ.get("RECOVERY_RESUME", "false").lower() == "true"
expected_initialization = (
    "random_seeded_exact_resume" if recovery_resume else "random_seeded"
)
if scope.get("policy_initialization") != expected_initialization:
    raise SystemExit(
        "Manifest policy initialization mismatch: "
        f"{scope.get('policy_initialization')} != {expected_initialization}"
    )
if scope.get("uses_bc_initialization") is not False:
    raise SystemExit("Manifest unexpectedly enables BC initialization")
architecture = policy_architecture_contract(args)
expected_architecture = {
    "policy_model": "recurrent_transformer",
    "transformer_layers": int(sys.argv[2]),
    "hidden_size": 256,
    "transformer_heads": 4,
    "transformer_dropout": 0.0,
    "transformer_norm_first": True,
    "transformer_observation_normalization": True,
    "transformer_observation_tokenization": "dense_temporal",
    "policy_head_init_std": 0.01,
    "transformer_memory_tokens": 1,
    "transformer_memory_context_length": 32,
}
observed_architecture = {
    name: architecture.get(name) for name in expected_architecture
}
if observed_architecture != expected_architecture:
    raise SystemExit(
        "Scratch GAIL actor is not the shared BC/GAIL actor: "
        f"{observed_architecture}"
    )
if recovery_resume:
    if args.get("initial_policy_checkpoint") or not args.get("resume_checkpoint"):
        raise SystemExit("Recovery must contain only an exact resume checkpoint")
    if int(args.get("expected_resume_round", 0)) <= 0:
        raise SystemExit("Recovery has no positive expected resume round")
else:
    if args.get("initial_policy_checkpoint") or args.get("resume_checkpoint"):
        raise SystemExit("Scratch job contains an initial or resume checkpoint")
for name in (
    "bc_pretrain_epochs",
    "policy_bc_regularization_coef",
    "policy_bc_regularization_final_coef",
    "policy_bc_regularization_decay_rounds",
):
    if float(args.get(name, 0)) != 0.0:
        raise SystemExit(f"Scratch job has nonzero {name}")
workers = int(args["num_rollout_workers"])
threads = int(args["rollout_worker_threads"])
eval_workers = int(args["evaluation_num_workers"])
eval_threads = int(args["evaluation_worker_threads"])
if (workers, threads, eval_workers, eval_threads) != (16, 2, 16, 2):
    raise SystemExit(
        "Expected proven 16x2 rollout/evaluation geometry, got "
        f"{workers}x{threads} and {eval_workers}x{eval_threads}"
    )
if int(args.get("vehicle_increase_soft_collision_rounds", 0)) != 5:
    raise SystemExit("Expected five soft-collision rounds after each vehicle-count increase")
if float(args.get("collision_proxy_penalty_coef", 0.0)) <= 0.0:
    raise SystemExit("Soft-collision gate requires a positive collision-proxy penalty")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected exactly one allocated GPU, found {torch.cuda.device_count()}")
if not getattr(wandb.Api(), "api_key", None):
    raise SystemExit("W&B online mode requested but no API key is configured")
print(
    f"scratch_initialization=true exact_resume_recovery={str(recovery_resume).lower()} "
    f"depth={sys.argv[2]}"
)
print("continuous_acceleration_range_mps2=[-5.0,5.0] explicit_data_contracts=true")
print("optimization_validation_horizon=finite_fallback terminal_coverage_gate=0.95")
print(f"worker_geometry=rollout:{workers}x{threads},evaluation:{eval_workers}x{eval_threads}")
print("vehicle_increase_soft_collision_rounds=5 collision_proxy_penalty_enabled=true")
print(f"allocated_gpu={torch.cuda.get_device_name(0)}")
PY

"${PYTHON_BIN}" -m scripts_gail.run_gail_airl_us_pilot_trial \
    --manifest "${PILOT_MANIFEST}" --repo "${REPODIR}" \
    --method gail --depth "${DEPTH}" --python "${PYTHON_BIN}" \
    --verify-only --include-large-data

child_pid=""
monitor_pid=""
stop_children() {
    local signal_name="${1:-USR1}"
    if [ -n "${child_pid}" ] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -s "${signal_name}" "${child_pid}" 2>/dev/null || true
    fi
    if [ -n "${monitor_pid}" ] && kill -0 "${monitor_pid}" 2>/dev/null; then
        kill -TERM "${monitor_pid}" 2>/dev/null || true
    fi
}
graceful_stop() {
    trap - USR1 TERM INT
    echo "Batch signal received; requesting a durable depth-${DEPTH} checkpoint." >&2
    stop_children USR1
    wait || true
    exit 99
}
trap graceful_stop USR1 TERM INT

trial_log="${PILOT_SLURM_LOG_ROOT}/gail_depth${DEPTH}_${SLURM_JOB_ID}.log"
srun --exclusive --exact --ntasks=1 --cpus-per-task=32 --mem=64G \
    --gres=gpu:L40S:1 --unbuffered \
    --output="${trial_log}" --error="${trial_log}" \
    "${PYTHON_BIN}" -m scripts_gail.run_gail_airl_us_pilot_trial \
    --manifest "${PILOT_MANIFEST}" --repo "${REPODIR}" \
    --method gail --depth "${DEPTH}" --python "${PYTHON_BIN}" &
child_pid="$!"

"${PYTHON_BIN}" -m scripts_gail.monitor_gail_airl_us_pilot_runtime \
    --manifest "${PILOT_MANIFEST}" --method gail --depth "${DEPTH}" &
monitor_pid="$!"
monitor_done=false
overall_status=0

while kill -0 "${child_pid}" 2>/dev/null; do
    if [ "${monitor_done}" = false ] && ! kill -0 "${monitor_pid}" 2>/dev/null; then
        set +e
        wait "${monitor_pid}"
        monitor_status=$?
        set -e
        monitor_done=true
        if [ "${monitor_status}" -ne 0 ]; then
            echo "Depth-${DEPTH} runtime projection failed with ${monitor_status}." >&2
            overall_status="${monitor_status}"
            kill -USR1 "${child_pid}" 2>/dev/null || true
        fi
    fi
    sleep 10
done

set +e
wait "${child_pid}"
child_status=$?
set -e
if [ "${child_status}" -ne 0 ] && [ "${overall_status}" -eq 0 ]; then
    overall_status="${child_status}"
fi
if [ "${monitor_done}" = false ]; then
    if [ "${child_status}" -ne 0 ]; then
        kill -TERM "${monitor_pid}" 2>/dev/null || true
    fi
    set +e
    wait "${monitor_pid}"
    monitor_status=$?
    set -e
    if [ "${monitor_status}" -ne 0 ] && [ "${overall_status}" -eq 0 ]; then
        overall_status="${monitor_status}"
    fi
fi
exit "${overall_status}"
