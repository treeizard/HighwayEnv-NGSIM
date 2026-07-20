#!/bin/bash
#SBATCH --job-name=gail_airl_pilot
#SBATCH --account=bt60
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/gail_airl_pilot_%j.out
#SBATCH --error=logs/slurm/gail_airl_pilot_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${REPODIR}/hpc/slurm/project_env.bash"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${VFI_LOG_ROOT}/slurm" "${PYTHONPYCACHEPREFIX}"
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

WARM_START="${GAIL_AIRL_WARM_START:-${VFI_RESULTS_ROOT}/runs/policies/bc/gail_airl_warm_start_20260717T0505Z/best.pt}"
EXPERT_DATA="${GAIL_AIRL_EXPERT_DATA:-${VFI_DATA_ROOT}/expert/ngsim_ps_unified_expert_continuous_55145982}"
EPISODE_ROOT="${GAIL_AIRL_EPISODE_ROOT:-${VFI_HIGHWAY_DATA_ROOT}/processed_20s}"
PILOT_ID="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}"
PILOT_ROOT="${GAIL_AIRL_PILOT_ROOT:-${VFI_RESULTS_ROOT}/runs/policies/gail_airl/warm_start_pilot_${PILOT_ID}}"

test -s "${WARM_START}"
test -s "${EXPERT_DATA}/manifest.json"
if [ -e "${PILOT_ROOT}/gail/final.pt" ] || [ -e "${PILOT_ROOT}/airl/final.pt" ]; then
    echo "Refusing to overwrite an existing pilot: ${PILOT_ROOT}" >&2
    exit 2
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable for the GAIL/AIRL pilot")
print(f"GAIL/AIRL pilot CUDA device: {torch.cuda.get_device_name(0)}")
PY

COMMON_ARGS=(
    --expert-data "${EXPERT_DATA}"
    --episode-root "${EPISODE_ROOT}"
    --prebuilt-split train
    --validation-prebuilt-split val
    --test-prebuilt-split test
    --validation-vehicle-mode training_count
    --test-vehicle-mode training_count
    --seed 0
    --device cuda
    --action-mode continuous
    --policy-model recurrent_transformer
    --hidden-size 256
    --transformer-layers 2
    --transformer-heads 4
    --transformer-dropout 0.1
    --transformer-memory-tokens 8
    --transformer-memory-context-length 32
    --initial-policy-checkpoint "${WARM_START}"
    --policy-bc-regularization-coef 0
    --total-rounds 5
    --max-expert-samples 100000
    --controlled-vehicle-curriculum
    --initial-controlled-vehicles 10
    --final-controlled-vehicles 10
    --controlled-vehicle-curriculum-rounds 5
    --rollout-target-agent-steps 2000
    --initial-rollout-target-agent-steps 2000
    --final-rollout-target-agent-steps 2000
    --rollout-target-agent-steps-curriculum-rounds 5
    --rollout-min-episodes 1
    --rollout-max-episode-steps 100
    --max-episode-steps 100
    --num-rollout-workers 1
    --rollout-worker-threads 2
    --evaluation-num-workers 2
    --evaluation-worker-threads 2
    --disc-learning-rate 0.0001
    --warmup-rounds 5
    --warmup-learning-rate 0.000005
    --warmup-disc-learning-rate 0.00005
    --warmup-clip-range 0.05
    --clip-range 0.1
    --value-clip-range 0.2
    --target-kl 0.005
    --ppo-epochs 2
    --batch-size 1024
    --disc-batch-size 1024
    --disc-updates-per-round 1
    --collision-mode-schedule '1:2:soft;3:4:mixed;5:5:full'
    --enable-collision
    --evaluate-initial-policy
    --validation-every 1
    --validation-episodes 3
    --validation-max-score-drop 5
    --validation-regression-patience 2
    --validation-stress-every 0
    --validation-stress-episodes 0
    --test-episodes 3
    --checkpoint-every 1
    --no-save-checkpoint-video
    --wandb-mode disabled
    --abort-on-health-failure
)

cd "${REPODIR}"
set +e
python -m scripts_gail.train_simple_ps_gail \
    "${COMMON_ARGS[@]}" \
    --algorithm-variant gail_bce \
    --run-name "${PILOT_ROOT}/gail" \
    --learning-rate 0.00003 \
    --entropy-coef 0.002
gail_status=$?

python -m scripts_gail.train_simple_airl \
    "${COMMON_ARGS[@]}" \
    --algorithm-variant airl_bce \
    --run-name "${PILOT_ROOT}/airl" \
    --learning-rate 0.00001 \
    --entropy-coef 0.003
airl_status=$?
set -e

python - "${PILOT_ROOT}" "${gail_status}" "${airl_status}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "gail_exit_code": int(sys.argv[2]),
    "airl_exit_code": int(sys.argv[3]),
    "gail_summary": str(root / "gail" / "evaluation_summary.json"),
    "airl_summary": str(root / "airl" / "evaluation_summary.json"),
}
root.mkdir(parents=True, exist_ok=True)
(root / "pilot_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [ "${gail_status}" -ne 0 ] || [ "${airl_status}" -ne 0 ]; then
    exit 1
fi
