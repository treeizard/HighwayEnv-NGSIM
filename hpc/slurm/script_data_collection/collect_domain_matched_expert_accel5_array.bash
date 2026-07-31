#!/bin/bash
#SBATCH --job-name=domain_expert_a5
#SBATCH --account=bt60
#SBATCH --array=0,1,3,4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/slurm/domain_expert_a5_%A_%a.out
#SBATCH --error=logs/slurm/domain_expert_a5_%A_%a.err

set -euo pipefail

: "${REPODIR:?Submit this script with the absolute component REPODIR exported}"
REPODIR="$(cd -- "${REPODIR}" && pwd)"
export REPODIR
source "${REPODIR}/hpc/slurm/project_env.bash"

assert_sha256() {
    local expected="$1"
    local path="$2"
    if [ -n "${expected}" ]; then
        local actual
        actual="$(sha256sum "${path}" | awk '{print $1}')"
        if [ "${actual}" != "${expected}" ]; then
            echo "Submission-locked source changed: ${path}" >&2
            echo "expected ${expected}, got ${actual}" >&2
            exit 2
        fi
    fi
}

: "${COLLECTION_EXPECTED_COLLECTOR_SHA256:?Missing collector source lock}"
: "${COLLECTION_EXPECTED_CONSTANTS_SHA256:?Missing constants source lock}"
: "${COLLECTION_EXPECTED_CONTRACTS_SHA256:?Missing contracts source lock}"
: "${COLLECTION_EXPECTED_REPLAY_SHA256:?Missing replay source lock}"
: "${COLLECTION_EXPECTED_TRAJECTORY_GEN_SHA256:?Missing trajectory_gen source lock}"
: "${COLLECTION_EXPECTED_NGSIM_ENV_SHA256:?Missing ngsim_env source lock}"
: "${COLLECTION_EXPECTED_LIDAR_SHA256:?Missing lidar source lock}"
: "${COLLECTION_EXPECTED_RUNNER_SHA256:?Missing collection runner source lock}"

assert_sha256 "${COLLECTION_EXPECTED_COLLECTOR_SHA256:-}" \
    "${REPODIR}/scripts_gail/build_ps_traj_expert_discrete.py"
assert_sha256 "${COLLECTION_EXPECTED_CONSTANTS_SHA256:-}" \
    "${REPODIR}/highway_env/ngsim_utils/core/constants.py"
assert_sha256 "${COLLECTION_EXPECTED_CONTRACTS_SHA256:-}" \
    "${REPODIR}/scripts_gail/ps_gail/contracts.py"
assert_sha256 "${COLLECTION_EXPECTED_REPLAY_SHA256:-}" \
    "${REPODIR}/highway_env/ngsim_utils/vehicles/replay.py"
assert_sha256 "${COLLECTION_EXPECTED_TRAJECTORY_GEN_SHA256:-}" \
    "${REPODIR}/highway_env/ngsim_utils/data/trajectory_gen.py"
assert_sha256 "${COLLECTION_EXPECTED_NGSIM_ENV_SHA256:-}" \
    "${REPODIR}/highway_env/envs/ngsim_env.py"
assert_sha256 "${COLLECTION_EXPECTED_LIDAR_SHA256:-}" \
    "${REPODIR}/highway_env/envs/common/observations/lidar.py"
assert_sha256 "${COLLECTION_EXPECTED_RUNNER_SHA256:-}" \
    "${REPODIR}/hpc/slurm/script_data_collection/collect_domain_matched_expert_accel5_array.bash"

DOMAINS=(us us us japanese japanese japanese)
SCENES=(us-101 us-101 us-101 japanese japanese japanese)
SPLITS=(train val test train val test)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#DOMAINS[@]}" ]; then
    echo "Invalid domain-matched expert array task: ${TASK_ID}" >&2
    exit 2
fi

DOMAIN="${DOMAINS[${TASK_ID}]}"
SCENE="${SCENES[${TASK_ID}]}"
SPLIT="${SPLITS[${TASK_ID}]}"
if [ "${SPLIT}" = "test" ] && [ "${COLLECTION_ALLOW_TEST:-false}" != "true" ]; then
    echo "Refusing to open the locked test split without COLLECTION_ALLOW_TEST=true." >&2
    exit 2
fi
COLLECTION_ID="${COLLECTION_ID:-domain_matched_accel5_v2}"
OUT="${VFI_DATA_ROOT}/expert/${COLLECTION_ID}/${DOMAIN}/${SPLIT}"
EXPERT_ACCELERATION_LIMIT_MPS2="${EXPERT_ACCELERATION_LIMIT_MPS2:-5.0}"
export NGSIM_ACCELERATION_LIMIT_MPS2="${EXPERT_ACCELERATION_LIMIT_MPS2}"

COLLECTION_WORKER_THREADS="${COLLECTION_WORKER_THREADS:-2}"
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-4}"
COLLECTION_WORKERS="${COLLECTION_WORKERS:-$((SLURM_CPUS / COLLECTION_WORKER_THREADS))}"
if [ "${COLLECTION_WORKER_THREADS}" -lt 1 ] || [ "${COLLECTION_WORKERS}" -lt 1 ]; then
    echo "Collection worker counts must be positive." >&2
    exit 3
fi

export OMP_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export MKL_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export OPENBLAS_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export NUMEXPR_NUM_THREADS="${COLLECTION_WORKER_THREADS}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${VFI_LOG_ROOT}/matplotlib_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"

cd "${REPODIR}"
mkdir -p \
    "${VFI_LOG_ROOT}/slurm" \
    "${PYTHONPYCACHEPREFIX}" \
    "${MPLCONFIGDIR}" \
    "${VFI_DATA_ROOT}/expert/${COLLECTION_ID}/${DOMAIN}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VFI_CONDA_ENV:-ngsim_env}"

python - <<'PY'
from highway_env.ngsim_utils.core.constants import ACCELERATION_RANGE

expected = float(__import__("os").environ["NGSIM_ACCELERATION_LIMIT_MPS2"])
assert ACCELERATION_RANGE == (-expected, expected), (
    f"Imported acceleration range {ACCELERATION_RANGE} does not match "
    f"the frozen collection limit {expected}."
)
print(f"Frozen expert acceleration range: {ACCELERATION_RANGE} m/s^2")
PY

if [ -e "${OUT}" ]; then
    echo "Refusing to overwrite or reuse collection output: ${OUT}" >&2
    exit 4
fi

python -m scripts_gail.build_ps_traj_expert_discrete \
    --scene "${SCENE}" \
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

EXPERT_DATA_PATH="${OUT}" EXPERT_DOMAIN="${DOMAIN}" EXPERT_SPLIT="${SPLIT}" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts_gail.ps_gail.contracts import (
    validate_declared_raw_observation_space,
)

root = Path(os.environ["EXPERT_DATA_PATH"]).resolve()
repodir = Path(os.environ["REPODIR"]).resolve()
limit = float(os.environ["NGSIM_ACCELERATION_LIMIT_MPS2"])
expected_steering_scale = float(np.pi / 4.0)
files = sorted(root.glob("*.npz"))
if not files:
    raise RuntimeError(f"No expert files were produced under {root}.")

manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
contract = manifest.get("continuous_action_contract")
if not isinstance(contract, dict):
    raise RuntimeError("Collection manifest has no continuous_action_contract.")
observation_contract = manifest.get("policy_observation_contract")
if not isinstance(observation_contract, dict):
    raise RuntimeError("Collection manifest has no policy_observation_contract.")

rows = 0
maximum_residual = 0.0
observation_field_extrema = {}
for path in files:
    with np.load(path, allow_pickle=False) as payload:
        normalized = np.asarray(payload["actions_continuous_env"], dtype=np.float64)
        physical = np.asarray(
            payload["actions_steering_acceleration"],
            dtype=np.float64,
        )
        observations = np.asarray(payload["observations"])
        next_observations = np.asarray(payload["next_observations"])
    observation_receipt = validate_declared_raw_observation_space(
        observations,
        observation_contract,
        context=f"{path} observations",
    )
    next_observation_receipt = validate_declared_raw_observation_space(
        next_observations,
        observation_contract,
        context=f"{path} next_observations",
    )
    for receipt in (observation_receipt, next_observation_receipt):
        for field_name, field in receipt["fields"].items():
            extrema = observation_field_extrema.setdefault(
                field_name,
                {
                    "minimum": float(field["minimum"]),
                    "maximum": float(field["maximum"]),
                },
            )
            extrema["minimum"] = min(
                extrema["minimum"],
                float(field["minimum"]),
            )
            extrema["maximum"] = max(
                extrema["maximum"],
                float(field["maximum"]),
            )
    if normalized.ndim != 2 or normalized.shape[1] != 2:
        raise RuntimeError(f"{path} has invalid normalized actions {normalized.shape}.")
    if physical.shape != normalized.shape:
        raise RuntimeError(f"{path} has misaligned physical actions {physical.shape}.")
    if len(observations) != len(normalized) or next_observations.shape != observations.shape:
        raise RuntimeError(
            f"{path} has misaligned observation/action arrays: "
            f"observations={observations.shape}, "
            f"next_observations={next_observations.shape}, "
            f"actions={normalized.shape}."
        )
    if not np.isfinite(normalized).all() or not np.isfinite(physical).all():
        raise RuntimeError(f"{path} contains non-finite actions.")
    if float(np.max(np.abs(normalized))) > 1.000001:
        raise RuntimeError(f"{path} has normalized actions outside [-1, 1].")
    expected = np.column_stack(
        (
            normalized[:, 1] * expected_steering_scale,
            normalized[:, 0] * limit,
        )
    )
    residual = float(np.max(np.abs(physical - expected)))
    if residual > 5e-5:
        raise RuntimeError(
            f"{path} violates the frozen action contract: residual={residual}."
        )
    rows += int(normalized.shape[0])
    maximum_residual = max(maximum_residual, residual)

if not np.allclose(contract.get("scales"), [limit, expected_steering_scale], atol=1e-8):
    raise RuntimeError(f"Manifest action contract is incorrect: {contract}.")

source_paths = {
    "collector": repodir / "scripts_gail/build_ps_traj_expert_discrete.py",
    "constants": repodir / "highway_env/ngsim_utils/core/constants.py",
    "contracts": repodir / "scripts_gail/ps_gail/contracts.py",
    "replay": repodir / "highway_env/ngsim_utils/vehicles/replay.py",
    "trajectory_gen": (
        repodir / "highway_env/ngsim_utils/data/trajectory_gen.py"
    ),
    "ngsim_env": repodir / "highway_env/envs/ngsim_env.py",
    "lidar": repodir / "highway_env/envs/common/observations/lidar.py",
}
source_sha256 = {
    name: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, path in source_paths.items()
}
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
summary = {
    "schema_version": 1,
    "status": "complete",
    "domain": os.environ["EXPERT_DOMAIN"],
    "split": os.environ["EXPERT_SPLIT"],
    "root": str(root),
    "file_count": len(files),
    "row_count": rows,
    "maximum_absolute_contract_residual": maximum_residual,
    "manifest_sha256": digest,
    "continuous_action_contract": contract,
    "policy_observation_contract": observation_contract,
    "raw_observation_space_validation": {
        "status": "passed",
        "files_checked": len(files),
        "rows_checked_per_array": rows,
        "arrays_checked": ["observations", "next_observations"],
        "field_extrema_across_both_arrays": observation_field_extrema,
    },
    "collection_source_sha256": source_sha256,
}
(root / "action_contract_validation.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
