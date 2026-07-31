#!/bin/bash
set -euo pipefail

# Submit only the recipe-locked BC confirmation matrix. The existing expert collection is
# re-audited, and the BC job runs from an immutable copy of the exact dirty
# source state validated locally.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPODIR="${SOURCE_REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SOURCE_REPODIR="$(cd -- "${SOURCE_REPODIR}" && pwd)"
: "${VFI_PROJECT_ROOT:?Export the validation_first_interpretability_dev project root}"
VFI_PROJECT_ROOT="$(cd -- "${VFI_PROJECT_ROOT}" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/scratch/bt60/ytao0016/conda/envs/ngsim_env/bin/python}"
CAMPAIGN_ID="${CAMPAIGN_ID:-bc_shared_transformer_$(date -u +%Y%m%dT%H%M%SZ)}"
COLLECTION_ID="${COLLECTION_ID:-domain_matched_accel5_v2}"
RUN_TESTS="${RUN_TESTS:-true}"
DRY_RUN="${DRY_RUN:-false}"
BC_TIME_LIMIT="${BC_TIME_LIMIT:-2-00:00:00}"

case "${CAMPAIGN_ID}" in
    *[!A-Za-z0-9._-]*|'')
        echo "Unsafe CAMPAIGN_ID: ${CAMPAIGN_ID}" >&2
        exit 2
        ;;
esac
case "${BC_TIME_LIMIT}" in
    *[!0-9:-]*|'')
        echo "Unsafe BC_TIME_LIMIT: ${BC_TIME_LIMIT}" >&2
        exit 2
        ;;
esac

for required in \
    "${PYTHON_BIN}" \
    "${SOURCE_REPODIR}/configs/bc_gail_aligned_accel5_v5.json"; do
    test -s "${required}"
done

"${PYTHON_BIN}" - \
    "${SOURCE_REPODIR}/configs/bc_gail_aligned_accel5_v5.json" <<'PY'
import json
import sys
from pathlib import Path

recipe = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
architecture = recipe.get("architecture", {})
benchmark = recipe.get("benchmark", {})
evaluation = recipe.get("evaluation", {})
expected = {
    "status": "locked",
    "policy_model": "recurrent_transformer",
    "depths": [2],
    "policy_seeds": [0, 1, 2, 3, 4],
    "transformer_observation_tokenization": "dense_temporal",
    "transformer_observation_normalization": True,
    "memory_tokens": 1,
}
actual = {
    "status": recipe.get("status"),
    "policy_model": architecture.get("policy_model"),
    "depths": architecture.get("depths"),
    "policy_seeds": architecture.get("policy_seeds"),
    "transformer_observation_tokenization": architecture.get(
        "transformer_observation_tokenization"
    ),
    "transformer_observation_normalization": architecture.get(
        "transformer_observation_normalization"
    ),
    "memory_tokens": architecture.get("memory_tokens"),
}
if actual != expected:
    raise SystemExit(f"Shared recurrent-transformer recipe is not locked: {actual}")
expected_benchmark = {
    "objective": "behavior_cloning_policy_realism_reference",
    "matrix_completion_gate": "all_training_artifacts_complete",
    "interpretability_eligibility_gate": "policy_realism_qualified",
    "closed_loop_metrics_role": "qualification_non_terminal_for_matrix_execution",
}
actual_benchmark = {
    key: benchmark.get(key)
    for key in expected_benchmark
}
if actual_benchmark != expected_benchmark:
    raise SystemExit(
        f"BC interpretability benchmark is not locked: {actual_benchmark}"
    )
if evaluation.get("vehicle_mode") != "single" or evaluation.get(
    "terminate_on_collision"
) is not False:
    raise SystemExit(
        "BC primary validation must use the selected single vehicle without "
        f"collision termination: {evaluation}"
    )
if evaluation.get("test_evaluation_mode") != "deferred":
    raise SystemExit("Locked test data must remain sealed during matrix selection.")
if evaluation.get("absolute_outcome_gate_role") != (
    "engineering_alert_non_publication_gate"
):
    raise SystemExit("Absolute collision/offroad thresholds are mislabelled.")
PY

deployment_root="${VFI_PROJECT_ROOT}/deployments/${CAMPAIGN_ID}"
repodir="${deployment_root}/HighwayEnv-NGSIM"
submission_dir="${VFI_PROJECT_ROOT}/results/runs/submissions/${CAMPAIGN_ID}"
slurm_log_root="${VFI_PROJECT_ROOT}/logs/slurm/${CAMPAIGN_ID}"
audit_root="${VFI_PROJECT_ROOT}/results/audits/${CAMPAIGN_ID}"
policy_root_template="${VFI_PROJECT_ROOT}/results/runs/policies/bc/gail_aligned_accel5_JOBID"
for target in \
    "${deployment_root}" \
    "${submission_dir}" \
    "${slurm_log_root}" \
    "${audit_root}"; do
    if [ -e "${target}" ]; then
        echo "Refusing to reuse deployment target: ${target}" >&2
        exit 3
    fi
done

mkdir -p \
    "${deployment_root}" \
    "${submission_dir}" \
    "${slurm_log_root}" \
    "${audit_root}"
git clone --quiet --no-hardlinks "${SOURCE_REPODIR}" "${repodir}"
rsync -a \
    --exclude='.git' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='logs/' \
    "${SOURCE_REPODIR}/" "${repodir}/"

recipe="${repodir}/configs/bc_gail_aligned_accel5_v5.json"
audit_runner="${repodir}/hpc/slurm/script_data_collection/audit_domain_matched_expert_accel5.bash"
bc_runner="${repodir}/hpc/slurm/script_full_training/run_bc_gail_aligned_accel5.bash"
audit_script="${repodir}/scripts_gail/audit_domain_matched_expert.py"
audit_out="${audit_root}/collection_contract_audit_train_val.json"
for required in "${recipe}" "${audit_runner}" "${bc_runner}" "${audit_script}"; do
    test -s "${required}"
done

if [ "${RUN_TESTS}" = true ]; then
    (
        cd "${repodir}"
        PYTHONPATH="${repodir}" CUDA_VISIBLE_DEVICES="" \
            "${PYTHON_BIN}" -m pytest -q \
            tests/test_recurrent_bc.py \
            tests/test_bc_load_once_matrix.py \
            tests/test_expert_observation_validation.py \
            tests/test_gail_airl_runtime_infrastructure.py \
            tests/test_bc_domain_depth_workflow.py \
            tests/test_ps_gail_training_logic.py
    )
fi

# The jobs write only to project data/results/log roots. Freeze the validated
# deployment tree itself so the absolute REPODIR remains an immutable source
# identity for the lifetime of the jobs.
chmod -R a-w "${repodir}"

audit_export="ALL,REPODIR=${repodir},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT}"
audit_export="${audit_export},VFI_CONDA_ENV=ngsim_env,COLLECTION_ID=${COLLECTION_ID}"
audit_export="${audit_export},EXPERT_ACCELERATION_LIMIT_MPS2=5.0,AUDIT_OUT=${audit_out}"
audit_export="${audit_export},AUDIT_SPLITS=train:val"
audit_export="${audit_export},AUDIT_EXPECTED_SCRIPT_SHA256=$(sha256sum "${audit_script}" | awk '{print $1}')"
audit_export="${audit_export},AUDIT_EXPECTED_RUNNER_SHA256=$(sha256sum "${audit_runner}" | awk '{print $1}')"

bc_export="ALL,REPODIR=${repodir},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT}"
bc_export="${bc_export},VFI_CONDA_ENV=ngsim_env,COLLECTION_ID=${COLLECTION_ID}"
bc_export="${bc_export},BC_LOCKED_RECIPE=${recipe},BC_AUDIT_PATH=${audit_out}"
bc_export="${bc_export},BC_EXPECTED_RECIPE_SHA256=$(sha256sum "${recipe}" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_RUNNER_SHA256=$(sha256sum "${bc_runner}" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_MATRIX_SHA256=$(sha256sum "${repodir}/scripts_gail/run_bc_domain_depth_matrix.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_TRAINER_SHA256=$(sha256sum "${repodir}/scripts_gail/train_recurrent_bc_policy.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_RECURRENT_BC_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/recurrent_bc.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_MODELS_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/models.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_CONTRACTS_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/contracts.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_DATA_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/data.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_CHECKPOINTS_SHA256=$(sha256sum "${repodir}/scripts_gail/ps_gail/checkpoints.py" | awk '{print $1}')"
bc_export="${bc_export},BC_EXPECTED_ARCHIVE_SHA256=$(sha256sum "${repodir}/scripts_gail/archive_bc_checkpoints.py" | awk '{print $1}')"

if [ "${DRY_RUN}" = true ]; then
    echo "deployment=${repodir}"
    echo "recipe=${recipe}"
    echo "audit_output=${audit_out}"
    echo "policy_root_template=${policy_root_template}"
    exit 0
fi

sbatch --test-only \
    --chdir="${repodir}" \
    --output="${slurm_log_root}/audit_expert_a5_%j.out" \
    --error="${slurm_log_root}/audit_expert_a5_%j.err" \
    "${audit_runner}"
sbatch --test-only \
    --chdir="${repodir}" \
    --time="${BC_TIME_LIMIT}" \
    --output="${slurm_log_root}/bc_shared_transformer_%j.out" \
    --error="${slurm_log_root}/bc_shared_transformer_%j.err" \
    "${bc_runner}"

audit_job="$(
    sbatch --parsable \
        --chdir="${repodir}" \
        --export="${audit_export}" \
        --output="${slurm_log_root}/audit_expert_a5_%j.out" \
        --error="${slurm_log_root}/audit_expert_a5_%j.err" \
        "${audit_runner}"
)"
audit_job="${audit_job%%;*}"

bc_job="$(
    sbatch --parsable \
        --chdir="${repodir}" \
        --time="${BC_TIME_LIMIT}" \
        --dependency="afterok:${audit_job}" \
        --export="${bc_export}" \
        --output="${slurm_log_root}/bc_shared_transformer_%j.out" \
        --error="${slurm_log_root}/bc_shared_transformer_%j.err" \
        "${bc_runner}"
)"
bc_job="${bc_job%%;*}"

export CAMPAIGN_ID repodir recipe audit_out audit_job bc_job
export slurm_log_root BC_TIME_LIMIT
"${PYTHON_BIN}" - "${submission_dir}/submission.json" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

deployment = Path(os.environ["repodir"])
locked_sources = (
    "hpc/slurm/script_full_training/run_bc_gail_aligned_accel5.bash",
    "scripts_gail/run_bc_domain_depth_matrix.py",
    "scripts_gail/train_recurrent_bc_policy.py",
    "scripts_gail/pretrain_continuous_bc_policy.py",
    "scripts_gail/train_simple_ps_gail.py",
    "scripts_gail/archive_bc_checkpoints.py",
    "scripts_gail/ps_gail/recurrent_bc.py",
    "scripts_gail/ps_gail/envs.py",
    "scripts_gail/ps_gail/training/evaluation.py",
    "scripts_gail/ps_gail/training/policy.py",
    "scripts_gail/ps_gail/training/ppo.py",
    "scripts_gail/ps_gail/models.py",
    "scripts_gail/ps_gail/contracts.py",
    "scripts_gail/ps_gail/data.py",
    "scripts_gail/ps_gail/checkpoints.py",
)
recipe_payload = json.loads(Path(os.environ["recipe"]).read_text(encoding="utf-8"))
depths = list(recipe_payload["architecture"]["depths"])
seeds = list(recipe_payload["architecture"]["policy_seeds"])
payload = {
    "schema_version": 1,
    "submitted_utc": datetime.now(timezone.utc).isoformat(),
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "study": "matched_bc_shared_dense_temporal_recurrent_transformer",
    "status": "validation_matrix_submitted",
    "scientific_qualification_status": "pending_locked_test_and_expert_replay",
    "deployment": os.environ["repodir"],
    "source_lock": {
        "git_revision": subprocess.check_output(
            ("git", "-C", str(deployment), "rev-parse", "HEAD"),
            text=True,
        ).strip(),
        "deployment_tree_write_protected": True,
        "files_sha256": {
            relative: sha256(str(deployment / relative))
            for relative in locked_sources
        },
    },
    "locked_recipe": os.environ["recipe"],
    "recipe_sha256": sha256(os.environ["recipe"]),
    "architecture_contract_id": "shared_dense_temporal_recurrent_transformer_v1",
    "benchmark_contract": {
        "objective": "behavior_cloning_policy_realism_reference",
        "matrix_completion_gate": "all_training_artifacts_complete",
        "interpretability_eligibility_gate": "policy_realism_qualified",
        "closed_loop_metrics_role": "qualification_non_terminal_for_matrix_execution",
        "shared_validation_framework": "shared_bc_gail_paper_metrics_v1",
        "validation_vehicle_mode": "single",
        "collision_termination_enabled": False,
        "test_evaluation_mode": "deferred",
    },
    "transformer_depths": depths,
    "policy_seeds": seeds,
    "policy_model_count": 2 * len(depths) * len(seeds),
    "bc_time_limit": os.environ["BC_TIME_LIMIT"],
    "historical_promotion_precondition": "removed",
    "jobs": {
        "expert_contract_audit": os.environ["audit_job"],
        "bc_matrix": os.environ["bc_job"],
    },
    "dependencies": {
        "bc_matrix": f"afterok:{os.environ['audit_job']}",
    },
    "audit_output": os.environ["audit_out"],
    "policy_root": (
        f"{Path(os.environ['repodir']).parents[2]}/results/runs/policies/bc/"
        f"gail_aligned_accel5_{os.environ['bc_job']}"
    ),
    "slurm_log_root": os.environ["slurm_log_root"],
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "submitted_audit_job=${audit_job}"
echo "submitted_bc_matrix_job=${bc_job}"
echo "deployment=${repodir}"
echo "submission=${submission_dir}/submission.json"
