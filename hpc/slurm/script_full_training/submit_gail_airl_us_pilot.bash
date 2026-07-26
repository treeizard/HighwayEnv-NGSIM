#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
REPODIR="$(cd "${REPODIR}" && pwd)"
: "${VFI_PROJECT_ROOT:?Export the canonical validation_first_interpretability project root}"
: "${LOCAL_SMOKE_REPORT:?Set LOCAL_SMOKE_REPORT to a passing local-GPU smoke_report.json}"

VFI_PROJECT_ROOT="$(cd "${VFI_PROJECT_ROOT}" && pwd)"
DRY_RUN="${DRY_RUN:-false}"
ALLOW_ACTIVE_BC_DRY_RUN="${ALLOW_ACTIVE_BC_DRY_RUN:-false}"
RUN_CPU_TESTS="${RUN_CPU_TESTS:-true}"
BC_JOB_ID="58391443"
CAMPAIGN_ID="${CAMPAIGN_ID:-gail_airl_us_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/bt60/ytao0016/conda/envs/ngsim_env/bin/python}"

if [ "${DRY_RUN}" != true ] && [ -n "$(git -C "${REPODIR}" status --porcelain --untracked-files=all)" ]; then
    echo "Official pilot submission requires a clean source checkout: ${REPODIR}" >&2
    exit 3
fi

bc_state=""
if command -v squeue >/dev/null 2>&1; then
    bc_state="$(squeue -h -j "${BC_JOB_ID}" -o '%T' | head -n 1)"
fi
if [ -z "${bc_state}" ] && command -v sacct >/dev/null 2>&1; then
    bc_state="$(sacct -X -n -P -j "${BC_JOB_ID}" --format=State | sed -n '1{s/+.*//;p;}')"
fi
bc_state="${bc_state:-UNKNOWN}"
if [ "${bc_state}" != COMPLETED ]; then
    if [ "${DRY_RUN}" != true ] || [ "${ALLOW_ACTIVE_BC_DRY_RUN}" != true ]; then
        echo "BC completion gate: job ${BC_JOB_ID} is ${bc_state}; required COMPLETED." >&2
        exit 4
    fi
    echo "Dry-run only: bypassing active BC state ${bc_state}." >&2
fi

case "${CAMPAIGN_ID}" in
    *[!A-Za-z0-9._-]*|'') echo "Unsafe CAMPAIGN_ID: ${CAMPAIGN_ID}" >&2; exit 2 ;;
esac
submission_dir="${VFI_PROJECT_ROOT}/results/runs/submissions/${CAMPAIGN_ID}"
run_root="${VFI_PROJECT_ROOT}/results/runs/policies/gail_airl/${CAMPAIGN_ID}"
slurm_log_root="${VFI_PROJECT_ROOT}/logs/slurm/${CAMPAIGN_ID}"
if [ -e "${submission_dir}" ] || [ -e "${run_root}" ]; then
    echo "Refusing to reuse campaign output: ${CAMPAIGN_ID}" >&2
    exit 3
fi

temporary_root="$(mktemp -d)"
cleanup() {
    if [ -n "${temporary_root:-}" ] && [ -d "${temporary_root}" ]; then
        rm -rf -- "${temporary_root}"
    fi
}
trap cleanup EXIT
manifest="${temporary_root}/pilot_manifest.json"
cd "${REPODIR}"
"${PYTHON_BIN}" -m scripts_gail.build_gail_airl_us_pilot \
    --repo "${REPODIR}" --project-root "${VFI_PROJECT_ROOT}" \
    --run-root "${run_root}" --campaign-id "${CAMPAIGN_ID}" \
    --output "${manifest}"

"${PYTHON_BIN}" - "${manifest}" "${LOCAL_SMOKE_REPORT}" <<'PY'
import json
import sys
from pathlib import Path
from scripts_gail.ps_gail.pilot import load_manifest, source_fingerprint

manifest = load_manifest(Path(sys.argv[1]))
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if report.get("passed") is not True or report.get("dry_run") is True:
    raise SystemExit("LOCAL_SMOKE_REPORT is not a passing real-GPU report")
if report.get("source_fingerprint") != source_fingerprint(manifest):
    raise SystemExit("Local GPU smoke source lock differs from submission source")
if len(report.get("results") or []) != 4:
    raise SystemExit("Local GPU smoke did not cover all four pilot trials")
PY

if [ "${RUN_CPU_TESTS}" = true ]; then
    "${PYTHON_BIN}" -m pytest -q \
        tests/test_gail_airl_study.py \
        tests/test_gail_airl_runtime_infrastructure.py \
        tests/test_ps_gail_training_logic.py
fi

"${PYTHON_BIN}" - <<'PY'
import wandb
if not getattr(wandb.Api(), "api_key", None):
    raise SystemExit("W&B online mode requested but no API key is configured")
PY

mkdir -p "${submission_dir}" "${slurm_log_root}"
mv "${manifest}" "${submission_dir}/pilot_manifest.json"
manifest="${submission_dir}/pilot_manifest.json"
runner="${REPODIR}/hpc/slurm/script_full_training/run_gail_airl_us_pilot.bash"
common_export="ALL,REPODIR=${REPODIR},PILOT_MANIFEST=${manifest},PILOT_SLURM_LOG_ROOT=${slurm_log_root}"
common_export="${common_export},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT},VFI_CONDA_ENV=ngsim_env,PYTHON_BIN=${PYTHON_BIN}"
gail_command=(sbatch --parsable --chdir="${REPODIR}" --job-name=gail_us_pilot \
    --export="${common_export},METHOD=gail" \
    --output="${slurm_log_root}/gail_%j.out" --error="${slurm_log_root}/gail_%j.err" "${runner}")
airl_command=(sbatch --parsable --chdir="${REPODIR}" --job-name=airl_us_pilot \
    --export="${common_export},METHOD=airl" \
    --output="${slurm_log_root}/airl_%j.out" --error="${slurm_log_root}/airl_%j.err" "${runner}")

if [ "${DRY_RUN}" = true ]; then
    printf '%q ' "${gail_command[@]}"; printf '\n'
    printf '%q ' "${airl_command[@]}"; printf '\n'
    exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable." >&2
    exit 127
fi

gail_job="$("${gail_command[@]}")"
gail_job="${gail_job%%;*}"
set +e
airl_output="$("${airl_command[@]}" 2>&1)"
airl_status=$?
set -e
if [ "${airl_status}" -ne 0 ]; then
    scancel "${gail_job}" || true
    echo "AIRL submission failed; cancelled paired GAIL job ${gail_job}: ${airl_output}" >&2
    exit "${airl_status}"
fi
airl_job="${airl_output%%;*}"

export CAMPAIGN_ID BC_JOB_ID bc_state manifest run_root slurm_log_root gail_job airl_job
"${PYTHON_BIN}" - "${submission_dir}/submission_metadata.json" <<'PY'
from datetime import datetime, timezone
import json
import os
import sys

payload = {
    "schema_version": 1,
    "submitted_utc": datetime.now(timezone.utc).isoformat(),
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "manifest": os.environ["manifest"],
    "run_root": os.environ["run_root"],
    "slurm_log_root": os.environ["slurm_log_root"],
    "bc_completion_gate": {"job_id": os.environ["BC_JOB_ID"], "state": os.environ["bc_state"]},
    "jobs": {"gail": os.environ["gail_job"], "airl": os.environ["airl_job"]},
    "arrays": False,
    "dependencies": False,
    "requeue": False,
    "automatic_retry": False,
}
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
echo "submitted_gail_job=${gail_job}"
echo "submitted_airl_job=${airl_job}"
echo "manifest=${manifest}"
