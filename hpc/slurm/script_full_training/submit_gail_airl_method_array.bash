#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
REPODIR="$(cd "${REPODIR}" && pwd)"

required_roots=(
    VFI_PROJECT_ROOT VFI_DATA_ROOT VFI_HIGHWAY_DATA_ROOT VFI_CHECKPOINT_ROOT
    VFI_RESULTS_ROOT VFI_LOG_ROOT VFI_ARTIFACT_ROOT
)
for variable in "${required_roots[@]}"; do
    value="${!variable:-}"
    if [ -z "${value}" ] || [[ "${value}" != /* ]]; then
        echo "${variable} must be exported as an explicit absolute canonical path." >&2
        exit 2
    fi
done
source "${REPODIR}/hpc/slurm/project_env.bash"

: "${SOURCE_MANIFEST:?Set SOURCE_MANIFEST to the generated study or confirmation trials.json}"
METHOD="${METHOD:?Set METHOD to gail or airl}"
LAUNCH_PROFILE="${LAUNCH_PROFILE:-canary}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROLLOUT_WORKER_THREADS="${ROLLOUT_WORKER_THREADS:-2}"
EVALUATION_WORKER_THREADS="${EVALUATION_WORKER_THREADS:-2}"
CANARY_COUNT="${CANARY_COUNT:-4}"
DRY_RUN="${DRY_RUN:-false}"
SIMULATOR_PROFILE="${SIMULATOR_PROFILE:-legacy}"
BLOCKING_JOB_IDS="${BLOCKING_JOB_IDS:-58391443}"
ALLOW_BLOCKING_JOB_OVERLAP="${ALLOW_BLOCKING_JOB_OVERLAP:-false}"
ALLOW_TEST_SQUEUE_FIXTURE="${ALLOW_TEST_SQUEUE_FIXTURE:-false}"

case "${METHOD}" in
    gail|airl) ;;
    *) echo "METHOD must be gail or airl, got: ${METHOD}" >&2; exit 2 ;;
esac
case "${LAUNCH_PROFILE}" in
    canary) concurrency=2; default_cpus=16; default_mem=64G; default_workers=8 ;;
    production)
        concurrency=4
        : "${CPUS_PER_TASK:?Production requires benchmark-selected CPUS_PER_TASK}"
        : "${MEMORY_PER_TASK:?Production requires benchmark-selected MEMORY_PER_TASK}"
        : "${NUM_ROLLOUT_WORKERS:?Production requires benchmark-selected NUM_ROLLOUT_WORKERS}"
        : "${EVALUATION_NUM_WORKERS:?Production requires benchmark-selected EVALUATION_NUM_WORKERS}"
        default_cpus="${CPUS_PER_TASK}"
        default_mem="${MEMORY_PER_TASK}"
        default_workers="${NUM_ROLLOUT_WORKERS}"
        ;;
    *) echo "LAUNCH_PROFILE must be canary or production, got: ${LAUNCH_PROFILE}" >&2; exit 2 ;;
esac
CPUS_PER_TASK="${CPUS_PER_TASK:-${default_cpus}}"
MEMORY_PER_TASK="${MEMORY_PER_TASK:-${default_mem}}"
NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${default_workers}}"
EVALUATION_NUM_WORKERS="${EVALUATION_NUM_WORKERS:-${default_workers}}"

if [ "${SIMULATOR_PROFILE}" = "optimized" ]; then
    echo "Optimized submissions require the two-domain live parity gate in submit_gail_airl_final_two_stage.bash." >&2
    exit 4
fi

blocking_states=""
active_blockers=""
for job_id in ${BLOCKING_JOB_IDS//,/ }; do
    [ -n "${job_id}" ] || continue
    if [ "${DRY_RUN}" = "true" ] && [ "${ALLOW_TEST_SQUEUE_FIXTURE}" = "true" ]; then
        state="TEST_FIXTURE_NOT_ACTIVE"
    elif command -v squeue >/dev/null 2>&1; then
        state="$(squeue -h -j "${job_id}" -o '%T' | paste -sd, -)"
        state="${state:-NOT_FOUND}"
    elif [ "${DRY_RUN}" = "true" ]; then
        state="UNKNOWN_DRY_RUN"
    else
        echo "squeue is required to enforce the active-job freeze." >&2
        exit 127
    fi
    blocking_states="${blocking_states}${blocking_states:+;}${job_id}=${state}"
    if [[ ",${state}," == *",RUNNING,"* ]] || [[ ",${state}," == *",PENDING,"* ]]; then
        active_blockers="${active_blockers}${active_blockers:+,}${job_id}=${state}"
    fi
done
if [ -n "${active_blockers}" ]; then
    if [ "${DRY_RUN}" != "true" ] || [ "${ALLOW_BLOCKING_JOB_OVERLAP}" != "true" ]; then
        echo "Active-job freeze: refusing overlap with ${active_blockers}." >&2
        exit 4
    fi
fi

case "${REPODIR}/" in
    "${VFI_PROJECT_ROOT}/"*)
        echo "REPODIR must be an isolated sibling checkout, not the canonical live project: ${REPODIR}" >&2
        exit 2
        ;;
esac
SOURCE_REVISION="$(git -C "${REPODIR}" rev-parse HEAD)"
if [ "${DRY_RUN}" != "true" ] && [ -n "$(git -C "${REPODIR}" status --porcelain)" ]; then
    echo "Refusing to submit from a dirty source checkout: ${REPODIR}" >&2
    exit 3
fi

SOURCE_MANIFEST="$(realpath "${SOURCE_MANIFEST}")"
submission_dir="${VFI_RESULTS_ROOT}/runs/submissions/gail_airl_${METHOD}_${LAUNCH_PROFILE}_${RUN_STAMP}"
method_manifest="${submission_dir}/trials.json"
run_root="${VFI_LOG_ROOT}/gail_airl_training"
run_prefix="${RUN_STAMP}/${LAUNCH_PROFILE}"
if [ -e "${submission_dir}" ]; then
    echo "Refusing to reuse submission directory: ${submission_dir}" >&2
    exit 3
fi

cd "${REPODIR}"
"${PYTHON_BIN}" -m scripts_gail.prepare_gail_airl_method_manifest \
    --source "${SOURCE_MANIFEST}" \
    --source-repo "${REPODIR}" \
    --source-revision "${SOURCE_REVISION}" \
    --output "${method_manifest}" \
    --method "${METHOD}" \
    --selection "${LAUNCH_PROFILE}" \
    --canary-count "${CANARY_COUNT}" \
    --run-root "${run_root}" \
    --run-prefix "${run_prefix}" \
    --num-rollout-workers "${NUM_ROLLOUT_WORKERS}" \
    --rollout-worker-threads "${ROLLOUT_WORKER_THREADS}" \
    --evaluation-num-workers "${EVALUATION_NUM_WORKERS}" \
    --evaluation-worker-threads "${EVALUATION_WORKER_THREADS}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --simulator-profile "${SIMULATOR_PROFILE}"

trial_count="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["trial_count"])' "${method_manifest}")"
if [ "${trial_count}" -le 0 ]; then
    echo "Prepared manifest has no trials: ${method_manifest}" >&2
    exit 3
fi
array_spec="0-$((trial_count - 1))%${concurrency}"
runner="${REPODIR}/hpc/slurm/script_full_training/run_gail_airl_method_array.bash"
slurm_log_root="${VFI_LOG_ROOT}/slurm/gail_airl_${METHOD}_${LAUNCH_PROFILE}_${RUN_STAMP}"
mkdir -p "${slurm_log_root}"

export_spec="ALL,REPODIR=${REPODIR},METHOD=${METHOD},STUDY_MANIFEST=${method_manifest},PYTHON_BIN=${PYTHON_BIN}"
export_spec="${export_spec},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT},VFI_DATA_ROOT=${VFI_DATA_ROOT},VFI_HIGHWAY_DATA_ROOT=${VFI_HIGHWAY_DATA_ROOT}"
export_spec="${export_spec},VFI_CHECKPOINT_ROOT=${VFI_CHECKPOINT_ROOT},VFI_RESULTS_ROOT=${VFI_RESULTS_ROOT},VFI_LOG_ROOT=${VFI_LOG_ROOT},VFI_ARTIFACT_ROOT=${VFI_ARTIFACT_ROOT}"
command=(
    sbatch --parsable --array="${array_spec}" --chdir="${REPODIR}"
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}"
    --job-name="${METHOD}_${LAUNCH_PROFILE}" --export="${export_spec}"
    --output="${slurm_log_root}/%A_%a.out" --error="${slurm_log_root}/%A_%a.err"
    "${runner}"
)

echo "method=${METHOD} profile=${LAUNCH_PROFILE} trials=${trial_count} array=${array_spec}"
echo "source=${REPODIR}"
echo "manifest=${method_manifest}"
echo "run_root=${run_root}/${run_prefix}/${METHOD}"
echo "blocking_job_states=${blocking_states} overlap_override=${ALLOW_BLOCKING_JOB_OVERLAP}"
export METHOD LAUNCH_PROFILE SIMULATOR_PROFILE blocking_states ALLOW_BLOCKING_JOB_OVERLAP
export SOURCE_REVISION method_manifest run_root run_prefix
"${PYTHON_BIN}" - "${submission_dir}/submission_metadata.json" <<'PY'
import json, os, sys
states = []
for item in filter(None, os.environ.get("blocking_states", "").split(";")):
    job_id, state = item.split("=", 1)
    states.append({"job_id": job_id, "state": state})
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump({
        "schema_version": 1,
        "method": os.environ["METHOD"],
        "launch_profile": os.environ["LAUNCH_PROFILE"],
        "simulator_profile": os.environ["SIMULATOR_PROFILE"],
        "source_revision": os.environ["SOURCE_REVISION"],
        "manifest": os.environ["method_manifest"],
        "run_root": os.environ["run_root"],
        "run_prefix": os.environ["run_prefix"],
        "blocking_jobs_checked": states,
        "blocking_overlap_override": os.environ["ALLOW_BLOCKING_JOB_OVERLAP"] == "true",
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
if [ "${DRY_RUN}" = "true" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found on PATH. Run this script on the Slurm login node." >&2
    exit 127
fi
job_id="$("${command[@]}")"
echo "submitted_job=${job_id%%;*}"
