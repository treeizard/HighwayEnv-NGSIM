#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
REPODIR="$(cd "${REPODIR}" && pwd)"
export REPODIR

required_roots=(
    VFI_PROJECT_ROOT VFI_DATA_ROOT VFI_HIGHWAY_DATA_ROOT VFI_CHECKPOINT_ROOT
    VFI_RESULTS_ROOT VFI_LOG_ROOT VFI_ARTIFACT_ROOT
)
for variable in "${required_roots[@]}"; do
    value="${!variable:-}"
    if [ -z "${value}" ] || [[ "${value}" != /* ]]; then
        echo "${variable} must be an explicit absolute canonical path." >&2
        exit 2
    fi
done
source "${REPODIR}/hpc/slurm/project_env.bash"

METHOD="${METHOD:?Set METHOD to gail or airl}"
LAUNCH_PROFILE="${LAUNCH_PROFILE:?Set LAUNCH_PROFILE to canary or production}"
STAGE1_MANIFEST="$(realpath "${STAGE1_MANIFEST:?Set the generated stage-1 manifest}")"
STAGE2_MANIFEST="$(realpath "${STAGE2_MANIFEST:?Set the generated stage-2 manifest}")"
BC_AUDIT_JSON="$(realpath "${BC_AUDIT_JSON:?Set the terminal BC warm-start audit JSON}")"
DEPENDENCY_MODE="${DEPENDENCY_MODE:-aftercorr}"
DRY_RUN="${DRY_RUN:-false}"
ALLOW_TEST_BENCHMARK_FIXTURE="${ALLOW_TEST_BENCHMARK_FIXTURE:-false}"
ALLOW_BLOCKING_JOB_OVERLAP="${ALLOW_BLOCKING_JOB_OVERLAP:-false}"
ALLOW_TEST_SQUEUE_FIXTURE="${ALLOW_TEST_SQUEUE_FIXTURE:-false}"
BLOCKING_JOB_IDS="${BLOCKING_JOB_IDS:-58391443}"
MIN_SIMULATOR_SPEEDUP="${MIN_SIMULATOR_SPEEDUP:-1.10}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${METHOD}" in gail|airl) ;; *) echo "METHOD must be gail or airl." >&2; exit 2 ;; esac
case "${LAUNCH_PROFILE}" in canary|production) ;; *) echo "Invalid LAUNCH_PROFILE." >&2; exit 2 ;; esac
case "${DEPENDENCY_MODE}" in aftercorr|per-cell-afterok) ;; *) echo "Invalid DEPENDENCY_MODE." >&2; exit 2 ;; esac
case "${DRY_RUN}" in true|false) ;; *) echo "DRY_RUN must be true or false." >&2; exit 2 ;; esac

case "${REPODIR}/" in
    "${VFI_PROJECT_ROOT}/"*)
        echo "REPODIR must be an isolated sibling checkout, not the canonical live project." >&2
        exit 2
        ;;
esac

if [ "${LAUNCH_PROFILE}" = "canary" ]; then
    concurrency=2
    CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
    MEMORY_PER_TASK="${MEMORY_PER_TASK:-64G}"
    NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-8}"
    EVALUATION_NUM_WORKERS="${EVALUATION_NUM_WORKERS:-8}"
else
    concurrency=4
    : "${CPUS_PER_TASK:?Production requires benchmark-selected CPUS_PER_TASK}"
    : "${MEMORY_PER_TASK:?Production requires benchmark-selected MEMORY_PER_TASK}"
    : "${NUM_ROLLOUT_WORKERS:?Production requires benchmark-selected NUM_ROLLOUT_WORKERS}"
    : "${EVALUATION_NUM_WORKERS:?Production requires benchmark-selected EVALUATION_NUM_WORKERS}"
    CANARY_AUDIT_JSON="$(realpath "${CANARY_AUDIT_JSON:?Production requires a passed canary audit}")"
    CANARY_STAGE2_MANIFEST="$(realpath "${CANARY_STAGE2_MANIFEST:?Production requires the audited canary stage-2 manifest}")"
fi
ROLLOUT_WORKER_THREADS="${ROLLOUT_WORKER_THREADS:-2}"
EVALUATION_WORKER_THREADS="${EVALUATION_WORKER_THREADS:-2}"

if [ $((NUM_ROLLOUT_WORKERS * ROLLOUT_WORKER_THREADS)) -gt "${CPUS_PER_TASK}" ]; then
    echo "Rollout worker geometry exceeds CPUS_PER_TASK." >&2
    exit 2
fi
if [ $((EVALUATION_NUM_WORKERS * EVALUATION_WORKER_THREADS)) -gt "${CPUS_PER_TASK}" ]; then
    echo "Evaluation worker geometry exceeds CPUS_PER_TASK." >&2
    exit 2
fi

if [ "${DRY_RUN}" != "true" ]; then
    if [ -n "$(git -C "${REPODIR}" status --porcelain --untracked-files=all)" ]; then
        echo "Refusing non-dry-run submission from a dirty isolated checkout." >&2
        exit 3
    fi
fi

trial_count="$(${PYTHON_BIN} - "${STAGE1_MANIFEST}" "${STAGE2_MANIFEST}" "${METHOD}" "${LAUNCH_PROFILE}" "${REPODIR}" "${BC_AUDIT_JSON}" "${CPUS_PER_TASK}" "${MEMORY_PER_TASK}" "${NUM_ROLLOUT_WORKERS}" "${ROLLOUT_WORKER_THREADS}" "${EVALUATION_NUM_WORKERS}" "${EVALUATION_WORKER_THREADS}" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

(stage1_path, stage2_path, method, profile, repo, bc_audit_path,
 cpus, memory, rollout_workers, rollout_threads, evaluation_workers, evaluation_threads) = sys.argv[1:]

def load(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"Manifest is not a JSON object: {path}")
    return value

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

s1, s2 = load(stage1_path), load(stage2_path)
for expected_stage, payload in ((1, s1), (2, s2)):
    if payload.get("manifest_kind") != "gail_airl_final_stage":
        raise SystemExit("Final launcher rejects screening/non-final manifests.")
    if payload.get("method") != method or int(payload.get("stage", -1)) != expected_stage:
        raise SystemExit("Method/stage manifest mismatch.")
    if payload.get("launch_profile") != profile:
        raise SystemExit("Manifest launch_profile mismatch.")
if s1.get("canonical_indices") != s2.get("canonical_indices"):
    raise SystemExit("Stage canonical-index mappings differ.")
expected_indices = [0, 3, 6, 9] if profile == "canary" else [1, 2, 4, 5, 7, 8, 10, 11]
if s1.get("canonical_indices") != expected_indices:
    raise SystemExit(f"Unexpected {profile} canonical indices: {s1.get('canonical_indices')}")
if int(s1.get("trial_count", -1)) != len(expected_indices) or int(s2.get("trial_count", -1)) != len(expected_indices):
    raise SystemExit("Manifest trial count is inconsistent.")
if s1.get("simulator_profile") != s2.get("simulator_profile"):
    raise SystemExit("Simulator profiles differ between stages.")
if s1.get("source_code") != s2.get("source_code"):
    raise SystemExit("Source locks differ between stages.")
source = s1["source_code"]
repo_path = Path(repo).resolve()
if Path(source.get("repo", "")).resolve() != repo_path:
    raise SystemExit("Source-lock repository mismatch.")
head = subprocess.check_output(("git", "-C", str(repo_path), "rev-parse", "HEAD"), text=True).strip()
tree = subprocess.check_output(("git", "-C", str(repo_path), "rev-parse", "HEAD^{tree}"), text=True).strip()
if source.get("revision") != head or source.get("tree") != tree:
    raise SystemExit("Checked-out source revision/tree differs from manifest lock.")
for relative, expected in source.get("files_sha256", {}).items():
    if digest(repo_path / relative) != expected:
        raise SystemExit(f"Source-lock hash mismatch: {relative}")
audit = load(bc_audit_path)
if audit.get("terminal") is not True or str(audit.get("status", "")).lower() not in {"complete", "passed"}:
    raise SystemExit("BC audit is not terminal/passed.")
recorded_audit = s1.get("bc_audit", {})
if Path(recorded_audit.get("path", "")).resolve() != Path(bc_audit_path).resolve() or recorded_audit.get("sha256") != digest(bc_audit_path):
    raise SystemExit("BC audit path/hash differs from the locked manifest.")
geometry = s1.get("resource_geometry", {})
expected_geometry = {
    "cpus_per_task": int(cpus), "memory_per_task": memory,
    "num_rollout_workers": int(rollout_workers), "rollout_worker_threads": int(rollout_threads),
    "evaluation_num_workers": int(evaluation_workers), "evaluation_worker_threads": int(evaluation_threads),
}
if geometry != expected_geometry or s2.get("resource_geometry") != expected_geometry:
    raise SystemExit(f"Requested resources differ from manifest: {expected_geometry} != {geometry}")
run_root = Path(s1["run_root"]).resolve()
for local_index, (row1, row2) in enumerate(zip(s1["trials"], s2["trials"], strict=True)):
    if int(row1["index"]) != local_index or int(row2["index"]) != local_index:
        raise SystemExit("Local array indices are not contiguous/aligned.")
    c1 = int(row1["arguments"]["study_cell_index"])
    c2 = int(row2["arguments"]["study_cell_index"])
    if c1 != expected_indices[local_index] or c2 != c1:
        raise SystemExit("Canonical cell mapping mismatch.")
    if int(row1["arguments"].get("study_stage", -1)) != 1 or int(row2["arguments"].get("study_stage", -1)) != 2:
        raise SystemExit("Study-stage identity mismatch.")
    stage1_dir = (run_root / row1["run_name"]).resolve()
    stage2_dir = (run_root / row2["run_name"]).resolve()
    if Path(row2["arguments"]["resume_checkpoint"]).resolve() != stage1_dir / "resume_latest.pt":
        raise SystemExit("Stage2 does not resume the matching stage1 artifact.")
    if stage1_dir.exists() or stage2_dir.exists():
        raise SystemExit(f"Refusing to reuse existing output directory: {stage1_dir}, {stage2_dir}")
print(len(expected_indices))
PY
)"

SIMULATOR_PROFILE="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["simulator_profile"])' "${STAGE1_MANIFEST}")"
if [ "${SIMULATOR_PROFILE}" = "optimized" ]; then
    BENCHMARK_EVIDENCE_JSON="$(realpath "${BENCHMARK_EVIDENCE_JSON:?Optimized simulator requires benchmark evidence JSON}")"
    "${PYTHON_BIN}" - "${BENCHMARK_EVIDENCE_JSON}" "${MIN_SIMULATOR_SPEEDUP}" "${DRY_RUN}" "${ALLOW_TEST_BENCHMARK_FIXTURE}" "${REPODIR}" "${STAGE1_MANIFEST}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
path, threshold, dry_run, allow_fixture, repo, manifest_path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    evidence = json.load(handle)
is_fixture = evidence.get("test_fixture") is True
if is_fixture:
    if dry_run != "true" or allow_fixture != "true":
        raise SystemExit("Benchmark fixtures are allowed only for explicit dry-run tests.")
else:
    if evidence.get("benchmark_kind") != "ngsim_live_data_parity" or evidence.get("live_data") is not True:
        raise SystemExit("Optimized evidence must come from the live-data NGSIM parity benchmark.")
    if str(evidence.get("status", "")).lower() != "passed":
        raise SystemExit("Optimized benchmark evidence is not passed.")
    cases = evidence.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("Optimized evidence must include per-domain/per-vehicle cases.")
    for scene in ("us-101", "japanese"):
        scene_cases = [row for row in cases if row.get("scene") == scene]
        for vehicles in (50, 100):
            matches = [row for row in scene_cases if int(row.get("controlled_vehicles", -1)) == vehicles]
            if len(matches) != 1:
                raise SystemExit(f"Evidence must contain exactly one {scene}/{vehicles}-vehicle case.")
            row = matches[0]
            if row.get("strict_parity") is not True or float(row.get("observation_max_abs_error", 1.0)) != 0.0:
                raise SystemExit(f"Strict zero-tolerance parity failed for {scene}/{vehicles} vehicles.")
        case100 = next(row for row in scene_cases if int(row.get("controlled_vehicles", -1)) == 100)
        if float(case100.get("speedup_ratio", 0.0)) < float(threshold):
            raise SystemExit(f"100-vehicle speed threshold failed for {scene}.")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    source = evidence.get("source_code") or {}
    manifest_source = manifest.get("source_code") or {}
    head = subprocess.check_output(("git", "-C", repo, "rev-parse", "HEAD"), text=True).strip()
    tree = subprocess.check_output(("git", "-C", repo, "rev-parse", "HEAD^{tree}"), text=True).strip()
    if source.get("repo") != str(Path(repo).resolve()) or source.get("revision") != head or source.get("tree") != tree:
        raise SystemExit("Benchmark evidence is not bound to this exact source checkout.")
    if source.get("revision") != manifest_source.get("revision") or source.get("tree") != manifest_source.get("tree"):
        raise SystemExit("Benchmark evidence source differs from campaign source lock.")
    episode_data = evidence.get("episode_data") or {}
    expected_roots = {}
    for row in manifest.get("trials", []):
        args = row["arguments"]
        expected_roots[args["scene"]] = str(Path(args["episode_root"]).resolve())
    for scene in ("us-101", "japanese"):
        if str(Path((episode_data.get(scene) or {}).get("episode_root", "")).resolve()) != expected_roots.get(scene):
            raise SystemExit(f"Benchmark episode-data identity differs for {scene}.")
PY
else
    BENCHMARK_EVIDENCE_JSON=""
fi

if [ "${LAUNCH_PROFILE}" = "production" ]; then
    "${PYTHON_BIN}" - "${CANARY_AUDIT_JSON}" "${CANARY_STAGE2_MANIFEST}" "${METHOD}" "${SIMULATOR_PROFILE}" "${STAGE1_MANIFEST}" <<'PY'
import hashlib
import json
import sys
audit_path, canary_manifest_path, method, simulator_profile, production_manifest_path = sys.argv[1:]
with open(audit_path, encoding="utf-8") as handle:
    audit = json.load(handle)
if audit.get("terminal") is not True or str(audit.get("status", "")).lower() != "passed":
    raise SystemExit("Production requires a terminal passed canary audit.")
if audit.get("canonical_indices") != [0, 3, 6, 9]:
    raise SystemExit("Canary audit does not cover canonical indices 0/3/6/9.")
if audit.get("method") != method or audit.get("simulator_profile") != simulator_profile:
    raise SystemExit("Canary audit method/simulator profile mismatch.")
with open(canary_manifest_path, "rb") as handle:
    canary_bytes = handle.read()
with open(canary_manifest_path, encoding="utf-8") as handle:
    canary_manifest = json.load(handle)
with open(production_manifest_path, encoding="utf-8") as handle:
    production_manifest = json.load(handle)
if audit.get("manifest") != canary_manifest_path or audit.get("manifest_sha256") != hashlib.sha256(canary_bytes).hexdigest():
    raise SystemExit("Canary audit is not bound to CANARY_STAGE2_MANIFEST.")
if canary_manifest.get("method") != method or int(canary_manifest.get("stage", -1)) != 2:
    raise SystemExit("Audited canary manifest method/stage mismatch.")
if audit.get("source_code") != canary_manifest.get("source_code") or audit.get("source_code") != production_manifest.get("source_code"):
    raise SystemExit("Canary audit source lock differs from production campaign.")
PY
else
    CANARY_AUDIT_JSON=""
    CANARY_STAGE2_MANIFEST=""
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

submission_dir="${VFI_RESULTS_ROOT}/runs/submissions/gail_airl_final_${METHOD}_${LAUNCH_PROFILE}_${RUN_STAMP}"
if [ -e "${submission_dir}" ]; then
    echo "Refusing to reuse submission directory: ${submission_dir}" >&2
    exit 3
fi
slurm_log_root="${VFI_LOG_ROOT}/slurm/gail_airl_final_${METHOD}_${LAUNCH_PROFILE}_${RUN_STAMP}"
mkdir -p "${submission_dir}" "${slurm_log_root}"

runner="${REPODIR}/hpc/slurm/script_full_training/run_gail_airl_method_array.bash"
array_spec="0-$((trial_count - 1))%${concurrency}"
source_revision="$(git -C "${REPODIR}" rev-parse HEAD)"

export_base="ALL,REPODIR=${REPODIR},METHOD=${METHOD},PYTHON_BIN=${PYTHON_BIN}"
export_base="${export_base},VFI_PROJECT_ROOT=${VFI_PROJECT_ROOT},VFI_DATA_ROOT=${VFI_DATA_ROOT},VFI_HIGHWAY_DATA_ROOT=${VFI_HIGHWAY_DATA_ROOT}"
export_base="${export_base},VFI_CHECKPOINT_ROOT=${VFI_CHECKPOINT_ROOT},VFI_RESULTS_ROOT=${VFI_RESULTS_ROOT},VFI_LOG_ROOT=${VFI_LOG_ROOT},VFI_ARTIFACT_ROOT=${VFI_ARTIFACT_ROOT}"

stage1_command=(
    sbatch --parsable --array="${array_spec}" --chdir="${REPODIR}"
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}" --signal=B:USR1@1800
    --job-name="${METHOD}_${LAUNCH_PROFILE}_s1"
    --export="${export_base},STUDY_MANIFEST=${STAGE1_MANIFEST}"
    --output="${slurm_log_root}/stage1_%A_%a.out" --error="${slurm_log_root}/stage1_%A_%a.err"
    "${runner}"
)

echo "method=${METHOD} profile=${LAUNCH_PROFILE} simulator=${SIMULATOR_PROFILE} trials=${trial_count}"
echo "canonical_source=${REPODIR}@${source_revision}"
echo "blocking_job_states=${blocking_states} overlap_override=${ALLOW_BLOCKING_JOB_OVERLAP}"

stage1_job_id="DRYRUN_STAGE1"
stage2_job_ids=""
if [ "${DRY_RUN}" = "true" ]; then
    printf '%q ' "${stage1_command[@]}"
    printf '\n'
else
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch was not found on PATH." >&2
        exit 127
    fi
    stage1_result="$("${stage1_command[@]}")"
    stage1_job_id="${stage1_result%%;*}"
fi

if [ "${DEPENDENCY_MODE}" = "aftercorr" ]; then
    stage2_command=(
        sbatch --parsable --array="${array_spec}" --dependency="aftercorr:${stage1_job_id}"
        --chdir="${REPODIR}" --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}"
        --signal=B:USR1@1800 --job-name="${METHOD}_${LAUNCH_PROFILE}_s2"
        --export="${export_base},STUDY_MANIFEST=${STAGE2_MANIFEST}"
        --output="${slurm_log_root}/stage2_%A_%a.out" --error="${slurm_log_root}/stage2_%A_%a.err"
        "${runner}"
    )
    if [ "${DRY_RUN}" = "true" ]; then
        printf '%q ' "${stage2_command[@]}"
        printf '\n'
        stage2_job_ids="DRYRUN_STAGE2_AFTERCORR"
    else
        stage2_result="$("${stage2_command[@]}")"
        stage2_job_ids="${stage2_result%%;*}"
    fi
else
    for ((task_index=0; task_index<trial_count; task_index++)); do
        dependency="afterok:${stage1_job_id}_${task_index}"
        stage2_command=(
            sbatch --parsable --array="${task_index}-${task_index}" --dependency="${dependency}"
            --chdir="${REPODIR}" --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}"
            --signal=B:USR1@1800 --job-name="${METHOD}_${LAUNCH_PROFILE}_s2_${task_index}"
            --export="${export_base},STUDY_MANIFEST=${STAGE2_MANIFEST}"
            --output="${slurm_log_root}/stage2_%A_%a.out" --error="${slurm_log_root}/stage2_%A_%a.err"
            "${runner}"
        )
        if [ "${DRY_RUN}" = "true" ]; then
            printf '%q ' "${stage2_command[@]}"
            printf '\n'
            result="DRYRUN_STAGE2_${task_index}"
        else
            submitted="$("${stage2_command[@]}")"
            result="${submitted%%;*}"
        fi
        stage2_job_ids="${stage2_job_ids}${stage2_job_ids:+,}${result}"
    done
fi

export METHOD LAUNCH_PROFILE SIMULATOR_PROFILE STAGE1_MANIFEST STAGE2_MANIFEST BC_AUDIT_JSON
export CANARY_AUDIT_JSON BENCHMARK_EVIDENCE_JSON MIN_SIMULATOR_SPEEDUP BLOCKING_JOB_IDS blocking_states
export CANARY_STAGE2_MANIFEST
export ALLOW_BLOCKING_JOB_OVERLAP CPUS_PER_TASK MEMORY_PER_TASK NUM_ROLLOUT_WORKERS
export ROLLOUT_WORKER_THREADS EVALUATION_NUM_WORKERS EVALUATION_WORKER_THREADS
export DEPENDENCY_MODE source_revision stage1_job_id stage2_job_ids DRY_RUN
"${PYTHON_BIN}" - "${submission_dir}/submission_metadata.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

states = []
for item in filter(None, os.environ.get("blocking_states", "").split(";")):
    job_id, state = item.split("=", 1)
    states.append({"job_id": job_id, "state": state})
payload = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "method": os.environ["METHOD"],
    "launch_profile": os.environ["LAUNCH_PROFILE"],
    "simulator_profile": os.environ["SIMULATOR_PROFILE"],
    "stage1_manifest": os.environ["STAGE1_MANIFEST"],
    "stage2_manifest": os.environ["STAGE2_MANIFEST"],
    "source_revision": os.environ["source_revision"],
    "bc_audit": os.environ["BC_AUDIT_JSON"],
    "canary_audit": os.environ.get("CANARY_AUDIT_JSON") or None,
    "benchmark_evidence": os.environ.get("BENCHMARK_EVIDENCE_JSON") or None,
    "minimum_simulator_speedup": float(os.environ["MIN_SIMULATOR_SPEEDUP"]),
    "blocking_jobs_checked": states,
    "blocking_overlap_override": os.environ["ALLOW_BLOCKING_JOB_OVERLAP"] == "true",
    "resources": {
        "cpus_per_task": int(os.environ["CPUS_PER_TASK"]),
        "memory_per_task": os.environ["MEMORY_PER_TASK"],
        "num_rollout_workers": int(os.environ["NUM_ROLLOUT_WORKERS"]),
        "rollout_worker_threads": int(os.environ["ROLLOUT_WORKER_THREADS"]),
        "evaluation_num_workers": int(os.environ["EVALUATION_NUM_WORKERS"]),
        "evaluation_worker_threads": int(os.environ["EVALUATION_WORKER_THREADS"]),
    },
    "dependency_mode": os.environ["DEPENDENCY_MODE"],
    "stage1_job_id": os.environ["stage1_job_id"],
    "stage2_job_ids": os.environ["stage2_job_ids"].split(","),
    "dry_run": os.environ["DRY_RUN"] == "true",
    "signal_contract": "SIGUSR1 1800 seconds before limit; exact checkpoint at next completed round; exit 99 blocks dependent stage",
}
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "stage1_job=${stage1_job_id}"
echo "stage2_jobs=${stage2_job_ids}"
echo "submission_metadata=${submission_dir}/submission_metadata.json"
