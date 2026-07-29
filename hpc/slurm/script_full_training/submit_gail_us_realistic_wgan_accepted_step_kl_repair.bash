#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
: "${VFI_PROJECT_ROOT:?Export the canonical validation_first_interpretability project root}"

CAMPAIGN_ID="${CAMPAIGN_ID:-gail_us_realistic_wgan_d3_accepted_kl_$(date -u +%Y%m%dT%H%M%SZ)}"
REPAIR_OF_FIRST_JOB="${REPAIR_OF_FIRST_JOB:-58575842}"
REPAIR_OF_SECOND_JOB="${REPAIR_OF_SECOND_JOB:-58581211}"
REPAIR_OF_THIRD_JOB="${REPAIR_OF_THIRD_JOB:-58603980}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/bt60/ytao0016/conda/envs/ngsim_env/bin/python}"

case "${REPAIR_OF_FIRST_JOB}:${REPAIR_OF_SECOND_JOB}:${REPAIR_OF_THIRD_JOB}" in
    *[!0-9:]*|:*|*:)
        echo "Repair job IDs must be positive integers." >&2
        exit 2
        ;;
esac

export REPODIR VFI_PROJECT_ROOT CAMPAIGN_ID PYTHON_BIN
export GAIL_DEPTHS=3
"${SCRIPT_DIR}/submit_gail_us_realistic_wgan.bash"

if [ "${DRY_RUN:-false}" = true ]; then
    echo "repair_provenance=not_created_in_dry_run"
    exit 0
fi

submission_dir="${VFI_PROJECT_ROOT}/results/runs/submissions/${CAMPAIGN_ID}"
repair_manifest="${submission_dir}/repair_provenance.json"
export repair_manifest
export REPAIR_OF_FIRST_JOB REPAIR_OF_SECOND_JOB REPAIR_OF_THIRD_JOB
"${PYTHON_BIN}" - <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

repo = Path(os.environ["REPODIR"]).resolve()
metadata_path = (
    Path(os.environ["VFI_PROJECT_ROOT"])
    / "results/runs/submissions"
    / os.environ["CAMPAIGN_ID"]
    / "submission_metadata.json"
)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
jobs = dict(metadata.get("jobs") or {})
if set(jobs) != {"gail_depth3"}:
    raise SystemExit(f"Depth-3 repair submitted an unexpected job set: {jobs}")
payload = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "submitted_job": jobs["gail_depth3"],
    "repair_of": [
        {
            "job_id": os.environ["REPAIR_OF_FIRST_JOB"],
            "terminal_reason": "target_kl_repeatedly_exceeded",
            "interpretation": "superseded finite-KL fatal classification",
        },
        {
            "job_id": os.environ["REPAIR_OF_SECOND_JOB"],
            "terminal_reason": "no_post_initialization_best_checkpoint",
            "interpretation": "epoch-level KL stop permitted oversize depth-3 updates",
        },
        {
            "job_id": os.environ["REPAIR_OF_THIRD_JOB"],
            "terminal_reason": "no_post_initialization_best_checkpoint",
            "interpretation": (
                "pre-step minibatch guard still accepted an over-target first "
                "optimizer step"
            ),
        },
    ],
    "repair": {
        "scope": "accepted PPO optimizer-step KL enforcement",
        "target_kl": 0.005,
        "previous_stop_granularity": "before the next minibatch optimizer step",
        "new_stop_granularity": "each proposed optimizer step before acceptance",
        "diagnostic_batch_is_stable_within_update": True,
        "optimizer_and_policy_state_restored_before_retry": True,
        "backtrack_factor": 0.5,
        "maximum_backtracks": 8,
        "fully_rejected_step_restores_pre_step_state": True,
        "fresh_bc_initialization": True,
        "resumes_regressed_round_100_state": False,
        "reruns_completed_depth2": False,
        "configured_optimizer_hyperparameters_changed": False,
        "policy_architecture_changed": False,
        "data_or_action_contract_changed": False,
        "learning_gate_relaxed": False,
    },
    "source": {
        "repo": str(repo),
        "revision": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "status_clean": not subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip(),
    },
    "claim_boundary": (
        "submission and local validation establish only the accepted-step "
        "control path; terminal learning, checkpoint, validation, and held-out "
        "qualification remain pending"
    ),
}
Path(os.environ["repair_manifest"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "repair_provenance=${repair_manifest}"
