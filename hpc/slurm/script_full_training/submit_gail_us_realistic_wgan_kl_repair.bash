#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="${REPODIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
: "${VFI_PROJECT_ROOT:?Export the canonical validation_first_interpretability project root}"

CAMPAIGN_ID="${CAMPAIGN_ID:-gail_us_realistic_wgan_klrepair_$(date -u +%Y%m%dT%H%M%SZ)}"
REPAIR_OF_CAMPAIGN_ID="${REPAIR_OF_CAMPAIGN_ID:-gail_us_realistic_wgan_20260727_prod01}"
REPAIR_OF_DEPTH2_JOB="${REPAIR_OF_DEPTH2_JOB:-58575841}"
REPAIR_OF_DEPTH3_JOB="${REPAIR_OF_DEPTH3_JOB:-58575842}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/bt60/ytao0016/conda/envs/ngsim_env/bin/python}"

case "${REPAIR_OF_CAMPAIGN_ID}" in
    *[!A-Za-z0-9._-]*|'')
        echo "Unsafe REPAIR_OF_CAMPAIGN_ID: ${REPAIR_OF_CAMPAIGN_ID}" >&2
        exit 2
        ;;
esac
case "${REPAIR_OF_DEPTH2_JOB}:${REPAIR_OF_DEPTH3_JOB}" in
    *[!0-9:]*|:*|*:)
        echo "Repair job IDs must be positive integers." >&2
        exit 2
        ;;
esac

export REPODIR VFI_PROJECT_ROOT CAMPAIGN_ID PYTHON_BIN
"${SCRIPT_DIR}/submit_gail_us_realistic_wgan.bash"

if [ "${DRY_RUN:-false}" = true ]; then
    echo "repair_provenance=not_created_in_dry_run"
    exit 0
fi

submission_dir="${VFI_PROJECT_ROOT}/results/runs/submissions/${CAMPAIGN_ID}"
repair_manifest="${submission_dir}/repair_provenance.json"
export repair_manifest REPAIR_OF_CAMPAIGN_ID
export REPAIR_OF_DEPTH2_JOB REPAIR_OF_DEPTH3_JOB
"${PYTHON_BIN}" - <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

repo = Path(os.environ["REPODIR"]).resolve()
payload = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "repair_of": {
        "campaign_id": os.environ["REPAIR_OF_CAMPAIGN_ID"],
        "jobs": {
            "gail_depth2": os.environ["REPAIR_OF_DEPTH2_JOB"],
            "gail_depth3": os.environ["REPAIR_OF_DEPTH3_JOB"],
        },
        "failure_reason": "target_kl_repeatedly_exceeded",
    },
    "repair": {
        "scope": "training-health classification and durable KL diagnostics only",
        "finite_repeated_target_kl": "recoverable_warning",
        "within_update_response": "existing PPO early stop remains enabled",
        "nonfinite_metrics": "fatal",
        "post_update_kl_persisted": True,
        "consecutive_kl_violation_count_persisted": True,
        "optimizer_hyperparameters_changed": False,
        "policy_architecture_changed": False,
        "data_contract_changed": False,
    },
    "source": {
        "repo": str(repo),
        "revision": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "status_clean": not subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
        ).strip(),
    },
    "claim_boundary": (
        "resubmission is pending until terminal scheduler, checkpoint, "
        "validation, and held-out evaluation gates pass"
    ),
}
Path(os.environ["repair_manifest"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "repair_provenance=${repair_manifest}"
