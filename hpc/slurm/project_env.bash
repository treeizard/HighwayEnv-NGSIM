#!/bin/bash

# Resolve generated state from the parent interpretability project when this
# fork is checked out as a submodule, while remaining usable standalone.
if [ -z "${VFI_PROJECT_ROOT:-}" ]; then
    candidate_root="$(cd "${REPODIR}/../.." 2>/dev/null && pwd || true)"
    if [ -f "${candidate_root}/pyproject.toml" ] \
        && grep -q "validation-first-interpretability" "${candidate_root}/pyproject.toml"; then
        VFI_PROJECT_ROOT="${candidate_root}"
    else
        VFI_PROJECT_ROOT="${REPODIR}"
    fi
fi

export VFI_PROJECT_ROOT
export VFI_DATA_ROOT="${VFI_DATA_ROOT:-${VFI_PROJECT_ROOT}/data}"
export VFI_HIGHWAY_DATA_ROOT="${VFI_HIGHWAY_DATA_ROOT:-${VFI_DATA_ROOT}/highway_env}"
export VFI_CHECKPOINT_ROOT="${VFI_CHECKPOINT_ROOT:-${VFI_DATA_ROOT}/checkpoints}"
export VFI_LOG_ROOT="${VFI_LOG_ROOT:-${VFI_PROJECT_ROOT}/logs}"
export VFI_ARTIFACT_ROOT="${VFI_ARTIFACT_ROOT:-${VFI_PROJECT_ROOT}/artifacts}"
