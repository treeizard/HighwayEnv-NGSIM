#!/usr/bin/env python3
"""Build a hash-locked US GAIL/AIRL pilot manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts_gail.ps_gail.pilot import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expert-data",
        type=Path,
        help=(
            "Explicit US training expert root. Omit only when reproducing the "
            "legacy pilot."
        ),
    )
    parser.add_argument(
        "--bc-root",
        type=Path,
        help=(
            "Explicit US BC policy root containing recurrent_transformer_* "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--policy-recipe",
        type=Path,
        help="Locked BC recipe that defines the shared BC/GAIL actor.",
    )
    parser.add_argument(
        "--shared-policy-seed",
        type=int,
        help="Use one paired initialization seed for both transformer depths.",
    )
    parser.add_argument(
        "--require-explicit-data-contracts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--gail-only",
        action="store_true",
        help="Build only the two US GAIL depth trials.",
    )
    parser.add_argument(
        "--no-bc-initialization",
        action="store_true",
        help="Start policies from their deterministic random initialization.",
    )
    parser.add_argument("--num-rollout-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest(
        repo=args.repo,
        project_root=args.project_root,
        run_root=args.run_root,
        campaign_id=args.campaign_id,
        methods=("gail",) if args.gail_only else ("gail", "airl"),
        use_bc_initialization=not args.no_bc_initialization,
        num_rollout_workers=args.num_rollout_workers,
        expert_data=args.expert_data,
        bc_root=args.bc_root,
        policy_recipe=args.policy_recipe,
        shared_policy_seed=args.shared_policy_seed,
        require_explicit_data_contracts=bool(
            args.require_explicit_data_contracts
        ),
    )
    write_manifest(args.output, payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
