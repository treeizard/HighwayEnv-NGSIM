#!/usr/bin/env python3
"""Generate the pre-registered 48-run GAIL/AIRL screening study."""

from __future__ import annotations

import argparse
import json
import os

import torch

from scripts_gail.ps_gail.study import build_screening_trials, write_study_files


def parse_args() -> argparse.Namespace:
    data_root = os.environ.get("VFI_HIGHWAY_DATA_ROOT", "data/highway_env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert-data",
        default="expert_data/ngsim_ps_unified_expert_continuous_55145982",
    )
    parser.add_argument(
        "--episode-root",
        default=os.path.join(data_root, "processed_20s"),
    )
    parser.add_argument(
        "--bc-checkpoint",
        required=True,
        help="Short recurrent-BC stabilization checkpoint used by both methods.",
    )
    parser.add_argument(
        "--bc-summary",
        default="",
        help="BC summary.json; defaults to the checkpoint's sibling summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("results", "gail_airl_training_study"),
    )
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.bc_checkpoint):
        raise FileNotFoundError(
            "The shared recurrent-BC checkpoint must exist before generating the study: "
            f"{args.bc_checkpoint}"
        )
    bc_summary_path = args.bc_summary or os.path.join(
        os.path.dirname(os.path.abspath(args.bc_checkpoint)),
        "summary.json",
    )
    if not os.path.isfile(bc_summary_path):
        raise FileNotFoundError(f"BC validation summary does not exist: {bc_summary_path}")
    with open(bc_summary_path, encoding="utf-8") as handle:
        bc_summary = json.load(handle)
    bc_gate = {
        "checkpoint_purpose": bc_summary.get("checkpoint_purpose"),
        "checkpoint_saved": bool(bc_summary.get("checkpoint_saved", False)),
        "warm_start_passed": bool(bc_summary.get("warm_start_passed", False)),
        "configured_epochs": int(bc_summary.get("configured_epochs", 0)),
        "relative_validation_improvement": float(
            bc_summary.get("relative_validation_improvement", float("-inf"))
        ),
    }
    if not (
        bc_gate["checkpoint_purpose"] == "warm_start"
        and bc_gate["checkpoint_saved"]
        and bc_gate["warm_start_passed"]
        and 1 <= bc_gate["configured_epochs"] <= 5
        and bc_gate["relative_validation_improvement"] >= 0.01
    ):
        raise RuntimeError(f"Shared recurrent-BC warm start failed the study gate: {bc_gate}")
    try:
        bc_checkpoint = torch.load(
            args.bc_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        bc_checkpoint = torch.load(args.bc_checkpoint, map_location="cpu")
    bc_policy_config = dict(bc_checkpoint.get("config") or {})
    if bc_checkpoint.get("checkpoint_kind") != "behaviour_cloning_warm_start":
        raise RuntimeError(
            "Study initialization must use a deliberately short BC warm-start checkpoint, got "
            f"{bc_checkpoint.get('checkpoint_kind')!r}."
        )
    if bc_policy_config.get("policy_model") != "recurrent_transformer":
        raise RuntimeError(
            "Study warm start must be a recurrent_transformer checkpoint, got "
            f"{bc_policy_config.get('policy_model')!r}."
        )
    trials = build_screening_trials(
        expert_data=os.path.abspath(args.expert_data),
        episode_root=os.path.abspath(args.episode_root),
        bc_checkpoint=os.path.abspath(args.bc_checkpoint),
        bc_policy_config=bc_policy_config,
        rounds=max(1, int(args.rounds)),
    )
    paths = write_study_files(args.output_dir, trials, python=args.python)
    print(json.dumps({"trial_count": len(trials), "bc_gate": bc_gate, **paths}, indent=2))


if __name__ == "__main__":
    main()
