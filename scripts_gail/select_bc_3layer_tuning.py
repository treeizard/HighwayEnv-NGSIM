"""Rank three-layer BC tuning trials and enforce anti-collapse promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing BC tuning summary: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("transformer_layers", -1)) != 3:
        raise ValueError(f"Tuning trial is not a 3-layer policy: {path}")
    return payload


def trial_record(path: Path, *, trial_id: str) -> dict[str, Any]:
    summary = read_summary(path)
    learning_gate = summary.get("learning_signal_gate") or {}
    return {
        "trial_id": trial_id,
        "summary": str(path.resolve()),
        "domain": summary.get("domain"),
        "seed": summary.get("seed"),
        "learning_rate": summary.get("learning_rate"),
        "max_grad_norm": summary.get("max_grad_norm"),
        "transformer_dropout": summary.get("transformer_dropout"),
        "validation_skill": summary.get("validation_skill"),
        "validation_mae": summary.get("validation_mae"),
        "learning_action_std_ratio": learning_gate.get("prediction_std_ratio"),
        "learning_action_correlation": learning_gate.get("prediction_target_correlation"),
        "learning_signal_passed": bool(summary.get("learning_signal_passed")),
        "metric_capability_passed": bool(summary.get("metric_capability_passed")),
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
    }


def select_tuning_candidate(
    screen_root: Path,
    *,
    expected_candidates: list[str],
    confirmation_root: Path | None = None,
    expected_confirmations: list[str] | None = None,
) -> dict[str, Any]:
    candidates = [
        trial_record(screen_root / candidate / "summary.json", trial_id=candidate)
        for candidate in expected_candidates
    ]
    passing = [candidate for candidate in candidates if candidate["metric_capability_passed"]]
    passing.sort(
        key=lambda candidate: (
            float(candidate["validation_skill"]),
            float(candidate["learning_action_correlation"]),
            float(candidate["learning_action_std_ratio"]),
        ),
        reverse=True,
    )
    selected = passing[0] if passing else None
    result: dict[str, Any] = {
        "schema_version": 1,
        "study": "bc_3layer_optimization_recovery_v1",
        "screen_root": str(screen_root.resolve()),
        "candidate_count": len(candidates),
        "screen_pass_count": len(passing),
        "screen_status": "selected" if selected is not None else "no_candidate_passed",
        "selected_candidate": selected,
        "candidates": candidates,
        "promotion_status": "pending_confirmation" if selected is not None else "rejected",
    }

    if confirmation_root is not None:
        if selected is None:
            raise ValueError("Cannot evaluate confirmations when no screen candidate passed.")
        confirmation_ids = list(expected_confirmations or [])
        confirmations = [
            trial_record(confirmation_root / confirmation_id / "summary.json", trial_id=confirmation_id)
            for confirmation_id in confirmation_ids
        ]
        confirmation_pass_count = sum(
            int(confirmation["metric_capability_passed"]) for confirmation in confirmations
        )
        promoted = bool(confirmations) and confirmation_pass_count == len(confirmations)
        result.update(
            {
                "confirmation_root": str(confirmation_root.resolve()),
                "confirmation_count": len(confirmations),
                "confirmation_pass_count": confirmation_pass_count,
                "confirmations": confirmations,
                "promotion_status": "promoted" if promoted else "rejected",
            }
        )
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", dest="candidates", required=True)
    parser.add_argument("--confirmation-root", type=Path)
    parser.add_argument("--confirmation", action="append", dest="confirmations", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--print-selected", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = select_tuning_candidate(
        args.screen_root,
        expected_candidates=list(args.candidates),
        confirmation_root=args.confirmation_root,
        expected_confirmations=list(args.confirmations),
    )
    write_json(args.out, result)
    if args.print_selected:
        selected = result.get("selected_candidate") or {}
        print(selected.get("trial_id") or "NONE")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
