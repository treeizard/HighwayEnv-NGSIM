#!/usr/bin/env python3
"""Exercise depth-2/depth-3 policy learning and reload on one local GPU."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts_gail.policy_api import (
    load_policy_bundle,
    make_actor_critic,
    runtime_continuous_action_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def _config(depth: int) -> dict[str, object]:
    return {
        "policy_model": "recurrent_transformer",
        "action_mode": "continuous",
        "continuous_action_dim": 2,
        "hidden_size": 64,
        "transformer_layers": int(depth),
        "transformer_heads": 4,
        "transformer_dropout": 0.0,
        "transformer_norm_first": True,
        "transformer_observation_normalization": True,
        "transformer_observation_tokenization": "dense_temporal",
        "policy_head_init_std": 0.01,
        "transformer_temporal_module": False,
        "transformer_memory_tokens": 1,
        "transformer_memory_context_length": 8,
        "transformer_use_causal_attention": True,
        "centralized_critic": False,
    }


def _build(config: dict[str, object], device: torch.device) -> torch.nn.Module:
    return make_actor_critic(
        str(config["policy_model"]),
        obs_dim=322,
        hidden_size=int(config["hidden_size"]),
        action_mode=str(config["action_mode"]),
        continuous_action_dim=int(config["continuous_action_dim"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        transformer_dropout=float(config["transformer_dropout"]),
        transformer_norm_first=bool(config["transformer_norm_first"]),
        transformer_observation_normalization=bool(
            config["transformer_observation_normalization"]
        ),
        transformer_observation_tokenization=str(
            config["transformer_observation_tokenization"]
        ),
        policy_head_init_std=float(config["policy_head_init_std"]),
        transformer_temporal_module=bool(
            config["transformer_temporal_module"]
        ),
        transformer_memory_tokens=int(config["transformer_memory_tokens"]),
        transformer_memory_context_length=int(
            config["transformer_memory_context_length"]
        ),
        transformer_use_causal_attention=bool(
            config["transformer_use_causal_attention"]
        ),
        centralized_critic=bool(config["centralized_critic"]),
    ).to(device)


def _run_depth(
    depth: int,
    *,
    output_root: Path,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed + depth)
    config = _config(depth)
    policy = _build(config, device)
    policy.train()
    observations = torch.randn(batch_size, 322, device=device)
    targets = torch.stack(
        (
            torch.tanh(0.7 * observations[:, 0] - 0.2 * observations[:, 2]),
            torch.tanh(0.6 * observations[:, 1] + 0.3 * observations[:, 3]),
        ),
        dim=-1,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-3)
    initial_parameters = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
    }
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = policy.actor(observations)
        loss = torch.nn.functional.mse_loss(predicted, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite depth-{depth} policy loss.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    policy.eval()
    with torch.no_grad():
        actions = policy.actor(observations)
    action_std = actions.std(dim=0)
    changed = any(
        not torch.equal(value.detach(), initial_parameters[name])
        for name, value in policy.named_parameters()
    )
    if not changed:
        raise RuntimeError(f"Depth-{depth} policy parameters did not update.")
    if not torch.all(torch.isfinite(actions)):
        raise RuntimeError(f"Depth-{depth} policy produced non-finite actions.")
    if torch.any(action_std <= 1.0e-6):
        raise RuntimeError(
            f"Depth-{depth} policy collapsed an action column: {action_std}."
        )
    if losses[-1] >= losses[0]:
        raise RuntimeError(
            f"Depth-{depth} supervised loss did not decrease: {losses}."
        )

    checkpoint_path = output_root / f"depth{depth}.pt"
    torch.save(
        {
            "checkpoint_kind": "local_policy_contract_gpu_smoke",
            "policy_state_dict": policy.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    reloaded = load_policy_bundle(checkpoint_path, device=device)
    with torch.no_grad():
        reloaded_actions = reloaded.policy.actor(observations)
    torch.testing.assert_close(reloaded_actions, actions, rtol=0.0, atol=0.0)
    return {
        "depth": depth,
        "steps": steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "action_std": [float(value) for value in action_std.cpu()],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": reloaded.checkpoint_sha256,
        "reload_exact": True,
        "parameters_changed": changed,
    }


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2.")
    if not torch.cuda.is_available():
        raise RuntimeError("Policy contract smoke requires a CUDA GPU.")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    output_root.mkdir(parents=True)
    device = torch.device("cuda")
    rows = [
        _run_depth(
            depth,
            output_root=output_root,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        for depth in (2, 3)
    ]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": "engineering_smoke",
        "scientific_claim_eligible": False,
        "passed": True,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "action_contract": runtime_continuous_action_contract(),
        "results": rows,
    }
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
