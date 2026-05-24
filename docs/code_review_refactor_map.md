# Code Review and Refactor Map

This document records the conservative cleanup standard for the NGSIM fork. It
keeps the first pass focused on maintainability while preserving existing import
paths, script entrypoints, and experiment commands.

## Documentation Standard

- Every Python module in `highway_env` and `scripts_gail/ps_gail` should have a
  short module docstring that explains the file's purpose.
- Public classes and public functions should have concise docstrings when their
  behavior is part of the package, experiment, or test surface.
- Private helpers should be documented only when the implementation is
  non-obvious, performance-sensitive, or encodes a research assumption.
- Comments should explain intent, invariants, and edge cases. Avoid comments that
  only restate the code.
- Sphinx-facing docstrings should remain compatible with the existing Napoleon
  configuration in `docs/conf.py`.

## Completed Conservative Split

- `scripts_gail/ps_gail/trainer.py` is now a compatibility facade. The
  implementation is grouped under `scripts_gail/ps_gail/training/` by data
  containers, Torch helpers, policy helpers, evaluation, reward shaping, rollout
  collection, discriminator updates, and PPO updates.
- `highway_env/envs/common/observation.py` is now a compatibility facade. The
  implementation is grouped under `highway_env/envs/common/observations/` by
  shared primitives, classic observations, lidar tracing, camera observations,
  and the observation factory.
- Existing callers should continue importing from the historical modules unless
  they are intentionally moving to the new internal layout.

## Deferred Rename Map

No physical folder or file renames are part of this pass. Use this map for a
later rename PR after tests and experiment scripts are stable.

| Current name | Candidate name | Compatibility requirement |
| --- | --- | --- |
| `slurum` | `slurm` | Keep forwarding docs or duplicate scheduler entrypoints until cluster scripts are updated. |
| `scripts_gail` | `scripts_imitation` or `experiments/imitation` | Keep old script paths as wrappers for active runbooks and checkpoints. |
| `scripts_env_test` | `scripts_env_diagnostics` | Preserve command examples in READMEs and local notes before removing old paths. |
| `scripts_setup` | `scripts_data_setup` | Update dataset setup docs and any SLURM/local scripts that call setup utilities. |
| `train_simple_*` | Names without `simple` | Rename only after each script's role is documented as a full workflow, smoke test, or diagnostic. |

## Review Checklist

- Confirm imports still work from compatibility facades before updating callers.
- Run `python -m compileall highway_env scripts_gail` after mechanical moves.
- Run the targeted PS-GAIL and observation tests after touching the split modules.
- Check generated docs or Sphinx references for public classes moved behind
  compatibility facades.
- Record any future physical rename in this map before applying it.
