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

## June 2026 Review Notes

- Added monitoring-only steering diagnostics for continuous PS-GAIL under the
  existing Vendi diagnostics gate. These metrics compare expert and policy
  normalized steering summaries, fixed-bin histograms, steering-window Vendi,
  safe steering-window Vendi, and steering MMD. They do not alter rewards,
  losses, entropy schedules, action scaling, or rollout behavior.
- Documented that steering event rates are threshold proxies on
  `abs(steering_norm)`, not true lane-change rates. True lane-change rates need
  lane-index transition collection.
- Kept BC fine-tuning disabled in the GAIL five-day finetune path. The PPO BC
  code remains available when explicitly configured, and recurrent PPO now
  validates expert observation/action length the same way the non-recurrent path
  does.
- Fixed rollout subsampling to preserve recurrent policy-step memory row shape
  instead of collapsing multi-dimensional memory arrays through scalar fallback
  handling.
- Cleaned stale patch-note comments in the expert tracker/replay path, flattened
  lane references on ingest, and added validation for
  `PurePursuitTracker.ref_lanes` length before target-lane lookup.
- Added normalized continuous-action range validation so expert
  `actions_continuous_env` files must stay within the policy/env `[-1, 1]`
  acceleration and steering action interval.
- Added finite-array validation for action-conditioned expert observations,
  next observations, trajectory states, rewards, vehicle ids, and timesteps so
  corrupted expert files fail at load time instead of surfacing as unstable
  training metrics later.
- Bound `PurePursuitTracker` default acceleration clamps to the shared
  NGSIM `[-10, 10]` acceleration constants so expert tracking and action
  normalization no longer drift apart.
- Fixed `ContinuousAction` range handling so NumPy array
  `acceleration_range`, `steering_range`, and `speed_range` configs no longer
  fail truthiness checks during environment construction.
- Removed copied dead imports from the split PS-GAIL training package after AST
  import-use scans showed they were not used by the local modules.
- Removed stale commented temporal-module scaffolding from the primary
  GAIL/AIRL pretrain and finetune Slurm scripts; active temporal-transformer
  support remains available in config/model paths and auxiliary scripts.
- Runtime note: keep `VENDI_MAX_WINDOWS` positive for production diagnostics.
  The default `2048` caps Vendi and MMD work; `0` means no subsampling and can
  make kernel/eigenvalue diagnostics scale poorly on large rollouts.
- Steering MMD now avoids full kernel matrices by using chunked kernel means and
  a one-dimensional median-bandwidth selector for normalized steering samples.
  Exact self-comparisons reuse the same subsample, so identical steering inputs
  report zero MMD even when `VENDI_MAX_WINDOWS` caps the samples.
- Safe-policy steering MMD uses the same zero-penalty window semantics as safe
  steering Vendi before comparing policy steering samples against expert data,
  and safe steering diagnostics respect the existing `vendi_safe_only` gate.
- Aligned the direct GAIL/AIRL stage-two finetune script defaults with the
  five-day submitter intent: ramp to 100 controlled vehicles, 40k rollout target
  agent steps, and `gamma=0.99` by round 100, then hold through round 200 with a
  nonzero entropy floor and `2.0` collision/offroad penalties. The full
  submitters export the finetune schedule and safety-penalty values instead of
  relying on stale stage-script breakpoints.
- Added a parser-aware regression test that extracts literal and `*_ARG`
  training flags from the four primary GAIL/AIRL stage scripts and verifies
  they are accepted by the Python training entrypoints.
- Updated the GAIL methodology note to use stable module/function/class anchors
  instead of stale line-number references, and documented the current
  continuous-action expert validation plus monitoring-only steering diagnostics.
  A regression test now prevents `file.py:line` anchors from returning to that
  methodology note.
- Updated the expert-dataset README with the NGSIM continuous-action contract:
  normalized `[acceleration_norm, steering_norm]`, `[-10, 10]` m/s^2
  acceleration mapping, `[-pi/4, pi/4]` steering mapping, and loader rejection
  of non-finite or out-of-range continuous expert arrays. A regression test now
  checks that this contract remains documented.

Validation evidence from this pass:

- Full local suite in `ngsim_env`: `199 passed, 1 skipped, 3 warnings`.
- Focused PS-GAIL training logic: `94 passed`.
- Focused tracker/action range slice: `10 passed`.
- Focused action-loader/steering diagnostics slice: `23 passed, 68 deselected`.
- Focused Slurm CLI/schedule slice: `7 passed, 85 deselected`.
- Focused methodology-doc/Slurm slice: `7 passed, 86 deselected`.
- Focused expert-data/action-contract docs slice: `11 passed, 93 deselected`.
- Steering MMD timing at the default cap was about `0.125s` for 2048 samples per
  side on the local machine.
- Local lightweight quality checks covered YAML parsing, TOML parsing via
  `tomli`, Python compile checks, and AST import-use scans. The configured
  pre-commit tools (`black`, `isort`, `flake8`, and `pre-commit`) were not
  installed in `ngsim_env`.
- GAIL and AIRL full-training/stage bash syntax checks passed with Git Bash.
- Parser-aware Slurm CLI audit is now covered by regression test and found no
  unknown Python flags in the primary GAIL/AIRL pretrain and finetune stage
  scripts.
- GAIL and AIRL full-training submitters were dry-run locally with a fake
  `sbatch`; finetune jobs included dependencies, resume checkpoints, 200-round
  schedule exports, 100-vehicle hold, 40k rollout target hold, gamma schedules,
  nonzero entropy schedules, and safety-penalty exports.
- `git diff --check` passed; only Windows LF/CRLF warnings were reported.

Remaining scope not proven by this pass:

- The configured pre-commit stack was not run because the tools were absent from
  `ngsim_env`.
- SLURM scripts were syntax-checked and dry-run locally with a fake `sbatch`,
  but not submitted to a real cluster.
- AIRL can reuse the diagnostics helper later, but this pass intentionally wired
  steering diagnostics into GAIL first.
- Older TODO comments remain in unrelated upstream/simulation modules; they were
  not changed without a concrete failing test or runtime symptom.
