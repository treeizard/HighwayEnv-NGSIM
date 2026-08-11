# Expert Dataset Pipeline

Status: component reference. For current four-method work, the parent
repository's `docs/plan/expert_replay_bc_recovery_20260808.md` is authoritative.

This module turns processed NGSIM trajectory replays into saved expert
**transition** datasets, not action-only tables.
Instead of reading low-level CSV action tables directly, it replays the real trajectories through
`NGSimEnv` and records the observations and expert actions that the simulator actually uses.

That gives us:

- observations in the same format used everywhere else in the repo
- actions aligned with the environment's discrete or continuous control interface
- episode metadata that is still rich enough for replay, debugging, and scene-level methods

## Main Entry Points

- `highway_env/imitation/expert_dataset.py`
  Canonical implementation for dataset collection, validation, metadata loading, and PyTorch datasets.
- `src/policy/data/collect_expert.py` in the parent repository
  Current contract-aware collection CLI.
- `src/policy/data/audit_expert.py` in the parent repository
  Independent collection and split audit.

## Data Source

The pipeline uses `NGSimEnv` expert replay mode with processed trajectories under:

`data/highway_env/processed_20s/<scene>/prebuilt/`

It relies on two existing properties of the environment:

- `NGSimEnv` already knows how to load processed replay episodes from the repo's prebuilt files
- `NGSimEnv.step()` exposes the internally applied expert action in `info`

This makes the saved dataset match the simulator's actual observation and action conventions.

The dependency graph is branched:

```text
raw trajectories -> cleaned/windowed processed_20s replay episodes
                 -> expert replay -> expert transition datasets

processed_20s replay episodes -> online rollouts and closed-loop evaluation
```

BC consumes supervised observation/action pairs. IQ-Learn consumes expert
transitions plus online simulator interactions. GAIL consumes expert occupancy
features plus generator rollouts. AIRL consumes full expert transitions plus
generator rollouts. Sharing a transition corpus does not authorize sharing a
trained BC model across these methods.

## Dataset Modes

Two save formats are supported.

### `per_vehicle`

Each saved dataset episode corresponds to one controlled vehicle rollout.
If multiple expert vehicles are replayed together, they are still written as separate saved episodes
that share the same `scenario_id`.

Saved top-level fields:

- `episode_id`: `int32 [E]`
- `scenario_id`: `object [E]`
- `episode_name`: `object [E]`
- `ego_id`: `int32 [E]`
- `source_split`: `object [E]`
- `observations`: `object [E]`, each item `float32 [T, *obs_shape]`
- `actions`: `object [E]`
- `next_observations`: `object [E]`, each item `float32 [T, *obs_shape]`
- `dones`: `object [E]`, each item `bool [T]`
- `rewards`: `object [E]`, each item `float32 [T]`
- `timesteps`: `object [E]`, each item `int32 [T]`
- `metadata_json`: JSON string with dataset-level metadata

Action storage:

- discrete mode: `int64 [T]`
- continuous mode: `float32 [T, *action_shape]`

### `scene`

Each saved dataset episode corresponds to one full traffic segment with all controlled vehicles kept
together. This is the right fit for multi-agent or PS-GAIL-style demonstrations.

Saved top-level fields:

- `episode_id`: `int32 [E]`
- `scenario_id`: `object [E]`
- `episode_name`: `object [E]`
- `agent_ids`: `object [E]`, each item `int32 [N]`
- `source_split`: `object [E]`
- `observations`: `object [E]`, each item `float32 [T, N, *obs_shape]`
- `actions`: `object [E]`
- `next_observations`: `object [E]`, each item `float32 [T, N, *obs_shape]`
- `dones`: `object [E]`, each item `bool [T]`
- `rewards`: `object [E]`, each item `float32 [T]`
- `timesteps`: `object [E]`, each item `int32 [T]`
- `alive_mask`: `object [E]`, each item `bool [T, N]`
- `metadata_json`: JSON string with dataset-level metadata

Action storage:

- discrete mode: `int64 [T, N]`
- continuous mode: `float32 [T, N, *action_shape]`

## Observation and Action Conventions

The component's legacy bare default observation is lidar `(128, 2)`. The
current four-method actor does not use that 256D flattening directly. Its v1
policy projection is 322D:

- vehicle lidar: `128 x 2 = 256`;
- lane camera: `21 x 3 = 63`; and
- ego fields: `[length_m, speed_mps, heading_rad] = 3`.

The component retains historical discrete support. Current four-method
comparisons use only:

- `continuous`: `ContinuousAction`, normalized `float32 [2]` stored as
  `[acceleration_norm, steering_norm]`

For NGSIM continuous control, the normalized action interval is always
`[-1, 1]`. `acceleration_norm=-1` maps to `-5 m/s^2`,
`acceleration_norm=1` maps to `5 m/s^2`, and `steering_norm` maps to
`[-pi/4, pi/4]` radians. Expert files produced for action-conditioned
GAIL/AIRL should also contain `actions_continuous_env` with these normalized
columns and, when available, `actions_steering_acceleration` with physical
`[steering_rad, acceleration_mps2]` columns. Loaders reject non-finite
continuous expert arrays and normalized continuous actions outside `[-1, 1]`.
The exact metadata stored with each dataset remains authoritative; a scale or
column mismatch must fail rather than be converted. The current recovery
forbids yaw-rate surrogate labels and steering/yaw adapters.

## Build a Dataset

Current production collection is launched through a source-locked `StudySpec`.
For a bounded local receipt test only, use the current module and explicit
expert-replay controls:

```bash
python -m policy.data.collect_expert \
  --scene us-101 \
  --prebuilt-split train \
  --episode-name t1118849739700 \
  --control-all-vehicles \
  --expert-control-mode continuous \
  --trajectory-state-source simulated \
  --no-allow-idm \
  --max-episodes 1 \
  --max-steps-per-episode 20 \
  --out /tmp/ngsim_expert_replay_receipt
```

This command is engineering-only and does not replace the active recovery's
dual-domain local gates. Useful CLI options include:

- `--episode-root`
  Override the processed trajectory root.
- `--prebuilt-split train|val`
  Choose which prebuilt split to sample from.
- `--episode-name`
  Restrict collection to one replay episode.
- `--max-steps-per-episode`
  Cap the number of collected steps per scenario.
- `--control-all-vehicles`
  Control every valid vehicle in the selected traffic segment.
- `--max-surrounding`
  Limit how many replay vehicles are spawned as context.

## Inspect a Saved Dataset

Use `python -m policy.data.audit_expert --help` for the independent collection
audit, or load a single file through the dataset classes below. Do not rely on
the removed `scripts_ngsim/inspect_expert_dataset.py` prototype.

## Load from Python

Use `ExpertTransitionDataset` for standard per-transition supervised or adversarial imitation learning:

```python
from highway_env.imitation import ExpertTransitionDataset

dataset = ExpertTransitionDataset(
    "expert_data/ngsim_expert_dataset_discrete.npz",
    flatten_observations=True,
)

sample = dataset[0]
print(sample["observation"].shape)
print(sample["action"])
```

Use `SceneTransitionDataset` when the saved dataset was built in `scene` mode:

```python
from highway_env.imitation import SceneTransitionDataset

scene_dataset = SceneTransitionDataset(
    "expert_data/ngsim_expert_scene_dataset_discrete.npz",
    flatten_observations=True,
)

sample = scene_dataset[0]
print(sample["agent_id"], sample["observation"].shape)
```

## Scene-Mode Notes

`scene` mode keeps interacting controlled vehicles in one trajectory, so it is the most natural
format for methods that need scene-level coordination.

Practical notes:

- `--control-all-vehicles` automatically disables surrounding replay vehicles because the controlled
  set already covers the relevant scene participants
- dense scenes are still expensive with `LidarObservation`, because each controlled vehicle computes
  its own local observation
- `--episode-name` and `--max-horizon` are the fastest way to debug scene collection before scaling up

## Assumptions

- the dataset root already contains repo-compatible processed trajectory files
- expert actions come from `NGSimEnv` expert replay, not from a learned policy
- rewards are saved when available; in the current `NGSimEnv` pipeline they are usually `0.0`
