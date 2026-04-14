# Expert Dataset Pipeline

This module turns processed NGSIM trajectory replays into saved imitation-learning datasets.
Instead of reading low-level CSV action tables directly, it replays the real trajectories through
`NGSimEnv` and records the observations and expert actions that the simulator actually uses.

That gives us:

- observations in the same format used everywhere else in the repo
- actions aligned with the environment's discrete or continuous control interface
- episode metadata that is still rich enough for replay, debugging, and scene-level methods

## Main Entry Points

- `highway_env/imitation/expert_dataset.py`
  Canonical implementation for dataset collection, validation, metadata loading, and PyTorch datasets.
- `scripts_ngsim/build_expert_dataset.py`
  CLI wrapper for building a dataset from processed replay episodes.
- `scripts_ngsim/inspect_expert_dataset.py`
  CLI tool for checking dataset stats and printing a replay command for one saved episode.

## Data Source

The pipeline uses `NGSimEnv` expert replay mode with processed trajectories under:

`highway_env/data/processed_20s/<scene>/prebuilt/`

It relies on two existing properties of the environment:

- `NGSimEnv` already knows how to load processed replay episodes from the repo's prebuilt files
- `NGSimEnv.step()` exposes the internally applied expert action in `info`

This makes the saved dataset match the simulator's actual observation and action conventions.

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

Default observation:

- `LidarObservation`
- shape `(128, 2)` before batching
- commonly flattened to `256` features for MLP-style imitation baselines

Supported action modes:

- `discrete`: `DiscreteSteerMetaAction`, scalar action in `{0, 1, 2, 3, 4}`
- `continuous`: `ContinuousAction`, normalized `float32 [2]` for acceleration and steering

## Build a Dataset

Example single-vehicle discrete dataset:

```bash
python scripts_ngsim/build_expert_dataset.py \
  --scene us-101 \
  --action-mode discrete \
  --episodes 32 \
  --out expert_data/ngsim_expert_dataset_discrete.npz
```

Example single-vehicle continuous dataset:

```bash
python scripts_ngsim/build_expert_dataset.py \
  --scene us-101 \
  --action-mode continuous \
  --episodes 32 \
  --out expert_data/ngsim_expert_dataset_continuous.npz
```

Example full-scene discrete dataset:

```bash
python scripts_ngsim/build_expert_dataset.py \
  --scene us-101 \
  --prebuilt-split train \
  --dataset-mode scene \
  --control-all-vehicles \
  --action-mode discrete \
  --episodes 32 \
  --out expert_data/ngsim_expert_scene_dataset_discrete.npz
```

Example targeted debug run on one known episode:

```bash
python scripts_ngsim/build_expert_dataset.py \
  --scene us-101 \
  --prebuilt-split train \
  --episode-name t1118849739700 \
  --dataset-mode scene \
  --control-all-vehicles \
  --action-mode discrete \
  --episodes 1 \
  --max-horizon 20 \
  --out /tmp/ngsim_scene_debug.npz
```

Useful CLI options:

- `--episode-root`
  Override the processed trajectory root.
- `--prebuilt-split train|val`
  Choose which prebuilt split to sample from.
- `--episode-name`
  Restrict collection to one replay episode.
- `--max-horizon`
  Cap the number of collected steps per scenario.
- `--controlled-vehicles`
  Replay a fixed number of expert-controlled vehicles together.
- `--control-all-vehicles`
  Control every valid vehicle in the selected traffic segment.
- `--dataset-mode scene`
  Save one scene-level sequence instead of separate per-vehicle rollouts.
- `--max-surrounding`
  Limit how many replay vehicles are spawned as context.

## Inspect a Saved Dataset

```bash
python scripts_ngsim/inspect_expert_dataset.py expert_data/ngsim_expert_dataset_discrete.npz
```

The inspector prints:

- number of saved dataset episodes
- total transitions
- trajectory length statistics
- observation and action shapes
- a few sample transitions
- a ready-to-run replay command for the first recorded episode

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
