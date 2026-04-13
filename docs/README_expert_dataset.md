# Expert Dataset Pipeline

## Source

The first expert dataset pipeline uses the repository's existing processed real trajectory data through `NGSimEnv` expert replay mode.

Why this source was chosen:

- `highway_env/envs/ngsim_env.py` already loads processed trajectory episodes from `highway_env/data/processed_20s/<scene>/prebuilt/`
- `NGSimEnv.step()` already exposes the internally applied expert action in `info`
- this produces observation/action pairs in the exact simulator convention used by the rest of the repo

This is preferable to reading lower-level CSV action tables directly because GAIL needs simulator observations as well as actions.

## Files

- `scripts_ngsim/build_expert_dataset.py`: builds a sample expert dataset
- `scripts_ngsim/inspect_expert_dataset.py`: prints dataset summary statistics and a replay command
- `highway_env/imitation/expert_dataset.py`: reusable collection and PyTorch loading utilities

## Saved Format

Datasets are stored as compressed `.npz` files with episode-preserving arrays.

Two dataset modes are supported:

- `per_vehicle`: one saved episode per controlled vehicle
- `scene`: one saved episode per full traffic segment, with all controlled vehicles stored together

Top-level fields:

- `episode_id`: `int32 [E]`
- `scenario_id`: `object [E]`
- `episode_name`: `object [E]`
- `ego_id`: `int32 [E]`
- `source_split`: `object [E]`
- `observations`: `object [E]`, each item is `float32 [T, *obs_shape]`
- `actions`: `object [E]`, each item is:
  - discrete mode: `int64 [T]`
  - continuous mode: `float32 [T, *action_shape]`
- `next_observations`: `object [E]`, each item is `float32 [T, *obs_shape]`
- `dones`: `object [E]`, each item is `bool [T]`
- `rewards`: `object [E]`, each item is `float32 [T]`
- `timesteps`: `object [E]`, each item is `int32 [T]`
- `metadata_json`: JSON string with dataset-level config and schema metadata

Each saved dataset episode corresponds to one controlled vehicle rollout. If multiple controlled vehicles are replayed together in the same simulator scenario, they are saved as separate episodes that share the same `scenario_id`.

In `scene` mode, the saved fields are:

- `episode_id`: `int32 [E]`
- `scenario_id`: `object [E]`
- `episode_name`: `object [E]`
- `agent_ids`: `object [E]`, each item is `int32 [N]`
- `source_split`: `object [E]`
- `observations`: `object [E]`, each item is `float32 [T, N, *obs_shape]`
- `actions`: `object [E]`, each item is:
  - discrete mode: `int64 [T, N]`
  - continuous mode: `float32 [T, N, *action_shape]`
- `next_observations`: `object [E]`, each item is `float32 [T, N, *obs_shape]`
- `dones`: `object [E]`, each item is `bool [T]`
- `rewards`: `object [E]`, each item is `float32 [T]`
- `timesteps`: `object [E]`, each item is `int32 [T]`
- `alive_mask`: `object [E]`, each item is `bool [T, N]`
- `metadata_json`

## Observation and Action Conventions

Default observation:

- `LidarObservation`
- shape `(128, 2)` before batching
- typically flattened to `256` features for simple MLP baselines

Supported action modes:

- `discrete`: `DiscreteSteerMetaAction`, scalar action in `{0,1,2,3,4}`
- `continuous`: `ContinuousAction`, normalized `float32 [2]` for acceleration and steering

## Build

Example discrete dataset:

```bash
python scripts_ngsim/build_expert_dataset.py \
  --scene us-101 \
  --action-mode discrete \
  --episodes 32 \
  --out expert_data/ngsim_expert_dataset_discrete.npz
```

Example continuous dataset:

```bash
python scripts_ngsim/build_expert_dataset.py \
  --scene us-101 \
  --action-mode continuous \
  --episodes 32 \
  --out expert_data/ngsim_expert_dataset_continuous.npz
```

Example full-scene dataset for PS-GAIL-style demonstrations:

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

Example targeted benchmark on one known small scene:

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

Optional controls:

- `--episode-root` to point at another processed dataset root
- `--prebuilt-split train|val` to choose the prebuilt split
- `--episode-name` to target a specific scene
- `--max-horizon` to cap steps per scenario
- `--controlled-vehicles` to replay multiple expert vehicles together
- `--control-all-vehicles` to control all valid vehicles in the selected traffic segment
- `--dataset-mode scene` to keep the whole traffic segment together
- `--max-surrounding` to control replay context

## Inspect

```bash
python scripts_ngsim/inspect_expert_dataset.py expert_data/ngsim_expert_dataset_discrete.npz
```

The inspector prints:

- number of dataset episodes
- total transitions
- trajectory length statistics
- observation and action shapes
- a few sample transitions
- a ready-to-run replay command for one recorded episode

## Python Loader

```python
from highway_env.imitation import ExpertTransitionDataset, SceneTransitionDataset

dataset = ExpertTransitionDataset(
    "expert_data/ngsim_expert_dataset_discrete.npz",
    flatten_observations=True,
)

sample = dataset[0]
print(sample["observation"].shape)
print(sample["action"])
```

```python
scene_dataset = SceneTransitionDataset(
    "expert_data/ngsim_expert_scene_dataset_discrete.npz",
    flatten_observations=True,
)

sample = scene_dataset[0]
print(sample["agent_id"], sample["observation"].shape)
```

## Full-Scene Replay Notes

`scene` mode is the closest fit to PS-GAIL style demonstrations because one dataset episode corresponds to one traffic segment and includes all controlled interacting vehicles together.

Performance notes:

- `--control-all-vehicles` automatically disables surrounding replay vehicles, because controlled vehicles already cover the valid scene participants
- dense scenes can still be expensive with `LidarObservation`, since each controlled vehicle computes its own local observation
- use `--episode-name` and `--max-horizon` first when benchmarking or debugging scene-level replay
- smaller scenes and shorter horizons are the fastest way to validate the pipeline before scaling collection up

## Assumptions

- The dataset root already contains repo-compatible processed trajectory files.
- Expert actions come from `NGSimEnv` expert replay rather than a learned policy.
- Rewards are stored when available from the environment; for the current `NGSimEnv`, reward is typically `0.0`.
- The first version targets a reliable single-agent transition dataset for later GAIL, while preserving episode metadata for future sequence batching and driver/scenario analysis.
