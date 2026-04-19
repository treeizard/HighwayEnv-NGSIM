# `scripts_setup`

Utility scripts for preparing NGSIM-style data and generating simple visualizations for this repository.

## Before You Run Anything

Run these commands from the repository root:

```bash
pip install -e .
```

Most scripts assume the project root is the current working directory so that paths like `highway_env/data/...` resolve correctly.

## Typical Workflow

1. Convert a raw NGSIM CSV into the repository's processed format with `dump_data_ngsim.py`.
2. Optionally split a raw CSV into fixed-length train/validation/test episodes with `dump_data_time_ngsim.py`.
3. Use `visualize.py` to render a high-resolution snapshot of a configured scene.

## Scripts

### `dump_data_ngsim.py`

Converts a raw NGSIM CSV file into the repository's base processed format used by older loading utilities.

#### Usage

```bash
python scripts_setup/dump_data_ngsim.py <path-to-csv> [--scene us-101]
```

#### Arguments

- `path`: path to the raw NGSIM CSV file.
- `--scene`: scene/location filter passed into `ngsim_data`. Default: `us-101`.

#### What It Does

- loads the raw CSV with `highway_env.data.ngsim.ngsim_data`
- filters rows by the selected `scene` when the CSV contains a `Location` column
- cleans the loaded data via `reader.clean()`
- writes processed files into:

```text
highway_env/data/processed/<scene>/
```

#### Output Files

- `vehicle_record_file.csv`
- `vehicle_file.csv`
- `snapshot_file.csv`

#### Example

```bash
python scripts_setup/dump_data_ngsim.py data/us101.csv --scene us-101
```

### `dump_data_time_ngsim.py`

Reads a raw NGSIM CSV and writes fixed-duration episodes, primarily for faster episode-based loading.

By default it creates 20-second non-overlapping windows and splits them chronologically into consecutive `train`, `val`, and `test` segments.

#### Usage

```bash
python scripts_setup/dump_data_time_ngsim.py <path-to-csv> \
  [--scene us-101] \
  [--out_root highway_env/data/processed_20s] \
  [--episode_len_sec 20.0] \
  [--stride_sec 20.0] \
  [--val_ratio 0.1] \
  [--test_ratio 0.1] \
  [--car_only]
```

#### Arguments

- `path`: path to the raw NGSIM CSV file.
- `--scene`: scene/location name. Default: `us-101`.
- `--out_root`: root directory for windowed output. Default: `highway_env/data/processed_20s`.
- `--episode_len_sec`: episode length in seconds. Default: `20.0`.
- `--stride_sec`: time between episode starts in seconds. Default: `20.0`.
- `--val_ratio`: fraction of windows assigned to validation. Default: `0.1`.
- `--test_ratio`: fraction of windows assigned to test. Default: `0.1`.
- `--car_only`: keep only car-sized vehicles in the written episode folders.

#### What It Does

- loads and cleans the raw CSV
- groups snapshots into fixed-length time windows
- optionally filters non-car vehicles using length/width thresholds
- splits windows by time into consecutive `train/`, `val/`, and `test/` folders
- stores each episode in its own `t<unix_time>` directory

#### Output Layout

```text
highway_env/data/processed_20s/<scene>/
  train/
    t1118846663000/
      vehicle_record_file.csv
      vehicle_file.csv
      snapshot_file.csv
  val/
    t1118846673000/
      vehicle_record_file.csv
      vehicle_file.csv
      snapshot_file.csv
  test/
    t1118846683000/
      vehicle_record_file.csv
      vehicle_file.csv
      snapshot_file.csv
```

#### Notes

- if `stride_sec == episode_len_sec`, episodes do not overlap
- if `stride_sec < episode_len_sec`, windows overlap
- `--car_only` is useful when downstream expert replay should only model passenger cars
- `train` gets the earliest windows, `val` the next windows, and `test` the latest windows

#### Example

```bash
python scripts_setup/dump_data_time_ngsim.py data/us101.csv --scene us-101
```

Example with overlapping 5-second stride and 10-second windows:

```bash
python scripts_setup/dump_data_time_ngsim.py data/us101.csv \
  --scene us-101 \
  --episode_len_sec 10 \
  --stride_sec 5
```

### `visualize.py`

Renders and saves a high-resolution PNG snapshot from the registered `NGSim-US101-v0` environment.

This script is currently written as a Python utility/example rather than a command-line tool. Running the file directly executes the hardcoded example at the bottom of the script.

#### Direct Run

```bash
python scripts_setup/visualize.py
```

With the current code, this saves:

```text
plots/japanese_road_only.png
```

#### Current Default Behavior

The built-in example:

- creates an environment with `scene="japanese"`
- renders a full scene (`road_only=False`)
- saves a large PNG image to `plots/japanese_road_only.png`

#### Programmatic Use

You can also import and call `save_highres_snapshot(...)` yourself:

```python
from scripts_setup.visualize import save_highres_snapshot

save_highres_snapshot(
    out_path="plots/example.png",
    width=5000,
    height=700,
    scaling=7.0,
    episode_name=None,
    ego_vehicle_ID=None,
    seed=42,
    road_only=False,
)
```

#### Parameters of `save_highres_snapshot(...)`

- `out_path`: output PNG path
- `width`: render width in pixels
- `height`: render height in pixels
- `scaling`: renderer zoom factor
- `episode_name`: optional fixed episode folder name
- `ego_vehicle_ID`: optional fixed ego vehicle id
- `seed`: random seed for environment reset
- `road_only`: if `True`, renders only the road layout

### `build_prebuilt_japanese.py`

Builds Japanese prebuilt trajectory caches directly from the filtered Morinomiya `.npy` artifact produced by the scripts in `raw_data/`.

This is useful when you want to regenerate:

- `veh_ids_train.npy`
- `trajectory_train.npy`
- `veh_ids_val.npy`
- `trajectory_val.npy`
- `veh_ids_test.npy`
- `trajectory_test.npy`

for a target folder such as `highway_env/data/processed_20s/japanese/prebuilt/`.

### `plot_vehicle_size_distribution.py`

Plots vehicle length/width distributions for both the repository's US-101 (NGSIM) and Japanese prebuilt datasets on a shared metric scale.

By default it reads:

- `highway_env/data/processed_20s/us-101/prebuilt/trajectory_train.npy`
- `highway_env/data/processed_10s/japanese/prebuilt/trajectory_train.npy`

and converts US-101 dimensions from feet to meters before plotting.

#### Usage

```bash
python scripts_setup/plot_vehicle_size_distribution.py
```

#### Outputs

By default the script writes:

```text
plots/vehicle_dimensions/vehicle_size_distribution_train.png
plots/vehicle_dimensions/vehicle_size_summary_train.txt
```

#### Useful Arguments

- `--split`: choose `train`, `val`, or `test`
- `--us-root`: root folder for the US-101 prebuilt files
- `--japanese-root`: root folder for the Japanese prebuilt files
- `--out-dir`: output directory for plots and summary text

#### Usage

```bash
python scripts_setup/build_prebuilt_japanese.py
```

#### Default Inputs and Outputs

- input: `raw_data/morinomiya_filtered_without_duplicates.npy`
- output root: `highway_env/data/processed_20s`
- scene: `japanese`
- window size: `20` seconds
- JST time filter: `09:00:00 <= time < 12:00:00`

#### What It Does

- loads the filtered Morinomiya record array
- reconstructs local XY coordinates
- estimates a curved-road remap using the mainline lanes
- smooths each vehicle trajectory
- slices the data into fixed-duration windows
- optionally filters to car-sized vehicles only
- splits episode windows into consecutive `train`, `val`, and `test`
- writes prebuilt `.npy` files under:

```text
highway_env/data/processed_20s/japanese/prebuilt/
```

#### Example

```bash
python scripts_setup/build_prebuilt_japanese.py \
  --input_npy raw_data/morinomiya_filtered_without_duplicates.npy \
  --episode_root highway_env/data/processed_20s \
  --window_sec 20
```

Example with an explicit JST clock filter:

```bash
python scripts_setup/build_prebuilt_japanese.py \
  --start_clock 09:00:00 \
  --end_clock 12:00:00
```

## Scene Notes

This repository contains scene-specific logic for at least:

- `us-101`
- `japanese`

Some older helper code also references `i-80`, but support depends on the downstream road-building and environment code you intend to use.
