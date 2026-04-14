# `scripts_ngsim`

Utilities in this folder are centered on NGSIM replay, expert dataset generation, and imitation learning.

## Dataset Tools

- `build_expert_dataset.py`
  Main CLI for building expert datasets from processed replay trajectories.
- `inspect_expert_dataset.py`
  Prints dataset statistics, sample transitions, and a replay command.
- `build_single_episode_expert_discrete.py`
  Convenience wrapper for collecting one targeted discrete dataset episode.

## Training Scripts

- `train1_rl.py`
  Reinforcement-learning baseline training.
- `train2_imitate.py`
  Imitation-learning training entry point.
- `train3_gail.py`
  GAIL training for continuous-action setups.
- `train3_gail_discrete.py`
  GAIL training for discrete-action expert datasets.
- `train3_gail_discrete_vae.py`
  Discrete GAIL variant with VAE components.
- `train_ps_gail_discrete.py`
  Scene-aware discrete PS-GAIL-style training that can consume `scene`-mode datasets.

## Legacy Collection Scripts

- `build_expert_data.py`
  Older continuous expert collection script kept for reference.
- `build_expert_data_discrete.py`
  Older discrete expert collection script kept for reference.

The canonical dataset API now lives in `highway_env/imitation/expert_dataset.py`.
