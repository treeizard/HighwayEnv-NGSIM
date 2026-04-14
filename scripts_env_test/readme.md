`scripts_test/` contains lightweight manual and regression-style scripts for exercising the NGSIM environment outside the training pipeline.

Files:

- `test_ngsim_continuous.py`
  Runs a short expert-replay rollout in continuous mode and records a video when `moviepy` is available. Useful for checking that continuous actions and rendering still work end-to-end.

- `test_ngsim_discrete.py`
  Runs a short expert-replay rollout in discrete meta-action mode, prints the action distribution taken by the expert policy, and reports trajectory replay metrics when available.

- `test_ngsim_manual.py`
  Opens the environment in human-render mode and lets you control the ego vehicle from the terminal using discrete commands. Good for quick interaction and controller debugging.

- `test_ngsim_sp_veh.py`
  Runs one targeted scenario with a specified scene / episode / ego vehicle, optionally records video, and saves a trajectory plot comparing the expert reference path against the replayed trajectory.

- `test_ngsim_expert_traj.py`
  Runs multiple expert-replay cases with `max_surrounding=0` and reports per-case and aggregate trajectory accuracy metrics such as ADE and FDE. This is the best quick regression script for expert trajectory tracking quality.

Notes:

- All scripts safely register `NGSim-US101-v0` only when it is not already present in the Gymnasium registry.
- Video recording scripts degrade gracefully when `moviepy` is not installed.
- Output folders such as `videos/` and `videos_discrete_test/` are created relative to the current working directory.

Example usage:

```bash
python scripts_test/test_ngsim_expert_traj.py --scene us-101 --action-mode discrete --cases 10
python scripts_test/test_ngsim_discrete.py
python scripts_test/test_ngsim_continuous.py
python scripts_test/test_ngsim_manual.py
```
