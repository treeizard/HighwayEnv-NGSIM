# Simple PS-GAIL

This package is a small modular trainer for the NGSIM discrete meta-action setup.

- Policy input: lidar + lane observation + `[length, velocity, heading]`.
- Policy output: discrete meta action index from `DiscreteSteerMetaAction`.
- Discriminator input: policy observation + trajectory state `[x, y, v]`.
- Discriminator does not receive the discrete meta action.
- Generator environment defaults to `enable_collision=True` and `allow_idm=True`.
- Optional scene discriminator: fixed-size full-road snapshots encoded as
  top-K `[presence, rel_x, rel_y, vx, vy]` road-vehicle rows. Rebuild expert
  data with the current `build_ps_traj_expert_discrete.py` so `scene_features`
  are present before enabling this.
- Optional sequence discriminator: GRU over fixed-length windows of the normal
  discriminator feature, grouped by vehicle trajectory id. By default, each
  sequence reward is assigned to the final transition of its window; set
  `--sequence-reward-assignment mean` or `sum` to run denser credit-assignment
  ablations without changing the discriminator itself.
- Unified expert datasets collected with `build_ps_traj_expert_discrete.py`
  preserve the existing GAIL fields and, when collected with continuous expert
  control, add `actions_continuous_env` with columns
  `[acceleration_norm, steering_norm]` plus `actions_steering_acceleration`
  with columns `[steering_rad, acceleration_mps2]`.
- Expert collection can run episodes in parallel with
  `--num-collection-workers N --collection-worker-threads 2`; each worker owns
  one environment at a time and caps native CPU thread pools.
- Use `--collect-all-split-episodes` or `--max-episodes 0` to collect every
  available fixed episode in the configured prebuilt split without duplicate
  random resets.
- Generator rollouts collect complete random NGSIM episodes by default. With the
  default `max_episode_steps=200`, `--rollout-min-episodes 4` means four full
  200-step episodes per training round, split across rollout workers.
- Training starts from 20% controlled vehicles by default, and the curriculum
  default initial fraction is also 20%.

Run a quick demonstration:

```bash
MPLCONFIGDIR=/tmp python scripts_gail/train_simple_ps_gail.py \
  --expert-data expert_data/ngsim_ps_traj_expert_discrete_54902119 \
  --total-rounds 10 \
  --rollout-steps 200 \
  --rollout-min-episodes 4 \
  --max-expert-samples 100000 \
  --run-name ps_gail_collision_idm
```

Enable Weights & Biases logging:

```bash
python scripts_gail/train_simple_ps_gail.py \
  --wandb-mode online \
  --wandb-project highwayenv-ps-gail \
  --wandb-tags ps-gail,collision,idm \
  --run-name ps_gail_collision_idm
```

Use `--wandb-mode offline` on a cluster node without internet, or keep the default
`--wandb-mode disabled` for local debugging.

For a tiny smoke test:

```bash
MPLCONFIGDIR=/tmp python scripts_gail/train_simple_ps_gail.py \
  --total-rounds 1 \
  --rollout-steps 2 \
  --max-expert-samples 128 \
  --hidden-size 32 \
  --batch-size 64 \
  --disc-batch-size 64 \
  --disc-updates-per-round 1 \
  --ppo-epochs 1 \
  --run-name smoke
```

Check the dual-discriminator setup on SLURM:

```bash
sbatch slurum/check_build_ps_gail_dual_disc_expert.bash
sbatch slurum/check_train_simple_ps_gail_dual_disc_gpu_32c.bash
```
