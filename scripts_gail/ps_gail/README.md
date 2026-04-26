# Simple PS-GAIL

This package is a small modular trainer for the NGSIM discrete meta-action setup.

- Policy input: lidar + lane observation + `[length, velocity, heading]`.
- Policy output: discrete meta action index from `DiscreteSteerMetaAction`.
- Discriminator input: policy observation + trajectory state `[x, y, v]`.
- Discriminator does not receive the discrete meta action.
- Generator environment defaults to `enable_collision=True` and `allow_idm=True`.

Run a quick demonstration:

```bash
MPLCONFIGDIR=/tmp python scripts_gail/train_simple_ps_gail.py \
  --expert-data expert_data/ngsim_ps_traj_expert_discrete_54902119 \
  --total-rounds 10 \
  --rollout-steps 128 \
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
