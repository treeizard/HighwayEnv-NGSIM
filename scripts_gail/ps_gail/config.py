from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PSGAILConfig:
    expert_data: str = "expert_data/ngsim_ps_traj_expert_discrete_54902119"
    run_name: str = "simple_ps_gail"
    scene: str = "us-101"
    episode_root: str = "highway_env/data/processed_20s"
    prebuilt_split: str = "train"
    seed: int = 0

    total_rounds: int = 10
    rollout_steps: int = 128
    max_expert_samples: int = 100_000
    max_surrounding: str | int = "all"
    control_all_vehicles: bool = True
    percentage_controlled_vehicles: float = 0.1
    enable_collision: bool = True
    allow_idm: bool = True

    cells: int = 128
    maximum_range: float = 64.0
    simulation_frequency: int = 10
    policy_frequency: int = 10
    max_episode_steps: int = 300

    hidden_size: int = 256
    learning_rate: float = 3e-4
    disc_learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 1024
    disc_batch_size: int = 1024
    disc_updates_per_round: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    checkpoint_every: int = 5
    device: str = "auto"

    wandb_mode: str = "disabled"
    wandb_project: str = "highwayenv-ps-gail"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""
    wandb_watch: bool = False
