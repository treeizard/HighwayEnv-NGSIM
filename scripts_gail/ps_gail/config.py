from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PSGAILConfig:
    expert_data: str = "expert_data/ngsim_ps_traj_expert_discrete_54902119"
    run_name: str = "simple_ps_gail"
    scene: str = "us-101"
    action_mode: str = "discrete"
    continuous_action_dim: int = 2
    episode_root: str = "highway_env/data/processed_20s"
    prebuilt_split: str = "train"
    seed: int = 0

    total_rounds: int = 10
    rollout_steps: int = 128
    rollout_min_episodes: int = 1
    rollout_full_episodes: bool = True
    rollout_max_episode_steps: int = 0
    num_rollout_workers: int = 1
    rollout_worker_threads: int = 1
    max_expert_samples: int = 100_000
    trajectory_frame: str = "relative"
    max_surrounding: str | int = "all"
    control_all_vehicles: bool = False
    percentage_controlled_vehicles: float = 0.2
    controlled_vehicle_curriculum: bool = False
    initial_controlled_vehicle_fraction: float = 0.2
    final_controlled_vehicle_fraction: float = 1.0
    controlled_vehicle_curriculum_rounds: int = 100
    enable_collision: bool = True
    terminate_when_all_controlled_crashed: bool = True
    allow_idm: bool = True

    cells: int = 128
    maximum_range: float = 64.0
    simulation_frequency: int = 10
    policy_frequency: int = 10
    max_episode_steps: int = 200

    policy_model: str = "mlp"
    hidden_size: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    learning_rate: float = 3e-4
    disc_learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 1024
    disc_batch_size: int = 1024
    disc_updates_per_round: int = 4
    disc_expert_label: float = 0.9
    disc_generator_label: float = 0.1
    cgail_k: float = 0.0
    enable_scene_discriminator: bool = False
    enable_sequence_discriminator: bool = False
    scene_max_vehicles: int = 64
    scene_feature_dim_per_vehicle: int = 5
    scene_reward_coef: float = 1.0
    sequence_length: int = 8
    sequence_stride: int = 1
    sequence_reward_coef: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_gail_reward: bool = True
    gail_reward_clip: float = 5.0
    collision_penalty: float = 2.0
    offroad_penalty: float = 2.0
    final_reward_clip: float = 10.0
    checkpoint_every: int = 5
    save_checkpoint_video: bool = False
    checkpoint_video_steps: int = 120
    checkpoint_video_dir: str = "videos"
    checkpoint_video_deterministic: bool = True
    checkpoint_video_width: int = 1200
    checkpoint_video_height: int = 608
    checkpoint_video_scaling: float = 5.5
    device: str = "auto"

    wandb_mode: str = "disabled"
    wandb_project: str = "highwayenv-ps-gail"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""
    wandb_watch: bool = False
