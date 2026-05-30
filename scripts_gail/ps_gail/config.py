"""Shared CLI-backed configuration for PS-GAIL/AIRL training.

The training scripts auto-generate argparse flags from this dataclass, so fields
should stay primitive and serializable. Empty strings and zeros are used as
"disabled/default" sentinels for several schedule-like options.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PSGAILConfig:
    # Dataset/env identity. These values define the observation/action contract;
    # changing them can invalidate expert-data compatibility.
    expert_data: str = "expert_data/ngsim_ps_traj_expert_discrete_54902119" # Define the path of the expert data. 
    run_name: str = "simple_ps_gail"
    resume_checkpoint: str = ""
    scene: str = "us-101"
    action_mode: str = "discrete"
    continuous_action_dim: int = 2 # -> [acceleration, lane change]
    episode_root: str = "highway_env/data/processed_20s"
    prebuilt_split: str = "train"
    seed: int = 0

    # Training/Roll out parameters. These values define the amount of simulated rollouts;
    # changing them can modify how training is to be conducted
    total_rounds: int = 10
    rollout_steps: int = 128
    rollout_min_episodes: int = 1
    rollout_full_episodes: bool = True
    rollout_max_episode_steps: int = 0
    rollout_target_aware_episodes: bool = True
    rollout_target_min_episodes: int = 1
    rollout_target_episode_safety_factor: float = 1.25
    rollout_training_subsample: bool = True
    rollout_training_agent_steps: int = 0

    # Threads/Training parameters. These values define the training setup and optimization parameters;
    num_rollout_workers: int = 1 # number of actives worker that will be rolling out trajectories in parallel.
    rollout_worker_threads: int = 1 # number of active threads can be utilized by each rollout worker. 
    evaluation_num_workers: int = 1 # persistent evaluation workers; 0 keeps old main-process serial eval.
    evaluation_worker_threads: int = 2 # native CPU threads available inside each evaluation worker.
    evaluation_cache_envs: bool = True # keep evaluation envs alive inside persistent eval workers.
    evaluation_max_cached_envs_per_worker: int = 0 # 0 means unlimited per-worker eval env cache.
    max_expert_samples: int = 100_000
    trajectory_frame: str = "relative" # Parameters that can be utilized for state-only GAIL
    max_surrounding: str | int = "all"
    control_all_vehicles: bool = False
    percentage_controlled_vehicles: float = 0.2 # percentage of vehicles controlled by policy. 
    controlled_vehicle_curriculum: bool = False # Automatic increase of percentage of controlled vehicles. 
    initial_controlled_vehicles: float = 0.2
    final_controlled_vehicles: float = 1.0
    controlled_vehicle_curriculum_rounds: int = 100
    controlled_vehicle_increment_rounds: int = 0
    controlled_vehicle_schedule: str = ""
    warmup_rounds: int = 0
    warmup_learning_rate: float = 0.0
    warmup_disc_learning_rate: float = 0.0
    warmup_entropy_coef: float = -1.0
    warmup_clip_range: float = 0.0
    warmup_disc_updates_per_round: int = 0
    warmup_gail_reward_clip: float = 0.0
    warmup_final_reward_clip: float = 0.0
    vehicle_increase_warmup_rounds: int = 0
    rollout_target_agent_steps: int = 0
    initial_rollout_target_agent_steps: int = 0
    final_rollout_target_agent_steps: int = 0
    rollout_target_agent_steps_curriculum_rounds: int = 0
    rollout_target_agent_steps_schedule: str = ""
    psro_lite: bool = False # the PRSO without nash equilibrium computation. 
    psro_archive_every: int = 20
    psro_archive_size: int = 5
    psro_mixture_after_jump_rounds: int = 20
    psro_current_policy_fraction: float = 0.65
    enable_collision: bool = True
    terminate_when_all_controlled_crashed: bool = True
    allow_idm: bool = True

    # Sensor Parameters. These values define the observation space and what information is available to the policy;
    # changing them can modify the state representation and thus the learning problem itself.
    cells: int = 128
    maximum_range: float = 64.0
    simulation_frequency: int = 10
    policy_frequency: int = 10
    max_episode_steps: int = 200
    max_episode_steps_schedule: str = ""

    # Policy/Discriminator architecture and optimization parameters. These values define the model architecture and optimization hyperparameters;
    # changing them can significantly affect learning dynamics and performance. (Currently support transformer and MLP policy architectures, and MLP discriminators.)
    policy_model: str = "mlp"
    hidden_size: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    transformer_temporal_module: bool = False
    transformer_temporal_kernel_size: int = 5
    transformer_temporal_layers: int = 1
    transformer_memory_tokens: int = 8
    transformer_memory_context_length: int = 32
    transformer_recurrent_sequence_length: int = 32
    transformer_recurrent_sequences_per_batch: int = 32
    transformer_recurrent_micro_batch_sequences: int = 8
    transformer_memory_storage_dtype: str = "float16"
    transformer_use_causal_attention: bool = True
    centralized_critic: bool = False
    central_critic_max_vehicles: int = 64
    central_critic_include_local_obs: bool = False
    central_critic_pooling: str = "flat"
    central_critic_attention_heads: int = 4
    learning_rate: float = 4e-4
    learning_rate_schedule: str = ""
    bc_pretrain_epochs: int = 0 # Behavior cloning pretraining epochs. Setting this to >0 will enable a BC pretraining phase before GAIL training. (Not utilized as of this moment)
    bc_pretrain_learning_rate: float = 3e-4
    bc_pretrain_batch_size: int = 4096
    bc_pretrain_micro_batch_size: int = 0
    bc_pretrain_weight_decay: float = 0.0
    bc_pretrain_validation_fraction: float = 0.1
    bc_pretrain_eval_episodes: int = 4
    bc_pretrain_eval_deterministic: bool = True
    bc_pretrain_min_mean_episode_length: float = 0.0
    bc_pretrain_abort_on_failed_eval: bool = False
    policy_bc_regularization_coef: float = 0.0
    policy_bc_regularization_final_coef: float = 0.0
    policy_bc_regularization_decay_rounds: int = 0
    disc_learning_rate: float = 3e-4
    disc_learning_rate_schedule: str = ""
    gamma: float = 0.99 # Discount factor for future rewards.
    initial_gamma: float = 0.0
    final_gamma: float = 0.0
    gamma_curriculum_rounds: int = 0
    gamma_schedule: str = ""
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_schedule: str = ""
    ppo_epochs: int = 6
    batch_size: int = 1024
    ppo_micro_batch_size: int = 0
    disc_batch_size: int = 1024
    disc_updates_per_round: int = 4
    disc_updates_per_round_schedule: str = ""
    discriminator_replay_rounds: int = 0
    discriminator_replay_max_samples: int = 0
    disc_expert_label: float = 0.9
    disc_generator_label: float = 0.1
    discriminator_input: str = "auto"
    discriminator_loss: str = "wgan_gp"
    discriminator_hidden_sizes: str = "128,128,64"
    discriminator_dropout: float = 0.0
    discriminator_spectral_norm: bool = False
    enable_hard_example_selection: bool = False
    hard_example_candidate_samples: int = 65_536
    hard_example_selected_fraction: float = 0.35
    hard_example_uniform_mix: float = 0.25
    hard_example_temperature: float = 1.0
    hard_example_min_samples: int = 4_096
    wgan_gp_lambda: float = 2.0
    wgan_reward_center: bool = False
    wgan_reward_clip: float = 0.0
    wgan_reward_scale: float = 1.0
    wgan_reward_norm_min_std: float = 1.0e-3
    wgan_reward_norm_clip: float = 5.0
    airl_policy_reward_mode: str = "shaped"
    normalize_discriminator_features: bool = True
    discriminator_feature_clip: float = 10.0
    cgail_k: float = 0.0
    enable_action_masking: bool = True
    enable_scene_discriminator: bool = False
    enable_sequence_discriminator: bool = False
    scene_max_vehicles: int = 64
    scene_feature_dim_per_vehicle: int = 5
    scene_reward_coef: float = 1.0
    sequence_feature_mode: str = "local_deltas"
    sequence_length: int = 8
    sequence_stride: int = 1
    sequence_reward_coef: float = 1.0
    sequence_reward_assignment: str = "last"
    entropy_coef: float = 0.015
    entropy_coef_schedule: str = ""
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_gail_reward: bool = True
    allow_wgan_reward_normalization: bool = False
    gail_reward_clip: float = 5.0
    collision_penalty: float = 2.0
    offroad_penalty: float = 2.0
    final_reward_clip: float = 10.0
    enable_player_challenge_reward: bool = False
    # Challenge reward parameters (In development). These values define the additional reward shaping based on specific challenge metrics (like time-to-collision, gap, crashes, offroad events, etc.);
    challenge_reward_coef: float = 0.2
    challenge_reward_clip: float = 0.25
    challenge_max_primary_reward_fraction: float = 0.10
    challenge_ttc_target: float = 0.0
    challenge_ttc_margin: float = 0.75
    challenge_ttc_floor: float = 0.0
    challenge_gap_target: float = 0.0
    challenge_gap_floor: float = 0.0
    challenge_ttc_weight: float = 0.6
    challenge_gap_weight: float = 0.4
    challenge_crash_weight: float = 4.0
    challenge_offroad_weight: float = 2.0
    challenge_risk_ema_beta: float = 0.95
    challenge_expert_like_quantile: float = 0.25
    checkpoint_every: int = 5
    save_checkpoint_video: bool = False
    checkpoint_video_every: int = 0
    checkpoint_video_steps: int = 120
    checkpoint_video_dir: str = "videos"
    checkpoint_video_deterministic: bool = True
    checkpoint_video_width: int = 1200
    checkpoint_video_height: int = 608
    checkpoint_video_scaling: float = 5.5
    validation_every: int = 20
    validation_episodes: int = 4
    validation_prebuilt_split: str = "val"
    validation_vehicle_mode: str = "training_count"
    validation_control_all_vehicles: bool = False
    validation_stress_every: int = 100
    validation_stress_episodes: int = 2
    validation_stress_vehicle_mode: str = "all"
    save_best_checkpoint: bool = True
    validation_min_delta: float = 0.0
    validation_score_horizon_seconds: int = 20
    validation_score_position_weight: float = 1.0
    validation_score_speed_weight: float = 0.5
    validation_score_lane_offset_weight: float = 2.0
    validation_score_crash_weight: float = 25.0
    validation_score_offroad_weight: float = 25.0
    validation_score_hard_brake_weight: float = 2.0
    test_episodes: int = 4
    test_prebuilt_split: str = "test"
    test_control_all_vehicles: bool = False
    evaluation_horizons_seconds: str = "1,5,10,20"
    hard_brake_accel_threshold: float = -3.0
    device: str = "auto"

    wandb_mode: str = "disabled"
    wandb_project: str = "highwayenv-ps-gail"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""
    wandb_watch: bool = False


def checkpoint_video_interval(cfg: PSGAILConfig) -> int:
    video_every = int(getattr(cfg, "checkpoint_video_every", 0))
    if video_every > 0:
        return video_every
    return max(0, int(getattr(cfg, "checkpoint_every", 0)))


def should_save_checkpoint_video(cfg: PSGAILConfig, step: int) -> bool:
    if not bool(getattr(cfg, "save_checkpoint_video", False)):
        return False
    interval = checkpoint_video_interval(cfg)
    return interval > 0 and int(step) % interval == 0
