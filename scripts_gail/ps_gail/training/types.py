"""Data containers used by PS-GAIL rollout and update routines."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass
class AgentTransition:
    vehicle_id: int
    policy_observation: np.ndarray
    next_policy_observation: np.ndarray
    critic_observation: np.ndarray
    next_critic_observation: np.ndarray
    action: object
    action_mask: np.ndarray
    log_prob: float
    value: float
    trajectory_id: int
    trajectory_state: np.ndarray
    scene_index: int
    env_penalty: float
    crashed: bool
    offroad: bool
    challenge_pressure: float
    challenge_payoff: float
    challenge_crash_rate_ema: float
    challenge_offroad_rate_ema: float
    challenge_ttc_target: float
    challenge_gap_target: float
    done: bool
    policy_step_memory: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float16))


@dataclass
class RolloutBatch:
    policy_observations: np.ndarray
    next_policy_observations: np.ndarray
    critic_observations: np.ndarray
    next_critic_observations: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    trajectory_ids: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    gail_rewards_raw: np.ndarray
    gail_rewards_normalized: np.ndarray
    env_penalties: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    generator_features: np.ndarray
    scene_features: np.ndarray
    transition_scene_indices: np.ndarray
    sequence_features: np.ndarray
    sequence_last_indices: np.ndarray
    sequence_transition_indices: np.ndarray
    num_env_steps: int
    num_agent_steps: int
    sequence_rewards_raw: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    sequence_rewards_assigned: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    sequence_window_counts: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    vehicle_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    policy_step_memories: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=np.float16))
    challenge_pressures: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    challenge_payoffs: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    challenge_bonuses: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    challenge_crash_rate_ema: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    challenge_offroad_rate_ema: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    challenge_ttc_targets: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    challenge_gap_targets: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    num_episodes: int = 0
    num_terminated: int = 0
    num_truncated: int = 0
    num_crash_events: int = 0
    num_offroad_events: int = 0
    crash_agent_fraction: float = 0.0
    offroad_agent_fraction: float = 0.0
    mean_env_penalty: float = 0.0
    mean_raw_gail_reward: float = 0.0
    mean_normalized_gail_reward: float = 0.0
    mean_episode_length: float = 0.0
    min_episode_length: int = 0
    max_episode_length: int = 0
    unique_episode_names: int = 0
    episode_names: tuple[str, ...] = ()
    mean_controlled_vehicles: float = 0.0
    mean_road_vehicles: float = 0.0
    psro_active: bool = False
    psro_current_decisions: int = 0
    psro_archive_decisions: int = 0
    psro_current_fraction: float = 1.0


__all__ = [
    'AgentTransition',
    'RolloutBatch'
]
