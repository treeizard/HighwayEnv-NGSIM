import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader

import ray
import gymnasium as gym
from gymnasium.envs.registration import register

# ---------------------------------------------------------------------
# Make sure the project root is importable (same as make_videos_multi.py)
# ---------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Register your NGSim env (same ID as the video script)
register(id="NGSim-US101", entry_point="highway_env.envs.ngsim_env:NGSimEnv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import your NGSIM trajectory tools
from highway_env.ngsim_utils.trajectory_gen import (
    build_trajectory,
    process_raw_trajectory,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot(actions: torch.Tensor, n_actions: int) -> torch.Tensor:
    return F.one_hot(actions, num_classes=n_actions).float()


def compute_returns(rews: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute discounted returns, resetting at episode boundaries.
    rews : [N]
    dones: [N] boolean (True at terminal step)
    """
    returns = torch.zeros_like(rews)
    G = 0.0
    for i in reversed(range(len(rews))):
        if dones[i]:
            G = 0.0
        G = rews[i] + gamma * G
        returns[i] = G
    return returns


# ---------------------------------------------------------------------
# Models (2-layer MLP policy + discriminator) for DISCRETE actions
# ---------------------------------------------------------------------
class DiscretePolicyNet(nn.Module):
    """
    2-layer MLP policy over discrete actions.
    Input: obs_dim
    Output: logits over n_actions
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.logits(x)  # [B, n_actions]

    def act(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        obs: 1D numpy array (single flattened state)
        returns: (action_int, log_prob_float)
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = self.forward(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item())


class DiscriminatorNet(nn.Module):
    """
    2-layer MLP discriminator D(s,a) -> (0,1) for GAIL.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + n_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, acts_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, acts_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return torch.sigmoid(logits)  # [B, 1]


# ---------------------------------------------------------------------
# Expert Dataset built directly from NGSIM ego trajectory
# ---------------------------------------------------------------------
class ExpertDataset(Dataset):
    """
    Expert dataset over (state, action).
    states : [N, obs_dim]
    actions: [N] (int indices into Discrete action space)
    """

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        assert len(states) == len(actions)
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]


def infer_meta_action(lane_t, lane_tp1, v_t, v_tp1, speed_eps: float = 0.3) -> int:
    """
    Very simple mapping from consecutive NGSIM (lane, speed) to DiscreteMetaAction index.

    Assumed mapping:
        0: LANE_LEFT
        1: IDLE
        2: LANE_RIGHT
        3: FASTER
        4: SLOWER
    """
    # lane change
    if lane_tp1 > lane_t:
        return 0  # LANE_LEFT
    elif lane_tp1 < lane_t:
        return 2  # LANE_RIGHT

    # same lane: use speed change
    if v_tp1 > v_t + speed_eps:
        return 3  # FASTER
    elif v_tp1 < v_t - speed_eps:
        return 4  # SLOWER
    else:
        return 1  # IDLE


def build_expert_dataset_from_ngsim(
    env_id: str,
    base_cfg: Dict[str, Any],
    scene: str = "us-101",
    ego_vehicle_ID: int = 121,
    period: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (states, actions) from NGSIM ego trajectory, using the env's own
    observation (Kinematics) and inferred discrete meta-actions.

    The idea:
      - Get ego NGSIM trajectory: [x_ft, y_ft, speed_ft, lane_id].
      - Convert to metric: [s_m, r_m, v_mps, lane].
      - For each consecutive (t, t+1), infer a DiscreteMetaAction.
      - Step the NGSimEnv with that action; store the env's observation as the state.
    """
    # 1) Load the NGSIM ego trajectory from your existing code
    traj_set = build_trajectory(scene, period, ego_vehicle_ID)
    ego_traj_raw = traj_set["ego"]["trajectory"]  # list of [x_ft, y_ft, speed, lane]

    if len(ego_traj_raw) < 2:
        raise RuntimeError("Ego trajectory too short to build expert dataset.")

    ego_traj = process_raw_trajectory(ego_traj_raw)  # [s_m, r_m, v_mps, lane]

    # 2) Build env and configure it like in training
    env = gym.make(env_id)
    env.unwrapped.configure(base_cfg)
    obs, info = env.reset(seed=0)

    obs_list = []
    act_list = []

    for t in range(len(ego_traj) - 1):
        s0, r0, v0, lane0 = ego_traj[t]
        s1, r1, v1, lane1 = ego_traj[t + 1]

        a_t = infer_meta_action(lane0, lane1, v0, v1)

        # Flatten obs from env
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs_list.append(obs_flat)
        act_list.append(a_t)

        # Step env forward with that action
        obs, r, terminated, truncated, info = env.step(a_t)
        if terminated or truncated:
            break

    env.close()

    states = np.stack(obs_list, axis=0) if obs_list else np.zeros((0,), dtype=np.float32)
    actions = np.array(act_list, dtype=np.int64)
    if states.ndim == 1:  # handle edge case if only one sample
        states = states.reshape(1, -1)
    return states, actions


def load_expert_loader_from_ngsim(
    env_id: str,
    base_cfg: Dict[str, Any],
    batch_size: int,
    scene: str = "us-101",
    ego_vehicle_ID: int = 121,
    period: int = 0,
) -> DataLoader:
    states, actions = build_expert_dataset_from_ngsim(
        env_id=env_id,
        base_cfg=base_cfg,
        scene=scene,
        ego_vehicle_ID=ego_vehicle_ID,
        period=period,
    )
    if len(states) == 0:
        raise RuntimeError("Expert dataset is empty. Check NGSIM trajectory / env setup.")
    dataset = ExpertDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


# ---------------------------------------------------------------------
# Ray Env Worker – uses the SAME config style as make_videos_multi.py
# ---------------------------------------------------------------------
@ray.remote
class EnvWorker:
    """
    Ray actor that runs rollouts in its own NGSim env instance.
    Uses the same base_cfg and configure(...) pattern as make_videos_multi.py.
    """

    def __init__(self, env_id: str, base_env_config: Dict[str, Any],
                 max_episode_steps: int, worker_seed: int):
        self.env_id = env_id
        self.base_cfg = dict(base_env_config)
        self.max_episode_steps = max_episode_steps
        self.worker_seed = worker_seed

        # Build and configure env
        self.env = gym.make(self.env_id)
        self.env.unwrapped.configure(self.base_cfg)

        # Inspect spaces
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        if isinstance(obs_space, gym.spaces.Box):
            self.obs_dim = int(np.prod(obs_space.shape))  # flatten
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError(f"This script assumes a discrete action space, got: {act_space}")

        self.n_actions = act_space.n

    def run_rollouts(self, policy_state_dict, n_episodes: int):
        """
        Collect n_episodes trajectories with current policy parameters.
        Returns dict: obs, acts, logp, dones, ep_ids (all concatenated).
        """
        # Rebuild policy (on CPU is fine)
        policy = DiscretePolicyNet(self.obs_dim, self.n_actions)
        policy.load_state_dict(policy_state_dict)
        policy.eval()

        all_obs = []
        all_acts = []
        all_logp = []
        all_dones = []
        all_ep_ids = []

        ep_id = 0
        for ep in range(n_episodes):
            # Seed per episode for reproducibility (optional)
            seed = self.worker_seed * 1000 + ep
            obs, info = self.env.reset(seed=seed)
            obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)

            done = False
            truncated = False
            steps = 0

            while not (done or truncated) and steps < self.max_episode_steps:
                action, logp = policy.act(obs_flat)
                next_obs, r, done, truncated, info = self.env.step(action)
                next_obs_flat = np.asarray(next_obs, dtype=np.float32).reshape(-1)

                all_obs.append(obs_flat.copy())
                all_acts.append(action)
                all_logp.append(logp)
                all_dones.append(done or truncated)
                all_ep_ids.append(ep_id)

                obs_flat = next_obs_flat
                steps += 1

            ep_id += 1

        batch = {
            "obs": np.asarray(all_obs, dtype=np.float32),
            "acts": np.asarray(all_acts, dtype=np.int64),
            "logp": np.asarray(all_logp, dtype=np.float32),
            "dones": np.asarray(all_dones, dtype=np.bool_),
            "ep_ids": np.asarray(all_ep_ids, dtype=np.int32),
        }
        return batch


# ---------------------------------------------------------------------
# GAIL Config
# ---------------------------------------------------------------------
@dataclass
class GAILConfig:
    env_id: str = "NGSim-US101"

    # Ray / rollout
    num_workers: int = 4
    episodes_per_worker: int = 4
    max_episode_steps: int = 450  # roughly 30s at 15 Hz if you like

    # Training
    total_iters: int = 1000
    gamma: float = 0.99
    pi_lr: float = 3e-4
    disc_lr: float = 3e-4
    hidden_dim: int = 128
    disc_updates_per_iter: int = 2
    disc_batch_size: int = 64

    # NGSIM expert source
    scene: str = "us-101"
    ego_vehicle_ID: int = 121
    period: int = 0

    # Seed
    seed: int = 123


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------
def main():
    cfg = GAILConfig()
    set_global_seed(cfg.seed)

    # ---- Base env config (mirrors make_videos_multi.py) ----
    base_cfg = {
        "scene": cfg.scene,
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction"},
        "show_trajectories": False,

        # Frequencies / rendering (rendering flags won't hurt training)
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "screen_width": 400, "screen_height": 150, "scaling": 2.0,
        "offscreen_rendering": True,

        # Ego spawn
        "ego_speed": 30.0,
        "ego_lane_index": 1,
        "ego_longitudinal_m": 30.0,

        # Replay controls (your NGSimEnv uses replay_period & ego_vehicle_ID)
        "replay_period": cfg.period,
        "ego_vehicle_ID": cfg.ego_vehicle_ID,

        # Surroundings
        "spawn_radius_m": 120.0,
        "max_surrounding": 60,

        # RL-specific
        "max_episode_steps": cfg.max_episode_steps,
    }

    # ---- Dummy env to inspect obs/action spaces using SAME configure pattern ----
    dummy_env = gym.make(cfg.env_id)
    dummy_env.unwrapped.configure(base_cfg)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    if isinstance(obs_space, gym.spaces.Box):
        obs_dim = int(np.prod(obs_space.shape))  # flatten anything
    else:
        raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

    if not isinstance(act_space, gym.spaces.Discrete):
        raise ValueError(f"This script assumes Discrete actions, got {act_space}")

    n_actions = act_space.n
    dummy_env.close()

    print(f"Obs space: {obs_space}, flattened dim = {obs_dim}, n_actions = {n_actions}")

    # ---- Models ----
    policy = DiscretePolicyNet(obs_dim, n_actions, hidden_dim=cfg.hidden_dim).to(device)
    disc = DiscriminatorNet(obs_dim, n_actions, hidden_dim=cfg.hidden_dim).to(device)

    pi_opt = torch.optim.Adam(policy.parameters(), lr=cfg.pi_lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=cfg.disc_lr)

    # ---- Expert dataset from NGSIM ----
    expert_loader = load_expert_loader_from_ngsim(
        env_id=cfg.env_id,
        base_cfg=base_cfg,
        batch_size=cfg.disc_batch_size,
        scene=cfg.scene,
        ego_vehicle_ID=cfg.ego_vehicle_ID,
        period=cfg.period,
    )
    expert_iter = iter(expert_loader)

    # ---- Ray init + workers ----
    if not ray.is_initialized():
        ray.init()

    workers = [
        EnvWorker.remote(
            cfg.env_id,
            base_cfg,
            cfg.max_episode_steps,
            worker_seed=cfg.seed + wid,
        )
        for wid in range(cfg.num_workers)
    ]

    # ---- Training loop ----
    for it in range(cfg.total_iters):
        # 1) Collect rollouts from workers
        with torch.no_grad():
            cpu_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}

        futures = [
            w.run_rollouts.remote(cpu_state_dict, cfg.episodes_per_worker)
            for w in workers
        ]
        worker_batches = ray.get(futures)

        # Concatenate data
        obs_np = np.concatenate([b["obs"] for b in worker_batches], axis=0)
        acts_np = np.concatenate([b["acts"] for b in worker_batches], axis=0)
        dones_np = np.concatenate([b["dones"] for b in worker_batches], axis=0)

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        acts_t = torch.tensor(acts_np, dtype=torch.long, device=device)
        dones_t = torch.tensor(dones_np.astype(np.float32), dtype=torch.float32, device=device)

        # 2) Update discriminator (multiple batches per iter)
        for _ in range(cfg.disc_updates_per_iter):
            try:
                expert_states, expert_actions = next(expert_iter)
            except StopIteration:
                expert_iter = iter(expert_loader)
                expert_states, expert_actions = next(expert_iter)

            expert_states = expert_states.to(device)
            expert_actions = expert_actions.to(device)

            expert_actions_oh = one_hot(expert_actions, n_actions)
            policy_actions_oh = one_hot(acts_t, n_actions)

            disc_opt.zero_grad()

            # D(expert)
            d_exp = disc(expert_states, expert_actions_oh)
            # D(policy)
            d_pol = disc(obs_t.detach(), policy_actions_oh.detach())

            # GAIL discriminator loss
            loss_disc = -(
                torch.log(d_exp + 1e-8).mean()
                + torch.log(1.0 - d_pol + 1e-8).mean()
            )
            loss_disc.backward()
            disc_opt.step()

        # 3) Compute GAIL rewards
        with torch.no_grad():
            policy_actions_oh = one_hot(acts_t, n_actions)
            d_pol = disc(obs_t, policy_actions_oh).squeeze(-1)
            # r_GAIL(s,a) = -log(1 - D(s,a))
            rewards_t = -torch.log(1.0 - d_pol + 1e-8)

        # 4) REINFORCE with returns
        returns_t = compute_returns(rewards_t, dones_t.bool(), cfg.gamma)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        pi_opt.zero_grad()
        logits = policy(obs_t)
        dist = Categorical(logits=logits)
        logp_pi = dist.log_prob(acts_t)  # [N]
        loss_pi = -(logp_pi * returns_t).mean()
        loss_pi.backward()
        pi_opt.step()

        # 5) Logging
        with torch.no_grad():
            mean_disc_exp = d_exp.mean().item()
            mean_disc_pol = d_pol.mean().item()
            mean_reward = rewards_t.mean().item()

        print(
            f"Iter {it:04d} | "
            f"L_disc={loss_disc.item():.3f} | "
            f"L_pi={loss_pi.item():.3f} | "
            f"D(exp)={mean_disc_exp:.3f} | "
            f"D(pol)={mean_disc_pol:.3f} | "
            f"r_gail={mean_reward:.3f}"
        )

    print("✅ Training finished.")
    torch.save(policy.state_dict(), os.path.join(parent_dir, "gail_policy_discrete.pt"))
    torch.save(disc.state_dict(), os.path.join(parent_dir, "gail_discriminator_discrete.pt"))


if __name__ == "__main__":
    main()
