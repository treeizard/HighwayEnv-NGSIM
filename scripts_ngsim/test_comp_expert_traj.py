# scripts/compare_expert_vs_replay_traj_multi.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.registration import register

# Make sure project root is importable
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Register env (safe if re-run)
try:
    register(id="NGSim-US101-v0", entry_point="highway_env.envs.ngsim_env:NGSimEnv")
except Exception:
    pass


def polyline_mse_xy(ref_xy: np.ndarray, rep_xy: np.ndarray, n_samples: int = 200) -> float:
    """Arc-length resampled XY MSE so different step counts are comparable."""
    def resample(xy, n):
        xy = np.asarray(xy, dtype=float)
        if len(xy) < 2:
            return np.repeat(xy[:1], n, axis=0)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 1e-9:
            return np.repeat(xy[:1], n, axis=0)
        t = np.linspace(0, s[-1], n)
        x = np.interp(t, s, xy[:, 0])
        y = np.interp(t, s, xy[:, 1])
        return np.stack([x, y], axis=1)

    r1 = resample(ref_xy, n_samples)
    r2 = resample(rep_xy, n_samples)
    return float(np.mean(np.sum((r1 - r2) ** 2, axis=1)))


def run_one(cfg: dict, seed: int, max_steps: int = 300) -> dict:
    """Run one expert-replay episode and return ref/replay XY and MSE."""
    print("seed:", seed)
    env = gym.make("NGSim-US101-v0", config=cfg)
    env.reset(seed=seed)
    uenv = env.unwrapped

    for _ in range(max_steps):
        # action doesn't matter in expert_test_mode if env overrides it
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break

    ref = np.asarray(getattr(uenv, "_expert_ref_xy_pol", []), dtype=float)
    rep = np.asarray(getattr(uenv, "_replay_xy_pol", []), dtype=float)

    # If you already log t=0 before stepping, DO NOT drop last element.
    # If replay is post-step only, drop last to align lengths.
    # Robust handling: just trim to min length without assuming off-by-one.
    if ref.size == 0 or rep.size == 0:
        env.close()
        return dict(ref=ref, rep=rep, mse=np.nan, seed=seed)

    T = min(len(ref), len(rep))
    ref, rep = ref[:T], rep[:T]

    mse = polyline_mse_xy(ref, rep)
    env.close()
    return dict(ref=ref, rep=rep, mse=mse, seed=seed)


def plot_grid(runs: list[dict], nrows: int = 4, ncols: int = 5) -> None:
    """Plot expert vs replay per episode in a tight 4x5 grid."""
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, 10),
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(-1)

    # Tight spacing (avoid constrained_layout)
    fig.subplots_adjust(
        left=0.04,
        right=0.995,
        bottom=0.06,
        top=0.92,
        wspace=0.12,
        hspace=0.18,
    )

    for i, ax in enumerate(axes):
        if i >= len(runs):
            ax.axis("off")
            continue

        r = runs[i]
        ref = r["ref"]
        rep = r["rep"]

        ax.plot(ref[:, 0], ref[:, 1], lw=2, label="expert")
        ax.plot(rep[:, 0], rep[:, 1], "--", lw=2, label="replay")

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        mse = r["mse"]
        mse_str = "nan" if not np.isfinite(mse) else f"{mse:.2e}"
        ax.set_title(f"Seed {r['seed']} | MSE={mse_str}", fontsize=9, pad=2)

        # Outer labels only
        if i // ncols == nrows - 1:
            ax.set_xlabel("x [m]")
        if i % ncols == 0:
            ax.set_ylabel("y [m]")

        # Put a small legend only in the first panel (no global legend = less whitespace)
        if i == 0:
            ax.legend(loc="best", fontsize=9, frameon=True)

    fig.suptitle("Expert vs Replay Trajectories (per episode)", fontsize=16, y=0.98)


def plot_mse_summary(runs: list[dict]) -> None:
    """Plot MSE per run."""
    mses = np.array([r["mse"] for r in runs], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mses, marker="o")
    ax.set_xlabel("run index")
    ax.set_ylabel("MSE (XY, arc-length)")
    if np.any(np.isfinite(mses)):
        ax.set_title(f"MSE summary | mean={np.nanmean(mses):.2e}, median={np.nanmedian(mses):.2e}")
    else:
        ax.set_title("MSE summary")
    ax.grid(True)


def main():
    cfg = {
        "scene": "us-101",
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "episode_root": "highway_env/data/processed_10s",
        "max_surrounding": 0,
        "expert_test_mode": True,
        # Optional: if you are saving videos or using render_mode, set here
        # "render_mode": "rgb_array",
    }

    runs = [run_one(cfg, seed=i, max_steps=300) for i in range(20)]

    plot_grid(runs, nrows=4, ncols=5)
    plot_mse_summary(runs)

    # Make sure windows actually appear
    plt.show(block=True)


if __name__ == "__main__":
    main()
