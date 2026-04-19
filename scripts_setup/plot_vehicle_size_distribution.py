#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from highway_env.ngsim_utils.constants import METERS_PER_FOOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot vehicle length/width distributions for the repository's US-101 "
            "(NGSIM) and Japanese prebuilt datasets."
        )
    )
    parser.add_argument(
        "--us-root",
        default="highway_env/data/processed_20s",
        help="Episode root containing us-101/prebuilt/trajectory_<split>.npy.",
    )
    parser.add_argument(
        "--japanese-root",
        default="highway_env/data/processed_10s",
        help="Episode root containing japanese/prebuilt/trajectory_<split>.npy.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Prebuilt split to read from each scene.",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/vehicle_dimensions",
        help="Directory where the plots and summary text will be written.",
    )
    return parser.parse_args()


def load_prebuilt(scene: str, episode_root: str, split: str) -> dict[str, dict[Any, Any]]:
    path = Path(episode_root) / scene / "prebuilt" / f"trajectory_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing prebuilt trajectory file: {path}")
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected prebuilt format in {path}")
    return data


def collect_vehicle_sizes(
    *,
    scene: str,
    episode_root: str,
    split: str,
    length_scale: float,
    width_scale: float,
) -> list[dict[str, float | int | str]]:
    traj_all = load_prebuilt(scene=scene, episode_root=episode_root, split=split)
    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[int, float, float]] = set()

    for episode_name, veh_dict in traj_all.items():
        del episode_name
        for vehicle_id, meta in veh_dict.items():
            length = float(meta.get("length", np.nan)) * float(length_scale)
            width = float(meta.get("width", np.nan)) * float(width_scale)
            if not (np.isfinite(length) and np.isfinite(width)):
                continue
            key = (int(vehicle_id), round(length, 6), round(width, 6))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "scene": scene,
                    "vehicle_id": int(vehicle_id),
                    "length_m": length,
                    "width_m": width,
                }
            )

    if not rows:
        raise RuntimeError(f"No vehicle size metadata found for scene={scene!r} split={split!r}.")
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    lengths = np.asarray([float(row["length_m"]) for row in rows], dtype=float)
    widths = np.asarray([float(row["width_m"]) for row in rows], dtype=float)
    areas = lengths * widths
    return {
        "count": float(len(rows)),
        "unique_sizes": float(len({(round(l, 4), round(w, 4)) for l, w in zip(lengths, widths)})),
        "length_min_m": float(np.min(lengths)),
        "length_max_m": float(np.max(lengths)),
        "length_mean_m": float(np.mean(lengths)),
        "width_min_m": float(np.min(widths)),
        "width_max_m": float(np.max(widths)),
        "width_mean_m": float(np.mean(widths)),
        "area_mean_m2": float(np.mean(areas)),
    }


def write_summary(out_dir: Path, summary_by_scene: dict[str, dict[str, float]], split: str) -> None:
    out_path = out_dir / f"vehicle_size_summary_{split}.txt"
    lines = [f"split={split}"]
    for scene, stats in summary_by_scene.items():
        lines.append(f"scene={scene}")
        lines.append(f"  count={int(stats['count'])}")
        lines.append(f"  unique_sizes={int(stats['unique_sizes'])}")
        lines.append(f"  length_min_m={stats['length_min_m']:.3f}")
        lines.append(f"  length_mean_m={stats['length_mean_m']:.3f}")
        lines.append(f"  length_max_m={stats['length_max_m']:.3f}")
        lines.append(f"  width_min_m={stats['width_min_m']:.3f}")
        lines.append(f"  width_mean_m={stats['width_mean_m']:.3f}")
        lines.append(f"  width_max_m={stats['width_max_m']:.3f}")
        lines.append(f"  area_mean_m2={stats['area_mean_m2']:.3f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scatter(
    out_dir: Path,
    rows_by_scene: dict[str, list[dict[str, float | int | str]]],
    summary_by_scene: dict[str, dict[str, float]],
    split: str,
) -> None:
    colors = {
        "us-101": "#1f77b4",
        "japanese": "#d95f02",
    }
    labels = {
        "us-101": "US-101 / NGSIM",
        "japanese": "Japanese",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})

    ax = axes[0]
    for scene, rows in rows_by_scene.items():
        lengths = [float(row["length_m"]) for row in rows]
        widths = [float(row["width_m"]) for row in rows]
        ax.scatter(
            lengths,
            widths,
            s=18,
            alpha=0.6,
            c=colors.get(scene, "0.5"),
            label=f"{labels.get(scene, scene)} (n={len(rows)})",
        )
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")
    ax.set_title(f"Vehicle Dimensions by Scene ({split})")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    bins = np.linspace(
        0.0,
        max(max(float(s["length_max_m"]) for s in summary_by_scene.values()), 1.0) + 0.5,
        28,
    )
    for scene, rows in rows_by_scene.items():
        lengths = [float(row["length_m"]) for row in rows]
        ax.hist(
            lengths,
            bins=bins,
            alpha=0.5,
            color=colors.get(scene, "0.5"),
            label=labels.get(scene, scene),
        )
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Vehicle Count")
    ax.set_title("Length Histogram")
    ax.grid(alpha=0.25)

    ax = axes[2]
    bins = np.linspace(
        0.0,
        max(max(float(s["width_max_m"]) for s in summary_by_scene.values()), 1.0) + 0.25,
        24,
    )
    for scene, rows in rows_by_scene.items():
        widths = [float(row["width_m"]) for row in rows]
        ax.hist(
            widths,
            bins=bins,
            alpha=0.5,
            color=colors.get(scene, "0.5"),
            label=labels.get(scene, scene),
        )
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Vehicle Count")
    ax.set_title("Width Histogram")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_dir / f"vehicle_size_distribution_{split}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_scene = {
        "us-101": collect_vehicle_sizes(
            scene="us-101",
            episode_root=str(args.us_root),
            split=str(args.split),
            length_scale=METERS_PER_FOOT,
            width_scale=METERS_PER_FOOT,
        ),
        "japanese": collect_vehicle_sizes(
            scene="japanese",
            episode_root=str(args.japanese_root),
            split=str(args.split),
            length_scale=1.0,
            width_scale=1.0,
        ),
    }
    summary_by_scene = {
        scene: summarize(rows)
        for scene, rows in rows_by_scene.items()
    }

    plot_scatter(
        out_dir=out_dir,
        rows_by_scene=rows_by_scene,
        summary_by_scene=summary_by_scene,
        split=str(args.split),
    )
    write_summary(out_dir=out_dir, summary_by_scene=summary_by_scene, split=str(args.split))

    for scene, stats in summary_by_scene.items():
        print(
            f"scene={scene} count={int(stats['count'])} unique_sizes={int(stats['unique_sizes'])} "
            f"length_range_m=[{stats['length_min_m']:.3f}, {stats['length_max_m']:.3f}] "
            f"width_range_m=[{stats['width_min_m']:.3f}, {stats['width_max_m']:.3f}]"
        )
    print(f"saved_plot={out_dir / f'vehicle_size_distribution_{args.split}.png'}")
    print(f"saved_summary={out_dir / f'vehicle_size_summary_{args.split}.txt'}")


if __name__ == "__main__":
    main()
