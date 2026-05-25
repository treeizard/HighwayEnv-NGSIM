#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from highway_env import utils  # noqa: E402
from highway_env.envs import ngsim_env as ngsim_env_module  # noqa: E402
from highway_env.envs.ngsim_env import NGSimEnv  # noqa: E402
from highway_env.ngsim_utils.core.constants import FEET_PER_METER  # noqa: E402
from highway_env.ngsim_utils.data.episode_selection import build_trajectory_set  # noqa: E402
from highway_env.ngsim_utils.data.prebuilt import load_prebuilt_data  # noqa: E402
from highway_env.ngsim_utils.data.trajectory_gen import (  # noqa: E402
    first_valid_index,
    process_raw_trajectory,
    trajectory_row_is_active,
)
from highway_env.ngsim_utils.road import lane_mapping  # noqa: E402
from highway_env.ngsim_utils.road.gen_road import create_japanese_road, create_ngsim_101_road  # noqa: E402
from highway_env.ngsim_utils.road.lane_mapping import (  # noqa: E402
    heading_from_trajectory_row,
    target_lane_index_from_lane_id,
)
from highway_env.ngsim_utils.vehicles import ego_factory, replay  # noqa: E402
from highway_env.ngsim_utils.vehicles.ego import EgoVehicle  # noqa: E402
from highway_env.ngsim_utils.vehicles.replay import NGSIMVehicle, road_entity_pose_polygon  # noqa: E402


Mapper = Callable[[Any, str, float, int], tuple[str, str, int] | None]


@dataclass(frozen=True)
class ScanConfig:
    scene: str = "japanese"
    episode_root: str = "highway_env/data/processed_20s"
    prebuilt_split: str = "train"
    episode_name: str = "t1577840400000"
    ego_vehicle_id: int | None = 2586
    max_steps: int = 90
    max_surrounding: str | int | None = "all"
    allow_idm: bool = True
    seed: int = 0
    percentage_controlled_vehicles: float = 1.0
    controlled_vehicle_min_occupancy: float = 0.8
    vehicle_ids: tuple[int, ...] = ()
    stop_after_first: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose active vehicle polygon overlaps in NGSIM/Japanese replay scenes. "
            "Run with the local conda environment, e.g. "
            "`conda run -n ngsim_env python scripts_env_test/diagnose_japanese_replay_overlaps.py`."
        )
    )
    parser.add_argument("--scene", default="japanese")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episode-name", default="t1577840400000")
    parser.add_argument("--ego-vehicle-id", type=int, default=2586)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--percentage-controlled-vehicles", type=float, default=1.0)
    parser.add_argument("--controlled-min-occupancy", type=float, default=0.8)
    parser.add_argument("--allow-idm", action="store_true", default=True)
    parser.add_argument("--no-allow-idm", dest="allow_idm", action="store_false")
    parser.add_argument(
        "--vehicle-id",
        type=int,
        action="append",
        default=[],
        help="Restrict reported overlap pairs to these vehicle ids. Repeat for multiple ids.",
    )
    parser.add_argument("--out-dir", default="artifacts/replay_overlap_diagnostics")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--skip-env-scan", action="store_true")
    parser.add_argument("--skip-raw-scan", action="store_true")
    parser.add_argument("--compare-lane3-boundaries", action="store_true", default=True)
    parser.add_argument("--no-compare-lane3-boundaries", dest="compare_lane3_boundaries", action="store_false")
    parser.add_argument(
        "--inspect-pair",
        type=int,
        nargs=2,
        default=[3165, 3167],
        metavar=("VEHICLE_A", "VEHICLE_B"),
    )
    parser.add_argument("--inspect-from-step", type=int, default=26)
    parser.add_argument("--inspect-to-step", type=int, default=32)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--sweep-episodes-per-split", type=int, default=10)
    parser.add_argument("--stop-after-first", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        scene=str(args.scene),
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.prebuilt_split),
        episode_name=str(args.episode_name),
        ego_vehicle_id=int(args.ego_vehicle_id) if args.ego_vehicle_id is not None else None,
        max_steps=int(args.max_steps),
        max_surrounding=_parse_max_surrounding(args.max_surrounding),
        allow_idm=bool(args.allow_idm),
        seed=int(args.seed),
        percentage_controlled_vehicles=float(args.percentage_controlled_vehicles),
        controlled_vehicle_min_occupancy=float(args.controlled_min_occupancy),
        vehicle_ids=tuple(int(vehicle_id) for vehicle_id in args.vehicle_id),
        stop_after_first=bool(args.stop_after_first),
    )


def _parse_max_surrounding(value: Any) -> str | int | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() == "none":
        return None
    if text.lower() == "all":
        return "all"
    return int(text)


def active_vehicle(vehicle: object) -> bool:
    return (
        not bool(getattr(vehicle, "remove_from_road", False))
        and bool(getattr(vehicle, "appear", True))
        and bool(getattr(vehicle, "visible", True))
        and float(getattr(vehicle, "LENGTH", 0.0)) > 0.0
        and float(getattr(vehicle, "WIDTH", 0.0)) > 0.0
    )


def inactive_render_risk(vehicle: object) -> bool:
    """Return whether a non-active vehicle still has a visible-looking footprint."""
    if active_vehicle(vehicle):
        return False
    position = getattr(vehicle, "position", None)
    if position is None or not np.all(np.isfinite(np.asarray(position, dtype=float))):
        return False
    return bool(
        bool(getattr(vehicle, "visible", True))
        and float(getattr(vehicle, "LENGTH", 0.0)) > 0.0
        and float(getattr(vehicle, "WIDTH", 0.0)) > 0.0
    )


def vehicle_id(vehicle: object) -> int | None:
    raw = getattr(vehicle, "vehicle_ID", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _json_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _array_list(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return [float(x) for x in arr]


def _vehicle_row_at(vehicle: object, row_index: int | None) -> list[float] | None:
    traj = getattr(vehicle, "ngsim_traj", None)
    if traj is None or row_index is None:
        return None
    if row_index < 0 or row_index >= len(traj):
        return None
    return _array_list(np.asarray(traj[row_index], dtype=float)[:4])


def current_replay_row_index(vehicle: object) -> int | None:
    traj = getattr(vehicle, "ngsim_traj", None)
    if traj is None or len(traj) == 0:
        return None
    sim_steps = int(getattr(vehicle, "sim_steps", 0))
    position = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=float)
    candidates = range(max(0, sim_steps - 3), min(len(traj), sim_steps + 2))
    best_idx = None
    best_dist = float("inf")
    for idx in candidates:
        row = np.asarray(traj[idx], dtype=float)
        if not trajectory_row_is_active(row):
            continue
        dist = float(np.linalg.norm(row[:2] - position))
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    if best_idx is not None and best_dist <= 1e-4:
        return int(best_idx)
    if bool(getattr(vehicle, "overtaken", False)):
        return int(np.clip(sim_steps, 0, len(traj) - 1))
    return int(np.clip(max(0, sim_steps - 1), 0, len(traj) - 1))


def polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray) -> tuple[bool, float]:
    intersecting, _will_intersect, translation = utils.are_polygons_intersecting(
        poly_a,
        poly_b,
        np.zeros(2, dtype=float),
        np.zeros(2, dtype=float),
    )
    penetration = 0.0 if translation is None else float(np.linalg.norm(translation))
    return bool(intersecting), penetration


def pair_geometry_metrics(
    pos_a: np.ndarray,
    heading_a: float,
    length_a: float,
    width_a: float,
    pos_b: np.ndarray,
    length_b: float,
    width_b: float,
) -> dict[str, float]:
    delta = np.asarray(pos_b, dtype=float) - np.asarray(pos_a, dtype=float)
    forward = np.array([np.cos(float(heading_a)), np.sin(float(heading_a))], dtype=float)
    lateral = np.array([-forward[1], forward[0]], dtype=float)
    longitudinal_sep = abs(float(delta.dot(forward)))
    lateral_sep = abs(float(delta.dot(lateral)))
    longitudinal_gap = longitudinal_sep - 0.5 * (float(length_a) + float(length_b))
    lateral_gap = lateral_sep - 0.5 * (float(width_a) + float(width_b))
    return {
        "center_distance": float(np.linalg.norm(delta)),
        "longitudinal_gap_estimate": float(longitudinal_gap),
        "longitudinal_overlap_estimate": float(max(0.0, -longitudinal_gap)),
        "lateral_gap_estimate": float(lateral_gap),
        "lateral_overlap_estimate": float(max(0.0, -lateral_gap)),
    }


def _pair_filter_allows(config: ScanConfig, id_a: int | None, id_b: int | None) -> bool:
    if not config.vehicle_ids:
        return True
    wanted = set(config.vehicle_ids)
    return id_a in wanted and id_b in wanted


def _env_config(config: ScanConfig, *, split: str | None = None, episode_name: str | None = None, ego_id: int | None = None) -> dict[str, Any]:
    return {
        "scene": config.scene,
        "episode_root": config.episode_root,
        "prebuilt_split": split or config.prebuilt_split,
        "seed": config.seed,
        "ego_vehicle_ID": ego_id if ego_id is not None else config.ego_vehicle_id,
        "percentage_controlled_vehicles": config.percentage_controlled_vehicles,
        "max_surrounding": config.max_surrounding,
        "show_trajectories": False,
        "allow_idm": bool(config.allow_idm),
        "offscreen_rendering": True,
        "controlled_vehicle_min_occupancy": config.controlled_vehicle_min_occupancy,
        "simulation_period": {"episode_name": episode_name or config.episode_name},
    }


def idle_action(env: NGSimEnv) -> Any:
    action_space = env.action_space
    if hasattr(action_space, "spaces"):
        return tuple(np.zeros(space.shape, dtype=np.float32) for space in action_space.spaces)
    return np.zeros(action_space.shape, dtype=np.float32)


def classify_env_overlap(vehicle_a: object, vehicle_b: object, raw_intersection: bool) -> str:
    replay_pair = isinstance(vehicle_a, NGSIMVehicle) and isinstance(vehicle_b, NGSIMVehicle)
    both_replaying = replay_pair and not bool(getattr(vehicle_a, "overtaken", False)) and not bool(
        getattr(vehicle_b, "overtaken", False)
    )
    if both_replaying and raw_intersection:
        return "raw-data"
    if both_replaying:
        return "mapping/replay"
    if bool(getattr(vehicle_a, "overtaken", False)) or bool(getattr(vehicle_b, "overtaken", False)):
        return "post-handover simulation"
    return "environment"


def env_overlap_record(step: int, vehicle_a: object, vehicle_b: object, penetration: float) -> dict[str, Any]:
    row_idx_a = current_replay_row_index(vehicle_a)
    row_idx_b = current_replay_row_index(vehicle_b)
    raw_row_a = _vehicle_row_at(vehicle_a, row_idx_a)
    raw_row_b = _vehicle_row_at(vehicle_b, row_idx_b)
    raw_intersection = False
    if raw_row_a is not None and raw_row_b is not None:
        raw_intersection, _raw_penetration = raw_rows_intersect(
            raw_row_a,
            float(getattr(vehicle_a, "LENGTH", 0.0)),
            float(getattr(vehicle_a, "WIDTH", 0.0)),
            raw_row_b,
            float(getattr(vehicle_b, "LENGTH", 0.0)),
            float(getattr(vehicle_b, "WIDTH", 0.0)),
            mapper=target_lane_index_from_lane_id,
            scene=str(getattr(vehicle_a, "SCENE", "japanese")),
        )
    metrics = pair_geometry_metrics(
        np.asarray(getattr(vehicle_a, "position"), dtype=float),
        float(getattr(vehicle_a, "heading", 0.0)),
        float(getattr(vehicle_a, "LENGTH", 0.0)),
        float(getattr(vehicle_a, "WIDTH", 0.0)),
        np.asarray(getattr(vehicle_b, "position"), dtype=float),
        float(getattr(vehicle_b, "LENGTH", 0.0)),
        float(getattr(vehicle_b, "WIDTH", 0.0)),
    )
    return {
        "source": "env",
        "classification": classify_env_overlap(vehicle_a, vehicle_b, raw_intersection),
        "step": int(step),
        "vehicle_id_a": vehicle_id(vehicle_a),
        "vehicle_id_b": vehicle_id(vehicle_b),
        "class_a": type(vehicle_a).__name__,
        "class_b": type(vehicle_b).__name__,
        "row_index_a": row_idx_a,
        "row_index_b": row_idx_b,
        "raw_row_a": raw_row_a,
        "raw_row_b": raw_row_b,
        "position_a": _array_list(getattr(vehicle_a, "position")),
        "position_b": _array_list(getattr(vehicle_b, "position")),
        "heading_a": float(getattr(vehicle_a, "heading", 0.0)),
        "heading_b": float(getattr(vehicle_b, "heading", 0.0)),
        "speed_a": float(getattr(vehicle_a, "speed", 0.0)),
        "speed_b": float(getattr(vehicle_b, "speed", 0.0)),
        "lane_id_a": None if raw_row_a is None else int(raw_row_a[3]),
        "lane_id_b": None if raw_row_b is None else int(raw_row_b[3]),
        "lane_index_a": str(getattr(vehicle_a, "lane_index", None)),
        "lane_index_b": str(getattr(vehicle_b, "lane_index", None)),
        "length_a": float(getattr(vehicle_a, "LENGTH", 0.0)),
        "length_b": float(getattr(vehicle_b, "LENGTH", 0.0)),
        "width_a": float(getattr(vehicle_a, "WIDTH", 0.0)),
        "width_b": float(getattr(vehicle_b, "WIDTH", 0.0)),
        "appear_a": bool(getattr(vehicle_a, "appear", True)),
        "appear_b": bool(getattr(vehicle_b, "appear", True)),
        "visible_a": bool(getattr(vehicle_a, "visible", True)),
        "visible_b": bool(getattr(vehicle_b, "visible", True)),
        "overtaken_a": bool(getattr(vehicle_a, "overtaken", False)),
        "overtaken_b": bool(getattr(vehicle_b, "overtaken", False)),
        "crashed_a": bool(getattr(vehicle_a, "crashed", False)),
        "crashed_b": bool(getattr(vehicle_b, "crashed", False)),
        "polygon_intersection": True,
        "raw_row_polygon_intersection": bool(raw_intersection),
        "penetration_norm": float(penetration),
        **metrics,
    }


def run_env_overlap_scan(config: ScanConfig) -> dict[str, Any]:
    env = NGSimEnv(_env_config(config))
    overlap_rows: list[dict[str, Any]] = []
    step_summary: list[dict[str, Any]] = []
    render_risks: list[dict[str, Any]] = []
    try:
        env.reset()
        action = idle_action(env)
        for step in range(int(config.max_steps)):
            active = [vehicle for vehicle in env.road.vehicles if active_vehicle(vehicle)]
            inactive_risks = [vehicle for vehicle in env.road.vehicles if inactive_render_risk(vehicle)]
            hits = []
            for vehicle_a, vehicle_b in _candidate_pairs(active):
                id_a, id_b = vehicle_id(vehicle_a), vehicle_id(vehicle_b)
                if not _pair_filter_allows(config, id_a, id_b):
                    continue
                intersecting, penetration = polygons_intersect(vehicle_a.polygon(), vehicle_b.polygon())
                if not intersecting:
                    continue
                row = env_overlap_record(step, vehicle_a, vehicle_b, penetration)
                overlap_rows.append(row)
                hits.append(row)

            step_summary.append(
                {
                    "step": int(step),
                    "road_vehicle_count": int(len(env.road.vehicles)),
                    "active_vehicle_count": int(len(active)),
                    "overlap_count": int(len(hits)),
                    "inactive_render_risk_count": int(len(inactive_risks)),
                }
            )
            for vehicle in inactive_risks:
                render_risks.append(
                    {
                        "step": int(step),
                        "vehicle_id": vehicle_id(vehicle),
                        "class": type(vehicle).__name__,
                        "position": _array_list(getattr(vehicle, "position")),
                        "length": float(getattr(vehicle, "LENGTH", 0.0)),
                        "width": float(getattr(vehicle, "WIDTH", 0.0)),
                        "appear": bool(getattr(vehicle, "appear", True)),
                        "visible": bool(getattr(vehicle, "visible", True)),
                    }
                )
            if config.stop_after_first and hits:
                break
            env.step(action)
    finally:
        env.close()

    return {
        "kind": "env",
        "config": _dataclass_json(config),
        "overlaps": overlap_rows,
        "step_summary": step_summary,
        "render_risks": render_risks,
    }


def _candidate_pairs(vehicles: list[object]) -> Iterable[tuple[object, object]]:
    for idx, vehicle_a in enumerate(vehicles):
        diagonal_a = float(getattr(vehicle_a, "diagonal", np.hypot(getattr(vehicle_a, "LENGTH", 0.0), getattr(vehicle_a, "WIDTH", 0.0))))
        position_a = np.asarray(getattr(vehicle_a, "position"), dtype=float)
        for vehicle_b in vehicles[idx + 1 :]:
            diagonal_b = float(getattr(vehicle_b, "diagonal", np.hypot(getattr(vehicle_b, "LENGTH", 0.0), getattr(vehicle_b, "WIDTH", 0.0))))
            if np.linalg.norm(position_a - np.asarray(getattr(vehicle_b, "position"), dtype=float)) > 0.5 * (
                diagonal_a + diagonal_b
            ):
                continue
            yield vehicle_a, vehicle_b


def _dataclass_json(config: ScanConfig) -> dict[str, Any]:
    return {
        "scene": config.scene,
        "episode_root": config.episode_root,
        "prebuilt_split": config.prebuilt_split,
        "episode_name": config.episode_name,
        "ego_vehicle_id": config.ego_vehicle_id,
        "max_steps": config.max_steps,
        "max_surrounding": config.max_surrounding,
        "allow_idm": config.allow_idm,
        "seed": config.seed,
        "percentage_controlled_vehicles": config.percentage_controlled_vehicles,
        "controlled_vehicle_min_occupancy": config.controlled_vehicle_min_occupancy,
        "vehicle_ids": list(config.vehicle_ids),
        "stop_after_first": config.stop_after_first,
    }


def road_network_for_scene(scene: str):
    if str(scene) == "japanese":
        return create_japanese_road()
    if str(scene) == "us-101":
        return create_ngsim_101_road()
    raise ValueError(f"Unsupported scene for diagnostics: {scene!r}")


def load_episode_trajectory_set(config: ScanConfig, *, split: str | None = None, episode_name: str | None = None, ego_id: int | None = None):
    cache: dict[Any, Any] = {}
    _prebuilt_dir, valid_ids_by_episode, traj_all_by_episode, _episodes = load_prebuilt_data(
        config.episode_root,
        config.scene,
        split or config.prebuilt_split,
        min_occupancy=float(config.controlled_vehicle_min_occupancy),
        cache=cache,
    )
    selected_episode = episode_name or config.episode_name
    valid_ids = [int(vehicle_id) for vehicle_id in valid_ids_by_episode[selected_episode]]
    selected_ego = int(ego_id if ego_id is not None else (config.ego_vehicle_id if config.ego_vehicle_id is not None else valid_ids[0]))
    if selected_ego not in valid_ids:
        raise ValueError(f"Ego vehicle {selected_ego} is not valid for episode {selected_episode}")
    trajectory_set = build_trajectory_set(traj_all_by_episode, selected_episode, [selected_ego])
    return trajectory_set, valid_ids


def resolve_shared_start_index(trajectory_set: dict[Any, Any]) -> int:
    ego_records = trajectory_set["ego"]
    starts = []
    for meta in ego_records.values():
        start = first_valid_index(np.asarray(meta["trajectory"], dtype=float))
        if start is None:
            raise RuntimeError("Selected ego has no active trajectory rows")
        starts.append(int(start))
    return max(starts) if starts else 0


def dimensions_for_scene(meta: dict[str, Any], scene: str) -> tuple[float, float]:
    length = float(meta.get("length", 0.0))
    width = float(meta.get("width", 0.0))
    if str(scene) == "us-101":
        return length / FEET_PER_METER, width / FEET_PER_METER
    return length, width


def selected_surrounding_records(config: ScanConfig, trajectory_set: dict[Any, Any]) -> list[tuple[int, dict[str, Any], np.ndarray]]:
    shared_start = resolve_shared_start_index(trajectory_set)
    ego_anchor_positions = []
    for ego_meta in trajectory_set.get("ego", {}).values():
        ego_traj_full = process_raw_trajectory(ego_meta["trajectory"], config.scene)
        ego_traj = ego_traj_full[shared_start:]
        first_idx = first_valid_index(ego_traj)
        if first_idx is not None:
            ego_anchor_positions.append(np.asarray(ego_traj[first_idx][:2], dtype=float))

    candidates = []
    for raw_vid, meta in trajectory_set.items():
        if raw_vid == "ego":
            continue
        vid = int(raw_vid)
        traj_full = process_raw_trajectory(meta["trajectory"], config.scene)
        if len(traj_full) <= shared_start:
            continue
        traj = traj_full[shared_start:]
        first_idx = first_valid_index(traj)
        if first_idx is None:
            continue
        first_pos = np.asarray(traj[first_idx][:2], dtype=float)
        priority = (
            min(float(np.linalg.norm(first_pos - ego_pos)) for ego_pos in ego_anchor_positions)
            if ego_anchor_positions
            else float("inf")
        )
        candidates.append((priority, vid, meta, traj))

    if config.max_surrounding == "all" or config.max_surrounding is None:
        return [(vid, meta, traj) for _priority, vid, meta, traj in candidates]
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(vid, meta, traj) for _priority, vid, meta, traj in candidates[: int(config.max_surrounding)]]


def heading_with_mapper(
    net: Any,
    scene: str,
    row: Any,
    *,
    next_row: Any | None,
    mapper: Mapper,
    fallback_heading: float = 0.0,
) -> float:
    row_arr = np.asarray(row, dtype=float)
    mapped_lane_index = mapper(net, scene, float(row_arr[0]), int(row_arr[3]))
    if mapped_lane_index is not None:
        lane = net.get_lane(mapped_lane_index)
        local_s, _local_r = lane.local_coordinates(row_arr[:2])
        return float(lane.heading_at(local_s))
    return motion_heading(row_arr, next_row, fallback_heading=fallback_heading)


def motion_heading(row: Any, next_row: Any | None, *, fallback_heading: float = 0.0) -> float:
    row_arr = np.asarray(row, dtype=float)
    if next_row is not None and trajectory_row_is_active(next_row):
        next_arr = np.asarray(next_row, dtype=float)
        delta = next_arr[:2] - row_arr[:2]
        if float(np.linalg.norm(delta)) > 1e-3:
            return float(np.arctan2(float(delta[1]), float(delta[0])))
    return float(fallback_heading)


def raw_rows_intersect(
    row_a: Any,
    length_a: float,
    width_a: float,
    row_b: Any,
    length_b: float,
    width_b: float,
    *,
    mapper: Mapper,
    scene: str,
    net: Any | None = None,
    next_row_a: Any | None = None,
    next_row_b: Any | None = None,
) -> tuple[bool, float]:
    net = net or road_network_for_scene(scene)
    row_a_arr = np.asarray(row_a, dtype=float)
    row_b_arr = np.asarray(row_b, dtype=float)
    heading_a = heading_with_mapper(net, scene, row_a_arr, next_row=next_row_a, mapper=mapper)
    heading_b = heading_with_mapper(net, scene, row_b_arr, next_row=next_row_b, mapper=mapper)
    poly_a = road_entity_pose_polygon(row_a_arr[:2], heading_a, float(length_a), float(width_a))
    poly_b = road_entity_pose_polygon(row_b_arr[:2], heading_b, float(length_b), float(width_b))
    return polygons_intersect(poly_a, poly_b)


def run_raw_overlap_scan(config: ScanConfig, *, mapper: Mapper = target_lane_index_from_lane_id, label: str = "raw-current") -> dict[str, Any]:
    trajectory_set, _valid_ids = load_episode_trajectory_set(config)
    records = selected_surrounding_records(config, trajectory_set)
    net = road_network_for_scene(config.scene)
    horizon = max((len(traj) for _vid, _meta, traj in records), default=0)
    max_steps = min(int(config.max_steps), int(horizon))
    overlap_rows: list[dict[str, Any]] = []
    step_summary: list[dict[str, Any]] = []

    for step in range(max_steps):
        active_rows = []
        for vid, meta, traj in records:
            if step >= len(traj):
                continue
            row = np.asarray(traj[step], dtype=float)
            if not trajectory_row_is_active(row):
                continue
            length, width = dimensions_for_scene(meta, config.scene)
            next_row = traj[step + 1] if step + 1 < len(traj) else None
            heading = heading_with_mapper(net, config.scene, row, next_row=next_row, mapper=mapper)
            lane_index = mapper(net, config.scene, float(row[0]), int(row[3]))
            poly = road_entity_pose_polygon(row[:2], heading, length, width)
            active_rows.append(
                {
                    "vehicle_id": int(vid),
                    "meta": meta,
                    "row": row,
                    "next_row": next_row,
                    "length": length,
                    "width": width,
                    "heading": heading,
                    "lane_index": lane_index,
                    "polygon": poly,
                    "diagonal": float(np.hypot(length, width)),
                }
            )

        hits = []
        for idx, vehicle_a in enumerate(active_rows):
            for vehicle_b in active_rows[idx + 1 :]:
                if not _pair_filter_allows(config, vehicle_a["vehicle_id"], vehicle_b["vehicle_id"]):
                    continue
                if np.linalg.norm(vehicle_a["row"][:2] - vehicle_b["row"][:2]) > 0.5 * (
                    vehicle_a["diagonal"] + vehicle_b["diagonal"]
                ):
                    continue
                intersecting, penetration = polygons_intersect(vehicle_a["polygon"], vehicle_b["polygon"])
                if not intersecting:
                    continue
                metrics = pair_geometry_metrics(
                    vehicle_a["row"][:2],
                    vehicle_a["heading"],
                    vehicle_a["length"],
                    vehicle_a["width"],
                    vehicle_b["row"][:2],
                    vehicle_b["length"],
                    vehicle_b["width"],
                )
                row = {
                    "source": label,
                    "classification": "raw-data",
                    "step": int(step),
                    "vehicle_id_a": int(vehicle_a["vehicle_id"]),
                    "vehicle_id_b": int(vehicle_b["vehicle_id"]),
                    "class_a": "raw",
                    "class_b": "raw",
                    "row_index_a": int(step),
                    "row_index_b": int(step),
                    "raw_row_a": _array_list(vehicle_a["row"][:4]),
                    "raw_row_b": _array_list(vehicle_b["row"][:4]),
                    "position_a": _array_list(vehicle_a["row"][:2]),
                    "position_b": _array_list(vehicle_b["row"][:2]),
                    "heading_a": float(vehicle_a["heading"]),
                    "heading_b": float(vehicle_b["heading"]),
                    "speed_a": float(vehicle_a["row"][2]),
                    "speed_b": float(vehicle_b["row"][2]),
                    "lane_id_a": int(vehicle_a["row"][3]),
                    "lane_id_b": int(vehicle_b["row"][3]),
                    "lane_index_a": str(vehicle_a["lane_index"]),
                    "lane_index_b": str(vehicle_b["lane_index"]),
                    "length_a": float(vehicle_a["length"]),
                    "length_b": float(vehicle_b["length"]),
                    "width_a": float(vehicle_a["width"]),
                    "width_b": float(vehicle_b["width"]),
                    "appear_a": True,
                    "appear_b": True,
                    "visible_a": True,
                    "visible_b": True,
                    "overtaken_a": False,
                    "overtaken_b": False,
                    "crashed_a": False,
                    "crashed_b": False,
                    "polygon_intersection": True,
                    "raw_row_polygon_intersection": True,
                    "penetration_norm": float(penetration),
                    **metrics,
                }
                overlap_rows.append(row)
                hits.append(row)

        step_summary.append(
            {
                "step": int(step),
                "active_vehicle_count": int(len(active_rows)),
                "overlap_count": int(len(hits)),
            }
        )
        if config.stop_after_first and hits:
            break

    return {
        "kind": label,
        "config": _dataclass_json(config),
        "overlaps": overlap_rows,
        "step_summary": step_summary,
    }


def japanese_lane3_315_mapper(net: Any, scene: str, x: float, lane_id: int) -> tuple[str, str, int] | None:
    if str(scene) == "japanese" and int(lane_id) == 3:
        x_m = float(x)
        if x_m < 150.0:
            return ("j", "b", 0)
        if x_m < 315.0:
            return ("b", "c", 2)
        return ("c", "d", 1)
    return target_lane_index_from_lane_id(net, scene, x, lane_id)


@contextmanager
def patched_japanese_lane3_315_mapping():
    originals = {
        lane_mapping: lane_mapping.target_lane_index_from_lane_id,
        replay: replay.target_lane_index_from_lane_id,
        ego_factory: ego_factory.target_lane_index_from_lane_id,
        ngsim_env_module: ngsim_env_module.target_lane_index_from_lane_id,
    }
    try:
        for module in originals:
            module.target_lane_index_from_lane_id = japanese_lane3_315_mapper
        yield
    finally:
        for module, original in originals.items():
            module.target_lane_index_from_lane_id = original


def compare_lane3_boundary_rows() -> list[dict[str, Any]]:
    net = create_japanese_road()
    rows = []
    for x in (149.0, 150.0, 259.9, 260.0, 276.7, 314.9, 315.0):
        rows.append(
            {
                "x": float(x),
                "current": str(target_lane_index_from_lane_id(net, "japanese", x, 3)),
                "road_315_boundary": str(japanese_lane3_315_mapper(net, "japanese", x, 3)),
            }
        )
    return rows


def inspect_pair_geometry_window(
    config: ScanConfig,
    pair: tuple[int, int],
    start_step: int,
    end_step: int,
) -> list[dict[str, Any]]:
    trajectory_set, _valid_ids = load_episode_trajectory_set(config)
    records = {vid: (meta, traj) for vid, meta, traj in selected_surrounding_records(config, trajectory_set)}
    if pair[0] not in records or pair[1] not in records:
        return []
    meta_a, traj_a = records[pair[0]]
    meta_b, traj_b = records[pair[1]]
    length_a, width_a = dimensions_for_scene(meta_a, config.scene)
    length_b, width_b = dimensions_for_scene(meta_b, config.scene)
    net = road_network_for_scene(config.scene)
    rows = []

    for step in range(int(start_step), int(end_step) + 1):
        if step >= len(traj_a) or step >= len(traj_b):
            continue
        row_a = np.asarray(traj_a[step], dtype=float)
        row_b = np.asarray(traj_b[step], dtype=float)
        if not trajectory_row_is_active(row_a) or not trajectory_row_is_active(row_b):
            continue
        next_a = traj_a[step + 1] if step + 1 < len(traj_a) else None
        next_b = traj_b[step + 1] if step + 1 < len(traj_b) else None
        variants: dict[str, tuple[float, float, str, str]] = {
            "current_mapped": (
                heading_with_mapper(net, config.scene, row_a, next_row=next_a, mapper=target_lane_index_from_lane_id),
                heading_with_mapper(net, config.scene, row_b, next_row=next_b, mapper=target_lane_index_from_lane_id),
                str(target_lane_index_from_lane_id(net, config.scene, float(row_a[0]), int(row_a[3]))),
                str(target_lane_index_from_lane_id(net, config.scene, float(row_b[0]), int(row_b[3]))),
            ),
            "motion_derived": (
                motion_heading(row_a, next_a),
                motion_heading(row_b, next_b),
                "motion",
                "motion",
            ),
            "lane3_315_boundary": (
                heading_with_mapper(net, config.scene, row_a, next_row=next_a, mapper=japanese_lane3_315_mapper),
                heading_with_mapper(net, config.scene, row_b, next_row=next_b, mapper=japanese_lane3_315_mapper),
                str(japanese_lane3_315_mapper(net, config.scene, float(row_a[0]), int(row_a[3]))),
                str(japanese_lane3_315_mapper(net, config.scene, float(row_b[0]), int(row_b[3]))),
            ),
        }
        for variant, (heading_a, heading_b, lane_a, lane_b) in variants.items():
            poly_a = road_entity_pose_polygon(row_a[:2], heading_a, length_a, width_a)
            poly_b = road_entity_pose_polygon(row_b[:2], heading_b, length_b, width_b)
            intersects, penetration = polygons_intersect(poly_a, poly_b)
            metrics = pair_geometry_metrics(row_a[:2], heading_a, length_a, width_a, row_b[:2], length_b, width_b)
            rows.append(
                {
                    "step": int(step),
                    "variant": variant,
                    "vehicle_id_a": int(pair[0]),
                    "vehicle_id_b": int(pair[1]),
                    "raw_row_a": _array_list(row_a[:4]),
                    "raw_row_b": _array_list(row_b[:4]),
                    "heading_a": float(heading_a),
                    "heading_b": float(heading_b),
                    "lane_index_a": lane_a,
                    "lane_index_b": lane_b,
                    "length_a": float(length_a),
                    "width_a": float(width_a),
                    "length_b": float(length_b),
                    "width_b": float(width_b),
                    "polygon_intersection": bool(intersects),
                    "penetration_norm": float(penetration),
                    **metrics,
                }
            )
    return rows


def boundary_comparison_scan(config: ScanConfig) -> dict[str, Any]:
    current = run_env_overlap_scan(config)
    with patched_japanese_lane3_315_mapping():
        patched = run_env_overlap_scan(config)
    return {
        "kind": "lane3_boundary_comparison",
        "current": summarize_overlap_result(current),
        "lane3_315_boundary": summarize_overlap_result(patched),
        "lane3_boundary_rows": compare_lane3_boundary_rows(),
    }


def summarize_overlap_result(result: dict[str, Any]) -> dict[str, Any]:
    step_summary = result.get("step_summary", [])
    overlaps = result.get("overlaps", [])
    return {
        "overlap_rows": int(len(overlaps)),
        "frames_with_overlaps": int(sum(1 for row in step_summary if int(row.get("overlap_count", 0)) > 0)),
        "max_overlaps_per_frame": int(max((int(row.get("overlap_count", 0)) for row in step_summary), default=0)),
        "first_overlap_step": next((int(row.get("step", 0)) for row in step_summary if int(row.get("overlap_count", 0)) > 0), None),
        "first_overlap": overlaps[0] if overlaps else None,
    }


def write_report(out_dir: Path, prefix: str, result: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    json_path = out_dir / f"{prefix}_{result['kind']}.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    paths["json"] = str(json_path)

    overlaps = result.get("overlaps")
    if isinstance(overlaps, list):
        csv_path = out_dir / f"{prefix}_{result['kind']}_overlaps.csv"
        write_csv(csv_path, overlaps)
        paths["overlaps_csv"] = str(csv_path)

    step_summary = result.get("step_summary")
    if isinstance(step_summary, list):
        summary_path = out_dir / f"{prefix}_{result['kind']}_steps.csv"
        write_csv(summary_path, step_summary)
        paths["steps_csv"] = str(summary_path)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return _json_value(value)
    return value


def default_prefix(config: ScanConfig) -> str:
    ego = "auto" if config.ego_vehicle_id is None else str(config.ego_vehicle_id)
    return f"{config.scene}_{config.prebuilt_split}_{config.episode_name}_ego{ego}"


def run_sweep(base_config: ScanConfig, splits: list[str], episodes_per_split: int) -> dict[str, Any]:
    rows = []
    for split in splits:
        cache: dict[Any, Any] = {}
        _prebuilt_dir, valid_ids_by_episode, _traj_all_by_episode, episodes = load_prebuilt_data(
            base_config.episode_root,
            base_config.scene,
            split,
            min_occupancy=float(base_config.controlled_vehicle_min_occupancy),
            cache=cache,
        )
        for episode_name in episodes[: int(episodes_per_split)]:
            valid_ids = [int(vehicle_id) for vehicle_id in valid_ids_by_episode[episode_name]]
            if not valid_ids:
                continue
            config = ScanConfig(
                scene=base_config.scene,
                episode_root=base_config.episode_root,
                prebuilt_split=split,
                episode_name=str(episode_name),
                ego_vehicle_id=valid_ids[0],
                max_steps=base_config.max_steps,
                max_surrounding=base_config.max_surrounding,
                allow_idm=base_config.allow_idm,
                seed=base_config.seed,
                percentage_controlled_vehicles=base_config.percentage_controlled_vehicles,
                controlled_vehicle_min_occupancy=base_config.controlled_vehicle_min_occupancy,
                vehicle_ids=base_config.vehicle_ids,
                stop_after_first=base_config.stop_after_first,
            )
            result = run_env_overlap_scan(config)
            summary = summarize_overlap_result(result)
            rows.append(
                {
                    "split": split,
                    "episode_name": str(episode_name),
                    "ego_vehicle_id": int(valid_ids[0]),
                    **summary,
                }
            )
    rows.sort(
        key=lambda row: (
            -int(row["max_overlaps_per_frame"]),
            -int(row["frames_with_overlaps"]),
            str(row["split"]),
            str(row["episode_name"]),
        )
    )
    return {"kind": "sweep", "config": _dataclass_json(base_config), "rows": rows}


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    out_dir = Path(args.out_dir)
    prefix = args.prefix or default_prefix(config)
    written: dict[str, dict[str, str]] = {}

    if not args.skip_env_scan:
        env_result = run_env_overlap_scan(config)
        written["env"] = write_report(out_dir, prefix, env_result)
        print("env_summary", json.dumps(summarize_overlap_result(env_result), sort_keys=True))

    if not args.skip_raw_scan:
        raw_result = run_raw_overlap_scan(config)
        written["raw"] = write_report(out_dir, prefix, raw_result)
        print("raw_summary", json.dumps(summarize_overlap_result(raw_result), sort_keys=True))

    if args.inspect_pair:
        pair_rows = inspect_pair_geometry_window(
            config,
            (int(args.inspect_pair[0]), int(args.inspect_pair[1])),
            int(args.inspect_from_step),
            int(args.inspect_to_step),
        )
        pair_result = {
            "kind": "pair_geometry",
            "config": _dataclass_json(config),
            "rows": pair_rows,
        }
        pair_path = out_dir / f"{prefix}_pair_geometry.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        pair_path.write_text(json.dumps(pair_result, indent=2, sort_keys=True), encoding="utf-8")
        write_csv(out_dir / f"{prefix}_pair_geometry.csv", pair_rows)
        written["pair_geometry"] = {"json": str(pair_path)}
        print("pair_geometry_rows", len(pair_rows))

    if args.compare_lane3_boundaries and config.scene == "japanese":
        comparison = boundary_comparison_scan(config)
        comparison_path = out_dir / f"{prefix}_lane3_boundary_comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        written["lane3_boundary_comparison"] = {"json": str(comparison_path)}
        print("lane3_boundary_comparison", json.dumps(comparison, sort_keys=True))

    if args.sweep:
        sweep = run_sweep(config, [str(split) for split in args.sweep_splits], int(args.sweep_episodes_per_split))
        sweep_path = out_dir / f"{prefix}_sweep.json"
        sweep_path.write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
        write_csv(out_dir / f"{prefix}_sweep.csv", sweep["rows"])
        written["sweep"] = {"json": str(sweep_path)}
        print("sweep_rows", len(sweep["rows"]))

    print("written", json.dumps(written, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
