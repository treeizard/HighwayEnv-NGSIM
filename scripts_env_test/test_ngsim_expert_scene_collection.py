#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm.auto import tqdm


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from highway_env.imitation.expert_dataset import ENV_ID, build_env_config, register_ngsim_env  # noqa: E402
from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle  # noqa: E402
from highway_env.vehicle.behavior import IDMVehicle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight expert-control regression check for scene collection. "
            "Uses cheap Kinematics observations so we can validate controlled-vehicle "
            "activity windows and hidden-vehicle ghosting without lidar/camera cost."
        )
    )
    parser.add_argument("--scene", default="us-101")
    parser.add_argument("--episode-root", default="highway_env/data/processed_20s")
    parser.add_argument("--prebuilt-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episode-name", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-surrounding", default="all")
    parser.add_argument("--controlled-vehicles", type=int, default=4)
    parser.add_argument("--max-controlled-vehicles", type=int, default=0)
    parser.add_argument(
        "--control-all-vehicles",
        action="store_true",
        default=True,
        help="Check every viable controlled vehicle in the chosen scene.",
    )
    parser.add_argument(
        "--no-control-all-vehicles",
        dest="control_all_vehicles",
        action="store_false",
    )
    parser.add_argument(
        "--expert-control-mode",
        choices=["teleport", "continuous"],
        default="continuous",
    )
    parser.add_argument(
        "--allow-idm",
        action="store_true",
        default=False,
        help="Allow surrounding replay vehicles to hand over to IDM.",
    )
    parser.add_argument(
        "--no-allow-idm",
        dest="allow_idm",
        action="store_false",
    )
    parser.add_argument(
        "--controlled-min-occupancy",
        type=float,
        default=0.8,
        help="Minimum required active ratio for each controlled vehicle.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record the regression rollout to disk.",
    )
    parser.add_argument(
        "--video-dir",
        default="expert_data/videos",
        help="Output directory for recorded videos.",
    )
    parser.add_argument(
        "--video-prefix",
        default="ngsim_expert_scene_collection",
        help="Filename prefix for recorded videos.",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1200,
        help="Render width used for video/offscreen rendering.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=300,
        help="Render height used for video/offscreen rendering.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=2.0,
        help="Scene zoom factor. Larger values zoom in more.",
    )
    parser.add_argument(
        "--print-idm-handover",
        action="store_true",
        help='Print "vehicle id changed to idm" when a replay vehicle hands over to IDM.',
    )
    parser.add_argument(
        "--print-idm-handover-id",
        dest="print_idm_handover_ids",
        type=int,
        action="append",
        default=[],
        help="Only print IDM handover for the specified vehicle id. Repeatable.",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug/idm_scene_collection",
        help="Directory where IDM debug traces are written.",
    )
    parser.add_argument(
        "--no-debug-idm-log",
        dest="debug_idm_log",
        action="store_false",
        default=True,
        help="Disable writing IDM debug logs to disk.",
    )
    return parser.parse_args()


def light_observation_config() -> dict[str, Any]:
    return {
        "type": "Kinematics",
        "vehicles_count": 1,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
        "clip": False,
        "absolute": True,
        "include_obstacles": True,
    }


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = build_env_config(
        scene=str(args.scene),
        action_mode=str(args.expert_control_mode),
        episode_root=str(args.episode_root),
        prebuilt_split=str(args.prebuilt_split),
        controlled_vehicles=max(1, int(args.controlled_vehicles)),
        control_all_vehicles=bool(args.control_all_vehicles),
        max_surrounding=args.max_surrounding,
        observation_config=light_observation_config(),
        simulation_frequency=10,
        policy_frequency=10,
        max_episode_steps=int(args.max_steps),
        seed=None,
        scene_dataset_collection_mode=True,
        allow_idm=bool(args.allow_idm),
    )
    cfg["expert_test_mode"] = str(args.expert_control_mode) != "teleport"
    cfg["disable_controlled_vehicle_collisions"] = True
    cfg["terminate_when_all_controlled_crashed"] = False
    cfg["controlled_vehicle_min_occupancy"] = float(args.controlled_min_occupancy)
    cfg["screen_width"] = int(args.screen_width)
    cfg["screen_height"] = int(args.screen_height)
    cfg["scaling"] = float(args.scaling)
    cfg["offscreen_rendering"] = bool(args.save_video)
    cfg["debug_idm_handover"] = bool(args.print_idm_handover)
    if args.print_idm_handover_ids:
        cfg["debug_idm_handover_ids"] = [int(vehicle_id) for vehicle_id in args.print_idm_handover_ids]
    if args.episode_name:
        cfg["simulation_period"] = {"episode_name": str(args.episode_name)}
    if int(args.max_controlled_vehicles) > 0:
        cfg["max_controlled_vehicles"] = int(args.max_controlled_vehicles)
    return cfg


def make_env(args: argparse.Namespace) -> gym.Env:
    cfg = build_config(args)
    render_mode = "rgb_array" if bool(args.save_video) else None
    env = gym.make(ENV_ID, render_mode=render_mode, config=cfg)
    if bool(args.save_video):
        os.makedirs(str(args.video_dir), exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(args.video_dir),
            name_prefix=str(args.video_prefix),
            episode_trigger=lambda episode_id: True,
            disable_logger=True,
        )
    return env


def idle_action(env: gym.Env, num_agents: int) -> Any:
    if hasattr(env.action_space, "spaces"):
        return tuple(1 for _ in range(num_agents))
    return 1


def hidden_vehicle_issues(base_env) -> list[str]:
    issues: list[str] = []
    for vehicle in list(base_env.road.vehicles):
        is_scene_controlled_hidden = hasattr(vehicle, "scene_collection_is_active") and not bool(
            getattr(vehicle, "scene_collection_is_active", True)
        )
        is_replay_hidden = hasattr(vehicle, "appear") and not bool(getattr(vehicle, "appear", True))
        if not (is_scene_controlled_hidden or is_replay_hidden):
            continue

        position = np.asarray(getattr(vehicle, "position", np.zeros(2, dtype=float)), dtype=float)
        length = float(getattr(vehicle, "LENGTH", 0.0))
        width = float(getattr(vehicle, "WIDTH", 0.0))
        speed = float(getattr(vehicle, "speed", 0.0))
        if (
            not np.allclose(position[:2], 0.0, atol=1e-6)
            or abs(length) > 1e-6
            or abs(width) > 1e-6
            or abs(speed) > 1e-6
        ):
            issues.append(
                f"hidden vehicle id={getattr(vehicle, 'vehicle_ID', None)} "
                f"pos={position[:2].tolist()} L={length:.4f} W={width:.4f} speed={speed:.4f}"
            )
    return issues


def update_path_errors(base_env, path_errors: dict[int, list[float]]) -> None:
    step_index = int(getattr(base_env, "steps", 0))
    for vehicle in base_env.controlled_vehicles:
        vehicle_id = int(getattr(vehicle, "vehicle_ID", -1))
        if vehicle_id not in path_errors:
            continue
        if not bool(getattr(vehicle, "scene_collection_is_active", False)):
            continue

        traj = np.asarray(getattr(vehicle, "scene_collection_full_traj", ()), dtype=float)
        if traj.ndim != 2 or step_index >= len(traj):
            continue

        ref_row = np.asarray(traj[step_index], dtype=float)
        ref_xy = ref_row[:2]
        sim_xy = np.asarray(getattr(vehicle, "position", np.zeros(2, dtype=float)), dtype=float)[:2]
        err = float(np.linalg.norm(sim_xy - ref_xy))
        path_errors[vehicle_id].append(err)


def path_metric_summary(path_errors: dict[int, list[float]]) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    per_vehicle: dict[int, dict[str, float]] = {}
    ades: list[float] = []
    fdes: list[float] = []

    for vehicle_id, errors in path_errors.items():
        if not errors:
            per_vehicle[vehicle_id] = {"T": 0.0, "ADE_m": np.nan, "FDE_m": np.nan}
            continue
        err_arr = np.asarray(errors, dtype=float)
        ade = float(np.mean(err_arr))
        fde = float(err_arr[-1])
        per_vehicle[vehicle_id] = {
            "T": float(len(err_arr)),
            "ADE_m": ade,
            "FDE_m": fde,
        }
        ades.append(ade)
        fdes.append(fde)

    overall = {
        "avg_ADE_m": float(np.mean(ades)) if ades else np.nan,
        "avg_FDE_m": float(np.mean(fdes)) if fdes else np.nan,
        "max_ADE_m": float(np.max(ades)) if ades else np.nan,
        "max_FDE_m": float(np.max(fdes)) if fdes else np.nan,
    }
    return per_vehicle, overall


def _idm_debug_folder(args: argparse.Namespace, episode_name: str, episode_idx: int) -> Path:
    root = Path(str(args.debug_dir))
    suffix = f"episode_{episode_idx + 1}"
    return root / str(args.scene) / str(episode_name) / suffix


def _safe_round(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)


def _as_lane_str(lane_index) -> str:
    return "" if lane_index is None else str(tuple(lane_index))


def _vehicle_id(vehicle) -> int | str:
    vehicle_id = getattr(vehicle, "vehicle_ID", None)
    if vehicle_id is None:
        return ""
    try:
        return int(vehicle_id)
    except (TypeError, ValueError):
        return str(vehicle_id)


def _vehicle_type(vehicle) -> str:
    return "" if vehicle is None else type(vehicle).__name__


def _vehicle_mode(vehicle) -> str:
    if isinstance(vehicle, NGSIMVehicle):
        if bool(getattr(vehicle, "overtaken", False)):
            return "idm_mobil"
        if not bool(getattr(vehicle, "appear", True)):
            return "replay_hidden"
        return "replay_tracking"
    if hasattr(vehicle, "scene_collection_is_active"):
        return (
            "expert_tracking_active"
            if bool(getattr(vehicle, "scene_collection_is_active", False))
            else "expert_tracking_hidden"
        )
    if isinstance(vehicle, IDMVehicle):
        return "idm_mobil"
    return "other"


def _lane_gap(vehicle, other, lane_index=None) -> float | None:
    if vehicle is None or other is None:
        return None
    if lane_index is not None:
        lane = vehicle.road.network.get_lane(lane_index)
        return float(vehicle.lane_distance_to(other, lane=lane))
    return float(vehicle.lane_distance_to(other))


def _reference_row_for_vehicle(base_env, vehicle) -> dict[str, Any]:
    step_index = int(getattr(base_env, "steps", 0))
    out: dict[str, Any] = {
        "ref_step": step_index,
        "ref_x": "",
        "ref_y": "",
        "ref_speed": "",
        "ref_lane_id": "",
        "ref_error_m": "",
    }
    traj = None
    if hasattr(vehicle, "scene_collection_full_traj"):
        traj = np.asarray(getattr(vehicle, "scene_collection_full_traj"), dtype=float)
    elif isinstance(vehicle, NGSIMVehicle) and getattr(vehicle, "ngsim_traj", None) is not None:
        traj = np.asarray(getattr(vehicle, "ngsim_traj"), dtype=float)
        step_index = int(getattr(vehicle, "sim_steps", step_index))
        out["ref_step"] = step_index

    if traj is None or traj.ndim != 2 or step_index < 0 or step_index >= len(traj):
        return out

    row = np.asarray(traj[step_index], dtype=float)
    out["ref_x"] = _safe_round(float(row[0]), 4)
    out["ref_y"] = _safe_round(float(row[1]), 4)
    out["ref_speed"] = _safe_round(float(row[2]), 4) if row.size > 2 else ""
    out["ref_lane_id"] = int(row[3]) if row.size > 3 and np.isfinite(row[3]) else ""
    position = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=float)[:2]
    out["ref_error_m"] = _safe_round(float(np.linalg.norm(position - row[:2])), 6)
    return out


def _debug_vehicle_state(base_env, vehicle) -> dict[str, Any]:
    ref = _reference_row_for_vehicle(base_env, vehicle)
    row = {
        "sim_step": int(getattr(base_env, "steps", 0)),
        "vehicle_id": _vehicle_id(vehicle),
        "vehicle_type": _vehicle_type(vehicle),
        "mode": _vehicle_mode(vehicle),
        "overtaken": bool(getattr(vehicle, "overtaken", False)),
        "appear": bool(getattr(vehicle, "appear", True)),
        "active": bool(getattr(vehicle, "scene_collection_is_active", True)),
        "crashed": bool(getattr(vehicle, "crashed", False)),
        "remove_from_road": bool(getattr(vehicle, "remove_from_road", False)),
        "position_x": _safe_round(float(vehicle.position[0]), 4),
        "position_y": _safe_round(float(vehicle.position[1]), 4),
        "heading": _safe_round(float(getattr(vehicle, "heading", 0.0)), 6),
        "speed": _safe_round(float(getattr(vehicle, "speed", 0.0)), 4),
        "target_speed": _safe_round(float(getattr(vehicle, "target_speed", 0.0)), 4),
        "length": _safe_round(float(getattr(vehicle, "LENGTH", 0.0)), 4),
        "width": _safe_round(float(getattr(vehicle, "WIDTH", 0.0)), 4),
        "lane_index": _as_lane_str(getattr(vehicle, "lane_index", None)),
        "target_lane_index": _as_lane_str(getattr(vehicle, "target_lane_index", None)),
        "action_acceleration": _safe_round(float(getattr(vehicle, "action", {}).get("acceleration", 0.0)), 6),
        "action_steering": _safe_round(float(getattr(vehicle, "action", {}).get("steering", 0.0)), 6),
        "handover_step": "" if getattr(vehicle, "idm_handover_step", None) is None else int(vehicle.idm_handover_step),
        "handover_reason": "" if getattr(vehicle, "idm_handover_reason", None) is None else str(vehicle.idm_handover_reason),
    }
    row.update(ref)
    return row


def _debug_idm_decision_state(base_env, vehicle: IDMVehicle) -> dict[str, Any]:
    decision = getattr(vehicle, "_last_idm_decision", {}) or {}
    current_front = decision.get("current_front")
    current_rear = decision.get("current_rear")
    target_front = decision.get("target_front")
    target_rear = decision.get("target_rear")
    current_gap = _lane_gap(vehicle, current_front)
    target_lane_index = decision.get("target_lane_index", getattr(vehicle, "target_lane_index", None))
    target_gap = _lane_gap(vehicle, target_front, target_lane_index)
    return {
        "sim_step": int(getattr(base_env, "steps", 0)),
        "vehicle_id": _vehicle_id(vehicle),
        "vehicle_type": _vehicle_type(vehicle),
        "mode": _vehicle_mode(vehicle),
        "lane_index": _as_lane_str(decision.get("lane_index", getattr(vehicle, "lane_index", None))),
        "target_lane_index": _as_lane_str(target_lane_index),
        "current_front_id": _vehicle_id(current_front),
        "current_front_type": _vehicle_type(current_front),
        "current_front_gap": _safe_round(current_gap, 6),
        "current_front_speed": "" if current_front is None else _safe_round(float(current_front.speed), 4),
        "current_rear_id": _vehicle_id(current_rear),
        "current_rear_type": _vehicle_type(current_rear),
        "target_front_id": _vehicle_id(target_front),
        "target_front_type": _vehicle_type(target_front),
        "target_front_gap": _safe_round(target_gap, 6),
        "target_front_speed": "" if target_front is None else _safe_round(float(target_front.speed), 4),
        "target_rear_id": _vehicle_id(target_rear),
        "target_rear_type": _vehicle_type(target_rear),
        "current_acceleration": _safe_round(decision.get("current_acceleration"), 6),
        "target_acceleration": _safe_round(decision.get("target_acceleration"), 6),
        "applied_acceleration": _safe_round(decision.get("applied_acceleration"), 6),
        "applied_steering": _safe_round(decision.get("applied_steering"), 6),
        "action_acceleration": _safe_round(float(getattr(vehicle, "action", {}).get("acceleration", 0.0)), 6),
        "action_steering": _safe_round(float(getattr(vehicle, "action", {}).get("steering", 0.0)), 6),
    }


def _debug_mobil_decision_rows(base_env, vehicle: IDMVehicle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in list(getattr(vehicle, "_last_mobil_decisions", []) or []):
        new_preceding = decision.get("new_preceding")
        new_following = decision.get("new_following")
        old_preceding = decision.get("old_preceding")
        old_following = decision.get("old_following")
        candidate_lane = decision.get("candidate_lane_index")
        rows.append(
            {
                "sim_step": int(getattr(base_env, "steps", 0)),
                "vehicle_id": _vehicle_id(vehicle),
                "vehicle_type": _vehicle_type(vehicle),
                "mode": _vehicle_mode(vehicle),
                "lane_index": _as_lane_str(getattr(vehicle, "lane_index", None)),
                "target_lane_index": _as_lane_str(getattr(vehicle, "target_lane_index", None)),
                "candidate_lane_index": _as_lane_str(candidate_lane),
                "accepted": bool(decision.get("accepted", False)),
                "reason": str(decision.get("reason", "")),
                "new_preceding_id": _vehicle_id(new_preceding),
                "new_preceding_type": _vehicle_type(new_preceding),
                "new_preceding_gap": _safe_round(_lane_gap(vehicle, new_preceding, candidate_lane), 6),
                "new_following_id": _vehicle_id(new_following),
                "new_following_type": _vehicle_type(new_following),
                "old_preceding_id": _vehicle_id(old_preceding),
                "old_preceding_type": _vehicle_type(old_preceding),
                "old_following_id": _vehicle_id(old_following),
                "old_following_type": _vehicle_type(old_following),
                "self_pred_a": _safe_round(decision.get("self_pred_a"), 6),
                "self_current_a": _safe_round(decision.get("self_current_a"), 6),
                "new_following_a": _safe_round(decision.get("new_following_a"), 6),
                "new_following_pred_a": _safe_round(decision.get("new_following_pred_a"), 6),
                "old_following_a": _safe_round(decision.get("old_following_a"), 6),
                "old_following_pred_a": _safe_round(decision.get("old_following_pred_a"), 6),
                "jerk": _safe_round(decision.get("jerk"), 6),
            }
        )
    return rows


def _debug_idm_state(base_env, vehicle: NGSIMVehicle) -> dict[str, Any]:
    current_front, current_rear = base_env.road.neighbour_vehicles(vehicle, vehicle.lane_index)
    current_gap = None if current_front is None else float(vehicle.lane_distance_to(current_front))
    current_desired_gap = (
        None if current_front is None else float(vehicle.desired_gap(vehicle, current_front))
    )
    current_safe = (
        None
        if current_front is None or current_gap is None
        else float(vehicle._required_braking_to_avoid_contact(vehicle, current_front, current_gap))
    )
    current_acc = float(vehicle.acceleration(vehicle, front_vehicle=current_front, rear_vehicle=current_rear))
    v0 = max(float(getattr(vehicle, "target_speed", 0.0)), float(vehicle.params.desired_speed))
    v = max(float(vehicle.speed), 0.0)
    a_free = float(vehicle.params.a_comf * (1.0 - (v / max(v0, 1e-9)) ** float(vehicle.params.delta)))
    a_int = None
    if current_front is not None and current_gap is not None:
        s_star = float(vehicle.desired_gap(vehicle, current_front, projected=True))
        a_int = -float(vehicle.params.a_comf * (s_star / max(abs(current_gap), 1e-9)) ** 2)

    target_front = target_rear = None
    target_gap = target_desired_gap = target_safe = target_acc = None
    if vehicle.target_lane_index is not None and vehicle.target_lane_index != vehicle.lane_index:
        target_front, target_rear = base_env.road.neighbour_vehicles(vehicle, vehicle.target_lane_index)
        target_acc = float(vehicle.acceleration(vehicle, front_vehicle=target_front, rear_vehicle=target_rear))
        if target_front is not None:
            target_lane = base_env.road.network.get_lane(vehicle.target_lane_index)
            target_gap = float(vehicle.lane_distance_to(target_front, lane=target_lane))
            target_desired_gap = float(vehicle.desired_gap(vehicle, target_front))
            target_safe = float(
                vehicle._required_braking_to_avoid_contact(vehicle, target_front, target_gap)
            )

    cause_parts: list[str] = []
    action_acc = float(vehicle.action.get("acceleration", 0.0))
    if bool(vehicle.crashed):
        cause_parts.append("crashed")
    if action_acc < -1e-6:
        if current_safe is not None and current_safe < -1e-6 and abs(current_acc - current_safe) < 1e-3:
            cause_parts.append("current_lane_safety_brake")
        elif current_front is not None and current_gap is not None and current_desired_gap is not None:
            if current_gap <= current_desired_gap + 0.5:
                cause_parts.append("current_lane_following")
        if target_acc is not None and target_acc < current_acc - 1e-3:
            cause_parts.append("target_lane_conservative_brake")
    if vehicle.target_lane_index != vehicle.lane_index:
        cause_parts.append("pending_lane_change")
    if not cause_parts and action_acc >= -1e-6:
        cause_parts.append("not_decelerating")

    return {
        "sim_step": int(getattr(base_env, "steps", 0)),
        "vehicle_id": int(getattr(vehicle, "vehicle_ID", -1)),
        "overtaken": bool(getattr(vehicle, "overtaken", False)),
        "crashed": bool(getattr(vehicle, "crashed", False)),
        "position_x": _safe_round(float(vehicle.position[0]), 4),
        "position_y": _safe_round(float(vehicle.position[1]), 4),
        "speed": _safe_round(float(vehicle.speed), 4),
        "action_acceleration": _safe_round(action_acc, 6),
        "action_steering": _safe_round(float(vehicle.action.get("steering", 0.0)), 6),
        "calc_acc_current_lane": _safe_round(current_acc, 6),
        "a_free": _safe_round(a_free, 6),
        "a_int": _safe_round(a_int, 6),
        "current_front_id": "" if current_front is None else int(getattr(current_front, "vehicle_ID", -1)),
        "current_front_type": "" if current_front is None else type(current_front).__name__,
        "current_gap": _safe_round(current_gap, 6),
        "current_desired_gap": _safe_round(current_desired_gap, 6),
        "current_safe_brake": _safe_round(current_safe, 6),
        "target_lane_acc": _safe_round(target_acc, 6),
        "target_front_id": "" if target_front is None else int(getattr(target_front, "vehicle_ID", -1)),
        "target_front_type": "" if target_front is None else type(target_front).__name__,
        "target_gap": _safe_round(target_gap, 6),
        "target_desired_gap": _safe_round(target_desired_gap, 6),
        "target_safe_brake": _safe_round(target_safe, 6),
        "lane_index": _as_lane_str(vehicle.lane_index),
        "target_lane_index": _as_lane_str(vehicle.target_lane_index),
        "handover_step": "" if getattr(vehicle, "idm_handover_step", None) is None else int(vehicle.idm_handover_step),
        "handover_reason": "" if getattr(vehicle, "idm_handover_reason", None) is None else str(vehicle.idm_handover_reason),
        "cause": "|".join(cause_parts),
    }


def _write_debug_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_idm_debug_logs(
    debug_dir: Path,
    rows: list[dict[str, Any]],
    *,
    vehicle_rows: list[dict[str, Any]] | None = None,
    idm_decision_rows: list[dict[str, Any]] | None = None,
    mobil_decision_rows: list[dict[str, Any]] | None = None,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    _write_debug_csv(debug_dir / "idm_overtaken_trace.csv", rows)
    _write_debug_csv(debug_dir / "vehicle_trajectory_trace.csv", vehicle_rows or [])
    _write_debug_csv(debug_dir / "idm_decision_trace.csv", idm_decision_rows or [])
    _write_debug_csv(debug_dir / "mobil_decision_trace.csv", mobil_decision_rows or [])

    manifest = {
        "idm_overtaken_trace.csv": "Legacy per-step trace for NGSIM replay vehicles after IDM handover.",
        "vehicle_trajectory_trace.csv": "All road vehicles each step: replay, IDM/MOBIL, and expert tracking vehicles with reference positions where available.",
        "idm_decision_trace.csv": "Latest IDM longitudinal/lateral command context for every IDMVehicle, including front/rear vehicles used on current and target lanes.",
        "mobil_decision_trace.csv": "Every MOBIL candidate lane evaluated in IDMVehicle.mobil(), including accept/reject reason and referenced vehicles.",
        "idm_summary.json": "Per-vehicle summary derived from idm_overtaken_trace.csv.",
        "idm_summary_preview.json": "Top summary entries for quick inspection.",
    }
    with (debug_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summary_by_vehicle: dict[int, dict[str, Any]] = {}
    for row in rows:
        vehicle_id = int(row["vehicle_id"])
        entry = summary_by_vehicle.setdefault(
            vehicle_id,
            {
                "vehicle_id": vehicle_id,
                "handover_step": row["handover_step"],
                "handover_reason": row["handover_reason"],
                "negative_accel_steps": 0,
                "crash_steps": 0,
                "pending_lane_change_steps": 0,
                "cause_counts": {},
                "first_negative_accel": None,
                "last_state": row,
            },
        )
        entry["last_state"] = row
        cause_parts = [part for part in str(row["cause"]).split("|") if part]
        for part in cause_parts:
            entry["cause_counts"][part] = int(entry["cause_counts"].get(part, 0)) + 1
        if float(row["action_acceleration"] or 0.0) < -1e-6:
            entry["negative_accel_steps"] += 1
            if entry["first_negative_accel"] is None:
                entry["first_negative_accel"] = row
        if bool(row["crashed"]):
            entry["crash_steps"] += 1
        if str(row["lane_index"]) != str(row["target_lane_index"]):
            entry["pending_lane_change_steps"] += 1

    ranked = sorted(
        summary_by_vehicle.values(),
        key=lambda item: (
            -int(item["negative_accel_steps"]),
            -int(item["crash_steps"]),
            int(item["vehicle_id"]),
        ),
    )
    with (debug_dir / "idm_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(ranked, handle, indent=2)

    preview_lines = []
    for item in ranked[:20]:
        preview_lines.append(
            {
                "vehicle_id": item["vehicle_id"],
                "handover_step": item["handover_step"],
                "handover_reason": item["handover_reason"],
                "negative_accel_steps": item["negative_accel_steps"],
                "crash_steps": item["crash_steps"],
                "pending_lane_change_steps": item["pending_lane_change_steps"],
                "top_causes": item["cause_counts"],
                "first_negative_accel": item["first_negative_accel"],
            }
        )
    with (debug_dir / "idm_summary_preview.json").open("w", encoding="utf-8") as handle:
        json.dump(preview_lines, handle, indent=2)


def run_episode(env: gym.Env, episode_idx: int, args: argparse.Namespace) -> None:
    obs, info = env.reset(seed=int(args.seed) + int(episode_idx))
    del obs, info

    base = env.unwrapped
    controlled_ids = [int(v.vehicle_ID) for v in base.controlled_vehicles]
    active_counts = {vehicle_id: 0 for vehicle_id in controlled_ids}
    path_errors = {vehicle_id: [] for vehicle_id in controlled_ids}
    total_steps = 0
    hidden_issues: list[str] = []
    idm_debug_rows: list[dict[str, Any]] = []
    vehicle_debug_rows: list[dict[str, Any]] = []
    idm_decision_rows: list[dict[str, Any]] = []
    mobil_decision_rows: list[dict[str, Any]] = []

    done = False
    progress = tqdm(
        total=int(args.max_steps),
        desc=f"episode {episode_idx + 1}",
        leave=False,
    )
    try:
        while not done and total_steps < int(args.max_steps):
            action = idle_action(env, len(base.controlled_vehicles))
            _obs, _reward, terminated, truncated, info = env.step(action)
            del _obs, _reward
            total_steps += 1
            progress.update(1)

            alive_ids = {int(v) for v in info.get("alive_controlled_vehicle_ids", []) if v is not None}
            progress.set_postfix(
                alive=len(alive_ids),
                controlled=len(controlled_ids),
                refresh=False,
            )
            for vehicle_id in alive_ids:
                if vehicle_id in active_counts:
                    active_counts[vehicle_id] += 1

            update_path_errors(base, path_errors)
            hidden_issues.extend(hidden_vehicle_issues(base))
            if bool(args.debug_idm_log):
                for vehicle in list(base.road.vehicles):
                    vehicle_debug_rows.append(_debug_vehicle_state(base, vehicle))
                    if isinstance(vehicle, IDMVehicle):
                        idm_decision_rows.append(_debug_idm_decision_state(base, vehicle))
                        mobil_decision_rows.extend(_debug_mobil_decision_rows(base, vehicle))
                    if isinstance(vehicle, NGSIMVehicle) and bool(getattr(vehicle, "overtaken", False)):
                        idm_debug_rows.append(_debug_idm_state(base, vehicle))
            done = bool(terminated or truncated)
    finally:
        progress.close()

    print(
        f"\nepisode={base.episode_name} controlled={len(controlled_ids)} "
        f"steps={total_steps} ids={controlled_ids}"
    )
    threshold = float(args.controlled_min_occupancy)
    per_vehicle_path_metrics, overall_path_metrics = path_metric_summary(path_errors)
    for vehicle_id in controlled_ids:
        ratio = (float(active_counts[vehicle_id]) / float(max(total_steps, 1)))
        status = "OK" if ratio >= threshold else "FAIL"
        metrics = per_vehicle_path_metrics[vehicle_id]
        ade_str = f"{metrics['ADE_m']:.4f}" if np.isfinite(metrics["ADE_m"]) else "nan"
        fde_str = f"{metrics['FDE_m']:.4f}" if np.isfinite(metrics["FDE_m"]) else "nan"
        print(
            f"  controlled_id={vehicle_id} active_steps={active_counts[vehicle_id]} "
            f"ratio={ratio:.3f} threshold={threshold:.3f} status={status} "
            f"ADE_m={ade_str} FDE_m={fde_str}"
        )

    avg_ade_str = (
        f"{overall_path_metrics['avg_ADE_m']:.4f}"
        if np.isfinite(overall_path_metrics["avg_ADE_m"])
        else "nan"
    )
    avg_fde_str = (
        f"{overall_path_metrics['avg_FDE_m']:.4f}"
        if np.isfinite(overall_path_metrics["avg_FDE_m"])
        else "nan"
    )
    max_ade_str = (
        f"{overall_path_metrics['max_ADE_m']:.4f}"
        if np.isfinite(overall_path_metrics["max_ADE_m"])
        else "nan"
    )
    max_fde_str = (
        f"{overall_path_metrics['max_FDE_m']:.4f}"
        if np.isfinite(overall_path_metrics["max_FDE_m"])
        else "nan"
    )
    print(
        f"  path_summary avg_ADE_m={avg_ade_str} avg_FDE_m={avg_fde_str} "
        f"max_ADE_m={max_ade_str} max_FDE_m={max_fde_str}"
    )

    deduped_hidden_issues = list(dict.fromkeys(hidden_issues))
    if deduped_hidden_issues:
        print("  hidden-state issues:")
        for line in deduped_hidden_issues[:20]:
            print(f"    {line}")
        if len(deduped_hidden_issues) > 20:
            print(f"    ... {len(deduped_hidden_issues) - 20} more")
    else:
        print("  hidden-state issues: none")

    if bool(args.debug_idm_log):
        debug_dir = _idm_debug_folder(args, str(base.episode_name), episode_idx)
        _write_idm_debug_logs(
            debug_dir,
            idm_debug_rows,
            vehicle_rows=vehicle_debug_rows,
            idm_decision_rows=idm_decision_rows,
            mobil_decision_rows=mobil_decision_rows,
        )
        print(f"  idm_debug_dir={debug_dir}")


def main() -> None:
    args = parse_args()
    register_ngsim_env()
    env = make_env(args)
    try:
        for episode_idx in range(int(args.episodes)):
            run_episode(env, episode_idx, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
