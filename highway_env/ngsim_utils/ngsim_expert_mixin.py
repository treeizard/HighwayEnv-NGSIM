from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from highway_env.envs.common.action import Action
from highway_env.ngsim_utils.constants import MAX_ACCEL, MAX_STEER
from highway_env.ngsim_utils.ego_vehicle import EgoVehicle


logger = logging.getLogger(__name__)


class NGSimExpertMixin:
    def _expert_action_strings(self) -> list[str]:
        if not hasattr(self.action_type, "actions"):
            return ["IDLE"]
        return [self.action_type.actions[i] for i in sorted(self.action_type.actions)]

    def _planning_vehicle_clone(self) -> EgoVehicle:
        vehicle = self.vehicle
        sim_vehicle = EgoVehicle(
            road=None,
            position=np.array(vehicle.position, dtype=float),
            heading=float(vehicle.heading),
            speed=float(vehicle.speed),
            target_speed=float(getattr(vehicle, "target_speed", vehicle.speed)),
            route=copy.deepcopy(getattr(vehicle, "route", None)),
            control_mode=str(getattr(vehicle, "control_mode", "discrete")),
            target_speeds=np.array(
                getattr(vehicle, "target_speeds", EgoVehicle.DEFAULT_TARGET_SPEEDS),
                dtype=float,
            ),
            lateral_offset_step=float(
                getattr(
                    vehicle,
                    "lateral_offset_step",
                    EgoVehicle.DEFAULT_LATERAL_OFFSET_STEP,
                )
            ),
            lateral_offset_max=float(
                getattr(
                    vehicle,
                    "lateral_offset_max",
                    EgoVehicle.DEFAULT_LATERAL_OFFSET_MAX,
                )
            ),
        )
        sim_vehicle.set_ego_dimension(
            width=float(getattr(vehicle, "WIDTH", sim_vehicle.WIDTH)),
            length=float(getattr(vehicle, "LENGTH", sim_vehicle.LENGTH)),
        )
        sim_vehicle.target_lane_index = copy.deepcopy(
            getattr(vehicle, "target_lane_index", None)
        )
        sim_vehicle.lane_index = copy.deepcopy(getattr(vehicle, "lane_index", None))
        sim_vehicle.lane = self.road.network.get_lane(sim_vehicle.lane_index)
        sim_vehicle.target_speed = float(
            getattr(vehicle, "target_speed", sim_vehicle.target_speed)
        )
        sim_vehicle.speed_index = int(
            getattr(vehicle, "speed_index", sim_vehicle.speed_index)
        )
        sim_vehicle.lateral_offset = float(getattr(vehicle, "lateral_offset", 0.0))
        sim_vehicle._last_low_level_action = dict(
            getattr(
                vehicle,
                "_last_low_level_action",
                {"steering": 0.0, "acceleration": 0.0},
            )
        )
        sim_vehicle.action = dict(
            getattr(vehicle, "action", {"steering": 0.0, "acceleration": 0.0})
        )
        sim_vehicle.crashed = bool(getattr(vehicle, "crashed", False))
        sim_vehicle.color = getattr(vehicle, "color", None)

        sim_vehicle.road = copy.copy(self.road)
        sim_vehicle.road.network = self.road.network
        sim_vehicle.road.objects = list(self.road.objects)
        sim_vehicle.road.vehicles = [
            sim_vehicle,
            *[v for v in self.road.vehicles if v is not self.vehicle],
        ]
        return sim_vehicle

    def _planning_vehicle_state(self, vehicle: EgoVehicle) -> dict[str, Any]:
        return {
            "position": np.array(vehicle.position, dtype=float),
            "heading": float(vehicle.heading),
            "speed": float(vehicle.speed),
            "target_speed": float(vehicle.target_speed),
            "speed_index": int(vehicle.speed_index),
            "lateral_offset": float(vehicle.lateral_offset),
            "target_lane_index": copy.deepcopy(vehicle.target_lane_index),
            "lane_index": copy.deepcopy(vehicle.lane_index),
            "action": dict(vehicle.action),
            "last_low_level_action": dict(vehicle._last_low_level_action),
            "crashed": bool(vehicle.crashed),
            "impact": (
                None
                if vehicle.impact is None
                else np.array(vehicle.impact, dtype=float)
            ),
        }

    def _restore_planning_vehicle_state(
        self, vehicle: EgoVehicle, state: dict[str, Any]
    ) -> None:
        vehicle.position = np.array(state["position"], dtype=float)
        vehicle.heading = float(state["heading"])
        vehicle.speed = float(state["speed"])
        vehicle.target_speed = float(state["target_speed"])
        vehicle.speed_index = int(state["speed_index"])
        vehicle.lateral_offset = float(state["lateral_offset"])
        vehicle.target_lane_index = copy.deepcopy(state["target_lane_index"])
        vehicle.lane_index = copy.deepcopy(state["lane_index"])
        vehicle.lane = (
            self.road.network.get_lane(vehicle.lane_index)
            if vehicle.lane_index is not None
            else None
        )
        vehicle.action = dict(state["action"])
        vehicle._last_low_level_action = dict(state["last_low_level_action"])
        vehicle.crashed = bool(state["crashed"])
        vehicle.impact = (
            None
            if state["impact"] is None
            else np.array(state["impact"], dtype=float)
        )

    def _planning_static_obstacles(self) -> tuple[np.ndarray, np.ndarray]:
        others = [other for other in self.road.vehicles if other is not self.vehicle]
        if not others:
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

        positions = np.asarray([other.position for other in others], dtype=float)
        diagonals = np.asarray(
            [
                float(getattr(other, "diagonal", np.hypot(other.LENGTH, other.WIDTH)))
                for other in others
            ],
            dtype=float,
        )
        return positions, diagonals

    def _planning_reference_index(self, offset: int) -> int:
        ref = getattr(self, "_expert_ref_xy_pol", None)
        if ref is None or len(ref) == 0:
            return 0
        return int(np.clip(self.steps + offset, 0, len(ref) - 1))

    def _reference_heading_at(self, ref_idx: int) -> float:
        ref = self._expert_ref_xy_pol
        if len(ref) < 2:
            return float(self.vehicle.heading)
        i0 = int(np.clip(ref_idx, 0, len(ref) - 2))
        i1 = i0 + 1
        dx = float(ref[i1, 0] - ref[i0, 0])
        dy = float(ref[i1, 1] - ref[i0, 1])
        return float(np.arctan2(dy, dx))

    def _planning_clearance_cost(
        self,
        sim_vehicle: EgoVehicle,
        other_positions: np.ndarray | None = None,
        other_diagonals: np.ndarray | None = None,
    ) -> float:
        weight = float(self.expert_cfg.get("planner_clearance_weight", 8.0))
        collision_cost = float(self.expert_cfg.get("planner_collision_cost", 1e6))

        if other_positions is None or other_diagonals is None:
            other_positions, other_diagonals = self._planning_static_obstacles()
        if other_positions.size == 0:
            return 0.0

        sim_diagonal = float(
            getattr(sim_vehicle, "diagonal", np.hypot(sim_vehicle.LENGTH, sim_vehicle.WIDTH))
        )
        center_dists = np.linalg.norm(other_positions - sim_vehicle.position, axis=1)
        clearances = center_dists - 0.5 * (sim_diagonal + other_diagonals)
        min_clearance = float(np.min(clearances))
        if min_clearance < 0.0:
            return collision_cost

        if not np.isfinite(min_clearance):
            return 0.0

        soft_margin = max(3.0, 0.20 * float(sim_vehicle.speed))
        if min_clearance >= soft_margin:
            return 0.0
        return weight * float((soft_margin - min_clearance) ** 2)

    def _planning_step_score(
        self,
        sim_vehicle: EgoVehicle,
        action_str: str,
        offset: int,
        previous_action: str | None,
        frames_per_action: int,
        dt: float,
        position_weight: float,
        heading_weight: float,
        speed_weight: float,
        action_change_weight: float,
        other_positions: np.ndarray,
        other_diagonals: np.ndarray,
    ) -> float:
        sim_vehicle.act(action_str)
        for _ in range(frames_per_action):
            sim_vehicle.act(None)
            sim_vehicle.step(dt)

        ref_idx = self._planning_reference_index(offset)
        ref_pos = self._expert_ref_xy_pol[ref_idx]
        ref_speed = float(self._expert_ref_v_pol[ref_idx])

        pos_err = float(np.linalg.norm(sim_vehicle.position - ref_pos))
        ref_heading = self._reference_heading_at(ref_idx)
        heading_err = float(
            np.arctan2(
                np.sin(sim_vehicle.heading - ref_heading),
                np.cos(sim_vehicle.heading - ref_heading),
            )
        )
        speed_err = float(sim_vehicle.speed - ref_speed)

        score = position_weight * pos_err * pos_err
        score += heading_weight * heading_err * heading_err
        score += speed_weight * speed_err * speed_err
        score += self._planning_clearance_cost(
            sim_vehicle,
            other_positions=other_positions,
            other_diagonals=other_diagonals,
        )

        if previous_action is not None and previous_action != action_str:
            score += action_change_weight

        return score

    def _rollout_action_tree(
        self,
        sim_vehicle: EgoVehicle,
        candidate_actions: list[str],
        horizon: int,
        frames_per_action: int,
        dt: float,
        position_weight: float,
        heading_weight: float,
        speed_weight: float,
        action_change_weight: float,
        other_positions: np.ndarray,
        other_diagonals: np.ndarray,
    ) -> tuple[str, float]:
        base_state = self._planning_vehicle_state(sim_vehicle)

        def search(
            depth: int,
            previous_action: str | None,
            running_score: float,
            best_score: float,
        ) -> float:
            if depth > horizon:
                return running_score
            if running_score >= best_score:
                return best_score

            for action_str in candidate_actions:
                state = self._planning_vehicle_state(sim_vehicle)
                step_score = self._planning_step_score(
                    sim_vehicle=sim_vehicle,
                    action_str=action_str,
                    offset=depth,
                    previous_action=previous_action,
                    frames_per_action=frames_per_action,
                    dt=dt,
                    position_weight=position_weight,
                    heading_weight=heading_weight,
                    speed_weight=speed_weight,
                    action_change_weight=action_change_weight,
                    other_positions=other_positions,
                    other_diagonals=other_diagonals,
                )
                candidate_score = search(
                    depth + 1,
                    action_str,
                    running_score + step_score,
                    best_score,
                )
                best_score = min(best_score, candidate_score)
                self._restore_planning_vehicle_state(sim_vehicle, state)
            return best_score

        best_first_action = candidate_actions[0]
        best_score = np.inf

        for action_str in candidate_actions:
            self._restore_planning_vehicle_state(sim_vehicle, base_state)
            step_score = self._planning_step_score(
                sim_vehicle=sim_vehicle,
                action_str=action_str,
                offset=1,
                previous_action=None,
                frames_per_action=frames_per_action,
                dt=dt,
                position_weight=position_weight,
                heading_weight=heading_weight,
                speed_weight=speed_weight,
                action_change_weight=action_change_weight,
                other_positions=other_positions,
                other_diagonals=other_diagonals,
            )
            total_score = search(
                depth=2,
                previous_action=action_str,
                running_score=step_score,
                best_score=best_score,
            )
            if total_score < best_score:
                best_score = total_score
                best_first_action = action_str

        self._restore_planning_vehicle_state(sim_vehicle, base_state)
        return best_first_action, float(best_score)

    def _select_discrete_expert_action(self) -> str:
        actions = self._expert_action_strings()
        if len(actions) == 1:
            return actions[0]

        horizon = int(max(1, self.expert_cfg.get("planner_horizon", 2)))
        branching = int(max(1, self.expert_cfg.get("planner_branching", len(actions))))
        prioritized = []

        if self._expert_actions_policy:
            last_action = self._expert_actions_policy[-1]
            if isinstance(last_action, str) and last_action in actions:
                prioritized.append(last_action)

        for candidate in ["IDLE", "STEER_LEFT", "STEER_RIGHT", "FASTER", "SLOWER"]:
            if candidate in actions and candidate not in prioritized:
                prioritized.append(candidate)
        for candidate in actions:
            if candidate not in prioritized:
                prioritized.append(candidate)

        candidate_actions = prioritized[:branching]
        frames_per_action = max(
            1,
            int(self.config["simulation_frequency"] // self.config["policy_frequency"]),
        )
        dt = 1.0 / float(self.config["simulation_frequency"])
        position_weight = float(self.expert_cfg.get("planner_position_weight", 3.0))
        heading_weight = float(self.expert_cfg.get("planner_heading_weight", 0.5))
        speed_weight = float(self.expert_cfg.get("planner_speed_weight", 0.2))
        action_change_weight = float(
            self.expert_cfg.get("planner_action_change_weight", 0.05)
        )
        other_positions, other_diagonals = self._planning_static_obstacles()
        sim_vehicle = self._planning_vehicle_clone()

        best_first_action, _ = self._rollout_action_tree(
            sim_vehicle=sim_vehicle,
            candidate_actions=candidate_actions,
            horizon=horizon,
            frames_per_action=frames_per_action,
            dt=dt,
            position_weight=position_weight,
            heading_weight=heading_weight,
            speed_weight=speed_weight,
            action_change_weight=action_change_weight,
            other_positions=other_positions,
            other_diagonals=other_diagonals,
        )
        return best_first_action

    def expert_replay_metrics(self):
        if not hasattr(self, "_expert_ref_xy_pol"):
            raise RuntimeError("No expert reference.")
        if not hasattr(self, "_replay_xy_pol"):
            raise RuntimeError("No replay recorded.")

        ref = np.asarray(self._expert_ref_xy_pol, dtype=float)
        rep = np.asarray(self._replay_xy_pol, dtype=float)

        T = min(len(ref), len(rep))
        ref = ref[:T]
        rep = rep[:T]

        err = np.linalg.norm(ref - rep, axis=1)
        ade = float(np.mean(err))
        fde = float(err[-1]) if T > 0 else np.nan

        return {"T": T, "ADE_m": ade, "FDE_m": fde, "err_per_step_m": err}

    def _resolve_expert_action(
        self,
    ) -> tuple[Action, np.ndarray | None, str | None, int | None]:
        pos = self.vehicle.position
        hdg = float(self.vehicle.heading)
        spd = float(self.vehicle.speed)

        steer_cmd, accel_cmd, i_near, i_tgt, _expert_target_lane_id = self._tracker.step(
            pos, hdg, spd
        )

        expert_action: np.ndarray | None = None
        expert_action_str: str | None = None
        expert_action_idx: int | None = None

        if self.control_mode == "continuous":
            accel_norm = float(np.clip(accel_cmd / MAX_ACCEL, -1.0, 1.0))
            steer_norm = float(np.clip(steer_cmd / MAX_STEER, -1.0, 1.0))
            expert_action = np.array([accel_norm, steer_norm], dtype=np.float32)
            action: Action = expert_action
        elif self.control_mode == "discrete":
            expert_action_str = self._select_discrete_expert_action()

            actions_indexes = getattr(self.action_type, "actions_indexes", None)
            if actions_indexes is None:
                raise RuntimeError(
                    "Action type mismatch. Config must use DiscreteSteerMetaAction."
                )

            if expert_action_str not in actions_indexes:
                logger.warning(
                    "Expert action %s invalid for current action type. Defaulting to IDLE.",
                    expert_action_str,
                )
                expert_action_str = "IDLE"

            expert_action_idx = int(actions_indexes[expert_action_str])
            action = expert_action_idx
        else:
            raise ValueError(f"Unknown action_mode={self.control_mode!r}")

        if (
            hasattr(self, "_expert_actions_policy")
            and self._expert_actions_policy is not None
        ):
            self._expert_actions_policy.append(
                expert_action.copy() if expert_action is not None else expert_action_str
            )
        if hasattr(self, "_tracker_dbg") and self._tracker_dbg is not None:
            self._tracker_dbg.append((int(i_near), int(i_tgt)))

        return action, expert_action, expert_action_str, expert_action_idx
