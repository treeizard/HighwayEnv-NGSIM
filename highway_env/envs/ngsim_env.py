from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List, Any

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.ngsim_utils.obs_vehicle import ReplayVehicle


from highway_env.ngsim_utils.trajectory_gen import build_trajectory, process_raw_trajectory

Observation = np.ndarray

class NGSimEnv(AbstractEnv):
    """
    NGSIM Driving Environment with defensive replay spawn + overlap diagnostics.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},

            "scene": "us-101",
            "show_trajectories": False,
            "simulation_frequency": 15,
            "policy_frequency": 5,

            "ego_speed": 30.0,
            "ego_lane_index": 1,
            "ego_longitudinal_m": 30.0,

            # Replay
            "replay_period": 0,

            # Spawn control
            "spawn_radius_m": 150.0,
            "max_surrounding": 80,
            "min_initial_gap_m": 2.0,  # prevent initial overlaps

            # Debug
            "log_overlaps": True,
        })
        return config

    @property
    def dt(self) -> float:
        return 1.0 / float(self.config.get("simulation_frequency", 15))

    # ---- lifecycle ----
    def _reset(self) -> None:
        self._load_trajectory()
        self._create_road()
        self._create_vehicles()

    # ---- road ----
    def _create_road(self) -> None:
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        length = 2150 / 3.281  # m
        width = 12 / 3.281     # m
        ends = [0, 560/3.281, (698+578+150)/3.281, length]

        # first section (5 lanes)
        line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
        for lane in range(5):
            origin = [ends[0], lane * width]
            end = [ends[1], lane * width]
            net.add_lane("s1", "s2", StraightLane(origin, end, width=width, line_types=line_types[lane]))

        # merge_in (forbidden)
        net.add_lane("merge_in", "s2", StraightLane([480/3.281, 5.5*width], [ends[1], 5*width], width=width, line_types=[c, c], forbidden=True))

        # second section (6 lanes)
        line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
        for lane in range(6):
            origin = [ends[1], lane * width]
            end = [ends[2], lane * width]
            net.add_lane("s2", "s3", StraightLane(origin, end, width=width, line_types=line_types[lane]))

        # third section (5 lanes)
        line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
        for lane in range(5):
            origin = [ends[2], lane * width]
            end = [ends[3], lane * width]
            net.add_lane("s3", "s4", StraightLane(origin, end, width=width, line_types=line_types[lane]))

        # merge_out (forbidden)
        net.add_lane("s3", "merge_out", StraightLane([ends[2], 5*width], [1550/3.281, 7*width], width=width, line_types=[c, c], forbidden=True))

        self.net = net
        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    # ---- vehicles ----
    def _create_vehicles(self) -> None:
        # ---------------- Ego ----------------
        main_edge = ("s1", "s2")
        num_main = len(self.net.graph[main_edge[0]][main_edge[1]])
        ego_lane_id = int(np.clip(self.config.get("ego_lane_index", 1), 0, max(0, num_main - 1)))
        ego_s = float(self.config.get("ego_longitudinal_m", 30.0))
        ego_speed = float(self.config.get("ego_speed", 30.0))
        ego_lane = self.net.get_lane((*main_edge, ego_lane_id))
        ego_xy = np.asarray(ego_lane.position(ego_s, 0.0), dtype=float)

        ego = self.action_type.vehicle_class(self.road, ego_xy, speed=ego_speed)
        ego.is_ego = True
        if hasattr(ego, "color"):
            ego.color = (30, 144, 255)
        self.road.vehicles.append(ego)
        self.vehicle = ego

        # ---------------- Surrounding (replay) ----------------
        ego_s_window = (ego_s - self.config["spawn_radius_m"], ego_s + self.config["spawn_radius_m"])
        min_gap = float(self.config.get("min_initial_gap_m", 2.0))

        count_added = 0
        for veh_id, meta in self.trajectory_set.items():
            if count_added >= int(self.config.get("max_surrounding", 80)):
                break

            traj_m = process_raw_trajectory(meta["trajectory"])  # rows: [s_abs_m, r_abs_m, v_mps, lane_id]
            if traj_m.shape[0] < 2:
                continue

            # first point inside window (prefer an explicit find over argmax)
            in_win = np.where((traj_m[:, 0] >= ego_s_window[0]) & (traj_m[:, 0] <= ego_s_window[1]))[0]
            if in_win.size == 0:
                continue
            idx0 = int(in_win[0])

            # Unpack first usable sample
            cur_s, cur_r_abs, v_cur, lane_id = traj_m[idx0]

            # Choose edge + local s from s_abs
            cut1 = 560/3.281
            cut2 = (698+578+150)/3.281
            if cur_s <= cut1:
                edge, edge_start = ("s1", "s2"), 0.0
            elif cur_s <= cut2:
                edge, edge_start = ("s2", "s3"), cut1
            else:
                edge, edge_start = ("s3", "s4"), cut2

            try:
                n_lanes_on_edge = len(self.net.graph[edge[0]][edge[1]])
            except Exception:
                n_lanes_on_edge = 1

            lane0 = int(np.clip(int(lane_id) - 1, 0, max(0, n_lanes_on_edge - 1)))
            lane = self.net.get_lane((edge[0], edge[1], lane0))

            local_s = float(cur_s - edge_start)

            # ---- KEY FIX: absolute -> relative lateral for spawn pose ----
            center_x, center_y = lane.position(local_s, 0.0)
            cur_r_rel = float(cur_r_abs) - float(center_y)

            # World pose at spawn (NO magic '-6' offset)
            xy0 = np.asarray(lane.position(local_s, cur_r_rel), dtype=float)

            # Initial speed (finite-diff on s if possible; fallback to v)
            if idx0 + 1 < traj_m.shape[0]:
                v0 = float(traj_m[idx0 + 1, 0] - cur_s) / utils.not_zero(self.dt)
            else:
                v0 = float(v_cur)

            # Build temp vehicle and check for spawn-time overlap
            temp = ReplayVehicle.create(
                self.road, veh_id, xy0,
                meta["length"]/3.281, meta["width"]/3.281,
                traj_m[idx0:], speed=v0
            )

            overlaps = False
            for other in self.road.vehicles:
                if hasattr(other, "aabb"):
                    if temp.overlaps_aabb(other):
                        overlaps = True
                        break
                else:
                    # Distance fallback if AABB unavailable
                    if np.linalg.norm(np.asarray(other.position) - xy0) < (0.5 * (temp.LENGTH + getattr(other, "LENGTH", 4.5)) + min_gap):
                        overlaps = True
                        break

            if overlaps:
                continue  # skip this vehicle; optional: jitter and retry instead

            self.road.vehicles.append(temp)
            count_added += 1

        # Color passive cars grey
        for v in self.road.vehicles:
            if getattr(v, "is_ego", False):
                continue
            if hasattr(v, "color"):
                v.color = (150, 150, 150)

    # ---- trajectories ----
    def _load_trajectory(self) -> None:
        self.trajectory_set = build_trajectory(self.config["scene"], self.config["replay_period"])

    # ---- rewards/termination (kept minimal) ----
    def _rewards(self, action: int) -> dict[str, float]:
        speed = float(getattr(self.vehicle, "speed", 0.0))
        scaled_speed = utils.lmap(speed, self.config.get("reward_speed_range",[20.0,30.0]), [0.0, 1.0])

        lane_tuple = getattr(self.vehicle, "lane_index", None)
        right_lane_reward = 0.0
        if lane_tuple is not None:
            try:
                n_on_edge = len(self.road.network.graph[lane_tuple[0]][lane_tuple[1]])
                right_lane_reward = (lane_tuple[2] / max(1, n_on_edge - 1)) if n_on_edge > 1 else 1.0
            except Exception:
                pass

        lane_change_pen = int(action in [0, 2])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": float(right_lane_reward),
            "high_speed_reward": float(scaled_speed),
            "lane_change_reward": float(lane_change_pen),
            "merging_speed_reward": 0.0,
        }

    def _reward(self, action: int) -> float:
        terms = self._rewards(action)
        raw = sum(self.config.get(name, 0.0) * val for name, val in terms.items())
        worst = self.config.get("collision_reward", -1.0) + self.config.get("merging_speed_reward", -0.5)
        best = self.config.get("high_speed_reward", 1.0) + self.config.get("right_lane_reward", 0.3)
        return utils.lmap(raw, [worst, best], [0.0, 1.0])

    def _is_terminated(self) -> bool:
        if self.vehicle.crashed: return True
        lane_tuple = getattr(self.vehicle, "lane_index", None)
        if lane_tuple is None: return False
        s, e, _ = lane_tuple
        if (s, e) == ("s3","s4"):
            return self.vehicle.position[0] > 0.95 * 655.0
        return False

    def _is_truncated(self) -> bool:
        return False

    def step(self, action: Action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Post-step overlap diagnostics
        if self.config.get("log_overlaps", True):
            overlaps: List[Tuple[int,int]] = []
            vehs = [v for v in self.road.vehicles if hasattr(v, "aabb")]
            for i in range(len(vehs)):
                for j in range(i+1, len(vehs)):
                    if vehs[i].overlaps_aabb(vehs[j]):
                        overlaps.append((getattr(vehs[i],"vehicle_ID",-1), getattr(vehs[j],"vehicle_ID",-1)))
            if overlaps:
                info = dict(info) if info else {}
                info["overlaps"] = overlaps  # visible in gym logs
        return obs, reward, terminated, truncated, info