# Modified by: Yide Tao (yide.tao@monash.edu)
# Reference: @article{huang2021driving,
#   title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
#   author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
#   journal={IEEE Transactions on Intelligent Transportation Systems},
#   year={2021},
#   publisher={IEEE}
# }
# @misc{highway-env,
#   author = {Leurent, Edouard},
#   title = {An Environment for Autonomous Driving Decision-Making},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/eleurent/highway-env}},
# }

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List, Any

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.ngsim_utils.obs_vehicle import NGSIMVehicle


from highway_env.ngsim_utils.trajectory_gen import build_trajectory, process_raw_trajectory, first_valid_index

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

            "ego_vehicle_ID": 121,

            # Replay
            "replay_period": 0,

            # Spawn control
            "spawn_radius_m": 150.0,
            "max_surrounding": 80,
            "min_initial_gap_m": 2.0,  

            "max_episode_steps": 600,

            "seed": None,

            # Debug
            "log_overlaps": True,
        })
        return config


    @property
    def dt(self) -> float:
        return 1.0 / float(self.config.get("simulation_frequency", 15))

    # ---- lifecycle ----
    def _reset(self) -> None:
        # NEW: episode step counter for truncation logic
        self.steps = 0

        # NEW: per-env seeding if provided
        seed = self.config.get("seed", None)
        if seed is not None and hasattr(self, "seed"):
            self.seed(seed)

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

        ego_rec = self.trajectory_set['ego']
        ego_traj = process_raw_trajectory(ego_rec['trajectory'])    
        ego_len  = ego_rec['length']
        ego_wid  = ego_rec['width']

        ego_start_idx = first_valid_index(ego_traj)
        if ego_start_idx is None:
            ego_lane_id = int(np.clip(self.config.get("ego_lane_index", 1),
                                0, max(0, num_main - 1)))
            ego_s = float(self.config.get("ego_longitudinal_m", 30.0))
            ego_speed = float(self.config.get("ego_speed", 30.0))
            ego_lane = self.net.get_lane((*main_edge, ego_lane_id))
            ego_xy = np.asarray(ego_lane.position(ego_s, 0.0), dtype=float)
        else:
            x0, y0, ego_speed, lane0 = ego_traj[ego_start_idx]
            ego_xy = np.array([x0, y0])

            # --- convert global NGSIM (x, y) to local lane-relative coordinates (s, r)
            ego_lane_id = int(np.clip(lane0, 0, max(0, num_main - 1)))
            ego_lane = self.net.get_lane((*main_edge, ego_lane_id))
            ego_s, ego_r = ego_lane.local_coordinates(ego_xy)

            # recompute world xy for correct lane alignment (optional)
            ego_xy = np.asarray(ego_lane.position(ego_s, ego_r), dtype=float)

        # --- instantiate the ego vehicle using relative coordinates
        ego = self.action_type.vehicle_class(
            road=self.road,
            position=ego_xy,
            speed=ego_speed,
            heading=ego_lane.heading_at(ego_s)
        )
        self.road.vehicles.append(ego)
        self.vehicle = ego
        
        # ---------------- Surrounding Vehicles---------------
        for veh_id, meta in list(self.trajectory_set.items())[1:]:
            other_trajectory = process_raw_trajectory(meta['trajectory'])[1:]
            v = NGSIMVehicle.create(
                    self.road,
                    veh_id,
                    other_trajectory[0][:2],
                    meta['length'] / 3.281,
                    meta['width'] / 3.281,
                    other_trajectory,
                    speed=other_trajectory[0][2],
                    color=(200, 0, 150),  # grey NGSIM cars
                )
           
            self.road.vehicles.append(
                v
            )
    # ---- trajectories ----
    def _load_trajectory(self) -> None:
        # NEW: configurable ego vehicle + replay period
        ego_id = int(self.config.get("ego_vehicle_ID", 121))
        period = int(self.config.get("replay_period", 0))
        self.trajectory_set = build_trajectory(
            self.config["scene"],
            period,
            ego_id,
        )

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
        """
        Time-limit truncation for RL. Ray/RLlib expects finite-length episodes.
        """
        max_steps = self.config.get("max_episode_steps", None)
        if max_steps is None:
            return False
        return getattr(self, "steps", 0) >= max_steps

    def step(self, action: Action):
        # Call AbstractEnv.step (handles dynamics, rewards, termination, truncation)
        obs, reward, terminated, truncated, info = super().step(action)

        # NEW: increment step counter
        self.steps = getattr(self, "steps", 0) + 1

        # NEW: enforce time-limit truncation on top of whatever AbstractEnv did
        if not truncated and self._is_truncated():
            truncated = True

        # Post-step overlap diagnostics
        if self.config.get("log_overlaps", True):
            overlaps: List[Tuple[int,int]] = []
            vehs = [v for v in self.road.vehicles if hasattr(v, "aabb")]
            for i in range(len(vehs)):
                for j in range(i+1, len(vehs)):
                    if vehs[i].overlaps_aabb(vehs[j]):
                        overlaps.append(
                            (getattr(vehs[i],"vehicle_ID",-1),
                             getattr(vehs[j],"vehicle_ID",-1))
                        )
            if overlaps:
                info = dict(info) if info else {}
                info["overlaps"] = overlaps  # visible in gym logs

        return obs, reward, terminated, truncated, info

