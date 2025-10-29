from __future__ import annotations
import os
import csv
import numpy as np
import bisect
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork

from highway_env.ngsim_utils.obs_vehicle import ReplayVehicle

Observation = np.ndarray


class NGSimEnv(AbstractEnv):
    """
    NGSIM Driving Environment with CSV trajectory replay.

    - Builds a US-101 (southbound) segment with an auxiliary lane section.
    - Spawns ONLY the ego vehicle (no synthetic IDM traffic).
    - Replays NGSIM trajectories as grey obstacle vehicles (no proximity filter).
    - Supports looping the full dataset or sweeping with a sliding time window.
    - Prints replay diagnostics to the console.
    """

    # ----------------------------
    # Config
    # ----------------------------
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # Observation / Action
                "observation": {"type": "Kinematics"},
                "action": {"type": "DiscreteMetaAction"},

                # Scene & rendering
                "scene": "us-101",
                "show_trajectories": False,
                "simulation_frequency": 15,   # Hz (dt = 1/sim_freq)
                "policy_frequency": 5,

                # Rewards
                "collision_reward": -1.0,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20.0, 30.0],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,

                # Replay (CSV)
                "replay_from_csv": False,
                "replay_folder": "./processed_ngsim",
                "replay_start_time_ms": None,  # None -> min in data
                "replay_end_time_ms": None,    # None -> max in data
                "replay_max_vehicles": 120,    # cap per (re)spawn
                "replay_interpolate": True,

                # 🔁 Replay cycling controls
                "replay_loop": True,           # loop when reaching end of dataset/window
                "replay_window_ms": None,      # None = play entire dataset span
                "replay_stride_ms": None,      # slide amount between windows; default = window_ms
                "replay_cycle_pause_s": 0.0,   # optional small pause between windows (visual)
            }
        )
        return config

    @property
    def dt(self) -> float:
        return 1.0 / float(self.config.get("simulation_frequency", 15))

    # ----------------------------
    # Lifecycle
    # ----------------------------
    def _reset(self) -> None:
        if getattr(self, "_replay", None) and self._replay.get("active"):
            self._clear_replay()
        self._create_road()
        self._create_vehicle()
        self._load_ngsim_replay()

        if self.config.get("replay_from_csv", False) and self._replay:
            t0_dataset = int(self._replay["t0"])
            t1_dataset = int(self._replay["t1"])

            # Choose initial window
            win = self.config.get("replay_window_ms")
            if win is None:
                start, end = t0_dataset, t1_dataset
            else:
                start = t0_dataset
                end = min(t0_dataset + int(win), t1_dataset)

            self._replay["window_start"] = int(start)
            self._replay["window_end"] = int(end)

            print(f"✅ [ReplayMode] Loaded {len(self._replay['veh2traj'])} trajectories "
                  f"from {self.config['replay_folder']}")
            print(f"▶️  [ReplayMode] Playing window [{self._replay['window_start']}, {self._replay['window_end']}] ms")

            self.road.sim_time_ms = int(start)
            self._spawn_replay_for_time(int(start))

    # ----------------------------
    # Road & Vehicles
    # ----------------------------
    def _create_road(self) -> None:
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

  
        length = 2150 / 3.281 # m
        width = 12 / 3.281 # m
        ends = [0, 560/3.281, (698+578+150)/3.281, length]

        # first section
        line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
        for lane in range(5):
            origin = [ends[0], lane * width]
            end = [ends[1], lane * width]
            net.add_lane('s1', 's2', StraightLane(origin, end, width=width, line_types=line_types[lane]))

        # merge_in lanes
        net.add_lane('merge_in', 's2', StraightLane([480/3.281, 5.5*width], [ends[1], 5*width], width=width, line_types=[c, c], forbidden=True))

        # second section
        line_types = [[c, n], [s, n], [s, n], [s, n], [s, n], [s, c]]
        for lane in range(6):
            origin = [ends[1], lane * width]
            end = [ends[2], lane * width]
            net.add_lane('s2', 's3', StraightLane(origin, end, width=width, line_types=line_types[lane]))
        
        # third section
        line_types = [[c, n], [s, n], [s, n], [s, n], [s, c]]
        for lane in range(5):
            origin = [ends[2], lane * width]
            end = [ends[3], lane * width]
            net.add_lane('s3', 's4', StraightLane(origin, end, width=width, line_types=line_types[lane]))

        # merge_out lanes
        net.add_lane('s3', 'merge_out', StraightLane([ends[2], 5*width], [1550/3.281, 7*width], width=width, line_types=[c, c], forbidden=True))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        '''
        net = RoadNetwork()
        C, S, N = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        # Geometry (feet -> meters)
        ft = 0.3048
        length = 2150 * ft
        lane_width = 12 * ft

        # Longitudinal split points (x)
        x0 = 0.0
        x1 = 560 * ft
        x2 = (698 + 578 + 150) * ft
        x3 = length

        # Section 1: 5-lane mainline (s1 -> s2)
        line_types_5 = [(C, N), (S, N), (S, N), (S, N), (S, C)]
        for i, lt in enumerate(line_types_5):
            start = [x0, i * lane_width]
            end = [x1, i * lane_width]
            net.add_lane("s1", "s2", StraightLane(start, end, width=lane_width, line_types=lt, speed_limit=33.3))

        # On-ramp connector (forbidden to ego)
        net.add_lane(
            "onramp", "s2",
            StraightLane([480 * ft, 5.5 * lane_width], [x1, 5 * lane_width],
                         width=lane_width, line_types=(C, C), forbidden=True, priority=-1, speed_limit=25.0),
        )

        # Section 2: 6-lane (aux exists) (s2 -> s3)
        line_types_6 = [(C, N), (S, N), (S, N), (S, N), (S, N), (S, C)]
        for i, lt in enumerate(line_types_6):
            start = [x1, i * lane_width]
            end = [x2, i * lane_width]
            net.add_lane("s2", "s3", StraightLane(start, end, width=lane_width, line_types=lt, speed_limit=33.3))

        # Section 3: 5-lane mainline resumes (s3 -> s4)
        for i, lt in enumerate(line_types_5):
            start = [x2, i * lane_width]
            end = [x3, i * lane_width]
            net.add_lane("s3", "s4", StraightLane(start, end, width=lane_width, line_types=lt, speed_limit=33.3))

        # Off-ramp connector (forbidden to ego)
        net.add_lane(
            "s3", "offramp",
            StraightLane([x2, 5 * lane_width], [1550 * ft, 7 * lane_width],
                         width=lane_width, line_types=(C, C), forbidden=True, priority=-1, speed_limit=25.0),
        )

        self.road = Road(network=net, np_random=self.np_random,
                         record_history=bool(self.config.get("show_trajectories", False)))
        '''
    def _create_vehicle(self) -> None:
        """Spawn ONLY the ego vehicle (no synthetic IDM traffic)."""
        road, net = self.road, self.road.network

        def n_lanes(edge):  # helper
            return len(net.graph[edge[0]][edge[1]])

        # Ego on ("s1","s2")
        main_edge = ("s1", "s2")
        num_main = n_lanes(main_edge)
        ego_lane_id = min(1, num_main - 1)
        ego_xy = net.get_lane((*main_edge, ego_lane_id)).position(30.0, 0.0)
        ego_speed = 30.0  # m/s
        ego_vehicle = self.action_type.vehicle_class(road, ego_xy, speed=ego_speed)
        ego_vehicle.is_ego = True
        ego_vehicle.color = (30, 144, 255)  # blue
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    # ----------------------------
    # Replay: load + spawn + cycle
    # ----------------------------
    def _load_ngsim_replay(self) -> None:
        """Load CSV dumps into memory for trajectory replay."""
        if not self.config.get("replay_from_csv", False):
            self._replay = None
            return

        folder = self.config["replay_folder"]
        vr_path = os.path.join(folder, "vehicle_record_file.csv")
        v_path = os.path.join(folder, "vehicle_file.csv")
        # snapshot_file.csv is optional and unused here

        # 1) Records (ID -> rec)
        id2rec = {}
        t_min, t_max = None, None
        with open(vr_path, "r", newline="") as f:
            r = csv.reader(f)
            for row in r:
                # ID, veh_ID, unixtime, x, y, lat, lon, len, wid, cls, spd, acc, lane_ID, pred, foll, shead, thead
                ID = int(row[0]); veh_ID = int(row[1]); t = int(row[2])
                rec = {
                    "ID": ID, "veh_ID": veh_ID, "t": t,
                    "x": float(row[3]), "y": float(row[4]),              # raw Local_X/Local_Y (feet)
                    "len": float(row[7]), "wid": float(row[8]),
                    "cls": int(row[9]), "spd": float(row[10]),
                    "lane": int(row[12]),
                }
                id2rec[ID] = rec
                t_min = t if t_min is None else min(t_min, t)
                t_max = t if t_max is None else max(t_max, t)

        # 2) Vehicle membership (veh_ID -> sorted trajectory)
        veh2traj = {}
        with open(v_path, "r", newline="") as f:
            r = csv.reader(f)
            for row in r:
                veh_id = int(row[0])
                ids = [int(x) for x in row[1:] if x != ""]
                traj = [id2rec[i] for i in ids if i in id2rec]
                traj.sort(key=lambda d: d["t"])
                veh2traj[veh_id] = [
                    {"t": d["t"], "x": d["x"], "y": d["y"], "spd": d["spd"], "len": d["len"], "wid": d["wid"]}
                    for d in traj
                ]

        t0 = self.config.get("replay_start_time_ms") or t_min
        t1 = self.config.get("replay_end_time_ms") or t_max

        self._replay = {
            "veh2traj": veh2traj,
            "t0": int(t0),
            "t1": int(t1),
            "active": [],        # list[ReplayVehicle]
            # window_start, window_end set in _reset()
        }

    # ---- helpers for cycling ----
    def _clear_replay(self):
        """Remove all active replay vehicles from the road and clear the list."""
        if not getattr(self, "_replay", None):
            return
        act = self._replay.get("active", [])
        if act:
            self.road.vehicles = [v for v in self.road.vehicles if v not in act]
        self._replay["active"] = []

    def _spawn_replay_for_time(self, t_ms: int):
        """(Re)spawn replay vehicles whose trajectories cover t_ms (no proximity filter)."""
        if not (self.config.get("replay_from_csv", False) and self._replay):
            return

        # Clear previously active replay vehicles
        self._clear_replay()

        road = self.road
        veh2traj = self._replay["veh2traj"]
        cap = int(self.config.get("replay_max_vehicles", 120))

        # --- Units / geometry (must match how you built the road) ---
        FT_TO_M = 0.3048
        LANE_WIDTH_FT = 12.0
        LANE_WIDTH_M = LANE_WIDTH_FT * FT_TO_M
        # NGSIM Local_X is measured from the *left edge*; our lane-0 center is at 0 → shift by half a lane
        LATERAL_OFFSET_FT = LANE_WIDTH_FT / 2.0

        # Longitudinal extent of your drawn segment (2150 ft total)
        ROAD_LEN_M = 2150.0 * FT_TO_M
        # Max lanes you display in any section: 6 (5 main + 1 aux); allow ± half a lane margin
        NUM_LANES_MAX = 6
        LAT_MIN_M = -0.5 * LANE_WIDTH_M
        LAT_MAX_M = (NUM_LANES_MAX - 0.5) * LANE_WIDTH_M

        def interp_pose(traj, t):
            """traj: list of dicts with keys t, x, y, spd, len, wid (times in ms, positions in feet)"""
            # Fast path: pre-extract times (do this once per traj and cache alongside traj)
            times = [p["t"] for p in traj]  # strictly non-decreasing, but may contain duplicates
            n = len(times)
            if n == 0:
                return None

            # Clamp to [t0, tn]
            if t <= times[0]:
                return traj[0]
            if t >= times[-1]:
                return traj[-1]

            # Find right neighbor index r: times[l] < t <= times[r]
            r = bisect.bisect_left(times, t)
            l = r - 1

            a = traj[l]
            b = traj[r]
            ta, tb = a["t"], b["t"]

            # If t matches exactly (or duplicates), pick the exact sample
            if tb == t:
                return b
            if ta == t:
                return a

            # Guard against zero dt (duplicate timestamps)
            dt = float(tb - ta)
            if dt <= 0.0:
                # fall back to a (could also choose b)
                return a

            u = float(t - ta) / dt
            return {
                "t": t,
                "x": a["x"] + u * (b["x"] - a["x"]),   # feet (lateral from left edge)
                "y": a["y"] + u * (b["y"] - a["y"]),   # feet (longitudinal)
                "spd": a.get("spd", 0.0) + u * (b.get("spd", 0.0) - a.get("spd", 0.0)),
                "len": a.get("len", 15.0),
                "wid": a.get("wid", 6.0),
            }

        def in_roi(p_ft):
            """Check if pose lies within our drawn segment/lane fan (in meters after conversion+offset)."""
            long_m = p_ft["y"] * FT_TO_M
            lat_m = (p_ft["x"] - LATERAL_OFFSET_FT) * FT_TO_M
            return (0.0 <= long_m <= ROAD_LEN_M) and (LAT_MIN_M <= lat_m <= LAT_MAX_M)

        # --- Select candidates present at t_ms and inside ROI ---
        # Sort by veh_id for deterministic order, then truncate to cap
        candidates = []
        for veh_id, traj in veh2traj.items():
            if not traj or not (traj[0]["t"] <= t_ms <= traj[-1]["t"]):
                continue
            p = interp_pose(traj, t_ms)
            if in_roi(p):
                candidates.append((veh_id, traj))
        candidates.sort(key=lambda kv: kv[0])  # deterministic
        candidates = candidates[:cap]

        # --- Spawn replay vehicles ---
        spawned = 0
        for _, traj in candidates:
            rv = ReplayVehicle(
                road, traj, start_t_ms=t_ms,
                interpolate=self.config.get("replay_interpolate", True),
            )
            # Grey tint for replay cars (width/length handled inside ReplayVehicle)
            rv.color = (120, 120, 120)
            road.vehicles.append(rv)
            self._replay["active"].append(rv)
            spawned += 1

        print(f"🔁 [ReplayMode] Spawned {spawned} replay vehicles at t={t_ms} ms (cap={cap})")
        # Optional quick sanity: show first few positions in meters
        for k, rv in enumerate(self._replay["active"][:3]):
            print(f"   ↳ #{k}: long={rv.position[0]:.2f} m, lat={rv.position[1]:.2f} m, L={rv.LENGTH:.2f} m, W={rv.WIDTH:.2f} m")

    # ----------------------------
    # Rewards / Termination
    # ----------------------------
    def _rewards(self, action: int) -> dict[str, float]:
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0.0, 1.0])

        lane_tuple = self.vehicle.lane_index  # (start, end, lane_id)
        try:
            n_on_edge = len(self.road.network.graph[lane_tuple[0]][lane_tuple[1]])
            right_lane_reward = (lane_tuple[2] / max(1, n_on_edge - 1))
        except Exception:
            right_lane_reward = 0.0

        # Example altruistic penalty (kept for compatibility; no ramp IDM now)
        def is_on_ramp_edge(v):
            s, e, _ = v.lane_index
            return (s, e) == ("onramp", "s2")

        ramp_penalty = 0.0
        for v in self.road.vehicles:
            if is_on_ramp_edge(v) and hasattr(v, "target_speed") and v.target_speed > 0:
                ramp_penalty += (v.target_speed - v.speed) / v.target_speed
        ramp_penalty = float(np.clip(ramp_penalty, 0.0, 1.0))

        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": right_lane_reward,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": int(action in [0, 2]),
            "merging_speed_reward": ramp_penalty,
        }

    def _reward(self, action: int) -> float:
        raw = sum(self.config.get(name, 0.0) * val for name, val in self._rewards(action).items())
        worst = self.config.get("collision_reward", -1.0) + self.config.get("merging_speed_reward", -0.5)
        best = self.config.get("high_speed_reward", 1.0) + self.config.get("right_lane_reward", 0.3)
        return utils.lmap(raw, [worst, best], [0.0, 1.0])

    def _is_terminated(self) -> bool:
        if self.vehicle.crashed:
            return True
        s, e, _ = self.vehicle.lane_index
        if (s, e) == ("s3", "s4"):
            return self.vehicle.position[0] > 0.95 * 655.0
        return False

    def _is_truncated(self) -> bool:
        return False
    def _reap_finished_replay(self):
        """Remove replay vehicles whose trajectories have ended."""
        if not (self.config.get("replay_from_csv", False) and getattr(self, "_replay", None)):
            return
        cur = int(self.road.sim_time_ms)
        keep = []
        removed = 0
        for rv in self._replay.get("active", []):
            last_t = int(rv.traj[-1]["t"])
            if cur > last_t:
                # yank from the road immediately
                if rv in self.road.vehicles:
                    self.road.vehicles.remove(rv)
                removed += 1
            else:
                keep.append(rv)
        self._replay["active"] = keep
        if removed:
            print(f"🧹 [ReplayMode] Despawned {removed} finished replay vehicles at t={cur} ms")
    # ----------------------------
    # Step (advance + cycle)
    # ----------------------------
    def step(self, action: Action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.config.get("replay_from_csv", False) and self._replay:
            # ---- advance replay clock (ms), clamp to dataset end ----
            self.road.sim_time_ms = min(
                int(self._replay["t1"]),
                int(self.road.sim_time_ms) + int(round(1000 * self.dt))
            )
            cur_t = int(self.road.sim_time_ms)
            win_end = int(self._replay["window_end"])

            # ---- reap finished replay vehicles immediately ----
            keep, removed = [], 0
            for rv in self._replay.get("active", []):
                # finished either because the vehicle said so, or we've passed its last sample time
                last_t = int(rv.traj[-1]["t"])
                if getattr(rv, "finished", False) or cur_t > last_t:
                    if rv in self.road.vehicles:
                        self.road.vehicles.remove(rv)
                    removed += 1
                else:
                    keep.append(rv)
            self._replay["active"] = keep
            if removed:
                print(f"🧹 [ReplayMode] Despawned {removed} replay vehicles at t={cur_t} ms")

            # ---- periodic heartbeat (debug) ----
            if (cur_t % 5000) < int(1000 * self.dt):
                print(f"🕒 [ReplayMode] sim_time_ms={cur_t} | active_replay={len(self._replay['active'])}")

            # ---- window boundary → cycle / loop ----
            if cur_t > win_end:
                t0_dataset = int(self._replay["t0"])
                t1_dataset = int(self._replay["t1"])
                win = self.config.get("replay_window_ms")
                stride = self.config.get("replay_stride_ms")

                if win is None:
                    # full-dataset mode
                    if self.config.get("replay_loop", True):
                        next_start = t0_dataset
                        next_end = t1_dataset
                    else:
                        # stop advancing (no respawn)
                        self.road.sim_time_ms = win_end
                        return obs, reward, terminated, truncated, info
                else:
                    # sliding window mode
                    if stride is None:
                        stride = win
                    next_start = int(self._replay["window_start"]) + int(stride)
                    if next_start > t1_dataset:
                        next_start = t0_dataset
                    next_end = min(next_start + int(win), t1_dataset)

                # optional short pause for visual separation
                pause_s = float(self.config.get("replay_cycle_pause_s", 0.0))
                if pause_s > 0.0 and getattr(self, "viewer", None) is not None:
                    import time as _time
                    _time.sleep(pause_s)

                # switch window & (re)spawn vehicles present at the new start time
                self._replay["window_start"] = int(next_start)
                self._replay["window_end"] = int(next_end)
                self.road.sim_time_ms = int(next_start)
                self._spawn_replay_for_time(int(next_start))
                print(f"🔁 [ReplayMode] Next window [{self._replay['window_start']}, {self._replay['window_end']}] ms")

        return obs, reward, terminated, truncated, info
