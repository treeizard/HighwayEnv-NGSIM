# highway_env/ngsim_utils/obs_vehicle.py
from __future__ import annotations
import numpy as np
from highway_env.vehicle.kinematics import Vehicle

class ReplayVehicle(Vehicle):
    """
    NGSIM trajectory follower with correct physical dimensions.

    Expected traj items (per point):
        {
          "t": <ms>,
          "x": <Local_X feet  (lateral)>,
          "y": <Local_Y feet  (longitudinal)>,
          "spd": <speed>,      # ft/s or m/s (see SPEED_IS_FT_PER_S)
          "len": <feet>,       # bumper-to-bumper
          "wid": <feet>,       # overall width
        }
    We convert feet -> meters, and map to road frame:
        position = [longitudinal_m, lateral_m] = [y_ft*0.3048, x_ft*0.3048]
    """
    FT_TO_M = 0.3048
    LATERAL_OFFSET_FT = 6.0

    # flip this to False if your `spd` field is already m/s in the processed CSV
    SPEED_IS_FT_PER_S = True

    # if True, re-apply LENGTH/WIDTH each step from the pose; otherwise freeze from first sample
    UPDATE_SIZE_EACH_STEP = True

    def __init__(self, road, trajectory, start_t_ms, interpolate=True):
        self.traj = trajectory
        self.interpolate = interpolate
        self.idx = 0
        self.prev_pos = None

        # Seed from t0
        p0 = self._pose_at(start_t_ms)
        pos0 = self._to_world_xy(p0)

        # Init base Vehicle
        spd0 = float(p0.get("spd", 0.0))
        if self.SPEED_IS_FT_PER_S:
            spd0 *= self.FT_TO_M
        super().__init__(road, pos0, speed=spd0)

        # Apply true size (feet → meters)
        self.LENGTH = float(p0.get("len", self.LENGTH)) * self.FT_TO_M
        self.WIDTH  = float(p0.get("wid", self.WIDTH)) * self.FT_TO_M

        # Optional: small safety inflation/deflation (e.g., +0.2 m length, +0.05 m width)
        # self.LENGTH += 0.0
        # self.WIDTH  += 0.0

        self.prev_pos = np.array(self.position, dtype=float)

    # ---------- sampling ----------
    def _pose_at(self, t_ms):
        traj = self.traj
        n = len(traj)
        while self.idx + 1 < n and traj[self.idx + 1]["t"] <= t_ms:
            self.idx += 1
        a = traj[self.idx]
        if (not self.interpolate) or self.idx + 1 >= n:
            return a
        b = traj[self.idx + 1]
        if b["t"] == a["t"]:
            return a
        u = np.clip((t_ms - a["t"]) / (b["t"] - a["t"]), 0.0, 1.0)
        return {
            "t": t_ms,
            "x": a["x"] + u * (b["x"] - a["x"]),
            "y": a["y"] + u * (b["y"] - a["y"]),
            "spd": a.get("spd", 0.0) + u * (b.get("spd", 0.0) - a.get("spd", 0.0)),
            "len": a.get("len", a.get("len", 15.0)),  # feet; fallback ≈ typical sedan
            "wid": a.get("wid", a.get("wid", 6.0)),   # feet
        }

    def _to_world_xy(self, p):
        # NGSIM: y = along-road [ft], x = from left edge [ft]
        long_m = float(p["y"]) * self.FT_TO_M
        lat_m  = (float(p["x"]) - self.LATERAL_OFFSET_FT) * self.FT_TO_M
        return [long_m, lat_m]

    # ---------- kinematics ----------
    def step(self, dt):
        # Time is driven by Road.sim_time_ms (advanced by env)
        t_ms = getattr(self.road, "sim_time_ms", 0)
        p = self._pose_at(t_ms)

        # Update position
        self.position = np.array(self._to_world_xy(p), dtype=float)

        # Update speed units
        spd = float(p.get("spd", 0.0))
        if self.SPEED_IS_FT_PER_S:
            spd *= self.FT_TO_M
        self.speed = spd

        # Optional: re-apply true size each step (robust to per-frame size entries)
        if self.UPDATE_SIZE_EACH_STEP:
            self.LENGTH = float(p.get("len", self.LENGTH / self.FT_TO_M)) * self.FT_TO_M
            self.WIDTH  = float(p.get("wid", self.WIDTH / self.FT_TO_M)) * self.FT_TO_M

        # Heading from displacement
        disp = self.position - self.prev_pos
        if np.linalg.norm(disp) > 1e-4:
            self.heading = float(np.arctan2(disp[1], disp[0]))
            self.prev_pos = self.position.copy()
