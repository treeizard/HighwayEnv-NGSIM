import numpy as np

try:
    from scipy.interpolate import UnivariateSpline
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_to_pi_scalar(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)

def steering_from_curvature_vehicle_model(
    kappa: np.ndarray,
    L_forward: float,
    max_steer: float = np.pi / 4,
) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=float)
    arg = kappa * float(L_forward) / 2.0
    arg = np.clip(arg, -0.999999, 0.999999)
    beta = np.arcsin(arg)
    delta = np.arctan(2.0 * np.tan(beta))
    return np.clip(delta, -max_steer, max_steer)


def traj_to_expert_actions(
    traj: np.ndarray,
    dt: float,
    L_forward: float,
    MAX_STEER: float = np.pi / 4,
    s_xy: float | None = None,
    v_turn_min: float = 0.3,
    hold_steer_when_stopped: bool = True,
    eps_v: float = 1e-3,
    use_arclength: bool = True,
    ds_min: float = 0.01,             # OPTIMIZED default: 1 cm (was 0.1 m)
    denom_min: float = 1e-8,          # slightly lower floor to reduce curvature flattening
    steer_rate_limit: float | None = 6.0,  # less constraining default than 2.0

    v_gate_floor: float = 0.05,
    v_gate_span: float = 0.6,

    jerk_limit: float | None = None,
) -> dict:
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 3:
        raise ValueError("expect [T,>=3] (x,y,speed,...)")

    x = traj[:, 0]
    y = traj[:, 1]
    v_ch = traj[:, 2]

    # --------------------------
    # 1) Valid region mask
    # --------------------------
    INVALID = np.array([0.0, -1.82871076, 0.0, 0.0])
    EPS = 1e-6

    if traj.shape[1] >= 4:
        invalid_sentinel = np.all(np.isclose(traj[:, :4], INVALID, atol=EPS), axis=1)
    else:
        invalid_sentinel = np.zeros(traj.shape[0], dtype=bool)

    invalid_zeros = np.all(np.isclose(traj[:, :3], 0.0, atol=EPS), axis=1)
    valid_mask = ~(invalid_sentinel | invalid_zeros)

    valid_idxs = np.where(valid_mask)[0]
    if valid_idxs.size < 3:
        raise ValueError("Not enough valid points in trajectory to fit spline.")

    start = int(valid_idxs[0])
    end = int(valid_idxs[-1])  # inclusive state index

    xv = x[start:end + 1]
    yv = y[start:end + 1]
    vv = v_ch[start:end + 1]
    T = xv.shape[0]
    if T < 3:
        raise ValueError("Valid window too short.")

    if not _HAVE_SCIPY:
        raise ImportError("SciPy not available. Install scipy or provide a no-scipy fallback.")

    # Clamp ds_min to something sane relative to typical step
    # (prevents accidental ds_min=0.1 from nuking curvature at 10Hz)
    dx0 = np.diff(xv)
    dy0 = np.diff(yv)
    ds0 = np.hypot(dx0, dy0)
    ds_med = float(np.median(ds0[ds0 > 0])) if np.any(ds0 > 0) else 0.0
    if ds_med > 0:
        ds_min_eff = float(np.clip(ds_min, 1e-4, max(0.25 * ds_med, 1e-3)))
    else:
        ds_min_eff = float(max(ds_min, 1e-4))

    # Default smoothing if not provided
    if s_xy is None:
        sigma = 0.02  # meters
        s_xy = float(T) * sigma * sigma

    # --------------------------
    # 2) Curvature estimate
    # --------------------------
    if use_arclength:
        # arc-length parameter s
        s_full = np.concatenate([[0.0], np.cumsum(ds0)])

        # compress near-duplicate points (in s)
        keep = np.ones(T, dtype=bool)
        keep[1:] = ds0 > ds_min_eff
        keep[-1] = True

        s2 = s_full[keep]
        x2 = xv[keep]
        y2 = yv[keep]

        if s2.size < 3 or s2[-1] <= 1e-6:
            kappa_state = np.zeros(T, dtype=float)
            kappa_valid_state = np.zeros(T, dtype=bool)
        else:
            # Scale smoothing for parameter spacing:
            # If the user tuned s_xy for t-grid, make it less sensitive to the chosen parameter.
            # Use ratio of typical step sizes as a heuristic.
            # (This prevents accidental over-smoothing when s has small increments.)
            # t step is dt; s step is ~ds_med.
            scale = (ds_med / max(dt, 1e-6)) if ds_med > 0 else 1.0
            s_xy_scaled = float(s_xy) * float(scale)

            sx = UnivariateSpline(s2, x2, s=s_xy_scaled, k=3)
            sy = UnivariateSpline(s2, y2, s=s_xy_scaled, k=3)

            x1 = sx.derivative(1)(s_full)
            y1 = sy.derivative(1)(s_full)
            x2d = sx.derivative(2)(s_full)
            y2d = sy.derivative(2)(s_full)

            denom = (x1 * x1 + y1 * y1) ** 1.5
            denom = np.maximum(denom, float(denom_min))

            kappa_state = (x1 * y2d - y1 * x2d) / denom

            # mark curvature valid when there is local motion
            motion_proxy = np.concatenate([ds0[:1], ds0])
            kappa_valid_state = motion_proxy > ds_min_eff
    else:
        # time parameter t
        t = np.arange(T, dtype=float) * float(dt)

        sx = UnivariateSpline(t, xv, s=float(s_xy), k=3)
        sy = UnivariateSpline(t, yv, s=float(s_xy), k=3)

        x1 = sx.derivative(1)(t)
        y1 = sy.derivative(1)(t)
        x2d = sx.derivative(2)(t)
        y2d = sy.derivative(2)(t)

        denom = (x1 * x1 + y1 * y1) ** 1.5
        denom = np.maximum(denom, float(denom_min))

        kappa_state = (x1 * y2d - y1 * x2d) / denom
        kappa_valid_state = vv > float(max(v_turn_min, eps_v))

    # --------------------------
    # 3) Transitions
    # --------------------------
    kappa_tr = kappa_state[:-1]
    v_state = vv[:-1]
    kappa_valid_tr = kappa_valid_state[:-1]

    # If curvature invalid, set to 0 (straight). Keep as float array.
    kappa_tr = np.where(kappa_valid_tr, kappa_tr, 0.0)

    # --------------------------
    # 4) Steering from curvature
    # --------------------------
    steer_raw = steering_from_curvature_vehicle_model(kappa_tr, L_forward=L_forward, max_steer=MAX_STEER)

    # --------------------------
    # 4b) Soft low-speed gating (initialized from first raw steer)
    # --------------------------
    if hold_steer_when_stopped:
        v0 = float(v_gate_floor)
        v1 = float(max(v0 + 1e-6, v_turn_min + v_gate_span))
        w = np.clip((v_state - v0) / (v1 - v0), 0.0, 1.0)

        steer_tr = np.empty_like(steer_raw)
        prev = float(steer_raw[0]) if np.isfinite(steer_raw[0]) else 0.0
        steer_tr[0] = prev
        for i in range(1, len(steer_raw)):
            prev = (1.0 - w[i]) * prev + w[i] * float(steer_raw[i])
            steer_tr[i] = prev
    else:
        steer_tr = steer_raw.copy()
        steer_tr[v_state < float(v_turn_min)] = 0.0

    # --------------------------
    # 4c) Steering rate limit (initialized from first value, not 0)
    # --------------------------
    if steer_rate_limit is not None and len(steer_tr) > 0:
        max_d = float(steer_rate_limit) * float(dt)
        prev = float(steer_tr[0]) if np.isfinite(steer_tr[0]) else 0.0
        steer_tr[0] = prev
        for i in range(1, len(steer_tr)):
            d = float(steer_tr[i]) - prev
            d = float(np.clip(d, -max_d, max_d))
            prev = prev + d
            steer_tr[i] = prev

    # --------------------------
    # 5) Acceleration from speed channel (+ optional jerk limit)
    # --------------------------
    accel_tr = (vv[1:] - vv[:-1]) / float(dt)

    if jerk_limit is not None and len(accel_tr) >= 2:
        jmax = float(jerk_limit)
        a_prev = float(accel_tr[0])
        for i in range(1, len(accel_tr)):
            da = float(accel_tr[i]) - a_prev
            da = float(np.clip(da, -jmax * float(dt), jmax * float(dt)))
            a_prev = a_prev + da
            accel_tr[i] = a_prev

    # --------------------------
    # 6) Embed back into full arrays
    # --------------------------
    steering = np.full(traj.shape[0], np.nan, dtype=float)
    accel = np.full(traj.shape[0], np.nan, dtype=float)

    steering[start:end] = steer_tr
    accel[start:end] = accel_tr

    return {
        "accel": accel,
        "steering": steering,
        "valid_mask": valid_mask,
        "start_idx": start,
        "end_idx": end,
        "spline_s": s_xy,
        "kappa_valid": kappa_state,
        "kappa_valid_mask": kappa_valid_state,
        "use_arclength": use_arclength,
        "ds_min": ds_min_eff,  # report effective ds_min
        "steer_raw_valid": steer_raw,
        "v_gate_floor": v_gate_floor,
        "v_gate_span": v_gate_span,
        "steer_rate_limit": steer_rate_limit,
        "jerk_limit": jerk_limit,
    }


class PurePursuitTracker:
    """
    Time-synchronised pure pursuit tracker.

    Compared to the open-loop / nearest-point version, this tracker:
      1) Anchors tracking to the *reference time index* k (policy step).
      2) Uses a progress controller on arc-length (s) to keep simulated motion aligned to
         the reference timeline as well as possible.
      3) Keeps lateral control "human-like" via speed-dependent lookahead + steering dynamics.

    Inputs per step:
      - pos_xy: current simulated position (x,y)
      - heading: current simulated heading (rad)
      - speed: current simulated speed (m/s)

    Outputs per step:
      - steering_cmd (rad), accel_cmd (m/s^2)
      - idx_near (time-anchored nearest/projection hint), idx_tgt (lookahead target index)

    Notes:
      - ref_xy and ref_v are assumed to be sampled at the tracker rate dt (policy rate).
      - If ref_v is None, we estimate v_ref from arc-length increments.
      - This tracker produces plausible controls for YOUR vehicle model; it will not exactly
        reproduce raw NGSIM controls, but will reduce both spatial drift and time warp.
    """

    def __init__(
        self,
        ref_xy: np.ndarray,
        ref_v: np.ndarray | None,
        *,
        dt: float,
        L_forward: float,
        max_steer: float = np.pi / 4,
        # lookahead (meters): Ld = Ld0 + Ld_k * v
        Ld0: float = 5.0,
        Ld_k: float = 0.6,
        Ld_min: float = 3.0,
        Ld_max: float = 30.0,
        # longitudinal (time sync)
        kp_v: float = 0.8,           # speed tracking P gain
        ks_s: float = 0.8,           # progress (arc-length) gain: v_cmd = v_ref + ks_s*(s_des-s_cur)
        v_cmd_min: float = 0.0,
        v_cmd_max: float = 40.0,
        a_min: float = -6.0,
        a_max: float = 4.0,
        jerk_limit: float | None = 10.0,
        # steering dynamics
        steer_rate_limit: float = 6.0,   # rad/s
        steer_lpf_tau: float = 0.15,     # seconds; 0 disables LPF
        # projection window around time index
        proj_back: int = 20,
        proj_fwd: int = 80,
        # if you want to cap how far the tracker can jump in time
        max_time_slip: int = 30,         # indices; limits target index relative to time anchor
    ):
        self.ref_xy = np.asarray(ref_xy, dtype=float)
        if self.ref_xy.ndim != 2 or self.ref_xy.shape[1] != 2:
            raise ValueError("ref_xy must be [N,2]")

        self.ref_v = None if ref_v is None else np.asarray(ref_v, dtype=float)
        if self.ref_v is not None and len(self.ref_v) != len(self.ref_xy):
            raise ValueError("ref_v must have same length as ref_xy (or be None)")

        self.dt = float(dt)
        self.L = float(L_forward)
        self.max_steer = float(max_steer)

        self.Ld0 = float(Ld0)
        self.Ld_k = float(Ld_k)
        self.Ld_min = float(Ld_min)
        self.Ld_max = float(Ld_max)

        self.kp_v = float(kp_v)
        self.ks_s = float(ks_s)
        self.v_cmd_min = float(v_cmd_min)
        self.v_cmd_max = float(v_cmd_max)

        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.jerk_limit = None if jerk_limit is None else float(jerk_limit)

        self.steer_rate_limit = float(steer_rate_limit)
        self.steer_lpf_tau = float(steer_lpf_tau)

        self.proj_back = int(proj_back)
        self.proj_fwd = int(proj_fwd)
        self.max_time_slip = int(max_time_slip)

        # internal state
        self._k = 0
        self._steer_prev = 0.0
        self._a_prev = 0.0

        # precompute arc-length schedule s_ref
        d = np.diff(self.ref_xy, axis=0)
        ds = np.hypot(d[:, 0], d[:, 1])
        self.s_ref = np.concatenate([[0.0], np.cumsum(ds)])  # length N
        self.N = len(self.ref_xy)

        # fallback v_ref from ds/dt (same length N)
        if self.N >= 2:
            v_est = ds / max(self.dt, 1e-6)
            self.v_ref_from_s = np.concatenate([v_est, [v_est[-1]]])
        else:
            self.v_ref_from_s = np.zeros(self.N, dtype=float)

    def reset(self, k0: int = 0) -> None:
        """Reset tracker internal state (call at env.reset)."""
        self._k = int(np.clip(k0, 0, max(0, self.N - 1)))
        self._steer_prev = 0.0
        self._a_prev = 0.0

    # -----------------------------
    # Reference utilities
    # -----------------------------
    def _ref_speed_time(self, k: int, fallback: float) -> float:
        if self.ref_v is not None:
            v = float(self.ref_v[int(np.clip(k, 0, self.N - 1))])
            if np.isfinite(v):
                return v
        # fallback to estimated v from arc-length
        if self.N > 0:
            v = float(self.v_ref_from_s[int(np.clip(k, 0, self.N - 1))])
            if np.isfinite(v):
                return v
        return float(fallback)

    def _project_to_polyline_s(self, pos_xy: np.ndarray, k_hint: int) -> float:
        """
        Project position onto the reference polyline near time index k_hint and return arc-length s.
        Uses a local segment window to avoid snapping to wrong loop/branch.
        """
        p = np.asarray(pos_xy, dtype=float)

        if self.N < 2:
            return 0.0

        i0 = int(np.clip(k_hint - self.proj_back, 0, self.N - 2))
        i1 = int(np.clip(k_hint + self.proj_fwd, 0, self.N - 2))

        best_s = float(self.s_ref[int(np.clip(k_hint, 0, self.N - 1))])
        best_d2 = float("inf")

        for i in range(i0, i1 + 1):
            a = self.ref_xy[i]
            b = self.ref_xy[i + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-12:
                continue
            t = float(np.dot(p - a, ab) / denom)
            t = float(np.clip(t, 0.0, 1.0))
            proj = a + t * ab
            d2 = float(np.dot(p - proj, p - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_s = float(self.s_ref[i] + t * np.sqrt(denom))

        return best_s

    def _target_index_from_time_anchor(self, k: int, Ld: float) -> int:
        """
        Choose a lookahead target index using arc-length *relative to the time anchor* k.
        This keeps lateral control time-consistent.
        """
        if self.N == 0:
            return 0

        k = int(np.clip(k, 0, self.N - 1))
        s0 = float(self.s_ref[k])
        s_target = s0 + float(Ld)

        # linear scan is fine for N~100-300; can replace with searchsorted later
        j = k
        while j + 1 < self.N and self.s_ref[j] < s_target:
            j += 1

        # limit time slip (optional safety)
        if self.max_time_slip > 0:
            j = int(np.clip(j, k - self.max_time_slip, k + self.max_time_slip))
        return j

    # -----------------------------
    # Main step
    # -----------------------------
    def step(
        self,
        pos_xy: np.ndarray,
        heading: float,
        speed: float,
    ) -> tuple[float, float, int, int]:
        """
        Compute closed-loop controls.

        Returns:
          steering_cmd [rad], accel_cmd [m/s^2], k_time (anchor index), target_idx
        """
        if self.N == 0:
            return 0.0, 0.0, 0, 0

        pos_xy = np.asarray(pos_xy, dtype=float)
        v = float(speed)

        # time anchor
        k_time = int(np.clip(self._k, 0, self.N - 1))

        # lookahead based on current speed
        Ld = self.Ld0 + self.Ld_k * max(v, 0.0)
        Ld = float(np.clip(Ld, self.Ld_min, self.Ld_max))

        # time-consistent lateral target
        i_tgt = self._target_index_from_time_anchor(k_time, Ld)
        tgt = self.ref_xy[i_tgt]

        # Pure Pursuit geometry
        dx = float(tgt[0] - pos_xy[0])
        dy = float(tgt[1] - pos_xy[1])
        alpha = wrap_to_pi_scalar(np.arctan2(dy, dx) - float(heading))

        # curvature for pure pursuit: kappa = 2*sin(alpha)/Ld
        kappa = (2.0 * np.sin(alpha)) / max(Ld, 1e-6)

        # Use your highway-env-consistent mapping: curvature -> steering
        steer_raw = steering_from_curvature_vehicle_model(
            np.array([kappa], dtype=float),
            L_forward=self.L,
            max_steer=self.max_steer,
        )[0]
        steer_cmd = float(steer_raw)

        # steering low-pass (human/actuator lag)
        if self.steer_lpf_tau > 1e-6:
            a = self.dt / (self.steer_lpf_tau + self.dt)
            steer_cmd = (1.0 - a) * self._steer_prev + a * steer_cmd

        # steering rate limit
        max_d = self.steer_rate_limit * self.dt
        dsteer = float(np.clip(steer_cmd - self._steer_prev, -max_d, max_d))
        steer_cmd = self._steer_prev + dsteer
        steer_cmd = float(np.clip(steer_cmd, -self.max_steer, self.max_steer))
        self._steer_prev = steer_cmd

        # -----------------------------
        # Time synchronisation (longitudinal)
        # -----------------------------
        # desired progress at this time index
        s_des = float(self.s_ref[k_time])

        # current progress by projecting onto polyline near time anchor
        s_cur = self._project_to_polyline_s(pos_xy, k_hint=k_time)

        # progress error (positive => behind schedule => speed up)
        s_err = s_des - s_cur

        # time-indexed reference speed
        v_ref_t = self._ref_speed_time(k_time, fallback=v)

        # commanded speed with progress correction
        v_cmd = v_ref_t + self.ks_s * s_err
        v_cmd = float(np.clip(v_cmd, self.v_cmd_min, self.v_cmd_max))

        # accel command to track v_cmd
        accel_cmd = self.kp_v * (v_cmd - v)
        accel_cmd = float(np.clip(accel_cmd, self.a_min, self.a_max))

        # jerk limit
        if self.jerk_limit is not None:
            max_da = self.jerk_limit * self.dt
            da = float(np.clip(accel_cmd - self._a_prev, -max_da, max_da))
            accel_cmd = self._a_prev + da
        self._a_prev = accel_cmd

        # advance reference time (one policy step)
        self._k = min(self._k + 1, self.N - 1)

        return steer_cmd, accel_cmd, k_time, i_tgt