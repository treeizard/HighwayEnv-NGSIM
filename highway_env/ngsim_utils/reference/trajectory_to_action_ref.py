# highway_env/ngsim_utils/trajectory_to_action.py
import numpy as np

try:
    from scipy.interpolate import UnivariateSpline
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def steering_from_curvature_vehicle_model(
    kappa: np.ndarray,
    L_forward: float,
    max_steer: float = np.pi / 4,
) -> np.ndarray:
    """
    Invert highway-env kinematic relation using curvature kappa = yaw_rate / v.

    Forward (your model):
        beta = atan(0.5 * tan(delta))
        yaw_rate = 2*v*sin(beta)/L
    => curvature kappa = yaw_rate / v = 2*sin(beta)/L

    Invert:
        sin(beta) = kappa * L / 2
        beta = arcsin(...)
        delta = atan(2*tan(beta))
    """
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
    # spline smoothing: higher => smoother (less curvature noise)
    s_xy: float | None = None,
    # low-speed handling
    v_turn_min: float = 0.8,          # m/s: below this, steer not observable
    hold_steer_when_stopped: bool = True,
    # numerical safety
    eps_v: float = 1e-3,
) -> dict:
    """
    Spline-based trajectory->(accel, steering) conversion.

    - Fits smoothing splines x(t), y(t)
    - Computes curvature kappa(t) from derivatives
    - Inverts curvature to steering using the SAME highway-env mapping
    - Uses speed-channel for acceleration (as before)

    Returns dict with accel/steering arrays of length T (NaNs outside valid range).
    Actions are defined on transitions t in [start, end).
    """
    traj = np.asarray(traj, dtype=float)
    assert traj.ndim == 2 and traj.shape[1] >= 3, "expect [T,>=3] (x,y,speed,...)"

    x = traj[:, 0]
    y = traj[:, 1]
    v_ch = traj[:, 2]

    # --------------------------
    # 1) Valid region mask (your existing logic)
    # --------------------------
    INVALID = np.array([0.0, -1.82871076, 0.0, 0.0])
    EPS = 1e-6

    invalid_sentinel = (
        np.all(np.isclose(traj[:, :4], INVALID, atol=EPS), axis=1)
        if traj.shape[1] >= 4
        else np.zeros(traj.shape[0], dtype=bool)
    )
    invalid_zeros = np.all(np.isclose(traj[:, :3], 0.0, atol=EPS), axis=1)
    valid_mask = ~(invalid_sentinel | invalid_zeros)

    valid_idxs = np.where(valid_mask)[0]
    if valid_idxs.size < 3:
        raise ValueError("Not enough valid points in trajectory to fit spline.")

    start = int(valid_idxs[0])
    end = int(valid_idxs[-1])  # inclusive state index

    xv = x[start : end + 1].copy()
    yv = y[start : end + 1].copy()
    vv = v_ch[start : end + 1].copy()
    T = xv.shape[0]
    if T < 3:
        raise ValueError("Valid window too short.")

    # time grid for valid states
    t = np.arange(T, dtype=float) * float(dt)

    # --------------------------
    # 2) Fit smoothing splines x(t), y(t)
    # --------------------------
    if not _HAVE_SCIPY:
        raise ImportError(
            "SciPy not available. Install scipy or tell me and I will provide a no-scipy spline/SG fallback."
        )

    # If user doesn't set s_xy, choose a conservative default.
    # Heuristic: s ~ N * sigma^2. If positions are ~cm noise, sigma=0.02m.
    if s_xy is None:
        sigma = 0.02  # meters (tune if needed)
        s_xy = float(T) * sigma * sigma

    sx = UnivariateSpline(t, xv, s=s_xy, k=3)
    sy = UnivariateSpline(t, yv, s=s_xy, k=3)

    # derivatives at state timestamps
    x1 = sx.derivative(1)(t)
    y1 = sy.derivative(1)(t)
    x2 = sx.derivative(2)(t)
    y2 = sy.derivative(2)(t)

    # --------------------------
    # 3) Curvature kappa(t) from derivatives
    # kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
    # --------------------------
    denom = (x1 * x1 + y1 * y1) ** 1.5
    denom = np.maximum(denom, 1e-9)
    kappa = (x1 * y2 - y1 * x2) / denom  # length T (states)

    # Actions live on transitions [0..T-2] within this valid window
    kappa_tr = kappa[:-1].copy()  # length T-1
    v_state = vv[:-1].copy()      # speed channel aligned to transitions

    # --------------------------
    # 4) Steering from curvature (model-consistent)
    # --------------------------
    steer_tr = steering_from_curvature_vehicle_model(
        kappa=kappa_tr, L_forward=L_forward, max_steer=MAX_STEER
    )

    # low-speed handling: when speed is too low, steering is not observable
    low_speed = v_state < float(v_turn_min)
    if np.any(low_speed):
        if hold_steer_when_stopped:
            # hold last non-low-speed value; if none, hold 0
            last = 0.0
            for i in range(len(steer_tr)):
                if low_speed[i]:
                    steer_tr[i] = last
                else:
                    last = steer_tr[i]
        else:
            steer_tr[low_speed] = 0.0

    # --------------------------
    # 5) Acceleration from speed channel (as before)
    # accel defined on transitions, length T-1
    # --------------------------
    accel_tr = (vv[1:] - vv[:-1]) / float(dt)

    # --------------------------
    # 6) Embed back into full arrays (length original T)
    # actions defined for transitions t in [start, end)
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
        # diagnostics (optional)
        "spline_s": s_xy,
        "kappa_valid": kappa,   # state curvature for valid slice
    }
