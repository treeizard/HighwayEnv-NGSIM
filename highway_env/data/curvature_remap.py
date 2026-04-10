from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def _principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return origin and rotation matrix for the dominant road direction."""
    origin = points.mean(axis=0)
    centered = points - origin
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    road_dir = vt[0]
    lateral_dir = vt[1]
    if road_dir[0] < 0:
        road_dir = -road_dir
        lateral_dir = -lateral_dir
    rotation = np.column_stack([road_dir, lateral_dir])
    return origin, rotation


def estimate_centerline_from_lanes(
    df: pd.DataFrame,
    lane_col: str = "traffic_lane",
    x_col: str = "x_m",
    y_col: str = "y_m",
    lanes: list[int] | None = None,
    bin_size_m: float = 5.0,
    spline_smooth: float | None = None,
) -> pd.DataFrame:
    """
    Estimate a smooth road centerline from lane-wise trajectory medians.

    The idea is:
    1. rotate the scene into a principal road-aligned frame,
    2. summarize each lane by robust medians in longitudinal bins,
    3. average the lane center traces,
    4. smooth the resulting centerline with a spline.
    """
    data = df.dropna(subset=[lane_col, x_col, y_col]).copy()
    data[[x_col, y_col]] = data[[x_col, y_col]].astype(float)
    data[lane_col] = data[lane_col].astype(int)

    if lanes is None:
        lane_counts = data[lane_col].value_counts().sort_index()
        lanes = lane_counts.index.tolist()
    else:
        lanes = [int(l) for l in lanes]

    data = data[data[lane_col].isin(lanes)].copy()
    if data.empty:
        raise ValueError("No rows remain after filtering lanes for centerline estimation.")

    points = data[[x_col, y_col]].to_numpy()
    origin, rotation = _principal_frame(points)
    aligned = (points - origin) @ rotation
    data["_u"] = aligned[:, 0]
    data["_v"] = aligned[:, 1]

    u_min = float(data["_u"].min())
    u_max = float(data["_u"].max())
    bins = np.arange(u_min, u_max + bin_size_m, bin_size_m)
    if len(bins) < 2:
        raise ValueError("Insufficient longitudinal span to estimate a centerline.")

    lane_profiles: list[pd.DataFrame] = []
    for lane in lanes:
        lane_df = data[data[lane_col] == lane].copy()
        if lane_df.empty:
            continue

        lane_df["_bin"] = pd.cut(lane_df["_u"], bins=bins, labels=False, include_lowest=True)
        profile = lane_df.groupby("_bin")[["_u", "_v"]].median().dropna()
        if len(profile) < 3:
            continue
        lane_profiles.append(profile)

    if not lane_profiles:
        raise ValueError("Could not form lane profiles from the selected lanes.")

    center_profile = pd.concat(lane_profiles).groupby(level=0)[["_u", "_v"]].mean()
    center_profile = center_profile.dropna().sort_values("_u")
    if len(center_profile) < 4:
        raise ValueError("Centerline estimate is too short to smooth reliably.")

    u = center_profile["_u"].to_numpy()
    v = center_profile["_v"].to_numpy()
    smooth = spline_smooth if spline_smooth is not None else len(u) * 0.5
    spline = UnivariateSpline(u, v, s=smooth)

    u_dense = np.linspace(float(u.min()), float(u.max()), num=max(200, len(u) * 4))
    v_dense = spline(u_dense)
    dv_du = spline.derivative(1)(u_dense)
    d2v_du2 = spline.derivative(2)(u_dense)

    aligned_center = np.column_stack([u_dense, v_dense])
    world_center = aligned_center @ rotation.T + origin

    ds = np.sqrt(np.diff(u_dense) ** 2 + np.diff(v_dense) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    curvature = np.abs(d2v_du2) / np.power(1.0 + dv_du**2, 1.5)

    tangent_aligned = np.column_stack([np.ones_like(dv_du), dv_du])
    tangent_aligned /= np.linalg.norm(tangent_aligned, axis=1, keepdims=True)
    normal_aligned = np.column_stack([-tangent_aligned[:, 1], tangent_aligned[:, 0]])

    tangent_world = tangent_aligned @ rotation.T
    normal_world = normal_aligned @ rotation.T

    return pd.DataFrame(
        {
            "center_x": world_center[:, 0],
            "center_y": world_center[:, 1],
            "s": s,
            "curvature": curvature,
            "tangent_x": tangent_world[:, 0],
            "tangent_y": tangent_world[:, 1],
            "normal_x": normal_world[:, 0],
            "normal_y": normal_world[:, 1],
        }
    )


def project_points_to_centerline(
    xy: np.ndarray,
    centerline_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world-frame points to a centerline and return:
    - longitudinal arc length s,
    - signed lateral offset d,
    - index of the closest centerline sample.
    """
    center_xy = centerline_df[["center_x", "center_y"]].to_numpy(dtype=float)
    s_values = centerline_df["s"].to_numpy(dtype=float)

    if len(center_xy) < 2:
        raise ValueError("Centerline must contain at least two samples for projection.")

    seg_start = center_xy[:-1]
    seg_end = center_xy[1:]
    seg_vec = seg_end - seg_start
    seg_len2 = np.sum(seg_vec * seg_vec, axis=1)
    seg_len = np.sqrt(seg_len2)

    # Project every point onto every segment and keep the closest valid projection.
    rel = xy[:, None, :] - seg_start[None, :, :]
    dot = np.sum(rel * seg_vec[None, :, :], axis=2)
    t = np.divide(dot, seg_len2[None, :], out=np.zeros_like(dot), where=seg_len2[None, :] > 0)
    t = np.clip(t, 0.0, 1.0)

    proj = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
    resid = xy[:, None, :] - proj
    dist2 = np.sum(resid * resid, axis=2)
    seg_idx = np.argmin(dist2, axis=1)

    chosen_t = t[np.arange(len(xy)), seg_idx]
    chosen_proj = proj[np.arange(len(xy)), seg_idx]
    chosen_vec = seg_vec[seg_idx]
    chosen_len = seg_len[seg_idx]

    tangent = np.divide(
        chosen_vec,
        chosen_len[:, None],
        out=np.zeros_like(chosen_vec),
        where=chosen_len[:, None] > 0,
    )
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    signed_d = np.sum((xy - chosen_proj) * normal, axis=1)
    s = s_values[seg_idx] + chosen_t * chosen_len
    return s, signed_d, seg_idx


def remap_dataframe_to_frenet(
    df: pd.DataFrame,
    centerline_df: pd.DataFrame,
    x_col: str = "x_m",
    y_col: str = "y_m",
    s_col: str = "x_curved",
    d_col: str = "y_curved",
) -> pd.DataFrame:
    """Attach curvature-aware straightened coordinates to a dataframe."""
    out = df.copy()
    valid_mask = out[[x_col, y_col]].notna().all(axis=1)
    xy = out.loc[valid_mask, [x_col, y_col]].to_numpy(dtype=float)

    s, d, idx = project_points_to_centerline(xy, centerline_df)
    out.loc[valid_mask, s_col] = s
    out.loc[valid_mask, d_col] = d
    out.loc[valid_mask, "centerline_idx"] = idx
    return out


def estimate_curvature_remap(
    df: pd.DataFrame,
    lane_col: str = "traffic_lane",
    x_col: str = "x_m",
    y_col: str = "y_m",
    lanes: list[int] | None = None,
    bin_size_m: float = 5.0,
    spline_smooth: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper returning:
    - remapped dataframe with (x_curved, y_curved)
    - sampled centerline with curvature
    """
    centerline_df = estimate_centerline_from_lanes(
        df=df,
        lane_col=lane_col,
        x_col=x_col,
        y_col=y_col,
        lanes=lanes,
        bin_size_m=bin_size_m,
        spline_smooth=spline_smooth,
    )
    remapped_df = remap_dataframe_to_frenet(
        df=df,
        centerline_df=centerline_df,
        x_col=x_col,
        y_col=y_col,
    )
    return remapped_df, centerline_df
