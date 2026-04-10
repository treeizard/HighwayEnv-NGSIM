import numpy as np
import pandas as pd
from pyproj import Geod
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt

def add_local_xy(df, lat_col="latitude", lon_col="longitude",
                 basis_lat=34.681580, basis_lon=135.527945):
    """
    Convert lat/lon to local Cartesian coordinates in meters.
    Origin is at (basis_lat, basis_lon).
    
    Returns a copy with:
        x_m: east displacement in meters
        y_m: north displacement in meters
    """
    R = 6378137.0  # WGS84 Earth radius approximation, meters

    out = df.copy()
    lat0 = np.deg2rad(basis_lat)
    lon0 = np.deg2rad(basis_lon)

    lat = np.deg2rad(out[lat_col].astype(float).to_numpy())
    lon = np.deg2rad(out[lon_col].astype(float).to_numpy())

    out["x_m"] = R * (lon - lon0) * np.cos(lat0)
    out["y_m"] = R * (lat - lat0)

    return out


def visualize_trajectories_first_100s(
    df,
    basis_lat=34.681580,
    basis_lon=135.527945,
    time_col="datetime",
    vehicle_col="vehicle_id",
    lat_col="latitude",
    lon_col="longitude",
    detected_flag_col="detected_flag",
    only_detected=True,
    time_unit="ms",      # "ms" or "s"
    figsize=(10, 8),
    linewidth=1.0,
    alpha=0.8,
    show_points=False
):
    """
    Visualize all vehicle trajectories during the first 100 seconds.

    Assumptions:
    - datetime is numeric
    - if time_unit == 'ms', first 100 seconds = first 100000 ms
    - if time_unit == 's', first 100 seconds = first 100 s
    """
    data = df.copy()

    # Optional filtering
    if only_detected and detected_flag_col in data.columns:
        data = data[data[detected_flag_col] == 1].copy()

    # Convert to local XY if not already present
    if "x_m" not in data.columns or "y_m" not in data.columns:
        data = add_local_xy(
            data,
            lat_col=lat_col,
            lon_col=lon_col,
            basis_lat=basis_lat,
            basis_lon=basis_lon
        )

    # Normalize time to start from zero
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col, vehicle_col, "x_m", "y_m"]).copy()

    t0 = data[time_col].min()
    if time_unit == "ms":
        horizon = 100 * 1000
    elif time_unit == "s":
        horizon = 100
    else:
        raise ValueError("time_unit must be 'ms' or 's'")

    data = data[(data[time_col] - t0) <= horizon].copy()

    # Sort for clean trajectory lines
    data = data.sort_values([vehicle_col, time_col])

    # Plot
    plt.figure(figsize=figsize)

    for vid, g in data.groupby(vehicle_col):
        plt.plot(g["x_m"], g["y_m"], linewidth=linewidth, alpha=alpha)
        if show_points:
            plt.scatter(g["x_m"], g["y_m"], s=4, alpha=alpha)

    plt.xlabel("x_m (East, meters)")
    plt.ylabel("y_m (North, meters)")
    plt.title("Vehicle trajectories during first 100 seconds")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return data


def add_local_xy_fast(df, lat_col="latitude", lon_col="longitude",
                      basis_lat=34.681580, basis_lon=135.527945):
    """
    Fast local tangent-plane approximation.
    x_m: east displacement (m)
    y_m: north displacement (m)
    """
    R = 6378137.0

    out = df.copy()
    lat0 = np.deg2rad(basis_lat)
    lon0 = np.deg2rad(basis_lon)

    lat = np.deg2rad(pd.to_numeric(out[lat_col], errors="coerce").to_numpy())
    lon = np.deg2rad(pd.to_numeric(out[lon_col], errors="coerce").to_numpy())

    out["x_m"] = R * (lon - lon0) * np.cos(lat0)
    out["y_m"] = R * (lat - lat0)
    return out


def _estimate_heading_for_vehicle(group, target_time, time_col="datetime", window: int = 2):
    """
    Estimate heading angle (radians) for one vehicle at target_time
    using nearby points in time.
    
    Returns angle measured from +x axis.
    If not enough information exists, returns 0.0.
    """
    g = group.sort_values(time_col)

    times = g[time_col].to_numpy()
    xs = g["x_m"].to_numpy()
    ys = g["y_m"].to_numpy()

    if len(g) == 1:
        return 0.0

    idx = np.argmin(np.abs(times - target_time))
    lo = max(0, idx - window)
    hi = min(len(g), idx + window + 1)

    # Fit a small local line in time to make the heading robust to discretization noise.
    t_local = times[lo:hi].astype(float)
    x_local = xs[lo:hi].astype(float)
    y_local = ys[lo:hi].astype(float)

    dx = dy = 0.0
    if len(t_local) >= 2 and np.ptp(t_local) > 0:
        x_coef = np.polyfit(t_local, x_local, deg=1)
        y_coef = np.polyfit(t_local, y_local, deg=1)
        dx = float(x_coef[0])
        dy = float(y_coef[0])
    elif 0 < idx < len(g) - 1:
        dx = xs[idx + 1] - xs[idx - 1]
        dy = ys[idx + 1] - ys[idx - 1]
    elif idx < len(g) - 1:
        dx = xs[idx + 1] - xs[idx]
        dy = ys[idx + 1] - ys[idx]
    elif idx > 0:
        dx = xs[idx] - xs[idx - 1]
        dy = ys[idx] - ys[idx - 1]

    if np.hypot(dx, dy) < 1e-6:
        return 0.0

    return np.arctan2(dy, dx)


def visualize_vehicle_snapshot(
    df,
    target_time=None,
    time_col="datetime",
    vehicle_col="vehicle_id",
    lat_col="latitude",
    lon_col="longitude",
    length_col="vehicle_length",
    detected_flag_col="detected_flag",
    only_detected=True,
    basis_lat=34.681580,
    basis_lon=135.527945,
    default_width_m=1.8,
    min_length_m=3.0,
    max_length_m=20.0,
    figsize=(14, 8),
    xlim=None,
    ylim=None,
    alpha=0.7,
    edgecolor="black",
    linewidth=0.8,
    annotate_ids=False,
    show_centers=False,
    use_nearest_timestamp=True,
):
    """
    Visualize one timestamp as rectangles representing vehicles.

    Parameters
    ----------
    target_time : numeric or None
        Requested timestamp in same units as df[time_col].
        If None, uses the minimum timestamp.
    use_nearest_timestamp : bool
        If True, chooses the nearest available global timestamp.
        If False, requires exact timestamp match.
    """

    data = df.copy()

    if only_detected and detected_flag_col in data.columns:
        data = data[pd.to_numeric(data[detected_flag_col], errors="coerce") == 1].copy()

    # Ensure local coordinates exist
    if "x_m" not in data.columns or "y_m" not in data.columns:
        data = add_local_xy_fast(
            data,
            lat_col=lat_col,
            lon_col=lon_col,
            basis_lat=basis_lat,
            basis_lon=basis_lon
        )

    # Clean numeric columns
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[vehicle_col] = pd.to_numeric(data[vehicle_col], errors="coerce")
    data[length_col] = pd.to_numeric(data[length_col], errors="coerce")

    data = data.dropna(subset=[time_col, vehicle_col, "x_m", "y_m"]).copy()

    if len(data) == 0:
        raise ValueError("No valid rows available after cleaning/filtering.")

    # Choose timestamp
    unique_times = np.sort(data[time_col].unique())
    if target_time is None:
        chosen_time = unique_times[0]
    else:
        if use_nearest_timestamp:
            chosen_time = unique_times[np.argmin(np.abs(unique_times - target_time))]
        else:
            if target_time not in set(unique_times):
                raise ValueError(f"Exact timestamp {target_time} not found in data.")
            chosen_time = target_time

    snapshot = data[data[time_col] == chosen_time].copy()

    if len(snapshot) == 0:
        raise ValueError(f"No rows found at chosen timestamp {chosen_time}.")

    # Clean / clamp lengths
    if length_col in snapshot.columns:
        snapshot[length_col] = snapshot[length_col].fillna(5.0).clip(min_length_m, max_length_m)
    else:
        snapshot[length_col] = 5.0

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Pre-group whole data once for heading estimation
    vehicle_groups = {vid: g for vid, g in data.groupby(vehicle_col)}

    for _, row in snapshot.iterrows():
        vid = row[vehicle_col]
        x = row["x_m"]
        y = row["y_m"]
        length = float(row[length_col])
        width = float(default_width_m)

        heading = _estimate_heading_for_vehicle(vehicle_groups[vid], chosen_time, time_col=time_col)
        heading_deg = np.rad2deg(heading)

        # Rectangle centered at (x, y), initially axis-aligned along +x
        rect = Rectangle(
            (x - length / 2.0, y - width / 2.0),
            length,
            width,
            facecolor="C0",
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha
        )

        transform = Affine2D().rotate_deg_around(x, y, heading_deg) + ax.transData
        rect.set_transform(transform)
        ax.add_patch(rect)

        if show_centers:
            ax.plot(x, y, marker="o", markersize=2)

        if annotate_ids:
            ax.text(x, y, str(int(vid)), fontsize=7, ha="center", va="center")

    ax.set_xlabel("x_m (East, meters)")
    ax.set_ylabel("y_m (North, meters)")
    ax.set_title(f"Vehicle snapshot at timestamp {chosen_time}")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return snapshot, chosen_time


def load_morinomiya_npy(
    npy_path,
    basis_lat=34.681580,
    basis_lon=135.527945,
    lat_col="latitude",
    lon_col="longitude",
    only_detected=False,
    detected_flag_col="detected_flag",
    x_m_max=None,
):
    """
    Load a saved Morinomiya .npy record array into a cleaned DataFrame.

    If x_m / y_m are missing, they are generated using the same local XY
    convention used elsewhere in this file.
    """
    arr = np.load(npy_path, allow_pickle=True)
    df = pd.DataFrame(arr).copy()

    numeric_cols = [
        "vehicle_id",
        "datetime",
        "traffic_lane",
        lat_col,
        lon_col,
        detected_flag_col,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["vehicle_id", "datetime", "traffic_lane"]).copy()
    df["vehicle_id"] = df["vehicle_id"].astype(np.int64)
    df["datetime"] = df["datetime"].astype(np.int64)
    df["traffic_lane"] = df["traffic_lane"].astype(np.int64)

    if only_detected and detected_flag_col in df.columns:
        df = df[df[detected_flag_col] == 1].copy()

    if "x_m" not in df.columns or "y_m" not in df.columns:
        df = add_local_xy_fast(
            df,
            lat_col=lat_col,
            lon_col=lon_col,
            basis_lat=basis_lat,
            basis_lon=basis_lon,
        )

    df["x_m"] = pd.to_numeric(df["x_m"], errors="coerce")
    df["y_m"] = pd.to_numeric(df["y_m"], errors="coerce")
    df = df.dropna(subset=["x_m", "y_m"]).copy()

    if x_m_max is not None:
        df = df[df["x_m"] <= x_m_max].copy()

    return df.sort_values(["vehicle_id", "datetime"]).reset_index(drop=True)


def build_spawn_summary(
    df,
    vehicle_col="vehicle_id",
    time_col="datetime",
    lane_col="traffic_lane",
):
    """
    One row per vehicle describing when and where it first appears.
    """
    ordered = df.sort_values([vehicle_col, time_col]).copy()
    spawn = (
        ordered.groupby(vehicle_col, as_index=False)
        .first()
        [[vehicle_col, time_col, lane_col, "x_m", "y_m"]]
        .rename(
            columns={
                time_col: "spawn_time",
                lane_col: "spawn_lane",
                "x_m": "spawn_x_m",
                "y_m": "spawn_y_m",
            }
        )
    )
    spawn["spawn_lane"] = spawn["spawn_lane"].astype(np.int64)
    return spawn.sort_values(["spawn_time", vehicle_col]).reset_index(drop=True)


def compute_vehicle_path_lengths(
    df,
    vehicle_col="vehicle_id",
    time_col="datetime",
    x_col="x_m",
    y_col="y_m",
):
    """
    Path length is the accumulated Euclidean distance along each trajectory.
    """
    summaries = []

    for vehicle_id, group in df.groupby(vehicle_col):
        g = group.sort_values(time_col)
        x = g[x_col].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)

        if len(g) < 2:
            path_length_m = 0.0
        else:
            segment_lengths = np.hypot(np.diff(x), np.diff(y))
            path_length_m = float(segment_lengths.sum())

        summaries.append(
            {
                vehicle_col: int(vehicle_id),
                "path_length_m": path_length_m,
                "n_points": int(len(g)),
                "start_time": int(g[time_col].iloc[0]),
                "end_time": int(g[time_col].iloc[-1]),
            }
        )

    return pd.DataFrame(summaries).sort_values(vehicle_col).reset_index(drop=True)


def build_lane_path_length_summary(
    df,
    lane_focus=3,
    vehicle_col="vehicle_id",
    time_col="datetime",
    lane_col="traffic_lane",
):
    """
    Combine spawn-lane metadata and per-vehicle path lengths.
    """
    spawn = build_spawn_summary(df, vehicle_col=vehicle_col, time_col=time_col, lane_col=lane_col)
    path_lengths = compute_vehicle_path_lengths(df, vehicle_col=vehicle_col, time_col=time_col)
    summary = spawn.merge(path_lengths, on=vehicle_col, how="inner")
    summary["spawn_group"] = np.where(
        summary["spawn_lane"] == lane_focus,
        f"lane_{lane_focus}",
        "other_lanes",
    )
    return summary


def find_same_spawn_time_lane_pair(
    df,
    lane_focus=3,
    matched_time=None,
    lane_vehicle_id=None,
    other_vehicle_id=None,
    vehicle_col="vehicle_id",
    time_col="datetime",
    lane_col="traffic_lane",
):
    """
    Find a lane_focus vehicle and a non-lane_focus vehicle that first appear
    at the same timestamp.
    """
    spawn = build_lane_path_length_summary(
        df,
        lane_focus=lane_focus,
        vehicle_col=vehicle_col,
        time_col=time_col,
        lane_col=lane_col,
    )

    if matched_time is None:
        counts = (
            spawn.groupby("spawn_time")
            .agg(
                lane_focus_count=("spawn_lane", lambda s: int((s == lane_focus).sum())),
                other_count=("spawn_lane", lambda s: int((s != lane_focus).sum())),
            )
            .reset_index()
        )
        counts = counts[(counts["lane_focus_count"] > 0) & (counts["other_count"] > 0)]
        if counts.empty:
            raise ValueError(
                f"No spawn time contains both a lane {lane_focus} vehicle and a non-lane {lane_focus} vehicle."
            )
        matched_time = int(counts.sort_values("spawn_time").iloc[0]["spawn_time"])

    candidates = spawn[spawn["spawn_time"] == matched_time].copy()
    lane_candidates = candidates[candidates["spawn_lane"] == lane_focus].copy()
    other_candidates = candidates[candidates["spawn_lane"] != lane_focus].copy()

    if lane_candidates.empty or other_candidates.empty:
        raise ValueError(
            f"Spawn time {matched_time} does not contain both lane {lane_focus} and non-lane {lane_focus} vehicles."
        )

    if lane_vehicle_id is not None:
        lane_candidates = lane_candidates[lane_candidates[vehicle_col] == lane_vehicle_id].copy()
    if other_vehicle_id is not None:
        other_candidates = other_candidates[other_candidates[vehicle_col] == other_vehicle_id].copy()

    if lane_candidates.empty:
        raise ValueError("Requested lane_focus vehicle is not available at the chosen spawn time.")
    if other_candidates.empty:
        raise ValueError("Requested comparison vehicle is not available at the chosen spawn time.")

    lane_row = lane_candidates.sort_values(["path_length_m", vehicle_col], ascending=[False, True]).iloc[0]
    other_row = other_candidates.sort_values(["path_length_m", vehicle_col], ascending=[False, True]).iloc[0]

    return {
        "matched_time": int(matched_time),
        "lane_focus_vehicle_id": int(lane_row[vehicle_col]),
        "other_vehicle_id": int(other_row[vehicle_col]),
        "lane_focus_spawn_lane": int(lane_row["spawn_lane"]),
        "other_spawn_lane": int(other_row["spawn_lane"]),
        "lane_focus_path_length_m": float(lane_row["path_length_m"]),
        "other_path_length_m": float(other_row["path_length_m"]),
    }


def plot_same_spawn_time_path_pair(
    df,
    lane_focus=3,
    matched_time=None,
    lane_vehicle_id=None,
    other_vehicle_id=None,
    vehicle_col="vehicle_id",
    time_col="datetime",
    lane_col="traffic_lane",
    use_shared_horizon=True,
    figsize=(10, 6),
    linewidth=2.0,
):
    """
    Plot a lane_focus trajectory against a same-spawn-time comparison vehicle.

    If use_shared_horizon=True, both paths are cropped to the shared time range
    so the visual comparison covers the same elapsed duration after spawning.
    """
    pair = find_same_spawn_time_lane_pair(
        df,
        lane_focus=lane_focus,
        matched_time=matched_time,
        lane_vehicle_id=lane_vehicle_id,
        other_vehicle_id=other_vehicle_id,
        vehicle_col=vehicle_col,
        time_col=time_col,
        lane_col=lane_col,
    )

    lane_traj = (
        df[df[vehicle_col] == pair["lane_focus_vehicle_id"]]
        .sort_values(time_col)
        .copy()
    )
    other_traj = (
        df[df[vehicle_col] == pair["other_vehicle_id"]]
        .sort_values(time_col)
        .copy()
    )

    if use_shared_horizon:
        shared_end = min(
            int(lane_traj[time_col].max()),
            int(other_traj[time_col].max()),
        )
        lane_traj = lane_traj[lane_traj[time_col] <= shared_end].copy()
        other_traj = other_traj[other_traj[time_col] <= shared_end].copy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        lane_traj["x_m"],
        lane_traj["y_m"],
        linewidth=linewidth,
        color="C3",
        label=f"Lane {lane_focus} vehicle {pair['lane_focus_vehicle_id']}",
    )
    ax.plot(
        other_traj["x_m"],
        other_traj["y_m"],
        linewidth=linewidth,
        color="C0",
        label=f"Other-lane vehicle {pair['other_vehicle_id']} (lane {pair['other_spawn_lane']})",
    )

    ax.scatter(lane_traj["x_m"].iloc[0], lane_traj["y_m"].iloc[0], color="C3", s=50, marker="o")
    ax.scatter(other_traj["x_m"].iloc[0], other_traj["y_m"].iloc[0], color="C0", s=50, marker="o")
    ax.scatter(lane_traj["x_m"].iloc[-1], lane_traj["y_m"].iloc[-1], color="C3", s=50, marker="x")
    ax.scatter(other_traj["x_m"].iloc[-1], other_traj["y_m"].iloc[-1], color="C0", s=50, marker="x")

    ax.set_xlabel("x_m (East, meters)")
    ax.set_ylabel("y_m (North, meters)")
    ax.set_title(
        f"Same-spawn-time path comparison at {pair['matched_time']}: "
        f"lane {lane_focus} vs other lane"
    )
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return pair, lane_traj, other_traj


def plot_average_path_length_comparison(
    df,
    lane_focus=3,
    vehicle_col="vehicle_id",
    time_col="datetime",
    lane_col="traffic_lane",
    figsize=(8, 5),
):
    """
    Compare average total path length for vehicles spawned from lane_focus
    against vehicles spawned from all other lanes.
    """
    summary = build_lane_path_length_summary(
        df,
        lane_focus=lane_focus,
        vehicle_col=vehicle_col,
        time_col=time_col,
        lane_col=lane_col,
    )

    comparison = (
        summary.groupby("spawn_group")
        .agg(
            mean_path_length_m=("path_length_m", "mean"),
            median_path_length_m=("path_length_m", "median"),
            vehicle_count=(vehicle_col, "count"),
        )
        .reset_index()
    )

    order = [f"lane_{lane_focus}", "other_lanes"]
    comparison["spawn_group"] = pd.Categorical(comparison["spawn_group"], categories=order, ordered=True)
    comparison = comparison.sort_values("spawn_group").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        comparison["spawn_group"].astype(str),
        comparison["mean_path_length_m"],
        color=["C3", "C0"],
        alpha=0.85,
    )

    for bar, row in zip(bars, comparison.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{row.mean_path_length_m:.1f} m\nn={row.vehicle_count}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Average path length (m)")
    ax.set_title(f"Average path length: lane {lane_focus} spawn vs other lanes")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return comparison, summary
