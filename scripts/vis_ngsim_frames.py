# scripts/vis_ngsim_frames.py
import argparse, os, csv
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FEET_TO_M = 1.0 / 3.281

# Columns in your file (no header):
# 0:ID, 1:veh_ID, 2:unixtime, 3:Local_X, 4:Local_Y, 5:Global_X, 6:Global_Y,
# 7:v_length, 8:v_Width, 9:v_Class, 10:v_Vel, 11:v_Acc, 12:Lane_ID,
# 13:Preceding, 14:Following, 15:Space_Headway, 16:Time_Headway

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot per-frame NGSIM snapshots: one rectangle per vehicle at each timestamp."
    )
    p.add_argument("--csv", default='./highway_env/data/processed/us-101/vehicle_record_file.csv', help="Path to vehicle_record_file.csv")
    p.add_argument("--outdir", default='./figs_frames_10min_every10s', help="Directory to write PNGs")
    p.add_argument("--start-ms", type=int, default=1118847219700, help="Start unixtime (ms)")
    p.add_argument("--end-ms", type=int, default=1118847339700, help="End unixtime (ms)")
    p.add_argument("--skip-frames", type=int, default=0,
                   help="Plot every N+1-th frame (0 = plot every frame in range)")
    p.add_argument("--units", choices=["feet", "meters"], default="meters")
    p.add_argument("--xlim", type=float, nargs=2, default=[0, 700],
                   help="Longitudinal axis limits (same units as --units)")
    p.add_argument("--ylim", type=float, nargs=2, default=[-5, 30],
                   help="Lateral axis limits (same units as --units)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional cap on number of frames to save")
    return p.parse_args()

def maybe_convert(val, units):
    return val * (FEET_TO_M if units == "meters" else 1.0)

def load_frames(csv_path, start_ms, end_ms):
    """Stream the CSV and bucket rows by exact unixtime (frame)."""
    frames = {}  # t_ms -> list of rows
    t_min, t_max = None, None
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            t = int(row[2])
            if (start_ms is not None and t < start_ms) or (end_ms is not None and t > end_ms):
                if t_min is None:
                    t_min = t
                t_max = t if t_max is None else max(t_max, t)
                continue
            frames.setdefault(t, []).append(row)
            if t_min is None:
                t_min = t
            t_max = t if t_max is None else max(t_max, t)
    return frames, t_min, t_max

def draw_snapshot(rows, units, xlim, ylim, out_path):
    """
    rows: list of CSV rows for a single timestamp.
    units: 'feet' or 'meters'
    xlim: [long_min, long_max]  (road direction = longitudinal)
    ylim: [lat_min, lat_max]    (across lanes = lateral)
    """
    fig, ax = plt.subplots(figsize=(12, 3), dpi=150)

    for row in rows:
        # Local_X (lateral), Local_Y (longitudinal), length/width in feet
        local_x = float(row[3])
        local_y = float(row[4])
        v_len   = float(row[7])
        v_wid   = float(row[8])

        # Convert to requested units
        longi = maybe_convert(local_y, units)
        lat   = maybe_convert(local_x, units)
        L     = maybe_convert(v_len,  units)
        W     = maybe_convert(v_wid,  units)

        # Draw rectangle centered at (longi, lat)
        llx = longi - L / 2.0
        lly = lat   - W / 2.0
        rect = Rectangle((llx, lly), L, W, facecolor="0.6", edgecolor="k", linewidth=0.5, alpha=0.9)
        ax.add_patch(rect)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f"Longitudinal ({'m' if units=='meters' else 'ft'})")
    ax.set_ylabel(f"Lateral ({'m' if units=='meters' else 'ft'})")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    frames, t_min, t_max = load_frames(args.csv, args.start_ms, args.end_ms)
    if not frames:
        print("No frames found in the specified time range.")
        if t_min is not None and t_max is not None:
            print(f"CSV spans [{t_min}, {t_max}] ms.")
        return

    times = sorted(frames.keys())
    print(f"Found {len(times)} frames in range.")

    saved = 0
    for idx, t in enumerate(times):
        if args.skip_frames and (idx % (args.skip_frames + 1)) != 0:
            continue
        out_path = os.path.join(args.outdir, f"frame_{t}.png")
        draw_snapshot(frames[t], args.units, args.xlim, args.ylim, out_path)
        saved += 1
        if args.max_frames is not None and saved >= args.max_frames:
            break
        if saved % 50 == 0:
            print(f"... saved {saved} images")

    print(f"✅ Done. Saved {saved} snapshots to {args.outdir}")

if __name__ == "__main__":
    main()
