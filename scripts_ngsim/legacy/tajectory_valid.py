#!/usr/bin/env python3
"""
Test meta-action replay using MDPVehicle.

- Reads raw trajectory + processed meta-actions from NGSIM-like data
- Uses MDPVehicle to replay the meta-actions
- Compares simulated speed & lane vs ground truth
- Plots results

Assumes the following folder structure (relative to repo root):

highway_env/
    data/
        processed_10s/
            us-101/
                t1118846989700/
                    vehicle_record_file.csv
                    processed_dma_data.csv   (created by traj_action)

You can change scene/timestamp via CLI args.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# --- Make sure we can import highway_env from parent dir ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir)

from highway_env.data.traj_to_action import traj_action, colidxmap
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.vehicle.controller import MDPVehicle


# ======================================================================
# Helpers for building a simple road & mapping lane_id <-> lane_index
# ======================================================================

def build_simple_us101_road(num_lanes: int = 5,
                            lane_width: float = 4.0,
                            length: float = 2000.0) -> Road:
    """
    Build a very simple straight multi-lane road:
    lanes are parallel straight lines along x, spaced along y.

    This is NOT the full NGSim geometry, just a simple approximation
    sufficient to exercise MDPVehicle + lane changes.
    """
    net = RoadNetwork()
    start = (0, 0)
    end = (1, 0)

    for lane_id in range(num_lanes):
        y = lane_width * lane_id
        lane = StraightLane(
            [0.0, y],
            [length, y],
            line_types=("continuous", "continuous"),
            width=lane_width,
        )
        # lane index: (start_node, end_node, index)
        net.add_lane(start, end, lane, lane_id)

    road = Road(network=net, np_random=np.random.RandomState(0), record_history=False)
    return road


def lane_id_to_lane_index(lane_id: int) -> tuple:
    """
    Map NGSIM lane_id (1..N) to highway-env lane_index (0..N-1)
    on our simple road.

    lane_id 1 -> index 0, lane_id 2 -> index 1, etc.
    """
    idx = int(lane_id) - 1
    return ("0,0", "1,0", idx)  # will fix this below once we see network keys


def lane_index_to_lane_id(lane_index) -> int:
    """
    Inverse mapping of lane_id_to_lane_index for simple road.
    """
    return int(lane_index[2]) + 1


# ======================================================================
# Core logic: load data, replay meta actions, compare & plot
# ======================================================================

def load_raw_data(save_path: str) -> pd.DataFrame:
    """
    Load raw vehicle_record_file.csv and rename columns using colidxmap.
    """
    input_path = os.path.join(save_path, "vehicle_record_file.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw trajectory file not found: {input_path}")

    data = pd.read_csv(input_path)
    data.columns = [list(colidxmap.keys())[i] for i in range(len(data.columns))]

    # Keep only relevant columns
    relevant = ["veh_ID", "time", "x", "y", "speed", "lane_id"]
    data = data[relevant].sort_values(["veh_ID", "time"]).reset_index(drop=True)
    return data


def load_actions(save_path: str) -> pd.DataFrame:
    """
    Load processed meta actions (output of traj_action).
    """
    action_path = os.path.join(save_path, "processed_dma_data.csv")
    if not os.path.exists(action_path):
        raise FileNotFoundError(f"Processed DMA file not found: {action_path}")

    df = pd.read_csv(action_path)
    # Expect at least: veh_ID, time, action
    return df.sort_values(["veh_ID", "time"]).reset_index(drop=True)


def replay_vehicle(road: Road,
                   veh_df: pd.DataFrame,
                   actions: list[str],
                   dt: float) -> list[MDPVehicle]:
    """
    Replay meta-actions for a single vehicle using MDPVehicle.

    veh_df: dataframe for ONE vehicle, sorted by time, with columns
            ["x", "y", "speed", "lane_id", ...]
    actions: list of meta actions (strings) of same length as veh_df
    dt: time step [s]
    """
    assert len(veh_df) == len(actions), \
        f"veh_df has {len(veh_df)} rows but got {len(actions)} actions"

    # Initialise from first data point
    x0 = veh_df["x"].iloc[0]
    y0 = veh_df["y"].iloc[0]
    v0 = veh_df["speed"].iloc[0]
    lane0 = int(veh_df["lane_id"].iloc[0])

    # Build mapping based on actual road network keys
    # Here we assume network has a single edge from node 0 to 1
    start_nodes = list(road.network.graph.keys())
    if not start_nodes:
        raise RuntimeError("Road network has no nodes.")

    start = start_nodes[0]
    end = list(road.network.graph[start].keys())[0]
    # lane index = (start, end, lane_idx)
    def _lane_id_to_lane_index(lid: int):
        return (start, end, int(lid) - 1)

    def _lane_index_to_lane_id(lindex) -> int:
        return int(lindex[2]) + 1

    # create MDPVehicle
    mdp_vehicle = MDPVehicle(
        road=road,
        position=[x0, y0],
        speed=v0,
        target_lane_index=_lane_id_to_lane_index(lane0),
        # use default target_speeds for now
    )

    # Replay actions
    states = []
    for a in actions:
        mdp_vehicle.act(a)
        mdp_vehicle.step(dt)
        states.append(copy.deepcopy(mdp_vehicle))

    return states, _lane_index_to_lane_id


def evaluate_and_plot(veh_id: int,
                      veh_df: pd.DataFrame,
                      actions_df: pd.DataFrame,
                      dt: float = 0.1) -> None:
    """
    Build simple road, replay meta-actions with MDPVehicle,
    compare speed & lane, and plot trajectories.
    """
    # Filter to this vehicle
    veh_traj = veh_df[veh_df["veh_ID"] == veh_id].sort_values("time").reset_index(drop=True)
    veh_actions = actions_df[actions_df["veh_ID"] == veh_id].sort_values("time").reset_index(drop=True)

    if len(veh_traj) == 0:
        raise ValueError(f"No trajectory found for veh_ID={veh_id}")
    if len(veh_actions) != len(veh_traj):
        raise ValueError(
            f"Trajectory length ({len(veh_traj)}) and action length ({len(veh_actions)}) do not match "
            f"for veh_ID={veh_id}"
        )

    actions = veh_actions["action"].tolist()

    # Build simple road (straight lanes)
    num_lanes = int(veh_traj["lane_id"].max())
    road = build_simple_us101_road(num_lanes=num_lanes)

    # Replay with MDPVehicle
    states, lane_index_to_lane_id = replay_vehicle(road, veh_traj, actions, dt)

    # Extract true and simulated
    true_t = veh_traj["time"].to_numpy()
    true_speed = veh_traj["speed"].to_numpy()
    true_lane = veh_traj["lane_id"].to_numpy()

    sim_speed = np.array([v.speed for v in states])
    sim_lane = np.array([lane_index_to_lane_id(v.lane_index) for v in states])

    # Align lengths
    n = min(len(true_speed), len(sim_speed))
    true_t = true_t[:n]
    true_speed = true_speed[:n]
    true_lane = true_lane[:n]
    sim_speed = sim_speed[:n]
    sim_lane = sim_lane[:n]

    # Metrics
    speed_rmse = np.sqrt(np.mean((sim_speed - true_speed) ** 2))
    lane_acc = np.mean(sim_lane == true_lane)

    print(f"Vehicle {veh_id}:")
    print(f"  Length: {n} steps")
    print(f"  Speed RMSE: {speed_rmse:.3f} m/s")
    print(f"  Lane accuracy: {lane_acc * 100:.1f}%")

    # --- Plots ---
    # Speed vs time
    plt.figure(figsize=(8, 4))
    plt.plot(true_t, true_speed, label="True speed")
    plt.plot(true_t, sim_speed, "--", label="Simulated speed")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title(f"Vehicle {veh_id} speed profile")
    plt.legend()
    plt.grid(True)

    # Lane vs time
    plt.figure(figsize=(8, 4))
    plt.step(true_t, true_lane, where="post", label="True lane")
    plt.step(true_t, sim_lane, where="post", linestyle="--", label="Simulated lane")
    plt.xlabel("Time [s]")
    plt.ylabel("Lane ID")
    plt.title(f"Vehicle {veh_id} lane profile")
    plt.legend()
    plt.grid(True)

    plt.show()


# ======================================================================
# Main entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Test meta-action replay with MDPVehicle")
    parser.add_argument("--scene", type=str, default="us-101/",
                        help="Scene name (subfolder), e.g. 'us-101/'")
    parser.add_argument("--timestamp", type=str, default="t1118846989700",
                        help="Timestamp folder, e.g. 't1118846989700'")
    parser.add_argument("--veh_id", type=int, default=None,
                        help="Vehicle ID to test (default: first one in data)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step [s], default 0.1")
    args = parser.parse_args()

    # Build save_path exactly like your snippet
    save_path = os.path.join(
        parent_dir,  # repo root
        "highway_env",
        "data",
        "processed_10s",
        args.scene,
        args.timestamp,
    )

    print(f"Using save_path: {save_path}")

    # 1) Generate meta-actions (processed_dma_data.csv)
    print("Running traj_action(...) to generate meta-actions...")
    traj_action(save_path)

    # 2) Load raw data and actions
    raw_df = load_raw_data(save_path)
    dma_df = load_actions(save_path)

    # 3) Determine veh_ID to test
    if args.veh_id is None:
        veh_id = int(raw_df["veh_ID"].iloc[0])
        print(f"No veh_id provided, using first vehicle: {veh_id}")
    else:
        veh_id = args.veh_id

    # 4) Evaluate and plot
    evaluate_and_plot(veh_id, raw_df, dma_df, dt=args.dt)


if __name__ == "__main__":
    main()
