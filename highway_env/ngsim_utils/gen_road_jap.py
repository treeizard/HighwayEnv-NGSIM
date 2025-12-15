import numpy as np
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import RoadNetwork

def create_hanshin_complex_road():
    net = RoadNetwork()
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

    # --- DIMENSIONS (Metric) ---
    W_L = 3.25   # Traffic Lane Width (3.25m)
    W_R = 3.25   # Ramp/Auxiliary Lane Width
    # Offset for the center of the first lane (Lane 0)
    Y0 = W_L / 2 
    
    # --- X-COORDINATES (Meters relative to 1.70KP) ---
    KP_START = 1.70 # Starting KP reference
    X_START = 0.0
    X_RAMP_DIVERGE = (1.78 - KP_START) * 1000 # ~ 80m
    X_MERGE_POINT  = (1.89 - KP_START) * 1000 # ~ 190m
    X_AUX_ENDS     = (2.0 - KP_START) * 1000  # ~ 300m
    X_END          = (2.30 - KP_START) * 1000 # ~ 600m

    # --- 1. OSAKA BOUND DIRECTION (3 Lanes + Ramp/Auxiliary) ---
    # Lanes indexed from right (median) to left (outer edge)

    # Section 1: Merge/Ramp Area (Entry: "merge_in", Exit: "s2")
    # This models the ramp merging into the main road (creating a temporary 3rd lane).
    
    # A. RAMP LANE (Lane 0) - Merges from an external source
    # Y-offset for the ramp start must be determined visually, let's assume it starts centered at Y0
    net.add_lane(
        "merge_in", "s2", 
        StraightLane([X_RAMP_DIVERGE, Y0], [X_MERGE_POINT, Y0], 
                     width=W_R, line_types=[c, c], forbidden=False)
    )

    # B. INNER MAIN LANE (Lane 1) - Moves from right-most main lane
    # Starts at an offset to accommodate the ramp
    net.add_lane(
        "s1", "s2", 
        StraightLane([X_START, Y0 + W_R], [X_MERGE_POINT, Y0 + W_R], 
                     width=W_L, line_types=[c, s]
        )
    )

    # C. OUTER MAIN LANE (Lane 2)
    net.add_lane(
        "s1", "s2", 
        StraightLane([X_START, Y0 + W_R + W_L], [X_MERGE_POINT, Y0 + W_R + W_L], 
                     width=W_L, line_types=[s, c]
        )
    )

    # Section 2: Main Road after Merge (Entry: "s2", Exit: "s3")
    # This models the road where the auxiliary lane is present before dropping off (likely 3 lanes here)
    for lane_index in range(3):
        net.add_lane(
            "s2", "s3", 
            StraightLane([X_MERGE_POINT, Y0 + lane_index * W_L], [X_AUX_ENDS, Y0 + lane_index * W_L], 
                         width=W_L, 
                         line_types=[c if lane_index == 0 else s, c if lane_index == 2 else s]
            )
        )
    
    # Section 3: Final 2-Lane Road (Entry: "s3", Exit: "exit")
    # The auxiliary lane drops off, returning to the standard 2-lane structure
    for lane_index in range(2):
        net.add_lane(
            "s3", "exit", 
            StraightLane([X_AUX_ENDS, Y0 + lane_index * W_L], [X_END, Y0 + lane_index * W_L], 
                         width=W_L, 
                         line_types=[c if lane_index == 0 else s, c if lane_index == 1 else s]
            )
        )

    # --- 2. OPPOSING DIRECTION (2 Lanes) ---
    # Need a large offset to account for the physical median gap. 
    # Let's use a MEDIAN_GAP of 10m (can be adjusted).
    MEDIAN_GAP = 10.0
    Y_OPPOSITE_OFFSET = Y0 + 2 * W_L + MEDIAN_GAP 
    
    # Opposing lanes flow in the opposite direction (from X_END to X_START)
    # The start/end points of the StraightLane need to be reversed, AND 
    # the y-coordinates might need adjustment depending on how you define "Lane 0" for this side.
    
    # Assuming the lanes are numbered from the outer edge inwards (0 is outer, 1 is inner/median)
    for lane_index in range(2):
        net.add_lane(
            "opp_exit", "opp_entry", 
            StraightLane([X_END, Y_OPPOSITE_OFFSET + lane_index * W_L], 
                         [X_START, Y_OPPOSITE_OFFSET + lane_index * W_L], 
                         width=W_L, 
                         line_types=[c if lane_index == 0 else s, c if lane_index == 1 else s],
                         # Optional: Reverse the line types if the simulation logic expects it
            )
        )
        
    return net