"""Define shared constants and configuration for NGSIM integration."""

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

import os

import numpy as np

# Dataset/unit conversion
FEET_PER_METER = 3.281
METERS_PER_FOOT = 1.0 / FEET_PER_METER

# Vehicle/control limits. The original NGSIM controller and the US expert
# collection both use a symmetric 5 m/s² limit. A June 2026 zero-centering
# change accidentally coupled that fix to a wider ±10 m/s² range. Keep the
# correct zero-centered mapping, restore the physical default to ±5 m/s², and
# retain an explicit environment override for auditable legacy prototypes.
_ACCELERATION_LIMIT_ENV = "NGSIM_ACCELERATION_LIMIT_MPS2"
_raw_acceleration_limit = os.environ.get(_ACCELERATION_LIMIT_ENV, "5.0")
try:
    ACCELERATION_LIMIT_MPS2 = float(_raw_acceleration_limit)
except ValueError as exc:
    raise ValueError(
        f"{_ACCELERATION_LIMIT_ENV} must be a finite positive number, "
        f"got {_raw_acceleration_limit!r}."
    ) from exc
if not np.isfinite(ACCELERATION_LIMIT_MPS2) or ACCELERATION_LIMIT_MPS2 <= 0.0:
    raise ValueError(
        f"{_ACCELERATION_LIMIT_ENV} must be a finite positive number, "
        f"got {ACCELERATION_LIMIT_MPS2!r}."
    )
ACCELERATION_RANGE = (-ACCELERATION_LIMIT_MPS2, ACCELERATION_LIMIT_MPS2)
MIN_ACCEL = ACCELERATION_RANGE[0]
MAX_ACCEL = ACCELERATION_RANGE[1]
MAX_STEER = np.pi / 4


def normalize_acceleration(acceleration: float) -> float:
    """Map physical acceleration to the normalized env action interval."""
    low, high = ACCELERATION_RANGE
    acceleration = float(acceleration)
    if acceleration >= 0.0:
        return float(acceleration / high) if high > 0.0 else 0.0
    return float(acceleration / abs(low)) if low < 0.0 else 0.0


def denormalize_acceleration(action: float) -> float:
    """Map normalized env acceleration action to physical m/s^2."""
    low, high = ACCELERATION_RANGE
    action = float(action)
    if action >= 0.0:
        return float(action * high)
    return float(action * abs(low))

# IDM / MOBIL parameter placeholders by dataset/region.
# These are configuration placeholders so the environment can cleanly select
# region-specific settings now and fill in calibrated values later.
IDM_PARAMETER_PRESETS = {
    "US": {
        "profile": "us",
        "dataset": "US",
        "idm": {
            "desired_speed": 19.6268,
            "time_headway": 1.2408,
            "min_gap": 4.1301,
            "acceleration": 1.6354,
            "comfortable_deceleration": 1.4806,
            "delta": 4.0,
        },
        "mobil": {
            "politeness": 0.0,
            "lane_change_min_acc_gain": 0.1841,
            "lane_change_max_braking_imposed": 0.9366,
            "lane_change_delay": 2.3499,
        },
    },
    "JAPAN": {
        "profile": "japanese",
        "dataset": "JAPAN",
        "idm": {
            "desired_speed": 18.8892,
            "time_headway": 1.1854,
            "min_gap": 5.2877,
            "acceleration": 1.2797,
            "comfortable_deceleration": 0.9570,
            "delta": 4.0,
        },
        "mobil": {
            "politeness": 0.2597,
            "lane_change_min_acc_gain": 0.0491,
            "lane_change_max_braking_imposed": 0.6554,
            "lane_change_delay": 5.4958,
        },
    },
}

SCENE_IDM_PARAMETER_KEY = {
    "us-101": "US",
    "i-80": "US",
    "lankershim": "US",
    "japanese": "JAPAN",
}

# US-101 geometry in dataset-native feet
US101_MAINLINE_LENGTH_FT = 2150.0
US101_LANE_WIDTH_FT = 12.0
US101_SECTION_1_LENGTH_FT = 560.0
US101_SECTION_2_LENGTH_FT = 698.0 + 578.0 + 150.0
US101_MERGE_IN_START_FT = 480.0
US101_MERGE_OUT_END_FT = 1550.0

# Derived US-101 geometry in meters
US101_MAINLINE_LENGTH_M = US101_MAINLINE_LENGTH_FT * METERS_PER_FOOT
US101_LANE_WIDTH_M = US101_LANE_WIDTH_FT * METERS_PER_FOOT
US101_SECTION_ENDS_M = [
    0.0,
    US101_SECTION_1_LENGTH_FT * METERS_PER_FOOT,
    US101_SECTION_2_LENGTH_FT * METERS_PER_FOOT,
    US101_MAINLINE_LENGTH_M,
]
US101_MERGE_IN_START_M = US101_MERGE_IN_START_FT * METERS_PER_FOOT
US101_MERGE_OUT_END_M = US101_MERGE_OUT_END_FT * METERS_PER_FOOT
