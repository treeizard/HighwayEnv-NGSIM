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

import numpy as np

# Dataset/unit conversion
FEET_PER_METER = 3.281
METERS_PER_FOOT = 1.0 / FEET_PER_METER

# Vehicle/control limits
MAX_ACCEL = 5.0
MAX_STEER = np.pi / 4

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
