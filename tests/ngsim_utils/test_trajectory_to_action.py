import numpy as np
import pytest

from highway_env.ngsim_utils.core.constants import MAX_ACCEL, MIN_ACCEL
from highway_env.ngsim_utils.expert.trajectory_to_action import PurePursuitTracker


def test_pure_pursuit_tracker_reports_target_lane_id():
    tracker = PurePursuitTracker(
        np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=float),
        np.asarray([5.0, 5.0, 5.0], dtype=float),
        dt=0.1,
        L_forward=4.5,
        ref_lanes=np.asarray([1, 2, 3], dtype=int),
    )

    _steer, _accel, _near, target_idx, target_lane_id = tracker.step(
        np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        speed=5.0,
    )

    assert 0 <= target_idx < 3
    assert target_lane_id == [1, 2, 3][target_idx]


def test_pure_pursuit_tracker_rejects_short_lane_reference():
    with pytest.raises(ValueError, match="ref_lanes must have at least as many entries"):
        PurePursuitTracker(
            np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=float),
            np.asarray([5.0, 5.0, 5.0], dtype=float),
            dt=0.1,
            L_forward=4.5,
            ref_lanes=np.asarray([1, 2], dtype=int),
        )


def test_pure_pursuit_tracker_accepts_column_lane_reference():
    tracker = PurePursuitTracker(
        np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=float),
        np.asarray([5.0, 5.0, 5.0], dtype=float),
        dt=0.1,
        L_forward=4.5,
        ref_lanes=np.asarray([[1], [2], [3]], dtype=int),
    )

    _steer, _accel, _near, target_idx, target_lane_id = tracker.step(
        np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        speed=5.0,
    )

    assert target_lane_id == [1, 2, 3][target_idx]


def test_pure_pursuit_tracker_uses_shared_acceleration_limits_by_default():
    tracker = PurePursuitTracker(
        np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=float),
        np.asarray([100.0, 100.0], dtype=float),
        dt=0.1,
        L_forward=4.5,
        jerk_limit=None,
    )

    _steer, accel, _near, _target_idx, _target_lane_id = tracker.step(
        np.asarray([0.0, 0.0], dtype=float),
        heading=0.0,
        speed=0.0,
    )

    assert tracker.a_min == pytest.approx(MIN_ACCEL)
    assert tracker.a_max == pytest.approx(MAX_ACCEL)
    assert accel == pytest.approx(MAX_ACCEL)
