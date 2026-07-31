from __future__ import annotations

import numpy as np

from highway_env.ngsim_utils.expert.ngsim_expert_mixin import NGSimExpertMixin


class _RecordingTracker:
    def __init__(self):
        self.received = None

    def step(self, position, heading, speed):
        self.received = (
            np.asarray(position, dtype=float).copy(),
            float(heading),
            float(speed),
        )
        return 0.0, 0.0, 0, 0, -1


class _ExpertHarness(NGSimExpertMixin):
    control_mode = "continuous"

    def __init__(self, *, collection_mode):
        self.scene_dataset_collection_mode = bool(collection_mode)
        self.tracker = _RecordingTracker()
        self.state = {
            "tracker": self.tracker,
            "actions_policy": [],
            "tracker_dbg": [],
        }

    def _expert_state_for_vehicle(self, vehicle):
        return self.state


class _Vehicle:
    position = np.asarray(
        [1_000_000.0123456789, -2_000_000.0987654321],
        dtype=np.float64,
    )
    heading = np.float64(1.234567890123)
    speed = np.float64(12.34567890123)


def test_scene_collection_tracker_uses_float32_archival_state():
    harness = _ExpertHarness(collection_mode=True)
    vehicle = _Vehicle()
    harness._resolve_expert_action(vehicle=vehicle)
    position, heading, speed = harness.tracker.received
    np.testing.assert_array_equal(
        position,
        np.asarray(vehicle.position, dtype=np.float32).astype(float),
    )
    assert heading == float(np.float32(vehicle.heading))
    assert speed == float(np.float32(vehicle.speed))


def test_noncollection_tracker_keeps_runtime_precision():
    harness = _ExpertHarness(collection_mode=False)
    vehicle = _Vehicle()
    harness._resolve_expert_action(vehicle=vehicle)
    position, heading, speed = harness.tracker.received
    np.testing.assert_array_equal(position, vehicle.position)
    assert heading == float(vehicle.heading)
    assert speed == float(vehicle.speed)
