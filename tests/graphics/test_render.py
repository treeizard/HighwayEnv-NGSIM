import gymnasium as gym
import highway_env
import numpy as np
import pygame
import pytest
from highway_env.road.graphics import LaneGraphics
from highway_env.road.lane import LineType, PolyLaneFixedWidth, StraightLane

gym.register_envs(highway_env)


class _IdentitySurface:
    WHITE = (255, 255, 255)
    origin = np.asarray([0.0, 0.0])
    scaling = 1.0

    @staticmethod
    def get_width():
        return 100

    @staticmethod
    def get_height():
        return 100

    @staticmethod
    def pix(length):
        return max(int(round(float(length))), 1)

    @staticmethod
    def vec2pix(position):
        return tuple(np.asarray(position, dtype=float))


def test_curved_continuous_line_samples_the_lane_boundary(monkeypatch):
    lane = PolyLaneFixedWidth(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 11.1)],
        width=4.0,
        line_types=[LineType.CONTINUOUS_LINE, LineType.NONE],
    )
    captured = {}

    def capture_lines(_surface, _color, closed, points, width):
        captured.update(closed=closed, points=list(points), width=width)

    monkeypatch.setattr(pygame.draw, "lines", capture_lines)
    LaneGraphics.continuous_line(
        lane,
        _IdentitySurface(),
        stripes_count=10,
        longitudinal=0.0,
        side=0,
    )

    expected_midpoint = tuple(lane.position(lane.length / 2.0, -lane.width / 2.0))
    points = captured["points"]
    assert captured["closed"] is False
    assert len(points) >= int(np.ceil(lane.length)) + 1
    assert any(np.allclose(point, expected_midpoint) for point in points)
    assert not np.allclose(expected_midpoint, 0.5 * (np.asarray(points[0]) + points[-1]))


def test_straight_continuous_line_retains_single_segment(monkeypatch):
    lane = StraightLane(
        [0.0, 0.0],
        [20.0, 0.0],
        width=4.0,
        line_types=[LineType.CONTINUOUS_LINE, LineType.NONE],
    )
    captured = []

    def capture_line(_surface, _color, start, end, width):
        captured.append((start, end, width))

    monkeypatch.setattr(pygame.draw, "line", capture_line)
    LaneGraphics.continuous_line(
        lane,
        _IdentitySurface(),
        stripes_count=10,
        longitudinal=0.0,
        side=0,
    )

    assert len(captured) == 1
    start, end, _width = captured[0]
    np.testing.assert_allclose(start, lane.position(0.0, -lane.width / 2.0))
    np.testing.assert_allclose(end, lane.position(lane.length, -lane.width / 2.0))


def test_marking_profile_renders_solid_approach_then_open_striped_merge(monkeypatch):
    lane = PolyLaneFixedWidth(
        [(0.0, 0.0), (20.5, 0.0)],
        width=4.0,
        line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
        marking_profile=[
            {
                "start_s_m": 0.0,
                "end_s_m": 10.0,
                "line_types": [LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
            },
            {
                "start_s_m": 10.0,
                "end_s_m": 20.5,
                "line_types": [LineType.NONE, LineType.STRIPED],
            },
        ],
    )
    solid_lines = []
    stripes = []

    def capture_lines(_surface, _color, _closed, points, _width):
        solid_lines.append(np.asarray(points, dtype=float))

    def capture_line(_surface, _color, start, end, _width):
        stripes.append((np.asarray(start, dtype=float), np.asarray(end, dtype=float)))

    monkeypatch.setattr(pygame.draw, "lines", capture_lines)
    monkeypatch.setattr(pygame.draw, "line", capture_line)

    LaneGraphics.display(lane, _IdentitySurface())

    assert len(solid_lines) == 2
    for side, points in enumerate(solid_lines):
        np.testing.assert_allclose(points[[0, -1], 0], [0.0, 10.0])
        np.testing.assert_allclose(points[:, 1], -2.0 if side == 0 else 2.0)
    assert stripes
    for start, end in stripes:
        assert 10.0 <= start[0] < end[0] <= lane.length
        np.testing.assert_allclose([start[1], end[1]], [2.0, 2.0])


def test_missing_marking_profile_uses_base_line_types(monkeypatch):
    lane = PolyLaneFixedWidth(
        [(0.0, 0.0), (20.5, 0.0)],
        width=4.0,
        line_types=[LineType.CONTINUOUS_LINE, LineType.NONE],
    )
    solid_lines = []

    def capture_lines(_surface, _color, _closed, points, _width):
        solid_lines.append(np.asarray(points, dtype=float))

    monkeypatch.setattr(pygame.draw, "lines", capture_lines)

    LaneGraphics.display(lane, _IdentitySurface())

    assert lane.marking_profile is None
    assert len(solid_lines) == 1
    np.testing.assert_allclose(solid_lines[0][[0, -1], 0], [0.0, lane.length])


@pytest.mark.parametrize("env_spec", ["highway-v0", "merge-v0"])
def test_render(env_spec):
    env = gym.make(env_spec, render_mode="rgb_array").unwrapped
    env.config.update({"offscreen_rendering": True})
    env.reset()
    img = env.render()
    env.close()
    assert isinstance(img, np.ndarray)
    assert img.shape == (
        env.config["screen_height"],
        env.config["screen_width"],
        3,
    )  # (H,W,C)


@pytest.mark.parametrize("env_spec", ["highway-v0", "merge-v0"])
def test_obs_grayscale(env_spec, stack_size=4):
    env = gym.make(env_spec).unwrapped
    env.config.update(
        {
            "offscreen_rendering": True,
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (
                    env.config["screen_width"],
                    env.config["screen_height"],
                ),
                "stack_size": stack_size,
                "weights": [0.2989, 0.5870, 0.1140],
            },
        }
    )
    obs, info = env.reset()
    env.close()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (
        stack_size,
        env.config["screen_width"],
        env.config["screen_height"],
    )
