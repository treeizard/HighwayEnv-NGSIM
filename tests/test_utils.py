import numpy as np

from highway_env.utils import are_polygons_intersecting, rotated_rectangles_intersect


def test_rotated_rectangles_intersect():
    assert rotated_rectangles_intersect(
        ([12.86076812, 28.60182391], 5.0, 2.0, -0.4675779906495494),
        ([9.67753944, 28.90585412], 5.0, 2.0, -0.3417019364473201),
    )
    assert rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([0, 1], 2, 1, 0))
    assert not rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([0, 2.1], 2, 1, 0))
    assert not rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([1, 1.1], 2, 1, 0))
    assert rotated_rectangles_intersect(([0, 0], 2, 1, np.pi / 4), ([1, 1.1], 2, 1, 0))


def test_polygon_collision_skips_degenerate_edges_without_nan_translation():
    rectangle = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]
    )
    line_with_duplicate_vertices = np.asarray(
        [[-0.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [-0.5, 0.0]]
    )

    intersecting, will_intersect, translation = are_polygons_intersecting(
        rectangle,
        line_with_duplicate_vertices,
        np.zeros(2),
        np.zeros(2),
    )

    assert intersecting
    assert will_intersect
    assert translation is not None
    assert np.all(np.isfinite(translation))


def test_two_zero_area_point_polygons_are_not_collision_surfaces():
    point = np.zeros((5, 2), dtype=float)

    intersecting, will_intersect, translation = are_polygons_intersecting(
        point,
        point,
        np.zeros(2),
        np.zeros(2),
    )

    assert not intersecting
    assert not will_intersect
    assert translation is None
