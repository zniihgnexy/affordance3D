import numpy as np
import pytest

from affordance3d.mem import scan_lifting as L
from tests.conftest import TEST_HW, frame_from_depth

# TEST_K: fx = fy = 100, cx = 31.5, cy = 23.5; the camera sits at the origin looking along +z.
A = (0.01, 0.01, 2.0)      # pixel (row 24, column 32), on the wall 2 m away
B = (0.11, 0.01, 2.0)      # pixel (24, 37)
NEAR = (0.012, 0.012, 2.4)  # pixel (24, 32), 0.4 m behind the wall: within 25% of 2 m, kept as in Fun3DU
FAR = (0.013, 0.013, 2.6)   # pixel (24, 32), 0.6 m behind: dropped
BEHIND = (0.01, 0.01, -2.0)


def _wall(depth: float = 2.0):
    return frame_from_depth(np.full(TEST_HW, depth))


def test_lift_mask_keeps_points_in_the_mask_whose_depth_agrees_within_a_quarter() -> None:
    mask = np.zeros(TEST_HW, dtype=bool)
    mask[24, 32] = True

    kept = L.lift_mask(np.array([A, B, NEAR, FAR, BEHIND]), _wall(), mask)

    assert kept.tolist() == [True, False, True, False, False]


def test_lift_mask_needs_depth_and_a_mask_of_the_frame_size() -> None:
    depth = np.full(TEST_HW, 2.0)
    depth[24, 32] = 0.0  # no depth here: the point cannot be kept
    mask = np.ones(TEST_HW, dtype=bool)

    assert L.lift_mask(np.array([A, B]), frame_from_depth(depth), mask).tolist() == [False, True]
    with pytest.raises(ValueError):
        L.lift_mask(np.array([A]), _wall(), np.ones((2, 2), dtype=bool))


def test_upsample_makes_every_pixel_a_block() -> None:
    assert L.upsample(np.array([[1, 0], [0, 1]]), 2).astype(int).tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]


def test_region_mask_marks_where_the_points_are_seen_and_most_visible_picks_that_frame() -> None:
    region = L.region_mask(np.array([A, B, FAR]), _wall())

    assert sorted(zip(*np.nonzero(region))) == [(24, 32), (24, 37)]
    assert L.most_visible(np.array([A, B]), [_wall(5.0), _wall(2.0)]) == 1
    assert L.most_visible(np.array([BEHIND]), [_wall(2.0)]) is None
