import numpy as np
import pytest

from affordance3d.geometry import backproject, project, rotate_frame_cw, subsample_frame

from .conftest import TEST_HW, frame_from_depth, pose


def textured_frame():
    rng = np.random.default_rng(1)
    depth = rng.uniform(1.0, 3.0, TEST_HW)
    depth[5, 7] = 0.0  # one missing depth value
    rgb = rng.integers(0, 255, (*TEST_HW, 3), dtype=np.uint8)
    return frame_from_depth(depth, pose(yaw_deg=30.0, translation=(0.2, -0.4, 1.0)), rgb=rgb)


def sorted_rows(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(points.T[::-1])]


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3, 5])
def test_rotation_keeps_every_world_point(quarter_turns: int) -> None:
    frame = textured_frame()
    rotated = rotate_frame_cw(frame, quarter_turns)
    original_points, _ = backproject(frame)
    rotated_points, _ = backproject(rotated)
    np.testing.assert_allclose(sorted_rows(rotated_points), sorted_rows(original_points), atol=1e-9)


def test_one_clockwise_turn_moves_the_left_edge_to_the_top() -> None:
    frame = textured_frame()
    rotated = rotate_frame_cw(frame, 1)
    height, width = TEST_HW
    assert rotated.image_hw == (width, height)
    # Pixel (u, v) moves to (H-1-v, u): the old left column (u = 0) becomes the new top row.
    np.testing.assert_array_equal(rotated.rgb[0, height - 1 - 10], frame.rgb[10, 0])
    assert rotated.depth_m[0, height - 1 - 10] == frame.depth_m[10, 0]


def test_rotated_frame_projects_world_points_onto_the_moved_pixels() -> None:
    frame = textured_frame()
    rotated = rotate_frame_cw(frame, 1)
    points, pixels = backproject(frame)
    uv, _, inside = project(points, rotated)
    height = TEST_HW[0]
    expected = np.stack([height - 1 - pixels[:, 1], pixels[:, 0]], axis=1)
    assert inside.all()
    np.testing.assert_allclose(uv, expected, atol=1e-9)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_subsampling_matches_strided_backprojection(step: int) -> None:
    frame = textured_frame()
    small = subsample_frame(frame, step)
    small_points, _ = backproject(small)
    strided_points, _ = backproject(frame, stride=step)
    np.testing.assert_allclose(small_points, strided_points, atol=1e-9)
    assert small.rgb.shape[:2] == small.depth_m.shape


def test_subsampling_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="step"):
        subsample_frame(textured_frame(), 0)
