import numpy as np

from affordance3d.frames import Frame
from affordance3d.mem.labels import part_label_map, projected_axis


def _camera_looking_along_x(depth: np.ndarray) -> Frame:
    """Camera at the origin looking along world +x (world z up), 20 x 20 pixels."""
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    intrinsics = np.array([[20.0, 0, 9.5], [0, 20.0, 9.5], [0, 0, 1]])
    return Frame(0, np.zeros((20, 20, 3), np.uint8), depth, intrinsics, cam_to_world)


def test_part_labels_mark_visible_parts_and_skip_occluded_ones() -> None:
    depth = np.full((20, 20), 2.0, dtype=np.float32)  # a wall 2 m ahead
    frame = _camera_looking_along_x(depth)
    on_wall = np.array([[2.0, 0.0, 0.0], [2.0, 0.01, 0.0]])
    behind_wall = np.array([[3.0, -0.5, 0.0]])

    labels = part_label_map([on_wall, behind_wall], frame, depth_tol=0.05)

    assert labels.shape == (20, 20)
    assert set(np.unique(labels)) == {-1, 0}
    assert labels[9:11, 9:11].max() == 0


def test_projected_axis_starts_at_the_origin_pixel_and_points_along_the_motion() -> None:
    frame = _camera_looking_along_x(np.full((20, 20), 2.0, dtype=np.float32))

    start, end = projected_axis(np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), frame, length_m=0.5)

    assert np.allclose(start, [9.5, 9.5])
    assert end[1] < start[1]  # world up is image up
    assert np.isclose(end[0], start[0])
