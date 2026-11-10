import numpy as np
import pytest

from affordance3d.frames import Frame
from affordance3d.geometry import project, rotate_frame_cw
from affordance3d.mem.inputs import lidar_to_image, nearest_index, upright_camera_pose


def test_lidar_depth_is_rotated_upright_masked_by_confidence_and_resized_to_the_image() -> None:
    depth = np.full((2, 4), 1.5, dtype=np.float32)  # landscape LiDAR frame, 2 rows x 4 columns
    depth[0, 3] = 3.0  # top-right pixel
    confidence = np.full((2, 4), 2, dtype=np.uint8)
    confidence[1, 0] = 1  # low confidence: dropped

    image_depth = lidar_to_image(depth, confidence, cw_turns=1, image_hw=(8, 4), min_confidence=2)

    # one clockwise turn: (row r, col c) -> (row c, col H-1-r); top-right becomes bottom-right
    assert image_depth.shape == (8, 4)
    assert np.all(image_depth[6:8, 2:4] == 3.0)
    assert np.all(image_depth[0:2, 0:2] == 0.0)  # the low-confidence pixel, now top-left
    assert np.all(image_depth[0:2, 2:4] == 1.5)


def test_nearest_index_rejects_matches_beyond_the_allowed_gap() -> None:
    times = np.array([10.0, 10.1, 10.2])

    assert nearest_index(times, 10.13, max_gap=0.05) == 1
    with pytest.raises(ValueError, match="no sample within"):
        nearest_index(times, 10.4, max_gap=0.05)


def test_upright_camera_pose_matches_rotating_the_whole_frame() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [0.2, -0.1, 0.5]
    frame = Frame(0, np.zeros((6, 8, 3), np.uint8), np.ones((6, 8), np.float32),
                  np.array([[5.0, 0, 3.5], [0, 5.0, 2.5], [0, 0, 1]]), pose)
    rotated = rotate_frame_cw(frame, 1)
    point = np.array([[0.4, 0.1, 2.0]])

    upright = upright_camera_pose(pose, cw_turns=1)

    assert np.allclose(upright, rotated.cam_to_world)
    uv, _, _ = project(point, Frame(0, rotated.rgb, rotated.depth_m, rotated.intrinsics, upright))
    uv_original, _, _ = project(point, frame)
    assert np.allclose(uv[0], [(6 - 1) - uv_original[0, 1], uv_original[0, 0]])
