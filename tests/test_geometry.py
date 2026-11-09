import numpy as np
import pytest

from affordance3d.frames import make_frame, read_intrinsics, read_log_poses
from affordance3d.geometry import backproject, depth_residuals, forward_warp, project, visible_mask

from .conftest import TEST_HW, TEST_K, frame_from_depth, pose


def test_backproject_then_project_recovers_pixels_and_depth() -> None:
    depth = np.full(TEST_HW, 2.0)
    depth[0, 0] = 0.0  # missing depth must be skipped
    frame = frame_from_depth(depth, pose(yaw_deg=20.0, translation=(0.3, -0.1, 0.5)))

    points, pixels = backproject(frame)
    uv, z, inside = project(points, frame)

    assert len(points) == depth.size - 1
    np.testing.assert_allclose(uv, pixels, atol=1e-9)
    np.testing.assert_allclose(z, 2.0, atol=1e-9)
    assert inside.all()


def test_backproject_matches_a_hand_computed_point() -> None:
    depth = np.zeros(TEST_HW)
    depth[23, 63] = 2.0  # row v = 23, column u = 63
    points, pixels = backproject(frame_from_depth(depth))
    # x = (u - cx) z / fx, y = (v - cy) z / fy with TEST_K: fx = fy = 100, cx = 31.5, cy = 23.5
    np.testing.assert_array_equal(pixels, [[63, 23]])
    np.testing.assert_allclose(points, [[0.63, -0.01, 2.0]], atol=1e-12)


def test_backproject_moves_points_with_the_camera_pose() -> None:
    depth = np.full(TEST_HW, 2.0)
    at_origin, _ = backproject(frame_from_depth(depth))
    shifted, _ = backproject(frame_from_depth(depth, pose(translation=(1.0, 2.0, 3.0))))
    np.testing.assert_allclose(shifted - at_origin, np.broadcast_to([1.0, 2.0, 3.0], shifted.shape))


def test_backproject_stride_subsamples_pixels(wall_frame) -> None:
    _, pixels = backproject(wall_frame, stride=4)
    assert len(pixels) == (TEST_HW[0] // 4) * (TEST_HW[1] // 4)
    assert set(np.unique(pixels[:, 0] % 4)) == {0}


def test_visibility_respects_occlusion_and_camera_bounds(wall_frame) -> None:
    points = np.array([
        [0.0, 0.0, 2.0],  # on the wall -> visible
        [0.0, 0.0, 3.0],  # behind the wall -> occluded
        [0.0, 0.0, -1.0],  # behind the camera
        [50.0, 0.0, 2.0],  # outside the image
    ])
    np.testing.assert_array_equal(visible_mask(points, wall_frame, depth_tol=0.05), [True, False, False, False])


def test_depth_residuals_detect_a_wrong_pose() -> None:
    wall = frame_from_depth(np.full(TEST_HW, 2.0))  # wall at z = 2 in world
    moved_forward = pose(translation=(0.0, 0.0, 0.5))  # camera 0.5 m closer: wall at depth 1.5
    correct = frame_from_depth(np.full(TEST_HW, 1.5), moved_forward, index=1)
    wrong = frame_from_depth(np.full(TEST_HW, 1.5), np.eye(4), index=1)

    np.testing.assert_allclose(depth_residuals(wall, correct), 0.0, atol=1e-6)
    np.testing.assert_allclose(depth_residuals(wall, wrong), 0.5, atol=1e-6)


def test_forward_warp_with_identical_cameras_reproduces_the_image() -> None:
    rgb = np.random.default_rng(0).integers(0, 255, (*TEST_HW, 3), dtype=np.uint8)
    frame = frame_from_depth(np.full(TEST_HW, 2.0), rgb=rgb)
    warped, filled = forward_warp(frame, frame)
    assert filled.all()
    np.testing.assert_array_equal(warped, rgb)


def test_forward_warp_keeps_the_nearest_surface() -> None:
    # Near (white) surface on the LEFT: pixels are written in row-major order, so the far
    # (black) right half is written last. Only a working z-buffer keeps white in the overlap.
    depth = np.full(TEST_HW, 2.0)
    depth[:, :32] = 1.0
    rgb = np.zeros((*TEST_HW, 3), np.uint8)
    rgb[:, :32] = 255
    src = frame_from_depth(depth, rgb=rgb)
    # Camera moves left: near points shift 10 px right, far points 5 px right.
    dst = frame_from_depth(np.full(TEST_HW, 2.0), pose(translation=(-0.1, 0.0, 0.0)), index=1)
    warped, filled = forward_warp(src, dst)
    overlap_column = 39  # near source column 29 and far source column 34 both land here
    assert filled[:, overlap_column].all()
    assert (warped[:, overlap_column] == 255).all()


def test_make_frame_rejects_mismatched_rgb_and_depth() -> None:
    with pytest.raises(ValueError, match="differ"):
        make_frame(0, np.zeros((4, 5, 3), np.uint8), np.zeros((4, 6), np.uint16), TEST_K, np.eye(4), 1000.0)


def test_make_frame_converts_millimetres_to_metres() -> None:
    frame = make_frame(0, np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 1500, np.uint16), TEST_K, np.eye(4), 1000.0)
    np.testing.assert_allclose(frame.depth_m, 1.5)


def test_readers_parse_open3d_formats(tmp_path) -> None:
    log = tmp_path / "poses.log"
    log.write_text("0 0 1\n1 0 0 2\n0 1 0 2\n0 0 1 -0.3\n0 0 0 1\n")
    (pose_matrix,) = read_log_poses(log)
    np.testing.assert_allclose(pose_matrix[:3, 3], [2.0, 2.0, -0.3])

    camera = tmp_path / "camera.json"
    camera.write_text('{"intrinsic_matrix": [525, 0, 0, 0, 525, 0, 319.5, 239.5, 1]}')
    np.testing.assert_allclose(read_intrinsics(camera), [[525, 0, 319.5], [0, 525, 239.5], [0, 0, 1]])
