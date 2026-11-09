import cv2
import numpy as np

from affordance3d.e0_checks import (
    epipolar_inliers,
    match_features,
    nearest_distances,
    reproject_matches,
    sift_features,
    sift_matches,
    visible_point_counts,
)
from affordance3d.frames import Frame
from affordance3d.geometry import backproject_pixels, forward_warp

from .conftest import TEST_HW, frame_from_depth, pose

WIDE_K = np.array([[300.0, 0.0, 159.5], [0.0, 300.0, 119.5], [0.0, 0.0, 1.0]])


def textured_wall(cam_to_world: np.ndarray, index: int = 0) -> Frame:
    blocks = np.random.default_rng(3).integers(0, 255, (30, 40, 3), dtype=np.uint8)
    rgb = cv2.resize(blocks, (320, 240), interpolation=cv2.INTER_NEAREST)
    return Frame(index, rgb, np.full((240, 320), 2.0, np.float32), WIDE_K, cam_to_world)


def test_backproject_pixels_uses_subpixel_rays_and_skips_missing_depth() -> None:
    depth = np.full(TEST_HW, 2.0)
    depth[0, 0] = 0.0
    frame = frame_from_depth(depth)
    points, valid = backproject_pixels(frame, np.array([[31.5, 23.5], [0.2, 0.1]]))
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_allclose(points[0], [0.0, 0.0, 2.0], atol=1e-12)  # the principal point


def test_backproject_pixels_reads_depth_at_the_nearest_pixel_and_rejects_outside() -> None:
    height, width = TEST_HW
    depth = 1.0 + np.arange(height * width, dtype=np.float64).reshape(TEST_HW) / 1000.0  # unique per pixel
    frame = frame_from_depth(depth)
    uv = np.array([[10.4, 5.6], [-0.6, 0.0], [width - 0.4, 0.0]])

    points, valid = backproject_pixels(frame, uv)

    np.testing.assert_array_equal(valid, [True, False, False])
    z = depth[6, 10]  # rint(5.6) = row 6, rint(10.4) = column 10
    expected = np.array([(10.4 - 31.5) / 100.0, (5.6 - 23.5) / 100.0, 1.0]) * z  # exact sub-pixel ray
    np.testing.assert_allclose(points[0], expected, atol=1e-6)


def test_matches_reproject_with_small_error_for_correct_poses() -> None:
    src = textured_wall(np.eye(4))
    dst_camera = pose(translation=(0.15, 0.0, 0.0))
    rendered, _ = forward_warp(src, Frame(1, src.rgb, src.depth_m, WIDE_K, dst_camera))
    dst = Frame(1, rendered, src.depth_m, WIDE_K, dst_camera)

    uv_src, uv_dst = sift_matches(src.rgb, dst.rgb)
    good = reproject_matches(src, dst, uv_src, uv_dst)
    wrong = reproject_matches(src, Frame(1, rendered, src.depth_m, WIDE_K, np.eye(4)), uv_src, uv_dst)

    assert len(good.errors_px) >= 20
    assert np.median(good.errors_px) < 1.5
    assert np.median(wrong.errors_px) > 15.0  # 0.15 m at 2 m depth is ~22 px with fx = 300


def two_depth_scene() -> tuple[Frame, Frame]:
    """Textured wall with a closer box in the middle, seen from two cameras (non-planar)."""
    src = textured_wall(np.eye(4))
    depth = src.depth_m.copy()
    depth[60:180, 100:220] = 1.4
    src = Frame(0, src.rgb, depth, WIDE_K, np.eye(4))
    dst_camera = pose(yaw_deg=3.0, translation=(0.15, 0.02, 0.0))
    rendered, _ = forward_warp(src, Frame(1, src.rgb, depth, WIDE_K, dst_camera))
    return src, Frame(1, rendered, depth, WIDE_K, dst_camera)


def test_epipolar_check_rejects_impossible_matches_using_images_only() -> None:
    src, dst = two_depth_scene()
    uv_src, uv_dst = sift_matches(src.rgb, dst.rgb)
    rng = np.random.default_rng(0)
    fake_src = rng.uniform([0, 0], [319, 239], (30, 2))
    fake_dst = rng.uniform([0, 0], [319, 239], (30, 2))

    keep = epipolar_inliers(np.vstack([uv_src, fake_src]), np.vstack([uv_dst, fake_dst]), threshold_px=1.0)

    assert keep[: len(uv_src)].mean() > 0.9  # true correspondences survive
    assert keep[len(uv_src):].mean() < 0.2  # random pairings are rejected


def test_epipolar_check_needs_enough_matches() -> None:
    few = np.zeros((5, 2))
    assert not epipolar_inliers(few, few, threshold_px=1.0).any()


def test_features_once_then_match_equals_direct_matching() -> None:
    src, dst = two_depth_scene()
    direct = sift_matches(src.rgb, dst.rgb)
    via_features = match_features(sift_features(src.rgb), sift_features(dst.rgb))
    np.testing.assert_allclose(via_features[0], direct[0])
    np.testing.assert_allclose(via_features[1], direct[1])


def test_nearest_distances_to_a_reference_cloud() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(nearest_distances(np.array([[0.1, 0.0, 0.0], [1.0, 0.5, 0.0]]), reference), [0.1, 0.5])


def test_visible_point_counts_per_element_and_frame(wall_frame) -> None:
    on_wall = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]])
    behind_wall = np.array([[0.0, 0.0, 3.0]])
    counts = visible_point_counts([on_wall, behind_wall], [wall_frame], depth_tol=0.05)
    np.testing.assert_array_equal(counts, [[2], [0]])
