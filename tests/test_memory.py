import numpy as np
import pytest

from affordance3d.features import PatchFeatures
from affordance3d.memory import Observation, contains, cosine_scores, coverage, fuse, observe

from .conftest import TEST_HW, frame_from_depth

VOXEL = 0.05


def observation(frame_index: int, xyz: list, features: list, rgb: list | None = None) -> Observation:
    points = np.array(xyz, dtype=np.float64)
    colours = np.zeros_like(points) if rgb is None else np.array(rgb, dtype=np.float32)
    return Observation(frame_index, points, colours, np.array(features, dtype=np.float32))


def test_fuse_merges_one_voxel_seen_in_two_frames() -> None:
    first = observation(0, [[0.01, 0.01, 0.01]], [[1.0, 0.0]], rgb=[[1.0, 0.0, 0.0]])
    second = observation(1, [[0.02, 0.02, 0.02]], [[0.0, 1.0]], rgb=[[0.0, 0.0, 1.0]])
    memory = fuse([first, second], VOXEL)

    assert memory.n_voxels == 1
    np.testing.assert_array_equal(memory.seen_by, [[True, True]])
    np.testing.assert_allclose(memory.rgb, [[0.5, 0.0, 0.5]])
    np.testing.assert_allclose(memory.features, [[2**-0.5, 2**-0.5]], atol=1e-6)
    np.testing.assert_array_equal(memory.first_seen, [0])
    np.testing.assert_array_equal(memory.last_seen, [1])


def test_fuse_keeps_separate_voxels_with_their_source_frames() -> None:
    memory = fuse([
        observation(3, [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]),
        observation(7, [[-1.0, 0.0, 0.0]], [[1.0, 0.0]]),
    ], VOXEL)

    assert memory.n_voxels == 2
    np.testing.assert_array_equal(memory.frame_ids, [3, 7])
    near_origin = np.argmin(np.linalg.norm(memory.xyz, axis=1))
    np.testing.assert_array_equal(memory.seen_by[near_origin], [True, False])
    np.testing.assert_array_equal(memory.seen_by[1 - near_origin], [True, True])


def test_fuse_requires_observations() -> None:
    with pytest.raises(ValueError, match="observation"):
        fuse([], VOXEL)


def test_contains_reports_whether_points_fall_in_remembered_voxels() -> None:
    memory = fuse([observation(0, [[0.0, 0.0, 0.0]], [[1.0, 0.0]])], VOXEL)
    np.testing.assert_array_equal(contains(memory, np.array([[0.01, 0.01, 0.01], [1.0, 1.0, 1.0]])), [True, False])


def test_cosine_scores_rank_the_matching_feature_first() -> None:
    memory = fuse([observation(0, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])], VOXEL)
    scores = cosine_scores(memory, np.array([0.0, 5.0]))
    assert scores[np.argmax(scores)] == pytest.approx(1.0)
    assert memory.xyz[np.argmax(scores), 0] == pytest.approx(1.0)


def test_observe_attaches_each_pixels_patch_feature_and_colour() -> None:
    height, width = TEST_HW
    rgb = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    frame = frame_from_depth(np.full(TEST_HW, 2.0), rgb=rgb)
    grid = np.eye(4, dtype=np.float32).reshape(2, 2, 4)  # one distinct feature per patch
    obs = observe(frame, PatchFeatures(grid, TEST_HW), stride=1)

    assert len(obs.xyz) == height * width
    top_right = width - 1  # samples are row-major: pixel (u=63, v=0); a row/column swap gives grid[1, 0]
    np.testing.assert_array_equal(obs.features[top_right], grid[0, 1])
    np.testing.assert_allclose(obs.rgb[top_right], rgb[0, width - 1] / 255.0)
    np.testing.assert_allclose(obs.xyz[:, 2], 2.0)


def test_coverage_separates_visible_out_of_view_occluded_and_new() -> None:
    wall = frame_from_depth(np.full(TEST_HW, 2.0))
    remembered = [[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, 0.0, 3.0]]  # on wall, behind camera, behind wall
    memory = fuse([observation(0, remembered, [[1.0], [1.0], [1.0]])], VOXEL)
    current = observation(1, [[0.0, 0.0, 2.0], [0.3, 0.0, 2.0]], [[1.0], [1.0]])

    result = coverage(memory, wall, current, depth_tol=0.05)

    order = np.argsort(memory.xyz[:, 2])  # behind camera, on wall, behind wall
    np.testing.assert_array_equal(result.visible_now[order], [False, True, False])
    np.testing.assert_array_equal(result.out_of_view[order], [True, False, False])
    np.testing.assert_array_equal(result.occluded[order], [False, False, True])
    np.testing.assert_array_equal(result.history_only, ~result.visible_now)
    np.testing.assert_array_equal(result.new_in_current, [False, True])
