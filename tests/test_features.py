import numpy as np
import pytest

from affordance3d.features import PatchFeatures, fit_pca_coloring, l2_normalize, patch_rc, tokens_to_grid


def test_patch_rc_maps_pixels_to_the_patch_that_covers_them() -> None:
    uv = np.array([[0, 0], [19, 0], [20, 0], [639, 479], [320, 240]])
    rc = patch_rc(uv, image_hw=(480, 640), grid_hw=(24, 32))  # 20x20 px per patch
    np.testing.assert_array_equal(rc, [[0, 0], [0, 0], [0, 1], [23, 31], [12, 16]])


def test_tokens_to_grid_drops_prefix_tokens_and_keeps_row_major_order() -> None:
    patches = np.arange(6 * 4, dtype=np.float32).reshape(1, 6, 4)
    tokens = np.concatenate([np.full((1, 1, 4), -1.0, np.float32), patches], axis=1)  # CLS first
    grid = tokens_to_grid(tokens, grid_hw=(2, 3))
    assert grid.shape == (2, 3, 4)
    np.testing.assert_array_equal(grid[1, 0], patches[0, 3])


def test_tokens_to_grid_rejects_too_few_tokens() -> None:
    with pytest.raises(ValueError, match="tokens"):
        tokens_to_grid(np.zeros((1, 5, 4)), grid_hw=(2, 3))


def test_patch_features_return_the_feature_of_each_pixel() -> None:
    grid = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    features = PatchFeatures(grid, image_hw=(4, 4))
    looked_up = features.at_pixels(np.array([[3, 0], [0, 3]]))  # (u, v): top-right, bottom-left
    np.testing.assert_array_equal(looked_up, [grid[0, 1], grid[1, 0]])


def test_l2_normalize_returns_unit_rows_and_tolerates_zeros() -> None:
    rows = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(rows, [[0.6, 0.8], [0.0, 0.0]])


def test_pca_coloring_is_bounded_and_separates_distinct_features() -> None:
    rng = np.random.default_rng(0)
    group_a = rng.normal(0.0, 0.05, (200, 16)) + np.eye(16)[0]
    group_b = rng.normal(0.0, 0.05, (200, 16)) + np.eye(16)[1]
    coloring = fit_pca_coloring(np.vstack([group_a, group_b]))

    colours_a, colours_b = coloring.to_rgb(group_a), coloring.to_rgb(group_b)
    assert colours_a.shape == (200, 3)
    assert colours_a.min() >= 0.0 and colours_b.max() <= 1.0
    assert abs(colours_a[:, 0].mean() - colours_b[:, 0].mean()) > 0.5
