"""affordance3d.mem.gaussian_map: frame split, initial Gaussians from posed depth, image measures (CPU, no gsplat)."""
import numpy as np
import pytest
import torch

from affordance3d.frames import Frame
from affordance3d.mem import gaussian_map as G


def frame(depth: float = 2.0, offset: float = 0.0) -> Frame:
    k = np.array([[10.0, 0, 2.0], [0, 10.0, 2.0], [0, 0, 1]])
    pose = np.eye(4)
    pose[0, 3] = offset
    rgb = np.full((5, 5, 3), 255, dtype=np.uint8)
    d = np.full((5, 5), depth, dtype=np.float32)
    d[0, 0] = 0.0  # missing depth is skipped
    return Frame(0, rgb, d, k, pose)


def test_split_holds_out_one_frame_in_each_block():
    train, test = G.split_frames(17, 8)
    assert test == [4, 12] and len(train) == 15 and not set(train) & set(test)


def test_split_rejects_too_few_frames():
    with pytest.raises(ValueError):
        G.split_frames(1, 8)


def test_coloured_points_lift_the_principal_pixel_onto_the_axis():
    points, colours = G.coloured_points([frame(2.0, offset=1.0)], pixel_stride=1)
    assert len(points) == 24  # 25 pixels, one without depth
    centre = points[np.argmin(np.abs(points[:, 0] - 1.0) + np.abs(points[:, 1]))]
    assert np.allclose(centre, [1.0, 0.0, 2.0]) and np.allclose(colours, 1.0)


def test_voxel_mean_merges_points_in_one_voxel():
    points = np.array([[0.001, 0.001, 0.001], [0.003, 0.003, 0.003], [0.5, 0.5, 0.5]])
    colours = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.2, 0.2]])
    merged, mean_colours = G.voxel_mean(points, colours, 0.01)
    assert len(merged) == 2
    first = np.argmin(merged[:, 0])
    assert np.allclose(merged[first], 0.002) and np.allclose(mean_colours[first], 0.5)


def test_initial_gaussians_encode_colour_and_opacity():
    g = G.initial_gaussians(np.zeros((2, 3)), np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]), 0.01, 0.1, sh_degree=3)
    assert g["shN"].shape == (2, 15, 3)
    assert torch.allclose(G.SH_C0 * g["sh0"][:, 0] + 0.5, torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(torch.sigmoid(g["opacities"]), torch.full((2,), 0.1))
    assert torch.allclose(torch.exp(g["scales"]), torch.full((2, 3), 0.005))


def test_scene_scale_is_the_largest_camera_distance_times_1_1():
    assert G.scene_scale([frame(offset=-1.0), frame(offset=1.0)]) == pytest.approx(1.1)


def test_image_measures():
    a = torch.rand(1, 3, 16, 16)
    assert float(G.ssim(a, a)) == pytest.approx(1.0, abs=1e-5)
    assert G.psnr(torch.zeros(4, 4), torch.full((4, 4), 0.1)) == pytest.approx(20.0)
    errors = G.depth_error(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 0.0, 2.0]))
    assert torch.allclose(errors, torch.tensor([0.5, 1.0]))
