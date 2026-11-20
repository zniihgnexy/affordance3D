"""3D Gaussian map of one video, the lower layer of the scene memory chosen on 2026-11-20 (week-2 notebook 7.11):
initial Gaussians from posed LiDAR depth, the train/test split of the frames, and the image measures. NumPy and torch
only; the fitting loop is in scripts/fit_gaussians.py and needs gsplat on a CUDA device (through the GPU queue).

Cameras follow affordance3d.geometry: OpenCV pinhole, the camera looks along +z, `Frame.cam_to_world` maps camera to
world coordinates. gsplat takes the inverse (world to camera) as its view matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from affordance3d.frames import Frame
from affordance3d.geometry import backproject

SH_C0 = 0.28209479177387814  # zeroth spherical-harmonic basis value: colour = SH_C0 * sh0 + 0.5
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5


def split_frames(n_frames: int, test_every: int) -> tuple[list[int], list[int]]:
    """Every `test_every`-th frame (starting with the 4th of each block) is held out for testing; the rest train."""
    if n_frames < 2 or test_every < 2:
        raise ValueError("need at least 2 frames and test_every >= 2")
    test = [i for i in range(n_frames) if i % test_every == test_every // 2]
    held = set(test)
    return [i for i in range(n_frames) if i not in held], test


def coloured_points(frames: Sequence[Frame], pixel_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) and colours in [0, 1] (N, 3) of every `pixel_stride`-th pixel with depth, over all frames."""
    points, colours = [], []
    for frame in frames:
        xyz, uv = backproject(frame, pixel_stride)
        points.append(xyz)
        colours.append(frame.rgb[uv[:, 1], uv[:, 0]].astype(np.float64) / 255.0)
    if not points or not sum(len(p) for p in points):
        raise ValueError("no pixel with depth in the given frames")
    return np.concatenate(points), np.concatenate(colours)


def voxel_mean(points: np.ndarray, colours: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    """One point per occupied voxel: the mean position and mean colour of the points that fall in it."""
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    mean_points = np.zeros((len(counts), 3))
    mean_colours = np.zeros((len(counts), 3))
    np.add.at(mean_points, inverse, points)
    np.add.at(mean_colours, inverse, colours)
    return mean_points / counts[:, None], mean_colours / counts[:, None]


def initial_gaussians(points: np.ndarray, colours: np.ndarray, voxel_m: float, opacity: float,
                      sh_degree: int) -> dict[str, torch.Tensor]:
    """Isotropic Gaussians at the points: radius half a voxel, identity rotation, the given opacity, colour as SH."""
    n = len(points)
    return {
        "means": torch.tensor(points, dtype=torch.float32),
        "scales": torch.full((n, 3), float(np.log(voxel_m / 2.0))),  # stored as log
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "opacities": torch.full((n,), float(np.log(opacity / (1.0 - opacity)))),  # stored as logit
        "sh0": torch.tensor((colours - 0.5) / SH_C0, dtype=torch.float32)[:, None, :],
        "shN": torch.zeros((n, (sh_degree + 1) ** 2 - 1, 3)),
    }


def scene_scale(frames: Sequence[Frame]) -> float:
    """1.1 times the largest distance of a camera centre from their mean (gsplat's simple trainer does the same)."""
    centres = np.stack([f.cam_to_world[:3, 3] for f in frames])
    return 1.1 * float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max())


def _window(channels: int, device: torch.device) -> torch.Tensor:
    x = torch.arange(SSIM_WINDOW, dtype=torch.float32, device=device) - SSIM_WINDOW // 2
    g = torch.exp(-(x**2) / (2 * SSIM_SIGMA**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(channels, 1, SSIM_WINDOW, SSIM_WINDOW).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of two image batches (B, C, H, W) in [0, 1], Gaussian window 11 / 1.5 (the usual 3DGS setting)."""
    channels = a.shape[1]
    w = _window(channels, a.device)
    blur = lambda x: F.conv2d(x, w, padding=SSIM_WINDOW // 2, groups=channels)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))).mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB of two images in [0, 1]."""
    mse = float(torch.mean((a - b) ** 2))
    return float("inf") if mse == 0 else -10.0 * float(np.log10(mse))


def depth_error(rendered: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    """Absolute depth differences on pixels with a measured depth (measured 0 = missing)."""
    valid = measured > 0
    return (rendered[valid] - measured[valid]).abs()
