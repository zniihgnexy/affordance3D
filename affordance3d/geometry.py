"""Pinhole projection between image pixels and world points (E0 alignment checks).

Conventions: pixel (u, v) = (column, row) with integer pixel centres; camera looks
along +z; `Frame.cam_to_world` maps camera coordinates to world coordinates:
    x_world = R (depth * K^-1 [u, v, 1]^T) + t
"""

from __future__ import annotations

import numpy as np

from affordance3d.frames import Frame

MIN_DEPTH_M = 1e-6


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N, 3) points."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def backproject(frame: Frame, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Lift every `stride`-th pixel with valid depth to world points.

    Returns world points (N, 3) and the integer pixels (N, 2) as (u, v) they came from.
    """
    height, width = frame.image_hw
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    depth = frame.depth_m[rows, cols]
    valid = depth > 0
    u, v, z = cols[valid], rows[valid], depth[valid].astype(np.float64)
    rays = np.linalg.inv(frame.intrinsics) @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)
    camera_points = (rays * z).T
    return transform_points(frame.cam_to_world, camera_points), np.stack([u, v], axis=1)


def backproject_pixels(frame: Frame, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lift arbitrary (sub-pixel) pixels: ray from the exact (u, v), depth from the nearest pixel.

    Returns world points (N, 3) and a mask of pixels inside the image with valid depth;
    points where the mask is False are meaningless.
    """
    height, width = frame.image_hw
    cols, rows = np.rint(uv[:, 0]).astype(np.int64), np.rint(uv[:, 1]).astype(np.int64)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    depth = np.zeros(len(uv))
    depth[inside] = frame.depth_m[rows[inside], cols[inside]]
    rays = np.linalg.inv(frame.intrinsics) @ np.vstack([uv.T, np.ones(len(uv))])
    points = transform_points(frame.cam_to_world, (rays * depth).T)
    return points, inside & (depth > 0)


def project(points_world: np.ndarray, frame: Frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a frame.

    Returns float pixels (N, 2) as (u, v), camera depth z (N,), and a mask of points in
    front of the camera whose nearest pixel lies inside the image.
    """
    camera_points = transform_points(np.linalg.inv(frame.cam_to_world), points_world)
    z = camera_points[:, 2]
    safe_z = np.where(np.abs(z) < MIN_DEPTH_M, MIN_DEPTH_M, z)
    homogeneous = camera_points @ frame.intrinsics.T
    uv = homogeneous[:, :2] / safe_z[:, None]
    height, width = frame.image_hw
    inside = (
        (z > MIN_DEPTH_M)
        & (uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5)
        & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5)
    )
    return uv, z, inside


def subsample_frame(frame: Frame, step: int) -> Frame:
    """Keep every `step`-th pixel of RGB and depth; intrinsics scale exactly (pixel centres).

    New pixel (u', v') is old pixel (step*u', step*v'), so K' = diag(1/step, 1/step, 1) K.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    scale = np.diag([1.0 / step, 1.0 / step, 1.0])
    return Frame(frame.index, frame.rgb[::step, ::step].copy(), frame.depth_m[::step, ::step].copy(),
                 scale @ frame.intrinsics, frame.cam_to_world)


def _rotate_once_cw(frame: Frame) -> Frame:
    """Rotate the image 90 deg clockwise and move the camera model with it.

    Pixel (u, v) goes to (H-1-v, u); camera axes become x' = -y, y' = x, z' = z, so the
    same world points are seen, only the image axes change.
    """
    height, _ = frame.image_hw
    k = frame.intrinsics
    intrinsics = np.array([[k[1, 1], 0.0, (height - 1) - k[1, 2]], [0.0, k[0, 0], k[0, 2]], [0.0, 0.0, 1.0]])
    new_to_old = np.eye(4)
    new_to_old[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T
    return Frame(frame.index, np.rot90(frame.rgb, k=-1).copy(), np.rot90(frame.depth_m, k=-1).copy(),
                 intrinsics, frame.cam_to_world @ new_to_old)


def rotate_frame_cw(frame: Frame, quarter_turns: int) -> Frame:
    """Rotate a frame clockwise by `quarter_turns` x 90 deg (e.g. to make images upright)."""
    rotated = frame
    for _ in range(quarter_turns % 4):
        rotated = _rotate_once_cw(rotated)
    return rotated


def _nearest_pixels(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.rint(uv).astype(np.int64)
    return pixels[:, 0], pixels[:, 1]


def visible_mask(points_world: np.ndarray, frame: Frame, depth_tol: float) -> np.ndarray:
    """True where a point is inside the view and agrees with the observed depth (not occluded)."""
    uv, z, inside = project(points_world, frame)
    visible = np.zeros(len(points_world), dtype=bool)
    u, v = _nearest_pixels(uv[inside])
    observed = frame.depth_m[v, u]
    visible[inside] = (observed > 0) & (np.abs(z[inside] - observed) < depth_tol)
    return visible


def depth_residuals(src: Frame, dst: Frame, stride: int = 1) -> np.ndarray:
    """Signed depth error (projected minus observed, metres) of `src` points seen by `dst`.

    Consistent poses, intrinsics and depth units give residuals near zero; a wrong pose,
    a mirrored image or a millimetre/metre mix-up gives large, structured residuals.
    """
    points, _ = backproject(src, stride)
    uv, z, inside = project(points, dst)
    u, v = _nearest_pixels(uv[inside])
    observed = dst.depth_m[v, u]
    has_depth = observed > 0
    return (z[inside] - observed)[has_depth]


def forward_warp(src: Frame, dst: Frame, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Render `src` colours from `dst`'s camera, keeping the nearest surface per pixel.

    Returns the warped image (H, W, 3) in dst's resolution and a mask of filled pixels.
    """
    points, src_pixels = backproject(src, stride)
    uv, z, inside = project(points, dst)
    u, v = _nearest_pixels(uv[inside])
    z, src_pixels = z[inside], src_pixels[inside]

    height, width = dst.image_hw
    flat = v * width + u
    z_buffer = np.full(height * width, np.inf)
    np.minimum.at(z_buffer, flat, z)
    nearest = z <= z_buffer[flat]

    warped = np.zeros((height * width, 3), dtype=np.uint8)
    warped[flat[nearest]] = src.rgb[src_pixels[nearest, 1], src_pixels[nearest, 0]]
    filled = np.isfinite(z_buffer)
    return warped.reshape(height, width, 3), filled.reshape(height, width)
