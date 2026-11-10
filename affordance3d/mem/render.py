"""Small CPU renderer for the humanoid display: triangle z-buffer, point splats and image compositing.

Open3D's offscreen renderer works on the Mac (Metal) for colour but cannot return depth there,
and occlusion against the scene depth needs the body's depth per pixel, so the body is
rasterised here. A 20k-triangle body at 720x960 takes well under a second.
Pixel convention as in geometry.py: (u, v) = (column, row), integer pixel centres.
"""

from __future__ import annotations

import numpy as np

BODY_COLOUR = np.array([0.62, 0.74, 0.95])
NEAR_M = 0.05


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world pose of a camera at `eye` looking at `target` (x right, y down, z forward)."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = eye
    return pose


def to_camera(points_world: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    world_to_cam = np.linalg.inv(cam_to_world)
    return points_world @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]


def rasterize(vertices_cam: np.ndarray, faces: np.ndarray, intrinsics: np.ndarray,
              image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel depth (inf where empty) and headlight Lambert shade in [0, 1] of a triangle mesh."""
    height, width = image_hw
    depth = np.full(image_hw, np.inf)
    shade = np.zeros(image_hw)
    z = vertices_cam[:, 2]
    uv = (vertices_cam @ intrinsics.T)[:, :2] / np.maximum(z, NEAR_M)[:, None]
    tri = vertices_cam[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    centres = tri.mean(axis=1)
    face_shade = np.abs(np.sum(normals * centres, axis=1)) / (np.linalg.norm(centres, axis=1) + 1e-12)
    for f, (a, b, c) in enumerate(faces):
        if min(z[a], z[b], z[c]) < NEAR_M:
            continue
        pts = uv[[a, b, c]]
        u0, v0 = np.floor(pts.min(axis=0)).astype(int)
        u1, v1 = np.ceil(pts.max(axis=0)).astype(int)
        u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, width - 1), min(v1, height - 1)
        if u0 > u1 or v0 > v1:
            continue
        us, vs = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        (xa, ya), (xb, yb), (xc, yc) = pts
        area = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
        if abs(area) < 1e-12:
            continue
        w1 = ((us - xa) * (yc - ya) - (xc - xa) * (vs - ya)) / area
        w2 = ((xb - xa) * (vs - ya) - (us - xa) * (yb - ya)) / area
        w0 = 1 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        # perspective-correct depth: interpolate 1/z
        inv_z = w0 / z[a] + w1 / z[b] + w2 / z[c]
        d = 1.0 / inv_z[inside]
        rows, cols = vs[inside], us[inside]
        nearer = d < depth[rows, cols]
        depth[rows[nearer], cols[nearer]] = d[nearer]
        shade[rows[nearer], cols[nearer]] = face_shade[f]
    return depth, shade


def splat(points_cam: np.ndarray, colours: np.ndarray, intrinsics: np.ndarray, image_hw: tuple[int, int],
          radius_px: int = 1, background: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Render points as small squares with a z-buffer; returns an RGB image in [0, 1] and depth (inf = empty)."""
    height, width = image_hw
    image = np.full((height, width, 3), background)
    depth = np.full(image_hw, np.inf)
    keep = points_cam[:, 2] > NEAR_M
    points, colours = points_cam[keep], colours[keep]
    uv = (points @ intrinsics.T)[:, :2] / points[:, 2:3]
    order = np.argsort(-points[:, 2])  # far first, so nearer points overwrite
    for du in range(-radius_px, radius_px + 1):
        for dv in range(-radius_px, radius_px + 1):
            cols = np.rint(uv[order, 0]).astype(int) + du
            rows = np.rint(uv[order, 1]).astype(int) + dv
            ok = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            z = points[order, 2][ok]
            r, c = rows[ok], cols[ok]
            closer = z < depth[r, c]  # resolves against earlier offsets; within one pass the last write is nearest
            depth[r[closer], c[closer]] = z[closer]
            image[r[closer], c[closer]] = colours[order][ok][closer]
    return image, depth


def overlay_body(rgb: np.ndarray, scene_depth: np.ndarray, body_depth: np.ndarray, body_shade: np.ndarray,
                 margin_m: float = 0.03, alpha: float = 0.9, colour: np.ndarray = BODY_COLOUR) -> np.ndarray:
    """Draw the body where it is in front of the scene (or where the scene has no depth)."""
    visible = np.isfinite(body_depth) & ((scene_depth <= 0) | (body_depth < scene_depth + margin_m))
    shaded = (0.35 + 0.65 * body_shade[..., None]) * colour * 255
    out = rgb.astype(np.float64).copy()
    out[visible] = (1 - alpha) * out[visible] + alpha * shaded[visible]
    return out.astype(np.uint8)
