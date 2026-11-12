"""The reconstruction baseline of phase 1: fuse the keyframes into one point cloud, then read a part from it.

Same inputs as the memory (per-keyframe depth and device poses) and the same labels; only the reading
differs. The memory reads a contact point from one evidence keyframe; here the depth of all keyframes is
fused first, each keyframe's label votes on the fused points, and the contact is read from the points
that the votes assign to the part.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from affordance3d.frames import Frame
from affordance3d.geometry import backproject, project, visible_mask
from affordance3d.mem.pipeline import voxel_points

VISIBLE_TOL_M = 0.05
MIN_VOTE_SHARE = 0.5
POINT_BYTES = 15  # three 32-bit coordinates and an 8-bit colour per point


def fuse_cloud(frames: Sequence[Frame], voxel_m: float, stride: int = 2) -> np.ndarray:
    """One point per occupied voxel of everything the keyframes saw."""
    return voxel_points(np.concatenate([backproject(frame, stride=stride)[0] for frame in frames]), voxel_m)


def vote_share(points: np.ndarray, frames: Sequence[Frame], masks: Sequence[np.ndarray | None]) -> np.ndarray:
    """For each point: among the keyframes that see it and show the part, the share whose label covers it."""
    seen = np.zeros(len(points))
    inside = np.zeros(len(points))
    for frame, mask in zip(frames, masks):
        if mask is None or not mask.any():
            continue
        visible = visible_mask(points, frame, VISIBLE_TOL_M)
        uv, _, _ = project(points[visible], frame)
        cols, rows = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
        seen[visible] += 1
        inside[np.flatnonzero(visible)[mask[rows, cols]]] += 1
    return np.divide(inside, seen, out=np.zeros(len(points)), where=seen > 0)


def fused_contact(points: np.ndarray, frames: Sequence[Frame], masks: Sequence[np.ndarray | None],
                  min_share: float = MIN_VOTE_SHARE) -> np.ndarray | None:
    """Contact point read from the fused cloud: the voted part point closest to the middle of the part; None without votes."""
    part = points[vote_share(points, frames, masks) >= min_share]
    if not len(part):
        return None
    return part[int(np.argmin(np.linalg.norm(part - np.median(part, axis=0), axis=1)))]


def cloud_bytes(points: np.ndarray) -> int:
    return len(points) * POINT_BYTES
