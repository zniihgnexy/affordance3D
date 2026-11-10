"""Training labels from the 3D ground truth: functional parts and motion axes projected into frames.

Labels are made offline with the ground-truth camera (COLMAP pose, depth rendered from the
scan); they supervise and evaluate per-frame predictions and are never model inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from affordance3d.frames import Frame
from affordance3d.geometry import project, visible_mask

BACKGROUND = -1


def part_label_map(part_points: Sequence[np.ndarray], frame: Frame, depth_tol: float) -> np.ndarray:
    """Per-pixel part index (BACKGROUND elsewhere) for the visible points of each part."""
    labels = np.full(frame.image_hw, BACKGROUND, dtype=np.int32)
    for index, points in enumerate(part_points):
        seen = visible_mask(points, frame, depth_tol)
        if not seen.any():
            continue
        uv, _, _ = project(points[seen], frame)
        cols, rows = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
        labels[rows, cols] = index
    return labels


def projected_axis(origin: np.ndarray, direction: np.ndarray, frame: Frame,
                   length_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of a motion axis drawn from its origin for `length_m` metres along `direction`."""
    tip = origin + length_m * direction / np.linalg.norm(direction)
    uv, _, _ = project(np.stack([origin, tip]), frame)
    return uv[0], uv[1]
