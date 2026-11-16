"""From a mask on one frame to points of the laser scan (discussion draft, section 27.2).

The rule is Fun3DU's for a single view (utils/sun3d/fusion_util.py of its release): every scan point is projected into
the frame and its pixel rounded; a point is kept when that pixel lies in the image and in the mask, and the frame's
depth d there agrees with the point's depth z: |d - z| <= 0.25 d (pixels without depth never keep a point). With one
view, Fun3DU's later min-max normalisation and "> 0.7" threshold return exactly these points; when no point passes,
the prediction is empty.
"""
from collections.abc import Sequence

import numpy as np

from affordance3d.frames import Frame
from affordance3d.geometry import project

REL_DEPTH_TOL = 0.25  # Fun3DU's visibility_threshold, relative to the frame's depth


def _pixels(points: np.ndarray, frame: Frame, rel_tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices of the points that are in view and agree with the frame's depth, and their pixel columns and rows."""
    uv, z, inside = project(points, frame)
    index = np.flatnonzero(inside)
    cols, rows = np.rint(uv[index, 0]).astype(np.int64), np.rint(uv[index, 1]).astype(np.int64)
    observed = frame.depth_m[rows, cols]
    agree = (observed > 0) & (np.abs(observed - z[index]) <= rel_tol * observed)
    return index[agree], cols[agree], rows[agree]


def lift_mask(points: np.ndarray, frame: Frame, mask: np.ndarray, rel_tol: float = REL_DEPTH_TOL) -> np.ndarray:
    """Boolean per point: projected into `mask` (same size as the frame) and agreeing with the frame's depth."""
    if mask.shape != frame.image_hw:
        raise ValueError(f"mask {mask.shape} but frame {frame.image_hw}")
    index, cols, rows = _pixels(points, frame, rel_tol)
    kept = np.zeros(len(points), dtype=bool)
    kept[index[mask[rows, cols]]] = True
    return kept


def upsample(mask: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour enlargement: every pixel becomes a factor x factor block."""
    return np.repeat(np.repeat(np.asarray(mask, dtype=bool), factor, axis=0), factor, axis=1)


def region_mask(points: np.ndarray, frame: Frame, rel_tol: float = REL_DEPTH_TOL) -> np.ndarray:
    """The pixels where `points` are seen in the frame by the same rule (for the lifting upper bound of 27.2, item 5)."""
    _, cols, rows = _pixels(points, frame, rel_tol)
    mask = np.zeros(frame.image_hw, dtype=bool)
    mask[rows, cols] = True
    return mask


def most_visible(points: np.ndarray, frames: Sequence[Frame], rel_tol: float = REL_DEPTH_TOL) -> int | None:
    """The frame in which most of `points` are seen (ties: the earlier); None when no frame sees any."""
    counts = [len(_pixels(points, frame, rel_tol)[0]) for frame in frames]
    return int(np.argmax(counts)) if counts and max(counts) > 0 else None
