"""Quantitative E0 checks on real data.

- Cross-frame reprojection of SIFT matches: correspondences come from the images alone, so
  they test RGB, depth, intrinsics and poses together (a wrong pose, rotation, scale or
  timestamp pairing shows up as large pixel errors).
- Distance from depth-lifted points to the laser scan: checks depth units and that poses
  are in the scan's (Faro) frame.
- Visible points per functional element and frame: how often each element is observed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from affordance3d.frames import Frame
from affordance3d.geometry import backproject_pixels, project, visible_mask

SIFT_RATIO = 0.75  # Lowe's ratio test
MAX_FEATURES = 4000


@dataclass(frozen=True)
class MatchReprojection:
    src_uv: np.ndarray  # (N, 2) keypoints in the source frame
    dst_uv: np.ndarray  # (N, 2) matched keypoints observed in the destination frame
    predicted_uv: np.ndarray  # (N, 2) source keypoints lifted with source depth+pose, projected into dst

    @property
    def errors_px(self) -> np.ndarray:
        return np.linalg.norm(self.predicted_uv - self.dst_uv, axis=1)


MIN_EPIPOLAR_MATCHES = 8  # the 8-point algorithm's minimum
RANSAC_CONFIDENCE = 0.999


@dataclass(frozen=True)
class SiftFeatures:
    uv: np.ndarray  # (N, 2) keypoint positions (u, v)
    descriptors: np.ndarray  # (N, 128) float32


def sift_features(rgb: np.ndarray, max_features: int = MAX_FEATURES) -> SiftFeatures:
    """Detect SIFT keypoints once per frame so many pairs can be matched cheaply."""
    keys, desc = cv2.SIFT_create(nfeatures=max_features).detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    if desc is None:
        return SiftFeatures(np.empty((0, 2)), np.empty((0, 128), np.float32))
    return SiftFeatures(np.array([k.pt for k in keys], dtype=np.float64), desc)


def match_features(a: SiftFeatures, b: SiftFeatures, ratio: float = SIFT_RATIO) -> tuple[np.ndarray, np.ndarray]:
    """Image-only correspondences (u, v) from Lowe's ratio test; no geometry is used."""
    if len(a.uv) == 0 or len(b.uv) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.descriptors, b.descriptors, k=2)
    good = [best for best, second in (p for p in pairs if len(p) == 2) if best.distance < ratio * second.distance]
    rows_a = np.array([m.queryIdx for m in good], dtype=np.int64)
    rows_b = np.array([m.trainIdx for m in good], dtype=np.int64)
    return a.uv[rows_a].reshape(-1, 2), b.uv[rows_b].reshape(-1, 2)


def sift_matches(rgb_a: np.ndarray, rgb_b: np.ndarray, ratio: float = SIFT_RATIO,
                 max_features: int = MAX_FEATURES) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: detect and match two images in one call."""
    return match_features(sift_features(rgb_a, max_features), sift_features(rgb_b, max_features), ratio)


def epipolar_inliers(uv_a: np.ndarray, uv_b: np.ndarray, threshold_px: float) -> np.ndarray:
    """Matches consistent with one epipolar geometry (RANSAC fundamental matrix).

    Uses the images' correspondences only, never our poses or depth, so filtering out
    impossible matches cannot bias the reprojection check in favour of the poses.
    """
    keep = np.zeros(len(uv_a), dtype=bool)
    if len(uv_a) < MIN_EPIPOLAR_MATCHES:
        return keep
    _, mask = cv2.findFundamentalMat(uv_a, uv_b, cv2.FM_RANSAC, threshold_px, RANSAC_CONFIDENCE)
    if mask is None:
        return keep
    return mask.ravel().astype(bool)


def reproject_matches(src: Frame, dst: Frame, uv_src: np.ndarray, uv_dst: np.ndarray) -> MatchReprojection:
    """Predict where each source keypoint should appear in dst; keep matches with valid depth in view."""
    points, valid = backproject_pixels(src, uv_src)
    predicted, _, inside = project(points[valid], dst)
    return MatchReprojection(uv_src[valid][inside], uv_dst[valid][inside], predicted[inside])


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Euclidean distance from each query point to its nearest reference point."""
    distances, _ = cKDTree(reference).query(query, workers=-1)
    return distances


def visible_point_counts(point_sets: Sequence[np.ndarray], frames: Sequence[Frame], depth_tol: float) -> np.ndarray:
    """(E, F) number of points of each set visible (in view, not occluded) in each frame."""
    return np.array([[int(visible_mask(points, frame, depth_tol).sum()) for frame in frames]
                     for points in point_sets], dtype=np.int64).reshape(len(point_sets), len(frames))
