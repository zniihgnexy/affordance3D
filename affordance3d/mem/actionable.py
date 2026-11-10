"""Actionable affordance: where to touch a part, how it moves, and where a person stands to act.

Everything here works in one world frame with a known gravity direction `up` (ARKit world:
+z). The same functions are used with oracle labels and, later, with predicted ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAND_DISTANCE_M = 0.5  # horizontal distance from the contact point to the body centre
BODY_RADIUS_M = 0.2  # body cylinder used for the free-space check
BODY_MIN_HEIGHT_M = 0.15  # ignore floor noise and low clutter below the shins
BODY_MAX_HEIGHT_M = 1.8
MIN_HORIZONTAL = 0.3  # below this, the normal is treated as vertical (part on a top surface)
FLOOR_BAND = 0.2  # fraction of lowest points searched for the floor
FLOOR_BIN_M = 0.02
ALTERNATIVE_ANGLES_DEG = (0.0, 30.0, -30.0, 60.0, -60.0)
ALTERNATIVE_EXTRA_M = (0.0, 0.15)


@dataclass(frozen=True)
class Stand:
    position: np.ndarray  # on the floor
    facing: np.ndarray  # horizontal unit vector towards the part
    free: bool  # body cylinder clear of observed points
    moved: bool  # an alternative to the straight-ahead stand point was needed


def surface_normal(points: np.ndarray, viewer: np.ndarray) -> np.ndarray:
    """Plane normal of a point neighbourhood, oriented towards the viewer."""
    centred = points - points.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    return normal if normal @ (viewer - points.mean(axis=0)) > 0 else -normal


def floor_height(heights: np.ndarray) -> float:
    """Most common height among the lowest points (robust to a few points below the floor)."""
    low = np.sort(heights)[: max(1, int(len(heights) * FLOOR_BAND))]
    bins = np.arange(low.min(), low.max() + 2 * FLOOR_BIN_M, FLOOR_BIN_M)
    counts, edges = np.histogram(low, bins=bins)
    best = int(np.argmax(counts))
    in_bin = low[(low >= edges[best]) & (low <= edges[best + 1])]
    return float(np.median(in_bin))


def _horizontal(vector: np.ndarray, up: np.ndarray) -> np.ndarray:
    return vector - (vector @ up) * up


def is_free(position: np.ndarray, occupied: np.ndarray, up: np.ndarray, max_points: int = 0,
            radius: float = BODY_RADIUS_M) -> bool:
    """True when at most `max_points` observed points fall inside the body cylinder at `position`."""
    if not len(occupied):
        return True
    offsets = occupied - position
    height = offsets @ up
    horizontal = np.linalg.norm(offsets - height[:, None] * up, axis=1)
    inside = (horizontal < radius) & (height > BODY_MIN_HEIGHT_M) & (height < BODY_MAX_HEIGHT_M)
    return int(inside.sum()) <= max_points


def _rotate_about(vector: np.ndarray, up: np.ndarray, degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return vector * np.cos(angle) + np.cross(up, vector) * np.sin(angle) + up * (up @ vector) * (1 - np.cos(angle))


def plan_stand(contact: np.ndarray, normal: np.ndarray, up: np.ndarray, floor_z: float, viewer: np.ndarray,
               occupied: np.ndarray, distance: float = STAND_DISTANCE_M, max_points: int = 0) -> Stand:
    """Stand on the floor in front of the part; if blocked, try turned or farther positions."""
    away = _horizontal(normal, up)
    if np.linalg.norm(away) < MIN_HORIZONTAL:  # part faces up/down: approach from where it was seen
        away = _horizontal(viewer - contact, up)
    away = away / np.linalg.norm(away)
    base = contact - ((contact @ up) - floor_z) * up
    first = None
    for degrees in ALTERNATIVE_ANGLES_DEG:
        direction = _rotate_about(away, up, degrees)
        for extra in ALTERNATIVE_EXTRA_M:
            position = base + (distance + extra) * direction
            free = is_free(position, occupied, up, max_points)
            stand = Stand(position, -direction, free, moved=bool(degrees or extra))
            if free:
                return stand
            first = first or stand
    return first


def distance_to_part(point: np.ndarray, part_points: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(part_points - point, axis=1)))


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two axes, ignoring their sign (a rotation axis has no preferred direction here)."""
    cosine = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
