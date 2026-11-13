"""Hand events from Aria MPS wrist/palm tracking, and their scoring against annotated pick-ups and put-downs (B-1).

Hand positions come in the device frame on the device clock (10 Hz). The device trajectory of the multi-recording
SLAM (`closed_loop_trajectory.csv`, device-to-world, same clock) moves them into the kitchen's world frame. An
interaction moment is where a palm slows down: within each run of slow samples, the sample of lowest speed is an
event (time, 3D palm position). Events are matched one to one with the annotated events within a time and a
distance tolerance.

The detector's parameters below were fixed before any scoring (discussion draft, section 25.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FPS = 30  # MP4 frames per second
MAX_POSE_GAP_US = 10_000  # a hand sample needs a trajectory pose at most this far away (the trajectory is ~1 kHz)
MAX_STEP_S = 0.15  # neighbouring hand samples further apart than this are not used for a speed
SLOW_SPEED = 0.15  # m/s: the operating point; a palm slower than this is "slow"
MIN_SLOW_SAMPLES = 2  # a run of slow samples must be at least this long to give an event
MERGE_S, MERGE_M = 0.5, 0.15  # events of the two hands this close in time and space are one event
MATCH_S, MATCH_M = 1.0, 0.30  # an event is found if a detection lies within 1 s and 30 cm


@dataclass(frozen=True)
class Event:
    """A moment of interaction: MP4 time in seconds and a 3D world position."""

    seconds: float
    position: tuple[float, float, float]
    speed: float = 0.0
    kind: str = ""


def rotate(quaternions_xyzw: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate each vector by its unit quaternion (x, y, z, w)."""
    q = quaternions_xyzw / np.linalg.norm(quaternions_xyzw, axis=-1, keepdims=True)
    axis, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(axis, vectors)
    return vectors + w * t + np.cross(axis, t)


def nearest_poses(trajectory_us: np.ndarray, at_us: np.ndarray, max_gap_us: int = MAX_POSE_GAP_US) -> np.ndarray:
    """Index of the nearest trajectory sample for each time, or -1 if none is within `max_gap_us`."""
    right = np.clip(np.searchsorted(trajectory_us, at_us), 1, len(trajectory_us) - 1)
    left = right - 1
    nearest = np.where(np.abs(trajectory_us[left] - at_us) <= np.abs(trajectory_us[right] - at_us), left, right)
    return np.where(np.abs(trajectory_us[nearest] - at_us) <= max_gap_us, nearest, -1)


def to_world(points_device: np.ndarray, translations: np.ndarray, quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Device-frame points moved into the world frame by device-to-world poses (one pose per point)."""
    return rotate(quaternions_xyzw, points_device) + translations


def mp4_seconds(device_us: np.ndarray, first_frame_us: int) -> np.ndarray:
    """MP4 time of device-clock times, given the device time of the first video frame."""
    return (np.asarray(device_us, dtype=np.int64) - first_frame_us) / 1e6


def speeds(seconds: np.ndarray, positions: np.ndarray, valid: np.ndarray,
           max_step_s: float = MAX_STEP_S) -> np.ndarray:
    """Central-difference speed (m/s) of each sample; NaN where a neighbour is missing, invalid or too far away."""
    result = np.full(len(seconds), np.nan)
    if len(seconds) < 3:
        return result
    dt = seconds[2:] - seconds[:-2]
    usable = valid[:-2] & valid[1:-1] & valid[2:] & (seconds[1:-1] - seconds[:-2] <= max_step_s) \
        & (seconds[2:] - seconds[1:-1] <= max_step_s)
    distance = np.linalg.norm(positions[2:] - positions[:-2], axis=1)
    result[1:-1] = np.where(usable, distance / np.where(dt > 0, dt, np.nan), np.nan)
    return result


def slow_events(seconds: np.ndarray, positions: np.ndarray, speed: np.ndarray, slow: float = SLOW_SPEED,
                min_samples: int = MIN_SLOW_SAMPLES) -> list[Event]:
    """One event per run of consecutive slow samples: the sample of lowest speed in the run."""
    is_slow = np.nan_to_num(speed, nan=np.inf) < slow
    events, start = [], None
    for index, flag in enumerate(np.append(is_slow, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_samples:
                best = start + int(np.argmin(speed[start:index]))
                position = tuple(float(v) for v in positions[best])
                events.append(Event(float(seconds[best]), position, float(speed[best])))
            start = None
    return events


def merge_hands(events: list[Event], max_s: float = MERGE_S, max_m: float = MERGE_M) -> list[Event]:
    """Events of both hands in time order; of two close in time and space, the slower one stays."""
    kept: list[Event] = []
    for event in sorted(events, key=lambda e: e.speed):
        if all(abs(event.seconds - other.seconds) > max_s
               or np.linalg.norm(np.subtract(event.position, other.position)) > max_m for other in kept):
            kept.append(event)
    return sorted(kept, key=lambda e: e.seconds)


def match(detections: list[Event], truths: list[Event], max_s: float = MATCH_S,
          max_m: float = MATCH_M) -> list[tuple[int, int]]:
    """One-to-one (detection, truth) pairs within both tolerances, taken greedily by time difference."""
    pairs = []
    truth_times = np.array([t.seconds for t in truths])
    for d, detection in enumerate(detections):
        if not len(truths):
            break
        for t in np.flatnonzero(np.abs(truth_times - detection.seconds) <= max_s):
            gap = np.linalg.norm(np.subtract(detection.position, truths[t].position))
            if gap <= max_m:
                pairs.append((abs(truth_times[t] - detection.seconds), gap, d, int(t)))
    used_d, used_t, result = set(), set(), []
    for _, _, d, t in sorted(pairs):
        if d not in used_d and t not in used_t:
            used_d.add(d)
            used_t.add(t)
            result.append((d, t))
    return result


def precision_recall(detections: list[Event], truths: list[Event], **tolerances: float) -> dict:
    """Matches, precision and recall of detections against truths."""
    found = len(match(detections, truths, **tolerances))
    return {"detections": len(detections), "truths": len(truths), "matched": found,
            "precision": found / len(detections) if detections else None,
            "recall": found / len(truths) if truths else None}


def annotated_events(priming: dict, fps: int = FPS) -> list[Event]:
    """Pick-ups ("start") and put-downs ("end") of one video's priming annotations that have a frame and a place."""
    events = []
    for entry in priming.values():
        for key, kind in (("start", "pick-up"), ("end", "put-down")):
            item = entry.get(key)
            if isinstance(item, dict) and isinstance(item.get("frame"), int) and item["frame"] >= 0 \
                    and len(item.get("3d_location") or []) == 3:
                events.append(Event(item["frame"] / fps, tuple(float(v) for v in item["3d_location"]), kind=kind))
    return sorted(events, key=lambda e: e.seconds)
