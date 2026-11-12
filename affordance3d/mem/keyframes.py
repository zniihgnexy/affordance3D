"""Choosing keyframes from the device poses alone: in each time window, the frame where the camera turns least.

A frame taken while the camera turns fast is blurred, and labels projected into it do not line up with the
image. The turn rate comes from the device trajectory (an input), so the choice needs no image and no label.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from affordance3d.mem.pipeline import Scene

KEYFRAME_FILE = "keyframes.json"  # written by affordance3d.scenefun3d_partial when only the keyframes were fetched
KINDS = ("uniform", "steady")

RATE_WINDOW_S = 0.1  # the turn rate at time t is measured between t - 0.1 s and t + 0.1 s


def turn_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Angle of the rotation that takes one camera orientation (3x3) to the other."""
    cosine = (np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def turn_rates(times: Sequence[float], pose_times: np.ndarray, rotations: Sequence[np.ndarray],
               window_s: float = RATE_WINDOW_S) -> np.ndarray:
    """Camera turn rate (degrees per second) at each time, from the trajectory samples around it."""
    rates = []
    for t in times:
        before = int(np.argmin(np.abs(pose_times - (t - window_s))))
        after = int(np.argmin(np.abs(pose_times - (t + window_s))))
        seconds = float(pose_times[after] - pose_times[before])
        rates.append(turn_angle_deg(rotations[before], rotations[after]) / seconds if seconds > 0 else np.inf)
    return np.array(rates)


def steady_indices(rates: np.ndarray, n: int) -> np.ndarray:
    """One index per window: the frames (in time order) are cut into `n` windows, each gives its steadiest frame."""
    windows = np.array_split(np.arange(len(rates)), min(n, len(rates)))
    return np.array([window[int(np.argmin(rates[window]))] for window in windows], dtype=int)


def uniform_indices(n_total: int, n: int) -> np.ndarray:
    """Indices spread evenly over time, first and last included."""
    return np.unique(np.round(np.linspace(0, n_total - 1, min(n, n_total))).astype(int))


def saved_keyframes(folder: Path, kind: str) -> list[str] | None:
    """The keyframe choice stored with a video that was fetched keyframes-only, if there is one."""
    file = folder / KEYFRAME_FILE
    return json.loads(file.read_text())[kind] if file.exists() else None


def scene_keyframes(scene: Scene, kind: str = "steady", n: int = 40) -> list[str]:
    """Keyframes of a scene: `uniform` (evenly spaced) or `steady` (the steadiest frame of each time window).

    Only frames that also have ground truth are candidates, so that the result can be scored. A choice saved
    with the video wins, because then only those frames are on disk.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    saved = saved_keyframes(scene.vid.folder, kind)
    if saved is not None:
        return saved
    times = [t for t in scene.vid.rgb_times if t in scene.gt_index.depth]
    if kind == "uniform":
        return [times[i] for i in uniform_indices(len(times), n)]
    rates = turn_rates([float(t) for t in times], scene.vid.arkit_times, [pose[:3, :3] for pose in scene.vid.arkit_poses])
    return [times[i] for i in steady_indices(rates, n)]
