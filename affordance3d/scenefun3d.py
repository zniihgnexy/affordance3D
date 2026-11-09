"""SceneFun3D loader: hires RGB + Faro-mesh-rendered depth + COLMAP poses -> `Frame` records.

Conventions taken from the official toolkit (third_party/scenefun3d/utils/data_parser.py):
- `hires_poses.traj` lines are `timestamp ax ay az tx ty tz`, a world-to-camera transform
  (axis-angle rotation, metres) in the Faro laser-scan frame; we invert it to camera-to-world.
- `hires_depth/*.png` is uint16 millimetres, rendered from the Faro mesh (not sensor depth).
- `hires_wide_intrinsics/*.pincam` holds `width height fx fy cx cy` for each frame.
- Upright orientation: the ARKitScenes `sky_direction` describes the landscape low-res
  frames, while hires frames may be stored in another orientation. We therefore derive the
  rotation from the poses (direction of world +z, up in the gravity-aligned Faro frame, as
  seen in the image) and rotate image, depth, K and pose together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.spatial.transform import Rotation

from affordance3d.frames import Frame, make_frame
from affordance3d.geometry import rotate_frame_cw, subsample_frame

DEPTH_SCALE = 1000.0  # uint16 millimetres -> metres
SKY_TO_CW_TURNS = {"Up": 0, "Left": 1, "Down": 2, "Right": 3}
TRAJ_VALUES = 7


@dataclass(frozen=True)
class FunctionalElement:
    annot_id: str
    label: str  # affordance type, e.g. "rotate", "hook_pull", "tip_push"
    indices: np.ndarray  # indices into the full (uncropped) laser scan


@dataclass(frozen=True)
class SequenceSample:
    """Frames loaded from one video plus the bookkeeping needed for a run manifest."""

    frames: list[Frame]
    timestamps: list[str]
    pose_gaps_s: np.ndarray  # |frame time - matched pose time| per loaded frame
    n_frames_available: int  # frames with RGB, depth, intrinsics and a pose within the gap
    raw_hw: tuple[int, int]  # image size as stored, before subsampling / rotation


def sky_angle_deg(cam_to_world: np.ndarray, world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> float:
    """Direction of world-up in the image: 0 = image up, 90 = right, -90 = left, +-180 = down."""
    up_in_camera = cam_to_world[:3, :3].T @ np.asarray(world_up, dtype=np.float64)
    return float(np.degrees(np.arctan2(up_in_camera[0], -up_in_camera[1])))


def upright_cw_turns(poses: list[np.ndarray]) -> int:
    """Clockwise quarter turns that make most frames upright (majority vote over poses)."""
    votes = [int(np.round(-sky_angle_deg(pose) / 90.0)) % 4 for pose in poses]
    return int(np.bincount(votes, minlength=4).argmax())


def read_traj_poses(path: Path) -> dict[str, np.ndarray]:
    """Timestamp -> 4x4 camera-to-world pose."""
    poses = {}
    for line in path.read_text().splitlines():
        values = line.split()
        if not values:
            continue
        if len(values) != TRAJ_VALUES:
            raise ValueError(f"{path}: expected {TRAJ_VALUES} values per line, got {len(values)}")
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = Rotation.from_rotvec([float(v) for v in values[1:4]]).as_matrix()
        world_to_camera[:3, 3] = [float(v) for v in values[4:7]]
        poses[values[0]] = np.linalg.inv(world_to_camera)
    return poses


def read_pincam(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the 3x3 intrinsics and the (width, height) they refer to."""
    width, height, fx, fy, cx, cy = np.loadtxt(path)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), (int(width), int(height))


def timestamp_of(path: Path) -> str:
    """`<video_id>_<timestamp>.<ext>` -> timestamp string."""
    return path.stem.split("_", 1)[1]


def match_nearest(frame_times: list[str], pose_times: list[str], max_gap: float) -> dict[str, tuple[str, float]]:
    """Pair each frame time with the nearest pose time, dropping pairs further apart than `max_gap` s."""
    pose_values = np.array([float(t) for t in pose_times])
    pairs = {}
    for frame_time in frame_times:
        gaps = np.abs(pose_values - float(frame_time))
        best = int(np.argmin(gaps))
        if gaps[best] <= max_gap:
            pairs[frame_time] = (pose_times[best], float(gaps[best]))
    return pairs


def evenly_spaced(n_total: int, n_wanted: int) -> np.ndarray:
    """Indices spread evenly over time, first and last included; never chosen by labels."""
    return np.unique(np.round(np.linspace(0, n_total - 1, min(n_wanted, n_total))).astype(int))


def read_annotations(path: Path) -> list[FunctionalElement]:
    """Functional elements of a scene; "exclude" regions are not elements and are dropped."""
    data = json.loads(path.read_text())["annotations"]
    return [FunctionalElement(a["annot_id"], a["label"], np.asarray(a["indices"], dtype=np.int64))
            for a in data if a["label"] != "exclude"]


def read_laser_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Full (uncropped) Faro laser scan: points (N, 3) in metres and colours (N, 3) in [0, 1].

    Annotation indices refer to this full scan, so crop with the crop mask only for display.
    """
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points():
        raise ValueError(f"{path} contains no points")
    return np.asarray(cloud.points), np.asarray(cloud.colors, dtype=np.float32)


def read_sky_direction(metadata_csv: Path, video_id: str) -> str:
    """Sky direction of a video from the ARKitScenes metadata shipped with the toolkit."""
    import csv

    with metadata_csv.open() as handle:
        for row in csv.DictReader(handle):
            if row["video_id"] == video_id:
                return row["sky_direction"]
    raise KeyError(f"video {video_id} not in {metadata_csv}")


def _by_timestamp(folder: Path, pattern: str) -> dict[str, Path]:
    return {timestamp_of(path): path for path in sorted(folder.glob(pattern))}


@dataclass(frozen=True)
class SequenceIndex:
    """What exists on disk for one video, keyed by frame timestamp."""

    folder: Path
    rgb: dict[str, Path]
    depth: dict[str, Path]
    pincam: dict[str, Path]
    poses: dict[str, np.ndarray]
    matched: dict[str, tuple[str, float]]  # frame time -> (pose time, |gap| s), for frames with RGB + K

    @property
    def rgb_times(self) -> list[str]:
        """Frames with RGB, intrinsics and a pose (usable as projection targets)."""
        return sorted(self.matched, key=float)

    @property
    def depth_times(self) -> list[str]:
        """Frames that additionally have depth (usable as projection sources)."""
        return [t for t in self.rgb_times if t in self.depth]


def index_sequence(data_root: Path, visit_id: str, video_id: str, max_pose_gap_s: float = 0.05) -> SequenceIndex:
    folder = data_root / visit_id / video_id
    rgb = _by_timestamp(folder / "hires_wide", "*.jpg")
    pincam = _by_timestamp(folder / "hires_wide_intrinsics", "*.pincam")
    poses = read_traj_poses(folder / "hires_poses.traj")
    with_camera = sorted(set(rgb) & set(pincam), key=float)
    matched = match_nearest(with_camera, list(poses), max_pose_gap_s)
    return SequenceIndex(folder, rgb, _by_timestamp(folder / "hires_depth", "*.png"), pincam, poses, matched)


def _read_frame(index: SequenceIndex, position: int, time: str) -> Frame:
    rgb = iio.imread(index.rgb[time])
    intrinsics, (width, height) = read_pincam(index.pincam[time])
    depth = iio.imread(index.depth[time]) if time in index.depth else np.zeros(rgb.shape[:2], np.uint16)
    frame = make_frame(position, rgb, depth, intrinsics, index.poses[index.matched[time][0]], DEPTH_SCALE)
    if frame.image_hw != (height, width):
        raise ValueError(f"{index.rgb[time].name}: image {frame.image_hw} but intrinsics are for {(height, width)}")
    return frame


def load_frames(index: SequenceIndex, times: list[str], cw_turns: int, pixel_step: int,
                first_position: int = 0) -> list[Frame]:
    """Load frames at the given times; frames without depth get an all-zero (missing) depth map."""
    return [rotate_frame_cw(subsample_frame(_read_frame(index, first_position + k, t), pixel_step), cw_turns)
            for k, t in enumerate(times)]


def pick_partners(src_times: list[str], candidates: list[str], min_gap: float, max_gap: float) -> list[str | None]:
    """For each source time, the first candidate between min_gap and max_gap seconds later (or None)."""
    values = sorted(candidates, key=float)
    partners: list[str | None] = []
    for src in src_times:
        window = [c for c in values if min_gap <= float(c) - float(src) <= max_gap]
        partners.append(window[0] if window else None)
    return partners


def load_sequence(data_root: Path, visit_id: str, video_id: str, n_frames: int, cw_turns: int,
                  pixel_step: int = 1, max_pose_gap_s: float = 0.05) -> SequenceSample:
    """Load `n_frames` evenly spaced frames that have depth, subsampled and rotated by `cw_turns`."""
    index = index_sequence(data_root, visit_id, video_id, max_pose_gap_s)
    usable = index.depth_times
    if not usable:
        raise ValueError(f"{index.folder}: no frame has RGB, depth, intrinsics and a pose within {max_pose_gap_s}s")
    chosen = [usable[i] for i in evenly_spaced(len(usable), n_frames)]
    frames = load_frames(index, chosen, cw_turns, pixel_step)
    raw_hw = iio.improps(index.rgb[chosen[0]]).shape[:2]
    return SequenceSample(frames, chosen, np.array([index.matched[t][1] for t in chosen]), len(usable), raw_hw)
