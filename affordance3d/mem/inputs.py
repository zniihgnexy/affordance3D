"""Video-first inputs for SceneFun3D: what the memory is allowed to see at test time.

Allowed: hires RGB (1440x1920, stored upright), per-frame hires intrinsics, the iPad LiDAR
depth (256x192, landscape, 60 FPS) with its ARKit confidence, and the ARKit camera poses
(`lowres_poses.traj`, landscape camera, ARKit world with +z up).
Not allowed as input: the Faro laser scan, `hires_depth` (rendered from the scan) and
`hires_poses` (COLMAP, registered to the scan). They are loaded separately as ground truth.

Conventions checked on scene 421069 / video 42444490 (2026-11-10):
- The LiDAR image and the ARKit camera become upright after the clockwise quarter turns that
  `upright_cw_turns` derives from the ARKit poses alone; the hires frames are stored upright.
- After the turn, a plain nearest-neighbour resize puts LiDAR depth on the hires pixel grid
  (median relative error 1.3% vs the rendered depth); reprojecting with the two .pincam
  principal points was worse (2.8%), so the pixel grids align even though those files disagree.
- `<video>_transform.npy` maps Faro coordinates to ARKit coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from affordance3d.frames import Frame
from affordance3d.scenefun3d import DEPTH_SCALE, read_pincam, read_traj_poses, timestamp_of, upright_cw_turns

HIGH_CONFIDENCE = 2  # ARKit depth confidence levels are 0 (low), 1 (medium), 2 (high)
MAX_POSE_GAP_S = 0.05
MAX_LIDAR_GAP_S = 0.02
# Camera axes after one clockwise image turn, as in geometry._rotate_once_cw (x' = -y, y' = x).
_ONE_TURN = np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def nearest_index(times: np.ndarray, t: float, max_gap: float) -> int:
    """Index of the sample closest to `t`; an error if none is within `max_gap` seconds."""
    best = int(np.argmin(np.abs(times - t)))
    if abs(times[best] - t) > max_gap:
        raise ValueError(f"no sample within {max_gap}s of {t} (closest {times[best]})")
    return best


def upright_camera_pose(cam_to_world: np.ndarray, cw_turns: int) -> np.ndarray:
    """Camera-to-world pose of the same camera after rotating its image clockwise `cw_turns` times."""
    return cam_to_world @ np.linalg.matrix_power(_ONE_TURN, cw_turns % 4)


def lidar_to_image(depth_m: np.ndarray, confidence: np.ndarray, cw_turns: int, image_hw: tuple[int, int],
                   min_confidence: int = HIGH_CONFIDENCE) -> np.ndarray:
    """LiDAR depth on the image pixel grid: drop low-confidence pixels, turn upright, nearest resize."""
    kept = np.where(confidence >= min_confidence, depth_m, 0.0).astype(np.float32)
    upright = np.rot90(kept, k=-(cw_turns % 4))
    height, width = image_hw
    return cv2.resize(np.ascontiguousarray(upright), (width, height), interpolation=cv2.INTER_NEAREST)


def _by_time(folder: Path, pattern: str) -> tuple[np.ndarray, list[Path]]:
    paths = sorted(folder.glob(pattern), key=lambda p: float(timestamp_of(p)))
    return np.array([float(timestamp_of(p)) for p in paths]), paths


@dataclass(frozen=True)
class VideoInputs:
    """Everything on disk for one video, split into test-time inputs and ground truth."""

    folder: Path
    rgb_times: list[str]  # hires frames that have intrinsics and an ARKit pose
    arkit_times: np.ndarray
    arkit_poses: list[np.ndarray]  # landscape ARKit cameras, camera-to-world
    lidar_times: np.ndarray
    lidar_paths: list[Path]
    cw_turns: int  # clockwise turns that make the LiDAR/ARKit camera upright (from ARKit poses only)
    faro_to_arkit: np.ndarray  # ground-truth registration, for evaluation only

    def rgb_path(self, t: str) -> Path:
        return self.folder / "hires_wide" / f"{self.folder.name}_{t}.jpg"

    def intrinsics(self, t: str) -> np.ndarray:
        return read_pincam(self.folder / "hires_wide_intrinsics" / f"{self.folder.name}_{t}.pincam")[0]

    def arkit_pose(self, t: str) -> np.ndarray:
        """Upright ARKit camera-to-world pose for hires frame `t` (ARKit world, +z up)."""
        pose = self.arkit_poses[nearest_index(self.arkit_times, float(t), MAX_POSE_GAP_S)]
        return upright_camera_pose(pose, self.cw_turns)

    def lidar_depth(self, t: str, image_hw: tuple[int, int], min_confidence: int = HIGH_CONFIDENCE) -> np.ndarray:
        path = self.lidar_paths[nearest_index(self.lidar_times, float(t), MAX_LIDAR_GAP_S)]
        depth = iio.imread(path).astype(np.float32) / DEPTH_SCALE
        confidence = iio.imread(self.folder / "confidence" / path.name)
        return lidar_to_image(depth, confidence, self.cw_turns, image_hw, min_confidence)

    def gt_depth(self, t: str) -> np.ndarray:
        """Depth rendered from the Faro scan: ground truth / oracle only, never a model input."""
        path = self.folder / "hires_depth" / f"{self.folder.name}_{t}.png"
        return iio.imread(path).astype(np.float32) / DEPTH_SCALE

    def frame(self, t: str, depth_m: np.ndarray, position: int = 0) -> Frame:
        """Input frame: hires RGB, the given depth (LiDAR, monocular or oracle) and the ARKit pose."""
        rgb = iio.imread(self.rgb_path(t))
        if depth_m.shape != rgb.shape[:2]:
            raise ValueError(f"{t}: depth {depth_m.shape} does not match the image {rgb.shape[:2]}")
        return Frame(position, rgb, depth_m.astype(np.float32), self.intrinsics(t), self.arkit_pose(t))


def index_video(data_root: Path, visit_id: str, video_id: str) -> VideoInputs:
    folder = data_root / visit_id / video_id
    arkit = read_traj_poses(folder / "lowres_poses.traj")
    arkit_times = np.array([float(t) for t in arkit])
    lidar_times, lidar_paths = _by_time(folder / "lowres_depth", "*.png")
    rgb = {timestamp_of(p) for p in (folder / "hires_wide").glob("*.jpg")}
    pincam = {timestamp_of(p) for p in (folder / "hires_wide_intrinsics").glob("*.pincam")}
    with_pose = [t for t in sorted(rgb & pincam, key=float)
                 if np.min(np.abs(arkit_times - float(t))) <= MAX_POSE_GAP_S
                 and np.min(np.abs(lidar_times - float(t))) <= MAX_LIDAR_GAP_S]
    return VideoInputs(folder, with_pose, arkit_times, list(arkit.values()), lidar_times, lidar_paths,
                       upright_cw_turns(list(arkit.values())), np.load(folder / f"{video_id}_transform.npy"))
