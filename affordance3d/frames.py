"""RGB-D frame records and the Open3D Redwood sample loader.

Every dataset loader returns a list of `Frame`, so the later steps (projection,
features, memory) do not depend on where the data came from. A SceneFun3D loader
only has to produce the same record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REDWOOD_DEPTH_SCALE = 1000.0  # depth PNGs store uint16 millimetres
LOG_ENTRY_LINES = 5  # one header line followed by four rows of a 4x4 matrix


@dataclass(frozen=True)
class Frame:
    """One time-stamped RGB-D observation with its camera."""

    index: int
    rgb: np.ndarray  # (H, W, 3) uint8
    depth_m: np.ndarray  # (H, W) float32 metres; 0 marks missing depth
    intrinsics: np.ndarray  # (3, 3) pinhole matrix K
    cam_to_world: np.ndarray  # (4, 4) camera-to-world transform

    @property
    def image_hw(self) -> tuple[int, int]:
        height, width = self.depth_m.shape
        return height, width


def read_log_poses(path: Path) -> list[np.ndarray]:
    """Parse an Open3D/Redwood `.log` trajectory into 4x4 camera-to-world matrices."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) % LOG_ENTRY_LINES:
        raise ValueError(f"{path} has {len(rows)} lines, not a multiple of {LOG_ENTRY_LINES}")
    return [
        np.array(rows[start + 1 : start + LOG_ENTRY_LINES], dtype=np.float64)
        for start in range(0, len(rows), LOG_ENTRY_LINES)
    ]


def read_intrinsics(path: Path) -> np.ndarray:
    """Read an Open3D camera JSON; its 3x3 matrix is stored column-major."""
    data = json.loads(path.read_text())
    return np.array(data["intrinsic_matrix"], dtype=np.float64).reshape(3, 3).T


def make_frame(index: int, rgb: np.ndarray, depth_raw: np.ndarray, intrinsics: np.ndarray,
               cam_to_world: np.ndarray, depth_scale: float) -> Frame:
    """Validate one RGB-D pair and convert depth to metres."""
    if rgb.shape[:2] != depth_raw.shape:
        raise ValueError(f"frame {index}: RGB {rgb.shape[:2]} and depth {depth_raw.shape} differ")
    if cam_to_world.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError(f"frame {index}: expected 4x4 pose and 3x3 intrinsics")
    depth_m = depth_raw.astype(np.float32) / depth_scale
    return Frame(index, rgb, depth_m, intrinsics, cam_to_world)


def load_redwood_sample(data_root: Path) -> list[Frame]:
    """Download (once) and load Open3D's 5-frame Redwood living-room RGB-D sample."""
    import open3d as o3d

    sample = o3d.data.SampleRedwoodRGBDImages(data_root=str(data_root))
    intrinsics = read_intrinsics(Path(sample.camera_intrinsic_path))
    poses = read_log_poses(Path(sample.odometry_log_path))
    pairs = zip(sample.color_paths, sample.depth_paths, poses, strict=True)
    return [
        make_frame(i, iio.imread(color), iio.imread(depth), intrinsics, pose, REDWOOD_DEPTH_SCALE)
        for i, (color, depth, pose) in enumerate(pairs)
    ]
