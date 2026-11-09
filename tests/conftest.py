"""Synthetic RGB-D frames with known geometry for the unit tests."""

import numpy as np
import pytest

from affordance3d.frames import Frame

TEST_K = np.array([[100.0, 0.0, 31.5], [0.0, 100.0, 23.5], [0.0, 0.0, 1.0]])
TEST_HW = (48, 64)


def pose(yaw_deg: float = 0.0, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Camera-to-world transform: rotation about the camera y axis, then translation."""
    a = np.deg2rad(yaw_deg)
    transform = np.eye(4)
    transform[:3, :3] = [[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]
    transform[:3, 3] = translation
    return transform


def frame_from_depth(depth: np.ndarray, cam_to_world: np.ndarray | None = None, index: int = 0,
                     rgb: np.ndarray | None = None) -> Frame:
    height, width = depth.shape
    colours = np.zeros((height, width, 3), np.uint8) if rgb is None else rgb
    return Frame(index, colours, depth.astype(np.float32), TEST_K,
                 np.eye(4) if cam_to_world is None else cam_to_world)


@pytest.fixture
def wall_frame() -> Frame:
    """Camera at the origin facing a flat wall 2 m away."""
    return frame_from_depth(np.full(TEST_HW, 2.0))
