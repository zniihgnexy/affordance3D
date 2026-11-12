import numpy as np
import pytest

from affordance3d.mem import fusion as F
from tests.conftest import TEST_HW, TEST_K, frame_from_depth, pose


def _wall_and_part() -> tuple[list, list[np.ndarray]]:
    """Two cameras 20 cm apart look at a wall 2 m away; the part is the same wall patch in both images."""
    frames = [frame_from_depth(np.full(TEST_HW, 2.0)), frame_from_depth(np.full(TEST_HW, 2.0), cam_to_world=pose(0.0, (0.2, 0.0, 0.0)))]
    masks = []
    for frame in frames:
        mask = np.zeros(TEST_HW, dtype=bool)
        centre = TEST_K @ (np.array([0.1, 0.0, 2.0]) - frame.cam_to_world[:3, 3])  # the wall point (0.1, 0, 2)
        u, v = (centre[:2] / centre[2]).round().astype(int)
        mask[v - 2:v + 3, u - 2:u + 3] = True
        masks.append(mask)
    return frames, masks


def test_fuse_cloud_keeps_one_point_per_voxel_of_the_wall() -> None:
    frames, _ = _wall_and_part()

    coarse, fine = F.fuse_cloud(frames, voxel_m=0.2, stride=1), F.fuse_cloud(frames, voxel_m=0.05, stride=1)

    assert len(coarse) < len(fine) and coarse[:, 2] == pytest.approx(np.full(len(coarse), 2.0), abs=1e-4)
    assert F.cloud_bytes(coarse) == len(coarse) * 15


def test_fused_contact_lands_on_the_part_both_views_label() -> None:
    frames, masks = _wall_and_part()
    points = F.fuse_cloud(frames, voxel_m=0.02, stride=1)

    contact = F.fused_contact(points, frames, masks)
    share = F.vote_share(points, frames, masks)

    assert contact == pytest.approx([0.1, 0.0, 2.0], abs=0.03)
    assert share.max() == pytest.approx(1.0) and (share > 0).sum() < len(points) // 10  # only the patch gets votes


def test_fused_contact_is_none_when_no_keyframe_shows_the_part() -> None:
    frames, _ = _wall_and_part()
    points = F.fuse_cloud(frames, voxel_m=0.05, stride=1)

    assert F.fused_contact(points, frames, [None, np.zeros(TEST_HW, dtype=bool)]) is None
