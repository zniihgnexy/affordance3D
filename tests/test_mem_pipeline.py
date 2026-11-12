import numpy as np
import pytest

from affordance3d.mem.actionable import Stand
from affordance3d.mem.pipeline import (
    MemoryEntry,
    approach_start,
    contact_and_normal,
    lidar_scaled,
    lidar_scaled_together,
    retrieve,
)
from tests.conftest import TEST_HW, frame_from_depth

UP = np.array([0.0, 0.0, 1.0])


def _entry(annot_id: str, stand_xy=(0.5, 0.0)) -> MemoryEntry:
    stand = Stand(np.array([*stand_xy, 0.0]), np.array([-1.0, 0.0, 0.0]), True, False)
    return MemoryEntry(0, annot_id, "rotate", "", "1.0", (0, 0), 20, ("1.0",), np.zeros(3), np.array([1.0, 0, 0]),
                       stand, None)


def test_lidar_scaling_uses_the_median_ratio_over_pixels_with_both_depths() -> None:
    depth = np.array([[1.0, 2.0], [4.0, 3.0]])
    lidar = np.array([[0.8, 1.6], [0.0, 2.4]])  # true scale 0.8; one pixel without LiDAR

    assert np.allclose(lidar_scaled(depth, lidar), depth * 0.8)
    with pytest.raises(ValueError, match="no pixel"):
        lidar_scaled(depth, np.zeros_like(lidar))


def test_a_frame_without_usable_lidar_takes_the_median_scale_of_the_other_frames() -> None:
    depth = np.array([[1.0, 2.0], [4.0, 3.0]])
    lidars = [depth * 0.8, np.zeros_like(depth), depth * 1.0, depth * 0.9]  # the second frame has no LiDAR pixel

    scaled = lidar_scaled_together([depth] * 4, lidars)

    assert [float(frame[0, 0]) for frame in scaled] == pytest.approx([0.8, 0.9, 1.0, 0.9])  # own scale, except the median for frame 2
    with pytest.raises(ValueError, match="no frame"):
        lidar_scaled_together([depth], [np.zeros_like(depth)])
    with pytest.raises(ValueError):
        lidar_scaled_together([depth, depth], lidars[:1])


def test_a_part_that_fills_the_window_takes_the_normal_of_its_own_surface() -> None:
    frame = frame_from_depth(np.full(TEST_HW, 2.0))  # a flat wall 2 m in front of the camera
    labels = np.zeros(TEST_HW, dtype=np.int32)  # one part covers the whole image: nothing around it

    pixel, contact, normal = contact_and_normal(frame, labels, part=0)

    assert contact[2] == pytest.approx(2.0) and 0 <= pixel[0] < TEST_HW[1] and 0 <= pixel[1] < TEST_HW[0]
    assert normal @ np.array([0.0, 0.0, -1.0]) > 0.99  # the wall faces the camera
    assert contact_and_normal(frame_from_depth(np.zeros(TEST_HW)), labels, part=0) is None  # no depth at all


def test_oracle_retrieval_returns_the_parts_a_task_description_names() -> None:
    tasks = [("Open the microwave door", ["a"]), ("Turn on the tap in the sink", ["b", "c"])]
    entries = [_entry("a"), _entry("b"), _entry("c"), _entry("d")]

    found = retrieve(entries, tasks, "turn on the tap in the sink ")

    assert [e.annot_id for e in found] == ["b", "c"]
    assert retrieve(entries, tasks, "open the fridge") == []


def test_approach_starts_between_the_stand_point_and_the_camera_on_the_floor() -> None:
    start = approach_start(_entry("a"), camera_centre=np.array([3.5, 0.0, 1.4]), up=UP, max_walk_m=1.5)

    assert np.allclose(start, [2.0, 0.0, 0.0])  # 1.5 m towards the camera, not up to it
