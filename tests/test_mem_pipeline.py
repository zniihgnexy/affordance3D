import numpy as np
import pytest

from affordance3d.mem.actionable import Stand
from affordance3d.mem.pipeline import MemoryEntry, approach_start, lidar_scaled, retrieve

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


def test_oracle_retrieval_returns_the_parts_a_task_description_names() -> None:
    tasks = [("Open the microwave door", ["a"]), ("Turn on the tap in the sink", ["b", "c"])]
    entries = [_entry("a"), _entry("b"), _entry("c"), _entry("d")]

    found = retrieve(entries, tasks, "turn on the tap in the sink ")

    assert [e.annot_id for e in found] == ["b", "c"]
    assert retrieve(entries, tasks, "open the fridge") == []


def test_approach_starts_between_the_stand_point_and_the_camera_on_the_floor() -> None:
    start = approach_start(_entry("a"), camera_centre=np.array([3.5, 0.0, 1.4]), up=UP, max_walk_m=1.5)

    assert np.allclose(start, [2.0, 0.0, 0.0])  # 1.5 m towards the camera, not up to it
