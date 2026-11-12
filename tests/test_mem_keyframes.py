import numpy as np
import pytest

from affordance3d.mem import keyframes as K
from tests.conftest import pose


def test_turn_angle_between_two_orientations() -> None:
    assert K.turn_angle_deg(pose(0)[:3, :3], pose(30)[:3, :3]) == pytest.approx(30.0)
    assert K.turn_angle_deg(pose(10)[:3, :3], pose(10)[:3, :3]) == pytest.approx(0.0, abs=1e-5)


def test_turn_rates_use_the_trajectory_samples_around_each_time() -> None:
    pose_times = np.arange(0.0, 2.01, 0.1)
    yaw = np.where(pose_times < 1.0, 0.0, (pose_times - 1.0) * 50.0)  # still for 1 s, then 50 degrees per second
    rotations = [pose(angle)[:3, :3] for angle in yaw]

    rates = K.turn_rates([0.5, 1.5], pose_times, rotations)

    assert rates == pytest.approx([0.0, 50.0], abs=1e-4)


def test_steady_indices_take_the_slowest_frame_of_each_window() -> None:
    rates = np.array([9.0, 1.0, 5.0, 7.0, 8.0, 2.0])

    assert K.steady_indices(rates, 2).tolist() == [1, 5]
    assert K.steady_indices(rates, 10).tolist() == [0, 1, 2, 3, 4, 5]  # never more windows than frames


def test_uniform_indices_include_both_ends() -> None:
    assert K.uniform_indices(11, 3).tolist() == [0, 5, 10]
    assert K.uniform_indices(2, 5).tolist() == [0, 1]


def test_saved_keyframes_are_read_from_the_video_folder(tmp_path) -> None:
    assert K.saved_keyframes(tmp_path, "steady") is None
    (tmp_path / K.KEYFRAME_FILE).write_text('{"uniform": ["1.0"], "steady": ["2.0", "3.0"]}')

    assert K.saved_keyframes(tmp_path, "steady") == ["2.0", "3.0"]
