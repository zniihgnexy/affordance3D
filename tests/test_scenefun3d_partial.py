import numpy as np

from affordance3d import scenefun3d_partial as P


def test_stamp_reads_the_timestamp_from_an_archive_member() -> None:
    assert P.stamp("hires_wide/42444755_80886.380.jpg") == "80886.380"


def test_usable_times_need_every_archive_and_nearby_poses_and_lidar() -> None:
    frames = [{"1.000", "2.000", "3.000", "4.000"}, {"1.000", "2.000", "3.000", "4.000"}, {"1.000", "2.000", "4.000"}]
    gt_pose_times = np.array([1.0, 2.0, 3.0, 4.0])
    arkit_times = np.array([1.01, 2.2, 4.0])  # no device pose near 2.0
    lidar_times = np.array([1.0, 2.0, 3.0, 4.5])  # no LiDAR frame near 4.0

    usable = P.usable_times(frames, gt_pose_times, arkit_times, lidar_times)

    assert usable == ["1.000"]  # 3.000 is missing from one archive


def test_choose_keyframes_gives_an_even_set_and_a_steady_set() -> None:
    times = [f"{t}.000" for t in range(6)]
    rates = np.array([9.0, 1.0, 5.0, 7.0, 8.0, 2.0])

    choice = P.choose_keyframes(times, rates, n=2)

    assert choice == {"uniform": ["0.000", "5.000"], "steady": ["1.000", "5.000"]}
