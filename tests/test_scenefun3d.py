import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from affordance3d.scenefun3d import (
    evenly_spaced,
    index_sequence,
    load_frames,
    load_sequence,
    match_nearest,
    pick_partners,
    read_annotations,
    read_laser_scan,
    read_pincam,
    read_sky_direction,
    read_traj_poses,
    sky_angle_deg,
    timestamp_of,
    upright_cw_turns,
)


def test_laser_scan_reader_returns_points_and_colours(tmp_path) -> None:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
    cloud.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]])
    path = tmp_path / "scan.ply"
    o3d.io.write_point_cloud(str(path), cloud)

    xyz, rgb = read_laser_scan(path)
    np.testing.assert_allclose(xyz, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    np.testing.assert_allclose(rgb, [[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]], atol=1 / 255)


def test_sky_direction_is_read_from_arkitscenes_metadata(tmp_path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text("video_id,visit_id,sky_direction,fold\n42444490,421069,Left,Training\n")
    assert read_sky_direction(path, "42444490") == "Left"
    with pytest.raises(KeyError, match="123"):
        read_sky_direction(path, "123")


def camera_pose(x_axis: list, y_axis: list, z_axis: list) -> np.ndarray:
    """Camera-to-world pose from the camera axes expressed in a z-up world."""
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    return transform


UPRIGHT = camera_pose([1, 0, 0], [0, 0, -1], [0, 1, 0])  # image down = world down
SKY_LEFT = camera_pose([0, 0, -1], [-1, 0, 0], [0, 1, 0])  # image right points down
SKY_RIGHT = camera_pose([0, 0, 1], [1, 0, 0], [0, 1, 0])  # image right points up
SKY_DOWN = camera_pose([-1, 0, 0], [0, 0, 1], [0, 1, 0])  # upside down


@pytest.mark.parametrize("pose_, angle, turns", [(UPRIGHT, 0.0, 0), (SKY_LEFT, -90.0, 1),
                                                 (SKY_DOWN, 180.0, 2), (SKY_RIGHT, 90.0, 3)])
def test_every_sky_direction_gives_its_clockwise_turns(pose_, angle, turns) -> None:
    assert abs(sky_angle_deg(pose_)) == pytest.approx(abs(angle))
    assert upright_cw_turns([pose_]) == turns


def test_sky_angle_is_zero_for_upright_and_minus_90_when_sky_is_left() -> None:
    assert sky_angle_deg(UPRIGHT) == pytest.approx(0.0)
    assert sky_angle_deg(SKY_LEFT) == pytest.approx(-90.0)


def test_upright_turns_follow_the_majority_of_poses() -> None:
    assert upright_cw_turns([UPRIGHT, UPRIGHT]) == 0
    assert upright_cw_turns([SKY_LEFT, SKY_LEFT, UPRIGHT]) == 1  # sky left -> one clockwise turn


def test_upright_turns_actually_make_the_camera_upright() -> None:
    from affordance3d.frames import Frame
    from affordance3d.geometry import rotate_frame_cw

    frame = Frame(0, np.zeros((4, 6, 3), np.uint8), np.ones((4, 6), np.float32), np.eye(3), SKY_LEFT)
    rotated = rotate_frame_cw(frame, upright_cw_turns([SKY_LEFT]))
    assert sky_angle_deg(rotated.cam_to_world) == pytest.approx(0.0, abs=1e-9)


def write_mini_sequence(root, times: list[str], depth_times: list[str]) -> None:
    """A tiny SceneFun3D-style folder: 6x4 images, identity rotation, camera moving along x."""
    import imageio.v3 as iio

    folder = root / "1" / "2"
    for sub in ("hires_wide", "hires_depth", "hires_wide_intrinsics"):
        (folder / sub).mkdir(parents=True)
    lines = []
    for k, t in enumerate(times):
        iio.imwrite(folder / "hires_wide" / f"2_{t}.jpg", np.full((4, 6, 3), 40 * k, np.uint8))
        (folder / "hires_wide_intrinsics" / f"2_{t}.pincam").write_text("6 4 10 10 2.5 1.5\n")
        lines.append(f"{t} 0 0 0 {-float(k)} 0 0")  # world-to-camera translation -k -> camera at x = k
        if t in depth_times:
            iio.imwrite(folder / "hires_depth" / f"2_{t}.png", np.full((4, 6), 1500, np.uint16))
    (folder / "hires_poses.traj").write_text("\n".join(lines) + "\n")


def test_load_sequence_reads_depth_frames_with_metres_and_camera_positions(tmp_path) -> None:
    write_mini_sequence(tmp_path, ["10.0", "10.5", "11.0"], depth_times=["10.0", "11.0"])
    sample = load_sequence(tmp_path, "1", "2", n_frames=5, cw_turns=0)

    assert sample.timestamps == ["10.0", "11.0"]  # only frames that have depth
    assert sample.n_frames_available == 2
    np.testing.assert_allclose(sample.frames[1].depth_m, 1.5)
    np.testing.assert_allclose(sample.frames[1].cam_to_world[:3, 3], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(sample.frames[0].intrinsics, [[10, 0, 2.5], [0, 10, 1.5], [0, 0, 1]])


def test_frames_without_depth_load_with_empty_depth(tmp_path) -> None:
    write_mini_sequence(tmp_path, ["10.0", "10.5"], depth_times=["10.0"])
    index = index_sequence(tmp_path, "1", "2")
    (frame,) = load_frames(index, ["10.5"], cw_turns=0, pixel_step=1)

    assert index.rgb_times == ["10.0", "10.5"] and index.depth_times == ["10.0"]
    assert not frame.depth_m.any()
    np.testing.assert_allclose(frame.cam_to_world[:3, 3], [1.0, 0.0, 0.0])


def test_subsampled_and_rotated_loading_sees_the_same_world_points(tmp_path) -> None:
    import imageio.v3 as iio

    from affordance3d.geometry import backproject

    write_mini_sequence(tmp_path, ["10.0"], depth_times=["10.0"])
    varying = (1000 + 10 * np.arange(24).reshape(4, 6)).astype(np.uint16)  # non-square, unique depths
    iio.imwrite(tmp_path / "1" / "2" / "hires_depth" / "2_10.0.png", varying)
    index = index_sequence(tmp_path, "1", "2")

    (plain,) = load_frames(index, ["10.0"], cw_turns=0, pixel_step=1)
    (transformed,) = load_frames(index, ["10.0"], cw_turns=1, pixel_step=2)

    assert transformed.image_hw == (3, 2)  # 4x6 -> subsample 2x3 -> rotate 3x2
    expected, _ = backproject(plain, stride=2)
    actual, _ = backproject(transformed)
    order = lambda p: p[np.lexsort(p.T[::-1])]
    np.testing.assert_allclose(order(actual), order(expected), atol=1e-9)


def test_partners_are_the_first_frame_inside_the_time_window() -> None:
    partners = pick_partners(["1.0", "5.0", "9.0"], ["1.1", "1.4", "2.0", "5.1", "9.2"], min_gap=0.3, max_gap=2.0)
    assert partners == ["1.4", None, None]  # 5.0 and 9.0 have no candidate 0.3-2.0 s later


def test_traj_line_is_world_to_camera_and_is_inverted(tmp_path) -> None:
    rotvec = np.array([0.1, -0.2, 0.3])
    translation = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "hires_poses.traj"
    path.write_text(f"123.456 {' '.join(map(str, rotvec))} {' '.join(map(str, translation))}\n")

    poses = read_traj_poses(path)

    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    world_to_camera[:3, 3] = translation
    assert list(poses) == ["123.456"]
    np.testing.assert_allclose(poses["123.456"] @ world_to_camera, np.eye(4), atol=1e-12)


def test_traj_rejects_malformed_lines(tmp_path) -> None:
    path = tmp_path / "bad.traj"
    path.write_text("1.0 0 0 0 1 2\n")
    with pytest.raises(ValueError, match="7 values"):
        read_traj_poses(path)


def test_pincam_gives_matrix_and_size(tmp_path) -> None:
    path = tmp_path / "frame.pincam"
    path.write_text("1920 1440 1500.5 1501.5 959.2 719.8\n")
    intrinsics, (width, height) = read_pincam(path)
    np.testing.assert_allclose(intrinsics, [[1500.5, 0, 959.2], [0, 1501.5, 719.8], [0, 0, 1]])
    assert (width, height) == (1920, 1440)


def test_timestamp_comes_from_the_file_name(tmp_path) -> None:
    assert timestamp_of(tmp_path / "42444490_3044.917.jpg") == "3044.917"


def test_match_nearest_pairs_within_the_gap_only() -> None:
    pairs = match_nearest(["1.000", "2.010", "5.000"], ["1.000", "2.000", "3.000"], max_gap=0.05)
    assert pairs == {"1.000": ("1.000", 0.0), "2.010": ("2.000", pytest.approx(0.01))}


def test_evenly_spaced_keeps_time_order_and_ends() -> None:
    np.testing.assert_array_equal(evenly_spaced(10, 4), [0, 3, 6, 9])
    np.testing.assert_array_equal(evenly_spaced(3, 10), [0, 1, 2])


def test_annotations_drop_excluded_regions(tmp_path) -> None:
    path = tmp_path / "ann.json"
    path.write_text(json.dumps({"annotations": [
        {"annot_id": "a", "label": "rotate", "indices": [3, 1]},
        {"annot_id": "b", "label": "exclude", "indices": [7]},
    ]}))
    (element,) = read_annotations(path)
    assert (element.annot_id, element.label) == ("a", "rotate")
    np.testing.assert_array_equal(element.indices, [3, 1])
