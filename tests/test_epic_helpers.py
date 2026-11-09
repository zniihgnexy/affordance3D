import numpy as np
from scipy.spatial.transform import Rotation

from report.epic_helpers import (
    Mask,
    Narration,
    VisorFrame,
    colmap_to_cam_to_world,
    frames_per_second,
    gravity_alignment,
    narration_at,
)


def test_colmap_world_to_camera_pose_is_inverted() -> None:
    rotation = Rotation.from_euler("z", 90, degrees=True)
    qx, qy, qz, qw = rotation.as_quat()
    translation = np.array([1.0, 2.0, 3.0])

    cam_to_world = colmap_to_cam_to_world([qw, qx, qy, qz, *translation])

    world_to_cam = np.eye(4)
    world_to_cam[:3, :3], world_to_cam[:3, 3] = rotation.as_matrix(), translation
    assert np.allclose(cam_to_world @ world_to_cam, np.eye(4))


def test_gravity_alignment_uses_the_plane_of_head_positions_not_the_viewing_direction() -> None:
    tilt = Rotation.from_euler("x", 30, degrees=True)  # reconstruction frame is tilted relative to gravity
    rng = np.random.default_rng(0)
    poses = []
    for angle in np.linspace(0, np.pi, 60):  # walking along an arc at head height, always looking down at the counter
        pose = np.eye(4)
        looking_down = Rotation.from_euler("x", -90 - 40, degrees=True)  # pitched 40 degrees below the horizon
        pose[:3, :3] = (tilt * looking_down).as_matrix()
        pose[:3, 3] = tilt.apply([3 * np.cos(angle), 1.5 * np.sin(angle), 1.6 + rng.normal(0, 0.02)])
        poses.append(pose)

    align = gravity_alignment(poses)

    assert np.allclose(align @ tilt.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=0.02)


def test_narration_at_prefers_the_running_action_then_the_latest_finished_one() -> None:
    narrations = [Narration("open cupboard", 100, 200), Narration("take bread", 150, 400), Narration("cut", 600, 700)]

    assert narration_at(narrations, 180) == "take bread"  # latest-started action that is still running
    assert narration_at(narrations, 500) == "take bread"
    assert narration_at(narrations, 50) == ""


def test_frames_per_second_is_derived_from_narration_timestamps() -> None:
    rows = [{"start_frame": "500", "start_timestamp": "00:00:10.00"},
            {"start_frame": "30000", "start_timestamp": "00:10:00.00"}]

    assert frames_per_second(rows) == 50.0


def test_contacts_skip_visor_placeholder_values() -> None:
    frame = VisorFrame(1, None, "s", (
        Mask("left hand", (), "knife"), Mask("right hand", (), "inconclusive"), Mask("knife", (), None)))

    assert frame.contacts == (("left hand", "knife"),)
