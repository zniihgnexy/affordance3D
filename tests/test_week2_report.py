import numpy as np
import pytest

from affordance3d.frames import Frame
from affordance3d.geometry import project
from report import week2_helpers as H
from report import week2_media as M

UP = np.array([0.0, 0.0, 1.0])


def _camera(pose: np.ndarray, hw: tuple[int, int] = (60, 80), focal: float = 50.0) -> Frame:
    intrinsics = np.array([[focal, 0, hw[1] / 2], [0, focal, hw[0] / 2], [0, 0, 1.0]])
    return Frame(0, np.zeros((*hw, 3), np.uint8), np.zeros(hw, np.float32), intrinsics, pose)


def test_look_at_puts_the_target_in_the_image_centre_with_up_at_the_top() -> None:
    pose = M.look_at(np.array([3.0, 0.0, 2.0]), np.zeros(3), UP)
    frame = _camera(pose)
    uv, z, inside = project(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]), frame)
    assert inside.all() and (z > 0).all()
    assert uv[0] == pytest.approx([40.0, 30.0], abs=1e-6)
    assert uv[1, 1] < uv[0, 1]  # a point above the target is drawn higher up


def test_splat_keeps_the_nearest_point_of_a_pixel() -> None:
    pose = M.look_at(np.array([0.0, -5.0, 0.0]), np.zeros(3), UP)
    near, far = np.array([[0.0, -1.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])
    image, depth = M.splat(np.vstack([far, near]), np.array([[255, 0, 0], [0, 0, 255]], np.uint8),
                           _camera(pose).intrinsics, pose, (60, 80))
    assert tuple(image[30, 40]) == (0, 0, 255)
    assert depth[30, 40] == pytest.approx(4.0)
    assert depth[0, 0] == 0  # nothing drawn there


def test_a_pulled_part_moves_out_of_its_surface_and_a_pressed_one_into_it() -> None:
    contact, normal = np.zeros(3), np.array([0.0, -1.0, 0.0])
    pulled = H.contact_path(contact, normal, "hook_pull", "trans", np.array([0.0, 1.0, 0.0]), contact, 60, 0.3)
    pressed = H.contact_path(contact, normal, "key_press", "trans", np.array([0.0, -1.0, 0.0]), contact, 60, 0.3)
    assert pulled[-1] == pytest.approx([0.0, -0.3, 0.0])
    assert pressed[-1] == pytest.approx([0.0, 0.3, 0.0])


def test_a_door_handle_swings_out_of_the_door_on_a_circle_round_the_hinge() -> None:
    contact, normal = np.array([0.6, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])  # hinge: the vertical through the origin
    path = H.contact_path(contact, normal, "hook_pull", "rot", UP, np.zeros(3), 60, 0.3)
    radii = np.linalg.norm(path[:, :2], axis=1)
    assert radii == pytest.approx(np.full(len(path), 0.6))
    assert path[1, 1] < 0  # its first step leaves the door towards the viewer
    assert np.degrees(np.arctan2(-path[-1, 1], path[-1, 0])) == pytest.approx(60.0)
