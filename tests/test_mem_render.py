import numpy as np

from affordance3d.mem.render import look_at, overlay_body, rasterize, splat, to_camera

K = np.array([[50.0, 0, 19.5], [0, 50.0, 19.5], [0, 0, 1]])
HW = (40, 40)


def _square(z: float, half: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def test_rasterized_square_has_its_depth_inside_and_nothing_outside() -> None:
    vertices, faces = _square(z=2.0, half=0.4)  # spans +-10 px around the centre

    depth, shade = rasterize(vertices, faces, K, HW)

    assert np.allclose(depth[15:25, 15:25], 2.0)
    assert np.isinf(depth[0, 0]) and np.isinf(depth[39, 39])
    assert shade[19, 19] > 0.99  # facing the camera (shade uses the direction to each face centre)


def test_the_nearer_surface_wins() -> None:
    far, faces = _square(z=3.0, half=1.2)  # covers the whole image
    near, _ = _square(z=1.0, half=0.1)

    depth, _ = rasterize(np.vstack([far, near]), np.vstack([faces, faces + 4]), K, HW)

    assert np.isclose(depth[19, 19], 1.0)
    assert np.isclose(depth[19, 5], 3.0)


def test_splat_keeps_the_nearest_point_per_pixel() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    colours = np.array([[1.0, 0, 0], [0, 1.0, 0]])

    image, depth = splat(points, colours, K, HW, radius_px=0)

    assert np.allclose(image[20, 20], [0, 1.0, 0]) and np.isclose(depth[20, 20], 1.0)


def test_look_at_puts_the_target_on_the_optical_axis_with_world_up_pointing_up() -> None:
    pose = look_at(np.array([0.0, -3.0, 1.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))

    target, above = to_camera(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]), pose)

    assert np.allclose(target, [0, 0, 3.0])
    assert above[1] < 0  # image y points down


def test_body_behind_the_scene_is_hidden_and_in_front_is_drawn() -> None:
    rgb = np.zeros((1, 2, 3), np.uint8)
    scene = np.array([[1.0, 3.0]])
    body = np.array([[2.0, 2.0]])

    out = overlay_body(rgb, scene, body, np.ones((1, 2)))

    assert out[0, 0].sum() == 0 and out[0, 1].sum() > 0
