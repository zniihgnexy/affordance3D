import numpy as np

from affordance3d.mem.humanoid import canonical_to_world, feet_on_floor

UP = np.array([0.0, 0.0, 1.0])


def test_canonical_axes_map_front_to_facing_and_up_to_up() -> None:
    rotation = canonical_to_world(facing=np.array([0.0, 2.0, 0.3]), up=UP)

    assert np.allclose(rotation @ [0, 0, 1], [0, 1, 0])  # front, horizontal
    assert np.allclose(rotation @ [0, 1, 0], UP)
    assert np.allclose(rotation @ [1, 0, 0], [-1, 0, 0])  # facing +y, the person's left is -x
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_feet_on_floor_moves_the_pelvis_over_the_stand_point_and_the_soles_to_the_floor() -> None:
    vertices = np.array([[0.0, 0.0, -0.9], [0.0, 0.0, 0.8]])  # soles at -0.9, head at 0.8
    pelvis = np.array([0.0, 0.0, 0.0])

    shift = feet_on_floor(vertices, pelvis, stand=np.array([2.0, 1.0, 0.1]), floor_z=0.1, up=UP)

    assert np.allclose(pelvis + shift, [2.0, 1.0, 1.0])
    assert np.isclose(np.min((vertices + shift) @ UP), 0.1)
