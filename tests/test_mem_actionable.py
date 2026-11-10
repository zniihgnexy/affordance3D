import numpy as np

from affordance3d.mem.actionable import (
    axis_angle_deg,
    distance_to_part,
    floor_height,
    is_free,
    plan_stand,
    surface_normal,
)

UP = np.array([0.0, 0.0, 1.0])


def test_surface_normal_of_a_wall_points_towards_the_camera() -> None:
    rng = np.random.default_rng(0)
    wall = np.column_stack([rng.normal(0, 0.001, 200), rng.uniform(-0.05, 0.05, 200), rng.uniform(0.95, 1.05, 200)])

    normal = surface_normal(wall, viewer=np.array([2.0, 0.0, 1.5]))

    assert np.allclose(normal, [1.0, 0.0, 0.0], atol=0.02)


def test_floor_height_ignores_a_few_spurious_low_points() -> None:
    heights = np.concatenate([np.full(1000, 0.02), np.linspace(0.1, 2.5, 3000), [-1.5, -1.2]])

    assert abs(floor_height(heights) - 0.02) < 0.01


def test_stand_point_is_on_the_floor_in_front_of_the_part_and_faces_it() -> None:
    stand = plan_stand(contact=np.array([0.0, 0.0, 1.0]), normal=np.array([1.0, 0.0, 0.0]), up=UP,
                       floor_z=0.0, viewer=np.array([2.0, 0.0, 1.5]), occupied=np.empty((0, 3)), distance=0.5)

    assert np.allclose(stand.position, [0.5, 0.0, 0.0])
    assert np.allclose(stand.facing, [-1.0, 0.0, 0.0])
    assert stand.free


def test_a_part_facing_up_is_approached_from_the_viewer_side() -> None:
    stand = plan_stand(contact=np.array([0.0, 0.0, 0.9]), normal=UP, up=UP, floor_z=0.0,
                       viewer=np.array([0.0, -2.0, 1.5]), occupied=np.empty((0, 3)), distance=0.5)

    assert np.allclose(stand.position, [0.0, -0.5, 0.0])


def test_a_blocked_stand_point_is_moved_to_the_nearest_free_alternative() -> None:
    blocker = np.column_stack([np.full(50, 0.5), np.zeros(50), np.linspace(0.2, 1.5, 50)])  # a pole at x=0.5

    stand = plan_stand(contact=np.array([0.0, 0.0, 1.0]), normal=np.array([1.0, 0.0, 0.0]), up=UP,
                       floor_z=0.0, viewer=np.array([2.0, 0.0, 1.5]), occupied=blocker, distance=0.5)

    assert stand.free
    assert stand.moved
    assert np.linalg.norm(stand.position[:2] - [0.5, 0.0]) > 0.2


def test_is_free_counts_only_points_inside_the_body_cylinder() -> None:
    occupied = np.array([[0.05, 0.0, 1.0], [0.05, 0.0, 0.02], [0.6, 0.0, 1.0]])  # inside / below knees / outside

    assert not is_free(np.array([0.0, 0.0, 0.0]), occupied, UP, max_points=0)
    assert is_free(np.array([0.0, 0.0, 0.0]), occupied[1:], UP, max_points=0)


def test_part_distance_and_undirected_axis_angle() -> None:
    part = np.array([[0.0, 0.0, 1.0], [0.0, 0.02, 1.0]])

    assert np.isclose(distance_to_part(np.array([0.03, 0.0, 1.0]), part), 0.03)
    assert np.isclose(axis_angle_deg(np.array([1.0, 0, 0]), np.array([-2.0, 0, 0])), 0.0)
    assert np.isclose(axis_angle_deg(np.array([1.0, 0, 0]), np.array([0, 3.0, 0])), 90.0)
