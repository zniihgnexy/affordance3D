import numpy as np
import pytest

from affordance3d.mem import action_records as A
from affordance3d.mem import part_vectors as V
from tests.conftest import TEST_K, frame_from_depth

# The camera sits at the origin and looks along +z: image u grows with world x, v with world y (down).
CONFIG = V.HingeConfig(plane_tol_m=0.02, gap_px=2, door_asymmetry=0.5)
UP = np.array([0.0, -1.0, 0.0])
HW = (96, 160)  # large enough that the normal's window (24 px) around an off-centre handle stays on the door
DOOR_COLUMNS = (10, 141)  # the door's flat front, 2 m away, from top to bottom; the room behind it is 3 m away
PART = 7


def _door(handle_columns: tuple[int, int], door_columns: tuple[int, int] = DOOR_COLUMNS):
    depth = np.full(HW, 3.0)
    depth[:, door_columns[0]:door_columns[1]] = 2.0
    labels = np.full(HW, -1, dtype=np.int32)
    labels[45:52, handle_columns[0]:handle_columns[1]] = PART
    return frame_from_depth(depth), labels


def _x_of_column(column: float, depth_m: float = 2.0) -> float:
    return (column - TEST_K[0, 2]) / TEST_K[0, 0] * depth_m


def test_line_angle_ignores_the_sign() -> None:
    assert V.line_angle_deg(np.array([1.0, 0, 0]), np.array([-2.0, 0, 0])) == pytest.approx(0.0)
    assert V.line_angle_deg(np.array([1.0, 0, 0]), np.array([0.0, 0, 3])) == pytest.approx(90.0)
    assert V.line_angle_deg(np.array([1.0, 1, 0]), np.array([1.0, 0, 0])) == pytest.approx(45.0)


def test_point_line_distance() -> None:
    assert V.point_line_distance(np.array([0.0, 2, 5]), np.zeros(3), np.array([0.0, 0, 1])) == pytest.approx(2.0)


def test_fuse_takes_the_median_contact_and_turns_normals_to_agree() -> None:
    first = V.Observation(0, (1, 1), 30, np.array([0.0, 0, 2]), np.array([0.0, 0, -1]))
    second = V.Observation(1, (1, 1), 20, np.array([0.2, 0, 2]), np.array([0.0, 0, 1]))
    third = V.Observation(2, (1, 1), 10, np.array([0.1, 0, 2]), np.array([0.0, 0, -1]))
    contact, normal = V.fuse([first, second, third])
    assert np.allclose(contact, [0.1, 0, 2])
    assert np.allclose(normal, [0, 0, -1])


def test_a_handle_near_one_edge_of_a_door_turns_about_the_far_edge() -> None:
    frame, labels = _door((108, 112))
    seen = V.observe(PART, [frame], [labels])
    found = V.hinge(frame, labels, PART, seen[0].pixel, seen[0].contact, seen[0].normal, CONFIG)
    assert found is not None and not found.at_border
    assert found.asymmetry > CONFIG.door_asymmetry
    assert V.line_angle_deg(found.axis, UP) == pytest.approx(0.0, abs=1e-6)
    assert found.origin[0] == pytest.approx(_x_of_column(DOOR_COLUMNS[0]), abs=0.03)


def test_a_surface_that_leaves_the_image_is_flagged() -> None:
    frame, labels = _door((108, 112), door_columns=(0, 141))
    seen = V.observe(PART, [frame], [labels])
    found = V.hinge(frame, labels, PART, seen[0].pixel, seen[0].contact, seen[0].normal, CONFIG)
    assert found is not None and found.at_border


def test_level_scan_runs_across_the_door_to_both_edges() -> None:
    frame, labels = _door((108, 112))
    seen = V.observe(PART, [frame], [labels])
    ends = V.level_scan(frame, labels, PART, seen[0].pixel, seen[0].contact, seen[0].normal, UP, CONFIG)
    assert ends is not None and not any(border for _, border in ends)
    xs = sorted(end[0] for end, _ in ends)
    assert xs[0] == pytest.approx(_x_of_column(DOOR_COLUMNS[0]), abs=0.03)
    assert xs[1] == pytest.approx(_x_of_column(DOOR_COLUMNS[1] - 1), abs=0.03)


def test_geometric_rule_calls_an_off_centre_pulled_handle_a_door_and_c1_a_drawer() -> None:
    frame, labels = _door((108, 112))
    seen = V.observe(PART, [frame], [labels])
    door = V.part_vector(PART, "hook_pull", seen, [frame], [labels], UP, "geometric", "best", CONFIG)
    drawer = V.part_vector(PART, "hook_pull", seen, [frame], [labels], UP, "c1", "best", CONFIG)
    assert door.kind == "rot" and V.line_angle_deg(door.axis, UP) == pytest.approx(0.0, abs=1e-6)
    assert drawer.kind == "trans" and V.line_angle_deg(drawer.axis, np.array([0.0, 0, 1])) == pytest.approx(0.0, abs=1.0)
    assert np.allclose(drawer.origin, drawer.contact)


def test_geometric_rule_calls_a_centred_pulled_handle_a_drawer() -> None:
    frame, labels = _door((72, 78))
    seen = V.observe(PART, [frame], [labels])
    vector = V.part_vector(PART, "hook_pull", seen, [frame], [labels], UP, "geometric", "best", CONFIG)
    assert vector.hinge is not None and vector.hinge.asymmetry < CONFIG.door_asymmetry
    assert vector.kind == "trans"


def test_oracle_kind_takes_the_annotated_type() -> None:
    frame, labels = _door((72, 78))
    seen = V.observe(PART, [frame], [labels])
    vector = V.part_vector(PART, "hook_pull", seen, [frame], [labels], UP, "oracle_kind", "best", CONFIG, true_kind="rot")
    assert vector.kind == "rot" and vector.hinge is not None


def test_unknown_rule_is_refused() -> None:
    frame, labels = _door((108, 112))
    seen = V.observe(PART, [frame], [labels])
    with pytest.raises(ValueError):
        V.part_vector(PART, "hook_pull", seen, [frame], [labels], UP, "magic", "best", CONFIG)


def _truth(kind: str, direction: list[float], origin: list[float]) -> A.PartMotion:
    return A.PartMotion(kind, A.unit(direction), np.array(origin, dtype=float), np.array(origin, dtype=float), "outwards")


def _vector(kind: str, axis: list[float], origin: list[float]) -> V.PartVector:
    return V.PartVector(0, "rotate", kind, np.zeros(3), np.array([0.0, 0, -1]), np.array(axis, dtype=float),
                        np.array(origin, dtype=float), 0, (0,), None)


def test_score_a_rotation_needs_type_axis_line_and_origin() -> None:
    truth = _truth("rot", [0, 1, 0], [1, 0, 2])
    assert V.score_part(_vector("rot", [0, -1, 0], [1, 5, 2]), truth, 15, 0.25) == V.PartScore(0.0, 0.0, True, True, True)
    far = V.score_part(_vector("rot", [0, 1, 0], [1.5, 0, 2]), truth, 15, 0.25)
    assert far.ma and not far.mao
    tilted = V.score_part(_vector("rot", [0, 1, 1], [1, 0, 2]), truth, 15, 0.25)
    assert tilted.m and not tilted.ma
    wrong_type = V.score_part(_vector("trans", [0, 1, 0], [1, 0, 2]), truth, 15, 0.25)
    assert not wrong_type.m and not wrong_type.mao


def test_score_a_translation_has_no_origin() -> None:
    score = V.score_part(_vector("trans", [0, 0, 1], [9, 9, 9]), _truth("trans", [0, 0, -1], [0, 0, 2]), 15, 0.25)
    assert score.origin_m is None and score.mao
