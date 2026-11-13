from dataclasses import replace

import numpy as np
import pytest

from affordance3d.mem import action_records as A
from tests.conftest import TEST_HW, frame_from_depth

# The camera sits at the origin and looks along +z: image u grows with world x, v with world y (down).
FRAME = frame_from_depth(np.full(TEST_HW, 2.0))


def _motion(kind: str, motion_dir: list[float], origin: list[float], centre: list[float], orient: str = "outwards",
            reading: str = "viewer") -> A.PartMotion:
    scan = np.array([origin, centre], dtype=float)
    entry = {"motion_type": kind, "motion_dir": motion_dir, "motion_origin_idx": 0, "motion_viz_orient": orient}
    return A.part_motion(entry, scan[1:], scan, reading)


def _labels(blocks: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    """A label map with each part filling a block (top, bottom, left, right)."""
    labels = np.full(TEST_HW, -1, dtype=np.int32)
    for part, (top, bottom, left, right) in blocks.items():
        labels[top:bottom, left:right] = part
    return labels


def test_inwards_is_drawn_against_motion_dir_and_outwards_along_it() -> None:
    assert np.allclose(A.signed_direction([0, 0, 2], "inwards"), [0, 0, -1])
    assert np.allclose(A.signed_direction([0, 0, 2], "outwards"), [0, 0, 1])
    with pytest.raises(ValueError):
        A.signed_direction([0, 0, 1], "sideways")
    with pytest.raises(ValueError):
        A.signed_direction([0, 0, 0], "inwards")


def test_a_button_pushed_inwards_moves_away_from_the_camera_and_a_drawer_comes_towards_it() -> None:
    # motion_dir points out of the wall (towards the camera) in both; the label of the orientation decides the sense
    button = _motion("trans", [0, 0, -1], [0, 0, 2], [0, 0, 2], orient="inwards")
    drawer = _motion("trans", [0, 0, -1], [0, 0, 2], [0, 0, 2], orient="outwards")

    assert A.push_or_pull(button, FRAME) == "push" and A.push_or_pull(drawer, FRAME) == "pull"
    assert np.allclose(button.anchor, [0, 0, 2]) and button.axis_distance_m == 0.0 and not button.hinged


def test_the_documentation_reading_keeps_motion_dir_for_every_translation_and_axis() -> None:
    viewer = _motion("trans", [0, 0, -1], [0, 0, 2], [0, 0, 2], orient="inwards")
    docs = _motion("trans", [0, 0, -1], [0, 0, 2], [0, 0, 2], orient="inwards", reading="docs")
    knob = _motion("rot", [0, 0, -2], [0, 0, 2], [0, 0, 2], orient="inwards", reading="docs")

    assert np.allclose(viewer.direction, [0, 0, 1]) and np.allclose(docs.direction, [0, 0, -1])
    assert np.allclose(knob.direction, [0, 0, -1]) and A.push_or_pull(docs, FRAME) == "pull"
    with pytest.raises(ValueError):
        _motion("trans", [0, 0, -1], [0, 0, 2], [0, 0, 2], reading="sideways")


def test_a_record_in_a_frame_that_hides_every_part_keeps_the_part_it_is_given() -> None:
    drawer = _motion("trans", [0.6, 0, -0.8], [0, 0, 2], [0, 0, 2])

    hidden = A.record_in("open it", (3, 5), ("a", "b"), ("pinch_pull", "hook_turn"), (drawer, None), _labels({}), FRAME,
                         shown_if_hidden=1)
    seen = A.record_in("open it", (3, 5), ("a", "b"), ("pinch_pull", "hook_turn"), (drawer, None), _labels({3: (0, 3, 0, 3)}), FRAME,
                       shown_if_hidden=1)

    assert hidden.shown == 1 and hidden.views[0].pixels == 0 and hidden.views[0].contact is None
    assert hidden.views[0].sense == "pull" and hidden.views[0].arrow is not None  # the 3D motion still projects
    assert seen.shown == 0 and seen.views[0].pixels == 9


def test_projected_direction_is_long_across_the_view_and_short_along_it() -> None:
    across = A.projected_segment(np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0]), FRAME)
    along = A.projected_segment(np.array([0.4, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]), FRAME)
    behind = A.projected_segment(np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, -1.0]), FRAME)

    assert A.segment_px(across) == pytest.approx(100 * A.ARROW_M / 2.0)  # focal 100 px, 0.2 m at 2 m
    assert A.segment_px(along) == pytest.approx(100 * 0.4 / 2.0 - 100 * 0.4 / 2.2)
    assert behind is None and A.segment_px(behind) == 0.0


def test_hinge_side_axis_class_and_opening_tangent_of_a_door_and_a_knob() -> None:
    left_door = _motion("rot", [0, 1, 0], [-0.3, 0.5, 2], [0, 0, 2])  # vertical hinge 0.3 m to the left of the centre
    right_door = _motion("rot", [0, -1, 0], [0.3, 0, 2], [0, 0, 2])
    lid = _motion("rot", [1, 0, 0], [0.2, 0.3, 2], [0, 0, 2])  # horizontal hinge below the centre (v grows downwards)
    knob = _motion("rot", [0, 0, -1], [0.02, 0, 2], [0, 0, 2], orient="inwards")

    assert np.allclose(left_door.foot, [-0.3, 0, 2]) and left_door.axis_distance_m == pytest.approx(0.3) and left_door.hinged
    assert [A.hinge_side(m, FRAME) for m in (left_door, right_door, lid, knob)] == ["left", "right", "bottom", A.ON_PART]
    assert [A.axis_class(m.direction, FRAME) for m in (left_door, lid, knob)] == list(A.AXIS_CLASSES[i] for i in (1, 0, 2))
    # right-hand rule about +y: a point at +x of the axis moves towards -z (the camera)
    assert np.allclose(A.opening_tangent(left_door), [0, 0, -1]) and not A.opening_tangent(knob).any()
    assert A.hinge_side(_motion("trans", [1, 0, 0], [0, 0, 2], [0, 0, 2]), FRAME) is None


def test_build_record_takes_the_keyframe_with_most_pixels_and_its_largest_part() -> None:
    drawer = _motion("trans", [0.6, 0, -0.8], [0, 0, 2], [0, 0, 2])
    maps = [_labels({3: (0, 1, 0, 2)}), _labels({3: (0, 2, 0, 2), 5: (10, 12, 10, 13)})]

    record = A.build_record("open it", (3, 5), ("a", "b"), ("pinch_pull", "hook_turn"), (drawer, None), maps, [FRAME, FRAME])

    assert record.keyframe == 1 and record.shown == 1 and not record.has_all_motions
    assert record.views[1].pixels == 6 and record.views[1].contact == (11, 10) and record.arrow_px == 0.0
    assert record.views[0].sense == "pull" and record.views[0].arrow is not None
    assert A.build_record("gone", (7,), ("c",), ("tip_push",), (None,), maps, [FRAME, FRAME]) is None


def _drawer_and_door() -> tuple[A.ActionRecord, np.ndarray]:
    """A drawer handle (part 0, pulled out and to the right) and a door handle (part 1, hinge on the left)."""
    drawer = _motion("trans", [0.6, 0, -0.8], [-0.2, 0, 2], [-0.2, 0, 2])
    door = _motion("rot", [0, 1, 0], [0.0, 0, 2], [0.3, 0, 2])
    labels = _labels({0: (20, 26, 18, 24), 1: (20, 26, 44, 50)})
    record = A.build_record("open", (0, 1), ("a", "b"), ("hook_pull", "hook_turn"), (drawer, door), [labels], [FRAME])
    return record, labels


def test_the_scorer_gives_the_truth_full_marks_and_a_contrary_answer_none() -> None:
    record, labels = _drawer_and_door()
    truth = A.truth_as_prediction(record)
    other = replace(truth, point=record.views[1 - record.shown].contact)

    right = [A.score(truth, record, labels), A.score(A.truth_as_prediction(replace(record, shown=1 - record.shown)), record, labels)]
    wrong = A.score(A.contrary(truth, TEST_HW), record, labels)

    assert all(s["joint"] and s["contact"] and s["label"] and s["type"] and s["arrow"] for s in right)
    assert {right[0]["sense"], right[1]["axis_class"], right[1]["hinge_side"]} == {True}
    assert right[0]["direction_deg"] == pytest.approx(0.0, abs=1e-6) and right[1]["direction_deg"] == pytest.approx(0.0, abs=1e-6)
    assert not any(value for value in wrong.values() if isinstance(value, bool))
    assert wrong["direction_deg"] == pytest.approx(90.0)
    # a point on the other part is judged against that part: its label differs from the shown part's
    assert A.score(other, record, labels)["label"] is False


def test_the_scorer_reads_rotations_as_lines_and_drops_the_sense_when_only_the_axis_counts() -> None:
    record, labels = _drawer_and_door()
    drawer = A.truth_as_prediction(replace(record, shown=0))
    reversed_drawer = replace(drawer, arrow=(-drawer.arrow[0], -drawer.arrow[1]), direction=-drawer.direction)
    door = A.truth_as_prediction(replace(record, shown=1))
    reversed_door = replace(door, arrow=(-door.arrow[0], -door.arrow[1]))

    signed, lines = A.score(reversed_drawer, record, labels), A.score(reversed_drawer, record, labels, axis_only=True)

    assert signed["arrow"] is False and signed["arrow_deg"] == pytest.approx(180.0) and signed["direction_deg"] == pytest.approx(180.0)
    assert lines["arrow"] is True and lines["sense"] is None and lines["direction_deg"] == pytest.approx(0.0, abs=1e-6)
    assert A.score(reversed_door, record, labels)["arrow"] is True
    assert A.score(replace(drawer, arrow=None), record, labels)["arrow"] is False
    assert A.score(replace(drawer, point=None), record, labels)["contact"] is False


def test_shares_count_only_the_tasks_where_a_field_applies() -> None:
    rows = [{"contact": True, "sense": None, "arrow_deg": 3.0}, {"contact": False, "sense": True, "arrow_deg": None}]

    assert A.shares(rows) == {"contact": {"share": 0.5, "n": 2}, "sense": {"share": 1.0, "n": 1}}
    assert A.shares([]) == {}
    assert A.angle_deg([1, 0], [-1, 0], as_lines=True) == pytest.approx(0.0) and A.angle_deg([1, 0], [0, 2], False) == pytest.approx(90.0)
    # equal directions give exactly 0, also where arccos of the rounded dot product would not
    awkward = np.array([0.1, 0.7, 0.3]) / 3.0
    assert A.angle_deg(awkward, awkward, as_lines=False) == 0.0 and A.angle_deg((-1.7, 0.3), (-1.7, 0.3), as_lines=False) == 0.0
