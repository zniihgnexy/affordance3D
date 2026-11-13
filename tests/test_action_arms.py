from dataclasses import replace

import numpy as np
import pytest

from affordance3d.mem import action_arms as C
from affordance3d.mem import action_records as A
from affordance3d.mem.tokens import ACTIONS
from tests.conftest import TEST_HW, TEST_K, frame_from_depth

# The camera sits at the origin and looks along +z: image u grows with world x, v with world y (down), so "up" is -y.
WALL = frame_from_depth(np.full(TEST_HW, 2.0))
UP = np.array([0.0, -1.0, 0.0])


def _tilted() -> np.ndarray:
    """Depth of the plane z = 2 + 0.5 x, seen by the test camera."""
    rows, cols = np.mgrid[0:TEST_HW[0], 0:TEST_HW[1]]
    ray_x = (cols - TEST_K[0, 2]) / TEST_K[0, 0]
    return 2.0 / (1.0 - 0.5 * ray_x)


def _motion(kind: str, motion_dir: list[float], origin: list[float], centre: list[float], orient: str = "outwards") -> A.PartMotion:
    scan = np.array([origin, centre], dtype=float)
    entry = {"motion_type": kind, "motion_dir": motion_dir, "motion_origin_idx": 0, "motion_viz_orient": orient}
    return A.part_motion(entry, scan[1:], scan, reading="docs")


def _labels(blocks: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    labels = np.full(TEST_HW, -1, dtype=np.int32)
    for part, (top, bottom, left, right) in blocks.items():
        labels[top:bottom, left:right] = part
    return labels


def _drawer_and_door() -> tuple[A.ActionRecord, np.ndarray]:
    """A drawer handle (part 0, pulled out towards the camera and to the right) and a door handle (part 1, hinge on the left)."""
    drawer = _motion("trans", [0.6, 0, -0.8], [-0.2, 0, 2], [-0.2, 0, 2])
    door = _motion("rot", [0, 1, 0], [0.0, 0, 2], [0.3, 0, 2])
    labels = _labels({0: (20, 26, 18, 24), 1: (20, 26, 44, 50)})
    return A.record_in("open", (0, 1), ("a", "b"), ("hook_pull", "hook_turn"), (drawer, door), labels, WALL), labels


def test_the_question_names_the_task_and_every_label() -> None:
    prompt = C.action_prompt("Open the drawer")

    assert '"Open the drawer"' in prompt and all(label in prompt for label in C.LABEL_HINTS)
    assert len(C.LABEL_HINTS) == 9 and set(C.PUSH_LABELS + C.PULL_LABELS + ("rotate", "hook_turn")) == set(C.LABEL_HINTS)


def test_the_answer_is_read_through_fences_prose_and_loose_wording() -> None:
    text = ('Sure.\n```json\n{"point": [500, 250], "label": "Tip push", "motion": "push", "arrow_end": [520.5, 260], '
            '"hinge_side": null}\n```')
    wrong = '{"point": [1200, 5], "label": "grab", "motion": "wiggle", "arrow_end": [1, 2, 3], "hinge_side": "behind"}'

    assert C.parse_action_answer(text) == C.ActionAnswer((500.0, 250.0), "tip_push", "push", (520.5, 260.0), A.ON_PART)
    assert C.parse_action_answer(wrong) == C.ActionAnswer(None, None, None, None, None)
    assert C.parse_action_answer("I cannot see it.") == C.ActionAnswer(None, None, None, None, None)
    loose = C.parse_action_answer("{'label': 'hook-turn', 'motion': 'turn', 'hinge_side': 'Left', 'point': [0, 1000]}")
    assert (loose.label, loose.motion, loose.hinge_side, loose.point) == ("hook_turn", "turn", "left", (0.0, 1000.0))


def test_relative_points_become_pixels_inside_the_image() -> None:
    assert C.to_pixel((0, 0), TEST_HW) == (0, 0)
    assert C.to_pixel((1000, 1000), TEST_HW) == (TEST_HW[1] - 1, TEST_HW[0] - 1)
    assert C.to_pixel((500, 500), TEST_HW) == (32, 24)


def test_an_arrow_is_lifted_with_the_depth_at_both_ends() -> None:
    assert np.allclose(C.lifted_direction(WALL, (31, 23), (41, 23)), [1, 0, 0])
    assert C.lifted_direction(WALL, (31, 23), (31, 23)) is None
    holey = frame_from_depth(np.where(np.arange(TEST_HW[1]) < 40, 2.0, 0.0)[None].repeat(TEST_HW[0], 0))
    assert C.lift(holey, (45, 10)) is None and C.lifted_direction(holey, (31, 23), (45, 23)) is None


def test_the_depth_normal_faces_the_camera_on_a_flat_and_a_tilted_wall() -> None:
    tilted = frame_from_depth(_tilted())

    assert np.allclose(C.depth_normal(WALL, (31, 23)), [0, 0, -1])
    assert np.allclose(C.depth_normal(tilted, (31, 23)), A.unit([0.5, 0, -1]), atol=1e-4)  # depth is float32
    assert C.depth_normal(frame_from_depth(np.zeros(TEST_HW)), (31, 23)) is None


def test_the_rule_pushes_in_pulls_out_turns_knobs_about_the_normal_and_handles_about_the_vertical() -> None:
    normal = np.array([0.0, 0.0, -1.0])  # facing the camera

    push, pull = C.rule_motion("key_press", normal, UP), C.rule_motion("unplug", normal, UP)
    knob, handle = C.rule_motion("rotate", normal, UP), C.rule_motion("hook_turn", normal, UP)

    assert (push.kind, push.sense) == ("trans", "push") and np.allclose(push.direction, [0, 0, 1])
    assert (pull.kind, pull.sense) == ("trans", "pull") and np.allclose(pull.direction, [0, 0, -1])
    assert knob.kind == "rot" and np.allclose(knob.direction, normal) and np.allclose(handle.direction, UP)
    assert C.rule_motion("foot_push", None, UP) == C.RuleMotion("trans", "push", None)
    assert C.rule_motion(None, normal, UP) == C.RuleMotion(None, None, None)


def test_predictions_of_v_and_of_the_rule() -> None:
    answer = C.ActionAnswer((500, 500), "pinch_pull", "pull", (700, 500), "left")

    v = C.answer_prediction(answer, WALL)
    g = C.rule_prediction(v.point, v.label, WALL, UP)

    assert (v.point, v.kind, v.sense, v.arrow, v.hinge_side) == ((32, 24), "trans", "pull", (12, 0), "left")
    assert np.allclose(v.direction, [1, 0, 0])
    assert (g.point, g.kind, g.sense, g.hinge_side) == ((32, 24), "trans", "pull", None) and np.allclose(g.direction, [0, 0, -1])
    assert g.arrow is not None
    assert C.answer_prediction(C.ActionAnswer(None, None, None, None, None), WALL) == A.Prediction(None, None, None, None, None, None, None, None)


def test_the_memory_region_label_comes_from_the_region_with_that_box() -> None:
    labels = _labels({0: (5, 15, 5, 15), 1: (30, 40, 40, 50)})
    actions = C.part_actions(["tip_push", "hook_pull"])

    assert actions == [ACTIONS.index("tip_push"), ACTIONS.index("hook_pull")] and C.part_actions(["exclude"]) == [-1]
    assert C.region_label(labels, actions, WALL.depth_m, (5, 5, 15, 15)) == ("tip_push", True)
    assert C.region_label(labels, actions, WALL.depth_m, (38, 28, 52, 42)) == ("hook_pull", False)
    assert C.region_label(labels, actions, WALL.depth_m, (20, 20, 25, 25)) == (None, False)


def test_the_truth_scores_full_marks_and_no_answer_scores_nothing() -> None:
    record, labels = _drawer_and_door()
    drawer, door = A.truth_as_prediction(record), A.truth_as_prediction(replace(record, shown=1))

    right = [C.score_arm(p, record, labels, WALL, UP, answers_hinge=True) for p in (drawer, door)]
    empty = [C.score_arm(None, replace(record, shown=s), labels, WALL, UP, answers_hinge=True) for s in (0, 1)]

    for score in right:
        assert all(score[f] for f in ("contact", "label", "type", "line_ok", "joint")) and score["line_deg"] == 0.0
    assert right[0]["sense"] and right[0]["signed_deg"] == 0.0 and right[1]["axis_class"] and right[1]["hinge_side"]
    assert not any(v for s in empty for k, v in s.items() if isinstance(v, bool) and not k.startswith("vertical"))
    assert C.score_arm(None, replace(record, shown=1), labels, WALL, UP, answers_hinge=False)["hinge_side"] is None


def test_the_joint_success_needs_a_line_within_20_degrees_and_ignores_the_sign() -> None:
    record, labels = _drawer_and_door()
    truth = A.truth_as_prediction(record)

    def turned(degrees: float, sign: float = 1.0) -> dict:
        a = np.radians(degrees)  # rotate the true direction about the y axis
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        return C.score_arm(replace(truth, direction=sign * rotation @ truth.direction), record, labels, WALL, UP, True)

    assert turned(15)["joint"] and not turned(25)["joint"] and turned(25)["line_deg"] == pytest.approx(25)
    reversed_ = turned(0, -1.0)
    assert reversed_["joint"] and reversed_["signed_deg"] == pytest.approx(180)
    assert not C.score_arm(replace(truth, kind="rot"), record, labels, WALL, UP, True)["joint"]


def test_the_always_vertical_floor_is_scored_on_rotations_only() -> None:
    record, labels = _drawer_and_door()
    knob = _motion("rot", [0, 0, -1], [0.3, 0, 2], [0.3, 0, 2], orient="inwards")
    knob_record = A.record_in("turn", (1,), ("b",), ("rotate",), (knob,), labels, WALL)

    door = C.score_arm(None, replace(record, shown=1), labels, WALL, UP, False)
    turning = C.score_arm(None, knob_record, labels, WALL, UP, False)
    sliding = C.score_arm(None, record, labels, WALL, UP, False)

    assert door["vertical_axis_class"] and door["vertical_line_ok"]
    assert turning["vertical_axis_class"] is False and turning["vertical_line_ok"] is False
    assert sliding["vertical_axis_class"] is None and sliding["sense"] is False
    with pytest.raises(ValueError):
        C.score_arm(None, replace(record, motions=(None, record.motions[1])), labels, WALL, UP, False)
