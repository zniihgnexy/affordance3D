import numpy as np

from affordance3d.scenefun3d import FunctionalElement
from report.report_helpers import CANDIDATES, CAPABILITIES, MARKS, MapProjection, pick_task_examples


def test_map_projection_keeps_aspect_centres_content_and_puts_north_up() -> None:
    projection = MapProjection.fit(np.array([[0.0, 0.0], [2.0, 1.0]]), size=100, margin=10)

    corners = projection.to_px(np.array([[0.0, 0.0], [2.0, 1.0]]))

    assert projection.scale == 40.0  # the longer side (2 m) fills 80 px
    assert corners.tolist() == [[10, 70], [90, 30]]  # 1 m side centred; larger y is higher (smaller v)


def test_task_examples_cover_types_first_and_skip_unseen_parts() -> None:
    elements = [FunctionalElement("a", "rotate", np.array([0])), FunctionalElement("b", "rotate", np.array([1])),
                FunctionalElement("c", "key_press", np.array([2])), FunctionalElement("d", "unplug", np.array([3]))]
    descriptions = [("turn a", ["a"]), ("turn b", ["b"]), ("press c", ["c"]), ("pull d", ["d"])]

    chosen = pick_task_examples(descriptions, elements, seen=np.array([True, True, True, False]), n=3)

    assert chosen == [("turn a", 0), ("press c", 2), ("turn b", 1)]


def test_every_candidate_marks_each_capability() -> None:
    for candidate in CANDIDATES:
        assert len(candidate.marks) == len(CAPABILITIES)
        assert set(candidate.marks) <= set(MARKS)
