import numpy as np
import pytest

from affordance3d.mem import grounding as G
from affordance3d.mem import policies
from affordance3d.mem import tool_maps as TM


def _labels(parts: dict[int, tuple[int, int, int, int]], hw: tuple[int, int] = (100, 100)) -> np.ndarray:
    """A label map with each part a filled rectangle (left, top, right, bottom)."""
    labels = np.full(hw, -1)
    for part, (left, top, right, bottom) in parts.items():
        labels[top:bottom, left:right] = part
    return labels


def test_the_part_names_are_those_of_the_prompts_definition_of_the_part() -> None:
    assert all(name in G.PART for name in TM.PART_NAMES) and len(TM.PART_NAMES) == 8
    assert all(name in TM.detection_prompt() for name in TM.PART_NAMES)


def test_parse_detections_reads_every_box_with_its_label_and_skips_unreadable_ones() -> None:
    answer = ('```json\n[{"bbox_2d": [100, 200, 300, 400], "label": "Handle"},\n'
              ' {"bbox_2d": [500, 500, 500, 600], "label": "knob"},\n'  # zero width: skipped
              ' {"bbox_2d": [600, 0, 700, 1000], "label": "socket"}, {"bbox_2d": [10, 20')  # cut off: skipped

    found = TM.parse_detections(answer, (480, 640))

    assert found == [((64, 96, 192, 192), "handle"), ((384, 0, 448, 480), "socket")]
    assert TM.parse_detections("[]", (480, 640)) == [] and TM.parse_detections("There is none.", (480, 640)) == []


@pytest.mark.parametrize(("label", "name"), [("handle", "handle"), ("drawer handle", "handle"), ("light switch", "switch"),
                                             ("keyboard key", "key"), ("switches", "switch"), ("door", "other"),
                                             ("keyboard", "other")])
def test_part_name_maps_a_label_to_the_name_it_contains(label: str, name: str) -> None:
    assert TM.part_name(label) == name


def test_mask_box_is_the_bounding_box_of_the_mask() -> None:
    mask = np.zeros((20, 30), bool)
    mask[5:9, 10:14] = True

    assert TM.mask_box(mask) == (10, 5, 14, 9)
    assert TM.mask_box(np.zeros((20, 30), bool)) is None


def test_merge_duplicates_keeps_the_best_of_overlapping_boxes_per_keyframe() -> None:
    detections = [TM.Detection(0, (0, 0, 10, 10), 0.6, "knob"), TM.Detection(0, (1, 0, 11, 10), 0.8, "handle"),
                  TM.Detection(1, (0, 0, 10, 10), 0.7, "handle"),  # another keyframe: kept
                  TM.Detection(0, (50, 50, 60, 60), 0.9, "switch")]

    merged = TM.merge_duplicates(detections)

    assert [(d.keyframe, d.label) for d in merged] == [(0, "switch"), (0, "handle"), (1, "handle")]
    unscored = [TM.Detection(2, (0, 0, 10, 10), None, "a"), TM.Detection(2, (0, 0, 10, 9), None, "b"),
                TM.Detection(2, (30, 0, 40, 10), None, "c")]
    assert [d.label for d in TM.merge_duplicates(unscored)] == ["a", "c"]  # no scores: the tool's order


def test_tool_detections_apply_the_threshold_and_order_by_keyframe() -> None:
    record = {"keyframes": {"3": {"seconds": 1.0, "detections": [{"box": [0, 0, 5, 5], "score": 0.9, "label": "knob"}]},
                            "1": {"seconds": 2.5, "detections": [{"box": [0, 0, 5, 5], "score": 0.4, "label": "knob"},
                                                                 {"box": [9, 9, 20, 20], "score": 0.55, "label": "key"}]}}}

    detections = TM.tool_detections(record)

    assert [(d.keyframe, d.label) for d in detections] == [(1, "key"), (3, "knob")]
    assert TM.tool_seconds(record) == pytest.approx(3.5)
    assert TM.missing_keyframes(record, 4) == [0, 2]
    candidates = TM.candidates_of(detections, (100, 100))
    assert candidates[0].box == (9, 9, 20, 20) and candidates[0].crop[2] - candidates[0].crop[0] >= 11


def test_budget_keyframes_take_every_fourth_and_every_second_keyframe_from_the_first() -> None:
    assert TM.budget_keyframes(40, 10) == list(range(0, 40, 4))
    assert TM.budget_keyframes(40, 20) == list(range(0, 40, 2))
    assert TM.budget_keyframes(40, 40) == list(range(40))
    with pytest.raises(ValueError):
        TM.budget_keyframes(40, 30)


def test_a_box_finds_a_part_when_its_centre_lies_on_the_part_widened_by_eight_pixels() -> None:
    labels = [_labels({3: (40, 40, 50, 50)}), _labels({3: (40, 40, 50, 50), 5: (0, 0, 20, 20)})]
    boxes = {0: [(52, 40, 62, 50)],  # centre (57, 45): 8 px from the part, inside the margin
             1: [(60, 40, 70, 50)]}  # centre (65, 45): 16 px away

    found = TM.sightings(labels, boxes)

    assert found == {(3, 0): True, (3, 1): False, (5, 1): False}


def test_part_outcomes_keep_the_parts_of_all_keyframes_as_denominator_at_every_budget() -> None:
    found = {(1, 0): False, (1, 2): True,  # part 1: missed on keyframe 0, found on keyframe 2
             (2, 0): False, (2, 1): False,  # part 2: always missed
             (3, 1): True}  # part 3: seen on keyframe 1 only

    every_second = TM.part_outcomes(found, [0, 2])
    all_keyframes = TM.part_outcomes(found, [0, 1, 2, 3])

    assert every_second["found"].tolist() == [True, False, False]  # part 3 is not visible with this budget
    assert every_second["visible"].tolist() == [True, True, False]
    assert every_second["always_missed"].tolist() == [False, True, False]
    assert all_keyframes["found"].tolist() == [True, False, True] and all_keyframes["always_missed"].tolist() == [False, True, False]


def test_visible_parts_need_the_phase1_minimum_of_pixels() -> None:
    labels = _labels({1: (0, 0, 10, 10), 2: (50, 50, 53, 53)})  # part 2 has 9 pixels: too few

    assert TM.visible_parts([labels, labels]) == {1: [0, 1]}


def test_false_alarms_count_boxes_whose_centre_is_on_no_part() -> None:
    labels = _labels({1: (40, 40, 50, 50), 2: (0, 0, 10, 10)})
    boxes = [(40, 40, 50, 50), (12, 0, 22, 10), (70, 70, 90, 90)]  # on part 1; centre 8 px from part 2; on nothing

    assert TM.false_alarms(labels, boxes) == 1
    assert TM.false_alarms(labels, []) == 0


def test_on_a_part_keeps_only_candidates_centred_on_an_annotated_part_of_their_keyframe() -> None:
    labels = [_labels({1: (40, 40, 50, 50)}), _labels({})]
    candidates = [policies.Candidate(0, (40, 40, 50, 50), (0, 0, 1, 1)), policies.Candidate(0, (70, 70, 80, 80), (0, 0, 1, 1)),
                  policies.Candidate(1, (40, 40, 50, 50), (0, 0, 1, 1))]

    assert TM.on_a_part(candidates, labels) == [0]


def test_shown_on_target_scores_the_shown_boxes_by_both_rules() -> None:
    labels = [_labels({4: (40, 40, 60, 44)})]
    centres = {0: [(50, 42)]}
    wide = [policies.Candidate(0, (20, 30, 90, 80), (0, 0, 1, 1))]  # holds the part's centre, but its own centre is off the part
    on = [policies.Candidate(0, (44, 38, 54, 46), (0, 0, 1, 1))]

    assert TM.shown_on_target(wide, labels, (4,), centres) == (False, True)
    assert TM.shown_on_target(on, labels, (4,), centres) == (True, True)
    assert TM.shown_on_target([], labels, (4,), centres) == (False, False)


@pytest.mark.parametrize(("recall", "tool"), [
    ({"sam3": (0.50, 0.44, 0.56), "qwen8b": (0.60, 0.55, 0.66)}, "sam3"),  # intervals overlap: a tie, SAM 3 decides
    ({"sam3": (0.40, 0.34, 0.46), "qwen8b": (0.60, 0.55, 0.66)}, "qwen8b"),
    ({"sam3": (0.70, 0.66, 0.75), "qwen8b": (0.60, 0.55, 0.64)}, "sam3"),
])
def test_deciding_tool_is_the_higher_recall_unless_the_intervals_overlap(recall: dict, tool: str) -> None:
    decision = TM.deciding_tool(recall)

    assert decision["tool"] == tool
    assert ("tie" in decision["reason"]) == (recall["sam3"][1] <= recall["qwen8b"][2] and recall["qwen8b"][1] <= recall["sam3"][2])


def test_task_prompt_marks_show_the_frames_of_the_best_boxes_and_their_best_boxes() -> None:
    detections = [TM.Detection(5, (0, 0, 10, 10), 0.9, "task"), TM.Detection(5, (50, 50, 60, 60), 0.6, "task"),
                  TM.Detection(2, (0, 0, 10, 10), 0.8, "task"), TM.Detection(7, (0, 0, 10, 10), 0.55, "task"),
                  TM.Detection(9, (0, 0, 10, 10), 0.52, "task"),  # fourth frame: not shown
                  TM.Detection(3, (0, 0, 10, 10), 0.2, "task")]  # below the threshold

    frames, marks = TM.task_prompt_marks(detections)

    assert frames == [5, 2, 7]
    assert [(d.keyframe, d.score) for d in marks] == [(5, 0.9), (5, 0.6), (2, 0.8), (7, 0.55)]


def test_task_prompt_marks_fall_back_to_all_boxes_when_none_passes_and_cap_the_marks() -> None:
    weak = [TM.Detection(1, (0, 0, 10, 10), 0.1, "task"), TM.Detection(4, (0, 0, 10, 10), 0.3, "task")]
    many = [TM.Detection(0, (3 * i, 0, 3 * i + 2, 2), 0.9 - i / 100, "task") for i in range(50)]

    assert TM.task_prompt_marks(weak)[0] == [4, 1]
    frames, marks = TM.task_prompt_marks(many)
    assert frames == [0] and len(marks) == TM.MAX_MARKS and marks[0].score == pytest.approx(0.9)


def test_task_detections_read_one_task_from_every_frame() -> None:
    record = {"frames": {"0": {"detections": [[{"box": [0, 0, 4, 4], "score": 0.7}], []]},
                         "6": {"detections": [[], [{"box": [1, 1, 5, 5], "score": 0.2}]]}}}

    assert TM.task_detections(record, 1) == [TM.Detection(6, (1, 1, 5, 5), 0.2, "task")]


def test_gate_keyframes_are_the_two_showing_most_parts_in_time_order() -> None:
    one, two = _labels({1: (0, 0, 10, 10)}), _labels({1: (0, 0, 10, 10), 2: (20, 20, 30, 30)})

    assert TM.gate_keyframes([one, two, _labels({}), two]) == [1, 3]
    assert TM.gate_keyframes([_labels({}), one, _labels({})]) == [0, 1]  # ties: the earlier
