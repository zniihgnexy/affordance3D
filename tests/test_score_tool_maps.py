import importlib.util
from pathlib import Path

import numpy as np
import pytest

from affordance3d.mem import policies
from report import mem_steps_helpers as H

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_tool_maps.py"
spec = importlib.util.spec_from_file_location("score_tool_maps", SCRIPT)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def _scene_labels() -> list[np.ndarray]:
    """40 keyframes: part 0 seen in keyframe 1 only, part 1 in keyframes 4 and 5, part 2 in keyframe 8."""
    labels = [np.full((30, 30), -1) for _ in range(40)]
    labels[1][0:5, 0:5] = 0
    labels[4][10:15, 10:15] = 1
    labels[5][10:15, 10:15] = 1
    labels[8][20:25, 20:25] = 2
    return labels


def _record(found_part_1: bool = True) -> dict:
    """A tool that finds part 0 (keyframe 1), part 1 only in keyframe 5, part 2 (keyframe 8) plus a false alarm there."""
    entry = {str(k): {"seconds": 0.5, "detections": []} for k in range(40)}
    entry["1"]["detections"] = [{"box": [0, 0, 5, 5], "score": 0.9, "label": "knob"}]
    entry["5"]["detections"] = [{"box": [10, 10, 15, 15], "score": 0.8 if found_part_1 else 0.1, "label": "handles"}]
    entry["8"]["detections"] = [{"box": [20, 20, 25, 25], "score": 0.7, "label": "switch"},
                                {"box": [0, 20, 4, 24], "score": 0.6, "label": "door"}]
    return {"keyframes": entry}


def test_scene_measures_score_recall_per_budget_false_alarms_and_time() -> None:
    measures = score.scene_measures(_record(), _scene_labels())

    outcomes = measures["outcomes"]
    assert outcomes[40]["found"].tolist() == [True, True, True]
    assert outcomes[20]["found"].tolist() == [False, False, True]  # part 0 is seen in an odd keyframe only
    assert outcomes[20]["visible"].tolist() == [False, True, True] and outcomes[20]["always_missed"].tolist() == [False, True, False]
    assert outcomes[10]["found"].tolist() == [False, False, True]
    assert measures["false_alarms"].sum() == 1 and measures["false_alarms"][8] == 1
    assert measures["boxes"][8] == 2 and measures["seconds"] == pytest.approx(20.0)
    assert measures["sightings_found"].tolist() == [1.0, 0.0, 1.0, 1.0]  # (0, 1), (1, 4), (1, 5), (2, 8)


def _scenes(n: int = 4) -> list[dict]:
    labels = _scene_labels()
    coverage = {"common": np.array([1.0, 0.0]), "phase1": np.array([1.0, 1.0])}
    return [{"visit": str(i), "annotated_coverage": coverage,
             "tools": {"sam3": {**score.scene_measures(_record(), labels), "coverage": None},
                       "qwen8b": {**score.scene_measures(_record(found_part_1=False), labels), "coverage": None}}}
            for i in range(n)]


def test_summary_pools_the_scenes_and_names_the_deciding_tool_only_when_every_map_is_complete() -> None:
    scenes = _scenes()

    summary = score.summarise(scenes, {}, [scene["visit"] for scene in scenes])

    sam3, qwen = summary["tools"]["sam3"], summary["tools"]["qwen8b"]
    assert sam3["parts"] == 12 and sam3["recall"]["40"]["rate"] == pytest.approx(1.0)
    assert qwen["recall"]["40"]["rate"] == pytest.approx(2 / 3)  # its box on part 1 scores below the threshold
    assert sam3["recall"]["10"]["rate"] == pytest.approx(1 / 3) and sam3["recall_gain_40_minus_10"]["rate"] == pytest.approx(2 / 3)
    assert sam3["always_missed"]["10"]["rate"] == pytest.approx(0.5)  # parts 1 and 2 are visible with 10 keyframes
    assert sam3["recall_gain_40_minus_10_among_visible"]["rate"] == pytest.approx(0.5)  # of those two, part 1 is gained
    assert sam3["false_alarms_per_keyframe"]["rate"] == pytest.approx(1 / 40)
    assert sam3["boxes_by_name"] == {"handle": 4, "knob": 4, "switch": 4, "other": 4}
    assert sam3["write_seconds_per_scene"]["median"] == pytest.approx(20.0) and "not yet" in sam3["coverage"]
    assert summary["deciding_tool"]["tool"] == "sam3" and summary["recall_difference"]["rate"] == pytest.approx(1 / 3)
    assert summary["annotated_coverage"]["phase1"]["rate"] == pytest.approx(1.0)
    assert score.summarise(scenes, {"qwen8b": ["9"]}, [*(s["visit"] for s in scenes), "9"])["deciding_tool"] is None


def test_coverage_asks_whether_a_box_shown_by_the_memory_lies_on_the_target() -> None:
    labels = [np.full((60, 60), -1) for _ in range(4)]
    labels[2][20:30, 20:30] = 7
    keyframe_vectors = np.eye(4)
    saved = {"queries": np.array([[0.0, 0.0, 1.0, 0.0]]), "keyframes": keyframe_vectors}
    candidates = [policies.Candidate(2, (20, 20, 30, 30), (0, 0, 1, 1)), policies.Candidate(0, (0, 0, 10, 10), (0, 0, 1, 1))]
    vectors = np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    task = H.Task("open it", (7,), frozenset({2}))

    rows = score.coverage(candidates, vectors, saved, [task], labels)

    assert rows["common"].tolist() == [1.0] and rows["phase1"].tolist() == [1.0]
    off = [policies.Candidate(2, (40, 40, 50, 50), (0, 0, 1, 1)), candidates[1]]
    assert score.coverage(off, vectors, saved, [task], labels)["common"].tolist() == [0.0]
