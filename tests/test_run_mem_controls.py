import importlib.util
from pathlib import Path

import numpy as np
import pytest

from affordance3d.mem import policies
from report import mem_phase1_helpers as R
from report import mem_steps_helpers as H
from tests.conftest import frame_from_depth

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_mem_controls.py"
spec = importlib.util.spec_from_file_location("run_mem_controls", SCRIPT)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_all_grids_cuts_every_phase1_level_and_orients_it_by_the_long_side() -> None:
    upright = run.all_grids(2, (120, 90))
    lying = run.all_grids(2, (90, 120))

    assert set(upright) == {f"{r}x{c}" for r, c in policies.GRID_SHAPES}
    assert sum(len(cells) for cells in upright.values()) == 2 * sum(r * c for r, c in policies.GRID_SHAPES)
    tall_cell = upright["2x1"][0].box  # two rows on an upright frame: each cell is half the height
    assert (tall_cell[3] - tall_cell[1], tall_cell[2] - tall_cell[0]) == (60, 90)
    wide_cell = lying["2x1"][0].box  # on a lying frame the two cells sit side by side
    assert (wide_cell[3] - wide_cell[1], wide_cell[2] - wide_cell[0]) == (90, 60)


def test_p3_indices_repeats_the_draw_of_the_phase1_runner() -> None:
    image_hw = (100, 80)
    rng = np.random.default_rng(3)
    p4 = [policies.Candidate(int(k), (0, 0, int(w), int(w)), (0, 0, 0, 0)) for k, w in zip(rng.integers(5, size=6), rng.integers(4, 20, size=6))]
    pool = [policies.Candidate(int(k), (0, 0, int(w), int(h)), (0, 0, 1, 1))
            for k, w, h in zip(rng.integers(5, size=40), rng.integers(2, 30, size=40), rng.integers(2, 30, size=40))]

    indices = run.p3_indices(p4, pool, 5, image_hw, seed=2)

    expected_rng = np.random.default_rng(R.RUNNER_SEED + 2)
    policies.random_candidates(p4, len(p4), 5, image_hw, expected_rng)
    expected = policies.size_matched(pool, p4, len(p4), expected_rng)
    assert [pool[i] for i in indices] == expected and len(set(indices)) == len(p4)


def test_annotated_on_keeps_the_regions_of_the_shown_keyframes_in_order_and_caps_them() -> None:
    regions = [policies.Candidate(k, (k, 0, k + 1, 1), (0, 0, 0, 0)) for k in (0, 2, 2, 5, 7)]

    assert [c.keyframe for c in run.annotated_on([2, 7], regions)] == [2, 2, 7]
    many = [policies.Candidate(1, (i, 0, i + 1, 1), (0, 0, 0, 0)) for i in range(run.MAX_MARKS + 5)]
    assert len(run.annotated_on([1], many)) == run.MAX_MARKS and run.annotated_on([1], many)[0].box == (0, 0, 1, 1)


def test_best_keyframes_without_the_memory_ranks_by_whole_keyframe_meaning() -> None:
    keyframes = np.eye(4)[:, :3] + 0.01
    keyframes = keyframes / np.linalg.norm(keyframes, axis=1, keepdims=True)
    query = np.array([0.0, 1.0, 0.2]) / np.linalg.norm([0.0, 1.0, 0.2])

    assert run.best_keyframes(query, keyframes)[0] == 1
    # a token on keyframe 3 that matches the query lifts it to the top once the memory is used
    with_memory = run.best_keyframes(query, keyframes, np.array([query, -query]), np.array([3, 0]))
    assert with_memory[0] == 3


class FakeReader:
    """Answers every question with the same text and remembers how many images it was shown."""

    def __init__(self, answer: str = "1") -> None:
        self.answer, self.images = answer, []

    def ask(self, key: str, images: list, prompt: str, **options: object) -> dict:
        self.images.append(len(images))
        return {"answer": self.answer, "seconds": 0.4}


def _scene() -> tuple[list, list[str], list[np.ndarray], H.Task]:
    """Four keyframes; the target (part 5) lies in keyframe 1."""
    frames = [frame_from_depth(np.full((48, 64), 2.0), index=k) for k in range(4)]
    labels = [np.full((48, 64), -1) for _ in range(4)]
    labels[1][10:20, 10:20] = 5
    return frames, ["t0", "t1", "t2", "t3"], labels, H.Task("press it", (5,), frozenset({1}))


def _tool_memory(labels: list[np.ndarray]) -> dict:
    candidates = [policies.Candidate(0, (2, 2, 8, 8), (0, 0, 9, 9)), policies.Candidate(1, (10, 10, 20, 20), (0, 0, 30, 30)),
                  policies.Candidate(1, (30, 30, 40, 40), (20, 20, 48, 48)), policies.Candidate(3, (5, 5, 10, 10), (0, 0, 9, 9))]
    vectors = np.eye(4)[[0, 1, 2, 3]]  # only the box on the target matches the query (axis 1)
    return {"candidates": candidates, "vectors": vectors, "on_part": run.TM.on_a_part(candidates, labels)}


def test_the_tool_arms_are_named_explicitly_and_the_default_arms_stay_as_they_were() -> None:
    assert run.ARMS == ("p4n", "c1", "c2", "c3", "b", "d", "cplus", "c4", "c1s", "c2s")
    assert set(run.TOOL_ARMS) == {"p4t_sam3", "p4t_qwen8b", "c4t_sam3", "c4t_qwen8b", "p4tp_sam3", "p4tp_qwen8b", "sam3q"}


def test_tool_c4_numbers_the_tool_boxes_on_the_keyframes_chosen_without_the_memory() -> None:
    frames, times, labels, task = _scene()
    memory, query, keyframe_vectors = _tool_memory(labels), np.eye(4)[1], np.eye(4)

    row = run.tool_arm(FakeReader("1"), "c4t_sam3", task, query, frames, times, keyframe_vectors, memory, labels)

    assert row["keyframes_shown"] == ["t1", "t0", "t2"] and row["marks"] == 3  # keyframe 3's box is not shown
    assert row["hit"] and row["box"] == [10, 10, 20, 20]  # number 1 is the first box of the first keyframe shown


def test_tool_p4_ranks_with_the_tool_boxes_and_the_diagnostic_arm_keeps_only_boxes_on_a_part() -> None:
    frames, times, labels, task = _scene()
    memory, query, keyframe_vectors = _tool_memory(labels), np.eye(4)[1], np.eye(4)

    p4t = run.tool_arm(FakeReader("2"), "p4t_qwen8b", task, query, frames, times, keyframe_vectors, memory, labels)
    p4tp = run.tool_arm(FakeReader("1"), "p4tp_qwen8b", task, query, frames, times, keyframe_vectors, memory, labels)

    assert p4t["keyframes_shown"] == ["t1", "t0", "t3"] and p4t["marks"] == 4
    assert p4t["box"] == [30, 30, 40, 40] and not p4t["hit"]  # number 2: the second box of keyframe 1, off the part
    assert memory["on_part"] == [1] and p4tp["marks"] == 1 and p4tp["hit"]


def test_sam3q_shows_the_frames_of_sam3s_best_boxes_for_the_task_text() -> None:
    frames, times, labels, task = _scene()
    record = {"frames": {"0": {"detections": [[{"box": [0, 0, 6, 6], "score": 0.9}]]},
                         "1": {"detections": [[{"box": [10, 10, 20, 20], "score": 0.95}, {"box": [30, 30, 40, 40], "score": 0.3}]]},
                         "2": {"detections": [[]]}, "3": {"detections": [[]]}}}
    reader = FakeReader("1")

    row = run.task_prompted(reader, record, 0, task, frames, times, labels)

    assert row["keyframes_shown"] == ["t1", "t0"] and row["marks"] == 2 and row["images_read"] == 2
    assert row["hit"] and reader.images == [2]


def test_tool_features_reuse_the_saved_vectors_and_refuse_vectors_of_other_boxes(tmp_path: Path) -> None:
    candidates = [policies.Candidate(0, (1, 2, 3, 4), (0, 0, 5, 5))]
    file = tmp_path / "features_1.npz"
    np.savez_compressed(file, vectors=np.ones((1, 3)), keyframe=np.array([0]), boxes=np.array([[1, 2, 3, 4]]), seconds=0.1)

    assert run.tool_features(file, [], candidates, 3).tolist() == [[1.0, 1.0, 1.0]]
    with pytest.raises(SystemExit):
        run.tool_features(file, [], [policies.Candidate(0, (1, 2, 3, 5), (0, 0, 5, 5))], 3)
