import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from affordance3d.mem import encoders
from affordance3d.mem import phase1_stats as A
from report import mem_steps_helpers as H
from tests.conftest import frame_from_depth

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pooled_reading.py"
spec = importlib.util.spec_from_file_location("run_pooled_reading", SCRIPT)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

N_SCENES = 30  # as many as phase 2b pooled
BOXES = np.array([[0, 0, 4, 4], [2, 2, 6, 6]], dtype=np.int64)


class FakeEncoder:
    def texts(self, texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 4), 0.5)


def _diagnostics(folder: Path, visit: str, rng: np.random.Generator) -> None:
    np.savez(folder / f"{visit}.npz", **{"keyframes": rng.normal(size=(2, 4)), "queries": rng.normal(size=(1, 4)),
                                         "vectors/P4": rng.normal(size=(2, 4)), "keyframe/P4": np.array([0, 1]),
                                         "boxes/P4": BOXES, "truth": np.array(json.dumps({"relevant": [[0]]}))})


def test_the_draws_are_those_pooled_retrieval_made(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # pooled_retrieval.main itself runs on 30 synthetic scenes; every memory it pools is recorded
    pr, rng = run.PR, np.random.default_rng(1)
    (tmp_path / "diagnostics").mkdir()
    results = [{"visit": str(400000 + i), "tasks": {"answerable": 1}} for i in range(N_SCENES)]
    for result in results:
        _diagnostics(tmp_path / "diagnostics", result["visit"], rng)
    pooled, real_pool = [], pr.pool

    def recording_pool(scenes: list[dict], members: list) -> dict:
        pooled.append([int(m) for m in members])
        return real_pool(scenes, members)

    monkeypatch.setattr(A, "load_results", lambda folder: results)
    monkeypatch.setattr(A, "bootstrap", lambda *args, **kwargs: (0.0, 0.0, 0.0))
    monkeypatch.setattr(pr, "slope", lambda gaps: (0.0, 0.0, 0.0))
    monkeypatch.setattr(pr, "ranks", lambda query, memory, arm, affinity: np.arange(len(memory["keyframes"])))
    monkeypatch.setattr(pr, "pool", recording_pool)
    monkeypatch.setattr(pr, "_action_key_module",
                        lambda: SimpleNamespace(region_actions=lambda result, runs, data: ([0, 0], BOXES, [set()])))
    monkeypatch.setattr(encoders, "ImageTextEncoder", FakeEncoder)
    monkeypatch.setattr(sys, "argv", ["pooled_retrieval.py", "--runs", str(tmp_path), "--data", str(tmp_path)])

    pr.main()

    every = run.retrieval_draws(N_SCENES)
    assert pooled == [[target, *draw] for k in pr.POOL_SIZES for target in range(N_SCENES) for draw in every[k][target]]
    assert run.first_draws(N_SCENES, 7) == {k: every[k][7][0] for k in run.POOL_SIZES}


def test_first_draws_hold_k_minus_one_other_scenes_and_none_at_one() -> None:
    draws = run.first_draws(N_SCENES, 4)

    assert draws[1] == []
    for k in (4, 16):
        assert len(draws[k]) == len(set(draws[k])) == k - 1 and 4 not in draws[k]
    with pytest.raises(ValueError):
        run.first_draws(N_SCENES, 4, sizes=(3,))


def test_the_memory_holds_its_scenes_in_visit_order_whatever_the_target() -> None:
    assert run.memory_order(5, [9, 2]) == [2, 5, 9]
    assert run.memory_order(0, []) == [0]
    with pytest.raises(ValueError):
        run.memory_order(5, [9, 5])


def _saved(keyframes: list[list[float]], regions: list[tuple[int, tuple[int, int, int, int], list[float]]]) -> dict:
    """One scene as the diagnostics keep it: keyframe vectors and P4 regions (keyframe, box, vector)."""
    return {"keyframes": np.array(keyframes, dtype=float), "keyframe": np.array([r[0] for r in regions], dtype=np.int64),
            "boxes": np.array([r[1] for r in regions], dtype=np.int64), "vectors": np.array([r[2] for r in regions], dtype=float)}


def test_pool_offsets_the_regions_and_maps_every_keyframe_back_to_its_scene() -> None:
    a = _saved([[1, 0], [0, 1]], [(1, (0, 0, 2, 2), [1, 0])])
    b = _saved([[1, 0], [1, 0], [0, 1]], [(0, (1, 1, 3, 3), [0, 1]), (2, (2, 2, 4, 4), [1, 0])])

    memory = run.pool({3: a, 8: b}, [3, 8])

    assert memory["keyframes"].shape == (5, 2) and memory["vectors"].shape == (3, 2)
    assert [c.keyframe for c in memory["candidates"]] == [1, 2, 4]
    assert [c.box for c in memory["candidates"]] == [(0, 0, 2, 2), (1, 1, 3, 3), (2, 2, 4, 4)]
    assert memory["scene"].tolist() == [3, 3, 8, 8, 8] and memory["local"].tolist() == [0, 1, 0, 1, 2]
    retrieval = run.PR.pool([{**a, "action": np.zeros(1)}, {**b, "action": np.zeros(2)}], [0, 1])  # as phase 2b pooled
    assert np.array_equal(retrieval["keyframes"], memory["keyframes"]) and retrieval["keyframe"].tolist() == [1, 2, 4]


def _unit(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=float) / np.linalg.norm(values)


def test_the_semantic_arm_numbers_the_regions_of_its_keyframes_in_memory_order_up_to_the_cap() -> None:
    # scene 1 is the target; the first 40 regions in memory order come from scene 0, so the cap does not favour the target
    many = run.RUN.MAX_MARKS
    other = _saved([[0, 1, 0], [1, 0, 0]], [(0, (i, 0, i + 2, 2), [0, 0, 1]) for i in range(many - 10)])
    target = _saved([[0, 0.9, 0.1], [0, 0, 1]], [(0, (i, 5, i + 2, 7), [0, 0, 1]) for i in range(20)])
    memory = run.pool({0: other, 1: target}, [0, 1])

    shown, kept = run.semantic_choice(np.array([0.0, 1.0, 0.0]), memory)

    assert shown[:2] == [0, 2]
    assert len(kept) == many and [c.keyframe for c in kept] == [0] * (many - 10) + [2] * 10
    assert kept[-1].box == (9, 5, 11, 7)


def _random_memory(rng: np.random.Generator, n_scenes: int = 3, n_keyframes: int = 4, per_keyframe: int = 20) -> dict:
    scenes = {}
    for m in range(n_scenes):
        regions = [(k, tuple(int(v) for v in (x, y, x + 6, y + 6)), list(rng.normal(size=6)))
                   for k in range(n_keyframes) for x, y in zip(rng.integers(0, 50, per_keyframe), rng.integers(0, 36, per_keyframe))]
        saved = _saved(rng.normal(size=(n_keyframes, 6)).tolist(), regions)
        saved["keyframes"] /= np.linalg.norm(saved["keyframes"], axis=1, keepdims=True)
        saved["vectors"] /= np.linalg.norm(saved["vectors"], axis=1, keepdims=True)
        scenes[m] = saved
    return run.pool(scenes, list(range(n_scenes)))


class FakeReader:
    """Answers every question with the same text and keeps the cache keys it was asked."""

    def __init__(self, answer: str = "1") -> None:
        self.answer, self.keys = answer, []

    def ask(self, key: str, images: list, prompt: str, **options: object) -> dict:
        self.keys.append(key)
        return {"answer": self.answer, "seconds": 0.1}


def _view(memory: dict, target: int, labels: list[np.ndarray]) -> "run.PooledView":
    scenes = sorted(set(memory["scene"].tolist()))
    counts = {m: int((memory["scene"] == m).sum()) for m in scenes}
    sources = {m: run.SceneFrames(str(400000 + m), [f"t{k}" for k in range(counts[m])],
                                  lambda t: frame_from_depth(np.full((48, 64), 2.0))) for m in scenes}
    return run.PooledView(memory, sources, target, labels)


def test_the_function_arm_asks_what_p4n_asks_on_the_same_memory() -> None:
    rng = np.random.default_rng(5)
    memory = _random_memory(rng)  # 80 regions per scene: the cap of 40 binds
    labels = [np.full((48, 64), -1) for _ in range(4)]
    view, task = _view(memory, 1, labels), H.Task("open it", (5,), frozenset({0}))
    for query in rng.normal(size=(5, 6)):
        query = query / np.linalg.norm(query)
        ours, theirs = FakeReader("3"), FakeReader("3")

        row = run.read_task(ours, "function index", task, query, memory, view)
        control = run.RUN.numbered_choice(theirs, task.text, query, view.frames, view.times, memory["keyframes"],
                                          memory["candidates"], memory["vectors"], view.labels, task.parts, run.KEY_ARM)

        assert ours.keys == theirs.keys and row["marks"] == control["marks"] == run.RUN.MAX_MARKS
        assert {key: row[key] for key in control} == control


def test_the_function_index_can_find_the_target_among_keyframes_that_look_alike() -> None:
    # two scenes whose keyframes look alike as a whole; only the target's region matches the query
    other = _saved([[1, 0]], [(0, (0, 0, 4, 4), [1, 0])])
    target = _saved([[1, 0]], [(0, (0, 0, 4, 4), [0, 1])])
    memory = run.pool({0: other, 1: target}, [0, 1])
    query = _unit([0.6, 0.8])

    assert run.semantic_choice(query, memory)[0][0] == 0  # a tie keeps memory order
    assert run.function_choice(query, memory)[0][0] == 1


def _two_scenes() -> tuple[dict, "run.PooledView", H.Task]:
    """Scene 0 is the target, part 5 on its keyframe 1; scene 1's keyframe 0 has a region at the same pixels."""
    target = _saved([[1, 0, 0], _unit([0, 1, 0.1]).tolist()], [(1, (10, 10, 20, 20), [0, 1, 0])])
    other = _saved([[0, 1, 0], [0, 0, 1]], [(0, (10, 10, 20, 20), [0, 1, 0])])
    memory = run.pool({0: target, 1: other}, [0, 1])
    labels = [np.full((48, 64), -1), np.full((48, 64), -1)]
    labels[1][10:20, 10:20] = 5
    return memory, _view(memory, 0, labels), H.Task("open the drawer", (5,), frozenset({1}))


def test_a_pick_counts_only_on_the_target_scene() -> None:
    memory, view, task = _two_scenes()
    query = np.array([0.0, 1.0, 0.0])

    wrong = run.read_task(FakeReader("1"), "semantic", task, query, memory, view)  # number 1: the other scene's box
    right = run.read_task(FakeReader("2"), "semantic", task, query, memory, view)

    assert wrong["keyframes_shown"] == ["400001/t0", "400000/t1", "400000/t0"]
    assert wrong["box"] == right["box"] == [10, 10, 20, 20]
    assert not wrong["hit"] and wrong["chosen_scene"] == "400001" and wrong["frame"] == "400001/t0"
    assert right["hit"] and right["chosen_scene"] == "400000"
    assert wrong["target_keyframes_shown"] == 2 and wrong["top3_right"] and wrong["a_box_right"]
    assert (view.labels[2] == -1).all()  # the other scene's keyframe has no labels, not the target's part numbers


def test_shown_facts_see_no_right_keyframe_when_only_other_scenes_are_shown() -> None:
    memory, view, task = _two_scenes()

    facts = run.shown_facts([2, 3], [memory["candidates"][1]], task, view)

    assert facts == {"target_keyframes_shown": 0, "top3_right": False, "a_box_right": False}


def test_scene_frames_load_a_keyframe_once_when_it_is_first_shown() -> None:
    loaded = []
    frames = run.SceneFrames("400000", ["t0", "t1"], lambda t: loaded.append(t) or frame_from_depth(np.full((48, 64), 2.0)))

    frames.frame(1)
    frames.frame(1)

    assert loaded == ["t1"] and frames.name(0) == "400000/t0"


def test_scored_scenes_refuse_a_run_folder_that_differs_from_the_one_phase_2b_drew_from(tmp_path: Path,
                                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    results = [{"visit": "400000", "tasks": {"answerable": 1, "texts": ["a", "b"]}}]
    monkeypatch.setattr(A, "load_results", lambda folder: results)
    (tmp_path / "pooled_retrieval.json").write_text(json.dumps({"scenes": 1, "tasks": 2}))

    assert run.scored_scenes(tmp_path) == results
    (tmp_path / "pooled_retrieval.json").write_text(json.dumps({"scenes": 1, "tasks": 3}))
    with pytest.raises(SystemExit):
        run.scored_scenes(tmp_path)
