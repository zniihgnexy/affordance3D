import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pooled_retrieval.py"
spec = importlib.util.spec_from_file_location("pooled_retrieval", SCRIPT)
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


def _scene(keyframes: list[list[float]], vectors: list[list[float]], keyframe: list[int], action: list[int]) -> dict:
    return {"keyframes": np.array(keyframes, dtype=float), "vectors": np.array(vectors, dtype=float),
            "keyframe": np.array(keyframe, dtype=np.int64), "action": np.array(action, dtype=np.int64)}


def test_pool_stacks_scenes_and_offsets_token_keyframes() -> None:
    a = _scene([[1, 0], [0, 1]], [[1, 0]], [1], [3])
    b = _scene([[1, 0], [1, 0], [0, 1]], [[0, 1], [1, 0]], [0, 2], [0, 3])

    memory = pr.pool([a, b], [0, 1])

    assert memory["keyframes"].shape == (5, 2)
    assert memory["keyframe"].tolist() == [1, 2, 4]
    assert memory["action"].tolist() == [3, 0, 3]
    assert pr.same_type_count(memory, {3}) == 2 and pr.same_type_count(memory, set()) == 0


def test_hits_reads_the_first_keyframe_and_the_first_three() -> None:
    assert pr.hits(np.array([5, 2, 0, 1]), [0]) == (0, 1)
    assert pr.hits(np.array([0, 2, 5]), [0]) == (1, 1)
    assert pr.hits(np.array([5, 2, 3, 0]), [0]) == (0, 0)


def test_the_function_index_can_pick_the_frame_whose_region_matches_when_frames_look_alike() -> None:
    # two scenes whose keyframes look alike as a whole; only the target's region matches the query
    target = _scene([[1, 0]], [[0, 1]], [0], [0])
    other = _scene([[1, 0]], [[1, 0]], [0], [0])
    memory = pr.pool([other, target], [0, 1])
    query = np.array([0.6, 0.8])

    assert pr.ranks(query, memory, "semantic", None)[0] == 0  # a tie keeps the earlier keyframe
    assert pr.ranks(query, memory, "function index", None)[0] == 1


def test_slope_finds_a_gap_that_grows_with_pool_size() -> None:
    gaps = {k: [np.full(4, 0.02 * np.log2(k) + 0.01 * i) for i in range(6)] for k in (1, 2, 4, 8)}

    rate, low, high = pr.slope(gaps, resamples=200)

    assert rate == pytest.approx(0.02) and low > 0
