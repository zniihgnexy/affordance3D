import numpy as np
import pytest

from affordance3d.mem import phase1_diagnostics as D
from affordance3d.mem import search as S
from affordance3d.mem.policies import Candidate

HW = (96, 64)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _candidate(keyframe: int, box: tuple[int, int, int, int]) -> Candidate:
    return Candidate(keyframe, box, (0, 0, 32, 32))


def test_keyframe_scores_repeat_the_search_formula_and_expose_its_constants() -> None:
    rng = np.random.default_rng(0)
    keyframes, tokens, query = _unit(rng.normal(size=(6, 8))), _unit(rng.normal(size=(7, 8))), _unit(rng.normal(size=(1, 8)))[0]
    token_keyframe = [0, 0, 2, 3, 3, 3, 5]  # keyframes 1 and 4 hold no token

    same = D.keyframe_scores(query, keyframes, tokens, token_keyframe)
    no_penalty = D.keyframe_scores(query, keyframes, tokens, token_keyframe, empty_z=0.0)
    half = D.keyframe_scores(query, keyframes, tokens, token_keyframe, weight=0.5)

    reference = S.score_keyframes(query, keyframes, tokens, token_keyframe, [0] * 7)
    assert same == pytest.approx(reference.combined)
    assert (no_penalty - same)[[1, 4]] == pytest.approx([3.0, 3.0]) and (no_penalty - same)[[0, 2, 3, 5]] == pytest.approx(0.0)
    assert half - S.standardise(keyframes @ query) == pytest.approx(0.5 * (same - S.standardise(keyframes @ query)))
    assert D.keyframe_scores(query, keyframes, np.zeros((0, 8)), []) == pytest.approx(S.standardise(keyframes @ query))


def test_shortlist_scores_keep_one_bit_per_keyframe_and_no_token_content() -> None:
    keyframes, query = _unit(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])), np.array([1.0, 0.0])
    has_token = np.array([False, True, True])

    scores = D.shortlist_scores(query, keyframes, has_token)

    semantic = S.standardise(keyframes @ query)
    assert scores == pytest.approx(semantic + np.array([S.EMPTY_KEYFRAME_Z, 0.0, 0.0]))
    assert D.first_right(scores, relevant=[1]) == 1 and D.first_right(semantic, relevant=[1]) == 0  # the bit moved keyframe 0 down
    assert D.first_right(np.array([1.0, 1.0]), relevant=[1]) == 0  # a tie goes to the earlier keyframe


def test_chance_rates_count_right_keyframes_among_all_and_among_those_with_a_token() -> None:
    has_token = np.array([True, True, False, False])

    assert D.chance_rates([0, 2], has_token) == (0.5, 0.5)
    assert D.chance_rates([0, 1], has_token) == (0.5, 1.0)
    assert D.chance_rates([2], np.zeros(4, dtype=bool)) == (0.25, 0.0)


def test_random_in_same_keyframes_keeps_keyframe_and_size_of_every_region() -> None:
    like = [_candidate(0, (2, 4, 12, 10)), _candidate(0, (20, 30, 24, 50)), _candidate(3, (0, 0, 8, 8))]

    made = D.random_in_same_keyframes(like, HW, np.random.default_rng(1))

    assert [c.keyframe for c in made] == [0, 0, 3]
    assert [(c.box[2] - c.box[0], c.box[3] - c.box[1]) for c in made] == [(10, 6), (4, 20), (8, 8)]
    assert all(0 <= c.box[0] and c.box[2] <= HW[1] and 0 <= c.box[1] and c.box[3] <= HW[0] for c in made)
    assert [c.box for c in made] != [c.box for c in like]


def test_generic_in_same_keyframes_draws_size_matched_pool_regions_without_repeats() -> None:
    like = [_candidate(0, (0, 0, 10, 10)), _candidate(0, (0, 0, 10, 10)), _candidate(2, (0, 0, 4, 4)), _candidate(2, (0, 0, 4, 4))]
    pool = [_candidate(0, (0, 0, 10, 10)), _candidate(0, (5, 5, 16, 16)), _candidate(0, (0, 0, 60, 60)),  # two close in area, one far
            _candidate(1, (0, 0, 10, 10)), _candidate(2, (1, 1, 5, 5))]  # keyframe 2 offers one region only

    drawn = D.generic_in_same_keyframes(pool, like, np.random.default_rng(0), nearest=1)

    assert sorted(c.box for c in drawn if c.keyframe == 0) == [(0, 0, 10, 10), (5, 5, 16, 16)]  # never the far one, no repeat
    assert [c.box for c in drawn if c.keyframe == 2] == [(1, 1, 5, 5)]  # the pool ran out: one region instead of two
    assert not [c for c in drawn if c.keyframe == 1]  # a keyframe without an affordance region gets nothing


def test_shown_candidates_are_the_best_tokens_of_the_best_keyframes() -> None:
    keyframes = _unit(np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]]))
    candidates = [_candidate(0, (0, 0, 4, 4)), _candidate(1, (0, 0, 4, 4)), _candidate(2, (0, 0, 4, 4)), _candidate(2, (8, 8, 12, 12))]
    vectors = _unit(np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.2, 0.8]]))

    shown = D.shown_candidates(np.array([0.0, 1.0]), keyframes, vectors, candidates, top_keyframes=1, max_marks=1)
    both = D.shown_candidates(np.array([0.0, 1.0]), keyframes, vectors, candidates, top_keyframes=1)

    assert shown == [candidates[2]] and both == [candidates[2], candidates[3]]
