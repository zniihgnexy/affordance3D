import numpy as np
import pytest

from affordance3d.mem import search as S


def test_standardise_centres_and_scales_and_tolerates_constant_values() -> None:
    assert S.standardise(np.array([1.0, 2.0, 3.0])) == pytest.approx([-1.2247449, 0.0, 1.2247449])
    assert S.standardise(np.array([4.0, 4.0])).tolist() == [0.0, 0.0]
    assert S.standardise(np.zeros(0)).shape == (0,)
    with pytest.raises(ValueError):
        S.standardise(np.array([1.0, np.nan]))


def test_action_affinity_prefers_the_closest_action_phrase_and_sums_to_one() -> None:
    action_texts = np.eye(3)
    query = np.array([0.1, 0.9, 0.2])

    affinity = S.action_affinity(query, action_texts, temperature=0.1)

    assert affinity.sum() == pytest.approx(1.0)
    assert int(np.argmax(affinity)) == 1


def test_score_keyframes_lets_an_affordance_token_lift_its_keyframe() -> None:
    query = np.array([1.0, 0.0])
    keyframes = np.array([[0.6, 0.8], [0.6, 0.8], [0.6, 0.8]])  # the whole frames look the same to the query
    tokens = np.array([[1.0, 0.0], [0.0, 1.0]])  # keyframe 1 holds the matching region, keyframe 2 another one

    scores = S.score_keyframes(query, keyframes, tokens, token_keyframe=[1, 2], token_action=[0, 0])

    assert scores.ranking().tolist() == [1, 2, 0]  # the keyframe without any region comes last
    assert scores.ranking("afford").tolist() == [1, 2, 0]
    assert scores.best_token.tolist() == [S.NO_TOKEN, 0, 1]
    assert scores.ranking("semantic").tolist() == [0, 1, 2]  # plain semantics cannot tell them apart


def test_score_keyframes_takes_the_best_of_several_tokens_in_one_keyframe() -> None:
    query = np.array([1.0, 0.0])
    keyframes = np.array([[1.0, 0.0], [1.0, 0.0]])
    tokens = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])  # keyframe 0 holds a poor and the best token

    scores = S.score_keyframes(query, keyframes, tokens, token_keyframe=[0, 0, 1], token_action=[0, 0, 0])

    assert scores.best_token.tolist() == [1, 2]
    assert scores.afford[0] == pytest.approx(scores.token_scores[1]) and scores.afford[0] > scores.afford[1]
    assert scores.ranking().tolist() == [0, 1]


def test_score_keyframes_uses_the_action_the_query_asks_for() -> None:
    query = np.array([1.0, 0.0])
    keyframes = np.array([[1.0, 0.0], [1.0, 0.0]])
    tokens = np.array([[1.0, 0.0], [1.0, 0.0]])  # both crops match the words equally well
    affinity = np.array([0.1, 0.9])  # the query asks for action 1

    scores = S.score_keyframes(query, keyframes, tokens, token_keyframe=[0, 1], token_action=[0, 1], affinity=affinity)
    ignored = S.score_keyframes(query, keyframes, tokens, token_keyframe=[0, 1], token_action=[0, 1], affinity=affinity,
                                action_weight=0.0)
    flat = S.score_keyframes(query, keyframes, tokens, token_keyframe=[0, 1], token_action=[0, 1], affinity=np.array([0.5, 0.5]))

    assert scores.ranking().tolist() == [1, 0]
    assert scores.token_scores == pytest.approx([-0.4, 0.4])  # how far each affinity lies above an even split
    assert ignored.ranking().tolist() == [0, 1] and flat.token_scores == pytest.approx([0.0, 0.0])


def test_score_keyframes_without_tokens_falls_back_to_semantics() -> None:
    query = np.array([1.0, 0.0])
    keyframes = np.array([[0.0, 1.0], [1.0, 0.0]])

    scores = S.score_keyframes(query, keyframes, np.zeros((0, 2)), token_keyframe=[], token_action=[])

    assert scores.ranking().tolist() == [1, 0]
    assert scores.best_token.tolist() == [S.NO_TOKEN, S.NO_TOKEN]


def test_first_hit_and_recall_at() -> None:
    hits = [S.first_hit([4, 2, 9], {9}), S.first_hit([4, 2, 9], {4, 2}), S.first_hit([4, 2, 9], {7})]

    assert hits == [3, 1, None]
    assert S.recall_at(hits, ks=(1, 3)) == {1: pytest.approx(1 / 3), 3: pytest.approx(2 / 3)}
    assert S.recall_at([], ks=(1,)) == {1: 0.0}


def test_mark_prompt_names_the_task_and_the_number_of_marks() -> None:
    prompt = S.mark_prompt("open the drawer", n_marks=7)

    assert '"open the drawer"' in prompt and "1 to 7" in prompt


@pytest.mark.parametrize(("answer", "expected"), [("3", 3), ("Box 12.", 12), ("The answer is 0 or 4", 4),
                                                  ("none of them", None), ("99", None)])
def test_parse_mark_reads_the_first_valid_number(answer: str, expected: int | None) -> None:
    assert S.parse_mark(answer, n_marks=12) == expected


def test_parse_mark_without_any_mark_returns_none() -> None:
    assert S.parse_mark("1", n_marks=0) is None
