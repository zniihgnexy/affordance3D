import importlib.util
from pathlib import Path

import numpy as np
import pytest

from affordance3d.mem import search as S
from affordance3d.mem import tokens as T

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_action_key.py"
spec = importlib.util.spec_from_file_location("score_action_key", SCRIPT)
key = importlib.util.module_from_spec(spec)
spec.loader.exec_module(key)


def test_key_texts_cover_every_action_type() -> None:
    for texts in key.KEYS.values():
        assert set(texts) == set(T.ACTIONS)


def test_true_affinity_shares_the_weight_among_the_true_types() -> None:
    affinity = key.true_affinity({1, 3})

    assert affinity.sum() == pytest.approx(1.0) and affinity[1] == affinity[3] == pytest.approx(0.5)
    assert np.count_nonzero(affinity) == 2


def test_an_action_key_can_move_the_first_keyframe_to_the_region_of_the_asked_type() -> None:
    # two keyframes alike in meaning; each holds one region alike in meaning, of different action types
    keyframes = np.array([[1.0, 0.0], [1.0, 0.0]])
    scene = {"keyframes": keyframes, "vectors": np.array([[1.0, 0.0], [1.0, 0.0]]), "keyframe": np.array([0, 1]),
             "action": np.array([0, 3])}
    query = np.array([1.0, 0.0])

    assert key.first_right(query, scene, [1], None) == 0  # a tie goes to the earlier keyframe
    assert key.first_right(query, scene, [1], key.true_affinity({3})) == 1
    assert key.first_right(query, scene, [1], key.true_affinity({0})) == 0
    assert S.action_affinity(query, np.eye(2)).shape == (2,)
