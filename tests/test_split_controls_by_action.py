import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "split_controls_by_action.py"
spec = importlib.util.spec_from_file_location("split_controls_by_action", SCRIPT)
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)


def test_spread_is_zero_when_every_action_has_the_same_gap() -> None:
    differences = np.array([1.0, 0.0, 1.0, 0.0])
    actions = np.array(["a", "a", "b", "b"])

    assert split.spread(differences, actions) == 0.0
    assert split.spread(np.array([1.0, 1.0, 0.0, 0.0]), actions) == 1.0  # 2 * 0.5**2 + 2 * 0.5**2


def test_heterogeneity_finds_a_real_difference_and_not_a_made_up_one() -> None:
    rng = np.random.default_rng(1)
    scenes = np.repeat(np.arange(20), 10)
    actions = np.tile(np.array(["rotate"] * 5 + ["pull"] * 5), 20)
    differing = np.where(actions == "rotate", 1.0, 0.0)
    alike = rng.integers(0, 2, size=200).astype(float)

    _, p_differing = split.heterogeneity(differing, actions, scenes, shuffles=500)
    _, p_alike = split.heterogeneity(alike, actions, scenes, shuffles=500)

    assert p_differing < 0.01
    assert p_alike > 0.05


def test_shuffles_stay_inside_each_scene() -> None:
    # one action per scene: shuffling within scenes cannot change anything, so every shuffle ties the observed spread
    scenes = np.array([0, 0, 1, 1])
    actions = np.array(["a", "a", "b", "b"])

    _, p_value = split.heterogeneity(np.array([1.0, 1.0, 0.0, 0.0]), actions, scenes, shuffles=50)

    assert p_value == 1.0
