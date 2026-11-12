import numpy as np
import pytest

from affordance3d.mem import phase1_stats as A


def _run(policy: str, budget: float, hit1: list[int], seed: int = 0, tokens: int = 10) -> dict:
    return {"policy": policy, "budget": budget, "seed": seed, "tokens": tokens, "hit1": hit1, "hit3": hit1, "on_part": []}


def test_scene_rates_average_the_seeds_of_one_policy_and_budget() -> None:
    result = {"search": [_run("P2 random places", 1.0, [1, 0], seed=0), _run("P2 random places", 1.0, [0, 0], seed=1),
                         _run("P2 random places", 2.0, [1, 1]), _run("P4 affordance regions", 1.0, [1, 1])]}

    assert A.scene_rates(result, A.P2, 1.0).tolist() == [0.5, 0.0]
    assert A.scene_rates(result, A.P4, 1.0).tolist() == [1.0, 1.0]
    assert A.scene_rates(result, A.P3, 1.0) is None


def test_bootstrap_gives_the_pooled_rate_and_an_interval_around_it() -> None:
    per_scene = [np.array([1.0, 1.0, 1.0]), np.array([0.0]), np.array([1.0, 0.0])]

    rate, low, high = A.bootstrap(per_scene, resamples=500)
    same, low_same, high_same = A.bootstrap([np.ones(4), np.ones(2)], resamples=100)

    assert rate == pytest.approx(4 / 6) and low < rate < high and 0.0 <= low and high <= 1.0
    assert (same, low_same, high_same) == (1.0, 1.0, 1.0)  # no variation between scenes, no interval


def test_paired_difference_compares_the_same_tasks() -> None:
    better = [np.array([1.0, 1.0]), np.array([1.0, 0.0])]
    worse = [np.array([0.0, 1.0]), np.array([0.0, 0.0])]

    difference, low, high = A.paired_difference(better, worse, resamples=200)

    assert difference == pytest.approx(0.5) and low > 0 and high <= 1.0
    with pytest.raises(ValueError):
        A.paired_difference(better, worse[:1])


def test_curve_groups_grids_by_shape_and_other_policies_by_budget() -> None:
    scenes = [{"search": [_run("P1 uniform grid 1x1", 0.6, [1, 0], tokens=40), _run("P1 uniform grid 2x1", 1.3, [1, 1], tokens=80),
                          _run("P4 affordance regions", 1.0, [1, 1], tokens=60)]},
              {"search": [_run("P1 uniform grid 1x1", 0.8, [0, 0], tokens=40), _run("P1 uniform grid 2x1", 1.6, [1, 0], tokens=80),
                          _run("P4 affordance regions", 1.0, [1, 0], tokens=50)]}]

    grid, afford = A.curve(scenes, A.P1), A.curve(scenes, A.P4)

    assert [p["policy"] for p in grid] == ["P1 uniform grid 1x1", "P1 uniform grid 2x1"]
    assert grid[0]["rate"] == pytest.approx(0.25) and grid[0]["budget"] == pytest.approx(0.7) and grid[0]["scenes"] == 2
    assert len(afford) == 1 and afford[0]["rate"] == pytest.approx(0.75) and afford[0]["tokens"] == pytest.approx(55)


@pytest.mark.parametrize(("generic", "random", "reading", "expected"), [
    ((0.10, 0.02, 0.2), (0.2, 0.1, 0.3), (0.6, 0.5), "passed"),
    ((0.10, -0.02, 0.2), (0.2, 0.1, 0.3), (0.6, 0.5), "partly passed"),
    ((0.10, 0.02, 0.2), (0.2, 0.1, 0.3), (0.4, 0.5), "partly passed"),
    ((0.10, 0.02, 0.2), (0.05, -0.1, 0.2), (0.6, 0.5), "not passed"),
    (None, (0.2, 0.1, 0.3), (None, None), "undetermined"),
])
def test_g1_verdict_follows_the_rule_written_before_the_experiment(generic, random, reading, expected: str) -> None:
    assert A.g1_verdict(generic, random, *reading).startswith(expected)
