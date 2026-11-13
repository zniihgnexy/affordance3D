import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simulate_map_errors.py"
spec = importlib.util.spec_from_file_location("simulate_map_errors", SCRIPT)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _scene() -> dict:
    """Four keyframes; the query matches the region in keyframe 2, which is the right keyframe."""
    keyframes = _unit(np.array([[1.0, 0.2], [0.9, 0.3], [0.2, 1.0], [0.8, 0.5]]))
    p4 = {"vectors/P4": _unit(np.array([[0.0, 1.0], [1.0, 0.0]])), "keyframe/P4": np.array([2, 0]),
          "boxes/P4": np.array([[0, 0, 10, 10], [0, 0, 2, 2]])}
    generic = {f"vectors/{name}": _unit(np.array([[1.0, 0.1], [0.9, 0.2]])) for name in sim.GENERIC_SEEDS}
    generic.update({f"keyframe/{name}": np.array([1, 3]) for name in sim.GENERIC_SEEDS})
    generic.update({f"boxes/{name}": np.array([[40, 40, 50, 50], [60, 60, 70, 70]]) for name in sim.GENERIC_SEEDS})
    return {"keyframes": keyframes, "queries": _unit(np.array([[0.0, 1.0]])), "truth": {"relevant": [[2]]}, **p4, **generic}


def test_kept_tokens_drops_at_random_or_the_smallest_first() -> None:
    areas = np.array([100, 4, 50, 9])

    assert sim.kept_tokens(areas, 0.5, "smallest_first", np.random.default_rng(0)).tolist() == [0, 2]
    assert len(sim.kept_tokens(areas, 0.25, "random", np.random.default_rng(0))) == 1
    assert sim.kept_tokens(areas, 1.0, "random", np.random.default_rng(0)).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        sim.kept_tokens(areas, 0.5, "biggest_first", np.random.default_rng(0))


def test_a_full_map_finds_the_keyframe_and_losing_its_region_loses_it() -> None:
    scene = {**_scene(), "alarm_pool": (np.zeros((0, 2)), np.zeros(0, dtype=np.int64))}

    assert sim.setting_hits(scene, 1.0, 0.0, "random", seed=0).tolist() == [1.0]
    # keeping only the larger region keeps the right one (area 100); keeping only the smaller one loses it
    assert sim.setting_hits(scene, 0.5, 0.0, "smallest_first", seed=0).tolist() == [1.0]
    scene_small_right = {**scene, "boxes/P4": np.array([[0, 0, 2, 2], [0, 0, 10, 10]])}
    assert sim.setting_hits(scene_small_right, 0.5, 0.0, "smallest_first", seed=0).tolist() == [0.0]


def test_false_alarms_come_from_the_generic_regions_and_reference_arms_are_scored_alike() -> None:
    scene = _scene()
    scene["alarm_pool"] = sim.false_alarm_pool(scene)

    with_alarms = sim.setting_hits(scene, 1.0, 2.0, "random", seed=0)
    assert with_alarms.shape == (1,)
    assert sim.reference_hits(scene, "P0").shape == (1,) and sim.reference_hits(scene, "P3").shape == (1,)
    assert sim.verdict((0.1, 0.02, 0.2)) == "clearly better" and sim.verdict((-0.1, -0.2, -0.01)) == "worse"
    assert sim.verdict((0.0, -0.05, 0.05)) == "no clear difference"


def test_false_alarm_pool_drops_repeats_and_regions_on_an_affordance_part() -> None:
    scene = _scene()
    scene["boxes/P3 s0"] = scene["boxes/P3 s1"] = scene["boxes/P3 s2"] = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    scene["keyframe/P3 s0"] = scene["keyframe/P3 s1"] = scene["keyframe/P3 s2"] = np.array([2, 3])

    vectors, keyframes = sim.false_alarm_pool(scene)

    # the first region repeats the affordance box of keyframe 2, the second appears in all three draws: one is left
    assert keyframes.tolist() == [3] and vectors.shape == (1, 2)


def test_threshold_walks_down_from_keeping_everything() -> None:
    def cell(verdict: str) -> dict:
        return {f"{sim.THRESHOLD_FALSE_ALARMS:g}": {"against_p3": {"verdict": verdict}}}

    grid = {"random": {"1": cell("clearly better"), "0.75": cell("clearly better"), "0.5": cell("no clear difference"),
                       "0.25": cell("clearly better")}}

    assert sim.threshold(grid, "random") == 0.75
    grid["random"]["1"] = cell("worse")
    assert sim.threshold(grid, "random") is None
