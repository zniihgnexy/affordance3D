import importlib.util
from pathlib import Path

import numpy as np

from affordance3d.mem import policies
from report import mem_phase1_helpers as R

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
