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
