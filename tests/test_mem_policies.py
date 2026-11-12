import numpy as np
import pytest

from affordance3d.mem import policies as W
from affordance3d.mem.tokens import Region

HW = (96, 64)  # an upright frame: more rows than columns


def _region(keyframe: int, box: tuple[int, int, int, int]) -> Region:
    return Region(keyframe, 1, 0, box, (box[0], box[1]), 16, 2.0)


def test_affordance_candidates_keep_the_region_and_read_a_wider_crop() -> None:
    candidates = W.affordance_candidates([_region(3, (20, 30, 28, 34))], HW)

    assert candidates[0].keyframe == 3 and candidates[0].box == (20, 30, 28, 34)
    left, top, right, bottom = candidates[0].crop
    assert left <= 20 and top <= 30 and right >= 28 and bottom >= 34 and right - left == bottom - top


def test_grid_candidates_tile_every_keyframe_without_gaps() -> None:
    shape = W.grid_shape(6, HW)
    candidates = W.grid_candidates(2, HW, shape)

    assert shape == (3, 2)  # more cells along the long side
    assert len(candidates) == 12 and {c.keyframe for c in candidates} == {0, 1}
    assert sum(W._area(c.box) for c in candidates if c.keyframe == 0) == HW[0] * HW[1]
    assert W.grid_shape(6, (64, 96)) == (2, 3)  # a wide frame gets the columns


def test_random_candidates_copy_the_sizes_and_stay_inside_the_image() -> None:
    like = W.affordance_candidates([_region(0, (0, 0, 10, 4)), _region(0, (0, 0, 6, 20))], HW)

    made = W.random_candidates(like, 50, n_keyframes=5, image_hw=HW, rng=np.random.default_rng(0))

    sizes = {(c.box[2] - c.box[0], c.box[3] - c.box[1]) for c in made}
    assert sizes <= {(10, 4), (6, 20)} and len(made) == 50
    assert all(0 <= c.box[0] and c.box[2] <= HW[1] and 0 <= c.box[1] and c.box[3] <= HW[0] for c in made)
    assert {c.keyframe for c in made} <= set(range(5)) and len({c.keyframe for c in made}) > 1


def test_size_matched_prefers_regions_of_the_same_area_and_never_repeats() -> None:
    pool = [W.Candidate(0, (0, 0, side, side), (0, 0, side, side)) for side in (4, 8, 16, 32)]
    like = [W.Candidate(0, (0, 0, 8, 8), (0, 0, 8, 8))]

    chosen = W.size_matched(pool, like, 2, np.random.default_rng(0))
    everything = W.size_matched(pool, like, 10, np.random.default_rng(0))

    assert chosen[0].box == (0, 0, 8, 8) and len({c.box for c in chosen}) == 2
    assert len(everything) == 4  # never more than the pool holds


def test_subsample_keeps_the_order_and_the_requested_number() -> None:
    candidates = [W.Candidate(k, (0, 0, 1, 1), (0, 0, 1, 1)) for k in range(10)]

    kept = W.subsample(candidates, 4, np.random.default_rng(1))

    assert len(kept) == 4 and [c.keyframe for c in kept] == sorted(c.keyframe for c in kept)
    assert len(W.subsample(candidates, 99, np.random.default_rng(1))) == 10


def test_covers_uses_the_region_box_with_a_margin() -> None:
    candidate = W.Candidate(0, (20, 30, 28, 34), (0, 0, 64, 64))

    assert W.covers(candidate, (24, 32)) and W.covers(candidate, (14, 32)) and not W.covers(candidate, (40, 32))
    assert W.median_area_share([candidate], HW) == pytest.approx(32 / (96 * 64))
