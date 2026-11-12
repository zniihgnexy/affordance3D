"""Where a memory puts its tokens: the write policies compared in phase 1.

Every policy yields candidates of the same kind (a keyframe, a region box, and the crop the image-text encoder
reads), so that the policies differ only in where the regions are. P4 takes the regions of the affordance
map; the others ignore it: a regular grid (P1), random places with the sizes of P4 (P2), and regions of a
segmenter that knows nothing about affordance (P3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from affordance3d.mem.tokens import Region, context_box

Box = tuple[int, int, int, int]  # left, top, right, bottom (exclusive), pixels
GRID_SHAPES = ((1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (6, 4))  # rows x columns for an upright frame


@dataclass(frozen=True)
class Candidate:
    keyframe: int
    box: Box  # the region the token stands for
    crop: Box  # what the image-text encoder reads


def affordance_candidates(regions: Sequence[Region], image_hw: tuple[int, int]) -> list[Candidate]:
    """P4: one token per region of the affordance map."""
    return [Candidate(region.keyframe, region.box, context_box(region.box, image_hw)) for region in regions]


def grid_shape(tokens_per_keyframe: float, image_hw: tuple[int, int]) -> tuple[int, int]:
    """The grid whose cell count is closest to the wanted tokens per keyframe; long side of the image gets more cells."""
    rows, columns = min(GRID_SHAPES, key=lambda shape: abs(shape[0] * shape[1] - tokens_per_keyframe))
    return (rows, columns) if image_hw[0] >= image_hw[1] else (columns, rows)


def grid_candidates(n_keyframes: int, image_hw: tuple[int, int], shape: tuple[int, int]) -> list[Candidate]:
    """P1: every keyframe cut into the same grid; a cell is both the region and the crop."""
    rows, columns = shape
    tops = np.linspace(0, image_hw[0], rows + 1).astype(int)
    lefts = np.linspace(0, image_hw[1], columns + 1).astype(int)
    cells = [(int(lefts[c]), int(tops[r]), int(lefts[c + 1]), int(tops[r + 1])) for r in range(rows) for c in range(columns)]
    return [Candidate(k, cell, cell) for k in range(n_keyframes) for cell in cells]


def random_candidates(like: Sequence[Candidate], n: int, n_keyframes: int, image_hw: tuple[int, int],
                      rng: np.random.Generator) -> list[Candidate]:
    """P2: `n` regions with the sizes of `like`, in random keyframes at random places."""
    made = []
    for _ in range(n):
        left, top, right, bottom = like[int(rng.integers(len(like)))].box
        width, height = min(right - left, image_hw[1]), min(bottom - top, image_hw[0])
        x, y = int(rng.integers(image_hw[1] - width + 1)), int(rng.integers(image_hw[0] - height + 1))
        box = (x, y, x + width, y + height)
        made.append(Candidate(int(rng.integers(n_keyframes)), box, context_box(box, image_hw)))
    return made


def size_matched(pool: Sequence[Candidate], like: Sequence[Candidate], n: int, rng: np.random.Generator) -> list[Candidate]:
    """P3: `n` regions drawn from `pool` so that their areas follow those of `like` (nearest area, without repeats)."""
    areas = np.array([_area(candidate.box) for candidate in pool], dtype=float)
    free = np.ones(len(pool), dtype=bool)
    chosen = []
    for _ in range(min(n, len(pool))):
        target = _area(like[int(rng.integers(len(like)))].box)
        index = int(np.argmin(np.where(free, np.abs(np.log(areas) - np.log(target)), np.inf)))
        free[index] = False
        chosen.append(pool[index])
    return chosen


def subsample(candidates: Sequence[Candidate], n: int, rng: np.random.Generator) -> list[Candidate]:
    """A random `n` of the candidates (all of them when there are fewer)."""
    keep = rng.permutation(len(candidates))[:n]
    return [candidates[int(i)] for i in sorted(keep)]


def _area(box: Box) -> int:
    return max(box[2] - box[0], 1) * max(box[3] - box[1], 1)


def covers(candidate: Candidate, pixel: tuple[int, int], pad: int = 8) -> bool:
    """True when the region box (with a small margin) contains the pixel (u, v)."""
    left, top, right, bottom = candidate.box
    return left - pad <= pixel[0] < right + pad and top - pad <= pixel[1] < bottom + pad


def median_area_share(candidates: Sequence[Candidate], image_hw: tuple[int, int]) -> float:
    """Median region area as a share of the image: how specific the regions are."""
    if not candidates:
        return float("nan")
    return float(np.median([_area(candidate.box) for candidate in candidates]) / (image_hw[0] * image_hw[1]))
