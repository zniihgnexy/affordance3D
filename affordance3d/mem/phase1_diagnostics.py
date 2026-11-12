"""Diagnostics for phase 1: is the gain of affordance tokens owed to where the tokens are, or to knowing which
keyframes hold an operable part?

A keyframe without any token gets a fixed low score in search.score_keyframes. With affordance tokens that is
information (no operable part is annotated there); with tokens placed otherwise it is chance. To separate the two,
the scoring is repeated here with its two constants exposed, a variant keeps only the one bit per keyframe, and two
token sets give the other policies exactly the keyframes and the per-keyframe token counts of the affordance tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from affordance3d.mem import search as S
from affordance3d.mem.policies import Candidate, _area
from affordance3d.mem.tokens import context_box

NEAREST_BY_AREA = 5  # a generic region is drawn among this many pool regions of the keyframe closest in area


def keyframe_scores(query: np.ndarray, keyframe_vectors: np.ndarray, token_vectors: np.ndarray, token_keyframe: Sequence[int],
                    weight: float = 1.0, empty_z: float = S.EMPTY_KEYFRAME_Z) -> np.ndarray:
    """What search.score_keyframes ranks by (no action affinity): standardised whole-keyframe similarity plus `weight`
    times the standardised best token of the keyframe; a keyframe without tokens gets `empty_z` for the second term."""
    semantic = S.standardise(keyframe_vectors @ query)
    if not len(token_vectors):
        return semantic
    token_scores = S.standardise(token_vectors @ query)
    best = np.full(len(keyframe_vectors), -np.inf)
    np.maximum.at(best, np.asarray(token_keyframe), token_scores)
    has_token = np.isfinite(best)
    afford = np.full(len(best), float(empty_z))
    afford[has_token] = S.standardise(best[has_token])
    return semantic + weight * afford


def shortlist_scores(query: np.ndarray, keyframe_vectors: np.ndarray, has_token: np.ndarray, weight: float = 1.0,
                     empty_z: float = S.EMPTY_KEYFRAME_Z) -> np.ndarray:
    """Whole keyframes plus one bit each: keyframes that would hold a token score as an average one, the others get
    `empty_z`. No token content enters."""
    return S.standardise(keyframe_vectors @ query) + weight * np.where(has_token, 0.0, float(empty_z))


def first_right(scores: np.ndarray, relevant: Sequence[int]) -> int:
    """1 when the best-scoring keyframe (ties: the earlier one) shows the part the task names."""
    return int(int(np.argsort(-scores, kind="stable")[0]) in set(relevant))


def chance_rates(relevant: Sequence[int], has_token: np.ndarray) -> tuple[float, float]:
    """Probability that a random keyframe is right: among all keyframes, and among those that hold a token."""
    right = np.zeros(len(has_token), dtype=bool)
    right[list(relevant)] = True
    among = float(right[has_token].mean()) if has_token.any() else 0.0
    return float(right.mean()), among


def random_in_same_keyframes(like: Sequence[Candidate], image_hw: tuple[int, int], rng: np.random.Generator) -> list[Candidate]:
    """For every region of `like`: a box of the same size at a random place in the same keyframe."""
    made = []
    for candidate in like:
        left, top, right, bottom = candidate.box
        width, height = min(right - left, image_hw[1]), min(bottom - top, image_hw[0])
        x, y = int(rng.integers(image_hw[1] - width + 1)), int(rng.integers(image_hw[0] - height + 1))
        box = (x, y, x + width, y + height)
        made.append(Candidate(candidate.keyframe, box, context_box(box, image_hw)))
    return made


def generic_in_same_keyframes(pool: Sequence[Candidate], like: Sequence[Candidate], rng: np.random.Generator,
                              nearest: int = NEAREST_BY_AREA) -> list[Candidate]:
    """For every region of `like`: a pool region of the same keyframe, drawn among the `nearest` unused ones closest
    in area. A keyframe whose pool runs out gives fewer regions than `like` has there."""
    chosen = []
    for keyframe in sorted({candidate.keyframe for candidate in like}):
        free = [candidate for candidate in pool if candidate.keyframe == keyframe]
        for target in (candidate for candidate in like if candidate.keyframe == keyframe):
            if not free:
                break
            gaps = np.abs(np.log([_area(candidate.box) for candidate in free]) - np.log(_area(target.box)))
            pick = int(rng.choice(np.argsort(gaps, kind="stable")[:nearest]))
            chosen.append(free.pop(pick))
    return chosen


def shown_candidates(query: np.ndarray, keyframe_vectors: np.ndarray, vectors: np.ndarray, candidates: Sequence[Candidate],
                     top_keyframes: int = 3, max_marks: int = 40) -> list[Candidate]:
    """The numbered boxes the model-reading step shows for a query: the best-scoring tokens of the best keyframes."""
    scores = S.score_keyframes(query, keyframe_vectors, vectors, [c.keyframe for c in candidates], [0] * len(candidates))
    top = {int(k) for k in scores.ranking()[:top_keyframes]}
    inside = sorted((i for i, c in enumerate(candidates) if c.keyframe in top), key=lambda i: -scores.token_scores[i])
    return [candidates[i] for i in inside[:max_marks]]
