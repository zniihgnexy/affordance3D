"""Token candidates of phase 1 and the first keyframe a query ranks: shared by the runner, the diagnostics, the
controls and the phase-1 notebook. Moved unchanged from report/mem_phase1_helpers.py, which re-exports it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from affordance3d.frames import Frame
from affordance3d.mem import phase1_stats as A
from affordance3d.mem import policies
from affordance3d.mem import search as S
from affordance3d.mem import tokens as T
from affordance3d.mem.pipeline import MIN_PART_PIXELS

RUNNER_SEED = 1000  # scripts/run_mem_phase1.py draws P2 and P3 of seed s from default_rng(1000 + s)


@dataclass(frozen=True)
class Pick:
    """What one way of placing tokens puts first for a task."""

    keyframe: int  # the keyframe ranked first
    box: policies.Box | None  # the best token of that keyframe; None when there is none (whole keyframes)
    right: bool  # the keyframe shows the part the task names


def generic_pool(regions_file: Path, image_hw: tuple[int, int]) -> list[policies.Candidate]:
    """The segments saved by scripts/make_generic_regions.py as token candidates (the pool P3 draws from)."""
    saved = json.loads(regions_file.read_text())
    return [policies.Candidate(k, (left, top, right, bottom), T.context_box((left, top, right, bottom), image_hw))
            for k, left, top, right, bottom in saved["regions"]]


def token_sets(frames: Sequence[Frame], action_maps: Sequence[np.ndarray], pool: Sequence[policies.Candidate],
               seed: int = 0) -> dict[str, list[policies.Candidate]]:
    """Where each policy puts its tokens at budget 1, drawn with the random numbers the runner uses for `seed`."""
    image_hw = frames[0].image_hw
    regions = [region for k, (actions, frame) in enumerate(zip(action_maps, frames))
               for region in T.find_regions(actions, frame.depth_m, k)[1]]
    afford = policies.affordance_candidates(regions, image_hw)
    rng = np.random.default_rng(RUNNER_SEED + seed)
    random = policies.random_candidates(afford, len(afford), len(frames), image_hw, rng)
    generic = policies.size_matched(pool, afford, len(afford), rng)
    grid = policies.grid_candidates(len(frames), image_hw, policies.grid_shape(len(afford) / len(frames), image_hw))
    return {A.P1: grid, A.P2: random, A.P3: generic, A.P4: afford}


def crops_of(frames: Sequence[Frame], candidates: Sequence[policies.Candidate]) -> list[np.ndarray]:
    return [frames[c.keyframe].rgb[c.crop[1]:c.crop[3], c.crop[0]:c.crop[2]] for c in candidates]


def target_pixels(part_labels: Sequence[np.ndarray], parts: Sequence[int]) -> dict[int, list[tuple[int, int]]]:
    """For each keyframe that shows a target part: the centre pixel (u, v) of every target part in it (ground truth)."""
    centres: dict[int, list[tuple[int, int]]] = {}
    for k, labels in enumerate(part_labels):
        for part in parts:
            rows_cols = np.argwhere(labels == part)
            if len(rows_cols) >= MIN_PART_PIXELS:
                v, u = np.median(rows_cols, axis=0)
                centres.setdefault(k, []).append((int(u), int(v)))
    return centres


def first_pick(query: np.ndarray, keyframe_vectors: np.ndarray, candidates: Sequence[policies.Candidate], vectors: np.ndarray,
               relevant: frozenset[int] | set[int]) -> Pick:
    """The keyframe a query ranks first with these tokens (none = whole keyframes only), scored as the runner scores it."""
    scores = S.score_keyframes(query, keyframe_vectors, vectors, [c.keyframe for c in candidates], [0] * len(candidates))
    keyframe = int(scores.ranking("combined" if candidates else "semantic")[0])
    token = int(scores.best_token[keyframe])
    return Pick(keyframe, None if token == S.NO_TOKEN else candidates[token].box, keyframe in relevant)
