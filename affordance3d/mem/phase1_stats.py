"""Phase-1 statistics: rates pooled over tasks with intervals from resampling scenes, paired differences, the G1 verdict.

Tasks of one scene share its keyframes and its memory, so the scene is the unit that is resampled.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

RESAMPLES = 2000
INTERVAL = (2.5, 97.5)
P0, P1, P2, P3, P4 = "P0", "P1", "P2", "P3", "P4"


def result_files(folder: Path) -> list[Path]:
    """The per-scene result files of a run folder, named <visit>.json; the summary and the answer caches are other files."""
    return sorted(path for path in folder.glob("*.json") if path.stem.isdigit())


def load_results(folder: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in result_files(folder)]


def scored(results: Sequence[dict]) -> list[dict]:
    """The scenes that can be scored: at least one task whose part is visible in a keyframe."""
    return [result for result in results if result["tasks"]["answerable"]]


def scene_rates(result: dict, policy: str, budget: float | None = None, key: str = "hit1") -> np.ndarray | None:
    """Per-task outcome of one scene for a policy (name prefix) at a budget, averaged over seeds; None if not run."""
    runs = [r for r in result["search"] if r["policy"].startswith(policy)
            and (budget is None or abs(r["budget"] - budget) < 1e-9) and len(r[key])]
    return np.mean([r[key] for r in runs], axis=0) if runs else None


def pooled(per_scene: Sequence[np.ndarray]) -> float:
    return float(np.concatenate(per_scene).mean())


def bootstrap(per_scene: Sequence[np.ndarray], resamples: int = RESAMPLES, seed: int = 0) -> tuple[float, float, float]:
    """Pooled rate with a 95% interval from resampling scenes with replacement."""
    rng = np.random.default_rng(seed)
    draws = [pooled([per_scene[i] for i in rng.integers(len(per_scene), size=len(per_scene))]) for _ in range(resamples)]
    low, high = np.percentile(draws, INTERVAL)
    return pooled(per_scene), float(low), float(high)


def paired_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray], resamples: int = RESAMPLES,
                      seed: int = 0) -> tuple[float, float, float]:
    """Pooled rate of `first` minus that of `second` on the same tasks, with a 95% interval over scenes."""
    difference = [a - b for a, b in zip(first, second, strict=True)]
    return bootstrap(difference, resamples, seed)


def affordance_against(results: Sequence[dict], other: str, budget: float | None, key: str = "hit1") -> tuple[float, float, float] | None:
    """Paired difference of P4 at budget 1 and `other` (a policy name or its prefix) at `budget` (None: the one budget
    it ran at), on the same tasks; None unless both were run in every scene."""
    ours = [scene_rates(result, P4, 1.0, key) for result in results]
    theirs = [scene_rates(result, other, budget, key) for result in results]
    if any(rates is None for rates in (*ours, *theirs)):
        return None
    return paired_difference(ours, theirs)


CLOSE_POINTS, WIDE_POINTS = 0.05, 0.10  # round-7 rule for the controls, fixed before they ran (discussion draft, section 24)


def closeness(difference: tuple[float, float, float], close: float = CLOSE_POINTS, wide: float = WIDE_POINTS) -> str:
    """How an arm stands against P4, from P4 minus the arm (mean, low, high): "close" when the interval's lower end is below
    `close`; "undetermined" when that happens only because the interval is wide (the mean itself is above `wide`); else "behind"."""
    mean, low, _ = difference
    if low >= close:
        return "behind"
    return "undetermined" if mean > wide else "close"


def curve(results: Sequence[dict], policy: str, key: str = "hit1") -> list[dict]:
    """Points of one policy's rate-against-budget curve: runs are grouped by policy name and nominal budget."""
    groups: dict[tuple[str, float], dict[int, list]] = {}
    for scene, result in enumerate(results):
        for run in result["search"]:
            if run["policy"].startswith(policy) and len(run[key]):
                nominal = round(run["budget"], 6) if policy != P1 else 0.0  # a grid's budget differs from scene to scene
                groups.setdefault((run["policy"], nominal), {}).setdefault(scene, []).append(run)
    points = []
    for (name, _), by_scene in groups.items():
        per_scene = [np.mean([run[key] for run in runs], axis=0) for runs in by_scene.values()]
        rate, low, high = bootstrap(per_scene)
        tokens = [run["tokens"] for runs in by_scene.values() for run in runs]
        budgets = [run["budget"] for runs in by_scene.values() for run in runs]
        points.append({"policy": name, "budget": float(np.median(budgets)), "tokens": float(np.mean(tokens)), "rate": rate,
                       "low": low, "high": high, "scenes": len(per_scene)})
    return sorted(points, key=lambda point: point["budget"])


def g1_verdict(against_generic: tuple[float, float, float] | None, against_random: tuple[float, float, float] | None,
               reading_affordance: float | None, reading_generic: float | None) -> str:
    """The pre-registered rule of the discussion draft, section 21.6 (each difference is mean, low, high at budget 1)."""
    if against_random is None:
        return "undetermined: the random-places policy has not been run in every scene"
    if against_random[1] <= 0:
        return "not passed: affordance regions are not better than random places"
    if against_generic is None or reading_affordance is None or reading_generic is None:
        return "undetermined: the generic-region policy or the model reading has not been run"
    if against_generic[1] > 0 and reading_affordance >= reading_generic:
        return "passed"
    return "partly passed: better than random places, not clearly better than generic regions"
