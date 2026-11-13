"""Phase 2b (discussion draft, section 26): does the function index beat semantic frame choice more as retrieval gets harder?

    python scripts/pooled_retrieval.py      # reads runs/mem_phase1/ and its diagnostics, writes runs/mem_phase1/pooled_retrieval.json

CPU only. Each scene's keyframes are pooled with those of K - 1 other scenes (K = 1, 2, 4, 8, 16; five draws each);
the scene's tasks are answered from the pool, and only its own keyframes count as right. Both arms see the same part
map (the annotated P4 regions): semantic choice ranks whole keyframes, the function index adds the region tokens as in
phase 1, and a third arm adds the action-type key of section 24.10.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.mem import phase1_stats as A
from affordance3d.mem import search as S
from affordance3d.mem import tokens as T

POOL_SIZES = (1, 2, 4, 8, 16)
DRAWS = 5
TOP = 3
MIN_GAIN_AT_LARGEST = 0.05
ARMS = ("semantic", "function index", "function index + action key")


def _action_key_module():
    spec = importlib.util.spec_from_file_location("score_action_key", REPO_ROOT / "scripts" / "score_action_key.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pool(scenes: list[dict], members: list[int]) -> dict:
    """One memory from several scenes: keyframes and region tokens stacked, token keyframes offset accordingly."""
    offsets = np.cumsum([0] + [len(scenes[m]["keyframes"]) for m in members[:-1]])
    return {"keyframes": np.concatenate([scenes[m]["keyframes"] for m in members]),
            "vectors": np.concatenate([scenes[m]["vectors"] for m in members]),
            "keyframe": np.concatenate([scenes[m]["keyframe"] + o for m, o in zip(members, offsets)]),
            "action": np.concatenate([scenes[m]["action"] for m in members])}


def ranks(query: np.ndarray, memory: dict, arm: str, affinity: np.ndarray | None) -> np.ndarray:
    """Keyframe indices of the pooled memory, best first, for one arm."""
    if arm == "semantic":
        return np.argsort(-(memory["keyframes"] @ query), kind="stable")
    scores = S.score_keyframes(query, memory["keyframes"], memory["vectors"], memory["keyframe"], memory["action"],
                               affinity if arm == "function index + action key" else None)
    return scores.ranking()


def hits(order: np.ndarray, right: list[int]) -> tuple[int, int]:
    """(the first keyframe is right, one of the first TOP is right); the target scene comes first in the pool."""
    right_set = set(right)
    return int(int(order[0]) in right_set), int(any(int(k) in right_set for k in order[:TOP]))


def same_type_count(memory: dict, true_types: set[int]) -> int:
    """Annotated regions in the pooled memory with one of the task's action types."""
    return int(np.isin(memory["action"], sorted(true_types)).sum()) if true_types else 0


def slope(per_scene_gaps: dict[int, list[np.ndarray]], resamples: int = A.RESAMPLES, seed: int = 0) -> tuple[float, float, float]:
    """Least-squares slope of the pooled gap over log2 K, with a 95% interval from resampling target scenes."""
    sizes = sorted(per_scene_gaps)
    x = np.log2(sizes)
    n_scenes = len(per_scene_gaps[sizes[0]])

    def fit(indices: np.ndarray) -> float:
        y = [A.pooled([per_scene_gaps[k][i] for i in indices]) for k in sizes]
        return float(np.polyfit(x, y, 1)[0])

    rng = np.random.default_rng(seed)
    draws = [fit(rng.integers(n_scenes, size=n_scenes)) for _ in range(resamples)]
    low, high = np.percentile(draws, A.INTERVAL)
    return fit(np.arange(n_scenes)), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    args = parser.parse_args()
    key = _action_key_module()
    from affordance3d.mem.encoders import ImageTextEncoder

    action_texts = ImageTextEncoder().texts([S.ACTION_PROMPTS[a] for a in T.ACTIONS])
    scenes = []
    for result in A.scored(A.load_results(args.runs)):
        with np.load(args.runs / "diagnostics" / f"{result['visit']}.npz") as saved:
            truth = json.loads(str(saved["truth"]))
            scene = {"keyframes": saved["keyframes"], "queries": saved["queries"], "vectors": saved["vectors/P4"],
                     "keyframe": saved["keyframe/P4"], "boxes": saved["boxes/P4"], "relevant": truth["relevant"]}
        actions, boxes, task_actions = key.region_actions(result, args.runs.parent, args.data)
        if not np.array_equal(boxes, scene["boxes"]):
            raise SystemExit(f"{result['visit']}: the recomputed P4 regions differ from the saved ones")
        scenes.append({**scene, "action": np.array(actions, dtype=np.int64), "task_actions": task_actions, "visit": result["visit"]})
        print(f"loaded {result['visit']}", flush=True)
    rng = np.random.default_rng(0)
    rows: dict = {k: {arm: {"top1": [], "top3": []} for arm in ARMS} for k in POOL_SIZES}
    same_type: dict = {k: [] for k in POOL_SIZES}
    for k in POOL_SIZES:
        for target, scene in enumerate(scenes):
            others = [i for i in range(len(scenes)) if i != target]
            draws = [[]] if k == 1 else [list(rng.choice(others, size=k - 1, replace=False)) for _ in range(DRAWS)]
            per_draw = {arm: {"top1": [], "top3": []} for arm in ARMS}
            counts = []
            for distractors in draws:
                memory = pool(scenes, [target, *distractors])
                for arm in ARMS:
                    outcome = [hits(ranks(q, memory, arm, S.action_affinity(q, action_texts)), right)
                               for q, right in zip(scene["queries"], scene["relevant"])]
                    per_draw[arm]["top1"].append([o[0] for o in outcome])
                    per_draw[arm]["top3"].append([o[1] for o in outcome])
                counts.append([same_type_count(memory, t) for t in scene["task_actions"]])
            for arm in ARMS:
                for metric in ("top1", "top3"):
                    rows[k][arm][metric].append(np.mean(per_draw[arm][metric], axis=0))
            same_type[k].append(np.mean(counts, axis=0))
        print(f"K = {k}: " + ", ".join(f"{arm} {A.pooled(rows[k][arm]['top3']):.3f}" for arm in ARMS), flush=True)
    summary: dict = {"scenes": len(scenes), "tasks": int(sum(len(s["queries"]) for s in scenes)), "pool_sizes": list(POOL_SIZES),
                     "per_size": {}, "readings": {}}
    gaps = {"function index": {}, "function index + action key": {}}
    for k in POOL_SIZES:
        entry = {arm: {m: dict(zip(("rate", "low", "high"), A.bootstrap(rows[k][arm][m]))) for m in ("top1", "top3")} for arm in ARMS}
        for arm, base in (("function index", "semantic"), ("function index + action key", "function index")):
            d = A.paired_difference(rows[k][arm]["top3"], rows[k][base]["top3"])
            entry[f"{arm} minus {base} (top3)"] = dict(zip(("rate", "low", "high"), d))
            gaps[arm][k] = [a - b for a, b in zip(rows[k][arm]["top3"], rows[k][base]["top3"])]
        entry["median_same_type_regions"] = float(np.median(np.concatenate(same_type[k])))
        summary["per_size"][str(k)] = entry
    for arm, base in (("function index", "semantic"), ("function index + action key", "function index")):
        s = slope(gaps[arm])
        largest = summary["per_size"][str(POOL_SIZES[-1])][f"{arm} minus {base} (top3)"]["rate"]
        supported = bool(s[1] > 0 and largest >= MIN_GAIN_AT_LARGEST)
        summary["readings"][f"{arm} minus {base}"] = {"slope_per_doubling": dict(zip(("rate", "low", "high"), s)),
                                                      "gap_at_largest": largest, "supported": supported}
    (args.runs / "pooled_retrieval.json").write_text(json.dumps(summary, indent=1))
    for name, reading in summary["readings"].items():
        sl = reading["slope_per_doubling"]
        print(f"{name}: slope per doubling {100 * sl['rate']:+.1f} [{100 * sl['low']:+.1f}, {100 * sl['high']:+.1f}] points; "
              f"gap at K = {POOL_SIZES[-1]} {100 * reading['gap_at_largest']:+.1f}; supported: {reading['supported']}")


if __name__ == "__main__":
    main()
