"""Misses and false alarms in the affordance map (R7-B; discussion draft, section 24.6): how good must a predicted map be?

    python scripts/simulate_map_errors.py      # reads runs/mem_phase1/diagnostics/<visit>.npz, writes runs/mem_phase1/map_errors.json

CPU only, from the features saved by scripts/diagnose_mem_phase1.py. A share of the affordance regions is kept (at random,
or dropping the smallest first) and generic regions are mixed in as false alarms; each setting is scored by "the first
keyframe shows the part" and compared, on the same tasks, with whole keyframes (P0) and generic regions (P3).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.mem import phase1_diagnostics as D
from affordance3d.mem import phase1_stats as A

KEEP = (1.0, 0.75, 0.5, 0.25)
FALSE_ALARMS = (0.0, 0.5, 1.0, 2.0)
SEEDS = 5
GENERIC_SEEDS = ("P3 s0", "P3 s1", "P3 s2")
THRESHOLD_FALSE_ALARMS = 0.5
SAME_PART_IOU = 0.5  # a generic region overlapping an affordance region this much is the same part, not a false alarm


def kept_tokens(areas: np.ndarray, share: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Indices of the affordance regions that survive: a random `share`, or the largest `share` ("smallest_first")."""
    n = max(1, round(share * len(areas))) if len(areas) else 0
    if mode == "random":
        return np.sort(rng.permutation(len(areas))[:n])
    if mode == "smallest_first":
        return np.sort(np.argsort(-areas, kind="stable")[:n])
    raise ValueError(f"unknown mode {mode!r}")


def iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Intersection over union of one box (left, top, right, bottom) with each of `others`."""
    width = np.clip(np.minimum(box[2], others[:, 2]) - np.maximum(box[0], others[:, 0]), 0, None)
    height = np.clip(np.minimum(box[3], others[:, 3]) - np.maximum(box[1], others[:, 1]), 0, None)
    inter = width * height
    union = (box[2] - box[0]) * (box[3] - box[1]) + (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def false_alarm_pool(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """Generic regions available as false alarms (vectors, keyframes): the size-matched draws saved for P3, each
    region once, without those that lie on an affordance region of the same keyframe."""
    seen, vectors, keyframes = set(), [], []
    affordance_boxes, affordance_keyframes = scene["boxes/P4"], scene["keyframe/P4"]
    for name in GENERIC_SEEDS:
        for vector, keyframe, box in zip(scene[f"vectors/{name}"], scene[f"keyframe/{name}"], scene[f"boxes/{name}"]):
            place = (int(keyframe), *(int(v) for v in box))
            same = affordance_boxes[affordance_keyframes == keyframe]
            if place in seen or (len(same) and iou(box, same).max() >= SAME_PART_IOU):
                continue
            seen.add(place)
            vectors.append(vector)
            keyframes.append(int(keyframe))
    width = scene["keyframes"].shape[1]
    return (np.array(vectors).reshape(len(vectors), width), np.array(keyframes, dtype=np.int64))


def setting_hits(scene: dict, share: float, alarms: float, mode: str, seed: int) -> np.ndarray:
    """Per task: 1 when the first keyframe is right, for one setting of misses and false alarms."""
    rng = np.random.default_rng(seed)
    boxes = scene["boxes/P4"]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else np.zeros(0)
    keep = kept_tokens(areas, share, mode, rng)
    vectors, keyframes = scene["vectors/P4"][keep], scene["keyframe/P4"][keep]
    pool_vectors, pool_keyframes = scene["alarm_pool"]
    n_alarms = min(round(alarms * len(boxes)), len(pool_vectors))
    if n_alarms:
        pick = rng.permutation(len(pool_vectors))[:n_alarms]
        vectors = np.concatenate([vectors, pool_vectors[pick]])
        keyframes = np.concatenate([keyframes, pool_keyframes[pick]])
    return np.array([D.first_right(D.keyframe_scores(query, scene["keyframes"], vectors, keyframes), right)
                     for query, right in zip(scene["queries"], scene["truth"]["relevant"])], dtype=float)


def reference_hits(scene: dict, name: str) -> np.ndarray:
    empty = np.zeros((0, scene["keyframes"].shape[1]))
    if name == "P0":
        return np.array([D.first_right(D.keyframe_scores(q, scene["keyframes"], empty, []), right)
                         for q, right in zip(scene["queries"], scene["truth"]["relevant"])], dtype=float)
    return np.mean([[D.first_right(D.keyframe_scores(q, scene["keyframes"], scene[f"vectors/{seed}"], scene[f"keyframe/{seed}"]), right)
                     for q, right in zip(scene["queries"], scene["truth"]["relevant"])] for seed in GENERIC_SEEDS], axis=0)


def verdict(difference: tuple[float, float, float]) -> str:
    return "clearly better" if difference[1] > 0 else "worse" if difference[2] < 0 else "no clear difference"


def threshold(grid: dict, mode: str) -> float | None:
    """The lowest kept share down to which every setting is still clearly better than P3, at the threshold
    false-alarm level (shares are walked from all kept downwards; None if keeping all is not clearly better)."""
    lowest = None
    for share in sorted(KEEP, reverse=True):
        if grid[mode][f"{share:g}"][f"{THRESHOLD_FALSE_ALARMS:g}"]["against_p3"]["verdict"] != "clearly better":
            break
        lowest = share
    return lowest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    results = A.scored(A.load_results(args.runs))
    scenes = []
    for result in results:
        with np.load(args.runs / "diagnostics" / f"{result['visit']}.npz") as saved:
            scene = {**{key: saved[key] for key in saved.files if key != "truth"}, "truth": json.loads(str(saved["truth"]))}
        scenes.append({**scene, "visit": result["visit"], "alarm_pool": false_alarm_pool(scene)})
    p0 = [reference_hits(scene, "P0") for scene in scenes]
    p3 = [reference_hits(scene, "P3") for scene in scenes]
    grid: dict = {"random": {}, "smallest_first": {}}
    for mode in grid:
        for share in KEEP:
            for alarms in FALSE_ALARMS:
                seeds = range(SEEDS) if mode == "random" or alarms else range(1)
                per_scene = [np.mean([setting_hits(scene, share, alarms, mode, seed) for seed in seeds], axis=0) for scene in scenes]
                against_p0, against_p3 = A.paired_difference(per_scene, p0), A.paired_difference(per_scene, p3)
                grid[mode].setdefault(f"{share:g}", {})[f"{alarms:g}"] = {
                    "rate": A.pooled(per_scene),
                    "against_p0": {"rate": against_p0[0], "low": against_p0[1], "high": against_p0[2], "verdict": verdict(against_p0)},
                    "against_p3": {"rate": against_p3[0], "low": against_p3[1], "high": against_p3[2], "verdict": verdict(against_p3)}}
    short = {s["visit"]: len(s["alarm_pool"][0]) for s in scenes
             if len(s["alarm_pool"][0]) < round(max(FALSE_ALARMS) * len(s["boxes/P4"]))}
    summary = {"scenes": len(scenes), "tasks": int(sum(len(s["queries"]) for s in scenes)), "p0": A.pooled(p0), "p3": A.pooled(p3),
               "scenes_with_fewer_false_alarms_than_asked": short,
               "grid": grid, "threshold_keep_share": {mode: threshold(grid, mode) for mode in grid},
               "threshold_false_alarms": THRESHOLD_FALSE_ALARMS}
    (args.runs / "map_errors.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['scenes']} scenes, {summary['tasks']} tasks; P0 {100 * summary['p0']:.1f}%, P3 {100 * summary['p3']:.1f}%")
    for mode, rows in grid.items():
        print(f"  {mode}: first keyframe right (%), rows = share kept, columns = false alarms {FALSE_ALARMS}")
        for share, row in rows.items():
            cells = [f"{100 * cell['rate']:5.1f}{'*' if cell['against_p3']['verdict'] == 'clearly better' else ' '}" for cell in row.values()]
            print(f"    keep {share:>4}: " + "  ".join(cells))
    print("  * = clearly better than P3; threshold (lowest share kept, false alarms 0.5x):", summary["threshold_keep_share"])


if __name__ == "__main__":
    main()
