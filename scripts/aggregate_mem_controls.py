"""Summarise the round-7 controls over all scenes and read them by the rule fixed before they ran (discussion draft, section 24).

    python scripts/aggregate_mem_controls.py      # reads runs/mem_phase1/controls/<visit>.json, writes runs/mem_phase1/controls_summary.json

Rates are pooled over tasks with 95% intervals from resampling scenes; P4 (arm p4n) is compared with every other arm on
the same tasks. An arm is "close" to P4 when the lower end of P4-minus-arm is below 5 points, "undetermined" when that
happens only because the interval is wide (P4-minus-arm above 10 points), and "behind" otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.mem import phase1_stats as A

NAMES = {"p4n": "P4: the memory picks the keyframes and numbers its regions", "c1": "c1: no memory, three best keyframes by meaning",
         "c2": "c2: the memory's keyframes, the model draws the box", "c3": "c3: no memory, the raw video (40 even frames)",
         "b": "b: P3, generic regions as many as P4's (5 seeds), read like P4",
         "d": "d: every generic region in the memory, read like P4",
         "cplus": "c+: no memory, our 40 keyframes, the model names one and draws the box"}
B_SEEDS = (0, 1, 2, 3, 4)
AVERAGED = ("hit", "hit_old", "frame_right", "seconds", "images_read")
READINGS = {  # fixed before the run (round-7 brief, question 4; round-8 brief, section 6)
    ("c1", "close"): "close to P4: without the memory the model does about as well; C1 does not hold; stop and discuss",
    ("c3", "close"): "close to P4: without the memory the model does about as well; C1 does not hold; stop and discuss",
    ("c2", "close"): "close to P4: the numbered boxes add nothing beyond choosing the keyframes",
    ("c1", "behind"): "behind P4: expected with the annotations; report only as room for phase 2",
    ("c2", "behind"): "behind P4: the numbered boxes add something beyond choosing the keyframes (with the annotations)",
    ("c3", "behind"): "behind P4: expected with the annotations; report only as room for phase 2",
    # second batch, fixed before the run (discussion draft, section 24.7)
    ("b", "close"): "close to P4: generic regions as many as P4's do about as well in the reading step",
    ("b", "behind"): "behind P4: expected with the annotations; report only as room for phase 2",
    ("d", "close"): "close to P4: in the reading step the annotated boxes are no better than the best of all generic regions",
    ("d", "behind"): "behind P4: expected with the annotations; report only as room for phase 2",
    ("cplus", "close"): "close to P4: without the memory the model does about as well; C1 does not hold; stop and discuss",
    ("cplus", "behind"): "behind P4: expected with the annotations; report only as room for phase 2",
}
UNDETERMINED = "undetermined: the interval is too wide to call it close"


def reading(arm: str, verdict: str) -> str:
    return UNDETERMINED if verdict == "undetermined" else READINGS[(arm, verdict)]


def triple(values: tuple[float, float, float]) -> dict:
    return {"rate": values[0], "low": values[1], "high": values[2]}


def with_seed_average(record: dict) -> dict:
    """A copy of `record` in which arm b is the per-task average of its seeds (b s0 ... b s4), when all of them ran."""
    seeds = [record["arms"].get(f"b s{seed}") for seed in B_SEEDS]
    if any(rows is None for rows in seeds):
        return record
    averaged = [{key: float(np.mean([rows[i][key] for rows in seeds])) for key in AVERAGED} for i in range(len(seeds[0]))]
    return {**record, "arms": {**record["arms"], "b": averaged}}


def per_scene(records: list[dict], arm: str, key: str) -> list[np.ndarray]:
    return [np.array([row[key] for row in record["arms"][arm]], dtype=float) for record in records]


def summarise(records: list[dict]) -> dict:
    records = [with_seed_average(record) for record in records]
    arms = [arm for arm in NAMES if all(arm in record["arms"] for record in records)]
    summary = {"scenes": len(records), "tasks": int(sum(len(record["tasks"]) for record in records)), "arms": {}, "against_p4": {}}
    for arm in arms:
        rows = [row for record in records for row in record["arms"][arm]]
        summary["arms"][arm] = {"name": NAMES[arm], "right_part": triple(A.bootstrap(per_scene(records, arm, "hit"))),
                                "right_part_phase1_rule": triple(A.bootstrap(per_scene(records, arm, "hit_old"))),
                                "right_frame": triple(A.bootstrap(per_scene(records, arm, "frame_right"))),
                                "seconds_per_question": float(np.median([row["seconds"] for row in rows])),
                                "images_per_question": float(np.mean([row["images_read"] for row in rows]))}
    if "p4n" in arms:
        p4 = per_scene(records, "p4n", "hit")
        for arm in (arm for arm in arms if arm != "p4n"):
            difference = A.paired_difference(p4, per_scene(records, arm, "hit"))
            verdict = A.closeness(difference)
            summary["against_p4"][arm] = {**triple(difference), "verdict": verdict, "reading": reading(arm, verdict)}
    if {"c1", "c2"} <= set(arms):
        summary["choosing_keyframes"] = triple(A.paired_difference(per_scene(records, "c2", "hit"), per_scene(records, "c1", "hit")))
    if {"cplus", "c3"} <= set(arms):
        summary["our_keyframes_vs_even_frames"] = triple(A.paired_difference(per_scene(records, "cplus", "hit"), per_scene(records, "c3", "hit")))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    files = sorted(path for path in (args.runs / "controls").glob("*.json") if path.stem.isdigit())
    if not files:
        raise SystemExit(f"no control results in {args.runs / 'controls'}")
    summary = summarise([json.loads(path.read_text()) for path in files])
    (args.runs / "controls_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"{summary['scenes']} scenes, {summary['tasks']} tasks")
    for arm, row in summary["arms"].items():
        part = row["right_part"]
        print(f"  {row['name']}: right part {100 * part['rate']:.1f}% [{100 * part['low']:.1f}, {100 * part['high']:.1f}]; "
              f"right frame {100 * row['right_frame']['rate']:.1f}%; {row['seconds_per_question']:.1f} s and "
              f"{row['images_per_question']:.0f} images per question")
    for arm, row in summary["against_p4"].items():
        print(f"  P4 minus {arm}: {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}] -> {row['reading']}")
    if "choosing_keyframes" in summary:
        row = summary["choosing_keyframes"]
        print(f"  c2 minus c1 (what choosing the keyframes is worth): {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]")
    if "our_keyframes_vs_even_frames" in summary:
        row = summary["our_keyframes_vs_even_frames"]
        print(f"  c+ minus c3 (our keyframes against even frames): {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]")


if __name__ == "__main__":
    main()
