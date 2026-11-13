"""Summarise the round-7 controls over all scenes and read them by the rule fixed before they ran (discussion draft, section 24).

    python scripts/aggregate_mem_controls.py      # reads runs/mem_phase1/controls/<visit>.json, writes runs/mem_phase1/controls_summary.json

Rates are pooled over tasks with 95% intervals from resampling scenes; P4 (arm p4n) is compared with every other arm on
the same tasks. An arm is "close" to P4 when the lower end of P4-minus-arm is below 5 points, "undetermined" when that
happens only because the interval is wide (P4-minus-arm above 10 points), and "behind" otherwise.

When the tool-map arms of section 24.9 ran (p4t_<tool>, c4t_<tool>, p4tp_<tool>, sam3q; usually with --model 8b), they
are read by that section's rules, with the deciding tool and the recall gain taken from runs/mem_phase1/tool_maps_summary.json
(scripts/score_tool_maps.py): first whether to stop, then how large the gap to the annotated map is, then how the gap
splits into misses and false alarms.
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
         "cplus": "c+: no memory, our 40 keyframes, the model names one and draws the box",
         "c4": "c4: the annotated boxes on keyframes chosen without the memory, numbered",
         "c1s": "c1': no memory, three keyframes by meaning, read like c3",
         "c2s": "c2': the memory's keyframes, read like c3",
         # round 9, question 2 (section 24.9): the map built by off-the-shelf tools
         "p4t_sam3": "tool P4 (SAM 3): the memory built from SAM 3's boxes, read like P4",
         "p4t_qwen8b": "tool P4 (Qwen3-VL-8B): the memory built from Qwen's boxes, read like P4",
         "c4t_sam3": "tool c4 (SAM 3): SAM 3's boxes on keyframes chosen without the memory, numbered",
         "c4t_qwen8b": "tool c4 (Qwen3-VL-8B): Qwen's boxes on keyframes chosen without the memory, numbered",
         "p4tp_sam3": "diagnostic: tool P4 (SAM 3) with only the boxes on an annotated part",
         "p4tp_qwen8b": "diagnostic: tool P4 (Qwen3-VL-8B) with only the boxes on an annotated part",
         "sam3q": "no memory: SAM 3 prompted with the task text on 40 even frames, its best boxes numbered"}
TOOLS = ("sam3", "qwen8b")
TOOL_ARMS = (*(f"{kind}_{tool}" for kind in ("p4t", "c4t", "p4tp") for tool in TOOLS), "sam3q")
TOOL_READING = "descriptive here; read by the rules of section 24.9 (see tool_maps)"
TOOL_SUMMARY = "tool_maps_summary.json"
GAP_POINTS, RISE_POINTS = 0.10, 0.05  # section 24.9, fixed before the run
# Which recall gain (scripts/score_tool_maps.py) rule 2 of section 24.9 reads. The literal one counts all 481 parts at every
# budget, so visibility alone lifts it (+44 points with the annotated boxes as the tool); "..._among_visible" counts only
# the parts the 10 keyframes show. Decide before the tool maps are scored; never after seeing them.
GAIN_KEY = "recall_gain_40_minus_10_among_visible"  # counted only over parts the 10 keyframes show, so that visibility alone cannot make recall rise (section 24.9, amendment before any tool ran)
FOLDERS = {"4b": ("controls", "controls_summary.json"), "8b": ("controls_8b", "controls_8b_summary.json")}
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
    # round 9, fixed before the run (discussion draft, section 24.8)
    ("c4", "close"): "close to P4: with the same annotated boxes the memory's choice of keyframes adds little; option B rests on the index, so stop and discuss",
    ("c4", "behind"): "behind P4: the memory's choice of keyframes adds something (an upper bound: the choice itself uses the annotations)",
    ("c1s", "close"): "close to P4: without the memory the model does about as well; C1 does not hold; stop and discuss",
    ("c1s", "behind"): "behind P4: expected with the annotations; report only as room for phase 2",
    ("c2s", "close"): "close to P4: the numbered boxes add nothing beyond choosing the keyframes",
    ("c2s", "behind"): "behind P4: the numbered boxes add something beyond choosing the keyframes (with the annotations)",
}
UNDETERMINED = "undetermined: the interval is too wide to call it close"


STOP_READING = ("stop: indexing by function shows no advantage under real conditions; options B and C do not hold; "
                "stop and discuss with the lead")
GO_READING = "no stop: none of the three conditions of section 24.9, point 1, holds"
GAP_READINGS = {  # section 24.9, point 2; suggestions only, the lead decides
    "small": "the off-the-shelf tools suffice: supports option B; a new way of building the map is no contribution for now (DF-4)",
    "large": "building the map has work to do: suggest reconsidering DF-4",
    "undetermined": "undetermined"}


def reading(arm: str, verdict: str) -> str:
    if arm in TOOL_ARMS:
        return TOOL_READING
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
    if {"c1s", "c2s"} <= set(arms):
        difference = A.paired_difference(per_scene(records, "c2s", "hit"), per_scene(records, "c1s", "hit"))
        summary["choosing_keyframes_same_reading"] = {**triple(difference), "verdict": "worth something" if difference[1] > 0 else "not visible"}
    if {"cplus", "c3"} <= set(arms):
        summary["our_keyframes_vs_even_frames"] = triple(A.paired_difference(per_scene(records, "cplus", "hit"), per_scene(records, "c3", "hit")))
    return summary


def against_smaller(records: list[dict], smaller: list[dict]) -> dict:
    """Per arm run by both models on the same tasks: this model minus the 4B model (descriptive)."""
    arms = [arm for arm in NAMES if all(arm in r["arms"] for r in records) and all(arm in r["arms"] for r in smaller)]
    return {arm: triple(A.paired_difference(per_scene(records, arm, "hit"), per_scene(smaller, arm, "hit"))) for arm in arms}


def ran_everywhere(records: list[dict], arms: tuple[str, ...]) -> list[str]:
    """The arms of `arms` missing in at least one scene."""
    return [arm for arm in arms if not all(arm in record["arms"] for record in records)]


def difference(records: list[dict], first: str, second: str) -> tuple[float, float, float]:
    return A.paired_difference(per_scene(records, first, "hit"), per_scene(records, second, "hit"))


def stop_reading(records: list[dict], tool: str) -> dict:
    """Section 24.9, point 1: stop when tool P4 minus tool c4 is close (lower end below 5 points; a difference above 10
    points with a wide interval is undetermined and does not stop), or when tool P4 minus c3 or minus sam3q has a lower
    end at or below zero."""
    missing = ran_everywhere(records, (f"p4t_{tool}", f"c4t_{tool}", "c3", "sam3q"))
    if missing:
        return {"reading": f"not read: arms {missing} have not run in every scene"}
    memory = difference(records, f"p4t_{tool}", f"c4t_{tool}")
    raw_video, task_text = difference(records, f"p4t_{tool}", "c3"), difference(records, f"p4t_{tool}", "sam3q")
    memory_verdict = A.closeness(memory)
    stop = memory_verdict == "close" or raw_video[1] <= 0 or task_text[1] <= 0
    return {"tool_p4_minus_tool_c4": {**triple(memory), "verdict": memory_verdict},
            "tool_p4_minus_c3": {**triple(raw_video), "lower_end_above_zero": raw_video[1] > 0},
            "tool_p4_minus_sam3q": {**triple(task_text), "lower_end_above_zero": task_text[1] > 0},
            "stop": stop, "reading": STOP_READING if stop else GO_READING}


def gap_reading(records: list[dict], tool: str, recall_gain: dict | None) -> dict:
    """Section 24.9, point 2: annotated P4 minus tool P4. Upper end below 10 points: the tools suffice. Lower end above
    10 points and recall rising with the write budget (40 minus 10 keyframes: lower end above zero, at least 5 points):
    building the map has work to do. Anything else is undetermined."""
    missing = ran_everywhere(records, ("p4n", f"p4t_{tool}"))
    if missing:
        return {"reading": f"not read: arms {missing} have not run in every scene"}
    gap = difference(records, "p4n", f"p4t_{tool}")
    rises = recall_gain is not None and recall_gain["low"] > 0 and recall_gain["rate"] >= RISE_POINTS
    verdict = "small" if gap[2] < GAP_POINTS else "large" if gap[1] > GAP_POINTS and rises else "undetermined"
    return {"annotated_p4_minus_tool_p4": triple(gap), "recall_rises_with_budget": rises, "recall_gain": recall_gain,
            "recall_gain_read": GAIN_KEY, "verdict": verdict, "reading": GAP_READINGS[verdict]}


def gap_split(records: list[dict], tool: str) -> dict:
    """Section 24.9, point 3: of the gap, what keeping only the right boxes wins back (false alarms: tool P4 with only the
    boxes on a part, minus tool P4) and what is left (misses: annotated P4 minus that)."""
    missing = ran_everywhere(records, ("p4n", f"p4t_{tool}", f"p4tp_{tool}"))
    if missing:
        return {"reading": f"not read: arms {missing} have not run in every scene"}
    gap = difference(records, "p4n", f"p4t_{tool}")
    alarms, misses = difference(records, f"p4tp_{tool}", f"p4t_{tool}"), difference(records, "p4n", f"p4tp_{tool}")
    share = (lambda part: part / gap[0]) if gap[0] > 0 else (lambda part: None)
    return {"false_alarms": {**triple(alarms), "share_of_gap": share(alarms[0])},
            "misses": {**triple(misses), "share_of_gap": share(misses[0])}}


def tool_map_readings(records: list[dict], tool_summary: dict | None) -> dict:
    """Every section-24.9 reading, per tool; the verdict is that of the deciding tool (named by scripts/score_tool_maps.py
    from the recall alone, before any reading)."""
    deciding = (tool_summary or {}).get("deciding_tool")
    out: dict = {"deciding_tool": deciding, "tools": {}}
    if not ran_everywhere(records, ("p4n", "c4")):
        out["annotated_p4_minus_annotated_c4"] = triple(difference(records, "p4n", "c4"))  # the same difference with annotations
    for tool in TOOLS:
        gain = (tool_summary or {}).get("tools", {}).get(tool, {}).get(GAIN_KEY)
        out["tools"][tool] = {"stop": stop_reading(records, tool), "gap": gap_reading(records, tool, gain),
                              "split": gap_split(records, tool)}
    if deciding is None:
        out["reading"] = "not read: no deciding tool yet (run scripts/score_tool_maps.py on all scenes first)"
    else:
        out["verdict"] = out["tools"][deciding["tool"]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--model", choices=sorted(FOLDERS), default="4b")
    args = parser.parse_args()
    folder, out = FOLDERS[args.model]
    files = sorted(path for path in (args.runs / folder).glob("*.json") if path.stem.isdigit())
    if not files:
        raise SystemExit(f"no control results in {args.runs / folder}")
    records = [json.loads(path.read_text()) for path in files]
    summary = summarise(records)
    if args.model != "4b":
        smaller = {path.stem: json.loads(path.read_text()) for path in (args.runs / FOLDERS["4b"][0]).glob("*.json") if path.stem.isdigit()}
        summary["against_4b"] = against_smaller(records, [smaller[record["visit"]] for record in records])
    if any(arm in summary["arms"] for arm in TOOL_ARMS):
        tool_file = args.runs / TOOL_SUMMARY
        summary["tool_maps"] = tool_map_readings(records, json.loads(tool_file.read_text()) if tool_file.exists() else None)
    (args.runs / out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"{summary['scenes']} scenes, {summary['tasks']} tasks")
    for arm, row in summary["arms"].items():
        part = row["right_part"]
        print(f"  {row['name']}: right part {100 * part['rate']:.1f}% [{100 * part['low']:.1f}, {100 * part['high']:.1f}]; "
              f"right frame {100 * row['right_frame']['rate']:.1f}%; {row['seconds_per_question']:.1f} s and "
              f"{row['images_per_question']:.0f} images per question")
    for arm, row in summary["against_p4"].items():
        print(f"  P4 minus {arm}: {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}] -> {row['reading']}")
    if "choosing_keyframes_same_reading" in summary:
        row = summary["choosing_keyframes_same_reading"]
        print(f"  c2' minus c1' (choosing the keyframes, same reading as c3): {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}] -> {row['verdict']}")
    for arm, row in summary.get("against_4b", {}).items():
        print(f"  {arm}, this model minus 4B: {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]")
    if "choosing_keyframes" in summary:
        row = summary["choosing_keyframes"]
        print(f"  c2 minus c1 (what choosing the keyframes is worth): {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]")
    if "our_keyframes_vs_even_frames" in summary:
        row = summary["our_keyframes_vs_even_frames"]
        print(f"  c+ minus c3 (our keyframes against even frames): {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]")
    if "tool_maps" in summary:
        print_tool_readings(summary["tool_maps"])


def points(row: dict) -> str:
    return f"{100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]"


def print_tool_readings(readings: dict) -> None:
    deciding = readings["deciding_tool"]
    print(f"  section 24.9, deciding tool: {deciding['tool'] + ' (' + deciding['reason'] + ')' if deciding else 'none yet'}")
    if "annotated_p4_minus_annotated_c4" in readings:
        print(f"    annotated P4 minus annotated c4 (reference): {points(readings['annotated_p4_minus_annotated_c4'])}")
    for tool, rows in readings["tools"].items():
        stop, gap, split = rows["stop"], rows["gap"], rows["split"]
        if "stop" in stop:
            print(f"    {tool}: tool P4 minus tool c4 {points(stop['tool_p4_minus_tool_c4'])} ({stop['tool_p4_minus_tool_c4']['verdict']}), "
                  f"minus c3 {points(stop['tool_p4_minus_c3'])}, minus sam3q {points(stop['tool_p4_minus_sam3q'])}")
        print(f"      1 stop? {stop['reading']}")
        if "verdict" in gap:
            print(f"      2 gap: annotated P4 minus tool P4 {points(gap['annotated_p4_minus_tool_p4'])} -> {gap['reading']}")
        else:
            print(f"      2 gap: {gap['reading']}")
        if "misses" in split:
            print(f"      3 split: false alarms {points(split['false_alarms'])}, misses {points(split['misses'])}")


if __name__ == "__main__":
    main()
