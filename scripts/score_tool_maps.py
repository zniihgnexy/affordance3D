"""Score the affordance maps built by off-the-shelf tools (discussion draft, section 24.9) and name the tool that decides.

    python scripts/score_tool_maps.py      # reads runs/mem_phase1/tool_maps/<tool>/<visit>.json, writes runs/mem_phase1/tool_maps_summary.json

CPU only, from the saved boxes and the projected annotations (ground truth, used for scoring only). For each tool:

    per-part recall      of the parts visible in the 40 steady keyframes (481 in phase 1), the share found on some
                         keyframe that shows it: a box there has its centre on the part widened by 8 pixels
    recall by budget     the same with only 10 (every 4th) and 20 (every 2nd) keyframes, over the same parts; the gain
                         40 minus 10 is a paired difference over the same parts. With the annotated boxes as the tool it
                         is +44 points (only 271 of the 481 parts are visible in the 10 keyframes), so the same gain among
                         the parts the 10 keyframes show is also stored (descriptive; see aggregate_mem_controls.GAIN_KEY)
    always missed        of the parts visible in the keyframes used, the share missed on every one of them
    false alarms         per keyframe, the boxes whose centre lies on no annotated part
    coverage per task    whether a box the reading step would show (the memory's three best keyframes, at most 40 boxes)
                         lies on the target part; needs the crop vectors written by run_mem_controls.py's tool arms and is
                         left out until they exist. Annotated P4 is scored the same way for reference (83.6% by the
                         phase-1 rule)
    write time           seconds the tool took for the 40 keyframes of a scene (model loading not included)

Intervals come from resampling scenes. The deciding tool is the one with the higher per-part recall at 40 keyframes;
when the two intervals overlap it is a tie and SAM 3 decides. It is named only when both maps are complete in every scene.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the log)

from affordance3d.mem import keyframes
from affordance3d.mem import phase1_diagnostics as D
from affordance3d.mem import phase1_stats as A
from affordance3d.mem import pipeline as P
from affordance3d.mem import policies
from affordance3d.mem import tool_maps as TM
from report import mem_phase1_helpers as R
from report import mem_steps_helpers as H

OUT = "tool_maps_summary.json"


def triple(values: tuple[float, float, float]) -> dict:
    return {"rate": values[0], "low": values[1], "high": values[2]}


def scene_measures(record: dict, labels: list[np.ndarray]) -> dict:
    """Everything one tool's map of one scene is scored by, per part, per keyframe and per budget."""
    n = len(labels)
    detections = TM.tool_detections(record)
    boxes = {k: [d.box for d in detections if d.keyframe == k] for k in range(n)}
    found = TM.sightings(labels, boxes)
    outcomes = {used: TM.part_outcomes(found, TM.budget_keyframes(n, used)) for used in TM.BUDGETS}
    return {"outcomes": outcomes, "false_alarms": np.array([TM.false_alarms(labels[k], boxes[k]) for k in range(n)], dtype=float),
            "boxes": np.array([len(boxes[k]) for k in range(n)], dtype=float), "seconds": TM.tool_seconds(record),
            "sightings_found": np.array(list(found.values()), dtype=float), "detections": detections}


def coverage(candidates: list[policies.Candidate], vectors: np.ndarray, saved: dict, tasks: list[H.Task],
             labels: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Per task: whether a shown box lies on the target part (common rule, phase-1 rule)."""
    rows = []
    for query, task in zip(saved["queries"], tasks):
        shown = D.shown_candidates(query, saved["keyframes"], vectors, candidates, TM.TOP_FRAMES, TM.MAX_MARKS)
        rows.append(TM.shown_on_target(shown, labels, task.parts, R.target_pixels(labels, task.parts)))
    rows = np.array(rows, dtype=float).reshape(len(tasks), 2)
    return {"common": rows[:, 0], "phase1": rows[:, 1]}


def tool_coverage(runs: Path, tool: str, visit: str, measures: dict, saved: dict, tasks: list[H.Task], labels: list[np.ndarray],
                  image_hw: tuple[int, int]) -> dict[str, np.ndarray] | None:
    """Coverage of a tool's map, from the crop vectors run_mem_controls.py saved; None until they exist."""
    file = runs / "tool_maps" / tool / f"features_{visit}.npz"
    if not file.exists():
        return None
    candidates = TM.candidates_of(measures["detections"], image_hw)
    with np.load(file) as features:
        if not np.array_equal(features["boxes"], np.array([c.box for c in candidates], dtype=np.int64).reshape(-1, 4)):
            raise SystemExit(f"{file} was encoded from other boxes than the {tool} map holds now")
        vectors, measures["encode_seconds"] = features["vectors"], float(features["seconds"])
    return coverage(candidates, vectors, saved, tasks, labels)


def annotated_coverage(saved: dict, tasks: list[H.Task], labels: list[np.ndarray]) -> dict[str, np.ndarray]:
    """The same for the annotated regions of phase 1 (the reference: 83.6% by the phase-1 rule)."""
    candidates = [policies.Candidate(int(k), tuple(int(v) for v in box), (0, 0, 0, 0))
                  for k, box in zip(saved["keyframe/P4"], saved["boxes/P4"])]
    return coverage(candidates, saved["vectors/P4"], saved, tasks, labels)


def load_scene_truth(runs: Path, result: dict) -> tuple[list[np.ndarray], list[H.Task], dict, tuple[int, int]]:
    """Label maps of the 40 steady keyframes, the answerable tasks, the saved phase-1 vectors, and the image size."""
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", result["visit"], result["video"])
    steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
    labels = P.label_maps(scene, P.gt_frames(scene, steady))
    seen = len(TM.visible_parts(labels))
    if seen != result["parts_seen"]["steady"]:
        raise SystemExit(f"{result['visit']}: {seen} parts visible now, {result['parts_seen']['steady']} in phase 1")
    tasks = [task for task in H.tasks_of(scene, labels) if task.keyframes]
    if [task.text for task in tasks] != result["tasks"]["texts"]:
        raise SystemExit(f"{result['visit']}: the tasks differ from those of the phase-1 run")
    with np.load(runs / "diagnostics" / f"{result['visit']}.npz") as saved:
        arrays = {key: saved[key] for key in ("keyframes", "queries", "vectors/P4", "keyframe/P4", "boxes/P4")}
    return labels, tasks, arrays, P.frame_hw(scene, steady[0], P.PIXEL_STEP)


def summarise_tool(scenes: list[dict], tool: str) -> dict:
    """One tool over all scenes it is complete in."""
    rows = [scene["tools"][tool] for scene in scenes]
    recall = {used: A.bootstrap([row["outcomes"][used]["found"].astype(float) for row in rows]) for used in TM.BUDGETS}
    missed = {used: A.bootstrap([row["outcomes"][used]["always_missed"][row["outcomes"][used]["visible"]].astype(float)
                                 for row in rows]) for used in TM.BUDGETS}
    gain = A.paired_difference([row["outcomes"][40]["found"].astype(float) for row in rows],
                               [row["outcomes"][10]["found"].astype(float) for row in rows])
    seen = [row["outcomes"][10]["visible"] for row in rows]  # descriptive: the gain among parts the 10 keyframes show,
    gain_seen = A.paired_difference([row["outcomes"][40]["found"][s].astype(float) for row, s in zip(rows, seen)],  # free of
                                    [row["outcomes"][10]["found"][s].astype(float) for row, s in zip(rows, seen)])  # visibility
    seconds = [row["seconds"] for row in rows]
    labels = [TM.part_name(d.label) for row in rows for d in row["detections"]]
    summary = {"scenes": len(rows), "parts": int(sum(len(row["outcomes"][40]["found"]) for row in rows)),
               "recall": {str(used): {**triple(recall[used]), "found": int(sum(row["outcomes"][used]["found"].sum() for row in rows)),
                                      "visible_in_keyframes_used": int(sum(row["outcomes"][used]["visible"].sum() for row in rows))}
                          for used in TM.BUDGETS},
               "always_missed": {str(used): triple(missed[used]) for used in TM.BUDGETS},
               "recall_gain_40_minus_10": triple(gain),
               "recall_gain_40_minus_10_among_visible": triple(gain_seen),
               "sightings_found": triple(A.bootstrap([row["sightings_found"] for row in rows])),
               "false_alarms_per_keyframe": triple(A.bootstrap([row["false_alarms"] for row in rows])),
               "boxes_per_keyframe": float(np.mean(np.concatenate([row["boxes"] for row in rows]))),
               "boxes_by_name": {name: labels.count(name) for name in (*TM.PART_NAMES, "other") if labels.count(name)},
               "write_seconds_per_scene": {"median": float(np.median(seconds)), "mean": float(np.mean(seconds)),
                                           "total_hours": float(np.sum(seconds) / 3600)}}
    covered = [row.get("coverage") for row in rows]
    if all(item is not None for item in covered):
        summary["coverage"] = {rule: triple(A.bootstrap([item[rule] for item in covered])) for rule in ("common", "phase1")}
        summary["encode_seconds_per_scene"] = float(np.median([row["encode_seconds"] for row in rows]))
    else:
        summary["coverage"] = "not yet: run the tool arms of scripts/run_mem_controls.py, which encode the crops"
    return summary


def score_scene(runs: Path, result: dict, tools: dict[str, dict]) -> dict:
    labels, tasks, saved, image_hw = load_scene_truth(runs, result)
    scene = {"visit": result["visit"], "tools": {}, "annotated_coverage": annotated_coverage(saved, tasks, labels)}
    for tool, record in tools.items():
        measures = scene_measures(record, labels)
        measures["coverage"] = tool_coverage(runs, tool, result["visit"], measures, saved, tasks, labels, image_hw)
        scene["tools"][tool] = measures
    return scene


def summarise(scenes: list[dict], incomplete: dict[str, list[str]], all_visits: list[str]) -> dict:
    tools = [tool for tool in TM.TOOLS if all(tool in scene["tools"] for scene in scenes)]
    summary = {"scenes": len(scenes), "visits": [scene["visit"] for scene in scenes], "incomplete": incomplete,
               "choices": {"score_threshold": TM.SCORE_THRESHOLD, "duplicates_merged_at_iou": TM.MERGE_IOU,
                           "budgets": {str(used): TM.budget_keyframes(TM.BUDGETS[-1], used) for used in TM.BUDGETS},
                           "visible": f"at least {P.MIN_PART_PIXELS} pixels in a keyframe", "version": TM.VERSION},
               "annotated_coverage": {rule: triple(A.bootstrap([scene["annotated_coverage"][rule] for scene in scenes]))
                                      for rule in ("common", "phase1")},
               "tools": {tool: summarise_tool(scenes, tool) for tool in tools}}
    complete = len(tools) == len(TM.TOOLS) and len(scenes) == len(all_visits)
    if complete:
        recall = {tool: tuple(summary["tools"][tool]["recall"]["40"][key] for key in ("rate", "low", "high")) for tool in tools}
        summary["deciding_tool"] = TM.deciding_tool(recall)
        first, second = tools
        summary["recall_difference"] = {"pair": f"{first} minus {second}", **triple(A.paired_difference(
            [scene["tools"][first]["outcomes"][40]["found"].astype(float) for scene in scenes],
            [scene["tools"][second]["outcomes"][40]["found"].astype(float) for scene in scenes]))}
    else:
        summary["deciding_tool"] = None
    return summary


def complete_maps(runs: Path, result: dict) -> tuple[dict[str, dict], list[str]]:
    """The tool maps of one scene that cover all its keyframes, and the tools whose map is missing or unfinished."""
    maps, unfinished = {}, []
    for tool in TM.TOOLS:
        file = runs / "tool_maps" / tool / f"{result['visit']}.json"
        record = json.loads(file.read_text()) if file.exists() else None
        if record is None or TM.missing_keyframes(record, result["n_keyframes"]):
            unfinished.append(tool)
        else:
            maps[tool] = record
    return maps, unfinished


def report(summary: dict) -> None:
    print(f"{summary['scenes']} scenes; incomplete maps: {summary['incomplete'] or 'none'}")
    for rule, row in summary["annotated_coverage"].items():
        print(f"  annotated P4, a shown box on the target ({rule} rule): {100 * row['rate']:.1f}% [{100 * row['low']:.1f}, {100 * row['high']:.1f}]")
    for tool, row in summary["tools"].items():
        recall = ", ".join(f"{used} keyframes {100 * cell['rate']:.1f}% [{100 * cell['low']:.1f}, {100 * cell['high']:.1f}]"
                           for used, cell in row["recall"].items())
        gain, alarms = row["recall_gain_40_minus_10"], row["false_alarms_per_keyframe"]
        print(f"  {TM.TOOL_NAMES[tool]}: recall of {row['parts']} parts: {recall}")
        print(f"    40 minus 10 keyframes {100 * gain['rate']:+.1f} [{100 * gain['low']:+.1f}, {100 * gain['high']:+.1f}]; "
              f"always missed (40) {100 * row['always_missed']['40']['rate']:.1f}%; false alarms {alarms['rate']:.2f} per keyframe "
              f"[{alarms['low']:.2f}, {alarms['high']:.2f}]; {row['write_seconds_per_scene']['median']:.0f} s per scene")
        if isinstance(row["coverage"], dict):
            cell = row["coverage"]["common"]
            print(f"    a shown box on the target: {100 * cell['rate']:.1f}% [{100 * cell['low']:.1f}, {100 * cell['high']:.1f}]")
    deciding = summary["deciding_tool"]
    print(f"  deciding tool: {deciding['tool']} ({deciding['reason']})" if deciding else "  deciding tool: not named (maps incomplete)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    results = A.scored(A.load_results(args.runs))
    scenes, incomplete = [], {}
    for result in results:
        maps, unfinished = complete_maps(args.runs, result)
        for tool in unfinished:
            incomplete.setdefault(tool, []).append(result["visit"])
        if maps:
            scenes.append(score_scene(args.runs, result, maps))
            print(f"{result['visit']}: scored {sorted(maps)}", flush=True)
    if not scenes:
        raise SystemExit(f"no finished tool map in {args.runs / 'tool_maps'}")
    summary = summarise([scene for scene in scenes if set(scene["tools"]) == set(TM.TOOLS)] or scenes, incomplete,
                        [result["visit"] for result in results])
    (args.runs / OUT).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    report(summary)


if __name__ == "__main__":
    main()
