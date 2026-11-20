"""Stage 0, tests T1 and T2 (proposal section 13; configs/stage0/task2_losses.yaml): where the memory arm loses Task 2,
what the parts SAM 3 never finds look like, and where the memory beats the raw video. CPU only, from existing results.

    python scripts/diagnose_task2_losses.py --config configs/stage0/task2_losses.yaml

A task's hit is the AP25 hit of the public evaluation's scorer (scripts/scenefun3d_eval.py, Fun3DU protocol: the
prediction's point precision on the target at least 0.25). The memory arm's misses end in one stage, checked in this
order: not_in_keyframes (no target part visible in a keyframe), never_found (SAM 3 found no target part on a keyframe
that shows it, tool-map rule), not_chosen (the VLM's chosen box is not on the target, controls "hit"), lost_in_mask
(chosen right, the lifted mask misses). Parts are those the 40 steady keyframes show (at least MIN_PART_PIXELS pixels);
a part's area is in working-resolution pixels, its depth the median rendered depth on it in its largest view.

Writes under <outputs>/<name>/: tasks.json, parts.json, summary.json (measures, and each prediction marked held or not).
"""
import argparse
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (imported quietly before the pipeline, which needs it)

from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402
from affordance3d.mem import keyframes  # noqa: E402
from affordance3d.mem import phase1_stats as A  # noqa: E402
from affordance3d.mem import pipeline as P  # noqa: E402
from affordance3d.mem import tool_maps as TM  # noqa: E402

DATA = REPO_ROOT / "data" / "scenefun3d"
_spec = importlib.util.spec_from_file_location("scenefun3d_eval_cli", REPO_ROOT / "scripts" / "scenefun3d_eval.py")
EVAL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EVAL)
STAGES = ("hit", "not_in_keyframes", "never_found", "not_chosen", "lost_in_mask")
MIN_LABEL_PARTS = 20  # T1-P3 compares labels with at least this many parts


def video_of(visit: str) -> str:
    videos = [p.name for p in (DATA / visit).iterdir() if p.is_dir() and p.name.isdigit()]
    if len(videos) != 1:
        raise SystemExit(f"{visit}: expected one video folder, found {videos}")
    return videos[0]


def scene_parts(visit: str, tool_dir: Path) -> tuple[P.Scene, dict[int, dict]]:
    """Every part the keyframes show: label, keyframes, largest area, depth there, found by SAM 3 or not."""
    scene = P.load_scene(DATA, visit, video_of(visit))
    frames = P.gt_frames(scene, keyframes.scene_keyframes(scene, "steady", 40))
    labels = P.label_maps(scene, frames)
    detections = TM.tool_detections(TM.read_tool_map(tool_dir / f"{visit}.json"))
    found = TM.sightings(labels, {k: [d.box for d in detections if d.keyframe == k] for k in range(len(labels))})
    parts = {}
    for part, shown in TM.visible_parts(labels).items():
        areas = [int((labels[k] == part).sum()) for k in shown]
        best = shown[int(np.argmax(areas))]
        depth = frames[best].depth_m[(labels[best] == part) & (frames[best].depth_m > 0)]
        parts[part] = {"visit": visit, "part": part, "label": scene.elements[part].label, "keyframes": shown,
                       "area": max(areas), "depth_m": float(np.median(depth)) if depth.size else None,
                       "found": any(found[(part, k)] for k in shown)}
    return scene, parts


def stage_of(hit: bool, visible: bool, found: bool, chosen: bool) -> str:
    if hit:
        return "hit"
    if not visible:
        return "not_in_keyframes"
    if not found:
        return "never_found"
    return "lost_in_mask" if chosen else "not_chosen"


def scene_tasks(visit: str, scene: P.Scene, parts: dict[int, dict], hits: dict[str, list[dict]], controls: dict,
                config: dict) -> list[dict]:
    """One row per task of the scene, in the descriptions' order (the scorer's order)."""
    descriptions = json.loads((DATA / visit / f"{visit}_descriptions.json").read_text())["descriptions"]
    chosen = dict(zip(controls["desc_ids"], (row["hit"] for row in controls["arms"][config["memory_arm"]]), strict=True))
    label_counts: dict[str, int] = {}
    for element in scene.elements:
        label_counts[element.label] = label_counts.get(element.label, 0) + 1
    rows = []
    for i, d in enumerate(descriptions):
        targets = [p for p, e in enumerate(scene.elements) if e.annot_id in d["annot_id"]]
        seen = [p for p in targets if p in parts]
        label = scene.elements[targets[0]].label if targets else None
        row = {"visit": visit, "desc_id": d["desc_id"], "label": label, "same_label": label_counts.get(label, 0),
               "visible": bool(seen), "found": any(parts[p]["found"] for p in seen), "chosen": bool(chosen[d["desc_id"]]),
               "area": max((parts[p]["area"] for p in seen), default=0),
               "keyframes": len({k for p in seen for k in parts[p]["keyframes"]})}
        row.update({f"hit_{arm}": bool(hits[arm][i]["AP25"]) for arm in hits})
        row["stage"] = stage_of(row[f"hit_{config['memory_arm']}"], row["visible"], row["found"], row["chosen"])
        rows.append(row)
    return rows


def share(rows: list[dict], test) -> float | None:
    return float(np.mean([bool(test(r)) for r in rows])) if rows else None


def paired(tasks: list[dict], first: str, second: str, keep, config: dict) -> dict:
    """first minus second hit rate over the kept tasks, 95% interval by resampling scenes that hold any of them."""
    by_scene: dict[str, list[dict]] = {}
    for row in tasks:
        if keep(row):
            by_scene.setdefault(row["visit"], []).append(row)
    a = [np.array([r[f"hit_{first}"] for r in rows], dtype=float) for rows in by_scene.values()]
    b = [np.array([r[f"hit_{second}"] for r in rows], dtype=float) for rows in by_scene.values()]
    if not a:
        return {"tasks": 0}
    rate, low, high = A.paired_difference(a, b, resamples=config["resamples"], seed=config["seed"])
    return {"tasks": sum(len(x) for x in a), "scenes": len(a), "diff": 100 * rate, "low": 100 * low, "high": 100 * high}


def summarise(tasks: list[dict], parts: list[dict], config: dict) -> dict:
    mem, ann, raw = config["memory_arm"], config["annotated_arm"], config["raw_arm"]
    lost = [r for r in tasks if r[f"hit_{ann}"] and not r[f"hit_{mem}"]]
    stages_all = {s: sum(r["stage"] == s for r in tasks) for s in STAGES}
    stages_lost = {s: sum(r["stage"] == s for r in lost) for s in STAGES}
    largest = max(stages_lost, key=stages_lost.get) if lost else None
    found_area = np.median([p["area"] for p in parts if p["found"]])
    missed_area = np.median([p["area"] for p in parts if not p["found"]])
    by_label = {}
    for label in sorted({p["label"] for p in parts}):
        group = [p for p in parts if p["label"] == label]
        by_label[label] = {"parts": len(group), "never_found": share(group, lambda p: not p["found"])}
    big = {k: v["never_found"] for k, v in by_label.items() if v["parts"] >= MIN_LABEL_PARTS}
    spread = max(big.values()) - min(big.values()) if len(big) >= 2 else None
    area_cut = float(np.median([r["area"] for r in tasks if r["visible"]]))
    strata = {"all": paired(tasks, mem, raw, lambda r: True, config),
              "small": paired(tasks, mem, raw, lambda r: r["visible"] and r["area"] < area_cut, config),
              "large": paired(tasks, mem, raw, lambda r: r["visible"] and r["area"] >= area_cut, config),
              "one_of_label": paired(tasks, mem, raw, lambda r: r["same_label"] == 1, config),
              "several_of_label": paired(tasks, mem, raw, lambda r: r["same_label"] >= 2, config),
              "few_keyframes": paired(tasks, mem, raw, lambda r: r["keyframes"] <= config["few_keyframes"], config),
              "many_keyframes": paired(tasks, mem, raw, lambda r: r["keyframes"] > config["few_keyframes"], config)}
    quartiles = np.percentile([p["area"] for p in parts], [25, 50, 75])
    by_size = {f"q{i + 1}": share([p for p in parts if np.searchsorted(quartiles, p["area"], side="right") == i],
                                  lambda p: not p["found"]) for i in range(4)}
    preds = {
        "T1-P1": {"held": bool(lost) and largest == "never_found" and stages_lost["never_found"] >= 0.5 * len(lost),
                  "largest_stage": largest, "never_found_share": stages_lost["never_found"] / len(lost) if lost else None},
        "T1-P2": {"held": bool(missed_area < 0.5 * found_area), "median_area_never_found": float(missed_area),
                  "median_area_found": float(found_area)},
        "T1-P3": {"held": spread is not None and spread >= 0.20, "spread": spread, "labels_compared": big},
        "T2-P1": {"held": strata["small"].get("diff", 0) > strata["large"].get("diff", 0)},
        "T2-P2": {"held": strata["several_of_label"].get("diff", 0) > strata["one_of_label"].get("diff", 0)},
    }
    concentrated = preds["T1-P2"]["held"] or preds["T1-P3"]["held"]
    if largest in ("not_chosen", "lost_in_mask"):
        reading = "the gap is in reading or lifting, not in the maps"
    elif preds["T1-P1"]["held"]:
        reading = ("misses concentrated: a targeted fix has a clear target; Task 2 stays a candidate pending one GPU try"
                   if concentrated else "misses spread out: Task 2 becomes a supporting metric")
    else:
        reading = "never_found is not the main stage; read the stage table"
    return {"tasks": len(tasks), "parts": len(parts), "stages_memory_arm": stages_all,
            "lost_to_annotated": {"tasks": len(lost), "stages": stages_lost},
            "never_found_by_label": by_label, "never_found_by_area_quartile": by_size,
            "area_quartiles": [float(q) for q in quartiles], "task_area_median": area_cut,
            "memory_minus_raw": strata, "predictions": preds, "reading_t1": reading,
            "hit_definition": "AP25 hit of the public evaluation's scorer: point precision on the target at least 0.25"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], help="overrides key=value")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, "diagnose", REPO_ROOT)
    pe = REPO_ROOT / config["public_eval"] / "lifted"
    arms = (config["memory_arm"], config["annotated_arm"], config["raw_arm"])
    rows_by_arm = {arm: EVAL.task_rows(pe / arm) for arm in arms}
    tasks, parts = [], []
    for visit in [v.strip() for v in (REPO_ROOT / config["scenes"]).read_text().splitlines() if v.strip()]:
        scene, scene_part_rows = scene_parts(visit, REPO_ROOT / config["tool_maps"])
        controls = json.loads((REPO_ROOT / config["controls"] / f"{visit}.json").read_text())
        tasks += scene_tasks(visit, scene, scene_part_rows, {arm: rows_by_arm[arm][visit] for arm in arms}, controls, config)
        parts += list(scene_part_rows.values())
        print(f"{visit}: {len(scene_part_rows)} parts, {len(tasks)} tasks so far", flush=True)
    summary = summarise(tasks, parts, config)
    for name, payload in (("tasks.json", tasks), ("parts.json", parts), ("summary.json", summary)):
        (folder / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("stages_memory_arm", "lost_to_annotated", "predictions", "reading_t1")}, indent=1))
    print(json.dumps(summary["memory_minus_raw"], indent=1))


if __name__ == "__main__":
    main()
