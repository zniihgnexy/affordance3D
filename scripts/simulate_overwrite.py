"""Overwrite check for the memory-form discussion (decisions 3 and 6, week-2 notebook section 7): within one video of a
scene that does not change, how often SAM 3 finds a part on a keyframe that shows it, and how many of the parts it found
somewhere a "one view decides" update rule would erase. A descriptive count from the stage-0 inputs (SAM 3 tool maps on
the 40 steady keyframes, projected annotations), CPU only, no predictions.

    python scripts/simulate_overwrite.py --config configs/stage0/overwrite_sim.yaml

Rules: last_view, the newest keyframe that shows the part decides whether it is in the map; largest_view, the keyframe
where it is largest decides. Both are compared with "kept if ever found". A part found nowhere is never erased.
Writes <outputs>/<name>/parts.json (per part: keyframes shown, found on how many, what each rule keeps) and summary.json.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
_spec = importlib.util.spec_from_file_location("diagnose_task2_losses", REPO_ROOT / "scripts" / "diagnose_task2_losses.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)  # brings in the stage-0 scene loading (open3d first, as there)

from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402

RULES = ("last_view", "largest_view")


def part_outcome(sightings: list[tuple[int, int, bool]]) -> dict:
    """One part's sightings (keyframe, area in pixels, found by SAM 3), in keyframe order -> counts and rule outcomes."""
    if not sightings:
        raise ValueError("a part needs at least one sighting")
    found = [hit for _, _, hit in sightings]
    largest = max(range(len(sightings)), key=lambda i: sightings[i][1])  # first of equal areas
    return {"shown": len(found), "found": sum(found), "ever": any(found),
            "kept_last_view": found[-1], "kept_largest_view": found[largest]}


def summarise(parts: list[dict]) -> dict:
    """Per-frame find rates and, for each rule, the share of ever-found parts it erases (all parts, and parts shown twice or more)."""
    ever = [p for p in parts if p["ever"]]
    twice = [p for p in ever if p["shown"] >= 2]
    rate = lambda group: sum(p["found"] for p in group) / sum(p["shown"] for p in group) if group else None  # noqa: E731
    erased = {rule: {"all": sum(not p[f"kept_{rule}"] for p in ever) / len(ever) if ever else None,
                     "shown_twice_or_more": sum(not p[f"kept_{rule}"] for p in twice) / len(twice) if twice else None}
              for rule in RULES}
    return {"parts": len(parts), "ever_found": len(ever), "ever_found_shown_twice_or_more": len(twice),
            "per_frame_found_rate": rate(parts), "per_frame_found_rate_ever_found": rate(ever), "erased": erased}


def scene_rows(visit: str, tool_dir: Path, n_keyframes: int) -> list[dict]:
    """Every part the keyframes of this visit show, with its sightings and outcome."""
    P, TM = D.P, D.TM
    scene = P.load_scene(D.DATA, visit, D.video_of(visit))
    frames = P.gt_frames(scene, D.keyframes.scene_keyframes(scene, "steady", n_keyframes))
    labels = P.label_maps(scene, frames)
    detections = TM.tool_detections(TM.read_tool_map(tool_dir / f"{visit}.json"))
    found = TM.sightings(labels, {k: [d.box for d in detections if d.keyframe == k] for k in range(len(labels))})
    rows = []
    for part, shown in TM.visible_parts(labels).items():
        sightings = [(k, int((labels[k] == part).sum()), bool(found[(part, k)])) for k in sorted(shown)]
        rows.append({"visit": visit, "part": part, "label": scene.elements[part].label,
                     "keyframes": [k for k, _, _ in sightings], **part_outcome(sightings)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], help="overrides key=value")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, "simulate", REPO_ROOT)
    parts = []
    for visit in [v.strip() for v in (REPO_ROOT / config["scenes"]).read_text().splitlines() if v.strip()]:
        parts += scene_rows(visit, REPO_ROOT / config["tool_maps"], config["keyframes"])
        print(f"{visit}: {len(parts)} parts so far", flush=True)
    summary = summarise(parts)
    (folder / "parts.json").write_text(json.dumps(parts, indent=1, ensure_ascii=False))
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
