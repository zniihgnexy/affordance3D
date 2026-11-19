"""Post-hoc diagnostic of a part-vector run (S0b, 2026-11-19): for rotations about an axis off the part (doors,
windows, lids), is the hinge error in the axis, in the side of the surface, or in where the surface's edge is found?

    python scripts/diagnose_door_hinges.py --config configs/part_vectors/s0b_val3.yaml

Same scenes, keyframes, masks (ORACLE) and depths as scripts/run_part_vectors.py, on each part's best keyframe. For
every off-part rotation it compares, by the same thresholds as the run (axis angle, distance of the origin to the
true axis line):

    rule         the hinge rule of affordance3d.mem.part_vectors (axis and origin from the surface's far edge)
    up_handle    axis = the vertical (gravity), origin = the contact point on the part
    up_far_end   axis = the vertical, origin = the far end of the flat surface along its most nearly level line
    up_right_end axis = the vertical, origin = whichever end of that line is nearer the truth (the side given:
                 the best any choice of side can do with these edges)

Writes <outputs>/<name>/door_hinges.json (per part and counts) next to the run's files.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_part_vectors as R  # noqa: E402

from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402
from affordance3d.mem import part_vectors as V  # noqa: E402

WAYS = ("rule", "up_handle", "up_far_end", "up_right_end")


def door_record(scene, frames: list, labels: list, part: int, config: dict) -> dict:
    """Axis angles and origin distances (to the true axis line) of every way, on the part's best keyframe."""
    truth = R.truth_of(scene, part)
    record = {"part": part, "label": scene.elements[part].label, "true_axis_vs_up_deg": round(V.line_angle_deg(truth.direction, scene.up), 1)}
    seen = V.observe(part, frames, labels)
    if not seen:
        return {**record, "seen": False}
    best, hinge_config = seen[0], V.HingeConfig(**config["hinge"])
    frame, label_map = frames[best.keyframe], labels[best.keyframe]
    to_axis = lambda point: round(V.point_line_distance(point, truth.origin, truth.direction), 3)  # noqa: E731
    up_deg = record["true_axis_vs_up_deg"]
    found = V.hinge(frame, label_map, part, best.pixel, best.contact, best.normal, hinge_config)
    ways = {"rule": None if found is None else (round(V.line_angle_deg(found.axis, truth.direction), 1), to_axis(found.origin)),
            "up_handle": (up_deg, to_axis(best.contact)), "up_far_end": None, "up_right_end": None}
    ends = V.level_scan(frame, label_map, part, best.pixel, best.contact, best.normal, scene.up, hinge_config)
    if ends is not None:
        far = ends[int(np.argmax([np.linalg.norm(end - best.contact) for end, _ in ends]))][0]
        ways["up_far_end"] = (up_deg, to_axis(far))
        ways["up_right_end"] = (up_deg, min(to_axis(end) for end, _ in ends))
        record["surface_width_m"] = round(float(np.linalg.norm(ends[0][0] - ends[1][0])), 3)
        record["an_end_at_image_border"] = any(border for _, border in ends)
    return {**record, "seen": True, "ways": ways}


def counts(records: list[dict], axis_deg: float, origin_m: float) -> dict:
    seen = [r for r in records if r["seen"]]
    out = {"doors": len(records), "seen": len(seen),
           "true_axis_vertical": sum(r["true_axis_vs_up_deg"] <= axis_deg for r in records),
           "an_end_at_image_border": sum(r.get("an_end_at_image_border", False) for r in seen)}
    for way in WAYS:
        done = [r["ways"][way] for r in seen if r["ways"][way] is not None]
        out[way] = {"axis_ok": sum(a <= axis_deg for a, _ in done), "axis_and_origin_ok": sum(a <= axis_deg and o < origin_m for a, o in done)}
    widths = [r["surface_width_m"] for r in seen if "surface_width_m" in r]
    out["surface_width_m_median"] = round(float(np.median(widths)), 3) if widths else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, "door_hinges", REPO_ROOT)
    records = {depth: [] for depth in config["depths"]}
    for visit, video in R.scene_list(config):
        scene, _, frames, labels = R.load_inputs(config, visit, video)
        parts, _ = R.scored_parts(scene, labels)
        doors = [p for p in parts if R.truth_of(scene, p).hinged]
        for depth in config["depths"]:
            records[depth].extend({"visit": visit, **door_record(scene, frames[depth], labels, p, config)} for p in doors)
    scoring = config["scoring"]
    summary = {depth: counts(recs, scoring["axis_deg"], scoring["origin_m"]) for depth, recs in records.items()}
    (folder / "door_hinges.json").write_text(json.dumps({"counts": summary, "parts": records}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
