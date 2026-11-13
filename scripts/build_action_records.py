"""C-0 (discussion draft, section 25.3): action records from the SceneFun3D ground truth for the 390 answerable tasks.

    python scripts/build_action_records.py                    # the 30 phase-1 scenes; writes runs/mem_c0/
    python scripts/build_action_records.py --visits 421254    # one scene, as a dry run (the summary then covers it only)
    python scripts/build_action_records.py --checks-only      # redraw the check images from runs/mem_c0/records.json

CPU only. For every answerable task (as in phase 1: a target part shows in at least one steady keyframe) the record
holds the parts, their affordance labels and motion types; a translation's signed direction (official viewer:
`inwards` -> -motion_dir); a rotation's axis and origin; and, in the keyframe that shows the task's parts best, each
part's contact pixel, its direction drawn 0.2 m long (the viewer's arrow; the axis for a rotation) and projected,
push or pull, the axis class and the hinge side (affordance3d.mem.action_records). Fixed before the run: go on to
C-1 only if all three hold:

    1. at least 90% of the tasks have a motion for every target part;
    2. for at least 80% of the tasks the projected direction of the shown part (the target part with most pixels
       in that keyframe) is at least 20 px long, at the working resolution of phase 1 (PIXEL_STEP);
    3. the scorer gives 100% to the ground truth itself (every field, every task).

It then draws 10 parts (3 pushed, 3 pulled, 2 hinged, 2 turning in place; at most one per scene; seed 0) with their
arrows on the keyframe, to check the sign conventions by eye (runs/mem_c0/checks/). If the conventions prove
inconsistent, the rule is to score the axis orientation only (action_records.score(..., axis_only=True)).

Writes runs/mem_c0/records.json (one row per task), summary.json and checks/. Ground truth only (part ids, motions,
laser scan, registered cameras): nothing here is a model input.
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401,E402  (its banner would end up in the log)

from affordance3d.frames import Frame  # noqa: E402
from affordance3d.geometry import project  # noqa: E402
from affordance3d.mem import action_records as A  # noqa: E402
from affordance3d.mem import keyframes  # noqa: E402
from affordance3d.mem import pipeline as P  # noqa: E402
from report import mem_steps_helpers as H  # noqa: E402

MOTIONS_SHARE, ARROW_SHARE = 0.9, 0.8  # section 25.3
STRATA = (("pushed", 3), ("pulled", 3), ("hinged", 2), ("turning in place", 2))
CHECK_SEED = 0
TANGENT_M = 0.1
ZOOM_PAD_PX = 60
AXIS_TIE = 0.1  # descriptive: the two largest camera components of a rotation axis this close make its class fragile


def _list(vector: np.ndarray | None) -> list[float] | None:
    return None if vector is None else [round(float(v), 5) for v in vector]


def part_motions(scene: P.Scene) -> dict[int, A.PartMotion]:
    """Every part of the scene that has a motion annotation, by element index."""
    return {p: A.part_motion(scene.motions[e.annot_id], scene.part_points(p), scene.scan_xyz)
            for p, e in enumerate(scene.elements) if e.annot_id in scene.motions}


def motion_row(motion: A.PartMotion | None) -> dict | None:
    if motion is None:
        return None
    return {"type": motion.kind, "viz_orient": motion.viz_orient, "direction": _list(motion.direction),
            "origin": _list(motion.origin), "centre": _list(motion.centre), "axis_distance_m": round(motion.axis_distance_m, 4),
            "opening_tangent": _list(A.opening_tangent(motion)) if motion.kind == "rot" else None}


def view_row(view: A.PartView) -> dict:
    return {"pixels": view.pixels, "contact": view.contact, "arrow": view.arrow, "arrow_px": round(A.segment_px(view.arrow), 2),
            "sense": view.sense, "axis_class": view.axis_class, "hinge_side": view.hinge_side}


def diagnostics(record: A.ActionRecord, frame: Frame) -> dict:
    """Descriptive checks of the sign conventions on the shown part (no part of the pass rule)."""
    motion, label = record.motions[record.shown], record.labels[record.shown]
    view = record.views[record.shown]
    out = {"sense_matches_label": None, "tangent_towards_camera": None, "anchor_in_image": None, "arrow_px_reversed": None,
           "axis_class_margin": None}
    if motion is None:
        return out
    out["anchor_in_image"] = bool(project(motion.anchor[None], frame)[2][0])
    if motion.kind == "trans":  # the length criterion under the opposite reading of the translation sign
        out["arrow_px_reversed"] = round(A.segment_px(A.projected_segment(motion.anchor, -motion.direction, frame)), 2)
    if motion.kind == "trans" and label in A.EXPECTED_SENSE:
        out["sense_matches_label"] = view.sense == A.EXPECTED_SENSE[label]
    if motion.hinged:
        out["tangent_towards_camera"] = bool(A.opening_tangent(motion) @ (A.camera_centre(frame) - motion.centre) > 0)
    if motion.kind == "rot":  # how clear the axis class is: a near tie of two camera components makes it fragile
        components = np.sort(np.abs(frame.cam_to_world[:3, :3].T @ motion.direction))
        out["axis_class_margin"] = round(float(components[2] - components[1]), 3)
    return out


def record_row(visit: str, keyframe_times: list[str], annot_ids: tuple[str, ...], record: A.ActionRecord,
               labels: np.ndarray, frame: Frame) -> dict:
    truth = A.truth_as_prediction(record)
    return {"visit": visit, "task": record.task, "keyframe": keyframe_times[record.keyframe], "annot_ids": list(annot_ids),
            "has_all_motions": all(a is not None for a in record.motions) and len(record.parts) == len(annot_ids),
            "parts": [{"part": p, "annot_id": a, "label": lab, "motion": motion_row(m), "view": view_row(v)}
                      for p, a, lab, m, v in zip(record.parts, record.annot_ids, record.labels, record.motions, record.views)],
            "shown": record.shown, "arrow_px": round(record.arrow_px, 2),
            "self_score": A.score(truth, record, labels), "contrary_score": A.score(A.contrary(truth, labels.shape), record, labels),
            "diagnostics": diagnostics(record, frame)}


def scene_rows(runs: Path, data_root: Path, visit: str) -> list[dict]:
    """The records of one scene's answerable tasks, checked against the phase-1 run."""
    result = json.loads((runs / f"{visit}.json").read_text())
    scene = P.load_scene(data_root, visit, result["video"])
    steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
    frames = P.gt_frames(scene, steady)
    labels = P.label_maps(scene, frames)
    pairs = [(ids, task) for (_, ids), task in zip(scene.tasks, H.tasks_of(scene, labels), strict=True) if task.keyframes]
    if [task.text for _, task in pairs] != result["tasks"]["texts"]:
        raise SystemExit(f"{visit}: the tasks differ from those of the phase-1 run")
    motions, rows = part_motions(scene), []
    for annot_ids, task in pairs:
        elements = [scene.elements[p] for p in task.parts]
        record = A.build_record(task.text, task.parts, [e.annot_id for e in elements], [e.label for e in elements],
                                [motions.get(p) for p in task.parts], labels, frames)
        rows.append(record_row(visit, steady, annot_ids, record, labels[record.keyframe], frames[record.keyframe]))
    return rows


# ---------------------------------------------------------------- summary

def _quantiles(values: list[float]) -> dict[str, float] | None:
    return {f"q{q}": round(float(np.percentile(values, q)), 2) for q in (10, 25, 50, 75, 90)} if values else None


def _shown(row: dict) -> dict:
    return row["parts"][row["shown"]]


def _kind(row: dict) -> str:
    motion = _shown(row)["motion"]
    return "no motion" if motion is None else motion["type"]


def _counts(values: list) -> dict[str, int]:
    keys = sorted({str(v) for v in values})
    return {k: sum(str(v) == k for v in values) for k in keys}


def _share(values: list) -> dict:
    kept = [v for v in values if v is not None]
    return {"share": float(np.mean(kept)) if kept else None, "n": len(kept)}


def criteria(rows: list[dict]) -> dict:
    with_motions = [row["has_all_motions"] for row in rows]
    long_enough = [row["arrow_px"] >= A.MIN_ARROW_PX for row in rows]
    full_res = [row["arrow_px"] * P.PIXEL_STEP >= A.MIN_ARROW_PX for row in rows]
    reversed_px = [row["arrow_px"] if row["diagnostics"]["arrow_px_reversed"] is None else row["diagnostics"]["arrow_px_reversed"]
                   for row in rows]
    self_shares = A.shares([row["self_score"] for row in rows])
    perfect = all(v["share"] == 1.0 for v in self_shares.values()) and self_shares["joint"]["n"] == len(rows)
    largest_error = max((row["self_score"][k] or 0.0) for row in rows for k in ("arrow_deg", "direction_deg"))
    return {"motions_for_every_part": {"share": float(np.mean(with_motions)), "threshold": MOTIONS_SHARE,
                                       "holds": bool(np.mean(with_motions) >= MOTIONS_SHARE)},
            "arrow_at_least_20px": {"share": float(np.mean(long_enough)), "threshold": ARROW_SHARE,
                                    "holds": bool(np.mean(long_enough) >= ARROW_SHARE),
                                    "share_at_full_resolution": float(np.mean(full_res)),
                                    "share_with_translations_reversed": float(np.mean([px >= A.MIN_ARROW_PX for px in reversed_px])),
                                    "by_type": {k: _share([row["arrow_px"] >= A.MIN_ARROW_PX for row in rows if _kind(row) == k])
                                                for k in ("trans", "rot", "no motion")}},
            "scorer_gives_truth_100": {"fields": self_shares, "largest_angle_error_deg": largest_error, "holds": perfect}}


def sign_diagnostics(rows: list[dict]) -> dict:
    by_label: dict[str, list] = {}
    for row in rows:
        by_label.setdefault(_shown(row)["label"], []).append(row["diagnostics"]["sense_matches_label"])
    rot = [row for row in rows if _kind(row) == "rot"]
    return {"translation_sense_matches_label": _share([row["diagnostics"]["sense_matches_label"] for row in rows]),
            "translation_sense_matches_label_by_label": {k: _share(v) for k, v in sorted(by_label.items()) if any(x is not None for x in v)},
            "hinged_tangent_towards_camera": _share([row["diagnostics"]["tangent_towards_camera"] for row in rot]),
            "rotation_axis_distance_m": _quantiles([_shown(row)["motion"]["axis_distance_m"] for row in rot]),
            "hinge_sides": _counts([_shown(row)["view"]["hinge_side"] for row in rot]),
            "axis_classes": _counts([_shown(row)["view"]["axis_class"] for row in rot]),
            "axis_class_margin_below_0.1": _share([row["diagnostics"]["axis_class_margin"] < AXIS_TIE for row in rot]),
            "anchor_in_image": _share([row["diagnostics"]["anchor_in_image"] for row in rows])}


def summary(rows: list[dict], seconds: float) -> dict:
    checks = criteria(rows)
    passed = all(c["holds"] for c in checks.values())
    missing = [{"visit": row["visit"], "task": row["task"],
                "parts_without_motion": [p["label"] for p in row["parts"] if p["motion"] is None],
                "annot_ids_not_in_scene": len(row["annot_ids"]) - len(row["parts"])} for row in rows if not row["has_all_motions"]]
    return {"scenes": len({row["visit"] for row in rows}), "tasks": len(rows), "criteria": checks,
            "verdict": "continue to C-1" if passed else "stop: a criterion of section 25.3 fails",
            "scorer_on_contrary_answers": A.shares([row["contrary_score"] for row in rows]),
            "tasks_by_shown_type": _counts([_kind(row) for row in rows]),
            "tasks_with_several_parts": sum(len(row["annot_ids"]) > 1 for row in rows),
            "arrow_px_by_type": {k: _quantiles([row["arrow_px"] for row in rows if _kind(row) == k]) for k in ("trans", "rot")},
            "tasks_missing_a_motion": missing, "sign_diagnostics": sign_diagnostics(rows), "seconds": round(seconds, 1)}


# ---------------------------------------------------------------- the check images

def stratum(row: dict) -> str | None:
    motion = _shown(row)["motion"]
    if motion is None:
        return None
    if motion["type"] == "trans":
        return "pushed" if motion["viz_orient"] == "inwards" else "pulled"
    return "hinged" if motion["axis_distance_m"] >= A.ON_PART_M else "turning in place"


def pick_checks(rows: list[dict], seed: int = CHECK_SEED) -> list[tuple[str, dict]]:
    """The parts to check by eye: a fixed number per stratum, drawn at random, at most one per scene."""
    rng, used, chosen = np.random.default_rng(seed), set(), []
    for name, wanted in STRATA:
        pool = [row for row in rows if stratum(row) == name]
        taken = 0
        for i in rng.permutation(len(pool)):
            if taken == wanted:
                break
            if pool[i]["visit"] not in used:
                used.add(pool[i]["visit"])
                chosen.append((name, pool[i]))
                taken += 1
    return chosen


def _arrow(axis: plt.Axes, segment: A.Segment | None, colour: str, lw: float = 2.5) -> None:
    """An arrow clipped to the axes (annotate would drop an arrow whose tip leaves the view)."""
    if segment is not None:
        axis.add_patch(FancyArrowPatch(segment[0], segment[1], arrowstyle="-|>", mutation_scale=18, color=colour, lw=lw,
                                       clip_on=True))


def _dot(axis: plt.Axes, point: np.ndarray, frame: Frame, colour: str, label: str) -> None:
    uv, z, _ = project(point[None], frame)
    if z[0] > 0:
        axis.plot(*uv[0], "o", color=colour, ms=7, mec="black", label=label)


def depth_text(motion: A.PartMotion, frame: Frame) -> str:
    """How far the drawn arrow's start and tip are from the camera, so that 'towards' or 'away' can be read off."""
    _, z, _ = project(np.stack([motion.anchor, motion.anchor + A.ARROW_M * motion.direction]), frame)
    return f"arrow start {z[0]:.2f} m from the camera, tip {z[1]:.2f} m ({'towards' if z[1] < z[0] else 'away from'} the camera)"


def draw_motion(axis: plt.Axes, motion: A.PartMotion, frame: Frame) -> None:
    if motion.kind == "trans":
        _arrow(axis, A.projected_segment(motion.anchor, -motion.direction, frame), "red", 1.2)
        _arrow(axis, A.projected_segment(motion.anchor, motion.direction, frame), "lime")
        _dot(axis, motion.anchor, frame, "lime", "motion origin")
        return
    _arrow(axis, A.projected_segment(motion.foot, -motion.direction, frame), "deepskyblue", 1.5)
    _arrow(axis, A.projected_segment(motion.foot, motion.direction, frame), "deepskyblue")
    _dot(axis, motion.origin, frame, "magenta", "motion origin")
    _dot(axis, motion.centre, frame, "white", "part centre")
    if motion.hinged:
        _arrow(axis, A.projected_segment(motion.centre, A.opening_tangent(motion), frame, TANGENT_M), "orange")


def zoom_box(labels: np.ndarray, part: int, motion: A.PartMotion, frame: Frame) -> tuple[float, float, float, float]:
    rows_cols = np.argwhere(labels == part)
    points = [np.stack([rows_cols[:, 1], rows_cols[:, 0]], axis=1).astype(float)]
    for start in (motion.anchor, motion.centre):
        for sign in (1.0, -1.0):
            segment = A.projected_segment(start, sign * motion.direction, frame)
            if segment is not None:
                points.append(np.array(segment))
    corners = np.concatenate(points)
    height, width = labels.shape
    left, top = np.clip(corners.min(axis=0) - ZOOM_PAD_PX, 0, (width - 1, height - 1))
    right, bottom = np.clip(corners.max(axis=0) + ZOOM_PAD_PX, 0, (width - 1, height - 1))
    return float(left), float(max(right, left + 1)), float(top), float(max(bottom, top + 1))


def check_text(stratum_name: str, row: dict, motion: A.PartMotion, frame: Frame) -> str:
    shown = _shown(row)
    lines = [f"{row['visit']} · keyframe {row['keyframe']} · stratum: {stratum_name}", f"task: {row['task']}",
             f"label {shown['label']} · type {motion.kind} · motion_viz_orient {motion.viz_orient}"]
    if motion.kind == "trans":
        lines += [f"green = planned reading ({'-' if motion.viz_orient == 'inwards' else '+'}motion_dir, 0.2 m) · thin red = the "
                  f"opposite · green computes as: {shown['view']['sense']} (label implies {A.EXPECTED_SENSE.get(shown['label'], '?')})",
                  "green " + depth_text(motion, frame)]
    else:
        lines += [f"thick blue = axis along +motion_dir, thin blue = along -motion_dir · magenta = motion origin · white = part centre · "
                  f"axis {motion.axis_distance_m:.2f} m from the centre",
                  f"hinge side {shown['view']['hinge_side']} · axis class {shown['view']['axis_class']} · " + depth_text(motion, frame)]
        if motion.hinged:
            lines.append("orange = where the part moves under a positive turn (right-hand rule); "
                         f"towards the camera: {row['diagnostics']['tangent_towards_camera']}")
    return "\n".join(lines)


def draw_check(path: Path, stratum_name: str, row: dict, frame: Frame, labels: np.ndarray, motion: A.PartMotion) -> None:
    part = _shown(row)["part"]
    image = (frame.rgb * 0.65).astype(np.uint8)
    image[labels == part] = (0.5 * image[labels == part] + 0.5 * np.array([255, 230, 0])).astype(np.uint8)
    height, width = labels.shape
    fig, axes = plt.subplots(1, 2, figsize=(16, 16 * height / (2 * width) + 1.6))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=1 - 1.5 / fig.get_figheight(), wspace=0.02)
    for axis, box in zip(axes, [(0, width - 1, 0, height - 1), zoom_box(labels, part, motion, frame)]):
        axis.imshow(image)
        draw_motion(axis, motion, frame)
        axis.set_xlim(box[0], box[1])
        axis.set_ylim(box[3], box[2])
        axis.axis("off")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.text(0.005, 0.995, check_text(stratum_name, row, motion, frame), fontsize=11, ha="left", va="top")
    fig.savefig(path, dpi=90)
    plt.close(fig)


def draw_checks(rows: list[dict], runs: Path, data_root: Path, out: Path) -> list[dict]:
    """Draw the chosen parts, scene by scene (the scan of each scene is read once)."""
    chosen = pick_checks(rows)
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for visit in sorted({row["visit"] for _, row in chosen}):
        result = json.loads((runs / f"{visit}.json").read_text())
        scene = P.load_scene(data_root, visit, result["video"])
        for number, (name, row) in enumerate(chosen, start=1):
            if row["visit"] != visit:
                continue
            frame = P.gt_frames(scene, [row["keyframe"]])[0]
            labels = P.label_maps(scene, [frame])[0]
            part = _shown(row)["part"]
            motion = A.part_motion(scene.motions[scene.elements[part].annot_id], scene.part_points(part), scene.scan_xyz)
            path = out / f"{number:02d}_{visit}_{name.replace(' ', '_')}_{_shown(row)['label']}.png"
            draw_check(path, name, row, frame, labels, motion)
            listing.append({"file": path.name, "stratum": name, "visit": visit, "task": row["task"], "label": _shown(row)["label"],
                            "keyframe": row["keyframe"], "motion": _shown(row)["motion"], "view": _shown(row)["view"],
                            "diagnostics": row["diagnostics"]})
    return sorted(listing, key=lambda item: item["file"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1", help="phase-1 results (tasks, keyframes)")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "mem_c0")
    parser.add_argument("--visits", default="", help="comma-separated visits (default: every phase-1 scene)")
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.checks_only:
        rows = json.loads((args.out / "records.json").read_text())
    else:
        visits = [v for v in args.visits.split(",") if v] or sorted(p.stem for p in args.runs.glob("[0-9]*.json"))
        rows = []
        for visit in visits:
            rows += scene_rows(args.runs, args.data, visit)
            print(f"{visit}: {sum(row['visit'] == visit for row in rows)} tasks ({time.time() - started:.0f} s)", flush=True)
        (args.out / "records.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        result = summary(rows, time.time() - started)
        (args.out / "summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
        report(result)
    listing = draw_checks(rows, args.runs, args.data, args.out / "checks")
    (args.out / "checks" / "checks.json").write_text(json.dumps(listing, indent=1, ensure_ascii=False))
    print(f"{len(listing)} check images in {args.out / 'checks'}")


def report(result: dict) -> None:
    c = result["criteria"]
    print(f"{result['scenes']} scenes, {result['tasks']} tasks")
    print(f"  1. a motion for every target part: {c['motions_for_every_part']['share']:.1%} (need >= {MOTIONS_SHARE:.0%})")
    print(f"  2. projected direction >= {A.MIN_ARROW_PX} px: {c['arrow_at_least_20px']['share']:.1%} (need >= {ARROW_SHARE:.0%}); "
          f"at full resolution {c['arrow_at_least_20px']['share_at_full_resolution']:.1%}")
    fields = ", ".join(f"{k} {v['share']:.0%} (n={v['n']})" for k, v in c["scorer_gives_truth_100"]["fields"].items())
    print(f"  3. scorer on the truth: {fields}")
    print(f"  verdict: {result['verdict']}")
    print(f"  sign diagnostics: {json.dumps(result['sign_diagnostics'])}")


if __name__ == "__main__":
    main()
