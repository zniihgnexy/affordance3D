"""S0 of the 2026-11-18 proposal: per-part affordance vectors by rules on one scene, scored against the annotated motions.

    python scripts/run_part_vectors.py --config configs/part_vectors/s0_421069.yaml [--set key=value ...]
    python scripts/run_part_vectors.py --config configs/part_vectors/s0b_val3.yaml
    python scripts/run_part_vectors.py --config configs/part_vectors/s0c_gravity_val3.yaml

CPU only, no model. Part masks and affordance labels come from the annotation (ORACLE); depth is either the device's
LiDAR with the ARKit pose (inputs) or the depth rendered from the laser scan with its registered pose (upper bound).
For every rule set (affordance3d.mem.part_vectors) and every way of using the keyframes, each part gets a vector,
scored per part by the SceneFun3D Task 3 definition (+M, +MA, +MAO). A config names one scene (visit, video) or a
list of them (scenes). Writes, under <outputs>/<name>/ for one scene and under <outputs>/<name>/<visit>/ for each of
several:

    parts.json        every part, variant by variant: the vector and its score
    summary.json      rates per variant and per label group, baselines, frame-to-frame spread, type-filter sizes
    part_table.json   the main variant's part table in the device's world (ARKit) frame: the robot-ready form
    keyframes.png     the main variant drawn on the keyframes that show the most parts (prediction and truth)

The main variant is the config's `main` [depth, rule, frames], by default MAIN.

and, for several scenes, <outputs>/<name>/summary.json: the rates and baselines over the parts of all scenes.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401,E402  (its banner would end up in the log)

from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402
from affordance3d.geometry import project, transform_points  # noqa: E402
from affordance3d.mem import action_records as A  # noqa: E402
from affordance3d.mem import keyframes  # noqa: E402
from affordance3d.mem import part_vectors as V  # noqa: E402
from affordance3d.mem import pipeline as P  # noqa: E402

MAIN = ("lidar", "geometric", "fused")  # default variant of the part table and the figure: inputs only, apart from the oracle masks
GROUPS = {"pressed": ("key_press", "tip_push", "plug_in", "foot_push"), "pulled": ("hook_pull", "pinch_pull", "unplug"),
          "rotate": ("rotate",), "hook_turn": ("hook_turn",)}
PRED_COLOUR, TRUE_COLOUR = "#D55E00", "#009E73"  # Okabe-Ito vermilion and green


def _list(vector: np.ndarray | None) -> list[float] | None:
    return None if vector is None else [round(float(v), 4) for v in vector]


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def main_variant(config: dict) -> tuple[str, str, str]:
    variant = tuple(config.get("main", MAIN))
    if variant[0] not in config["depths"] or variant[1] not in config["rules"] or variant[2] not in config["frames"]:
        raise SystemExit(f"the main variant {list(variant)} is not among the run's depths, rules and frames")
    return variant


def scene_list(config: dict) -> list[tuple[str, str]]:
    """(visit, video) of every scene: the `scenes` list, or the single `visit` and `video`."""
    if "scenes" in config:
        return [(str(s["visit"]), str(s["video"])) for s in config["scenes"]]
    return [(str(config["visit"]), str(config["video"]))]


def load_inputs(config: dict, visit: str, video: str) -> tuple[P.Scene, list[str], dict[str, list], list[np.ndarray]]:
    scene = P.load_scene(REPO_ROOT / config["data_root"], visit, video)
    times = keyframes.scene_keyframes(scene, config["keyframes"]["kind"], config["keyframes"]["n"])
    rendered = P.gt_frames(scene, times)
    lidar = [P.input_frame(scene, t, P.lidar_depth(scene, t)) for t in times]
    if any(a.image_hw != b.image_hw for a, b in zip(rendered, lidar)):
        raise SystemExit("the LiDAR and rendered frames differ in size")
    return scene, times, {"lidar": lidar, "rendered": rendered}, P.label_maps(scene, rendered)


def scored_parts(scene: P.Scene, labels: list[np.ndarray]) -> tuple[list[int], dict]:
    """Parts with a label other than exclude, a motion annotation and at least one keyframe that shows them."""
    counts = {"elements": len(scene.elements), "excluded": 0, "no_motion": 0, "not_seen": 0}
    parts = []
    for p, element in enumerate(scene.elements):
        if element.label == "exclude":
            counts["excluded"] += 1
        elif element.annot_id not in scene.motions:
            counts["no_motion"] += 1
        elif max(int((lab == p).sum()) for lab in labels) < P.MIN_PART_PIXELS:
            counts["not_seen"] += 1
        else:
            parts.append(p)
    counts["scored"] = len(parts)
    return parts, counts


def truth_of(scene: P.Scene, part: int) -> A.PartMotion:
    return A.part_motion(scene.motions[scene.elements[part].annot_id], scene.part_points(part), scene.scan_xyz)


def run_variants(config: dict, scene: P.Scene, frames: dict[str, list], labels: list[np.ndarray], parts: list[int]) -> tuple[list[dict], dict]:
    hinge_config = V.HingeConfig(**config["hinge"])
    scoring = config["scoring"]
    rows, vectors = [], {}
    for depth in config["depths"]:
        observed = {p: V.observe(p, frames[depth], labels) for p in parts}
        for rule in config["rules"]:
            for mode in config["frames"]:
                for p in parts:
                    element, truth = scene.elements[p], truth_of(scene, p)
                    row = {"depth": depth, "rule": rule, "frames": mode, "part": p, "annot_id": element.annot_id,
                           "label": element.label, "true_kind": truth.kind, "hinged": truth.hinged,
                           "task": scene.task_of(element.annot_id), "seen_in": len(observed[p])}
                    if not observed[p]:
                        rows.append({**row, "vector": None, "score": {"axis_deg": None, "origin_m": None, "m": False, "ma": False, "mao": False}})
                        continue
                    vector = V.part_vector(p, element.label, observed[p], frames[depth], labels, scene.up, rule, mode,
                                           hinge_config, true_kind=truth.kind)
                    vectors[(depth, rule, mode, p)] = vector
                    score = V.score_part(vector, truth, scoring["axis_deg"], scoring["origin_m"])
                    hinge = vector.hinge
                    rows.append({**row, "vector": {"kind": vector.kind, "contact": _list(vector.contact), "normal": _list(vector.normal),
                                                   "axis": _list(vector.axis), "origin": _list(vector.origin), "keyframe": vector.keyframe,
                                                   "hinge": None if hinge is None else {"asymmetry": _round(hinge.asymmetry), "at_border": hinge.at_border}},
                                 "score": {"axis_deg": _round(score.axis_deg, 1), "origin_m": _round(score.origin_m), "m": score.m,
                                           "ma": score.ma, "mao": score.mao}})
    return rows, vectors


def rates(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    angles = [r["score"]["axis_deg"] for r in rows if r["score"]["axis_deg"] is not None]
    origins = [r["score"]["origin_m"] for r in rows if r["score"]["origin_m"] is not None]
    return {"n": len(rows), **{k: round(100 * np.mean([r["score"][k] for r in rows]), 1) for k in ("m", "ma", "mao")},
            "axis_deg_median": _round(np.median(angles), 1) if angles else None,
            "origin_m_median": _round(np.median(origins)) if origins else None}


def variant_rates(config: dict, rows: list[dict]) -> dict:
    """Rates per variant: all parts, rotations, rotations about an axis off the part (doors), and per label group."""
    variants = {}
    for depth in config["depths"]:
        for rule in config["rules"]:
            for mode in config["frames"]:
                mine = [r for r in rows if (r["depth"], r["rule"], r["frames"]) == (depth, rule, mode)]
                variants[f"{depth}/{rule}/{mode}"] = {"all": rates(mine), "rotations": rates([r for r in mine if r["true_kind"] == "rot"]),
                                                      "off_part_rotations": rates([r for r in mine if r["hinged"]]),
                                                      **{g: rates([r for r in mine if r["label"] in labs]) for g, labs in GROUPS.items()}}
    return variants


def summarise(config: dict, rows: list[dict], scene: P.Scene, observations: dict[str, dict[int, list]], parts: list[int], counts: dict) -> dict:
    return {"counts": counts, "variants": variant_rates(config, rows), "baselines": baselines(config, scene, observations, parts),
            "spread": {d: spread_summary(observations[d]) for d in observations},
            "type_filter": type_filter(scene, parts), "truth": truth_notes(scene, observations["rendered"], parts)}


def baselines(config: dict, scene: P.Scene, observations: dict[str, dict[int, list]], parts: list[int]) -> dict:
    """Rotations only: share whose true axis is within the angle of the vertical, or of the fused surface normal."""
    limit, out, hits = config["scoring"]["axis_deg"], {}, {}
    rotations = [p for p in parts if truth_of(scene, p).kind == "rot"]
    hits["always_vertical"] = int(np.sum([V.line_angle_deg(scene.up, truth_of(scene, p).direction) <= limit for p in rotations]))
    for depth, seen in observations.items():
        hits[f"always_normal_{depth}"] = int(np.sum([V.line_angle_deg(V.fuse(seen[p])[1], truth_of(scene, p).direction) <= limit
                                                     for p in rotations if seen[p]]))
    out = {name: round(100 * k / len(rotations), 1) if rotations else None for name, k in hits.items()}
    return {**out, "rotations": len(rotations), "hits": hits}


def spread_summary(seen: dict[int, list]) -> dict:
    spreads = [V.spread(obs) for obs in seen.values() if len(obs) >= 2]
    normal = [s["normal_deg"] for s in spreads]
    contact = [s["contact_m"] for s in spreads]
    return {"parts_with_2_or_more_frames": len(spreads), "frames_per_part_median": float(np.median([len(o) for o in seen.values() if o])),
            "normal_deg_median": _round(np.median(normal), 1) if normal else None, "contact_m_median": _round(np.median(contact)) if contact else None}


def type_filter(scene: P.Scene, parts: list[int]) -> dict:
    """ORACLE upper bound of the type channel: with the right label, how many scored parts share it."""
    labels = [scene.elements[p].label for p in parts]
    shares = [labels.count(label) / len(labels) for label in labels]
    return {"parts": len(labels), "share_median": round(float(np.median(shares)), 3), "share_mean": round(float(np.mean(shares)), 3)}


def truth_notes(scene: P.Scene, seen: dict[int, list], parts: list[int]) -> list[dict]:
    """Descriptive, per rotating part: how its true axis sits against the vertical and the surface normal."""
    notes = []
    for p in parts:
        truth = truth_of(scene, p)
        if truth.kind != "rot" or not seen[p]:
            continue
        notes.append({"part": p, "label": scene.elements[p].label, "hinged": truth.hinged, "axis_distance_m": _round(truth.axis_distance_m),
                      "axis_vs_vertical_deg": _round(V.line_angle_deg(truth.direction, scene.up), 1),
                      "axis_vs_normal_deg": _round(V.line_angle_deg(truth.direction, V.fuse(seen[p])[1]), 1)})
    return notes


def part_table(scene: P.Scene, times: list[str], vectors: dict, parts: list[int], variant: tuple[str, str, str]) -> list[dict]:
    """The main variant in the device's world frame (ARKit; z is up): one rigid transform from the scan frame."""
    to_world = scene.vid.faro_to_arkit
    table = []
    for p in parts:
        vector = vectors.get((*variant, p))
        if vector is None:
            continue
        point = lambda x: _list(transform_points(to_world, x[None])[0])  # noqa: E731
        turn = lambda x: None if x is None else _list(to_world[:3, :3] @ x)  # noqa: E731
        element = scene.elements[p]
        table.append({"part": p, "annot_id": element.annot_id, "label": element.label, "task": scene.task_of(element.annot_id),
                      "kind": vector.kind, "contact": point(vector.contact), "normal": turn(vector.normal), "axis": turn(vector.axis),
                      "origin": point(vector.origin), "evidence": [times[k] for k in vector.evidence]})
    return table


def _segment(frame, start: np.ndarray, direction: np.ndarray, kind: str | None, length: float):
    ends = np.stack([start - length * direction, start + length * direction]) if kind == "rot" else np.stack([start, start + length * direction])
    uv, z, _ = project(ends, frame)
    return None if (z <= 0.05).any() else uv


def pick_keyframes(shown: list[list[int]], n: int) -> list[int]:
    """Greedy: each next keyframe adds the most parts not shown yet (ties: more parts in all), so each part shows once."""
    chosen, covered = [], set()
    for _ in range(n):
        gains = [(-1, -1) if k in chosen else (len(set(parts) - covered), len(parts)) for k, parts in enumerate(shown)]
        best = int(np.argmax([g[0] * 1000 + g[1] for g in gains]))
        if gains[best][0] <= 0:
            break
        chosen.append(best)
        covered |= set(shown[best])
    return sorted(chosen)


def _zoom(ax, points: list[np.ndarray], image_hw: tuple[int, int], pad: int = 70, least: int = 320) -> None:
    """Crop the panel to what is drawn on it, with a margin, at least `least` pixels on a side."""
    if not points:
        return
    uv = np.array(points)
    height, width = image_hw
    centre, half = (uv.min(axis=0) + uv.max(axis=0)) / 2, np.maximum((uv.max(axis=0) - uv.min(axis=0)) / 2 + pad, least / 2)
    left, right = np.clip([centre[0] - half[0], centre[0] + half[0]], 0, width)
    top, bottom = np.clip([centre[1] - half[1], centre[1] + half[1]], 0, height)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)


def draw(path: Path, config: dict, scene: P.Scene, times: list[str], frames: list, labels: list[np.ndarray], vectors: dict, parts: list[int]) -> list[str]:
    """The main variant on keyframes chosen to show every part once where possible: prediction (a rotation's axis through
    its origin, a translation's direction from the contact) and the annotation, 2 x arrow_m long."""
    length, variant = config["figure"]["arrow_m"], main_variant(config)
    shown = [[p for p in parts if (lab == p).sum() >= P.MIN_PART_PIXELS and (*variant, p) in vectors] for lab in labels]
    order = pick_keyframes(shown, config["figure"]["keyframes"])
    columns = (len(order) + 1) // 2
    figure, axes = plt.subplots(2, columns, figsize=(3.6 * columns, 7.6))
    for ax, k in zip(axes.flat, order):
        frame, drawn = frames[k], []
        ax.imshow(frame.rgb)
        for p in shown[k]:
            vector, truth = vectors[(*variant, p)], truth_of(scene, p)
            for start, direction, kind, colour in ((vector.origin, vector.axis, vector.kind, PRED_COLOUR),
                                                   (truth.anchor, truth.direction, truth.kind, TRUE_COLOUR)):
                uv = None if direction is None else _segment(frame, start, direction, kind, length)
                if uv is not None:
                    arrow = ax.annotate("", uv[1], uv[0], arrowprops={"arrowstyle": "-" if kind == "rot" else "->", "color": colour, "lw": 2})
                    arrow.arrow_patch.set_clip_box(ax.bbox)  # stay inside the (zoomed) panel
                    drawn.extend(uv)
            uv, _, inside = project(vector.contact[None], frame)
            if inside[0]:
                drawn.append(uv[0])
                ax.plot(*uv[0], "o", ms=4, color=PRED_COLOUR, mec="white", mew=0.8)
                ax.text(uv[0, 0] + 6, uv[0, 1] - 6, vector.label, fontsize=6, color="white",
                        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "lw": 0})
        _zoom(ax, drawn, frame.image_hw)
        ax.set_title(f"{times[k]} s, {len(shown[k])} parts", fontsize=8)
        ax.set_axis_off()
    for ax in list(axes.flat)[len(order):]:
        ax.set_axis_off()
    figure.legend(handles=[plt.Line2D([], [], color=PRED_COLOUR, lw=2, label=f"computed by the {variant[1]} rules from {'LiDAR' if variant[0] == 'lidar' else 'rendered'} depth"),
                           plt.Line2D([], [], color=TRUE_COLOUR, lw=2, label="annotated motion"),
                           plt.Line2D([], [], color="grey", lw=2, label="line: rotation axis    arrow: translation")],
                  loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [times[k] for k in order]


def run_scene(config: dict, visit: str, video: str, out: Path) -> tuple[list[dict], dict]:
    """Every variant on one scene; writes the scene's files to `out` and returns its rows and summary."""
    scene, times, frames, labels = load_inputs(config, visit, video)
    parts, counts = scored_parts(scene, labels)
    rows, vectors = run_variants(config, scene, frames, labels, parts)
    rows = [{"visit": visit, **row} for row in rows]
    observations = {d: {p: V.observe(p, frames[d], labels) for p in parts} for d in config["depths"]}
    summary = summarise(config, rows, scene, observations, parts, counts)
    out.mkdir(parents=True, exist_ok=True)
    variant = main_variant(config)
    summary["figure_keyframes"] = draw(out / "keyframes.png", config, scene, times, frames[variant[0]], labels, vectors, parts)
    (out / "parts.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "part_table.json").write_text(json.dumps(part_table(scene, times, vectors, parts, variant), indent=1, ensure_ascii=False))
    return rows, summary


def pooled(config: dict, rows: list[dict], summaries: dict[str, dict]) -> dict:
    """Over the parts of all scenes: counts, rates per variant, and the rotation baselines from their hit counts."""
    counts = {k: sum(s["counts"][k] for s in summaries.values()) for k in next(iter(summaries.values()))["counts"]}
    rotations = sum(s["baselines"]["rotations"] for s in summaries.values())
    hits = {k: sum(s["baselines"]["hits"][k] for s in summaries.values()) for k in next(iter(summaries.values()))["baselines"]["hits"]}
    return {"scenes": list(summaries), "counts": counts, "variants": variant_rates(config, rows),
            "baselines": {**{k: round(100 * h / rotations, 1) if rotations else None for k, h in hits.items()}, "rotations": rotations}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.sub=value override")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    main_variant(config)  # stop before any work if the figure's variant is not run
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, "run", REPO_ROOT)
    scenes = scene_list(config)
    several = len(scenes) > 1
    all_rows, summaries = [], {}
    for visit, video in scenes:
        rows, summaries[visit] = run_scene(config, visit, video, folder / visit if several else folder)
        all_rows.extend(rows)
        print(f"{visit}: {summaries[visit]['counts']}", flush=True)
    summary = pooled(config, all_rows, summaries) if several else summaries[scenes[0][0]]
    if several:
        (folder / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v["all"] for k, v in summary["variants"].items()}, indent=1))
    print(f"wrote {folder}")


if __name__ == "__main__":
    main()
