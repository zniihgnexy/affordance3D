"""Summarise C-1 over all scenes and read it by the rule fixed before the run (discussion draft, section 25.5).

    python scripts/aggregate_mem_c1.py                       # reads runs/mem_c1/4b/<visit>.json, writes runs/mem_c1/4b_summary.json
    python scripts/aggregate_mem_c1.py --model 8b            # V with the 8B model (also: 8B minus 4B for V and V')
    python scripts/aggregate_mem_c1.py --folder runs/mem_c1/preview   # the CPU-only arms (M, O)

Rates are pooled over tasks with 95% intervals from resampling scenes; arms are compared on the same tasks. The reading:

    C stands on its own   M minus V in joint success has its interval's lower end >= 10 points, and, on the tasks where
                          both put the point on the part, M minus V in direction (3D line error <= 20 degrees) is >= 5 points
    C merges into A       on those tasks, V is within 5 points of M in direction (M minus V < 5 points)
    direction from geometry   G is close to M in joint success (round-7 rule: the lower end of M minus G below 5 points)

O must score 100% on every field. The "always vertical" axis is reported as a floor for the rotation fields, and push or
pull (documentation's reading) as an extra outside the plan.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.mem import phase1_stats as A  # noqa: E402

NAMES = {"V": "V: plain Qwen3-VL on c3's frame", "G": "G: V's point and label, direction by the depth rule",
         "M": "M: P4's frame and region, the same rule (ORACLE)", "O": "O: the ground truth (scorer check)",
         "V'": "V' on the C-0 keyframe", "G'": "G' on the C-0 keyframe", "M'": "M' on the C-0 keyframe (ORACLE)"}
PRIMARY = ("contact", "label", "type", "line_ok", "axis_class", "hinge_side", "joint")
FLOOR = ("vertical_axis_class", "vertical_line_ok")
EXTRA = ("sense",)
ANGLES = ("line_deg", "line_2d_deg", "signed_deg")
READ_FIELDS = ("point", "label", "type", "arrow", "hinge_side", "direction")
ALONE_POINTS, DIRECTION_POINTS = 0.10, 0.05  # section 25.5 (from the feasibility record, sections 5 and 6)


def triple(values: tuple[float, float, float]) -> dict:
    return {"rate": values[0], "low": values[1], "high": values[2]}


def scores(record: dict, arm: str) -> list[dict]:
    return [entry["score"] for entry in record["arms"][arm]]


def per_scene(records: list[dict], arm: str, field: str, keep: Callable[[dict, int], bool] | None = None) -> list[np.ndarray]:
    """Per scene, the field over the tasks where it applies (and `keep(record, task)` holds), as 0/1."""
    return [np.array([s[field] for i, s in enumerate(scores(record, arm)) if isinstance(s[field], bool)
                      and (keep is None or keep(record, i))], dtype=float) for record in records]


def interval(per_scene_values: list[np.ndarray]) -> dict | None:
    kept = [values for values in per_scene_values if len(values)]
    return {**triple(A.bootstrap(kept)), "n": int(sum(len(v) for v in kept))} if kept else None


def paired(records: list[dict], first: str, second: str, field: str, keep: Callable[[dict, int], bool] | None = None) -> dict | None:
    """`first` minus `second` on the same tasks (where the field applies to both), 95% interval over scenes."""
    differences = []
    for record in records:
        a, b = scores(record, first), scores(record, second)
        rows = [(x[field], y[field]) for i, (x, y) in enumerate(zip(a, b, strict=True))
                if isinstance(x[field], bool) and isinstance(y[field], bool) and (keep is None or keep(record, i))]
        if rows:
            differences.append(np.array([float(x) - float(y) for x, y in rows]))
    return {**triple(A.bootstrap(differences)), "n": int(sum(len(d) for d in differences))} if differences else None


def both_on_part(first: str, second: str) -> Callable[[dict, int], bool]:
    return lambda record, i: scores(record, first)[i]["contact"] and scores(record, second)[i]["contact"]


def fields_read(records: list[dict], arm: str) -> dict[str, float]:
    """Share of the tasks in which each field of the arm's answer could be read (the pilot's check of V's JSON)."""
    predictions = [entry.get("prediction") or {} for record in records for entry in record["arms"][arm]]
    if not predictions:
        return {}
    return {field: float(np.mean([p.get(field) is not None for p in predictions])) for field in READ_FIELDS}


def arm_summary(records: list[dict], arm: str) -> dict:
    rows = [s for record in records for s in scores(record, arm)]
    out = {"name": NAMES[arm], **{field: interval(per_scene(records, arm, field)) for field in PRIMARY + FLOOR + EXTRA},
           "read": fields_read(records, arm)}
    for field in ANGLES:
        values = [s[field] for s in rows if s[field] is not None]
        out[f"median_{field}"] = float(np.median(values)) if values else None
        out[f"answered_{field}"] = len(values)
    out["line_ok_given_contact"] = interval(per_scene(records, arm, "line_ok", lambda record, i: scores(record, arm)[i]["contact"]))
    return out


def verdicts(records: list[dict], arms: list[str]) -> dict:
    """The reading of section 25.5, where the arms it needs are present."""
    out: dict = {}
    if {"M", "V"} <= set(arms):
        joint = paired(records, "M", "V", "joint")
        direction = paired(records, "M", "V", "line_ok", both_on_part("M", "V"))
        out["M_minus_V_joint"], out["M_minus_V_direction_both_on_part"] = joint, direction
        alone = joint is not None and direction is not None and joint["low"] >= ALONE_POINTS and direction["rate"] >= DIRECTION_POINTS
        merges = direction is not None and direction["rate"] < DIRECTION_POINTS
        out["C_stands_alone"], out["C_merges_into_A"] = bool(alone), bool(merges)
    if {"M", "G"} <= set(arms):
        joint = paired(records, "M", "G", "joint")
        out["M_minus_G_joint"] = joint
        out["G_vs_M"] = None if joint is None else A.closeness((joint["rate"], joint["low"], joint["high"]))
        out["direction_from_geometry"] = out["G_vs_M"] == "close"
    return out


def scorer_check(records: list[dict]) -> dict:
    """O must be right on every field wherever it applies."""
    fields = {field: interval(per_scene(records, "O", field)) for field in PRIMARY + EXTRA}
    perfect = all(v is None or v["rate"] == 1.0 for v in fields.values())
    return {"fields": {k: None if v is None else {"rate": v["rate"], "n": v["n"]} for k, v in fields.items()}, "holds": perfect}


def summarise(records: list[dict]) -> dict:
    arms = [arm for arm in NAMES if all(arm in record["arms"] for record in records)]
    summary = {"scenes": len(records), "tasks": int(sum(len(record["tasks"]) for record in records)), "arms": {},
               "readings": verdicts(records, arms)}
    for arm in arms:
        summary["arms"][arm] = arm_summary(records, arm)
    if "O" in arms:
        summary["scorer_check"] = scorer_check(records)
    if {"M'", "V'"} <= set(arms):
        summary["diagnostic"] = {"M'_minus_V'_joint": paired(records, "M'", "V'", "joint"),
                                 "M'_minus_G'_joint": paired(records, "M'", "G'", "joint")}
    regions = [entry.get("region_exact") for record in records for arm in ("M", "M'") if arm in record["arms"]
               for entry in record["arms"][arm] if entry.get("region_exact") is not None]
    summary["memory_region_label_exact"] = float(np.mean(regions)) if regions else None
    return summary


def against_smaller(records: list[dict], smaller: list[dict]) -> dict:
    """V and V' with this model minus the 4B model, joint success on the same tasks (descriptive)."""
    out = {}
    for arm in ("V", "V'"):
        if all(arm in r["arms"] for r in records + smaller):
            merged = [{"arms": {"a": r["arms"][arm], "b": s["arms"][arm]}} for r, s in zip(records, smaller, strict=True)]
            out[arm] = paired(merged, "a", "b", "joint")
    return out


def _pct(row: dict | None) -> str:
    return "-" if row is None else f"{100 * row['rate']:.1f} [{100 * row['low']:.1f}, {100 * row['high']:.1f}] (n={row['n']})"


def report(summary: dict) -> None:
    print(f"{summary['scenes']} scenes, {summary['tasks']} tasks")
    for arm, row in summary["arms"].items():
        print(f"  {row['name']}")
        for field in PRIMARY + EXTRA + FLOOR:
            if row[field] is not None:
                print(f"    {field:20s} {_pct(row[field])}")
        print(f"    median 3D line error {row['median_line_deg']} deg over {row['answered_line_deg']} answered")
        print("    fields read: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in row["read"].items()))
    if "scorer_check" in summary:
        print(f"  scorer gives O 100%: {summary['scorer_check']['holds']}")
    for key, value in summary["readings"].items():
        print(f"  {key}: {_pct(value) if isinstance(value, dict) else value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "mem_c1")
    parser.add_argument("--model", default="4b")
    parser.add_argument("--folder", type=Path, default=None, help="read this folder instead of <out>/<model>")
    args = parser.parse_args()
    folder = args.folder or args.out / args.model
    files = sorted(path for path in folder.glob("*.json") if path.stem.isdigit())
    if not files:
        raise SystemExit(f"no C-1 results in {folder}")
    records = [json.loads(path.read_text()) for path in files]
    summary = summarise(records)
    smaller_folder = args.out / "4b"
    if args.folder is None and args.model != "4b" and smaller_folder.exists():
        smaller = {path.stem: json.loads(path.read_text()) for path in smaller_folder.glob("*.json") if path.stem.isdigit()}
        if all(record["visit"] in smaller for record in records):
            summary["against_4b"] = against_smaller(records, [smaller[record["visit"]] for record in records])
    (folder.parent / f"{folder.name}_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    report(summary)


if __name__ == "__main__":
    main()
