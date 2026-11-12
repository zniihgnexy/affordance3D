"""Aggregate the phase-1 results of all scenes: table, rate-against-budget curves, the G1 verdict.

    python scripts/aggregate_mem_phase1.py            # reads runs/mem_phase1/*.json, writes runs/mem_phase1/summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.mem import phase1_stats as A

ON_PART_CM = 3.0


def triple(values: tuple[float, float, float]) -> dict:
    return {"rate": values[0], "low": values[1], "high": values[2]}


def at_budget_one(results: list[dict], key: str) -> dict:
    """Each policy at budget 1 (whole keyframes: no budget), and the paired differences of P4 against the others."""
    rows, per_scene = {}, {}
    for name, budget in ((A.P0, None), (A.P2, 1.0), (A.P3, 1.0), (A.P4, 1.0)):
        rates = [A.scene_rates(result, name, budget, key) for result in results]
        if all(rate is not None for rate in rates):
            per_scene[name] = rates
            rows[name] = triple(A.bootstrap(rates))
    for other in (A.P0, A.P2, A.P3):
        if other in per_scene and A.P4 in per_scene:
            rows[f"P4 minus {other}"] = triple(A.paired_difference(per_scene[A.P4], per_scene[other]))
    return rows


def reading(results: list[dict]) -> dict:
    rows = {}
    for name in ("P4 affordance regions", "P3 generic regions"):
        per_scene = [np.array(r["reading"][name]["right_part"], dtype=float) for r in results if name in r.get("reading", {})]
        if len(per_scene) == len(results):
            rows[name] = triple(A.bootstrap(per_scene))
            rows[name]["seconds_per_query"] = float(np.median(np.concatenate(
                [r["reading"][name]["seconds"] for r in results if r["reading"][name]["seconds"]])))
    return rows


def geometry(results: list[dict]) -> dict:
    def stats(errors: list[float | None]) -> dict:
        found = np.array([e for e in errors if e is not None], dtype=float)
        return {"parts": len(errors), "read": len(found), "median_cm": float(np.median(found)),
                "within_3cm": float(np.mean(found <= ON_PART_CM)), "within_3cm_of_all": float(np.sum(found <= ON_PART_CM) / len(errors))}

    rows = {name: stats([e for r in results for e in r["geometry"][name]["contact_cm"]]) for name in ("single_frame", "l2_readout")}
    for name in ("single_frame", "l2_readout"):
        rows[name]["stand_free"] = float(np.mean([f for r in results for f in r["geometry"][name]["stand_free"]]))
    for voxel in results[0]["geometry"]["fused"]:
        rows[f"fused {voxel} m"] = stats([e for r in results for e in r["geometry"]["fused"][voxel]["contact_cm"]])
        rows[f"fused {voxel} m"].update({"bytes_per_scene": float(np.mean([r["geometry"]["fused"][voxel]["bytes"] for r in results])),
                                         "build_s": float(np.mean([r["geometry"]["fused"][voxel]["build_s"] for r in results]))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    results = [json.loads(path.read_text()) for path in sorted(args.runs.glob("*.json")) if path.name != "summary.json"]
    if not results:
        raise SystemExit(f"no scene results in {args.runs}")
    summary = {"scenes": len(results), "tasks": int(sum(r["tasks"]["answerable"] for r in results)),
               "tasks_all": int(sum(r["tasks"]["all"] for r in results)), "keyframes": results[0]["keyframes"],
               "first_keyframe_right": at_budget_one(results, "hit1"), "right_keyframe_in_top3": at_budget_one(results, "hit3"),
               "curves": {name: A.curve(results, name) for name in (A.P1, A.P2, A.P3, A.P4)},
               "reading": reading(results), "geometry": geometry(results),
               "parts_seen": {kind: int(sum(r["parts_seen"][kind] for r in results)) for kind in results[0]["parts_seen"]},
               "parts": int(sum(r["parts"] for r in results)),
               "memory": {key: int(sum(r["memory"][key] for r in results)) for key in ("tokens", "parts", "true_parts_with_token", "mixed_regions")},
               "storage_bytes_per_scene": {key: float(np.mean([r["storage"][key] for r in results])) for key in results[0]["storage"]},
               "write_s_per_scene": float(np.mean([r["write_s"] for r in results]))}
    first, read = summary["first_keyframe_right"], summary["reading"]
    against = {name: (first[f"P4 minus {name}"]["rate"], first[f"P4 minus {name}"]["low"], first[f"P4 minus {name}"]["high"])
               if f"P4 minus {name}" in first else None for name in (A.P2, A.P3)}
    summary["g1"] = A.g1_verdict(against[A.P3], against[A.P2], read.get("P4 affordance regions", {}).get("rate"),
                                 read.get("P3 generic regions", {}).get("rate"))
    (args.runs / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['scenes']} scenes, {summary['tasks']} answerable tasks of {summary['tasks_all']}")
    for name, row in first.items():
        print(f"  first keyframe right, {name}: {row['rate']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
    for name, row in read.items():
        print(f"  model picks the right part, {name}: {row['rate']:.3f} [{row['low']:.3f}, {row['high']:.3f}]")
    print("  G1:", summary["g1"])


if __name__ == "__main__":
    main()
