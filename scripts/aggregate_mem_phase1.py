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
READING = ("P4 affordance regions", "P3 generic regions")  # P4 first: the paired difference is P4 minus P3
MORE_BUDGETS = (2.0, 4.0, 8.0)


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


def more_budget(results: list[dict]) -> dict:
    """P4 at budget 1 against the other policies given more tokens. Not part of the G1 rule: a check that the gap at
    budget 1 is not owed to keyframes that random draws leave without any token."""
    rows = {}
    for other in (A.P3, A.P2):
        for budget in MORE_BUDGETS:
            difference = A.affordance_against(results, other, budget)
            if difference is not None:
                rows[f"P4 at 1x minus {other} at {budget:g}x"] = triple(difference)
    grids = {run["policy"]: run["tokens"] for r in results for run in r["search"] if run["policy"].startswith(A.P1)}
    for grid in sorted(grids, key=grids.get):  # a fine grid did not run where it exceeded the budget cap: paired on its own scenes
        ran = [r for r in results if A.scene_rates(r, grid) is not None]
        difference = A.affordance_against(ran, grid, None)
        if difference is not None:
            budget = np.median([run["budget"] for r in ran for run in r["search"] if run["policy"] == grid])
            rows[f"P4 at 1x minus {grid} (about {budget:.1f}x, {len(ran)} scenes)"] = triple(difference)
    return rows


def against_whole(results: list[dict]) -> dict:
    """What tokens add to whole keyframes, paired on the same tasks (not part of the G1 rule). The affordance tokens at
    a quarter and a half of their number stand for an affordance map that misses parts."""
    whole = [A.scene_rates(r, A.P0) for r in results]
    arms = [(f"{A.P3} at {budget:g}x", A.P3, budget) for budget in (1.0, 4.0, 8.0)]
    arms += [(f"{A.P2} at {budget:g}x", A.P2, budget) for budget in (1.0, 8.0)]
    arms += [("P1 uniform grid 2x1", "P1 uniform grid 2x1", None), (f"{A.P4} at 0.25x", A.P4, 0.25), (f"{A.P4} at 0.5x", A.P4, 0.5)]
    rows = {}
    for label, policy, budget in arms:
        rates = [A.scene_rates(r, policy, budget) for r in results]
        if all(rate is not None for rate in (*whole, *rates)):
            rows[f"{label} minus P0"] = triple(A.paired_difference(rates, whole))
    return rows


def seed_spread(results: list[dict]) -> dict:
    """Budget 1: the pooled rate of every random seed, lowest and highest (P3's draws overlap between seeds)."""
    spread = {}
    for policy in (A.P2, A.P3):
        seeds = sorted({run["seed"] for r in results for run in r["search"] if run["policy"].startswith(policy) and run["budget"] == 1.0})
        rates = [A.pooled([np.array(run["hit1"], dtype=float) for r in results for run in r["search"]
                           if run["policy"].startswith(policy) and run["budget"] == 1.0 and run["seed"] == seed]) for seed in seeds]
        if rates:
            spread[policy] = {"seeds": len(rates), "lowest": min(rates), "highest": max(rates)}
    return spread


def geometry_paired(results: list[dict], resamples: int = A.RESAMPLES) -> dict:
    """Fused cloud against one keyframe on the parts both can read: median of (fused minus single-frame) error in cm,
    with a 95% interval from resampling scenes, and how often the single keyframe is the closer one."""
    rows, rng = {}, np.random.default_rng(0)
    for voxel in results[0]["geometry"]["fused"]:
        per_scene = [np.array([fused - single for fused, single in zip(r["geometry"]["fused"][voxel]["contact_cm"],
                                                                        r["geometry"]["single_frame"]["contact_cm"]) if fused is not None])
                     for r in results]
        per_scene = [scene for scene in per_scene if len(scene)]
        draws = [np.median(np.concatenate([per_scene[i] for i in rng.integers(len(per_scene), size=len(per_scene))]))
                 for _ in range(resamples)]
        pooled = np.concatenate(per_scene)
        rows[f"fused {voxel} m"] = {"parts": len(pooled), "median_difference_cm": float(np.median(pooled)),
                                    "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5)),
                                    "single_frame_closer": float(np.mean(pooled > 0)),
                                    "read_ms_per_part": float(np.mean([r["geometry"]["fused"][voxel]["read_ms_per_part"] for r in results]))}
    return rows


def reading(results: list[dict]) -> dict:
    rows, per_policy = {}, {}
    for name in READING:
        per_scene = [np.array(r["reading"][name]["right_part"], dtype=float) for r in results if name in r.get("reading", {})]
        if len(per_scene) == len(results):
            seconds = [s for r in results for s in r["reading"][name]["seconds"]]
            rows[name] = triple(A.bootstrap(per_scene))
            rows[name]["seconds_per_query"] = float(np.median(seconds)) if seconds else float("nan")
            per_policy[name] = per_scene
    if len(per_policy) == 2:
        rows["P4 minus P3"] = triple(A.paired_difference(*per_policy.values()))
    return rows


def missing(results: list[dict]) -> dict:
    """Scenes that lack a policy at budget 1 or a model reading: such a policy is left out of the summary."""
    gaps = {name: [r["visit"] for r in results if A.scene_rates(r, name, None if name == A.P0 else 1.0) is None]
            for name in (A.P0, A.P2, A.P3, A.P4)}
    gaps.update({f"reading, {name}": [r["visit"] for r in results if name not in r.get("reading", {})] for name in READING})
    return {name: visits for name, visits in gaps.items() if visits}


def geometry(results: list[dict]) -> dict:
    def stats(errors: list[float | None]) -> dict:
        found = np.array([e for e in errors if e is not None], dtype=float)
        if not len(found):
            return {"parts": len(errors), "read": 0, "median_cm": float("nan"), "within_3cm": float("nan"), "within_3cm_of_all": 0.0}
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
    everything = A.load_results(args.runs)
    results = A.scored(everything)  # a scene without an answerable task has nothing to score
    if not results:
        raise SystemExit(f"no scene results in {args.runs}")
    summary = {"scenes": len(results), "visits": [r["visit"] for r in results], "scenes_run": len(everything),
               "scenes_without_tasks": [r["visit"] for r in everything if not r["tasks"]["answerable"]], "missing": missing(results),
               "keyframes_without_lidar": {r["visit"]: r["keyframes_without_lidar"] for r in everything if r.get("keyframes_without_lidar")},
               "tasks": int(sum(r["tasks"]["answerable"] for r in results)),
               "tasks_all": int(sum(r["tasks"]["all"] for r in everything)), "keyframes": results[0]["keyframes"],
               "first_keyframe_right": at_budget_one(results, "hit1"), "right_keyframe_in_top3": at_budget_one(results, "hit3"),
               "curves": {name: A.curve(results, name) for name in (A.P1, A.P2, A.P3, A.P4)},
               "affordance_at_one_against_more": more_budget(results), "others_against_whole": against_whole(results),
               "seed_spread": seed_spread(results), "geometry_paired": geometry_paired(results),
               "reading": reading(results), "geometry": geometry(results),
               "parts_seen": {kind: int(sum(r["parts_seen"][kind] for r in everything)) for kind in results[0]["parts_seen"]},
               "parts": int(sum(r["parts"] for r in everything)),
               "memory": {key: int(sum(r["memory"][key] for r in results)) for key in ("tokens", "parts", "true_parts_with_token", "mixed_regions")},
               "storage_bytes_per_scene": {key: float(np.mean([r["storage"][key] for r in results])) for key in results[0]["storage"]},
               "write_s_per_scene": float(np.mean([r["write_s"] for r in results]))}
    first, read = summary["first_keyframe_right"], summary["reading"]
    against = {name: (first[f"P4 minus {name}"]["rate"], first[f"P4 minus {name}"]["low"], first[f"P4 minus {name}"]["high"])
               if f"P4 minus {name}" in first else None for name in (A.P2, A.P3)}
    summary["rule_21_6_category"] = A.g1_verdict(against[A.P3], against[A.P2], read.get("P4 affordance regions", {}).get("rate"),
                                 read.get("P3 generic regions", {}).get("rate"))
    (args.runs / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['scenes']} scenes, {summary['tasks']} answerable tasks of {summary['tasks_all']}")
    for name, row in first.items():
        print(f"  first keyframe right, {name}: {row['rate']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
    for name, row in read.items():
        print(f"  model picks the right part, {name}: {row['rate']:+.3f} [{row['low']:+.3f}, {row['high']:+.3f}]")
    for name, visits in summary["missing"].items():
        print(f"  WARNING: {name} is missing in {len(visits)} scenes and is left out: {' '.join(visits)}")
    print("  rule of section 21.6, category:", summary["rule_21_6_category"], "(whether gate G1 is passed is the project lead's decision)")


if __name__ == "__main__":
    main()
