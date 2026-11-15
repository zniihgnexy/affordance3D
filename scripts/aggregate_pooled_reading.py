"""Phase 2b end to end (discussion draft, section 26.2): the readings fixed before the run, on the CPU.

    python scripts/aggregate_pooled_reading.py      # reads runs/mem_phase1/pooled_reading/, writes runs/mem_phase1/pooled_reading_summary.json

Per pool size K: each arm's rate of right picks (95% interval from resampling target scenes), and function index minus
semantic on the same tasks. The reading: the least-squares slope of that gap over log2 K, with its interval; "supported
end to end" when the slope's lower end is above 0 and the gap at K = 16 is at least 5 points, otherwise "not supported
end to end either" (then gate G2 of the round-10 assessment: the project lead decides).

Check: K = 1 is the single scene, where the arms are p4n and c4 of section 24.8 (Qwen3-VL-8B: 54.4% and 47.7%); the
rates are set beside those, and, when runs/mem_phase1/controls_8b/ holds their records, task by task: the same question
(keyframes shown, number of boxes), the same outcome.

Descriptive only: per arm and K, the share of tasks where a shown keyframe shows a target part (the retrieval hit of
section 26, first draw only), where one of the numbered boxes is right, and where the pick lies on another scene.
Only complete records count; the scenes still missing are listed.
"""

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.mem import phase1_stats as A


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


READER = _script("run_pooled_reading")
PR = READER.PR

OURS, BASE = "function index", "semantic"
GAP = f"{OURS} minus {BASE}"
REFERENCE_K1 = {OURS: ("p4n", 0.544), BASE: ("c4", 0.477)}  # section 24.8, Qwen3-VL-8B, the same 390 tasks
VALUES: dict[str, Callable[[dict, dict], bool]] = {
    "hit": lambda record, row: row["hit"],
    "top3_right": lambda record, row: row["top3_right"],
    "a_box_right": lambda record, row: row["a_box_right"],
    "on_another_scene": lambda record, row: row["chosen_scene"] not in (None, record["visit"]),
    "no_pick": lambda record, row: row["frame"] is None,
}
DESCRIPTIVE = ("top3_right", "a_box_right", "on_another_scene", "no_pick")


def interval(values: Sequence[float]) -> dict[str, float]:
    return dict(zip(("rate", "low", "high"), (float(v) for v in values)))


def load_records(folder: Path) -> list[dict]:
    """The per-scene records (<visit>.json); the answer caches are other files."""
    return [json.loads(path.read_text()) for path in A.result_files(folder)]


def check_records(records: Sequence[dict]) -> tuple[list[dict], list[str]]:
    """The complete records, after refusing any that differ in scene list, version, model or draws; and the scenes of the
    scene list that still lack a complete record."""
    if not records:
        raise SystemExit("no records yet: run scripts/run_pooled_reading.py first")
    first = records[0]
    for record in records:
        for key in ("scenes", "version", "model"):
            if record[key] != first[key]:
                raise SystemExit(f"{record['visit']}: its {key} differs from that of {first['visit']}")
        scenes = record["scenes"]
        expected = READER.first_draws(len(scenes), scenes.index(record["visit"]))
        if record["draws"] != {str(k): [scenes[i] for i in others] for k, others in expected.items()}:
            raise SystemExit(f"{record['visit']}: its draws are not the first draws of scripts/pooled_retrieval.py")
    complete = [record for record in records if record.get("complete")]
    done = {record["visit"] for record in complete}
    return complete, [visit for visit in first["scenes"] if visit not in done]


def outcomes(records: Sequence[dict], k: int, arm: str, value: str = "hit") -> list[np.ndarray]:
    """Per target scene, one value per task (0 or 1) for an arm at pool size `k`."""
    return [np.array([VALUES[value](record, row) for row in record["per_size"][str(k)][arm]], dtype=float) for record in records]


def gaps(records: Sequence[dict], k: int) -> list[np.ndarray]:
    """Per target scene and task: function index minus semantic (right picks)."""
    return [ours - base for ours, base in zip(outcomes(records, k, OURS), outcomes(records, k, BASE), strict=True)]


def size_entry(records: Sequence[dict], k: int) -> dict:
    """The rates of one pool size: right picks with intervals, the paired gap, and the descriptive shares."""
    entry: dict = {arm: {"hit": interval(A.bootstrap(outcomes(records, k, arm))),
                         **{name: A.pooled(outcomes(records, k, arm, name)) for name in DESCRIPTIVE}}
                   for arm in READER.ARMS}
    entry[GAP] = interval(A.paired_difference(outcomes(records, k, OURS), outcomes(records, k, BASE)))
    return entry


def reading(records: Sequence[dict]) -> dict:
    """The pre-registered reading: the gap's slope per doubling of K and the gap at the largest K."""
    sizes = READER.POOL_SIZES
    slope = PR.slope({k: gaps(records, k) for k in sizes})
    largest = A.pooled(gaps(records, sizes[-1]))
    supported = bool(slope[1] > 0 and largest >= PR.MIN_GAIN_AT_LARGEST)
    verdict = ("supported end to end" if supported else
               "not supported end to end either (gate G2 of the round-10 assessment: the project lead decides)")
    return {"slope_per_doubling": interval(slope), "gap_at_largest": largest, "largest_k": sizes[-1],
            "min_gap_at_largest": PR.MIN_GAIN_AT_LARGEST, "supported": supported, "verdict": verdict}


def control_rows(records: Sequence[dict], controls: Path, arm: str) -> list[list[dict]] | None:
    """The single-scene arm's rows for the same tasks, per record; None unless every scene has them."""
    rows = []
    for record in records:
        file = controls / f"{record['visit']}.json"
        if not file.exists():
            return None
        control = json.loads(file.read_text())
        if arm not in control["arms"] or control["tasks"] != record["tasks"]:
            return None
        rows.append(control["arms"][arm])
    return rows


def same_question(ours: dict, theirs: dict) -> bool:
    """The same keyframes shown (ours are named "<visit>/<time>") and as many numbered boxes."""
    shown = [name.split("/", 1)[1] for name in ours["keyframes_shown"]]
    return shown == theirs["keyframes_shown"] and ours["marks"] == theirs["marks"]


def k1_check(records: Sequence[dict], controls: Path) -> dict:
    """K = 1 beside p4n and c4 of section 24.8; task by task when their records are there."""
    check = {}
    for arm, (control, reference) in REFERENCE_K1.items():
        rate = A.pooled(outcomes(records, 1, arm))
        entry = {"rate": rate, "control": control, "reference": reference, "minus_reference": rate - reference}
        theirs = control_rows(records, controls, control)
        if theirs is not None:
            pairs = [(ours, their) for record, rows in zip(records, theirs) for ours, their in zip(record["per_size"]["1"][arm], rows)]
            entry.update({"control_rate": float(np.mean([their["hit"] for _, their in pairs])),
                          "same_question": float(np.mean([same_question(ours, their) for ours, their in pairs])),
                          "same_outcome": float(np.mean([ours["hit"] == their["hit"] for ours, their in pairs]))})
        check[arm] = entry
    return check


def summarise(records: Sequence[dict], missing: Sequence[str], controls: Path) -> dict:
    return {"scenes": len(records), "tasks": int(sum(len(record["tasks"]) for record in records)), "missing": list(missing),
            "complete": not missing, "model": records[0]["model"], "version": records[0]["version"],
            "pool_sizes": list(READER.POOL_SIZES), "per_size": {str(k): size_entry(records, k) for k in READER.POOL_SIZES},
            "reading": reading(records), "k1_check": k1_check(records, controls)}


def report(summary: dict) -> None:
    print(f"{summary['scenes']} scenes, {summary['tasks']} tasks" + (f"; missing {summary['missing']}" if summary["missing"] else ""))
    for k, entry in summary["per_size"].items():
        gap = entry[GAP]
        print(f"  K = {k}: semantic {100 * entry[BASE]['hit']['rate']:.1f}%, function index {100 * entry[OURS]['hit']['rate']:.1f}%; "
              f"gap {100 * gap['rate']:+.1f} [{100 * gap['low']:+.1f}, {100 * gap['high']:+.1f}]")
    result = summary["reading"]
    slope = result["slope_per_doubling"]
    print(f"slope per doubling {100 * slope['rate']:+.1f} [{100 * slope['low']:+.1f}, {100 * slope['high']:+.1f}] points; gap at "
          f"K = {result['largest_k']} {100 * result['gap_at_largest']:+.1f}: {result['verdict']}")
    for arm, check in summary["k1_check"].items():
        extra = f"; task by task: same question {check['same_question']:.0%}, same outcome {check['same_outcome']:.0%}" \
            if "same_outcome" in check else ""
        print(f"K = 1 {arm}: {100 * check['rate']:.1f}% against {check['control']} {100 * check['reference']:.1f}%{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--folder", type=Path, help="the records (default: <runs>/pooled_reading)")
    parser.add_argument("--controls", type=Path, help="the 8B controls of section 24.8 (default: <runs>/controls_8b)")
    args = parser.parse_args()
    folder = args.folder or args.runs / READER.OUT_FOLDERS["8b"]
    records, missing = check_records(load_records(folder))
    if not records:
        raise SystemExit(f"no complete record yet; missing {missing}")
    summary = summarise(records, missing, args.controls or args.runs / "controls_8b")
    (args.runs / f"{folder.name}_summary.json").write_text(json.dumps(summary, indent=1))  # pooled_reading_summary.json
    report(summary)


if __name__ == "__main__":
    main()
