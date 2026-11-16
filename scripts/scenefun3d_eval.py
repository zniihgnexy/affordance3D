"""Score predictions with the official SceneFun3D task-driven affordance grounding scorer (Task 2), unchanged.

    python scripts/scenefun3d_eval.py oracle --mode elements --out runs/scenefun3d_eval/oracle_elements
    python scripts/scenefun3d_eval.py score --pred_dir runs/scenefun3d_eval/oracle_elements

`oracle` writes the annotations themselves as predictions, to check the scoring chain: "elements" (every target
element its own mask; should score 100), "union" (all target elements of a task merged into one mask) and "first" (only
the first target element), which show what a method giving one mask per task can reach at best; "fun3du" writes
the ground truth in the Fun3DU format for the `fun3du` scorer (should score 100). `score` runs
third_party/scenefun3d/eval/affordance_grounding/evaluate.py against the ground truth written by that toolkit's
prepare_gt_val_data.py, and stores AP, AP50 and AP25 in <pred_dir>/score.json. CPU only.

    python -I scripts/scenefun3d_eval.py fun3du --pred_dir <dir> --out <file.json> [--reference <json>] [--fingerprints <json>]

    python -I scripts/scenefun3d_eval.py compare --first <dir> --second <dir> --out <file.json>

`compare` takes the paired difference first minus second on every task (section 27.2) and reads it by
scenefun3d_eval.compare_verdict. `fun3du` scores by the protocol of the published numbers (affordance3d.mem.scenefun3d_eval, Fun3DU / UniFunc3D):
<pred_dir>/<visit_id>.json holds [{"desc_id": ..., "pred_mask": [sorted cropped-scan indices]}] for every task of the
30 validation scenes. --reference compares every task with per-task values released by another method (lists keyed
by metric, with visit_id and desc_id); --fingerprints compares the sha256 of our crop masks, descriptions and
annotations with those another release was scored on. Downloaded files are read as data only, never imported.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from affordance3d.mem import scenefun3d_eval as SE  # noqa: E402

TOOLKIT = REPO_ROOT / "third_party" / "scenefun3d"
DATA = REPO_ROOT / "data" / "scenefun3d"
GT_DIR = REPO_ROOT / "runs" / "scenefun3d_eval" / "gt_val"
SCENES = TOOLKIT / "benchmark_file_lists" / "val_scenes.txt"
MODES = ("elements", "union", "first", "fun3du")


def write_oracle(out: Path, mode: str) -> int:
    count = 0
    for visit in [line.strip() for line in SCENES.read_text().splitlines() if line.strip()]:
        n_points = SE.scan_points(DATA / visit / f"{visit}_laser_scan.ply")
        annotations = json.loads((DATA / visit / f"{visit}_annotations.json").read_text())["annotations"]
        for task in json.loads((DATA / visit / f"{visit}_descriptions.json").read_text())["descriptions"]:
            masks = SE.element_masks(n_points, annotations, task["annot_id"])
            if mode == "union":
                masks = [np.logical_or.reduce(masks)]
            elif mode == "first":
                masks = masks[:1]
            SE.write_task(out, visit, task["desc_id"], masks, [1.0] * len(masks), n_points)
            count += 1
    return count


def write_oracle_fun3du(out: Path) -> int:
    """The ground truth itself in the Fun3DU format (the union of each task's elements on the cropped scan)."""
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for visit in [line.strip() for line in SCENES.read_text().splitlines() if line.strip()]:
        crop = np.load(DATA / visit / f"{visit}_crop_mask.npy", allow_pickle=False)
        annotations = json.loads((DATA / visit / f"{visit}_annotations.json").read_text())["annotations"]
        tasks = json.loads((DATA / visit / f"{visit}_descriptions.json").read_text())["descriptions"]
        entries = [{"desc_id": t["desc_id"], "pred_mask": SE.task_ground_truth(crop, annotations, t["annot_id"]).tolist()}
                   for t in tasks]
        (out / f"{visit}.json").write_text(json.dumps(entries))
        count += len(entries)
    return count


# Two NumPy 2 incompatibilities of the toolkit, bridged without changing its code: it calls np.in1d, which NumPy 2
# removed (np.isin on the flattened input is the documented replacement, same result); and its loader passes each
# confidence as a one-element array, which NumPy 2 will not write into a single slot (the same values are passed as
# plain numbers). The loading and the scoring otherwise run unchanged.
RUN_OFFICIAL = ("import numpy as np, runpy\n"
                "if not hasattr(np, 'in1d'):\n"
                "    np.in1d = lambda a, b, **kw: np.isin(np.ravel(a), b, **kw)\n"
                "import eval.affordance_grounding.eval_utils.eval_script as es\n"
                "_evaluate = es.evaluate\n"
                "def evaluate(preds, gt_path, print_result=True):\n"
                "    for item in preds.values():\n"
                "        item['pred_scores'] = np.asarray(item['pred_scores'], dtype=float).ravel()\n"
                "    return _evaluate(preds, gt_path, print_result)\n"
                "es.evaluate = evaluate\n"
                "runpy.run_module('eval.affordance_grounding.evaluate', run_name='__main__', alter_sys=True)\n")


def score(pred_dir: Path, gt_dir: Path) -> dict:
    result = subprocess.run([sys.executable, "-c", RUN_OFFICIAL, "--pred_dir", str(pred_dir.resolve()), "--gt_dir", str(gt_dir.resolve())],
                            cwd=TOOLKIT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"the official scorer failed:\n{result.stderr[-3000:]}")
    match = re.search(r"^average\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise SystemExit(f"could not read the scorer's output:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    scores = {"AP": float(match[1]), "AP50": float(match[2]), "AP25": float(match[3]),
              "tasks": len(list(pred_dir.glob("*_*.txt"))), "scorer": "SceneFun3D toolkit, eval/affordance_grounding/evaluate.py"}
    (pred_dir / "score.json").write_text(json.dumps(scores, indent=1))
    return scores


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fun3du(pred_dir: Path, reference: Path | None, fingerprints: Path | None) -> dict:
    expected = json.loads(fingerprints.read_text()) if fingerprints else {}
    refs = {}
    if reference:
        table = json.loads(reference.read_text())
        keys = list(zip(table["visit_id"], table["desc_id"]))
        refs = {key: {m: table[m][i] for m in SE.FUN3DU_METRICS} for i, key in enumerate(keys)}
    rows, mismatches, fingerprint_errors, empty, compared, hashes = [], [], [], 0, 0, 0
    for visit in [line.strip() for line in SCENES.read_text().splitlines() if line.strip()]:
        folder = DATA / visit
        files = {"crop_sha256": folder / f"{visit}_crop_mask.npy", "descriptions_sha256": folder / f"{visit}_descriptions.json",
                 "annotations_sha256": folder / f"{visit}_annotations.json"}
        for key, path in files.items():
            if key in expected.get(visit, {}):
                hashes += 1
                if expected[visit][key] != sha256(path):
                    fingerprint_errors.append(f"{visit} {key}")
        crop = np.load(files["crop_sha256"], allow_pickle=False)
        annotations = json.loads(files["annotations_sha256"].read_text())["annotations"]
        tasks = json.loads(files["descriptions_sha256"].read_text())["descriptions"]
        predicted = {entry["desc_id"]: np.asarray(entry["pred_mask"], dtype=np.int64)
                     for entry in json.loads((pred_dir / f"{visit}.json").read_text())}
        if set(predicted) != {task["desc_id"] for task in tasks}:
            raise SystemExit(f"{visit}: the predictions do not cover exactly the scene's tasks")
        n_points = int((crop == 1).sum())
        for task in tasks:
            pred = predicted[task["desc_id"]]
            if pred.size and (pred[0] < 0 or pred[-1] >= n_points or np.any(np.diff(pred) <= 0)):
                raise SystemExit(f"{visit}/{task['desc_id']}: indices must be sorted, unique and in the cropped scan")
            empty += int(pred.size == 0)
            gt = SE.task_ground_truth(crop, annotations, task["annot_id"])
            if gt.size == 0:
                raise SystemExit(f"{visit}/{task['desc_id']}: the task has no ground-truth point on the cropped scan")
            row = SE.fun3du_scores(gt, pred)
            rows.append(row)
            want = refs.get((visit, task["desc_id"]))
            if want is not None:
                compared += 1
                if any(abs(row[m] - want[m]) > (1e-7 if m == "mIoU" else 0) for m in SE.FUN3DU_METRICS):
                    mismatches.append(f"{visit}/{task['desc_id']}")
    if refs and compared != len(rows):
        raise SystemExit(f"the reference covers {compared} of {len(rows)} tasks; nothing is checked silently")
    scenes = len([line for line in SCENES.read_text().splitlines() if line.strip()])
    if expected and hashes != 3 * scenes:
        raise SystemExit(f"only {hashes} of {3 * scenes} fingerprints (3 files of every scene) could be compared")
    return {"protocol": "Fun3DU point precision / recall (as released by UniFunc3D)", "tasks": len(rows),
            "empty_predictions": empty, "means": SE.fun3du_means(rows),
            "reference_tasks_compared": compared, "reference_mismatches": mismatches,
            "fingerprints_compared": hashes, "fingerprint_mismatches": fingerprint_errors}


def task_rows(pred_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Per scene, the Fun3DU-protocol scores of every task in the descriptions order."""
    rows = {}
    for visit in [line.strip() for line in SCENES.read_text().splitlines() if line.strip()]:
        folder = DATA / visit
        crop = np.load(folder / f"{visit}_crop_mask.npy", allow_pickle=False)
        annotations = json.loads((folder / f"{visit}_annotations.json").read_text())["annotations"]
        tasks = json.loads((folder / f"{visit}_descriptions.json").read_text())["descriptions"]
        predicted = {e["desc_id"]: np.asarray(e["pred_mask"], dtype=np.int64) for e in json.loads((pred_dir / f"{visit}.json").read_text())}
        if set(predicted) != {task["desc_id"] for task in tasks}:
            raise SystemExit(f"{pred_dir.name}/{visit}: the predictions do not cover exactly the scene's tasks")
        rows[visit] = [SE.fun3du_scores(SE.task_ground_truth(crop, annotations, t["annot_id"]), predicted[t["desc_id"]]) for t in tasks]
    return rows


def compare(first: Path, second: Path) -> dict:
    """first minus second on the same tasks, every Fun3DU metric, with 95% intervals from resampling scenes (2,000
    resamples, seed 0, pooled over tasks), read by the rule of section 27.2 on AP25 and mIoU."""
    from affordance3d.mem import phase1_stats as A

    a, b = task_rows(first), task_rows(second)
    diffs = {}
    for metric in SE.FUN3DU_METRICS:
        values = A.paired_difference([np.array([r[metric] for r in a[v]]) for v in a], [np.array([r[metric] for r in b[v]]) for v in a])
        diffs[metric] = tuple(100 * x for x in values)
    return {"first": str(first.relative_to(REPO_ROOT)) if first.is_relative_to(REPO_ROOT) else first.name,
            "second": str(second.relative_to(REPO_ROOT)) if second.is_relative_to(REPO_ROOT) else second.name,
            "tasks": sum(len(r) for r in a.values()), "first_minus_second": {m: dict(zip(("rate", "low", "high"), d)) for m, d in diffs.items()},
            "verdict": SE.compare_verdict(diffs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    oracle = sub.add_parser("oracle")
    oracle.add_argument("--mode", choices=MODES, required=True)
    oracle.add_argument("--out", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--pred_dir", type=Path, required=True)
    scoring.add_argument("--gt_dir", type=Path, default=GT_DIR)
    graded = sub.add_parser("fun3du")
    graded.add_argument("--pred_dir", type=Path, required=True)
    graded.add_argument("--out", type=Path, required=True)
    graded.add_argument("--reference", type=Path)
    graded.add_argument("--fingerprints", type=Path)
    comparing = sub.add_parser("compare")
    comparing.add_argument("--first", type=Path, required=True)
    comparing.add_argument("--second", type=Path, required=True)
    comparing.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare(args.first.resolve(), args.second.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
    elif args.command == "fun3du":
        result = fun3du(args.pred_dir, args.reference, args.fingerprints)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v for k, v in result.items() if k != "reference_mismatches"}, indent=1))
        print(f"{len(result['reference_mismatches'])} tasks differ from the reference")
    elif args.command == "oracle":
        written = write_oracle_fun3du(args.out) if args.mode == "fun3du" else write_oracle(args.out, args.mode)
        print(f"wrote {written} tasks to {args.out}")
    else:
        print(json.dumps(score(args.pred_dir, args.gt_dir)))


if __name__ == "__main__":
    main()
