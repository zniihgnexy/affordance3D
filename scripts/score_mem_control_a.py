"""Control a of round 7 (discussion draft, section 24.7): write every generic region, or every grid level, into the memory.

    python scripts/score_mem_control_a.py     # reads runs/mem_phase1/diagnostics/ and controls/features_<visit>.npz, writes runs/mem_phase1/control_a.json

CPU only. Retrieval is scored as in section 23 ("the first keyframe shows the part", weight 1, empty keyframe -3) and
compared on the same tasks with P4 at budget 1; an arm is close to P4 by the rule of section 24.3.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.mem import phase1_diagnostics as D
from affordance3d.mem import phase1_stats as A

P3_SEEDS = ("P3 s0", "P3 s1", "P3 s2")
READINGS = {"close": "close to P4: writing everything does about as well; P4's advantage is only that it stores far less",
            "behind": "behind P4: P4 is more accurate as well as smaller (with the annotations)",
            "undetermined": "undetermined: the interval is too wide to call it close"}


def first_right(scene: dict, vectors: np.ndarray, keyframe: np.ndarray) -> np.ndarray:
    """Per task: 1 when the best-scoring keyframe shows the part, with these tokens in the memory."""
    return np.array([D.first_right(D.keyframe_scores(query, scene["keyframes"], vectors, keyframe), right)
                     for query, right in zip(scene["queries"], scene["truth"]["relevant"])], dtype=float)


def memories(scene: dict, features: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Tokens of each arm: (vectors, keyframe of each token)."""
    grids = sorted(key.split("/")[0] for key in features if key.startswith("grid ") and key.endswith("/vectors"))
    width = scene["keyframes"].shape[1]
    return {"P0": (np.zeros((0, width)), np.zeros(0, dtype=np.int64)),
            "P4": (scene["vectors/P4"], scene["keyframe/P4"]),
            "a: every generic region": (features["pool/vectors"], features["pool/keyframe"]),
            "a: every grid level": (np.concatenate([features[f"{g}/vectors"] for g in grids]),
                                    np.concatenate([features[f"{g}/keyframe"] for g in grids]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    per_arm: dict[str, list[np.ndarray]] = {}
    tokens: dict[str, list[int]] = {}
    for result in A.scored(A.load_results(args.runs)):
        visit = result["visit"]
        with np.load(args.runs / "diagnostics" / f"{visit}.npz") as saved:
            scene = {**{key: saved[key] for key in saved.files if key != "truth"}, "truth": json.loads(str(saved["truth"]))}
        features_file = args.runs / "controls" / f"features_{visit}.npz"
        if not features_file.exists():
            raise SystemExit(f"{visit}: no {features_file.name}; run scripts/run_mem_controls.py --arms d first")
        with np.load(features_file) as saved:
            features = {key: saved[key] for key in saved.files}
        for name, (vectors, keyframe) in memories(scene, features).items():
            per_arm.setdefault(name, []).append(first_right(scene, vectors, keyframe))
            tokens.setdefault(name, []).append(len(vectors))
        per_arm.setdefault("P3", []).append(np.mean([first_right(scene, scene[f"vectors/{s}"], scene[f"keyframe/{s}"]) for s in P3_SEEDS], axis=0))
        tokens.setdefault("P3", []).append(len(scene["vectors/P3 s0"]))
    summary = {"scenes": len(per_arm["P4"]), "tasks": int(sum(len(x) for x in per_arm["P4"])), "arms": {}, "against_p4": {}}
    for name, rows in per_arm.items():
        rate = A.bootstrap(rows)
        summary["arms"][name] = {"rate": rate[0], "low": rate[1], "high": rate[2], "tokens_per_scene": float(np.median(tokens[name]))}
    for name in (n for n in per_arm if n.startswith("a: ")):
        difference = A.paired_difference(per_arm["P4"], per_arm[name])
        verdict = A.closeness(difference)
        summary["against_p4"][name] = {"rate": difference[0], "low": difference[1], "high": difference[2], "verdict": verdict,
                                       "reading": READINGS[verdict]}
    (args.runs / "control_a.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['scenes']} scenes, {summary['tasks']} tasks; first keyframe right:")
    for name, row in summary["arms"].items():
        print(f"  {name}: {100 * row['rate']:.1f}% [{100 * row['low']:.1f}, {100 * row['high']:.1f}], {row['tokens_per_scene']:.0f} tokens per scene")
    for name, row in summary["against_p4"].items():
        print(f"  P4 minus {name}: {100 * row['rate']:+.1f} [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}] -> {row['reading']}")


if __name__ == "__main__":
    main()
