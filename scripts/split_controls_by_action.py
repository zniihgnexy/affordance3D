"""The round-7 controls split by the action type of each task's target part, with a test of whether the gap differs
between action types (discussion draft, section 24.5; CPU only; annotations are used for grouping only).

    python scripts/split_controls_by_action.py     # reads runs/mem_phase1/controls/<visit>.json, writes runs/mem_phase1/controls_by_action.json

The test: the per-task difference P4 minus c3 is averaged per action type; the statistic is the task-weighted spread of
those averages around the overall one. Action labels are shuffled among the tasks of the same scene (scenes keep their
own difficulty), and the p-value is the share of shuffles with a spread at least as large as the observed one.
"""

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.mem import phase1_stats as A

ARMS = ("p4n", "c1", "c2", "c3")
MIN_TASKS = 10
SHUFFLES = 10000
SEED = 0


def spread(differences: np.ndarray, actions: np.ndarray) -> float:
    """Task-weighted spread of the per-action mean differences around the overall mean."""
    overall = differences.mean()
    return float(sum((actions == a).sum() * (differences[actions == a].mean() - overall) ** 2 for a in np.unique(actions)))


def heterogeneity(differences: np.ndarray, actions: np.ndarray, scenes: np.ndarray, shuffles: int = SHUFFLES,
                  seed: int = SEED) -> tuple[float, float]:
    """Observed spread and its p-value, shuffling action labels within each scene."""
    rng, observed = np.random.default_rng(seed), spread(differences, actions)
    groups = [np.flatnonzero(scenes == s) for s in np.unique(scenes)]
    at_least = 0
    for _ in range(shuffles):
        shuffled = actions.copy()
        for rows in groups:
            shuffled[rows] = actions[rng.permutation(rows)]
        at_least += spread(differences, shuffled) >= observed - 1e-12
    return observed, (at_least + 1) / (shuffles + 1)


def task_actions(data: Path, record: dict) -> list[str]:
    """The action type(s) of the part each task names, in the order of the record's tasks."""
    with contextlib.redirect_stdout(io.StringIO()):
        import open3d  # noqa: F401  (its banner would end up in the output)
    from affordance3d.mem import pipeline as P

    scene = P.load_scene(data, record["visit"], record["video"])
    labels_of = {}
    for text, annot_ids in scene.tasks:
        actions = sorted({e.label for e in scene.elements if e.annot_id in annot_ids})
        labels_of.setdefault(text, "+".join(actions) if actions else "none")
    return [labels_of.get(text, "unknown") for text in record["tasks"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    args = parser.parse_args()
    by_action: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    rows = []  # (scene, action, P4 - c3) per task
    for path in sorted((args.runs / "controls").glob("*.json")):
        if not path.stem.isdigit():
            continue
        record = json.loads(path.read_text())
        per_task = task_actions(args.data, record)
        hits = {arm: np.array([answer["hit"] for answer in record["arms"][arm]], dtype=float) for arm in ARMS}
        for action in set(per_task):
            chosen = [i for i, a in enumerate(per_task) if a == action]
            for arm in ARMS:
                by_action[action][arm].append(hits[arm][chosen])
        rows += [(record["visit"], action, hits["p4n"][i] - hits["c3"][i]) for i, action in enumerate(per_task)]
    counts = {action: int(sum(len(a) for a in arms["p4n"])) for action, arms in by_action.items()}
    summary: dict = {"actions": {}}
    for action in sorted(counts, key=lambda a: -counts[a]):
        row = {"tasks": counts[action], **{arm: A.pooled(by_action[action][arm]) for arm in ARMS}}
        if counts[action] >= MIN_TASKS:
            row["p4_minus_c3"] = A.paired_difference(by_action[action]["p4n"], by_action[action]["c3"])
            row["p4_minus_c1"] = A.paired_difference(by_action[action]["p4n"], by_action[action]["c1"])
        summary["actions"][action] = row
    kept = [r for r in rows if counts[r[1]] >= MIN_TASKS]
    scenes, actions = np.array([r[0] for r in kept]), np.array([r[1] for r in kept])
    observed, p_value = heterogeneity(np.array([r[2] for r in kept]), actions, scenes)
    summary["heterogeneity_p4_minus_c3"] = {"tasks": len(kept), "action_types": int(len(np.unique(actions))),
                                            "spread": observed, "p_value": p_value, "shuffles": SHUFFLES}
    (args.runs / "controls_by_action.json").write_text(json.dumps(summary, indent=1))
    for action, row in summary["actions"].items():
        gap = row.get("p4_minus_c3")
        extra = f" | P4-c3 {100 * gap[0]:+.1f} [{100 * gap[1]:+.1f}, {100 * gap[2]:+.1f}]" if gap else ""
        print(f"{action:22s} n={row['tasks']:3d} | P4 {100 * row['p4n']:5.1f} | c3 {100 * row['c3']:5.1f}{extra}")
    print(f"gap differs between action types? {len(kept)} tasks, {summary['heterogeneity_p4_minus_c3']['action_types']} types, "
          f"p = {p_value:.4f}")


if __name__ == "__main__":
    main()
