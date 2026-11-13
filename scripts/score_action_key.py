"""Does a functional key help keyframe retrieval beyond meaning? (discussion draft, section 24.10; CPU; an upper bound)

    python scripts/score_action_key.py      # reads runs/mem_phase1/ and its diagnostics, writes runs/mem_phase1/action_key.json

Every P4 region gets the annotated action type of its patch; a task's affinity over the nine types comes from its
words and one of three key texts (the action phrase, the part words alone, the verb alone), or is set to its true
type. Retrieval is scored as in section 23 ("the first keyframe shows the part") and compared, on the same tasks,
with no key at all.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the output)

from affordance3d.mem import keyframes, policies
from affordance3d.mem import phase1_stats as A
from affordance3d.mem import pipeline as P
from affordance3d.mem import search as S
from affordance3d.mem import tokens as T
from report import mem_steps_helpers as H
from report.mem_demo_helpers import scaled_mono_depths

MONO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
PART_WORDS = {  # the action phrases of search.ACTION_PROMPTS without their verbs
    "rotate": "a knob or a dial", "key_press": "a key or a button on a keyboard or a remote",
    "tip_push": "a switch or a button", "hook_pull": "a handle of a drawer, a door or a cabinet",
    "pinch_pull": "a small knob", "hook_turn": "a door handle or a window handle", "foot_push": "a pedal",
    "plug_in": "a socket", "unplug": "a plug in a socket"}
VERBS = {"rotate": "rotate", "key_press": "press a key", "tip_push": "push with a fingertip", "hook_pull": "pull",
         "pinch_pull": "pinch and pull", "hook_turn": "turn", "foot_push": "push with the foot", "plug_in": "plug in",
         "unplug": "unplug"}
KEYS = {"action phrase": S.ACTION_PROMPTS, "part words only": PART_WORDS, "verb only": VERBS}
CLOSE_TO_NOTHING = 0.05  # the true type adding less than this means little room for an action key


def true_affinity(task_actions: set[int], n_actions: int = len(T.ACTIONS)) -> np.ndarray:
    """One for the task's true action type(s), shared evenly; zero elsewhere."""
    affinity = np.zeros(n_actions)
    affinity[sorted(task_actions)] = 1.0 / len(task_actions)
    return affinity


def first_right(query: np.ndarray, scene: dict, right: list[int], affinity: np.ndarray | None) -> int:
    scores = S.score_keyframes(query, scene["keyframes"], scene["vectors"], scene["keyframe"], scene["action"], affinity)
    return int(int(scores.ranking()[0]) in set(right))


def region_actions(result: dict, runs: Path, data: Path) -> tuple[list[int], np.ndarray, list[set[int]]]:
    """The annotated action of each P4 region (in the order of the phase-1 draws), their boxes, and each task's
    true action types; recomputed as scripts/diagnose_mem_phase1.py builds P4."""
    scene = P.load_scene(data, result["visit"], result["video"])
    times = keyframes.scene_keyframes(scene, result["keyframes"], result["n_keyframes"])
    mono = scaled_mono_depths(scene, times, MONO_ID, runs / "mono" / result["video"])
    frames = [P.input_frame(scene, t, d) for t, d in zip(times, mono)]
    labels = P.label_maps(scene, P.gt_frames(scene, times))
    part_actions = [T.ACTIONS.index(e.label) if e.label in T.ACTIONS else T.NO_ACTION for e in scene.elements]
    regions = [region for k, (label, frame) in enumerate(zip(labels, frames))
               for region in T.find_regions(T.action_map(label, part_actions), frame.depth_m, k)[1]]
    boxes = np.array([c.box for c in policies.affordance_candidates(regions, frames[0].image_hw)], dtype=np.int64)
    tasks = [task for task in H.tasks_of(scene, labels) if task.keyframes]
    if [task.text for task in tasks] != result["tasks"]["texts"]:
        raise SystemExit(f"{result['visit']}: the tasks differ from those of the phase-1 run")
    task_actions = [{part_actions[p] for p in task.parts if part_actions[p] != T.NO_ACTION} for task in tasks]
    return [r.action for r in regions], boxes.reshape(len(regions), 4), task_actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    args = parser.parse_args()
    from affordance3d.mem.encoders import ImageTextEncoder

    encoder = ImageTextEncoder()
    key_vectors = {name: encoder.texts([texts[a] for a in T.ACTIONS]) for name, texts in KEYS.items()}
    arms = ["no key", *KEYS, "true type"]
    per_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
    for result in A.scored(A.load_results(args.runs)):
        with np.load(args.runs / "diagnostics" / f"{result['visit']}.npz") as saved:
            truth = json.loads(str(saved["truth"]))
            scene = {"keyframes": saved["keyframes"], "queries": saved["queries"], "vectors": saved["vectors/P4"],
                     "keyframe": saved["keyframe/P4"], "boxes": saved["boxes/P4"]}
        actions, boxes, task_actions = region_actions(result, args.runs.parent, args.data)
        if not np.array_equal(boxes, scene["boxes"]):
            raise SystemExit(f"{result['visit']}: the recomputed P4 regions differ from the saved ones")
        scene["action"] = np.array(actions, dtype=np.int64)
        for arm in arms:
            rows = []
            for query, right, true_types in zip(scene["queries"], truth["relevant"], task_actions):
                if arm == "no key":
                    affinity = None
                elif arm == "true type":
                    affinity = true_affinity(true_types) if true_types else None
                else:
                    affinity = S.action_affinity(query, key_vectors[arm])
                rows.append(first_right(query, scene, right, affinity))
            per_arm[arm].append(np.array(rows, dtype=float))
        print(f"{result['visit']}: " + ", ".join(f"{arm} {np.mean(per_arm[arm][-1]):.2f}" for arm in arms), flush=True)
    summary: dict = {"scenes": len(per_arm["no key"]), "tasks": int(sum(len(x) for x in per_arm["no key"])), "arms": {}, "against_no_key": {}}
    for arm, rows in per_arm.items():
        rate = A.bootstrap(rows)
        summary["arms"][arm] = {"rate": rate[0], "low": rate[1], "high": rate[2]}
    for arm in arms[1:]:
        d = A.paired_difference(per_arm[arm], per_arm["no key"])
        summary["against_no_key"][arm] = {"rate": d[0], "low": d[1], "high": d[2], "helps": bool(d[1] > 0)}
    verb_part = A.paired_difference(per_arm["verb only"], per_arm["part words only"])
    summary["verb_minus_part"] = {"rate": verb_part[0], "low": verb_part[1], "high": verb_part[2]}
    summary["little_room"] = bool(summary["against_no_key"]["true type"]["rate"] < CLOSE_TO_NOTHING)
    (args.runs / "action_key.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['scenes']} scenes, {summary['tasks']} tasks; first keyframe right:")
    for arm, row in summary["arms"].items():
        extra = summary["against_no_key"].get(arm)
        tail = f"  minus no key {100 * extra['rate']:+.1f} [{100 * extra['low']:+.1f}, {100 * extra['high']:+.1f}]" if extra else ""
        print(f"  {arm}: {100 * row['rate']:.1f}% [{100 * row['low']:.1f}, {100 * row['high']:.1f}]{tail}")
    print(f"  verb only minus part words only: {100 * verb_part[0]:+.1f} [{100 * verb_part[1]:+.1f}, {100 * verb_part[2]:+.1f}]")


if __name__ == "__main__":
    main()
