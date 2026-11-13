"""How often the frames a reading arm can choose from show the target part at all (discussion draft, section 24.5; CPU).

    python scripts/count_frame_ceiling.py     # writes runs/mem_phase1/frame_ceiling.json

The 390 tasks are those whose part is visible in at least one of our 40 keyframes; c3 chooses among 40 evenly spaced
frames instead, so its ceiling may be lower. Counted from the projected annotations, for scoring only.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the output)

from affordance3d.mem import keyframes
from affordance3d.mem import phase1_stats as A
from affordance3d.mem import pipeline as P
from report import mem_phase1_helpers as R
from report import mem_steps_helpers as H


def shows_target(labels: list, parts: tuple[int, ...]) -> bool:
    """Whether any of the frames shows a target part (the centre pixel of one of its parts lies in a frame)."""
    return bool(R.target_pixels(labels, parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    per_scene = {}
    for result in A.scored(A.load_results(args.runs)):
        scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", result["visit"], result["video"])
        steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
        uniform = keyframes.scene_keyframes(scene, "uniform", result["n_keyframes"])
        steady_labels = P.label_maps(scene, P.gt_frames(scene, steady))
        uniform_labels = P.label_maps(scene, P.gt_frames(scene, uniform))
        tasks = [task for task in H.tasks_of(scene, steady_labels) if task.keyframes]
        per_scene[result["visit"]] = [shows_target(uniform_labels, task.parts) for task in tasks]
        print(f"{result['visit']}: {sum(per_scene[result['visit']])} of {len(tasks)}", flush=True)
    flat = [shown for rows in per_scene.values() for shown in rows]
    summary = {"tasks": len(flat), "shown_in_uniform_frames": sum(flat), "share": sum(flat) / len(flat), "per_scene": per_scene}
    (args.runs / "frame_ceiling.json").write_text(json.dumps(summary, indent=1))
    print(f"the 40 even frames show the target in {sum(flat)} of {len(flat)} tasks ({100 * summary['share']:.1f}%)")


if __name__ == "__main__":
    main()
