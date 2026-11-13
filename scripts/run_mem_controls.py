"""Controls of round 7 on one scene (discussion draft, section 24): can a vision-language model find the part without this memory?

    python scripts/run_mem_controls.py --visit 421254                      # the first batch: p4n, c1, c2, c3
    python scripts/run_mem_controls.py --visit 421254 --arms c1            # one arm

Every arm uses Qwen3-VL-4B and is scored by one rule (grounding.hits_part): the centre of the chosen box lies on the
target part, widened by 8 pixels. The phase-1 rule (the box covers the centre of a target part) is stored alongside.

    p4n  the memory: P4's three best keyframes with its regions numbered; a neutral prompt; the model picks a number
    c1   no memory: the three best keyframes by whole-frame meaning (P0); the model draws the box itself
    c2   the memory's keyframes without its boxes: P4's three best keyframes; the model draws the box itself
    c3   no memory, the raw video: 40 frames taken evenly in time, shrunk; the model names a frame, then draws the box
         in that frame at full size

Needs the phase-1 results and the saved features of scripts/diagnose_mem_phase1.py (rankings come from them). Writes
runs/mem_phase1/controls/<visit>.json; the model's answers are cached in runs/mem_phase1/controls/answers_<visit>.json.
The ground truth (part ids projected from the scan) is used for scoring only.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the log)

from affordance3d.mem import grounding as G
from affordance3d.mem import keyframes, policies
from affordance3d.mem import phase1_diagnostics as D
from affordance3d.mem import pipeline as P
from affordance3d.mem import search as S
from affordance3d.mem.encoders import VLM_ID, VLM_REVISION, VisionLanguageModel
from report import mem_phase1_helpers as R
from report import mem_steps_helpers as H

ARMS = ("p4n", "c1", "c2", "c3")
CONTROLS_VERSION = "controls-v3"  # v2: every prompt defines "the part"; v3: longer answers (pilot, 2026-11-13)  # part of the answer cache key: change it when a prompt or an image changes
TOP_KEYFRAMES, MAX_MARKS, SHRINK = 3, 40, 2
ANSWER_TOKENS = {"number": 8, "box": 160}  # some answers explain first, then give the box


def answer_key(arm: str, prompt: str, frames: list[str], boxes: list) -> str:
    payload = json.dumps([CONTROLS_VERSION, H.MARKS_VERSION, VLM_REVISION, arm, prompt, frames, boxes])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def rgb_frames(scene: P.Scene, times: list[str]) -> list:
    """Input frames at the working resolution; only the colour image is needed here (depth is left empty)."""
    return [P.input_frame(scene, t, np.zeros(P.frame_hw(scene, t, P.PIXEL_STEP), dtype=np.float32)) for t in times]


def shrink(rgb: np.ndarray, factor: int = SHRINK) -> np.ndarray:
    return cv2.resize(rgb, (rgb.shape[1] // factor, rgb.shape[0] // factor), interpolation=cv2.INTER_AREA)


def scored(box: G.Box | None, labels: np.ndarray, parts: tuple[int, ...]) -> tuple[bool, bool, bool]:
    """(hit by the common rule, hit by the phase-1 rule, the frame shows a target part at all)."""
    centres = R.target_pixels([labels], parts).get(0, [])
    return G.hits_part(box, labels, parts), G.covers_a_centre(box, centres), bool(centres)


def free_box(reader: H.CachedReader, arm: str, task: str, frames: list, keyframes_shown: list[int], times: list[str],
             labels: list[np.ndarray], parts: tuple[int, ...]) -> dict:
    """c1, c2: the model sees three keyframes (numbered) and draws a box in one of them."""
    prompt = G.grounding_prompt(task, len(keyframes_shown))
    reply = reader.ask(answer_key(arm, prompt, [times[k] for k in keyframes_shown], []), [frames[k].rgb for k in keyframes_shown],
                       prompt, labels=G.image_labels(len(keyframes_shown)), max_new_tokens=ANSWER_TOKENS["box"])
    grounding = G.parse_grounding(reply["answer"], len(keyframes_shown))
    keyframe = None if grounding.image is None else keyframes_shown[grounding.image - 1]
    box = None if keyframe is None or grounding.box is None else G.to_pixels(grounding.box, frames[keyframe].image_hw)
    hit, hit_old, shows = scored(box, labels[keyframe], parts) if keyframe is not None else (False, False, False)
    record = G.answer_record(grounding, box, None if keyframe is None else times[keyframe], hit, hit_old, [reply["answer"]],
                             reply["seconds"], len(keyframes_shown))
    return {**record, "frame_right": shows, "keyframes_shown": [times[k] for k in keyframes_shown]}


def numbered_choice(reader: H.CachedReader, task: str, query: np.ndarray, frames: list, times: list[str], keyframe_vectors: np.ndarray,
                    candidates: list[policies.Candidate], vectors: np.ndarray, labels: list[np.ndarray], parts: tuple[int, ...]) -> dict:
    """p4n: as the phase-1 reading step (same keyframes, same numbered regions), with the neutral prompt."""
    scores = S.score_keyframes(query, keyframe_vectors, vectors, [c.keyframe for c in candidates], [0] * len(candidates))
    top = [int(k) for k in scores.ranking()[:TOP_KEYFRAMES]]
    inside = sorted((i for i, c in enumerate(candidates) if c.keyframe in top), key=lambda i: -scores.token_scores[i])[:MAX_MARKS]
    kept = [candidates[i] for i in sorted(inside)]
    images, marks = H.mark_boxes(frames, [(c.keyframe, c.box) for c in kept], top)
    if not marks:
        return {**G.answer_record(G.Grounding(None, None), None, None, False, False, [], 0.0, len(top)), "frame_right": False,
                "keyframes_shown": [times[k] for k in top], "marks": 0}
    prompt = G.numbered_prompt(task, len(marks))
    reply = reader.ask(answer_key("p4n", prompt, [times[k] for k in top], [list(kept[i].box) for i in marks]), images, prompt,
                       max_new_tokens=ANSWER_TOKENS["number"])
    mark = S.parse_mark(reply["answer"], len(marks))
    chosen = None if mark is None else kept[marks[mark - 1]]
    box = None if chosen is None else chosen.box
    hit, hit_old, shows = scored(box, labels[chosen.keyframe], parts) if chosen is not None else (False, False, False)
    record = G.answer_record(G.Grounding(mark, None), box, None if chosen is None else times[chosen.keyframe], hit, hit_old,
                             [reply["answer"]], reply["seconds"], len(top))
    return {**record, "frame_right": shows, "keyframes_shown": [times[k] for k in top], "marks": len(marks)}


def raw_video(reader: H.CachedReader, task: str, video: list, times: list[str], labels: list[np.ndarray], parts: tuple[int, ...]) -> dict:
    """c3: the model sees 40 evenly spaced frames (shrunk), names one, then draws the box in it at full size."""
    first = G.frame_prompt(task, len(video))
    pick = reader.ask(answer_key("c3-frame", first, times, []), [shrink(frame.rgb) for frame in video], first,
                      labels=G.image_labels(len(video), "Frame"), max_new_tokens=ANSWER_TOKENS["number"])
    number = S.parse_mark(pick["answer"], len(video))
    if number is None:
        return {**G.answer_record(G.Grounding(None, None), None, None, False, False, [pick["answer"]], pick["seconds"], len(video)),
                "frame_right": False}
    index = number - 1
    second = G.single_grounding_prompt(task)
    reply = reader.ask(answer_key("c3-box", second, [times[index]], []), [video[index].rgb], second, max_new_tokens=ANSWER_TOKENS["box"])
    grounding = G.parse_grounding(reply["answer"], 1)
    box = None if grounding.box is None else G.to_pixels(grounding.box, video[index].image_hw)
    hit, hit_old, shows = scored(box, labels[index], parts)
    record = G.answer_record(G.Grounding(number, grounding.box), box, times[index], hit, hit_old, [pick["answer"], reply["answer"]],
                             pick["seconds"] + reply["seconds"], len(video) + 1)
    return {**record, "frame_right": shows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    arms = [arm for arm in args.arms.split(",") if arm]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {ARMS}")
    result = json.loads((args.runs / f"{args.visit}.json").read_text())
    started = time.time()
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", args.visit, result["video"])
    steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
    labels = P.label_maps(scene, P.gt_frames(scene, steady))  # ground truth: scoring only
    tasks = [task for task in H.tasks_of(scene, labels) if task.keyframes]
    if [task.text for task in tasks] != result["tasks"]["texts"]:
        raise SystemExit(f"{args.visit}: the tasks differ from those of the phase-1 run")
    with np.load(args.runs / "diagnostics" / f"{args.visit}.npz") as saved:
        keyframe_vectors, queries = saved["keyframes"], saved["queries"]
        p4_vectors, p4_keyframes, p4_boxes = saved["vectors/P4"], saved["keyframe/P4"], saved["boxes/P4"]
    p4 = [policies.Candidate(int(k), tuple(int(v) for v in box), (0, 0, 0, 0)) for k, box in zip(p4_keyframes, p4_boxes)]
    frames = rgb_frames(scene, steady)
    out = args.runs / "controls"
    reader = H.CachedReader(out / f"answers_{args.visit}.json", VisionLanguageModel)
    record_file = out / f"{args.visit}.json"
    record = json.loads(record_file.read_text()) if record_file.exists() else {"visit": args.visit, "video": result["video"], "arms": {}}
    record.update({"model": f"{VLM_ID}@{VLM_REVISION}", "controls_version": CONTROLS_VERSION, "tasks": [task.text for task in tasks]})
    uniform = video = uniform_labels = None
    for arm in arms:
        rows = []
        for task, query in zip(tasks, queries):
            if arm == "p4n":
                rows.append(numbered_choice(reader, task.text, query, frames, steady, keyframe_vectors, p4, p4_vectors, labels, task.parts))
            elif arm in ("c1", "c2"):
                scores = (D.keyframe_scores(query, keyframe_vectors, np.zeros((0, keyframe_vectors.shape[1])), []) if arm == "c1"
                          else D.keyframe_scores(query, keyframe_vectors, p4_vectors, p4_keyframes))
                shown = [int(k) for k in np.argsort(-scores, kind="stable")[:TOP_KEYFRAMES]]
                rows.append(free_box(reader, arm, task.text, frames, shown, steady, labels, task.parts))
            else:
                if video is None:
                    uniform = keyframes.scene_keyframes(scene, "uniform", result["n_keyframes"])
                    video = rgb_frames(scene, uniform)
                    uniform_labels = P.label_maps(scene, P.gt_frames(scene, uniform))
                rows.append(raw_video(reader, task.text, video, uniform, uniform_labels, task.parts))
        record["arms"][arm] = rows
        hits = np.mean([row["hit"] for row in rows])
        print(f"{args.visit} {arm}: {hits:.0%} of {len(rows)} tasks right, {np.median([row['seconds'] for row in rows]):.1f} s per question",
              flush=True)
    record["total_s"] = round(time.time() - started, 1)
    out.mkdir(parents=True, exist_ok=True)
    record_file.write_text(json.dumps(record, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
