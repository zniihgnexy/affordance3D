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

The second batch (section 24.7):

    b      P3 as in phase 1 (generic regions as many and as large as P4's, seeds 0 to 4), read like p4n; stored per seed
    d      every generic region written into the memory, read like p4n (the 40 best-scoring regions of the top keyframes)
    cplus  no memory: our 40 keyframes, shrunk; the model names one, then draws the box in it at full size (as c3)

Round 9 (section 24.8), to measure what the memory itself is worth:

    c4     the annotated regions of c1's three keyframes (chosen without the memory), numbered; the model picks one
    c1s    c1's three keyframes read like c3 (shrunk: name one; then draw the box at full size)   [c1' in the draft]
    c2s    P4's three keyframes read the same way                                                 [c2' in the draft]

Round 9, question 2 (section 24.9): off-the-shelf tools build the map. Their boxes come from scripts/run_tool_maps.py
(runs/mem_phase1/tool_maps/<tool>/<visit>.json; <tool> is sam3 or qwen8b) and are named on the command line explicitly:

    p4t_<tool>   P4 with the tool's boxes as the regions: encoded, ranked and read as p4n (every action the same type)
    c4t_<tool>   c4 with the tool's boxes: c1's three keyframes, their boxes numbered in keyframe order (at most 40)
    p4tp_<tool>  diagnostic: p4t with only the boxes whose centre lies on an annotated part (widened by 8 pixels)
    sam3q        no memory: SAM 3 prompted with the task text on c3's 40 even frames (tool_maps/sam3_tasks/<visit>.json);
                 the three frames with its best boxes, its best boxes there (at most 40) numbered; the model picks one

    python scripts/run_mem_controls.py --visit 421254 --model 8b --arms p4t_sam3,p4t_qwen8b,c4t_sam3,p4tp_sam3,sam3q

The crops around a tool's boxes are encoded once and kept in runs/mem_phase1/tool_maps/<tool>/features_<visit>.npz.

--model 8b runs Qwen3-VL-8B instead of the 4B model; its records and answers go to runs/mem_phase1/controls_8b/.

b and d need the vectors of every generic region; they are encoded once per scene, together with every grid level (for
control a, scored on the CPU by scripts/score_mem_control_a.py), and saved in runs/mem_phase1/controls/features_<visit>.npz.

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
from affordance3d.mem import tool_maps as TM
from affordance3d.mem.encoders import VLMS, VisionLanguageModel
from report import mem_phase1_helpers as R
from report import mem_steps_helpers as H

ARMS = ("p4n", "c1", "c2", "c3", "b", "d", "cplus", "c4", "c1s", "c2s")  # the default of --arms
TOOL_ARMS = (*(f"{kind}_{tool}" for kind in ("p4t", "c4t", "p4tp") for tool in TM.TOOLS), "sam3q")  # named explicitly
OUT_FOLDERS = {"4b": "controls", "8b": "controls_8b"}
P3_SEEDS = (0, 1, 2, 3, 4)
CONTROLS_VERSION = "controls-v3"  # v2: every prompt defines "the part"; v3: longer answers (pilot, 2026-11-13)  # part of the answer cache key: change it when a prompt or an image changes
TOP_KEYFRAMES, MAX_MARKS, SHRINK = 3, 40, 2
ANSWER_TOKENS = {"number": 8, "box": 160}  # some answers explain first, then give the box


def answer_key(arm: str, prompt: str, frames: list[str], boxes: list, revision: str = VLMS["4b"][1]) -> str:
    """Cache key of one question; each model keeps its answers in its own folder, so the 4B revision stays in the key."""
    payload = json.dumps([CONTROLS_VERSION, H.MARKS_VERSION, revision, arm, prompt, frames, boxes])
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
                    candidates: list[policies.Candidate], vectors: np.ndarray, labels: list[np.ndarray], parts: tuple[int, ...],
                    arm: str = "p4n") -> dict:
    """p4n, b, d: as the phase-1 reading step (the memory's three best keyframes, its best regions there numbered), with the
    neutral prompt; `candidates` and `vectors` are what the memory holds."""
    scores = S.score_keyframes(query, keyframe_vectors, vectors, [c.keyframe for c in candidates], [0] * len(candidates))
    top = [int(k) for k in scores.ranking()[:TOP_KEYFRAMES]]
    inside = sorted((i for i, c in enumerate(candidates) if c.keyframe in top), key=lambda i: -scores.token_scores[i])[:MAX_MARKS]
    return numbered_reading(reader, arm, task, frames, times, top, [candidates[i] for i in sorted(inside)], labels, parts)


def numbered_reading(reader: H.CachedReader, arm: str, task: str, frames: list, times: list[str], top: list[int],
                     kept: list[policies.Candidate], labels: list[np.ndarray], parts: tuple[int, ...]) -> dict:
    """The model sees the keyframes `top` with the regions `kept` numbered on them and picks one (neutral prompt)."""
    images, marks = H.mark_boxes(frames, [(c.keyframe, c.box) for c in kept], top)
    if not marks:
        return {**G.answer_record(G.Grounding(None, None), None, None, False, False, [], 0.0, len(top)), "frame_right": False,
                "keyframes_shown": [times[k] for k in top], "marks": 0}
    prompt = G.numbered_prompt(task, len(marks))
    reply = reader.ask(answer_key(arm, prompt, [times[k] for k in top], [list(kept[i].box) for i in marks]), images, prompt,
                       max_new_tokens=ANSWER_TOKENS["number"])
    mark = S.parse_mark(reply["answer"], len(marks))
    chosen = None if mark is None else kept[marks[mark - 1]]
    box = None if chosen is None else chosen.box
    hit, hit_old, shows = scored(box, labels[chosen.keyframe], parts) if chosen is not None else (False, False, False)
    record = G.answer_record(G.Grounding(mark, None), box, None if chosen is None else times[chosen.keyframe], hit, hit_old,
                             [reply["answer"]], reply["seconds"], len(top))
    return {**record, "frame_right": shows, "keyframes_shown": [times[k] for k in top], "marks": len(marks)}


def raw_video(reader: H.CachedReader, task: str, video: list, times: list[str], labels: list[np.ndarray], parts: tuple[int, ...],
              arm: str = "c3") -> dict:
    """c3, cplus, c1s, c2s: the model sees the frames (shrunk), names one, then draws the box in it at full size."""
    first = G.frame_prompt(task, len(video))
    pick = reader.ask(answer_key(f"{arm}-frame", first, times, []), [shrink(frame.rgb) for frame in video], first,
                      labels=G.image_labels(len(video), "Frame"), max_new_tokens=ANSWER_TOKENS["number"])
    number = S.parse_mark(pick["answer"], len(video))
    if number is None:
        return {**G.answer_record(G.Grounding(None, None), None, None, False, False, [pick["answer"]], pick["seconds"], len(video)),
                "frame_right": False}
    index = number - 1
    second = G.single_grounding_prompt(task)
    reply = reader.ask(answer_key(f"{arm}-box", second, [times[index]], []), [video[index].rgb], second, max_new_tokens=ANSWER_TOKENS["box"])
    grounding = G.parse_grounding(reply["answer"], 1)
    box = None if grounding.box is None else G.to_pixels(grounding.box, video[index].image_hw)
    hit, hit_old, shows = scored(box, labels[index], parts)
    record = G.answer_record(G.Grounding(number, grounding.box), box, times[index], hit, hit_old, [pick["answer"], reply["answer"]],
                             pick["seconds"] + reply["seconds"], len(video) + 1)
    return {**record, "frame_right": shows}


def best_keyframes(query: np.ndarray, keyframe_vectors: np.ndarray, vectors: np.ndarray | None = None,
                   token_keyframes: np.ndarray | None = None) -> list[int]:
    """The three best keyframes: by whole-keyframe meaning alone (no memory), or with the memory's tokens as in phase 1."""
    if vectors is None:
        vectors, token_keyframes = np.zeros((0, keyframe_vectors.shape[1])), np.zeros(0, dtype=np.int64)
    scores = D.keyframe_scores(query, keyframe_vectors, vectors, token_keyframes)
    return [int(k) for k in np.argsort(-scores, kind="stable")[:TOP_KEYFRAMES]]


def annotated_on(shown: list[int], regions: list[policies.Candidate]) -> list[policies.Candidate]:
    """c4: the annotated regions that lie on the keyframes `shown`, in keyframe order, at most MAX_MARKS (no ranking:
    ranking them would use the memory)."""
    return [c for c in regions if c.keyframe in set(shown)][:MAX_MARKS]


def all_grids(n_keyframes: int, image_hw: tuple[int, int]) -> dict[str, list[policies.Candidate]]:
    """Every grid level of phase 1, each cut the way policies.grid_shape orients it (long side of the image, more cells)."""
    upright = image_hw[0] >= image_hw[1]
    return {f"{rows}x{columns}": policies.grid_candidates(n_keyframes, image_hw, (rows, columns) if upright else (columns, rows))
            for rows, columns in policies.GRID_SHAPES}


def p3_indices(p4: list[policies.Candidate], pool: list[policies.Candidate], n_keyframes: int, image_hw: tuple[int, int],
               seed: int) -> list[int]:
    """Indices into `pool` of P3 for `seed`, drawn with the random numbers of the phase-1 runner (P2 first, then P3)."""
    rng = np.random.default_rng(R.RUNNER_SEED + seed)
    policies.random_candidates(p4, len(p4), n_keyframes, image_hw, rng)
    position = {id(candidate): i for i, candidate in enumerate(pool)}
    return [position[id(candidate)] for candidate in policies.size_matched(pool, p4, len(p4), rng)]


def generic_features(file: Path, frames: list, pool: list[policies.Candidate]) -> dict[str, np.ndarray]:
    """Vectors of every generic region and every grid cell of one scene, encoded once and kept in `file`."""
    if file.exists():
        with np.load(file) as saved:
            return {key: saved[key] for key in saved.files}
    from affordance3d.mem.encoders import ImageTextEncoder

    encoder = ImageTextEncoder()
    arrays = {"pool/vectors": encoder.images(R.crops_of(frames, pool)),
              "pool/keyframe": np.array([c.keyframe for c in pool], dtype=np.int64),
              "pool/boxes": np.array([c.box for c in pool], dtype=np.int64).reshape(len(pool), 4)}
    for name, cells in all_grids(len(frames), frames[0].image_hw).items():
        arrays[f"grid {name}/vectors"] = encoder.images(R.crops_of(frames, cells))
        arrays[f"grid {name}/keyframe"] = np.array([c.keyframe for c in cells], dtype=np.int64)
    del encoder
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file.with_suffix(".tmp.npz"), **arrays)
    file.with_suffix(".tmp.npz").rename(file)
    return arrays


def tool_features(file: Path, frames: list, candidates: list[policies.Candidate], width: int) -> np.ndarray:
    """Vectors of the crops around a tool's boxes, encoded as generic_features encodes regions, once, and kept in `file`
    with the boxes they belong to and the seconds the encoding took."""
    boxes = np.array([c.box for c in candidates], dtype=np.int64).reshape(len(candidates), 4)
    keyframe = np.array([c.keyframe for c in candidates], dtype=np.int64)
    if file.exists():
        with np.load(file) as saved:
            if not (np.array_equal(saved["boxes"], boxes) and np.array_equal(saved["keyframe"], keyframe)):
                raise SystemExit(f"{file} was encoded from other boxes than the tool map holds now; move it away")
            return saved["vectors"]
    from affordance3d.mem.encoders import ImageTextEncoder

    encoder = ImageTextEncoder()
    start = time.time()
    vectors = encoder.images(R.crops_of(frames, candidates)) if candidates else np.zeros((0, width), dtype=np.float32)
    seconds = time.time() - start
    del encoder
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file.with_suffix(".tmp.npz"), vectors=vectors, keyframe=keyframe, boxes=boxes, seconds=seconds)
    file.with_suffix(".tmp.npz").rename(file)
    return vectors


def tool_memory(runs: Path, tool: str, visit: str, frames: list, labels: list[np.ndarray], width: int) -> dict:
    """A tool's map as the memory holds it: its boxes as regions, their vectors, and which of them lie on an annotated
    part (for the diagnostic arm only)."""
    record = TM.read_tool_map(runs / "tool_maps" / tool / f"{visit}.json")
    missing = TM.missing_keyframes(record, len(frames))
    if missing:
        raise SystemExit(f"{visit}: the {tool} map lacks keyframes {missing}; run scripts/run_tool_maps.py to the end first")
    candidates = TM.candidates_of(TM.tool_detections(record), frames[0].image_hw)
    vectors = tool_features(runs / "tool_maps" / tool / f"features_{visit}.npz", frames, candidates, width)
    return {"candidates": candidates, "vectors": vectors, "on_part": TM.on_a_part(candidates, labels)}


def tool_arm(reader: H.CachedReader, arm: str, task: H.Task, query: np.ndarray, frames: list, times: list[str],
             keyframe_vectors: np.ndarray, memory: dict, labels: list[np.ndarray]) -> dict:
    """p4t_<tool>, c4t_<tool>, p4tp_<tool>: the tool's boxes in the place of the annotated regions."""
    kind = arm.split("_", 1)[0]
    candidates, vectors = memory["candidates"], memory["vectors"]
    if kind == "c4t":
        shown = best_keyframes(query, keyframe_vectors)
        return numbered_reading(reader, arm, task.text, frames, times, shown, annotated_on(shown, candidates), labels, task.parts)
    if kind == "p4tp":
        candidates, vectors = [candidates[i] for i in memory["on_part"]], vectors[memory["on_part"]]
    return numbered_choice(reader, task.text, query, frames, times, keyframe_vectors, candidates, vectors, labels, task.parts, arm)


def task_prompts(runs: Path, visit: str, texts: list[str], uniform: list[str]) -> dict:
    """What SAM 3 found for every task's text on the even frames (scripts/run_tool_maps.py --tool sam3_tasks)."""
    record = TM.read_tool_map(runs / "tool_maps" / TM.TASK_PROMPTED / f"{visit}.json")
    if record["tasks"] != texts or record["times"] != uniform or len(record["frames"]) != len(uniform):
        raise SystemExit(f"{visit}: the {TM.TASK_PROMPTED} file holds other tasks or frames, or is not finished")
    return record


def task_prompted(reader: H.CachedReader, record: dict, index: int, task: H.Task, video: list, times: list[str],
                  labels: list[np.ndarray]) -> dict:
    """sam3q: SAM 3's best boxes for the task text, on the even frames that hold its best boxes, numbered."""
    top, marks = TM.task_prompt_marks(TM.task_detections(record, index))
    kept = [policies.Candidate(d.keyframe, d.box, (0, 0, 0, 0)) for d in marks]
    return numbered_reading(reader, "sam3q", task.text, video, times, top, kept, labels, task.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--model", choices=sorted(VLMS), default="4b")
    args = parser.parse_args()
    model_id, revision = VLMS[args.model]
    arms = [arm for arm in args.arms.split(",") if arm]
    unknown = set(arms) - set(ARMS) - set(TOOL_ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {ARMS + TOOL_ARMS}")
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
        saved_p3 = {seed: (saved[f"boxes/P3 s{seed}"], saved[f"vectors/P3 s{seed}"]) for seed in P3_SEEDS if f"boxes/P3 s{seed}" in saved.files}
    p4 = [policies.Candidate(int(k), tuple(int(v) for v in box), (0, 0, 0, 0)) for k, box in zip(p4_keyframes, p4_boxes)]
    frames = rgb_frames(scene, steady)
    out = args.runs / OUT_FOLDERS[args.model]
    reader = H.CachedReader(out / f"answers_{args.visit}.json", lambda: VisionLanguageModel(model_id=model_id, revision=revision))
    record_file = out / f"{args.visit}.json"
    record = json.loads(record_file.read_text()) if record_file.exists() else {"visit": args.visit, "video": result["video"], "arms": {}}
    pool = generic = None
    if {"b", "d"} & set(arms):
        pool = R.generic_pool(args.runs / "regions" / f"{args.visit}.json", frames[0].image_hw)
        generic = generic_features(args.runs / "controls" / f"features_{args.visit}.npz", frames, pool)
        draws = {seed: p3_indices(p4, pool, len(frames), frames[0].image_hw, seed) for seed in P3_SEEDS}
        agreement = []
        for seed, (boxes, vectors) in saved_p3.items():
            if not np.array_equal(generic["pool/boxes"][draws[seed]], boxes):
                raise SystemExit(f"{args.visit}: P3 seed {seed} differs from the draw saved by the diagnostics")
            agreement.append(float(np.min(np.sum(generic["pool/vectors"][draws[seed]] * vectors, axis=1))))
        record["encoder_agreement"] = min(agreement) if agreement else None  # lowest cosine, this run's vectors against phase 1's
    record.update({"model": f"{model_id}@{revision}", "controls_version": CONTROLS_VERSION, "tasks": [task.text for task in tasks]})
    uniform = video = uniform_labels = prompted = None
    memories: dict[str, dict] = {}  # the tool maps, loaded when an arm needs one
    for arm in arms:
        if arm == "b":
            for seed in P3_SEEDS:
                chosen = [pool[i] for i in draws[seed]]
                record["arms"][f"b s{seed}"] = [numbered_choice(reader, task.text, query, frames, steady, keyframe_vectors, chosen,
                                                                generic["pool/vectors"][draws[seed]], labels, task.parts, f"b s{seed}")
                                                for task, query in zip(tasks, queries)]
            rows = [row for seed in P3_SEEDS for row in record["arms"][f"b s{seed}"]]
            print(f"{args.visit} b: {np.mean([row['hit'] for row in rows]):.0%} of {len(tasks)} tasks right over {len(P3_SEEDS)} seeds",
                  flush=True)
            continue
        rows = []
        for index, (task, query) in enumerate(zip(tasks, queries)):
            if arm in TOOL_ARMS and arm != "sam3q":
                tool = arm.split("_", 1)[1]
                if tool not in memories:
                    memories[tool] = tool_memory(args.runs, tool, args.visit, frames, labels, keyframe_vectors.shape[1])
                rows.append(tool_arm(reader, arm, task, query, frames, steady, keyframe_vectors, memories[tool], labels))
            elif arm == "c4":
                shown = best_keyframes(query, keyframe_vectors)
                rows.append(numbered_reading(reader, "c4", task.text, frames, steady, shown, annotated_on(shown, p4), labels, task.parts))
            elif arm in ("c1s", "c2s"):
                shown = best_keyframes(query, keyframe_vectors) if arm == "c1s" else best_keyframes(query, keyframe_vectors, p4_vectors, p4_keyframes)
                rows.append(raw_video(reader, task.text, [frames[k] for k in shown], [steady[k] for k in shown],
                                      [labels[k] for k in shown], task.parts, arm))
            elif arm == "d":
                rows.append(numbered_choice(reader, task.text, query, frames, steady, keyframe_vectors, pool, generic["pool/vectors"],
                                            labels, task.parts, "d"))
            elif arm == "cplus":
                rows.append(raw_video(reader, task.text, frames, steady, labels, task.parts, "cplus"))
            elif arm == "p4n":
                rows.append(numbered_choice(reader, task.text, query, frames, steady, keyframe_vectors, p4, p4_vectors, labels, task.parts))
            elif arm in ("c1", "c2"):
                shown = best_keyframes(query, keyframe_vectors) if arm == "c1" else best_keyframes(query, keyframe_vectors, p4_vectors, p4_keyframes)
                rows.append(free_box(reader, arm, task.text, frames, shown, steady, labels, task.parts))
            else:
                if video is None:
                    uniform = keyframes.scene_keyframes(scene, "uniform", result["n_keyframes"])
                    video = rgb_frames(scene, uniform)
                    uniform_labels = P.label_maps(scene, P.gt_frames(scene, uniform))
                if arm == "sam3q":
                    prompted = prompted or task_prompts(args.runs, args.visit, [t.text for t in tasks], uniform)
                    rows.append(task_prompted(reader, prompted, index, task, video, uniform, uniform_labels))
                else:
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
