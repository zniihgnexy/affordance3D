"""C-1 (discussion draft, section 25.5): beyond where the part is, can the memory say how to act on it? One scene per call.

    python scripts/run_mem_c1.py --visit 421254                    # every arm with Qwen3-VL-4B (GPU: through the queue)
    python scripts/run_mem_c1.py --visit 421254 --model 8b         # V and V' again with Qwen3-VL-8B (M' keeps the 4B choice)
    python scripts/run_mem_c1.py --visit 421254 --no-vlm --out runs/mem_c1/preview   # CPU only: M and O

Tasks: those of C-0 whose every target part has a motion (runs/mem_c0/records.json, 380 over the 30 scenes). Each arm
answers in one frame, and is scored in that frame (affordance3d.mem.action_arms.score_arm):

    V   Qwen3-VL sees the frame c3 chose (controls, 4B) and the task; JSON: point, label, push/pull/rotate, arrow end, hinge side
    G   V's point and label; the direction from the depth map's normal at the point, by a rule on the label
    M   the frame and region p4n chose (controls, 4B): region centre, the region's annotated label, G's rule (ORACLE)
    O   the ground truth in the C-0 keyframe (the scorer must give it 100%)
    V', G', M'  the same in the C-0 keyframe (the steady keyframe where the target parts show most), which takes the
                choice of the frame out; M' lets the 4B model pick one of P4's numbered regions of that keyframe, as p4n does

Depth: monocular depth rescaled by the device's LiDAR, cached as in phase 1 (runs/mem_phase1/mono/<video>/); the frames
c3 chose are mostly not cached yet, so a V run computes them first (the depth model then needs the GPU as well).
Writes runs/mem_c1/<model>/<visit>.json; the model's answers are cached in runs/mem_c1/<model>/answers_<visit>.json.
Ground truth (part ids, motions, laser scan, registered cameras) is used for scoring only.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the log)

from affordance3d.frames import Frame  # noqa: E402
from affordance3d.mem import action_arms as C  # noqa: E402
from affordance3d.mem import action_records as A  # noqa: E402
from affordance3d.mem import grounding as G  # noqa: E402
from affordance3d.mem import keyframes  # noqa: E402
from affordance3d.mem import pipeline as P  # noqa: E402
from affordance3d.mem import search as S  # noqa: E402
from affordance3d.mem.encoders import VLMS, VisionLanguageModel  # noqa: E402
from report import mem_steps_helpers as H  # noqa: E402
from report.mem_demo_helpers import scaled_mono_depths  # noqa: E402

MONO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"  # as phase 1 (scripts/run_mem_phase1.py)
C1_VERSION = "c1-v1"  # part of the answer cache key: change it when a prompt or an image changes
ANSWER_TOKENS = {"json": 256, "number": 8}
MAX_MARKS = 40  # as p4n
VLM_ARMS = ("V", "G", "V'", "G'", "M'")
ANSWERS_HINGE = {"V": True, "V'": True, "O": True}  # G and M (and their primes) give no hinge side


@dataclass(frozen=True)
class FrameSet:
    inputs: Frame  # RGB, monocular depth x LiDAR, device pose: what the arms see
    gt: Frame  # registered camera, rendered depth: scoring only
    labels: np.ndarray  # part ids: scoring only


def answer_key(arm: str, prompt: str, frames: list[str], boxes: list, revision: str) -> str:
    payload = json.dumps([C1_VERSION, H.MARKS_VERSION, revision, arm, prompt, frames, boxes])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def c1_tasks(c0_rows: list[dict], visit: str, control_tasks: list[str]) -> list[tuple[int, dict]]:
    """(position among the scene's answerable tasks, C-0 row) of every task whose parts all have a motion."""
    rows = [row for row in c0_rows if row["visit"] == visit]
    if [row["task"] for row in rows] != control_tasks:
        raise SystemExit(f"{visit}: the C-0 tasks differ from those of the controls")
    return [(i, row) for i, row in enumerate(rows) if row["has_all_motions"]]


def frame_sets(scene: P.Scene, times: set[str], cache_dir: Path) -> dict[str, FrameSet]:
    ordered = sorted(times, key=float)
    depths = scaled_mono_depths(scene, ordered, MONO_ID, cache_dir)
    gts = P.gt_frames(scene, ordered)
    labels = P.label_maps(scene, gts)
    return {t: FrameSet(P.input_frame(scene, t, d), g, lab) for t, d, g, lab in zip(ordered, depths, gts, labels, strict=True)}


def truth_in(row: dict, motions: dict[int, A.PartMotion], frames: FrameSet) -> A.ActionRecord:
    """The ground truth of one task as seen in one frame (motions in the documentation's reading)."""
    parts = [p["part"] for p in row["parts"]]
    return A.record_in(row["task"], parts, [p["annot_id"] for p in row["parts"]], [p["label"] for p in row["parts"]],
                       [motions[p] for p in parts], frames.labels, frames.gt, shown_if_hidden=row["shown"])


def prediction_row(prediction: A.Prediction | None) -> dict | None:
    if prediction is None:
        return None
    return {"point": prediction.point, "label": prediction.label, "type": prediction.kind, "sense": prediction.sense,
            "arrow": prediction.arrow, "hinge_side": prediction.hinge_side,
            "direction": None if prediction.direction is None else [round(float(v), 5) for v in prediction.direction]}


def arm_row(arm: str, frame: str | None, prediction: A.Prediction | None, row: dict, motions: dict, sets: dict[str, FrameSet],
            up: np.ndarray, replies: list[dict], **extra: object) -> dict:
    """Score one arm's answer in its frame; without a frame, in the C-0 keyframe (and every applicable field is wrong)."""
    where = sets[frame or row["keyframe"]]
    record = truth_in(row, motions, where)
    score = C.score_arm(prediction if frame else None, record, where.labels, where.gt, up, ANSWERS_HINGE.get(arm.rstrip("'"), False))
    return {"frame": frame, "answers": [r["answer"] for r in replies], "seconds": round(sum(r["seconds"] for r in replies), 2),
            "prediction": prediction_row(prediction), "score": score, **extra}


def ask_action(reader: H.CachedReader, revision: str, task: str, t: str, sets: dict[str, FrameSet]) -> tuple[dict, A.Prediction]:
    """V (or V'): the model sees one frame at the working resolution and answers in JSON."""
    prompt = C.action_prompt(task)
    reply = reader.ask(answer_key("action", prompt, [t], [], revision), [sets[t].inputs.rgb], prompt, max_new_tokens=ANSWER_TOKENS["json"])
    return reply, C.answer_prediction(C.parse_action_answer(reply["answer"]), sets[t].inputs)


def memory_prediction(box: list[int], t: str, row: dict, sets: dict[str, FrameSet], actions: list[int],
                      up: np.ndarray) -> tuple[A.Prediction, bool]:
    """M (or M'): the region's centre and its annotated label (ORACLE), the motion by G's rule."""
    label, exact = C.region_label(sets[t].labels, actions, sets[t].inputs.depth_m, tuple(box))
    return C.rule_prediction(G.box_centre(tuple(box)), label, sets[t].inputs, up), exact


def pick_region(reader: H.CachedReader, revision: str, task: str, t: str, boxes: list[tuple[int, ...]],
                sets: dict[str, FrameSet]) -> tuple[dict | None, tuple[int, ...] | None]:
    """M': the model picks one of P4's numbered regions of one keyframe (the p4n question)."""
    kept = boxes[:MAX_MARKS]
    images, marks = H.mark_boxes([sets[t].inputs], [(0, box) for box in kept], [0])
    if not marks:
        return None, None
    prompt = G.numbered_prompt(task, len(marks))
    reply = reader.ask(answer_key("M'", prompt, [t], [list(kept[i]) for i in marks], revision), images, prompt,
                       max_new_tokens=ANSWER_TOKENS["number"])
    mark = S.parse_mark(reply["answer"], len(marks))
    return reply, None if mark is None else kept[marks[mark - 1]]


def readers(out: Path, visit: str, model: str) -> tuple[H.CachedReader, H.CachedReader]:
    """The reader for V and V' (this model) and the one for M' (always the 4B model; an 8B run reuses its answers)."""
    model_id, revision = VLMS[model]
    own = H.CachedReader(out / model / f"answers_{visit}.json", lambda: VisionLanguageModel(model_id=model_id, revision=revision))
    if model == "4b":
        return own, own

    def no_model() -> None:
        raise SystemExit(f"{visit}: M' needs the 4B answers; run --model 4b first")

    return own, H.CachedReader(out / "4b" / f"answers_{visit}.json", no_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--model", choices=sorted(VLMS), default="4b")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1", help="phase-1 results and controls")
    parser.add_argument("--c0", type=Path, default=REPO_ROOT / "runs" / "mem_c0" / "records.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "mem_c1")
    parser.add_argument("--no-vlm", action="store_true", help="only the arms without a model (M, O): runs on the CPU")
    args = parser.parse_args()
    started = time.time()
    result = json.loads((args.runs / f"{args.visit}.json").read_text())
    controls = json.loads((args.runs / "controls" / f"{args.visit}.json").read_text())  # 4B: c3's and p4n's choices
    tasks = c1_tasks(json.loads(args.c0.read_text()), args.visit, controls["tasks"])
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", args.visit, result["video"])
    steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
    c3, p4n = controls["arms"]["c3"], controls["arms"]["p4n"]
    times = {row["keyframe"] for _, row in tasks} | {p4n[i]["frame"] for i, _ in tasks if p4n[i]["frame"]}
    if not args.no_vlm:
        times |= {c3[i]["frame"] for i, _ in tasks if c3[i]["frame"]}
    sets = frame_sets(scene, times, args.runs / "mono" / result["video"])
    motions = {p: A.part_motion(scene.motions[e.annot_id], scene.part_points(p), scene.scan_xyz, reading="docs")
               for p, e in enumerate(scene.elements) if e.annot_id in scene.motions}
    actions = C.part_actions([e.label for e in scene.elements])
    with np.load(args.runs / "diagnostics" / f"{args.visit}.npz") as saved:
        p4_keyframe, p4_boxes = saved["keyframe/P4"], saved["boxes/P4"]
    reader, reader_4b = (None, None) if args.no_vlm else readers(args.out, args.visit, args.model)
    revision = VLMS[args.model][1]
    arms: dict[str, list[dict]] = {}

    def add(arm: str, entry: dict) -> None:
        arms.setdefault(arm, []).append(entry)

    for i, row in tasks:
        k = row["keyframe"]
        truth = truth_in(row, motions, sets[k])
        add("O", arm_row("O", k, A.truth_as_prediction(truth), row, motions, sets, scene.up, []))
        m_frame = p4n[i]["frame"]
        m_pred, m_exact = memory_prediction(p4n[i]["box"], m_frame, row, sets, actions, scene.up) if m_frame else (None, None)
        add("M", arm_row("M", m_frame, m_pred, row, motions, sets, scene.up, [], region_exact=m_exact))
        if args.no_vlm:
            continue
        for arm, frame in (("V", c3[i]["frame"]), ("V'", k)):
            reply, v_pred = ask_action(reader, revision, row["task"], frame, sets) if frame else ({"answer": "", "seconds": 0.0}, None)
            add(arm, arm_row(arm, frame, v_pred, row, motions, sets, scene.up, [reply]))
            g_pred = None if v_pred is None else C.rule_prediction(v_pred.point, v_pred.label, sets[frame].inputs, scene.up)
            add("G" + arm[1:], arm_row("G" + arm[1:], frame, g_pred, row, motions, sets, scene.up, []))
        boxes = [tuple(int(v) for v in box) for box, kf in zip(p4_boxes, p4_keyframe) if steady[int(kf)] == k]
        reply, box = pick_region(reader_4b, VLMS["4b"][1], row["task"], k, boxes, sets)
        mp_pred, mp_exact = memory_prediction(list(box), k, row, sets, actions, scene.up) if box else (None, None)
        add("M'", arm_row("M'", k if box else None, mp_pred, row, motions, sets, scene.up, [reply] if reply else [],
                          region_exact=mp_exact, marks=len(boxes[:MAX_MARKS])))
    record = {"visit": args.visit, "video": result["video"], "model": None if args.no_vlm else "@".join(VLMS[args.model]),
              "c1_version": C1_VERSION, "tasks": [row["task"] for _, row in tasks], "arms": arms,
              "total_s": round(time.time() - started, 1)}
    folder = args.out if args.no_vlm else args.out / args.model
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{args.visit}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    joint = {arm: np.mean([entry["score"]["joint"] for entry in rows]) for arm, rows in arms.items()}
    print(f"{args.visit}: {len(tasks)} tasks; joint success " + ", ".join(f"{arm} {share:.0%}" for arm, share in joint.items())
          + f" ({record['total_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
