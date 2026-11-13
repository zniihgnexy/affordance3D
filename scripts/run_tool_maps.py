"""Off-the-shelf tools build the affordance map of one scene (discussion draft, section 24.9): their boxes, keyframe by keyframe.

    python scripts/run_tool_maps.py --visit 421254 --tool sam3          # SAM 3, prompted once with each of the eight part names
    python scripts/run_tool_maps.py --visit 421254 --tool qwen8b        # Qwen3-VL-8B, asked once per keyframe for JSON boxes
    python scripts/run_tool_maps.py --visit 421254 --tool sam3 --gate   # only the two keyframes shown to the lead at the gate
    python scripts/run_tool_maps.py --visit 421254 --tool sam3_tasks    # SAM 3 prompted with each task's text on c3's 40
                                                                        # even frames (for arm sam3q of run_mem_controls.py)

The tools see the 40 steady keyframes of phase 1 (at the working resolution, as every arm does) and the part names only:
no annotation, no task text (sam3_tasks excepted: its prompt is the task). Writes runs/mem_phase1/tool_maps/<tool>/<visit>.json
after every keyframe, so a stopped run goes on where it stopped; Qwen's answers are also cached in answers_<visit>.json.
Each keyframe stores its boxes (pixels), scores, labels and the seconds the tool took. Run on the GPU through the queue
(the job sets AFFORDANCE3D_DEVICE=cuda). --gate chooses its keyframes with the projected annotations
(tool_maps.gate_keyframes): ground truth chooses what the lead is shown and is never given to a tool.
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the log)

from affordance3d.device import default_device
from affordance3d.mem import keyframes
from affordance3d.mem import pipeline as P
from affordance3d.mem import tool_maps as TM
from affordance3d.mem.encoders import VisionLanguageModel, model_source

CHOICES = (*TM.TOOLS, TM.TASK_PROMPTED)
TEXT_TOKENS = 32  # SAM 3's text encoder reads at most 32 tokens; a longer task text is cut
MASK_THRESHOLD = 0.5  # as SAM 3's post-processing
QWEN_TOKENS = 1024  # about 30 boxes; an answer cut off at this length keeps the boxes it completed


class Sam3Detector:
    """SAM 3 (transformers Sam3Model): text-prompted instances; one image encoding serves every prompt of a frame."""

    def __init__(self, device: str | None = None) -> None:
        from transformers import Sam3Model, Sam3Processor

        source = model_source(*TM.SAM3)
        self.device = device or default_device()
        self._processor = Sam3Processor.from_pretrained(**source)
        self._model = Sam3Model.from_pretrained(**source).eval().to(self.device)

    def tokens(self, prompt: str) -> int:
        """Length of a prompt in the text encoder's tokens (more than TEXT_TOKENS: the prompt is cut)."""
        return len(self._processor.tokenizer(prompt)["input_ids"])

    def detect(self, rgb: np.ndarray, prompts: list[str], threshold: float | None = TM.SCORE_THRESHOLD,
               keep: int | None = None) -> list[list[dict]]:
        """For each prompt, its instances best first: those scoring above `threshold`, or (threshold None) the `keep` best."""
        import torch
        from PIL import Image

        inputs = self._processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        found = []
        with torch.inference_mode():
            vision = self._model.get_vision_features(pixel_values=inputs.pixel_values)
            for prompt in prompts:
                text = self._processor.tokenizer(prompt, padding="max_length", max_length=TEXT_TOKENS, truncation=True,
                                                 return_tensors="pt").to(self.device)
                outputs = self._model(vision_embeds=vision, input_ids=text.input_ids, attention_mask=text.attention_mask)
                found.append(instances(outputs, rgb.shape[:2], threshold, keep))
        return found


def chosen_queries(scores: np.ndarray, threshold: float | None, keep: int | None) -> np.ndarray:
    """Indices of the instances kept, best first: above `threshold`, or the `keep` best when there is no threshold."""
    order = np.argsort(-scores, kind="stable")
    if threshold is not None:
        return order[scores[order] > threshold]
    return order[:keep]


def pixel_box(mask: np.ndarray, predicted: np.ndarray, image_hw: tuple[int, int]) -> TM.Box:
    """The box of an instance: around its mask; the predicted box (0-1 corners, scaled to pixels) when the mask is empty."""
    box = TM.mask_box(mask)
    if box is not None:
        return box
    height, width = image_hw
    left, top = int(np.floor(predicted[0] * width)), int(np.floor(predicted[1] * height))
    right, bottom = int(np.ceil(predicted[2] * width)), int(np.ceil(predicted[3] * height))
    left, top = min(max(left, 0), width - 1), min(max(top, 0), height - 1)
    return left, top, min(max(right, left + 1), width), min(max(bottom, top + 1), height)


def instances(outputs: object, image_hw: tuple[int, int], threshold: float | None, keep: int | None) -> list[dict]:
    """SAM 3's output for one prompt: scores as in its post-processing (query score times presence score), masks
    upsampled to the image and cut at MASK_THRESHOLD, each turned into a box."""
    import torch

    scores = (outputs.pred_logits.sigmoid() * outputs.presence_logits.sigmoid())[0].float().cpu().numpy()
    chosen = chosen_queries(scores, threshold, keep)
    if not len(chosen):
        return []
    index = torch.as_tensor(chosen, device=outputs.pred_masks.device)
    masks = torch.nn.functional.interpolate(outputs.pred_masks[0, index][None].sigmoid().float(), size=tuple(image_hw),
                                            mode="bilinear", align_corners=False)[0] > MASK_THRESHOLD
    masks, boxes = masks.cpu().numpy(), outputs.pred_boxes[0, index].float().cpu().numpy()
    return [{"box": list(pixel_box(mask, box, image_hw)), "score": round(float(scores[q]), 4)}
            for mask, box, q in zip(masks, boxes, chosen)]


def answer_key(prompt: str, time_stamp: str, revision: str) -> str:
    payload = json.dumps([TM.VERSION, revision, prompt, time_stamp])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save(record: dict, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    temporary.rename(file)


def start_record(file: Path, fresh: dict) -> dict:
    """The record on disk when it was made the same way (same tool, model, version, frames), else `fresh`."""
    if not file.exists():
        return fresh
    record = json.loads(file.read_text())
    same = ("tool", "model", "version", "times", "image_hw")
    if any(record.get(key) != fresh[key] for key in same):
        raise SystemExit(f"{file} was made differently ({', '.join(k for k in same if record.get(k) != fresh[k])}); "
                         "move it away to start again")
    return record


def sam3_entry(detector: Sam3Detector, rgb: np.ndarray) -> dict:
    start = time.time()
    found = detector.detect(rgb, list(TM.PART_NAMES))
    seconds = time.time() - start
    detections = [{**instance, "label": name} for name, per_prompt in zip(TM.PART_NAMES, found) for instance in per_prompt]
    return {"seconds": round(seconds, 2), "detections": detections}


def qwen_entry(reader: object, rgb: np.ndarray, time_stamp: str, revision: str) -> dict:
    prompt = TM.detection_prompt()
    reply = reader.ask(answer_key(prompt, time_stamp, revision), [rgb], prompt, max_new_tokens=QWEN_TOKENS)
    boxes = TM.parse_detections(reply["answer"], rgb.shape[:2])
    return {"seconds": reply["seconds"], "detections": [{"box": list(box), "score": None, "label": label} for box, label in boxes],
            "answer": reply["answer"], "complete": reply["answer"].rstrip().rstrip("`").rstrip().endswith("]")}


def run_part_names(args: argparse.Namespace, scene: P.Scene, steady: list[str], indices: list[int]) -> None:
    """sam3 or qwen8b on the chosen keyframes; the record is saved after every keyframe."""
    from report import mem_steps_helpers as H

    model_id, revision = TM.TOOLS[args.tool]
    out = args.runs / "tool_maps" / args.tool
    file = out / f"{args.visit}.json"
    image_hw = P.frame_hw(scene, steady[0], P.PIXEL_STEP)
    fresh = {"visit": args.visit, "tool": args.tool, "model": f"{model_id}@{revision}", "version": TM.VERSION, "times": steady,
             "image_hw": list(image_hw), "prompts": list(TM.PART_NAMES) if args.tool == "sam3" else TM.detection_prompt(),
             "threshold": TM.SCORE_THRESHOLD if args.tool == "sam3" else None, "keyframes": {}}
    record = start_record(file, fresh)
    todo = [k for k in indices if str(k) not in record["keyframes"]]
    if args.tool == "sam3":
        detector = Sam3Detector() if todo else None
    else:
        reader = H.CachedReader(out / f"answers_{args.visit}.json", lambda: VisionLanguageModel(model_id=model_id, revision=revision))
    for k in todo:
        rgb = P.input_frame(scene, steady[k], np.zeros(image_hw, dtype=np.float32)).rgb
        entry = sam3_entry(detector, rgb) if args.tool == "sam3" else qwen_entry(reader, rgb, steady[k], revision)
        record["keyframes"][str(k)] = {"time": steady[k], **entry}
        save(record, file)
    done = len(record["keyframes"])
    boxes = sum(len(entry["detections"]) for entry in record["keyframes"].values())
    print(f"{args.visit} {args.tool}: {done} of {len(steady)} keyframes done, {boxes} raw boxes, "
          f"{TM.tool_seconds(record):.0f} s of tool time", flush=True)


def run_task_prompts(args: argparse.Namespace, scene: P.Scene, result: dict) -> None:
    """sam3_tasks: every task's text as a SAM 3 prompt on c3's even frames; KEEP_PER_FRAME best instances per frame and
    task, whatever their score (arm sam3q applies the threshold); the record is saved after every frame."""
    uniform = keyframes.scene_keyframes(scene, "uniform", result["n_keyframes"])
    texts = result["tasks"]["texts"]
    image_hw = P.frame_hw(scene, uniform[0], P.PIXEL_STEP)
    file = args.runs / "tool_maps" / TM.TASK_PROMPTED / f"{args.visit}.json"
    fresh = {"visit": args.visit, "tool": TM.TASK_PROMPTED, "model": "@".join(TM.SAM3), "version": TM.VERSION, "times": uniform,
             "image_hw": list(image_hw), "tasks": texts, "keep_per_frame": TM.KEEP_PER_FRAME, "frames": {}}
    record = start_record(file, fresh)
    if record["tasks"] != texts:
        raise SystemExit(f"{file} holds other tasks; move it away to start again")
    todo = [f for f in range(len(uniform)) if str(f) not in record["frames"]]
    detector = Sam3Detector() if todo else None
    if detector is not None:
        record["cut_texts"] = [i for i, text in enumerate(texts) if detector.tokens(text) > TEXT_TOKENS]
    for f in todo:
        rgb = P.input_frame(scene, uniform[f], np.zeros(image_hw, dtype=np.float32)).rgb
        start = time.time()
        found = detector.detect(rgb, texts, threshold=None, keep=TM.KEEP_PER_FRAME)
        record["frames"][str(f)] = {"time": uniform[f], "seconds": round(time.time() - start, 2), "detections": found}
        save(record, file)
    print(f"{args.visit} {TM.TASK_PROMPTED}: {len(record['frames'])} of {len(uniform)} frames done, {len(texts)} tasks", flush=True)


def keyframes_to_run(args: argparse.Namespace, scene: P.Scene, steady: list[str]) -> list[int]:
    if args.gate:
        labels = P.label_maps(scene, P.gt_frames(scene, steady))  # ground truth: only to choose what the lead is shown
        return TM.gate_keyframes(labels)
    if args.keyframes:
        return sorted({int(k) for k in args.keyframes.split(",")})
    return list(range(len(steady)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--tool", required=True, choices=CHOICES)
    parser.add_argument("--keyframes", default="", help="comma-separated keyframe indices (default: all 40)")
    parser.add_argument("--gate", action="store_true", help="only the two keyframes of the gate figure")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    args = parser.parse_args()
    result = json.loads((args.runs / f"{args.visit}.json").read_text())
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", args.visit, result["video"])
    if args.tool == TM.TASK_PROMPTED:
        run_task_prompts(args, scene, result)
        return
    steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
    run_part_names(args, scene, steady, keyframes_to_run(args, scene, steady))


if __name__ == "__main__":
    main()
