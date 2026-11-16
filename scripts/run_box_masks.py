"""SAM 3 masks of the boxes the model chose (discussion draft, section 27.2): one per task and arm.

    python scripts/run_box_masks.py --visit 421254 [--arms p4t_sam3,c3,sam3q,p4n]

Reads runs/mem_phase1/controls_8b_all/<visit>.json (scripts/run_mem_controls.py --all-tasks). For every arm and task
with a chosen box, SAM 3's interactive mode (transformers Sam3TrackerModel: one box prompts one object, as in SAM 2)
runs on the frame the model saw, at the working resolution; of its candidate masks the one with the highest predicted
quality is kept, with that score (the confidence the official scorer needs). A task without a box gets no mask. When
SAM 3 returns an empty mask, the box itself is the mask, with score 0. Writes runs/scenefun3d_eval/box_masks/<visit>.npz:
per arm the packed masks, their scores, a status per task (mask, box, none), the frame times and the image size.
GPU through the queue (the job sets AFFORDANCE3D_DEVICE=cuda).
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (imported quietly before the pipeline, which needs it)

from affordance3d.mem import pipeline as P  # noqa: E402
from affordance3d.mem import tool_maps as TM  # noqa: E402
from affordance3d.mem.encoders import default_device, model_source  # noqa: E402

ARMS = ("p4t_sam3", "c3", "sam3q", "p4n")
OUT = REPO_ROOT / "runs" / "scenefun3d_eval" / "box_masks"
RECORDS = REPO_ROOT / "runs" / "mem_phase1" / "controls_8b_all"


def best_mask(masks: np.ndarray, scores: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, float, str]:
    """The candidate with the highest predicted quality; the box itself (score 0) when that mask is empty."""
    best = int(np.argmax(scores))
    mask = np.asarray(masks[best], dtype=bool)
    if mask.any():
        return mask, float(scores[best]), "mask"
    left, top, right, bottom = box
    fallback = np.zeros(mask.shape, dtype=bool)
    fallback[max(top, 0):bottom, max(left, 0):right] = True
    return fallback, 0.0, "box"


class BoxSegmenter:
    """SAM 3 in its interactive mode: a box in, candidate masks of that one object and their predicted quality out."""

    def __init__(self) -> None:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        source = model_source(*TM.SAM3)
        self.device = default_device()
        self._processor = Sam3TrackerProcessor.from_pretrained(**source)
        self._model = Sam3TrackerModel.from_pretrained(**source).eval().to(self.device)

    def segment(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from PIL import Image

        inputs = self._processor(images=Image.fromarray(rgb), input_boxes=[[[float(v) for v in box]]], return_tensors="pt")
        sizes = inputs["original_sizes"]
        with torch.inference_mode():
            out = self._model(**inputs.to(self.device), multimask_output=True)
        masks = self._processor.post_process_masks(out.pred_masks.cpu(), sizes)[0][0].numpy().astype(bool)
        return masks, out.iou_scores[0, 0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", type=Path, default=OUT, help="folder for <visit>.npz (default: runs/scenefun3d_eval/box_masks)")
    args = parser.parse_args()
    arms = [arm for arm in args.arms.split(",") if arm]
    if set(arms) - set(ARMS):
        raise SystemExit(f"arms must be among {ARMS}")
    record = json.loads((RECORDS / f"{args.visit}.json").read_text())
    missing = [arm for arm in arms if arm not in record["arms"]]
    if missing or len(record.get("desc_ids", [])) != len(record["tasks"]):
        raise SystemExit(f"{args.visit}: the all-tasks record lacks arms {missing} or its description ids")
    result = json.loads((REPO_ROOT / "runs" / "mem_phase1" / f"{args.visit}.json").read_text())
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", args.visit, result["video"])
    segmenter, images, saved, started = BoxSegmenter(), {}, {}, time.time()
    for arm in arms:
        rows = record["arms"][arm]
        if len(rows) != len(record["tasks"]):
            raise SystemExit(f"{args.visit} {arm}: {len(rows)} answers for {len(record['tasks'])} tasks")
        packed, scores, status, times, shape = [], [], [], [], None
        for row in rows:
            times.append(row["frame"] or "")
            if row["box"] is None or row["frame"] is None:
                packed.append(None), scores.append(np.nan), status.append("none")
                continue
            t = row["frame"]
            if t not in images:
                images[t] = P.input_frame(scene, t, np.zeros(P.frame_hw(scene, t, P.PIXEL_STEP), dtype=np.float32)).rgb
            if shape is None:
                shape = images[t].shape[:2]
            elif images[t].shape[:2] != shape:
                raise SystemExit(f"{args.visit} {arm}: frames of different sizes")
            candidates, quality = segmenter.segment(images[t], tuple(int(v) for v in row["box"]))
            mask, score, how = best_mask(candidates, quality, tuple(int(v) for v in row["box"]))
            packed.append(np.packbits(mask.ravel())), scores.append(score), status.append(how)
        width = len(np.packbits(np.zeros(int(np.prod(shape)), dtype=bool))) if shape else 0
        saved[f"{arm}/masks"] = np.stack([p if p is not None else np.zeros(width, np.uint8) for p in packed]) if shape else np.zeros((len(rows), 0), np.uint8)
        saved[f"{arm}/scores"] = np.array(scores, dtype=np.float32)
        saved[f"{arm}/status"] = np.array(status)
        saved[f"{arm}/frames"] = np.array(times)
        saved[f"{arm}/image_hw"] = np.array(shape if shape else (0, 0))
        counts = {how: status.count(how) for how in ("mask", "box", "none")}
        print(f"{args.visit} {arm}: {counts}", flush=True)
    saved["desc_ids"] = np.array(record["desc_ids"])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.visit}.tmp.npz", **saved)
    (args.out / f"{args.visit}.tmp.npz").rename(args.out / f"{args.visit}.npz")
    print(f"{args.visit}: {len(images)} frames, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
