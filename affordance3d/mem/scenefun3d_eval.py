"""Predictions in the submission format of the SceneFun3D task-driven affordance grounding benchmark (Task 2).

The official scorer (third_party/scenefun3d/eval/affordance_grounding/evaluate.py, MIT licence) reads, for every task,
`<visit_id>_<desc_id>.txt` with one line per predicted mask, "<mask path> <confidence>", and each mask as a run-length
encoding over the points of the 5 mm laser scan. Each target element of a task is its own ground-truth instance, and
points labelled "exclude" are dropped before scoring.

The published Task 2 numbers (Fun3DU, UniFunc3D, AffordBot, AFFORDMEM) do not use that scorer. They follow Fun3DU's
protocol, reimplemented here from the description and code that UniFunc3D released (https://github.com/garry1ng/unifunc3d,
commit f9cccff, release_lib/common.py; that repository has no licence file, so this is a rewrite, not a copy): on the
cropped laser scan
(crop_mask == 1, original vertex order), one predicted point set per task is compared with the union of the task's
annotated elements (exclude annotations left out). AP25 and AP50 say whether the point precision reaches 0.25 and 0.5,
mAP averages that over the thresholds 0.50 to 0.95, AR25, AR50 and mAR do the same with the point recall, and mIoU is
the point IoU; every number is a mean over tasks. These "AP" values are not instance-detection AP.
"""
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MASK_FOLDER = "predicted_masks"
FUN3DU_METRICS = ("mAP", "AP50", "AP25", "mAR", "AR50", "AR25", "mIoU")
# float32, as torch.linspace(0.5, 0.95, 10) gives them in the historical code
FUN3DU_THRESHOLDS = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], dtype=np.float32)


def rle_encode(mask: np.ndarray) -> str:
    """Run-length encoding of a 1-D binary mask, as the official scorer decodes it: "start length" pairs, 1-based."""
    flat = np.asarray(mask, dtype=bool).ravel().astype(np.int8)
    padded = np.concatenate([[0], flat, [0]])
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def rle_decode(counts: str, length: int) -> np.ndarray:
    """The inverse of rle_encode (the scorer's convention), for tests and checks."""
    values = np.asarray(counts.split(), dtype=np.int64)
    mask = np.zeros(length, dtype=bool)
    for start, run in zip(values[0::2] - 1, values[1::2]):
        mask[start:start + run] = True
    return mask


def write_task(pred_dir: Path, visit: str, desc_id: str, masks: Sequence[np.ndarray], scores: Sequence[float],
               n_points: int) -> Path:
    """Write one task's predictions. With no mask, one empty mask is written: the scorer needs a line for every task
    and drops masks without points, so the task counts as unanswered (a miss)."""
    if len(masks) != len(scores):
        raise ValueError(f"{len(masks)} masks but {len(scores)} scores")
    if any(len(mask) != n_points for mask in masks):
        raise ValueError(f"every mask must cover the {n_points} laser-scan points")
    if not masks:
        masks, scores = [np.zeros(n_points, dtype=bool)], [0.0]
    (pred_dir / MASK_FOLDER).mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        name = f"{MASK_FOLDER}/{visit}_{desc_id}_{i:03d}.txt"
        (pred_dir / name).write_text(rle_encode(mask))
        lines.append(f"{name} {float(score):.6f}")
    task_file = pred_dir / f"{visit}_{desc_id}.txt"
    task_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    return task_file


def element_masks(n_points: int, annotations: Sequence[dict], annot_ids: Sequence[str]) -> list[np.ndarray]:
    """One mask over the laser-scan points per annotated element of a task, in the task's order. A missing annotation
    id raises KeyError, as the official ground-truth preparation stops on it (task_ground_truth, for the Fun3DU
    protocol, skips it instead, as the released evaluation does)."""
    by_id = {item["annot_id"]: item["indices"] for item in annotations}
    masks = []
    for annot_id in annot_ids:
        mask = np.zeros(n_points, dtype=bool)
        mask[np.asarray(by_id[annot_id], dtype=np.int64)] = True
        masks.append(mask)
    return masks


def scan_points(ply: Path) -> int:
    """The number of points of a laser scan, from its PLY header."""
    with ply.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"{ply}: no vertex count in the header")


def cropped_indices(crop: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Positions in the cropped scan (crop == 1, original order) of full-scan point indices; points outside the crop
    are dropped. Sorted and unique."""
    crop = np.asarray(crop).ravel()
    full = np.unique(np.asarray(full, dtype=np.int64))
    if full.size and (full[0] < 0 or full[-1] >= len(crop)):
        raise ValueError("full-scan indices out of bounds")
    kept = np.flatnonzero(crop == 1)
    return np.searchsorted(kept, full[crop[full] == 1])


def task_ground_truth(crop: np.ndarray, annotations: Sequence[dict], annot_ids: Sequence[str]) -> np.ndarray:
    """The union of a task's annotated elements on the cropped scan; ids without an annotation and "exclude"
    annotations contribute nothing (as the released evaluation does)."""
    by_id = {item["annot_id"]: item for item in annotations}
    chunks = [np.asarray(by_id[a]["indices"], dtype=np.int64) for a in annot_ids
              if a in by_id and by_id[a]["label"] != "exclude"]
    return cropped_indices(crop, np.concatenate(chunks)) if chunks else np.empty(0, dtype=np.int64)


def fun3du_scores(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """One task by the Fun3DU protocol; `gt` and `pred` are sorted unique cropped-scan indices."""
    intersection = np.intersect1d(gt, pred, assume_unique=True).size
    precision = np.float32(intersection / len(pred)) if len(pred) else np.float32(0)
    recall = np.float32(intersection / len(gt)) if len(gt) else np.float32(0)
    union = len(gt) + len(pred) - intersection
    return {"mAP": float(np.mean(precision >= FUN3DU_THRESHOLDS, dtype=np.float32)),
            "AP50": float(precision >= np.float32(0.5)), "AP25": float(precision >= np.float32(0.25)),
            "mAR": float(np.mean(recall >= FUN3DU_THRESHOLDS, dtype=np.float32)),
            "AR50": float(recall >= np.float32(0.5)), "AR25": float(recall >= np.float32(0.25)),
            "mIoU": intersection / union if union else 0.0}


def fun3du_means(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Every metric as a mean over tasks, in percent."""
    return {metric: 100 * float(np.mean([row[metric] for row in rows])) for metric in FUN3DU_METRICS}
