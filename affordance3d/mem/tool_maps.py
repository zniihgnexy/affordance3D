"""Off-the-shelf tools build the affordance map (discussion draft, section 24.9): their boxes, and how the boxes are scored.

Two frozen tools look at the 40 steady keyframes of a scene and are given nothing but the image and the eight part names
of the prompts' definition of "the part": SAM 3, prompted once per name, and Qwen3-VL-8B, asked once per keyframe for JSON
boxes of those parts. scripts/run_tool_maps.py saves their boxes, scripts/score_tool_maps.py scores them against the
projected annotations, and scripts/run_mem_controls.py reads them as the memory's regions. A box finds a part when its
centre lies on the part widened by grounding.MARGIN_PX (8 pixels): the rule every control arm is scored by.

Fixed before anything ran (also listed in the summary that scripts/score_tool_maps.py writes):
- SAM 3 keeps the instances scoring above 0.5, the default of its release and model card; a mask becomes its bounding box
  (the predicted box when the mask is empty).
- Within one keyframe, a box overlapping a better one by IoU 0.5 or more is the same find (found under two names, or
  twice) and is dropped; a tool without scores (Qwen) keeps its own order.
- The write budgets of 10 and 20 keyframes take every 4th and every 2nd keyframe, starting with the first.
- A part is visible in a keyframe when it covers at least MIN_PART_PIXELS pixels there, as in phase 1.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from affordance3d.mem import grounding as G
from affordance3d.mem import policies
from affordance3d.mem.encoders import VLMS
from affordance3d.mem.pipeline import MIN_PART_PIXELS
from affordance3d.mem.tokens import context_box

PART_NAMES = ("handle", "knob", "button", "switch", "key", "pedal", "plug", "socket")  # the list in grounding.PART
SAM3 = ("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7")  # SAM License
TOOLS = {"sam3": SAM3, "qwen8b": VLMS["8b"]}  # the two tools that build a map from the part names
TASK_PROMPTED = "sam3_tasks"  # SAM 3 prompted with each task's text on c3's even frames (arm sam3q)
TOOL_NAMES = {"sam3": "SAM 3", "qwen8b": "Qwen3-VL-8B", TASK_PROMPTED: "SAM 3, task text"}
TIE_TOOL = "sam3"  # section 24.9: recall intervals that overlap are a tie, and SAM 3 decides
SCORE_THRESHOLD = 0.5
MERGE_IOU = 0.5
BUDGETS = (10, 20, 40)
TOP_FRAMES, MAX_MARKS = 3, 40  # as the reading step of phase 1 and the controls
KEEP_PER_FRAME = 40  # sam3_tasks stores this many best instances of every frame and task, whatever their score
VERSION = "tool-maps-v1"  # stored with every map; change it when a prompt, a threshold or the image given to a tool changes
Box = tuple[int, int, int, int]  # left, top, right, bottom (exclusive), pixels


@dataclass(frozen=True)
class Detection:
    keyframe: int  # index into the frames the tool was run on
    box: Box
    score: float | None  # None for a tool that gives no score
    label: str


# ---------------------------------------------------------------- reading a tool's output

def detection_prompt() -> str:
    """Qwen3-VL as a detector: every one of the eight kinds of part in one keyframe, as JSON boxes."""
    names = ", ".join(PART_NAMES[:-1]) + f" and {PART_NAMES[-1]}"
    return (f"This image is a frame from a video of one room. Find every {names} in it: the small elements a hand or "
            f"foot touches, not whole objects. Answer in JSON as a list [{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": "
            f"<name>}}, ...], with coordinates from 0 to 1000 relative to the image and each label one of these names. "
            f"Answer [] if there is none.")


def parse_detections(answer: str, image_hw: tuple[int, int]) -> list[tuple[Box, str]]:
    """Every readable box of an answer such as [{"bbox_2d": [10, 20, 300, 400], "label": "handle"}, ...], in pixels.

    Each object is read by grounding.parse_grounding, so the same tolerance applies; an object whose box cannot be read
    (outside 0-1000, zero size, cut off) is skipped.
    """
    found = []
    for item in re.findall(r"\{[^{}]*\}", answer):
        grounding = G.parse_grounding(item, 1)
        if grounding.box is None:
            continue
        label = re.search(r'"?label"?\s*[:=]\s*"([^"]*)"', item)
        found.append((G.to_pixels(grounding.box, image_hw), label.group(1).strip().lower() if label else ""))
    return found


def part_name(label: str) -> str:
    """The part name a label stands for (the first of PART_NAMES it contains as a word, plural allowed), or "other"."""
    return next((name for name in PART_NAMES if re.search(rf"\b{name}(?:e?s)?\b", label.lower())), "other")


def mask_box(mask: np.ndarray) -> Box | None:
    """Bounding box of a binary mask; None when the mask is empty."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def box_iou(first: Box, second: Box) -> float:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    inter = width * height
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_duplicates(detections: Sequence[Detection], iou: float = MERGE_IOU) -> list[Detection]:
    """One detection per find, in keyframe order and, within a keyframe, best first: a box overlapping a kept box of the
    same keyframe by `iou` or more is dropped. Detections without a score keep the tool's order."""
    order = sorted(range(len(detections)), key=lambda i: (detections[i].keyframe, -(detections[i].score or 0.0), i))
    kept: list[Detection] = []
    for i in order:
        detection = detections[i]
        same = (other for other in kept if other.keyframe == detection.keyframe)
        if all(box_iou(other.box, detection.box) < iou for other in same):
            kept.append(detection)
    return kept


def read_tool_map(file: Path) -> dict:
    if not file.exists():
        raise FileNotFoundError(f"{file} is missing: run scripts/run_tool_maps.py for this scene and tool first")
    return json.loads(file.read_text())


def missing_keyframes(record: dict, n_keyframes: int) -> list[int]:
    """Keyframes a tool has not been run on yet."""
    return [k for k in range(n_keyframes) if str(k) not in record["keyframes"]]


def tool_detections(record: dict, threshold: float = SCORE_THRESHOLD) -> list[Detection]:
    """A tool's finds in a saved map: above the score threshold (a tool without scores keeps all), duplicates merged,
    in keyframe order and, within a keyframe, best first."""
    raw = [Detection(int(k), tuple(int(v) for v in found["box"]), found.get("score"), found.get("label", ""))
           for k, entry in record["keyframes"].items() for found in entry["detections"]
           if found.get("score") is None or found["score"] >= threshold]
    return merge_duplicates(raw)


def tool_seconds(record: dict) -> float:
    """How long the tool ran on the keyframes of the record (model loading not included)."""
    return float(sum(entry["seconds"] for entry in record["keyframes"].values()))


def candidates_of(detections: Sequence[Detection], image_hw: tuple[int, int]) -> list[policies.Candidate]:
    """The finds as memory regions: the box, and the crop around it that the image-text encoder reads (as for P4)."""
    return [policies.Candidate(d.keyframe, d.box, context_box(d.box, image_hw)) for d in detections]


def task_detections(record: dict, task: int) -> list[Detection]:
    """The instances stored by sam3_tasks for one task (its index in the record), over all frames."""
    return [Detection(int(k), tuple(int(v) for v in found["box"]), float(found["score"]), "task")
            for k, entry in record["frames"].items() for found in entry["detections"][task]]


def task_prompt_marks(detections: Sequence[Detection], threshold: float = SCORE_THRESHOLD, top_frames: int = TOP_FRAMES,
                      max_marks: int = MAX_MARKS) -> tuple[list[int], list[Detection]]:
    """Arm sam3q for one task, SAM 3 prompted with its text on the even frames: the frames that hold the best boxes (at
    most `top_frames`, best first) and their best boxes (at most `max_marks`; by frame, then best first).

    Boxes above the default threshold count; when none passes in any frame all stored boxes count, so the arm always has
    candidates (this only helps the arm). Duplicates are merged as in the tool maps.
    """
    kept = merge_duplicates([d for d in detections if d.score >= threshold] or list(detections))
    best: dict[int, float] = {}
    for detection in kept:
        best[detection.keyframe] = max(best.get(detection.keyframe, -np.inf), detection.score)
    frames = sorted(best, key=lambda k: (-best[k], k))[:top_frames]
    inside = sorted((d for d in kept if d.keyframe in frames), key=lambda d: -d.score)[:max_marks]
    rank = {k: i for i, k in enumerate(frames)}
    return frames, sorted(inside, key=lambda d: (rank[d.keyframe], -d.score))


# ---------------------------------------------------------------- scoring against the annotations (ground truth)

def visible_parts(labels: Sequence[np.ndarray]) -> dict[int, list[int]]:
    """Every annotated part seen in at least MIN_PART_PIXELS pixels of some keyframe, with those keyframes."""
    seen: dict[int, list[int]] = {}
    for k, label_map in enumerate(labels):
        parts, counts = np.unique(label_map[label_map >= 0], return_counts=True)
        for part, count in zip(parts, counts):
            if count >= MIN_PART_PIXELS:
                seen.setdefault(int(part), []).append(k)
    return dict(sorted(seen.items()))


def centres_on(boxes: Sequence[Box], mask: np.ndarray) -> np.ndarray:
    """Whether the centre of each box lies on the mask."""
    height, width = mask.shape
    centres = [G.box_centre(box) for box in boxes]
    return np.array([0 <= v < height and 0 <= u < width and bool(mask[v, u]) for u, v in centres], dtype=bool)


def any_part_mask(labels: np.ndarray) -> np.ndarray:
    """Every annotated pixel of one keyframe, widened by MARGIN_PX."""
    return G.target_mask(labels, np.unique(labels[labels >= 0]).tolist())


def sightings(labels: Sequence[np.ndarray], boxes: Mapping[int, Sequence[Box]]) -> dict[tuple[int, int], bool]:
    """For every (part, keyframe) where the part is visible: whether a box of that keyframe finds it."""
    found = {}
    for part, keyframes in visible_parts(labels).items():
        for k in keyframes:
            here = boxes.get(k, ())
            found[(part, k)] = bool(len(here)) and bool(centres_on(here, G.target_mask(labels[k], (part,))).any())
    return found


def budget_keyframes(n_keyframes: int, used: int) -> list[int]:
    """The keyframes of a smaller write budget: every (n_keyframes / used)-th, starting with the first."""
    step = n_keyframes // used
    if used < 1 or step * used != n_keyframes:
        raise ValueError(f"{used} keyframes do not divide {n_keyframes} evenly")
    return list(range(0, n_keyframes, step))


def part_outcomes(found: Mapping[tuple[int, int], bool], used: Sequence[int]) -> dict[str, np.ndarray]:
    """Per part visible in any keyframe (the fixed denominator, in part order), with only the keyframes `used`:
    found (on a used keyframe that shows it), visible (in a used keyframe), and always missed (visible, never found)."""
    used = set(used)
    by_part: dict[int, list[bool]] = {}
    for (part, k), hit in sorted(found.items()):
        by_part.setdefault(part, [])
        if k in used:
            by_part[part].append(hit)
    visible = np.array([bool(hits) for hits in by_part.values()], dtype=bool)
    found_any = np.array([any(hits) for hits in by_part.values()], dtype=bool)
    return {"found": found_any, "visible": visible, "always_missed": visible & ~found_any}


def false_alarms(labels: np.ndarray, boxes: Sequence[Box]) -> int:
    """Boxes of one keyframe whose centre lies on no annotated part (every part widened by MARGIN_PX)."""
    if not len(boxes):
        return 0
    return int((~centres_on(boxes, any_part_mask(labels))).sum())


def on_a_part(candidates: Sequence[policies.Candidate], labels: Sequence[np.ndarray]) -> list[int]:
    """Indices of the candidates whose box centre lies on an annotated part of its keyframe (the diagnostic arm p4tp)."""
    masks: dict[int, np.ndarray] = {}
    kept = []
    for i, candidate in enumerate(candidates):
        if candidate.keyframe not in masks:
            masks[candidate.keyframe] = any_part_mask(labels[candidate.keyframe])
        if centres_on([candidate.box], masks[candidate.keyframe])[0]:
            kept.append(i)
    return kept


def shown_on_target(shown: Sequence[policies.Candidate], labels: Sequence[np.ndarray], parts: Sequence[int],
                    centres: Mapping[int, Sequence[tuple[int, int]]]) -> tuple[bool, bool]:
    """Coverage of one task: whether a shown box lies on the target part, by the common rule (its centre on the part
    widened by MARGIN_PX) and by the phase-1 rule (the box, widened, holds the centre of a target part; `centres` per
    keyframe as report.mem_phase1_helpers.target_pixels gives them)."""
    masks = {k: G.target_mask(labels[k], parts) for k in {c.keyframe for c in shown}}
    common = any(centres_on([c.box], masks[c.keyframe])[0] for c in shown)
    phase1 = any(policies.covers(c, pixel) for c in shown for pixel in centres.get(c.keyframe, []))
    return bool(common), bool(phase1)


def deciding_tool(recall: Mapping[str, tuple[float, float, float]], tie: str = TIE_TOOL) -> dict:
    """Section 24.9: the tool with the higher per-part recall (rate, low, high); when the two 95% intervals overlap it is
    a tie and `tie` decides."""
    if len(recall) != 2 or tie not in recall:
        raise ValueError(f"needs the recall of two tools, one of them {tie!r}; got {sorted(recall)}")
    (first, a), (second, b) = recall.items()
    if a[1] <= b[2] and b[1] <= a[2]:
        return {"tool": tie, "reason": "tie: the two recall intervals overlap, so SAM 3 decides"}
    best = first if a[0] > b[0] else second
    return {"tool": best, "reason": "higher per-part recall, the two intervals apart"}


def gate_keyframes(labels: Sequence[np.ndarray], n: int = 2) -> list[int]:
    """The keyframes shown to the lead at the gate: the `n` showing most annotated parts (ties: the earlier), in time
    order. Ground truth only chooses what the lead sees; no tool is given it."""
    counts = [0] * len(labels)
    for keyframes in visible_parts(labels).values():
        for k in keyframes:
            counts[k] += 1
    return sorted(sorted(range(len(labels)), key=lambda k: (-counts[k], k))[:n])
