"""Letting a vision-language model find the part by itself: the questions, reading its answers, and one scoring rule.

Used by the controls of round 7 (discussion draft, section 24). Qwen3-VL reports a box as [x1, y1, x2, y2] on a
0-1000 scale relative to the image it was shown. Every arm, whether it picks one of our numbered boxes or draws a box
itself, is scored the same way: the centre of the chosen box must lie on the target part, widened by `MARGIN_PX`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

RELATIVE_SCALE = 1000.0
MARGIN_PX = 8
PART = ("the part that must be operated to do the task (the small element a hand or foot touches, such as a handle, knob, "
        "button, switch, key, pedal, plug or socket, not the whole object)")  # SceneFun3D annotates these elements
Box = tuple[int, int, int, int]  # left, top, right, bottom (exclusive), pixels


def numbered_prompt(task: str, n_marks: int) -> str:
    """Pick one of our numbered boxes; neutral wording: a box is a region, not said to be operable."""
    return (f"These images are keyframes from a walk through one room. Boxes numbered 1 to {n_marks} mark regions in the "
            f"images. Task: \"{task}\". Which numbered box shows {PART}? Answer with the number only.")


def grounding_prompt(task: str, n_images: int) -> str:
    """Find the part in one of several numbered images and draw a box around it."""
    return (f"These {n_images} images are keyframes from a walk through one room, numbered 1 to {n_images} in the order "
            f"shown. Task: \"{task}\". Find {PART}. Answer in JSON as "
            f"{{\"image\": <image number>, \"bbox_2d\": [x1, y1, x2, y2]}}, with the box in coordinates from 0 to 1000 "
            f"relative to that image.")


def frame_prompt(task: str, n_frames: int) -> str:
    """Pick the frame of a video in which the part is best visible."""
    return (f"These {n_frames} frames are taken evenly from a video of one room, numbered 1 to {n_frames} in time order. "
            f"Task: \"{task}\". In which frame is {PART} best visible? Answer with the frame number only.")


def single_grounding_prompt(task: str) -> str:
    """Draw a box around the part in one image."""
    return (f"This image is a frame from a video of one room. Task: \"{task}\". Find {PART}. Answer in JSON as "
            f"{{\"bbox_2d\": [x1, y1, x2, y2]}}, with coordinates from 0 to 1000 relative to the image.")


def image_labels(n: int, word: str = "Image") -> list[str]:
    """Text put before each image so that the model can refer to it by number."""
    return [f"{word} {i}:" for i in range(1, n + 1)]


@dataclass(frozen=True)
class Grounding:
    image: int | None  # 1-based number of the chosen image; None when the answer names none or an impossible one
    box: tuple[float, float, float, float] | None  # 0-1000 relative to that image; None when unreadable


def _numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", text)]


def parse_grounding(answer: str, n_images: int = 1) -> Grounding:
    """Read the image number and the box from an answer such as {"image": 2, "bbox_2d": [10, 20, 300, 400]}.

    Tolerates code fences, a list around the object, extra keys and reversed corners. A box outside the 0-1000 range or
    of zero size is unreadable. With one image, the image number is 1 unless the answer says otherwise.
    """
    image = None
    match = re.search(r'"?image"?\s*[:=]\s*(\d+)', answer)
    if match and 1 <= int(match.group(1)) <= n_images:
        image = int(match.group(1))
    elif n_images == 1 and not match:
        image = 1
    box = None
    found = re.search(r'bbox_2d"?\s*[:=]\s*\[([^\]]*)\]', answer) or re.search(r"\[([^\[\]]*)\]", answer)
    if found:
        values = _numbers(found.group(1))
        if len(values) == 4:
            x1, y1, x2, y2 = values
            x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            if 0 <= x1 < x2 <= RELATIVE_SCALE and 0 <= y1 < y2 <= RELATIVE_SCALE:
                box = (x1, y1, x2, y2)
    return Grounding(image, box)


def to_pixels(box: tuple[float, float, float, float], image_hw: tuple[int, int]) -> Box:
    """A 0-1000 relative box in the pixels of an image of the given height and width."""
    height, width = image_hw
    x1, y1, x2, y2 = box
    left, right = int(np.floor(x1 / RELATIVE_SCALE * width)), int(np.ceil(x2 / RELATIVE_SCALE * width))
    top, bottom = int(np.floor(y1 / RELATIVE_SCALE * height)), int(np.ceil(y2 / RELATIVE_SCALE * height))
    return max(left, 0), max(top, 0), min(max(right, left + 1), width), min(max(bottom, top + 1), height)


def box_centre(box: Box) -> tuple[int, int]:
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def target_mask(labels: np.ndarray, parts: Sequence[int], margin_px: int = MARGIN_PX) -> np.ndarray:
    """The pixels of the target parts in one frame, widened by `margin_px` (ground truth: scoring only)."""
    mask = np.isin(labels, list(parts)).astype(np.uint8)
    if margin_px > 0 and mask.any():
        mask = cv2.dilate(mask, np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8))
    return mask.astype(bool)


def hits_part(box: Box | None, labels: np.ndarray, parts: Sequence[int], margin_px: int = MARGIN_PX) -> bool:
    """The rule used for every arm: the centre of the chosen box lies on a target part (widened by `margin_px`)."""
    if box is None:
        return False
    u, v = box_centre(box)
    height, width = labels.shape
    return bool(0 <= v < height and 0 <= u < width and target_mask(labels, parts, margin_px)[v, u])


def covers_a_centre(box: Box | None, centres: Sequence[tuple[int, int]], margin_px: int = MARGIN_PX) -> bool:
    """The phase-1 rule, reported alongside: the box, widened by `margin_px`, contains the centre of a target part."""
    if box is None:
        return False
    left, top, right, bottom = box
    return any(left - margin_px <= u < right + margin_px and top - margin_px <= v < bottom + margin_px for u, v in centres)


def answer_record(grounding: Grounding, box: Box | None, frame: str | None, hit: bool, hit_old: bool, answers: Sequence[str],
                  seconds: float, images_read: int) -> dict:
    """What one arm stores per task."""
    return {"image": grounding.image, "box": None if box is None else list(box), "frame": frame, "hit": bool(hit),
            "hit_old": bool(hit_old), "answers": list(answers), "seconds": round(float(seconds), 2), "images_read": int(images_read)}


def dumps(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False)
