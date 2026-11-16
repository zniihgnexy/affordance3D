"""The reading step of the memory: tasks, numbered marks on keyframes, and the cache of the model's answers.

Shared by every experiment that asks the vision-language model to pick a numbered region (phase 1, the controls, C-1,
the pooled reading, the public evaluation). Moved unchanged from report/mem_steps_helpers.py, which re-exports it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from affordance3d.frames import Frame
from affordance3d.mem import pipeline as P
from affordance3d.mem import tokens as T

MARK_COLOUR, CHOSEN_COLOUR, RIGHT_COLOUR = (255, 60, 60), (40, 220, 90), (60, 140, 255)
MARKS_VERSION = "marks-v2"  # part of the answer cache key: change it when the drawing of the marks changes
BOX_PAD_PX = 8


@dataclass(frozen=True)
class Task:
    text: str
    parts: tuple[int, ...]  # ground-truth parts the task names
    keyframes: frozenset[int]  # keyframes that show one of them (ground truth, for scoring)


def tasks_of(scene: P.Scene, part_labels: Sequence[np.ndarray]) -> list[Task]:
    tasks = []
    for text, annot_ids in scene.tasks:
        parts = tuple(i for i, element in enumerate(scene.elements) if element.annot_id in annot_ids)
        shown = frozenset(k for k, labels in enumerate(part_labels)
                          if any((labels == p).sum() >= P.MIN_PART_PIXELS for p in parts))
        tasks.append(Task(text, parts, shown))
    return tasks


Box = tuple[int, int, int, int]  # left, top, right, bottom


def _overlap(a: Box, b: Box) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def label_spot(box: Box, size: tuple[int, int], taken: Sequence[Box], image_hw: tuple[int, int]) -> Box:
    """Where to write a mark's number: beside its box, on the side that covers least of the other boxes and numbers."""
    (left, top, right, bottom), (width, height) = box, size
    spots = [(left, top - height, left + width, top), (left, bottom, left + width, bottom + height),
             (left - width, top, left, top + height), (right, top, right + width, top + height)]

    def cost(spot: Box) -> int:
        outside = spot[0] < 0 or spot[1] < 0 or spot[2] > image_hw[1] or spot[3] > image_hw[0]
        return sum(_overlap(spot, other) for other in taken) + (10 ** 9 if outside else 0)

    return min(spots, key=cost)


def marked_keyframes(frames: Sequence[Frame], memory: T.L2Memory, keyframe_ids: Sequence[int], chosen: int | None = None,
                     right: Sequence[int] = ()) -> tuple[list[np.ndarray], list[int]]:
    """Copies of the keyframes with every affordance region boxed and numbered 1, 2, ...; and the token behind each number.

    `chosen` and `right` only colour the display copy (the model's pick, the true part); the model sees neither.
    """
    return mark_boxes(frames, [(token.region.keyframe, token.region.box) for token in memory.tokens], keyframe_ids, chosen, right)


def mark_boxes(frames: Sequence[Frame], regions: Sequence[tuple[int, Box]], keyframe_ids: Sequence[int],
               chosen: int | None = None, right: Sequence[int] = ()) -> tuple[list[np.ndarray], list[int]]:
    """Copies of the keyframes with the given (keyframe, box) regions numbered 1, 2, ...; and the region behind each number."""
    images, tokens = [], []
    for k in keyframe_ids:
        image = frames[k].rgb.copy()
        here = [i for i, (keyframe, _) in enumerate(regions) if keyframe == k]
        boxes = [(b[0] - BOX_PAD_PX, b[1] - BOX_PAD_PX, b[2] + BOX_PAD_PX, b[3] + BOX_PAD_PX)
                 for b in (regions[i][1] for i in here)]
        taken = list(boxes)
        for index, box in zip(here, boxes):
            tokens.append(index)
            number = len(tokens)
            colour = CHOSEN_COLOUR if number == chosen else RIGHT_COLOUR if number in right else MARK_COLOUR
            cv2.rectangle(image, box[:2], box[2:], colour, 4 if colour != MARK_COLOUR else 2)
            (width, height), _ = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            spot = label_spot(box, (width + 6, height + 8), [other for other in taken if other != box], frames[k].image_hw)
            taken.append(spot)
            cv2.rectangle(image, spot[:2], spot[2:], colour, -1)
            cv2.putText(image, str(number), (spot[0] + 3, spot[3] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        images.append(image)
    return images, tokens


def reading_key(model: str, prompt: str, keyframe_ids: Sequence[int], boxes: Sequence[Box]) -> str:
    """Cache key of one question to the model: it changes with the model, the wording, the keyframes shown,
    the numbered boxes and the way the marks are drawn."""
    payload = json.dumps([MARKS_VERSION, model, prompt, [int(k) for k in keyframe_ids], [list(map(int, box)) for box in boxes]])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class CachedReader:
    """Asks the vision-language model once per question and keeps the answers (and how long they took) on disk."""

    def __init__(self, cache: Path, make_model: Callable[[], object]) -> None:
        self._cache, self._make_model, self._model = cache, make_model, None
        self.answers: dict[str, dict] = json.loads(cache.read_text()) if cache.exists() else {}

    def ask(self, key: str, images: Sequence[np.ndarray], prompt: str, **options: object) -> dict:
        """The model's answer for `key`, asked now if it is not on disk; `options` go to the model (e.g. labels, max_new_tokens)."""
        if key not in self.answers:
            self._model = self._model or self._make_model()
            start = time.time()
            answer = self._model.ask(images, prompt, **options)
            self.answers[key] = {"prompt": prompt, "answer": answer, "seconds": round(time.time() - start, 1)}
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            self._cache.write_text(json.dumps(self.answers, indent=1))
        return self.answers[key]
