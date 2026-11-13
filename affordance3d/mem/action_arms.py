"""C-1 (discussion draft, section 25.5): the arms that say how to act on a part, and the scorer that compares them.

V   a vision-language model answers in JSON: the contact point, one of the nine affordance labels, push/pull/rotate,
    the end of an arrow drawn from the point (the way the part moves; for a turning part, along its axis) and the
    hinge side. Its arrow is lifted to 3D with the depth map at both ends.
G   V's point and label; the direction is the depth map's normal at the point, turned into a motion by a rule on the
    label (rule_motion).
M   the memory's region (its centre and its annotated label: ORACLE) with the same rule.
O   the ground truth itself (A.truth_as_prediction): the scorer must give it 100%.

The primary scores compare directions as lines, sign ignored (C-0 found the planned sign reading wrong for pushed
parts). The joint "action-ready" success is: the point on the part, the right motion type, and a 3D line error of at
most LINE_OK_DEG. Push or pull is scored as an extra, against the documentation's reading (motion_dir is the way the
part moves). Arms see only inputs (RGB, monocular depth rescaled by LiDAR, device pose); the ground truth (part ids,
motions, registered cameras) is for scoring only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from affordance3d.frames import Frame
from affordance3d.geometry import backproject_pixels
from affordance3d.mem import action_records as A
from affordance3d.mem import tokens as T
from affordance3d.mem.actionable import surface_normal
from affordance3d.mem.grounding import PART, RELATIVE_SCALE, Box
from affordance3d.mem.pipeline import NORMAL_WINDOW_PX

LINE_OK_DEG = 20.0  # section 25.5: the joint success needs the 3D line error within this
MIN_NORMAL_POINTS = 3
PUSH_LABELS = ("key_press", "tip_push", "plug_in", "foot_push")  # the plan names the first three; foot_push is pressed too
PULL_LABELS = ("hook_pull", "pinch_pull", "unplug")
LABEL_HINTS = {  # the nine SceneFun3D affordance labels, as the question explains them
    "rotate": "turn a knob or dial", "key_press": "press a key", "tip_push": "push with a fingertip, e.g. a switch or button",
    "hook_pull": "pull with hooked fingers, e.g. a handle", "pinch_pull": "pull with a pinch, e.g. a small knob",
    "hook_turn": "hook the hand and turn, e.g. a door handle", "foot_push": "push with the foot, e.g. a pedal",
    "plug_in": "insert a plug", "unplug": "pull a plug out"}
MOTION_WORDS = {"push": ("trans", "push"), "press": ("trans", "push"), "pull": ("trans", "pull"),
                "rotate": ("rot", None), "turn": ("rot", None)}
HINGE_WORDS = {"left": "left", "right": "right", "top": "top", "bottom": "bottom", "none": A.ON_PART, "null": A.ON_PART}
EMPTY = A.Prediction(None, None, None, None, None, None, None, None)


# ---------------------------------------------------------------- V: the question and its answer

def action_prompt(task: str) -> str:
    labels = "; ".join(f"{name} ({hint})" for name, hint in LABEL_HINTS.items())
    return (f"This image is a frame from a video of one room. Task: \"{task}\". Find {PART}, and say how a hand or foot "
            f"operates it. Answer in JSON as {{\"point\": [x, y], \"label\": \"...\", \"motion\": \"...\", \"arrow_end\": "
            f"[x, y], \"hinge_side\": \"...\"}}, with coordinates from 0 to 1000 relative to the image. point: where the "
            f"hand or foot touches the part. label: one of {labels}. motion: push or pull if the part is pressed or slides, "
            f"rotate if it turns. arrow_end: the end of an arrow drawn from point, showing the way the part moves, or, if it "
            f"turns, along the axis it turns about. hinge_side: if the part swings on a hinge (a door or a lid), the side of "
            f"the part where the hinge is in the image (left, right, top or bottom); otherwise none.")


@dataclass(frozen=True)
class ActionAnswer:
    point: tuple[float, float] | None  # 0-1000, relative to the image
    label: str | None  # one of LABEL_HINTS
    motion: str | None  # push, pull or rotate
    arrow_end: tuple[float, float] | None
    hinge_side: str | None  # left, right, top, bottom or A.ON_PART


QUOTE = "[\"']?"


def _pair(answer: str, key: str) -> tuple[float, float] | None:
    found = re.search(rf"{QUOTE}{key}{QUOTE}\s*[:=]\s*\[([^\]]*)\]", answer)
    if not found:
        return None
    values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", found.group(1))]
    if len(values) != 2 or not all(0 <= v <= RELATIVE_SCALE for v in values):
        return None
    return values[0], values[1]


def _word(answer: str, key: str) -> str | None:
    found = re.search(rf"{QUOTE}{key}{QUOTE}\s*[:=]\s*{QUOTE}([A-Za-z][A-Za-z _\-]*)", answer)
    return None if not found else re.sub(r"[ \-]+", "_", found.group(1).strip().lower())


def parse_action_answer(answer: str) -> ActionAnswer:
    """Read V's JSON answer; tolerates code fences, prose around it, spaces or hyphens in the label and a few synonyms.
    A field that is missing, outside 0-1000 or not one of the allowed words is None."""
    label, motion, hinge = _word(answer, "label"), _word(answer, "motion"), _word(answer, "hinge_side")
    return ActionAnswer(_pair(answer, "point"), label if label in LABEL_HINTS else None,
                        motion if motion in MOTION_WORDS else None, _pair(answer, "arrow_end"), HINGE_WORDS.get(hinge or ""))


def to_pixel(xy: tuple[float, float], image_hw: tuple[int, int]) -> A.Pixel:
    """A 0-1000 relative point in the pixels of an image of the given height and width."""
    height, width = image_hw
    return (int(np.clip(np.floor(xy[0] / RELATIVE_SCALE * width), 0, width - 1)),
            int(np.clip(np.floor(xy[1] / RELATIVE_SCALE * height), 0, height - 1)))


# ---------------------------------------------------------------- geometry from the depth map (inputs only)

def lift(frame: Frame, pixel: A.Pixel) -> np.ndarray | None:
    """World point under a pixel, from the frame's depth; None where the depth is missing."""
    points, valid = backproject_pixels(frame, np.array([pixel], dtype=float))
    return points[0] if valid[0] else None


def lifted_direction(frame: Frame, start: A.Pixel, end: A.Pixel) -> np.ndarray | None:
    """A 2D arrow lifted to 3D with the depth at both ends (unit); None if an end has no depth or both lift to one point."""
    first, last = lift(frame, start), lift(frame, end)
    if first is None or last is None or np.allclose(first, last):
        return None
    return A.unit(last - first)


def depth_normal(frame: Frame, pixel: A.Pixel, window_px: int = NORMAL_WINDOW_PX) -> np.ndarray | None:
    """Normal of the plane through the depth points in a square window around the pixel, facing the camera."""
    u, v = pixel
    height, width = frame.image_hw
    rows, cols = np.mgrid[max(v - window_px, 0):min(v + window_px + 1, height), max(u - window_px, 0):min(u + window_px + 1, width)]
    valid = frame.depth_m[rows, cols] > 0
    if valid.sum() < MIN_NORMAL_POINTS:
        return None
    points, _ = backproject_pixels(frame, np.stack([cols[valid], rows[valid]], axis=1).astype(float))
    return surface_normal(points, A.camera_centre(frame))


@dataclass(frozen=True)
class RuleMotion:
    kind: str | None
    sense: str | None
    direction: np.ndarray | None


def rule_motion(label: str | None, normal: np.ndarray | None, up: np.ndarray) -> RuleMotion:
    """G's and M's rule: pressed parts move into the surface (-normal), pulled parts out of it (+normal), a knob turns
    about the normal, a door handle or hinge (hook_turn) about the vertical. `normal` faces the camera."""
    if label in PUSH_LABELS:
        return RuleMotion("trans", "push", None if normal is None else -normal)
    if label in PULL_LABELS:
        return RuleMotion("trans", "pull", normal)
    if label == "rotate":
        return RuleMotion("rot", None, normal)
    if label == "hook_turn":
        return RuleMotion("rot", None, A.unit(up))
    return RuleMotion(None, None, None)


# ---------------------------------------------------------------- the arms' predictions

def answer_prediction(answer: ActionAnswer, frame: Frame) -> A.Prediction:
    """V: its answer in the pixels of the (input) frame it saw; the arrow lifted to 3D with that frame's depth."""
    point = None if answer.point is None else to_pixel(answer.point, frame.image_hw)
    end = None if answer.arrow_end is None else to_pixel(answer.arrow_end, frame.image_hw)
    kind, sense = MOTION_WORDS.get(answer.motion or "", (None, None))
    has_arrow = point is not None and end is not None
    arrow = (end[0] - point[0], end[1] - point[1]) if has_arrow else None
    return A.Prediction(point, answer.label, kind, sense, arrow, None, answer.hinge_side,
                        lifted_direction(frame, point, end) if has_arrow else None)


def rule_prediction(point: A.Pixel | None, label: str | None, frame: Frame, up: np.ndarray) -> A.Prediction:
    """G and M: a point and a label; the motion from the depth normal at the point (rule_motion). No hinge side."""
    normal = None if point is None else depth_normal(frame, point)
    rule = rule_motion(label, normal, up)
    anchor = None if point is None else lift(frame, point)
    segment = None if rule.direction is None or anchor is None else A.projected_segment(anchor, rule.direction, frame)
    arrow = None if segment is None else (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])
    return A.Prediction(point, label, rule.kind, rule.sense, arrow, None, None, rule.direction)


def part_actions(labels: Sequence[str]) -> list[int]:
    """Each part's index into tokens.ACTIONS (as phase 1 built the affordance map)."""
    return [T.ACTIONS.index(label) if label in T.ACTIONS else T.NO_ACTION for label in labels]


def region_label(label_map: np.ndarray, actions_of_parts: Sequence[int], depth_m: np.ndarray, box: Box) -> tuple[str | None, bool]:
    """M's label (ORACLE): that of the frame's affordance-map region with exactly this box, as phase 1 found the regions;
    failing an exact match, the most common label inside the box. Also says whether the match was exact."""
    actions = T.action_map(label_map, actions_of_parts)
    _, regions = T.find_regions(actions, depth_m, 0)
    for region in regions:
        if region.box == tuple(box):
            return T.ACTIONS[region.action], True
    inside = actions[box[1]:box[3], box[0]:box[2]]
    inside = inside[inside != T.NO_ACTION]
    return (T.ACTIONS[int(np.bincount(inside).argmax())] if inside.size else None), False


# ---------------------------------------------------------------- scoring

def _line_2d(prediction: A.Prediction, view: A.PartView) -> float | None:
    if A.segment_px(view.arrow) == 0 or prediction.arrow is None or not any(prediction.arrow):
        return None
    truth = (view.arrow[1][0] - view.arrow[0][0], view.arrow[1][1] - view.arrow[0][1])
    return A.angle_deg(prediction.arrow, truth, as_lines=True)


def score_arm(prediction: A.Prediction | None, record: A.ActionRecord, labels: np.ndarray, frame: Frame, up: np.ndarray,
              answers_hinge: bool) -> dict[str, bool | float | None]:
    """One arm's answer against the truth in the frame it answered in (`labels`, `frame`: that frame's part ids and
    registered camera). True/False where a field applies, None where it does not; angles in degrees.

    Primary: contact (on a target part, widened by 8 px), label, type, line_ok (3D line error <= LINE_OK_DEG), and for
    rotations axis_class and hinge_side (only for arms that answer it); joint = contact and type and line_ok.
    Descriptive: line_deg, line_2d_deg. Baseline on the same frame: vertical_* (the axis always answered vertical).
    Extra (outside the plan): sense, push or pull, and signed_deg, for translations, in the documentation's reading.
    A missing prediction is wrong wherever a field applies.
    """
    prediction = prediction or EMPTY
    ref = A.reference_part(prediction, record, labels)
    motion, view = record.motions[ref], record.views[ref]
    if motion is None:
        raise ValueError(f"{record.task!r}: C-1 scores only tasks whose parts all have a motion")
    line = None if prediction.direction is None else A.angle_deg(prediction.direction, motion.direction, as_lines=True)
    out: dict[str, bool | float | None] = {
        "contact": A.on_target(prediction.point, record, labels), "label": prediction.label == record.labels[ref],
        "type": prediction.kind == motion.kind, "line_deg": line, "line_ok": line is not None and line <= LINE_OK_DEG,
        "line_2d_deg": _line_2d(prediction, view), "axis_class": None, "hinge_side": None,
        "vertical_axis_class": None, "vertical_line_ok": None, "sense": None, "signed_deg": None}
    if motion.kind == "rot":
        out["axis_class"] = prediction.direction is not None and A.axis_class(prediction.direction, frame) == view.axis_class
        if answers_hinge and view.hinge_side is not None:
            out["hinge_side"] = prediction.hinge_side == view.hinge_side
        out["vertical_axis_class"] = A.axis_class(A.unit(up), frame) == view.axis_class
        out["vertical_line_ok"] = A.angle_deg(up, motion.direction, as_lines=True) <= LINE_OK_DEG
    else:
        out["sense"] = prediction.sense == view.sense
        out["signed_deg"] = None if prediction.direction is None else A.angle_deg(prediction.direction, motion.direction, as_lines=False)
    out["joint"] = bool(out["contact"] and out["type"] and out["line_ok"])
    return out
