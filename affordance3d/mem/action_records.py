"""Action records from the SceneFun3D ground truth (C-0, discussion draft section 25.3): how the part of each task moves.

One record per task: the parts it names, with their affordance labels and motion types; for a translation the signed
direction, for a rotation the axis and its origin; and, in the keyframe that shows the task's parts best, each part's
contact pixel, projected direction, push or pull, axis class and hinge side. A scorer compares a predicted record with
the true one, field by field.

Sign of a translation, two readings. "viewer" (the C-0 plan, from third_party/scenefun3d/examples/show_annotations.py):
a translation marked `inwards` moves along -motion_dir, one marked `outwards` along +motion_dir. "docs" (the SceneFun3D
documentation, adopted for C-1's extra after C-0 found the viewer reading wrong for pushed parts): motion_dir is the way
the part moves; the viewer only draws an `inwards` line outside the surface. A rotation is an unsigned axis through its
origin either way; whether the axis sign says which way the part opens is not documented.

Everything here (part ids, motions, laser scan, registered cameras) is ground truth: scoring only, never an input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from affordance3d.frames import Frame
from affordance3d.geometry import MIN_DEPTH_M, project
from affordance3d.mem.grounding import MARGIN_PX, target_mask

ARROW_M = 0.2  # length of the drawn direction: the official viewer draws a translation as motion_dir / 5
MIN_ARROW_PX = 20  # C-0: the projected direction must be at least this long (working resolution)
ON_PART_M = 0.1  # an axis closer than this to the part's centre turns the part itself (knob, handle): no hinge side
SIDE_STEP_M = 0.05  # short step from the part's centre towards its hinge, projected to read the side
ANGLE_EPS_DEG = 1e-9
KINDS = ("trans", "rot")
READINGS = ("viewer", "docs")
ON_PART = "on the part"
MIRROR = {"left": "right", "right": "left", "top": "bottom", "bottom": "top", ON_PART: "left"}
AXIS_CLASSES = ("across the image", "up and down the image", "towards the camera")
SENSES = ("push", "pull")
EXPECTED_SENSE = {"tip_push": "push", "key_press": "push", "foot_push": "push", "plug_in": "push",
                  "pinch_pull": "pull", "hook_pull": "pull", "unplug": "pull"}  # translations only

Pixel = tuple[int, int]  # (u, v) = (column, row)
Segment = tuple[tuple[float, float], tuple[float, float]]


# ---------------------------------------------------------------- 3D: how one part moves

@dataclass(frozen=True)
class PartMotion:
    """How one part moves, in the laser-scan frame."""

    kind: str  # "trans" or "rot"
    direction: np.ndarray  # unit; translation: the way the part moves (viewer convention); rotation: the axis as annotated
    origin: np.ndarray  # translation: the arrow's start (a point of the part); rotation: a point on the axis
    centre: np.ndarray  # mean of the part's scan points
    viz_orient: str

    @property
    def foot(self) -> np.ndarray:
        """The point of the motion's line nearest the part's centre."""
        return axis_foot(self.origin, self.direction, self.centre)

    @property
    def anchor(self) -> np.ndarray:
        """Where the direction is drawn from: a translation's origin, or the point of the axis next to the part."""
        return self.origin if self.kind == "trans" else self.foot

    @property
    def axis_distance_m(self) -> float:
        """Distance from the part's centre to the rotation axis (0 for a translation)."""
        return 0.0 if self.kind == "trans" else float(np.linalg.norm(self.centre - self.foot))

    @property
    def hinged(self) -> bool:
        return self.kind == "rot" and self.axis_distance_m >= ON_PART_M


def unit(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("a direction of length 0")
    return vector / length


def signed_direction(motion_dir: Sequence[float], viz_orient: str) -> np.ndarray:
    """The way a translating part moves, by the official viewer's convention: `inwards` -> -motion_dir."""
    if viz_orient == "inwards":
        return -unit(motion_dir)
    if viz_orient == "outwards":
        return unit(motion_dir)
    raise ValueError(f"unknown motion_viz_orient {viz_orient!r}")


def axis_foot(origin: np.ndarray, direction: np.ndarray, point: np.ndarray) -> np.ndarray:
    """The point of the line through `origin` along unit `direction` that is nearest `point`."""
    return origin + float((point - origin) @ direction) * direction


def part_motion(motion: dict, part_points: np.ndarray, scan_xyz: np.ndarray, reading: str = "viewer") -> PartMotion:
    """One entry of `<visit>_motions.json` as a PartMotion; `part_points` are the part's scan points; `reading` is how a
    translation's sign is read (see the module docstring)."""
    kind = motion["motion_type"]
    if kind not in KINDS:
        raise ValueError(f"unknown motion_type {kind!r}")
    if reading not in READINGS:
        raise ValueError(f"unknown reading {reading!r}")
    viewer_sign = kind == "trans" and reading == "viewer"
    direction = signed_direction(motion["motion_dir"], motion["motion_viz_orient"]) if viewer_sign else unit(motion["motion_dir"])
    return PartMotion(kind, direction, np.asarray(scan_xyz[motion["motion_origin_idx"]], dtype=np.float64),
                      part_points.mean(axis=0), motion["motion_viz_orient"])


def opening_tangent(motion: PartMotion) -> np.ndarray:
    """Where the part's centre moves under a positive turn about the axis (right-hand rule); zeros if not a hinge."""
    if not motion.hinged:
        return np.zeros(3)
    return unit(np.cross(motion.direction, motion.centre - motion.foot))


# ---------------------------------------------------------------- 2D: the part in a keyframe

def projected_segment(start: np.ndarray, direction: np.ndarray, frame: Frame, length_m: float = ARROW_M) -> Segment | None:
    """Pixels of a segment drawn from `start` for `length_m` along `direction`; None if an end is behind the camera."""
    uv, z, _ = project(np.stack([start, start + length_m * direction]), frame)
    if (z <= MIN_DEPTH_M).any():
        return None
    return (float(uv[0, 0]), float(uv[0, 1])), (float(uv[1, 0]), float(uv[1, 1]))


def segment_px(segment: Segment | None) -> float:
    return 0.0 if segment is None else float(np.hypot(segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]))


def camera_centre(frame: Frame) -> np.ndarray:
    return frame.cam_to_world[:3, 3]


def push_or_pull(motion: PartMotion, frame: Frame) -> str:
    """A translation that moves towards the keyframe's camera pulls; one that moves away pushes."""
    return "pull" if float(motion.direction @ (camera_centre(frame) - motion.anchor)) > 0 else "push"


def axis_class(axis: np.ndarray, frame: Frame) -> str:
    """Which camera axis a rotation axis is closest to: across the image, up and down it, or towards the camera."""
    in_camera = frame.cam_to_world[:3, :3].T @ axis
    return AXIS_CLASSES[int(np.argmax(np.abs(in_camera)))]


def image_side(du: float, dv: float) -> str:
    """The side a 2D step points to (image u to the right, v down)."""
    if abs(du) >= abs(dv):
        return "left" if du < 0 else "right"
    return "top" if dv < 0 else "bottom"


def hinge_side(motion: PartMotion, frame: Frame) -> str | None:
    """On which side of the part, in the image, its rotation axis lies; ON_PART for a knob or a handle."""
    if motion.kind != "rot":
        return None
    if not motion.hinged:
        return ON_PART
    step = projected_segment(motion.centre, unit(motion.foot - motion.centre), frame, SIDE_STEP_M)
    if step is None:
        return None
    return image_side(step[1][0] - step[0][0], step[1][1] - step[0][1])


def contact_pixel(labels: np.ndarray, part: int) -> Pixel | None:
    """The part's pixel nearest the median of its pixels (as pipeline.contact_and_normal picks it)."""
    rows_cols = np.argwhere(labels == part)
    if not len(rows_cols):
        return None
    centre = np.median(rows_cols, axis=0)
    v, u = rows_cols[np.argmin(np.linalg.norm(rows_cols - centre, axis=1))]
    return int(u), int(v)


# ---------------------------------------------------------------- the record of one task

@dataclass(frozen=True)
class PartView:
    """One target part as seen in the task's keyframe."""

    pixels: int
    contact: Pixel | None
    arrow: Segment | None  # projected direction (translation) or axis (rotation), ARROW_M long
    sense: str | None  # translations: push or pull, relative to the keyframe's camera
    axis_class: str | None  # rotations
    hinge_side: str | None  # rotations


@dataclass(frozen=True)
class ActionRecord:
    task: str
    parts: tuple[int, ...]  # indices of the scene's elements (as in the label maps)
    annot_ids: tuple[str, ...]
    labels: tuple[str, ...]
    motions: tuple[PartMotion | None, ...]  # None: the part has no motion annotation
    keyframe: int  # the keyframe where the target parts have most pixels
    shown: int  # position in `parts` of the part with most pixels there
    views: tuple[PartView, ...]

    @property
    def has_all_motions(self) -> bool:
        return all(motion is not None for motion in self.motions)

    @property
    def arrow_px(self) -> float:
        """Length of the shown part's projected direction (0 when it has no motion or an end is behind the camera)."""
        return segment_px(self.views[self.shown].arrow)


def part_view(labels: np.ndarray, part: int, motion: PartMotion | None, frame: Frame) -> PartView:
    pixels = int((labels == part).sum())
    if motion is None:
        return PartView(pixels, contact_pixel(labels, part), None, None, None, None)
    rot = motion.kind == "rot"
    return PartView(pixels, contact_pixel(labels, part), projected_segment(motion.anchor, motion.direction, frame),
                    None if rot else push_or_pull(motion, frame), axis_class(motion.direction, frame) if rot else None,
                    hinge_side(motion, frame))


def build_record(task: str, parts: Sequence[int], annot_ids: Sequence[str], labels: Sequence[str],
                 motions: Sequence[PartMotion | None], label_maps: Sequence[np.ndarray], frames: Sequence[Frame]) -> ActionRecord | None:
    """The record of one task; None when no keyframe shows any of its parts."""
    counts = np.array([[int((lab == p).sum()) for p in parts] for lab in label_maps]).reshape(len(label_maps), len(parts))
    if not counts.size or counts.sum(axis=1).max() == 0:
        return None
    keyframe = int(np.argmax(counts.sum(axis=1)))
    return record_in(task, parts, annot_ids, labels, motions, label_maps[keyframe], frames[keyframe], keyframe=keyframe)


def record_in(task: str, parts: Sequence[int], annot_ids: Sequence[str], labels: Sequence[str], motions: Sequence[PartMotion | None],
              label_map: np.ndarray, frame: Frame, keyframe: int = 0, shown_if_hidden: int = 0) -> ActionRecord:
    """The record of one task as seen in one given frame (an arm's frame in C-1). The shown part is the one with most
    pixels there; when the frame shows none of the parts, `shown_if_hidden` (e.g. the part C-0 showed)."""
    views = tuple(part_view(label_map, p, m, frame) for p, m in zip(parts, motions, strict=True))
    pixels = [view.pixels for view in views]
    shown = int(np.argmax(pixels)) if max(pixels, default=0) > 0 else shown_if_hidden
    return ActionRecord(task, tuple(parts), tuple(annot_ids), tuple(labels), tuple(motions), keyframe, shown, views)


# ---------------------------------------------------------------- scoring a prediction against the record

@dataclass(frozen=True)
class Prediction:
    """A method's answer for one task, in the record's terms; any field may be missing."""

    point: Pixel | None
    label: str | None
    kind: str | None
    sense: str | None
    arrow: tuple[float, float] | None  # 2D direction in the keyframe (any length)
    axis_class: str | None
    hinge_side: str | None
    direction: np.ndarray | None  # 3D, laser-scan frame


def truth_as_prediction(record: ActionRecord) -> Prediction:
    """The record itself, written as a prediction of its shown part (the scorer must give it 100%)."""
    part, view, motion = record.shown, record.views[record.shown], record.motions[record.shown]
    arrow = None if view.arrow is None else (view.arrow[1][0] - view.arrow[0][0], view.arrow[1][1] - view.arrow[0][1])
    return Prediction(view.contact, record.labels[part], None if motion is None else motion.kind, view.sense, arrow,
                      view.axis_class, view.hinge_side, None if motion is None else motion.direction)


def contrary(prediction: Prediction, image_hw: tuple[int, int]) -> Prediction:
    """Every field made wrong on purpose (point to the far corner, another label and type, reversed sense and arrow,
    the next axis class, the mirrored hinge side): the scorer must not give this 100%."""
    point = None
    if prediction.point is not None:
        u, v = prediction.point
        point = (0 if u > image_hw[1] / 2 else image_hw[1] - 1, 0 if v > image_hw[0] / 2 else image_hw[0] - 1)
    arrow = None if prediction.arrow is None else (-prediction.arrow[1], prediction.arrow[0])  # perpendicular: wrong as a line too
    direction = None if prediction.direction is None else np.cross(prediction.direction, _not_parallel(prediction.direction))
    return Prediction(point, None if prediction.label is None else f"not {prediction.label}",
                      None if prediction.kind is None else KINDS[1 - KINDS.index(prediction.kind)],
                      None if prediction.sense is None else SENSES[1 - SENSES.index(prediction.sense)], arrow,
                      None if prediction.axis_class is None else AXIS_CLASSES[(AXIS_CLASSES.index(prediction.axis_class) + 1) % 3],
                      None if prediction.hinge_side is None else MIRROR[prediction.hinge_side], direction)


def _not_parallel(vector: np.ndarray) -> np.ndarray:
    return np.eye(3)[int(np.argmin(np.abs(vector)))]


def angle_deg(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, as_lines: bool) -> float:
    """Angle between two 2D or 3D directions; as lines (sign ignored) it is at most 90 degrees.

    atan2 of the cross and dot products, not arccos of the dot: equal directions give exactly 0 (arccos of a dot
    product rounded to 1 - 1e-16 gives about 1e-6 degrees).
    """
    a, b = unit(a), unit(b)
    cross = abs(a[0] * b[1] - a[1] * b[0]) if len(a) == 2 else float(np.linalg.norm(np.cross(a, b)))
    dot = float(a @ b)
    return float(np.degrees(np.arctan2(cross, abs(dot) if as_lines else dot)))


def reference_part(prediction: Prediction, record: ActionRecord, labels: np.ndarray) -> int:
    """The target part a prediction is about: the one under its point (nearest, if several are within the margin),
    otherwise the part the keyframe shows best."""
    if prediction.point is None:
        return record.shown
    u, v = prediction.point
    best, best_distance = record.shown, np.inf
    for position, part in enumerate(record.parts):
        rows_cols = np.argwhere(labels == part)
        if not len(rows_cols):
            continue
        # Chebyshev distance, the reach of the square widening in grounding.target_mask
        distance = float(np.min(np.maximum(np.abs(rows_cols[:, 1] - u), np.abs(rows_cols[:, 0] - v))))
        if distance <= MARGIN_PX and distance < best_distance:
            best, best_distance = position, distance
    return best


def on_target(point: Pixel | None, record: ActionRecord, labels: np.ndarray) -> bool:
    """The contact rule of the controls: the point lies on a target part, widened by MARGIN_PX."""
    if point is None:
        return False
    height, width = labels.shape
    u, v = point
    return bool(0 <= v < height and 0 <= u < width and target_mask(labels, record.parts, MARGIN_PX)[v, u])


def score(prediction: Prediction, record: ActionRecord, labels: np.ndarray, axis_only: bool = False,
          angle_tol_deg: float = 0.0) -> dict[str, bool | float | None]:
    """Field by field: True/False where the field applies, None where it does not; angles in degrees.

    contact: the point lies on a target part (label map of the record's keyframe), widened by MARGIN_PX.
    label, type: of the part under the point (else the shown part). sense: push or pull, translations only.
    arrow: the 2D direction within `angle_tol_deg` (signed for translations; as a line for rotations).
    axis_class, hinge_side: rotations only. direction_deg: 3D angle error, if the prediction has a 3D direction.
    `axis_only` (the rule when the sign conventions prove inconsistent): no sense, every direction compared as a line.
    joint: every applicable True/False field is True. C-1 sets its own `angle_tol_deg` before it runs.
    """
    ref = reference_part(prediction, record, labels)
    motion, view = record.motions[ref], record.views[ref]
    result: dict[str, bool | float | None] = {"contact": on_target(prediction.point, record, labels),
                                              "label": prediction.label == record.labels[ref]}
    result.update(_motion_fields(prediction, motion, view, axis_only, angle_tol_deg))
    result["joint"] = all(value for key, value in result.items() if isinstance(value, bool))
    return result


def _motion_fields(prediction: Prediction, motion: PartMotion | None, view: PartView, axis_only: bool,
                   angle_tol_deg: float) -> dict[str, bool | float | None]:
    fields: dict[str, bool | float | None] = dict.fromkeys(("type", "sense", "arrow", "arrow_deg", "axis_class", "hinge_side",
                                                           "direction_deg"))
    if motion is None:
        return fields
    rot = motion.kind == "rot"
    as_lines = rot or axis_only
    fields["type"] = prediction.kind == motion.kind
    if not rot and not axis_only:
        fields["sense"] = prediction.sense == view.sense
    if segment_px(view.arrow) > 0:  # the field applies only where the true direction shows in the image
        truth = (view.arrow[1][0] - view.arrow[0][0], view.arrow[1][1] - view.arrow[0][1])
        has_arrow = prediction.arrow is not None and any(prediction.arrow)
        fields["arrow_deg"] = angle_deg(prediction.arrow, truth, as_lines) if has_arrow else None
        fields["arrow"] = has_arrow and fields["arrow_deg"] <= angle_tol_deg + ANGLE_EPS_DEG
    if rot:
        fields["axis_class"] = prediction.axis_class == view.axis_class
        fields["hinge_side"] = None if view.hinge_side is None else prediction.hinge_side == view.hinge_side
    if prediction.direction is not None:
        fields["direction_deg"] = angle_deg(prediction.direction, motion.direction, as_lines)
    return fields


def shares(scores: Sequence[dict[str, bool | float | None]]) -> dict[str, dict[str, float | int]]:
    """Per True/False field: the share of True over the tasks where it applies, and how many those are."""
    out = {}
    for key in scores[0] if scores else ():
        values = [s[key] for s in scores if isinstance(s[key], bool)]
        if values:
            out[key] = {"share": float(np.mean(values)), "n": len(values)}
    return out
