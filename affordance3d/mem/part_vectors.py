"""Per-part affordance vectors from rules (proposal of 2026-11-18, step W5): where to touch a part and how it moves,
as 3D quantities in world coordinates, from a part mask, its affordance label, depth and the camera pose. No model.

A vector holds the contact point, the normal of the surface around the part, the motion type ("trans" or "rot"), the
direction of a translation or the axis of a rotation, and a point on that axis (the origin). Three rule sets:

- ``c1``: the rules of C-1's G and M arms (action_arms.rule_motion): pressed parts move into the surface, pulled parts
  out of it, a knob (rotate) turns about the normal through the contact, a hook_turn part about the vertical; the
  origin is the contact.
- ``geometric``: c1, but a pulled part (hook_pull, pinch_pull) is put on a door or a drawer by where it sits on its
  flat surface: far off the surface's centre means a door. A door, and every hook_turn part, then turns about the
  surface's far edge (the hinge): the axis lies in the surface, across the direction in which the part is most off
  centre, and the origin is that edge (a light version of REACT3D's rule).
- ``oracle_kind``: geometric's hinge rule with the annotated motion type (diagnostic: the geometry given the right type).

Scoring follows the SceneFun3D Task 3 definition in the paper, per part: +M type right, +A axis within the angle
(compared as a line, signs being unreliable, section 25.4), +O a rotation's origin within the distance of the true
axis line. Coordinates are the laser-scan (Faro) frame of affordance3d.mem.pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from affordance3d.frames import Frame
from affordance3d.geometry import backproject_pixels
from affordance3d.mem import action_records as A
from affordance3d.mem.action_arms import rule_motion
from affordance3d.mem.pipeline import MIN_PART_PIXELS, contact_and_normal

RULES = ("c1", "geometric", "oracle_kind")
FRAME_MODES = ("best", "fused")
DOOR_LABELS = ("hook_pull", "pinch_pull")  # pulled by a handle: a drawer (translation) or a door (rotation)
HINGED_LABELS = DOOR_LABELS + ("hook_turn",)


@dataclass(frozen=True)
class HingeConfig:
    plane_tol_m: float
    gap_px: int
    door_asymmetry: float


@dataclass(frozen=True)
class Observation:
    """One part in one keyframe."""

    keyframe: int
    pixel: A.Pixel
    pixels: int
    contact: np.ndarray
    normal: np.ndarray  # of the surface around the part, facing the camera


@dataclass(frozen=True)
class Hinge:
    """The far edge of the flat surface the part sits on."""

    origin: np.ndarray
    axis: np.ndarray
    asymmetry: float  # 0 = the part sits in the middle of the surface, 1 = at one end
    at_border: bool  # the surface ran out of the image before it ended


@dataclass(frozen=True)
class PartVector:
    part: int
    label: str
    kind: str | None
    contact: np.ndarray
    normal: np.ndarray
    axis: np.ndarray | None  # translation: the direction it moves; rotation: the axis
    origin: np.ndarray  # rotation: a point of the axis; translation: the contact
    keyframe: int  # the keyframe the hinge was read from (most part pixels)
    evidence: tuple[int, ...]  # every keyframe that shows the part
    hinge: Hinge | None


@dataclass(frozen=True)
class PartScore:
    axis_deg: float | None  # line angle to the true direction or axis, whatever the predicted type
    origin_m: float | None  # rotations: distance from the predicted origin to the true axis line
    m: bool
    ma: bool
    mao: bool


# ---------------------------------------------------------------- small geometry

def line_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two lines (0 to 90 degrees): the sign of either direction does not matter."""
    cosine = abs(float(A.unit(a) @ A.unit(b)))
    return float(np.degrees(np.arccos(min(cosine, 1.0))))


def point_line_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    return float(np.linalg.norm(point - A.axis_foot(origin, A.unit(direction), point)))


def in_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """`vector` with its component along `normal` removed, as a unit vector."""
    normal = A.unit(normal)
    return A.unit(vector - (vector @ normal) * normal)


# ---------------------------------------------------------------- observations and fusion

def observe(part: int, frames: Sequence[Frame], labels: Sequence[np.ndarray]) -> list[Observation]:
    """The part in every keyframe that shows it in at least MIN_PART_PIXELS pixels with depth; most pixels first."""
    seen = []
    for k, (frame, label_map) in enumerate(zip(frames, labels, strict=True)):
        pixels = int((label_map == part).sum())
        if pixels < MIN_PART_PIXELS:
            continue
        found = contact_and_normal(frame, label_map, part)
        if found is not None:
            seen.append(Observation(k, found[0], pixels, found[1], found[2]))
    return sorted(seen, key=lambda o: -o.pixels)


def fuse(observations: Sequence[Observation]) -> tuple[np.ndarray, np.ndarray]:
    """Contact = median of the frames' contacts; normal = mean of the frames' normals, each turned to agree with
    the best frame's (the camera can see a surface from one side only, so they should already agree)."""
    reference = observations[0].normal
    normals = np.array([o.normal if o.normal @ reference >= 0 else -o.normal for o in observations])
    return np.median([o.contact for o in observations], axis=0), A.unit(normals.mean(axis=0))


def spread(observations: Sequence[Observation]) -> dict[str, float | None]:
    """How much single frames disagree: median normal angle to the fused normal, median contact distance."""
    if len(observations) < 2:
        return {"normal_deg": None, "contact_m": None}
    contact, normal = fuse(observations)
    return {"normal_deg": float(np.median([line_angle_deg(o.normal, normal) for o in observations])),
            "contact_m": float(np.median([np.linalg.norm(o.contact - contact) for o in observations]))}


# ---------------------------------------------------------------- the hinge: far edge of the flat surface

def _run_ends(on: np.ndarray, start: int, gap_px: int) -> tuple[int, int]:
    """First and last index of the run of `on` through `start`, bridging holes up to `gap_px` long."""
    ends = []
    for step in (-1, 1):
        last, i, hole = start, start + step, 0
        while 0 <= i < len(on):
            if on[i]:
                last, hole = i, 0
            else:
                hole += 1
                if hole > gap_px:
                    break
            i += step
        ends.append(last)
    return ends[0], ends[1]


def _scan(frame: Frame, label_map: np.ndarray, part: int, pixel: A.Pixel, contact: np.ndarray, normal: np.ndarray,
          horizontal: bool, config: HingeConfig) -> list[tuple[np.ndarray, bool]] | None:
    """The two ends (3D) of the flat surface along the image row (or column) through the contact pixel, each with
    whether it is the image border. The part's own pixels count as surface (a handle sticks out of it)."""
    u, v = pixel
    height, width = frame.image_hw
    line = np.arange(width if horizontal else height)
    uv = np.stack([line, np.full_like(line, v)] if horizontal else [np.full_like(line, u), line], axis=1).astype(float)
    points, valid = backproject_pixels(frame, uv)
    flat = valid & (np.abs((points - contact) @ normal) <= config.plane_tol_m)
    own = (label_map[v, :] if horizontal else label_map[:, u]) == part
    first, last = _run_ends(flat | own, u if horizontal else v, config.gap_px)
    ends = [i for i in (first, last) if valid[i] and not own[i]]
    if len(ends) < 2:
        return None
    return [(points[first], first == 0), (points[last], last == len(line) - 1)]


def hinge(frame: Frame, label_map: np.ndarray, part: int, pixel: A.Pixel, contact: np.ndarray, normal: np.ndarray,
          config: HingeConfig) -> Hinge | None:
    """The far edge of the surface, along the image direction (row or column) in which the part is most off centre.

    A direction whose surface runs out of the image says nothing about where the surface ends, so a direction with
    both ends inside the image wins over one without; among equals, the more off-centre one wins."""
    candidates = []
    for horizontal in (True, False):
        found = _scan(frame, label_map, part, pixel, contact, normal, horizontal, config)
        if found is None:
            continue
        distances = [float(np.linalg.norm(end - contact)) for end, _ in found]
        if sum(distances) == 0:
            continue
        far, border = found[int(np.argmax(distances))]
        inside = not any(at_edge for _, at_edge in found)
        candidates.append((inside, abs(distances[0] - distances[1]) / sum(distances), far, border))
    if not candidates:
        return None
    _, asymmetry, edge, border = max(candidates, key=lambda c: (c[0], c[1]))
    if np.allclose(edge, contact):
        return None
    across = in_plane(edge - contact, normal)
    return Hinge(edge, A.unit(np.cross(normal, across)), float(asymmetry), bool(border))


def level_scan(frame: Frame, label_map: np.ndarray, part: int, pixel: A.Pixel, contact: np.ndarray, normal: np.ndarray,
               up: np.ndarray, config: HingeConfig) -> list[tuple[np.ndarray, bool]] | None:
    """The two ends of the flat surface along the image direction (row or column) that runs most nearly level in the
    world, each with whether it is the image border. A vertical hinge is at one of these ends (diagnostic)."""
    scans = [found for horizontal in (True, False)
             if (found := _scan(frame, label_map, part, pixel, contact, normal, horizontal, config)) is not None
             and not np.allclose(found[0][0], found[1][0])]
    if not scans:
        return None
    return min(scans, key=lambda found: abs(float(in_plane(found[1][0] - found[0][0], normal) @ A.unit(up))))


# ---------------------------------------------------------------- the vector

def rule_kind(rule: str, label: str, door: bool | None, true_kind: str | None) -> str | None:
    """Motion type under a rule set; `door` is the geometric door-or-drawer call (None when no hinge was found)."""
    if rule == "oracle_kind":
        return true_kind
    c1 = rule_motion(label, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])).kind
    if rule == "geometric" and label in DOOR_LABELS and door is not None:
        return "rot" if door else "trans"
    return c1


def part_vector(part: int, label: str, observations: Sequence[Observation], frames: Sequence[Frame],
                labels: Sequence[np.ndarray], up: np.ndarray, rule: str, frame_mode: str, config: HingeConfig,
                true_kind: str | None = None) -> PartVector:
    if rule not in RULES or frame_mode not in FRAME_MODES:
        raise ValueError(f"unknown rule {rule!r} or frame mode {frame_mode!r}")
    if not observations:
        raise ValueError(f"part {part} has no observation")
    best = observations[0]
    contact, normal = fuse(observations) if frame_mode == "fused" else (best.contact, best.normal)
    found = None
    if rule != "c1" and label in HINGED_LABELS:
        found = hinge(frames[best.keyframe], labels[best.keyframe], part, best.pixel, best.contact, best.normal, config)
    door = None if found is None else found.asymmetry >= config.door_asymmetry
    kind = rule_kind(rule, label, door, true_kind)
    evidence = tuple(sorted(o.keyframe for o in observations))
    if kind == "rot" and label in HINGED_LABELS and found is not None and rule != "c1":
        return PartVector(part, label, kind, contact, normal, found.axis, found.origin, best.keyframe, evidence, found)
    motion = rule_motion(label, normal, up)
    axis = motion.direction
    if kind != motion.kind:  # the type came from the geometric call or the annotation, not from the label
        axis = A.unit(up) if kind == "rot" and label in HINGED_LABELS else normal if kind is not None else None
    return PartVector(part, label, kind, contact, normal, axis, contact, best.keyframe, evidence, found)


def score_part(vector: PartVector, truth: A.PartMotion, axis_deg: float, origin_m: float) -> PartScore:
    """SceneFun3D Task 3 per part: +M type, +MA and axis line within `axis_deg`, +MAO and a rotation's origin within
    `origin_m` of the true axis line (a translation has no origin to score, so +MAO = +MA for it)."""
    angle = None if vector.axis is None else line_angle_deg(vector.axis, truth.direction)
    distance = point_line_distance(vector.origin, truth.origin, truth.direction) if truth.kind == "rot" else None
    m = vector.kind == truth.kind
    ma = m and angle is not None and angle <= axis_deg
    mao = ma and (distance is None or distance < origin_m)
    return PartScore(angle, distance, m, ma, mao)
