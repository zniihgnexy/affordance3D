"""Game-style interaction overlay for the week-2 walkthrough video (made by scripts/make_week2_media.py).

Every memory part visible in a frame is highlighted where it is: its annotated 3D points rendered with the frame's
depth (so a part behind a wall stays hidden), filled in a translucent colour and outlined. An arrow at its contact
point slides back and forth the way the memory says to move it, and a label names the object (from the part's task
text) and the action (from its affordance label). Colours mark the action category. Settings: configs/report/
week2_media.yaml, walkthrough.interaction. Display only: masks and names come from the annotation, vectors from the
part-vector run.

How each part's arrow moves (one back-and-forth every period_s seconds of video):
  translation, push category (push, press, step on, plug in)  straight arrow into the surface, ending at the contact
  translation, pull category (pull, unplug)                    straight arrow out of the surface, starting at the contact
  rotation about an axis off the part (door, window, lid)      straight arrow along the opening direction, out of the surface
  rotation about an axis through the part (knob, lever)        curved arrow round the axis, swinging back and forth
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from affordance3d.frames import Frame
from affordance3d.geometry import project
from affordance3d.mem import pipeline as P

NEAR_M = 0.05  # a point closer to the camera than this is not drawn
CUTS = (" located", " next to", " near", " close to", " of the", " of a", " in the", " on the", " at the", " above",
        " below", " under", " beside", " behind", " between", " to the", " from", " that", " which", " with", ",")
LEADS = tuple(re.compile(p, re.IGNORECASE) for p in (  # stripped from the front of a task's object, repeatedly
    r"^(on|off|in|out|up|down)\s+",  # "turn on the light"
    r"^(the|a|an)\s+",
    r"^(one|any|either|each) of (the\s+)?",  # "one of the sockets"
    r"^(device|plug|cable|charger)\s+(in|into)\s+",  # "plug the device in the power supply": the socket is the object
))
USING = re.compile(r"\susing\s+", re.IGNORECASE)  # "turn on the TV using the remote control": the remote is the object


@dataclass(frozen=True)
class Style:
    colours: dict[str, tuple[int, int, int]]  # action category -> RGB
    actions: dict[str, tuple[str, str]]  # affordance label -> (action name, category)
    swing: tuple[str, str]  # (action name, category) of a rotation about an axis off the part
    objects: dict[str, str]  # affordance label -> object name when the part has no task text
    swing_min_offset_m: float
    alpha: float
    grow_px: int
    min_pixels: int
    period_s: float
    arrow_m: float
    travel_m: float
    arc_radius_m: float
    arc_span_deg: float
    arc_sweep_deg: float
    line_px: int
    font_size: int
    max_words: int
    tag_lines: int  # 2: object over action; 1: "object > action" on one line (for crowded views)
    group_px: float  # parts with the same object and action whose anchors are this close share one tag (0: never)

    @classmethod
    def from_config(cls, c: dict) -> Style:
        return cls({k: tuple(v) for k, v in c["colours"].items()}, {k: tuple(v) for k, v in c["actions"].items()},
                   tuple(c["swing"]), dict(c["objects"]), c["swing_min_offset_m"], c["alpha"], c["grow_px"], c["min_pixels"], c["period_s"],
                   c["arrow_m"], c["travel_m"], c["arc_radius_m"], c["arc_span_deg"], c["arc_sweep_deg"], c["line_px"],
                   c["font_size"], c["max_words"], c["tag_lines"], c["group_px"])


# ---------------------------------------------------------------- pure helpers (tested)

def object_name(task: str, fallback: str, max_words: int = 3) -> str:
    """The object a task names: "Open the left cabinet door of the TV stand" -> "left cabinet door"."""
    text = task.strip().rstrip(".")
    parts = text.split(" ", 1)
    if len(parts) < 2:
        return fallback
    rest = _strip_leads(parts[1])
    using = USING.search(rest)
    rest = _strip_leads(rest[using.end():]) if using else rest
    low = rest.lower()
    cut = min((i for i in (low.find(c) for c in CUTS) if i > 0), default=len(rest))
    name = re.sub(r"'s\b.*$", "", rest[:cut]).split()[-max_words:]
    while name and name[0].lower() in ("the", "a", "an"):
        name = name[1:]
    return " ".join(name) if name else fallback


def _strip_leads(text: str) -> str:
    while found := next((m for m in (lead.match(text) for lead in LEADS) if m), None):
        text = text[found.end():]
    return text


def is_swing(kind: str | None, contact: np.ndarray, origin: np.ndarray, axis: np.ndarray | None, min_offset_m: float) -> bool:
    """A rotation whose axis passes at least `min_offset_m` from the contact (a door or window), not a twist in place."""
    if kind != "rot" or axis is None:
        return False
    offset = contact - origin
    return float(np.linalg.norm(offset - (offset @ axis) * axis)) >= min_offset_m


def action_of(label: str, swing: bool, style: Style) -> tuple[str, str]:
    """(action name, category) of a part."""
    if swing:
        return style.swing
    return style.actions.get(label, (label.replace("_", " "), "push"))


def phase(frame_index: int, fps: float, period_s: float) -> float:
    """0 -> 1 -> 0 once every `period_s` seconds of video, smoothly (cosine)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * frame_index / fps / period_s)


def toward(direction: np.ndarray, normal: np.ndarray, outward: bool) -> np.ndarray:
    """`direction` signed to point out of (outward) or into the surface whose normal is `normal`."""
    side = float(direction @ normal)
    return direction if (side >= 0) == outward else -direction


def straight_arrow(contact: np.ndarray, direction: np.ndarray, t: float, length_m: float, travel_m: float,
                   inward: bool) -> tuple[np.ndarray, np.ndarray]:
    """3D (tail, head) at phase t: a push slides in and ends at the contact, a pull starts at it and slides out."""
    if inward:
        head = contact - (1 - t) * travel_m * direction
        return head - length_m * direction, head
    tail = contact + t * travel_m * direction
    return tail, tail + length_m * direction


def arc(centre: np.ndarray, axis: np.ndarray, reference: np.ndarray, radius_m: float, start_deg: float, span_deg: float,
        n: int = 24) -> np.ndarray:
    """Points (n, 3) of a circular arc round `axis` through `centre`, starting at `reference` turned by `start_deg`."""
    u = reference - (reference @ axis) * axis
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.radians(start_deg + np.linspace(0.0, span_deg, n))
    return centre + radius_m * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))


# ---------------------------------------------------------------- drawing

def facing(normal: np.ndarray | None, contact: np.ndarray, frame: Frame) -> np.ndarray:
    """The surface normal turned toward this frame's camera (the side one sees)."""
    to_camera = frame.cam_to_world[:3, 3] - contact
    n = normal if normal is not None else to_camera / np.linalg.norm(to_camera)
    return n if n @ to_camera >= 0 else -n


def glyph(part, swing: bool, category: str, t: float, frame: Frame, style: Style) -> np.ndarray:
    """3D polyline of the part's arrow at phase t as seen from `frame` (last segment carries the arrowhead)."""
    return glyph_at(part, swing, category, t, facing(part.normal, part.contact, frame), style)


def glyph_at(part, swing: bool, category: str, t: float, normal: np.ndarray, style: Style) -> np.ndarray:
    """3D polyline of the part's arrow at phase t, for the surface side `normal` faces (last segment: arrowhead)."""
    if part.kind == "rot" and part.axis is not None and not swing:
        start = -style.arc_span_deg / 2 + (t - 0.5) * style.arc_sweep_deg
        return arc(part.contact, part.axis, normal, style.arc_radius_m, start, style.arc_span_deg)
    if swing:
        lever = part.contact - part.origin
        direction = toward(np.cross(part.axis, lever - (lever @ part.axis) * part.axis), normal, outward=True)
    else:
        direction = toward(part.axis if part.axis is not None else normal, normal, outward=category != "push")
    direction = direction / np.linalg.norm(direction)
    tail, head = straight_arrow(part.contact, direction, t, style.arrow_m, style.travel_m, inward=category == "push")
    return np.stack([tail, head])


def draw_polyline_arrow(image_bgr: np.ndarray, frame: Frame, points: np.ndarray, colour: tuple, width: int) -> None:
    uv, z, _ = project(points, frame)
    if (z <= NEAR_M).any():
        return
    pts = np.rint(uv).astype(np.int32)
    bgr = tuple(int(c) for c in colour[::-1])
    base = pts[max(0, len(pts) - 5)]  # an arc's head points along its last few segments
    span = max(1.0, float(np.linalg.norm(pts[-1] - base)))
    tip = min(0.9, max(0.2, 22.0 / span))  # about 22 px of arrowhead whatever the arrow's length on screen
    for thickness, ink in ((width + 4, (20, 20, 20)), (width, bgr)):  # dark rim first, so it reads on any background
        if len(pts) > 2:
            cv2.polylines(image_bgr, [pts], False, ink, thickness, cv2.LINE_AA)
        cv2.arrowedLine(image_bgr, tuple(base), tuple(pts[-1]), ink, thickness, cv2.LINE_AA, tipLength=tip)


def highlight(image_bgr: np.ndarray, mask: np.ndarray, colour: tuple, alpha: float, grow_px: int) -> None:
    """Translucent fill and a bright outline over the (grown) mask."""
    if grow_px > 0:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)) > 0
    bgr = np.array(colour[::-1], dtype=np.float32)
    image_bgr[mask] = (image_bgr[mask] * (1 - alpha) + bgr * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image_bgr, contours, -1, tuple(int(c) for c in bgr), 2, cv2.LINE_AA)
    cv2.drawContours(image_bgr, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)


@lru_cache(maxsize=8)
def fonts(size: int) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    bold = font_manager.findfont("DejaVu Sans:bold", fallback_to_default=True)
    regular = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    return ImageFont.truetype(bold, size), ImageFont.truetype(regular, size - 1)


Box = tuple[float, float, float, float]  # left, top, right, bottom (pixels)


def overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def place_tag(anchor: tuple[float, float], w: float, h: float, blocked: Sequence[Box], image_wh: tuple[int, int],
              gap: float = 26.0, margin: float = 4.0) -> Box:
    """A w x h tag box near `anchor`, inside the image and clear of `blocked` (parts, arrows, earlier tags) when it can be:
    tries above-right, above-left, below-right, below-left, right, left, then two and three times as far; else the
    in-image spot overlapping least."""
    x, y = anchor
    width, height = image_wh
    sides = ((-1, 1), (-1, -1), (1, 1), (1, -1), (0, 1), (0, -1))  # (down, right) signs
    candidates = [(x + reach * gap if dx > 0 else x - reach * gap - w,
                   y - h / 2 if dy == 0 else y + reach * gap if dy > 0 else y - reach * gap - h)
                  for reach in (1, 2, 3) for dy, dx in sides]
    boxes = [(min(max(left, margin), width - w - margin), min(max(top, margin), height - h - margin)) for left, top in candidates]
    boxes = [(left, top, left + w, top + h) for left, top in boxes]
    return min(boxes, key=lambda box: (sum(overlap(box, b) for b in blocked), boxes.index(box)))


def group_tags(items: Sequence[tuple], radius_px: float) -> list[tuple]:
    """Tags of parts with the same object and action whose anchors lie within `radius_px` of a group's first anchor
    merged into one: mean anchor, "name ×n", the union of their keep-out boxes."""
    groups: list[list[tuple]] = []
    for item in items:
        home = next((g for g in groups if g[0][1:4] == item[1:4] and np.hypot(g[0][0][0] - item[0][0], g[0][0][1] - item[0][1])
                     <= radius_px), None)
        if home is None:
            groups.append([item])
        else:
            home.append(item)
    merged = []
    for g in groups:
        anchor = (float(np.mean([i[0][0] for i in g])), float(np.mean([i[0][1] for i in g])))
        boxes = np.array([i[4] for i in g])
        name = g[0][1] if len(g) == 1 else f"{g[0][1]} \u00d7{len(g)}"
        merged.append((anchor, name, g[0][2], g[0][3], (*boxes[:, :2].min(axis=0), *boxes[:, 2:].max(axis=0))))
    return merged


def draw_labels(canvas_rgba: Image.Image, items: Sequence[tuple[tuple[float, float], str, str, tuple, Box]], size: int,
                lines: int = 2) -> None:
    """Rounded game-UI tags (object on top, action below in its colour; or both on one line), each linked to its part
    by a thin line and placed clear of every part, arrow and earlier tag when possible."""
    bold, regular = fonts(size)
    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    blocked = [item[4] for item in items]
    for (x, y), name, action, colour, _ in sorted(items, key=lambda item: item[0][1]):
        line2 = f"\u25b6 {action}"
        gap = 6 if lines == 1 else 0
        name_w, action_w = draw.textlength(name, font=bold), draw.textlength(line2, font=regular)
        w = int(name_w + gap + action_w if lines == 1 else max(name_w, action_w)) + 18
        h = 2 * size + 12 if lines == 2 else size + 10
        left, top, right, bottom = place_tag((x, y), w, h, blocked, canvas_rgba.size)
        blocked.append((left, top, right, bottom))
        edge = (min(max(x, left), right), min(max(y, top), bottom))  # nearest point of the tag to the anchor
        draw.line([(x, y), edge], fill=(*colour, 230), width=2)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(*colour, 255), outline=(255, 255, 255, 255))
        draw.rounded_rectangle([left, top, right, bottom], radius=7, fill=(18, 20, 24, 205), outline=(*colour, 255), width=2)
        draw.rounded_rectangle([left, top, left + 5, bottom], radius=3, fill=(*colour, 255))
        draw.text((left + 11, top + 4), name, font=bold, fill=(255, 255, 255, 255))
        second = (left + 11 + name_w + gap, top + 5) if lines == 1 else (left + 11, top + size + 7)
        draw.text(second, line2, font=regular, fill=(*colour, 255))
    canvas_rgba.alpha_composite(overlay)


def bands(image_rgb: np.ndarray, header: str, style: Style, band_px: int = 30) -> np.ndarray:
    """Header band (time, parts in view) and a legend band with one colour chip per action category; even size (H.264)."""
    height, width = image_rgb.shape[:2]
    canvas = Image.new("RGB", (width, height + 2 * band_px), (24, 24, 24))
    canvas.paste(Image.fromarray(image_rgb), (0, band_px))
    draw = ImageDraw.Draw(canvas)
    _, regular = fonts(style.font_size)
    draw.text((8, 7), header, font=regular, fill=(235, 235, 235))
    x, y = 8, height + band_px + 7
    names = {category: [] for category in style.colours}
    for name, category in [*style.actions.values(), style.swing]:
        if name not in names[category]:
            names[category].append(name)
    for category, colour in style.colours.items():
        text = " / ".join(names[category][:3])
        draw.rounded_rectangle([x, y + 1, x + 14, y + 15], radius=3, fill=colour)
        draw.text((x + 19, y), text, font=regular, fill=(235, 235, 235))
        x += 19 + int(draw.textlength(text, font=regular)) + 14
    array = np.asarray(canvas)
    return array[: array.shape[0] // 2 * 2, : width // 2 * 2]


def part_actions(parts: Sequence, style: Style) -> dict[int, tuple[str, str, bool]]:
    """Per part index: (action name, category, swing)."""
    out = {}
    for p in parts:
        swing = is_swing(p.kind, p.contact, p.origin, p.axis, style.swing_min_offset_m)
        out[p.part] = (*action_of(p.label, swing, style), swing)
    return out


def draw_frame(frame: Frame, labels: np.ndarray, parts: Sequence, actions: dict, t: float, width: int, style: Style,
               header: str) -> tuple[np.ndarray, int]:
    """One video frame (RGB, with bands) and how many parts were drawn on it."""
    image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    scale, tags = width / image.shape[1], []
    for part in parts:
        mask = labels == part.part
        if int(mask.sum()) < style.min_pixels:
            continue
        name, category, swing = actions[part.part]
        colour = style.colours[category]
        highlight(image, mask, colour, style.alpha * (0.7 + 0.3 * t), style.grow_px)  # a gentle pulse with the arrow
        draw_polyline_arrow(image, frame, glyph(part, swing, category, t, frame, style), colour, style.line_px)
        rows, cols = np.nonzero(mask)
        sweep = np.vstack([glyph(part, swing, category, end, frame, style) for end in (0.0, 1.0)])
        uv, z, _ = project(sweep, frame)
        uv = uv[z > NEAR_M] if (z > NEAR_M).any() else np.array([[cols.mean(), rows.mean()]])
        keep_out = (min(cols.min(), uv[:, 0].min()) * scale - 6, min(rows.min(), uv[:, 1].min()) * scale - 6,
                    max(cols.max(), uv[:, 0].max()) * scale + 6, max(rows.max(), uv[:, 1].max()) * scale + 6)
        tags.append(((float(cols.mean()) * scale, float(rows.mean()) * scale),
                     object_name(part.task, style.objects.get(part.label, part.label.replace("_", " ")), style.max_words),
                     name, colour, keep_out))
    small = cv2.resize(image, (width, int(np.rint(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw_labels(pil, group_tags(tags, style.group_px) if style.group_px > 0 else tags, style.font_size, style.tag_lines)
    return bands(np.asarray(pil.convert("RGB")), header.format(n=len(tags)), style), len(tags)


def render_walkthrough(scene: P.Scene, parts: Sequence, out_path: Path, width: int, fps: int, style: Style, writer_for,
                       chunk: int = 40) -> dict:
    """Every frame of the video that has a registered camera and depth, with each visible part highlighted, its arrow
    moving and its tag; `writer_for(path, hw, fps)` opens the H.264 writer."""
    times = scene.gt_index.depth_times
    actions = part_actions(parts, style)
    writer, counts = None, []
    for start in range(0, len(times), chunk):
        frames = P.gt_frames(scene, times[start:start + chunk])
        for time, frame, labels in zip(times[start:start + chunk], frames, P.label_maps(scene, frames)):
            header = f"t = {float(time) - float(times[0]):5.1f} s    interactable parts in view: {{n}}"
            canvas, n = draw_frame(frame, labels, parts, actions, phase(len(counts), fps, style.period_s), width, style, header)
            counts.append(n)
            writer = writer or writer_for(out_path, canvas.shape[:2], fps)
            writer.send(np.ascontiguousarray(canvas))
    writer.close()
    return {"frames": len(times), "parts_in_view_median": float(np.median(counts)),
            "frames_with_parts": int(np.sum(np.array(counts) > 0)), "style": "interaction"}
