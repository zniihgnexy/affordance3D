"""Videos of the week-2 report (report/2026-11-20_week2_summary.ipynb), made by scripts/make_week2_media.py.

1. walkthrough: the part-vector memory of one scene drawn on every frame of its video, so that one sees the vectors
   stay on their parts while the camera moves (the memory was written once, from 40 keyframes);
2. orbit: a camera circling above the scene's laser scan (ceiling cut away) with every part's vector.

Display only. The frames use the dataset's scan-registered cameras and rendered depth (hires poses, hires depth), so
that a drawing sits where its vector is and parts behind a wall are hidden. The vectors themselves come from a
part-vector run (scripts/run_part_vectors.py), which used the device's own poses and LiDAR depth.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from affordance3d.frames import Frame
from affordance3d.geometry import project
from affordance3d.mem import action_records as A
from affordance3d.mem import pipeline as P

PRED = (213, 94, 0)  # Okabe-Ito vermilion (RGB): computed by rules
TRUE = (0, 158, 115)  # Okabe-Ito green (RGB): annotation
NEAR_M = 0.05  # a point closer to the camera than this is not drawn


@dataclass(frozen=True)
class DrawnPart:
    label: str
    kind: str | None
    contact: np.ndarray
    axis: np.ndarray | None
    origin: np.ndarray
    true_kind: str
    true_direction: np.ndarray
    true_anchor: np.ndarray


def load_parts(run_dir: Path, variant: Sequence[str], truths: dict[int, A.PartMotion]) -> list[DrawnPart]:
    """The variant's vectors of a one-scene part-vector run, with the annotated motion of each part."""
    rows = json.loads((run_dir / "parts.json").read_text())
    depth, rule, frames = variant
    parts = []
    for row in rows:
        if (row["depth"], row["rule"], row["frames"]) != (depth, rule, frames) or row["vector"] is None:
            continue
        vector, truth = row["vector"], truths[row["part"]]
        axis = None if vector["axis"] is None else np.array(vector["axis"])
        parts.append(DrawnPart(row["label"], vector["kind"], np.array(vector["contact"]), axis, np.array(vector["origin"]),
                               truth.kind, truth.direction, truth.anchor))
    return parts


def segment(start: np.ndarray, direction: np.ndarray, kind: str | None, length: float) -> np.ndarray:
    """3D ends of a drawn vector: a rotation axis as a line through `start`, a translation as an arrow from it."""
    if kind == "rot":
        return np.stack([start - length * direction, start + length * direction])
    return np.stack([start, start + length * direction])


def _line(image: np.ndarray, frame: Frame, ends: np.ndarray, colour: tuple, width: int, arrow: bool) -> None:
    uv, z, _ = project(ends, frame)
    if (z <= NEAR_M).any():
        return
    a, b = (tuple(int(np.rint(c)) for c in p) for p in uv)
    bgr = colour[::-1]
    if arrow:
        cv2.arrowedLine(image, a, b, bgr, width, cv2.LINE_AA, tipLength=0.25)
    else:
        cv2.line(image, a, b, bgr, width, cv2.LINE_AA)


def visible(frame: Frame, point: np.ndarray, tolerance_m: float) -> tuple[int, int] | None:
    """Pixel of `point` if it is in the image and not hidden behind the frame's depth (0 = no surface: hidden)."""
    uv, z, inside = project(point[None], frame)
    if not inside[0] or z[0] <= NEAR_M:
        return None
    u, v = (int(np.rint(c)) for c in uv[0])
    surface = float(frame.depth_m[v, u])
    if surface <= 0 or z[0] > surface + tolerance_m:
        return None
    return u, v


def draw_parts(image_bgr: np.ndarray, frame: Frame, parts: Sequence[DrawnPart], tolerance_m: float, length_m: float,
               width: int = 3, labels: bool = True) -> int:
    """Draw every visible part (truth green, then prediction vermilion) onto a BGR image of the frame's size; returns how
    many were drawn."""
    drawn = 0
    for part in parts:
        pixel = visible(frame, part.contact, tolerance_m)
        if pixel is None:
            continue
        drawn += 1
        _line(image_bgr, frame, segment(part.true_anchor, part.true_direction, part.true_kind, length_m), TRUE, width,
              part.true_kind != "rot")
        if part.axis is not None:
            _line(image_bgr, frame, segment(part.origin, part.axis, part.kind, length_m), PRED, width, part.kind != "rot")
        cv2.circle(image_bgr, pixel, width + 2, PRED[::-1], -1, cv2.LINE_AA)
        cv2.circle(image_bgr, pixel, width + 2, (255, 255, 255), 1, cv2.LINE_AA)
        if labels:
            put_text(image_bgr, part.label, (pixel[0] + 8, pixel[1] - 8), 0.5)
    return drawn


def put_text(image_bgr: np.ndarray, text: str, origin: tuple[int, int], scale: float) -> None:
    """White text on a dark box."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(image_bgr, (x - 2, y - h - 3), (x + w + 2, y + base), (30, 30, 30), -1)
    cv2.putText(image_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def with_bands(image_bgr: np.ndarray, header: str, legend: str, band_px: int = 28) -> np.ndarray:
    """The image between a header band and a legend band; height and width made even (H.264)."""
    height, width = image_bgr.shape[:2]
    canvas = np.full((height + 2 * band_px, width, 3), 24, dtype=np.uint8)
    canvas[band_px:band_px + height] = image_bgr
    for text, y in ((header, band_px - 9), (legend, height + 2 * band_px - 9)):
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas[: canvas.shape[0] // 2 * 2, : width // 2 * 2]


def _writer(path: Path, hw: tuple[int, int], fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(path), (hw[1], hw[0]), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=2, output_params=["-crf", "28", "-movflags", "+faststart"])
    writer.send(None)
    return writer


def render_walkthrough(scene: P.Scene, parts: Sequence[DrawnPart], out_path: Path, width: int, fps: int, tolerance_m: float,
                       length_m: float, legend: str, chunk: int = 40) -> dict:
    """Every frame of the video that has a registered camera and depth, with the memory drawn on it."""
    times = scene.gt_index.depth_times
    writer, counts = None, []
    for start in range(0, len(times), chunk):
        for time, frame in zip(times[start:start + chunk], P.gt_frames(scene, times[start:start + chunk])):
            image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            counts.append(draw_parts(image, frame, parts, tolerance_m, length_m))
            scale = width / image.shape[1]
            image = cv2.resize(image, (width, int(np.rint(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            header = f"t = {float(time) - float(times[0]):5.1f} s   memory parts in view: {counts[-1]}"
            canvas = with_bands(image, header, legend)
            writer = writer or _writer(out_path, canvas.shape[:2], fps)
            writer.send(np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
    writer.close()
    return {"frames": len(times), "parts_in_view_median": float(np.median(counts)), "frames_with_parts": int(np.sum(np.array(counts) > 0))}


# ---------------------------------------------------------------- orbit around the laser scan

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world pose (x right, y down, z forward) of a camera at `eye` looking at `target`."""
    forward = A.unit(target - eye)
    right = A.unit(np.cross(forward, up))
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    pose[:3, 3] = eye
    return pose


def splat(xyz: np.ndarray, rgb: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray, hw: tuple[int, int],
          background: int = 245) -> tuple[np.ndarray, np.ndarray]:
    """Points drawn one pixel each with a depth buffer: (RGB image, depth in metres, 0 where nothing was drawn)."""
    height, width = hw
    world_to_cam = np.linalg.inv(cam_to_world)
    cam = xyz @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    front = cam[:, 2] > NEAR_M
    cam, colours = cam[front], rgb[front]
    u = np.rint(intrinsics[0, 0] * cam[:, 0] / cam[:, 2] + intrinsics[0, 2]).astype(int)
    v = np.rint(intrinsics[1, 1] * cam[:, 1] / cam[:, 2] + intrinsics[1, 2]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    index, z, colours = v[inside] * width + u[inside], cam[inside, 2], colours[inside]
    depth = np.full(height * width, np.inf)
    np.minimum.at(depth, index, z)
    nearest = z <= depth[index]
    image = np.full((height * width, 3), background, dtype=np.uint8)
    image[index[nearest]] = colours[nearest]
    depth[np.isinf(depth)] = 0
    return image.reshape(height, width, 3), depth.reshape(height, width)


def orbit_points(scene: P.Scene, n_points: int, cut_above_floor_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Cropped scan without its ceiling (points higher than `cut_above_floor_m` over the floor), subsampled; floor height."""
    keep = scene.crop.astype(bool) if scene.crop.dtype == bool or scene.crop.max() <= 1 else np.ones(len(scene.scan_xyz), bool)
    xyz, rgb = scene.scan_xyz[keep], scene.scan_rgb[keep]
    height = xyz @ A.unit(scene.up)
    floor = float(np.percentile(height, 2))
    below = height <= floor + cut_above_floor_m
    xyz, rgb = xyz[below], rgb[below]
    chosen = np.random.default_rng(0).permutation(len(xyz))[:n_points]
    return xyz[chosen], (np.clip(rgb[chosen], 0, 1) * 255).astype(np.uint8), floor


def render_orbit(scene: P.Scene, parts: Sequence[DrawnPart], out_path: Path, hw: tuple[int, int], n_frames: int, fps: int,
                 n_points: int, cut_above_floor_m: float, elevation_deg: float, radius_scale: float, fov_deg: float,
                 length_m: float, legend: str) -> dict:
    """A full circle above the scan, looking at the parts' centre, with every part's vector (hidden ones left out)."""
    xyz, rgb, floor = orbit_points(scene, n_points, cut_above_floor_m)
    up = A.unit(scene.up)
    contacts = np.array([p.contact for p in parts])
    target = contacts.mean(axis=0)
    target = target + (floor + 0.9 - target @ up) * up
    across = A.unit(np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0]))
    side = np.cross(up, across)
    flat = xyz - np.outer(xyz @ up, up)
    radius = radius_scale * float(np.percentile(np.linalg.norm(flat - (target - (target @ up) * up), axis=1), 95))
    focal = hw[1] / 2 / np.tan(np.radians(fov_deg) / 2)
    intrinsics = np.array([[focal, 0, hw[1] / 2], [0, focal, hw[0] / 2], [0, 0, 1.0]])
    writer, counts = None, []
    for k in range(n_frames):
        angle = 2 * np.pi * k / n_frames
        eye = target + radius * (np.cos(angle) * across + np.sin(angle) * side) + radius * np.tan(np.radians(elevation_deg)) * up
        pose = look_at(eye, target, up)
        image, depth = splat(xyz, rgb, intrinsics, pose, hw)
        frame = Frame(k, image, depth.astype(np.float32), intrinsics, pose)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        counts.append(draw_parts(bgr, frame, parts, tolerance_m=0.08, length_m=length_m, width=3, labels=False))
        canvas = with_bands(bgr, f"laser scan, ceiling removed   parts drawn: {counts[-1]} of {len(parts)}", legend)
        writer = writer or _writer(out_path, canvas.shape[:2], fps)
        writer.send(np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
    writer.close()
    return {"frames": n_frames, "points": len(xyz), "radius_m": round(radius, 2), "parts_drawn_median": float(np.median(counts))}
