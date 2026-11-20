"""Videos of the week-2 report (report/week2_2026-11-16_to_11-22.ipynb), made by scripts/make_week2_media.py.

Both videos show the part-vector memory of one scene in the same game-style overlay (report/week2_interaction.py: each
visible part highlighted in its action colour, an arrow sliding the way to move it, a tag with object and action):
1. walkthrough: every frame of the scene's video, so that one sees the parts and arrows stay put while the camera
   moves (the memory was written once, from 40 keyframes);
2. orbit: a camera circling above the scene's laser scan (ceiling cut away); a part is drawn where its scan points
   are visible in that view (depth test against the drawn scan).

Display only. The walkthrough uses the dataset's scan-registered cameras and rendered depth (hires poses, hires depth),
so that a drawing sits where its part is and parts behind a wall are hidden. The vectors themselves come from a
part-vector run (scripts/run_part_vectors.py), which used the device's own poses and LiDAR depth; part masks and names
come from the annotation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from affordance3d.frames import Frame
from affordance3d.mem import action_records as A
from affordance3d.mem import pipeline as P
from affordance3d.mem.labels import part_label_map
from report import week2_interaction as G

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
    part: int = -1  # index of the scene's element (as in the label maps)
    normal: np.ndarray | None = None
    task: str = ""  # a task text that names the part


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
        normal = None if vector.get("normal") is None else np.array(vector["normal"])
        parts.append(DrawnPart(row["label"], vector["kind"], np.array(vector["contact"]), axis, np.array(vector["origin"]),
                               truth.kind, truth.direction, truth.anchor, row["part"], normal, row.get("task", "")))
    return parts


def video_writer(path: Path, hw: tuple[int, int], fps: int):
    """An H.264 writer of RGB frames of size `hw` (height, width), primed and ready for send()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(path), (hw[1], hw[0]), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=2, output_params=["-crf", "28", "-movflags", "+faststart"])
    writer.send(None)
    return writer


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
                 depth_tol_m: float, style: G.Style) -> dict:
    """A full circle above the scan, looking at the parts' centre, in the walkthrough's game style: a part is drawn where
    its scan points pass the depth test against the drawn scan (`depth_tol_m`), with its arrow and tag."""
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
    actions = G.part_actions(parts, style)
    part_points = [scene.part_points(p) for p in range(len(scene.elements))]
    header = f"laser scan from above, ceiling removed    interactable parts in view: {{n}} of {len(parts)}"
    writer, counts = None, []
    for k in range(n_frames):
        angle = 2 * np.pi * k / n_frames
        eye = target + radius * (np.cos(angle) * across + np.sin(angle) * side) + radius * np.tan(np.radians(elevation_deg)) * up
        pose = look_at(eye, target, up)
        image, depth = splat(xyz, rgb, intrinsics, pose, hw)
        frame = Frame(k, image, depth.astype(np.float32), intrinsics, pose)
        labels = part_label_map(part_points, frame, depth_tol_m)
        canvas, n = G.draw_frame(frame, labels, parts, actions, G.phase(k, fps, style.period_s), hw[1], style, header)
        counts.append(n)
        writer = writer or video_writer(out_path, canvas.shape[:2], fps)
        writer.send(np.ascontiguousarray(canvas))
    writer.close()
    return {"frames": n_frames, "points": len(xyz), "radius_m": round(radius, 2), "parts_drawn_median": float(np.median(counts)),
            "style": "interaction"}
