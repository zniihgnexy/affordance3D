"""One EPIC-KITCHENS video for the report: VISOR masks + contacts, narrations, EPIC-Fields poses.

VISOR annotations and narrations are ground truth used for display only. EPIC-Fields poses and
points come from structure-from-motion: the scale is arbitrary and there is no measured depth.
"""

from __future__ import annotations

import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation

from affordance3d.viz import Z_UP_VIEW, points_trace, scene_figure
from report.report_helpers import MapProjection, outlined_text

NOT_A_CONTACT = {"inconclusive", "hand-not-in-contact", "none-of-the-above"}
HAND_COLOURS = {"left hand": (66, 133, 244), "right hand": (255, 140, 0)}


@dataclass(frozen=True)
class Mask:
    name: str
    polygons: tuple[np.ndarray, ...]  # (K, 2) pixel polygons at the original 1920x1080
    contact: str | None  # hands only: name of the touched object, or a VISOR placeholder


@dataclass(frozen=True)
class VisorFrame:
    frame: int  # EPIC-KITCHENS frame number
    path: Path | None
    subsequence: str
    masks: tuple[Mask, ...]

    @property
    def contacts(self) -> tuple[tuple[str, str], ...]:
        return tuple((m.name, m.contact) for m in self.masks if m.contact and m.contact not in NOT_A_CONTACT)


@dataclass(frozen=True)
class Narration:
    text: str
    start: int
    stop: int


@dataclass(frozen=True)
class FieldsReconstruction:
    intrinsics: np.ndarray  # 3x3 at `image_wh`
    image_wh: tuple[int, int]
    poses: dict[int, np.ndarray]  # frame -> 4x4 camera-to-world
    points: np.ndarray  # (N, 3) sparse SfM points
    colours: np.ndarray  # (N, 3) in [0, 1]


# --------------------------------------------------------------------------- reading


def read_visor(annotation_path: Path, frames_dir: Path, mapping_path: Path, video: str) -> list[VisorFrame]:
    mapping = json.loads(mapping_path.read_text())[video]  # VISOR image name -> EPIC frame name
    frames = []
    for entry in json.loads(annotation_path.read_text())["video_annotations"]:
        names = {a["id"]: a["name"] for a in entry["annotations"]}
        masks = tuple(Mask(a["name"], tuple(np.asarray(s, dtype=np.float32) for s in a["segments"] if len(s) >= 3),
                           names.get(a["in_contact_object"], a["in_contact_object"]) if "in_contact_object" in a
                           else None) for a in entry["annotations"])
        name = entry["image"]["name"]
        frame = int(mapping[name].split("_")[1].split(".")[0])
        frames.append(VisorFrame(frame, frames_dir / name, entry["image"]["subsequence"], masks))
    return sorted(frames, key=lambda f: f.frame)


def _seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def frames_per_second(rows: list[dict]) -> float:
    rates = [int(r["start_frame"]) / _seconds(r["start_timestamp"]) for r in rows if _seconds(r["start_timestamp"]) > 5]
    return float(np.round(np.median(rates), 1))


def read_narrations(csv_path: Path, video: str) -> tuple[list[Narration], float]:
    with csv_path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["video_id"] == video]
    narrations = [Narration(r["narration"], int(r["start_frame"]), int(r["stop_frame"])) for r in rows]
    return sorted(narrations, key=lambda n: n.start), frames_per_second(rows)


def narration_at(narrations: list[Narration], frame: int) -> str:
    """The latest-started action still running at `frame`, else the most recently finished one."""
    running = [n for n in narrations if n.start <= frame <= n.stop]
    if running:
        return max(running, key=lambda n: n.start).text
    finished = [n for n in narrations if n.stop < frame]
    return max(finished, key=lambda n: n.stop).text if finished else ""


def colmap_to_cam_to_world(qvec_tvec: list[float]) -> np.ndarray:
    """COLMAP image pose (qw, qx, qy, qz, tx, ty, tz; world-to-camera) as a 4x4 camera-to-world."""
    qw, qx, qy, qz, *translation = qvec_tvec
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -rotation.T @ np.asarray(translation)
    return pose


def read_epic_fields(path: Path) -> FieldsReconstruction:
    data = json.loads(path.read_text())
    camera = data["camera"]
    fx, fy, cx, cy = camera["params"][:4]
    poses = {int(name.split("_")[1].split(".")[0]): colmap_to_cam_to_world(values)
             for name, values in data["images"].items()}
    points = np.asarray(data["points"], dtype=np.float64)
    return FieldsReconstruction(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), (camera["width"], camera["height"]),
                                poses, points[:, :3], (points[:, 3:6] / 255).astype(np.float32))


def gravity_alignment(poses: list[np.ndarray]) -> np.ndarray:
    """Rotation taking world "up" to +z for a head-mounted camera.

    The head moves roughly in a horizontal plane, so "up" is the direction in which the camera
    centres vary least (robust to the 5 % most extreme positions per axis). The mean image-down
    axis only picks the sign: it is biased whenever the wearer keeps looking down at a counter.
    """
    centres = np.stack([pose[:3, 3] for pose in poses])
    low, high = np.percentile(centres, [5, 95], axis=0)
    inliers = centres[((centres >= low) & (centres <= high)).all(axis=1)]
    _, axes = np.linalg.eigh(np.cov((inliers - inliers.mean(axis=0)).T))
    up = axes[:, 0]
    if up @ np.mean([pose[:3, 1] for pose in poses], axis=0) > 0:
        up = -up
    rotation, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])
    return rotation.as_matrix()


# --------------------------------------------------------------------------- tables and drawing


def overview_table(frames: list[VisorFrame], narrations: list[Narration], recon: FieldsReconstruction,
                   fps: float) -> pd.DataFrame:
    last_frame = max(max(recon.poses), max(n.stop for n in narrations))
    minutes, seconds = divmod(last_frame / fps, 60)
    contact_frames = sum(bool(f.contacts) for f in frames)
    names = {m.name for f in frames for m in f.masks}
    return pd.DataFrame({"value": {
        "video": f"P28_101 · {int(minutes)} min {seconds:.0f} s · {fps:.0f} fps · head-mounted, 1920×1080",
        "narrated actions (EPIC-KITCHENS-100)": f"{len(narrations)} ({len({n.text for n in narrations})} distinct)",
        "VISOR annotated frames": f"{len(frames)} in {len({f.subsequence for f in frames})} short subsequences",
        "VISOR masks": f"{sum(len(f.masks) for f in frames):,} masks, {len(names)} hand / object classes",
        "frames with a hand–object contact": f"{contact_frames}",
        "EPIC-Fields camera poses": f"{len(recon.poses):,} frames ({len(recon.poses) / last_frame:.1%})",
        "EPIC-Fields 3D points": f"{len(recon.points):,} (SfM, arbitrary scale, no depth maps)",
    }})


def object_colours(frames: list[VisorFrame]) -> dict[str, tuple[int, int, int]]:
    names = sorted({m.name for f in frames for m in f.masks} - set(HAND_COLOURS))
    palette = plt.get_cmap("tab20")
    return {**HAND_COLOURS, **{n: tuple(int(c * 255) for c in palette(i % 20)[:3]) for i, n in enumerate(names)}}


def draw_masks(image: np.ndarray, frame: VisorFrame, scale: float, colours: dict, labels: bool = True,
               only: set[str] | None = None) -> np.ndarray:
    """Masks filled at 45 % opacity; objects a hand touches get a white outline."""
    touched = {obj for _, obj in frame.contacts}
    shown = [m for m in frame.masks if only is None or m.name in only]
    overlay = image.copy()
    for mask in shown:
        polygons = [np.rint(p * scale).astype(np.int32) for p in mask.polygons]
        cv2.fillPoly(overlay, polygons, colours[mask.name])
    out = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    for mask in shown:
        polygons = [np.rint(p * scale).astype(np.int32) for p in mask.polygons]
        if not polygons:
            continue
        touched_now = mask.name in touched
        cv2.polylines(out, polygons, True, (255, 255, 255) if touched_now else colours[mask.name],
                      3 if touched_now else 1, cv2.LINE_AA)
        if labels:
            top = min(polygons, key=lambda p: p[:, 1].min())
            x, y = int(top[:, 0].min()), max(14, int(top[:, 1].min()) - 4)
            outlined_text(out, mask.name.replace("hand", "").strip() or mask.name, (x, y), colours[mask.name])
    return out


def _clock(frame: int, fps: float) -> str:
    minutes, seconds = divmod(int(frame / fps), 60)
    return f"{minutes:02d}:{seconds:02d}"


# --------------------------------------------------------------------------- video, 3D, examples


@dataclass(frozen=True)
class AlignedScene:
    """EPIC-Fields reconstruction rotated so that +z is up; points above head height or far outliers dropped."""

    points: np.ndarray
    colours: np.ndarray
    centres: dict[int, np.ndarray]
    forwards: dict[int, np.ndarray]

    @classmethod
    def from_recon(cls, recon: FieldsReconstruction) -> AlignedScene:
        align = gravity_alignment(list(recon.poses.values()))
        centres = {f: align @ p[:3, 3] for f, p in recon.poses.items()}
        points = recon.points @ align.T
        heads = np.percentile([c[2] for c in centres.values()], 90)
        low, high = np.percentile(points, [2, 98], axis=0)
        keep = (points[:, 2] < heads) & ((points >= low) & (points <= high)).all(axis=1)
        return cls(points[keep], recon.colours[keep], centres,  # no ceiling and no far SfM outliers
                   {f: align @ p[:3, 2] for f, p in recon.poses.items()})

    def path(self, step: int = 10) -> tuple[np.ndarray, np.ndarray]:
        frames = np.array(sorted(self.centres))[::step]
        return frames, np.stack([self.centres[f] for f in frames])


def _map_background(projection: MapProjection, xy: np.ndarray, path_px: np.ndarray) -> np.ndarray:
    size = projection.size
    counts = np.zeros((size, size))
    px = np.clip(projection.to_px(xy), 0, size - 1)
    np.add.at(counts, (px[:, 1], px[:, 0]), 1)
    shade = 255 - np.log1p(counts) / np.log1p(counts).max() * 200
    base = np.repeat(shade[..., None], 3, axis=2).astype(np.uint8)
    cv2.polylines(base, [path_px.reshape(-1, 1, 2)], False, (170, 190, 230), 1, cv2.LINE_AA)
    return base


def render_visor_walkthrough(frames: list[VisorFrame], narrations: list[Narration], scene: AlignedScene,
                             fps_source: float, out_path: Path, fps: int = 6, width: int = 640,
                             map_size: int = 360, header_px: int = 56) -> None:
    """MP4: VISOR frame with masks + contacts | top-down EPIC-Fields map with camera and past contacts."""
    colours = object_colours(frames)
    height = width * 9 // 16
    path_frames, path_xyz = scene.path()
    projection = MapProjection.fit(scene.points[:, :2], size=map_size, margin=14)
    path_px = projection.to_px(path_xyz[:, :2])
    base = _map_background(projection, scene.points[:, :2], path_px)
    canvas_hw = (height + header_px, 1024)
    writer = imageio_ffmpeg.write_frames(str(out_path), canvas_hw[::-1], fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", macro_block_size=16,
                                         output_params=["-crf", "30", "-movflags", "+faststart"])
    writer.send(None)
    remembered: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for vf in frames:
        image = cv2.resize(iio.imread(vf.path), (width, height), interpolation=cv2.INTER_AREA)
        image = draw_masks(image, vf, width / 1920, colours)
        panel = base.copy()
        cv2.polylines(panel, [path_px[path_frames <= vf.frame].reshape(-1, 1, 2)], False, (30, 60, 160), 2,
                      cv2.LINE_AA)
        for xy, colour in remembered:
            cv2.circle(panel, tuple(int(v) for v in xy), 5, colour, -1)
        if vf.frame in scene.centres:
            tip = projection.to_px(scene.centres[vf.frame][:2])[0]
            direction = scene.forwards[vf.frame][:2] / (np.linalg.norm(scene.forwards[vf.frame][:2]) + 1e-9)
            end = tip + np.rint(40 * np.array([direction[0], -direction[1]])).astype(int)
            cv2.line(panel, tuple(int(v) for v in tip), tuple(int(v) for v in end), (220, 30, 30), 2, cv2.LINE_AA)
            cv2.circle(panel, tuple(int(v) for v in tip), 6, (220, 30, 30), -1)
            remembered += [(tip, colours[obj]) for _, obj in vf.contacts]
        canvas = np.zeros((*canvas_hw, 3), dtype=np.uint8)
        canvas[header_px:, :width] = image
        canvas[header_px + (height - map_size) // 2:header_px + (height - map_size) // 2 + map_size,
               width + 12:width + 12 + map_size] = panel
        contacts = ", ".join(f"{hand} -> {obj}" for hand, obj in vf.contacts) or "none"
        cv2.putText(canvas, f"EPIC-KITCHENS P28_101   t = {_clock(vf.frame, fps_source)}   narration: "
                    f"\"{narration_at(narrations, vf.frame)}\"", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"VISOR: {len(vf.masks)} masks   contact: {contacts}      map: camera path, dots = "
                    "past contacts", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        writer.send(np.ascontiguousarray(canvas))
    writer.close()


def epic_3d_figure(scene: AlignedScene, frames: list[VisorFrame], fps_source: float,
                   top_objects: int = 8) -> go.Figure:
    colours = object_colours(frames)
    _, path_xyz = scene.path()
    traces = [points_trace(scene.points, scene.colours, name="EPIC-Fields points (below head height)", size=1.5,
                           max_points=40_000),
              go.Scatter3d(x=path_xyz[:, 0], y=path_xyz[:, 1], z=path_xyz[:, 2], mode="lines",
                           line={"color": "black", "width": 4}, name="camera path")]
    by_object: dict[str, list[tuple[np.ndarray, str]]] = {}
    for vf in frames:
        for hand, obj in vf.contacts:
            if vf.frame in scene.centres:
                label = f"{_clock(vf.frame, fps_source)} {hand} → {obj}"
                by_object.setdefault(obj, []).append((scene.centres[vf.frame], label))
    ranked = sorted(by_object.items(), key=lambda kv: -len(kv[1]))
    groups = [(f"contact: {obj}", items, "rgb({},{},{})".format(*colours[obj])) for obj, items in ranked[:top_objects]]
    others = [item for _, items in ranked[top_objects:] for item in items]
    groups += [(f"contact: {len(ranked) - top_objects} other objects", others, "grey")] if others else []
    for name, items, colour in groups:
        xyz = np.stack([p for p, _ in items])
        traces.append(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name=name,
                                   text=[t for _, t in items], hoverinfo="text", marker={"size": 6, "color": colour}))
    view = {**Z_UP_VIEW, "eye": {"x": 1.5, "y": 1.5, "z": 1.3}}
    return scene_figure(traces, "EPIC-Fields reconstruction of P28_101 · camera path · where each hand contact "
                                "happened (camera position)", height=640, view=view)


def contact_examples_figure(frames: list[VisorFrame], narrations: list[Narration], objects: list[str],
                            fps_source: float, per_object: int = 5) -> plt.Figure:
    colours = object_colours(frames)
    fig, axes = plt.subplots(len(objects), per_object, figsize=(3.3 * per_object, 2.55 * len(objects)),
                             squeeze=False)
    for row, obj in enumerate(objects):
        hits = [f for f in frames if any(o == obj for _, o in f.contacts)]
        chosen = [hits[i] for i in np.unique(np.linspace(0, len(hits) - 1, per_object).round().astype(int))] if hits else []
        for col in range(per_object):
            axis = axes[row, col]
            axis.set_axis_off()
            if col >= len(chosen):
                continue
            vf = chosen[col]
            image = cv2.resize(iio.imread(vf.path), (480, 270), interpolation=cv2.INTER_AREA)
            axis.imshow(draw_masks(image, vf, 480 / 1920, colours, labels=False, only={obj, *HAND_COLOURS}))
            axis.set_title(f"{_clock(vf.frame, fps_source)} · {textwrap.shorten(narration_at(narrations, vf.frame), 26)}",
                           fontsize=9)
        axes[row, 0].text(-0.04, 0.5, f"{obj}\n({len(hits)} contact frames)", transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=10, fontweight="bold")
    fig.suptitle("Query: object → every earlier hand contact with it (VISOR ground truth, in time order)")
    fig.tight_layout()
    return fig
