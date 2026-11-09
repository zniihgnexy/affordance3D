"""Figures and media for the dataset report notebook in this folder.

Only local data and earlier run outputs are read. SceneFun3D functional annotations are used
for ORACLE display and statistics only, never as input to a model or memory.
"""

from __future__ import annotations

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

from affordance3d.geometry import project, visible_mask
from affordance3d.scenefun3d import FunctionalElement, SequenceIndex, load_frames
from affordance3d.viz import Z_UP_VIEW, points_trace, scene_figure

# --------------------------------------------------------------------------- candidate landscape

CAPABILITIES = ("Continuous video", "Depth + poses", "3D geometry", "Functional-part labels",
                "Language", "Observed interaction", "State change", "Robot actions")
MARKS = {"Y": "✓", "~": "~", "-": "–"}  # provided / partial, derived or proxy / not provided
MARK_STYLE = {"✓": "background-color:#2e7d32;color:white", "~": "background-color:#f9c74f;color:#333",
              "–": "background-color:#eeeeee;color:#999"}


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str  # "real" or "sim"
    marks: str  # one of "Y~-" per capability, in CAPABILITIES order
    role: str
    status: str


# Condensed from docs/outputs/pipeline_v3_research_202611/*.xlsx and the 2026-11-05 English guide.
CANDIDATES = (
    Candidate("SceneFun3D", "real", "YYYYY---", "E0 ✓ · E1 · E2 (12 scenes)", "1 scene local · E0 pass"),
    Candidate("EPIC-KITCHENS + VISOR (+2HANDS)", "real", "Y~~~YY~-", "real-video history retrieval",
              "1 video local (P28_101)"),
    Candidate("HOI4D", "real", "YYY--YY-", "interaction phases + geometry", "survey"),
    Candidate("3RScan", "real", "YYY---Y-", "real rescans: change / stale memory", "survey"),
    Candidate("Aria Digital Twin", "real", "YYY--~Y-", "dynamic objects, revisits", "survey"),
    Candidate("UMD Part Affordance", "real", "-~-Y----", "backup: local affordance head", "survey"),
    Candidate("ContactPose", "real", "-YY~-Y--", "aux: observed-contact check", "survey"),
    Candidate("ManiSkill / SAPIEN", "sim", "YYY~-YYY", "E3: controlled occupancy / state", "survey"),
    Candidate("AI2-THOR", "sim", "YYY~-~Y~", "E3 alt.: room revisits, object states", "survey"),
)


def capability_table(candidates: tuple[Candidate, ...] = CANDIDATES) -> pd.io.formats.style.Styler:
    rows = {c.name: {"type": c.kind, **{cap: MARKS[m] for cap, m in zip(CAPABILITIES, c.marks, strict=True)},
                     "role in plan": c.role, "local": c.status} for c in candidates}
    table = pd.DataFrame.from_dict(rows, orient="index")
    return (table.style.map(lambda v: MARK_STYLE.get(v, ""), subset=list(CAPABILITIES))
            .set_properties(subset=list(CAPABILITIES), **{"text-align": "center", "font-weight": "bold"})
            .set_properties(subset=["role in plan", "local"], **{"text-align": "left"}))


# --------------------------------------------------------------------------- images


def _load_rgb(image: Path | np.ndarray) -> np.ndarray:
    array = iio.imread(image) if isinstance(image, Path) else image
    return array[..., :3] if array.ndim == 3 else array


def show_row(items: list[tuple[Path | np.ndarray, str]], width: float = 15.0, title: str | None = None,
             fontsize: int = 10) -> plt.Figure:
    """Images side by side at a common height (panel widths follow the image aspect ratios)."""
    images = [_load_rgb(image) for image, _ in items]
    ratios = [im.shape[1] / im.shape[0] for im in images]
    fig, axes = plt.subplots(1, len(images), figsize=(width, width / sum(ratios) + 0.5),
                             gridspec_kw={"width_ratios": ratios}, squeeze=False)
    for axis, image, (_, caption) in zip(axes[0], images, items):
        axis.imshow(image)
        axis.set_title(caption, fontsize=fontsize)
        axis.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=fontsize + 2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- SceneFun3D overview

def split_counts(file_lists: Path) -> pd.DataFrame:
    scenes = {s: set((file_lists / f"{s}_scenes.txt").read_text().split()) for s in ("train", "val", "test")}
    train_val = pd.read_csv(file_lists / "train_val_set.csv", dtype=str)
    test = pd.read_csv(file_lists / "test_set.csv", dtype=str)
    videos = {"train": train_val[train_val.visit_id.isin(scenes["train"])],
              "val": train_val[train_val.visit_id.isin(scenes["val"])], "test": test}
    return pd.DataFrame({"scenes (visits)": {s: len(v) for s, v in scenes.items()},
                         "videos": {s: len(v) for s, v in videos.items()}})


def label_colours(elements: list[FunctionalElement]) -> dict[str, tuple[float, float, float]]:
    """Same colours as notebooks/02 (tab10 over the sorted labels)."""
    palette = plt.get_cmap("tab10")
    return {label: palette(i)[:3] for i, label in enumerate(sorted({e.label for e in elements}))}


def camera_path(index: SequenceIndex, times: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Camera centres and viewing directions (world) at the given frame times."""
    poses = np.stack([index.poses[index.matched[t][0]] for t in times])
    return poses[:, :3, 3], poses[:, :3, 2]


def scene_3d_figure(display_xyz: np.ndarray, display_rgb: np.ndarray, elements: list[FunctionalElement],
                    scan_xyz: np.ndarray, colours: dict, centres: np.ndarray) -> go.Figure:
    traces = [points_trace(display_xyz, display_rgb, name="laser scan (ceiling cut)", size=1.2, max_points=40_000)]
    for label, colour in colours.items():
        points = np.concatenate([scan_xyz[e.indices] for e in elements if e.label == label])
        css = "rgb({},{},{})".format(*(int(c * 255) for c in colour))
        traces.append(points_trace(points, css, name=label, size=3, max_points=3_000))
    traces.append(go.Scatter3d(x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], mode="lines",
                               line={"color": "black", "width": 5}, name="camera path"))
    return scene_figure(traces, "SceneFun3D 421069 · laser scan + functional parts + camera path of video 42444490",
                        height=640, view=Z_UP_VIEW)


# --------------------------------------------------------------------------- walkthrough video


@dataclass(frozen=True)
class MapProjection:
    """Top-down (world xy) to image pixels; north up, aspect preserved, content centred."""

    origin: np.ndarray
    scale: float  # pixels per metre
    size: int
    margin: int

    @classmethod
    def fit(cls, xy: np.ndarray, size: int, margin: int) -> MapProjection:
        low, extent = xy.min(axis=0), np.ptp(xy, axis=0)
        span = float(extent.max())
        return cls(low - (span - extent) / 2, (size - 2 * margin) / span, size, margin)

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        offset = (np.atleast_2d(xy) - self.origin) * self.scale
        u = self.margin + offset[:, 0]
        v = self.size - self.margin - offset[:, 1]
        return np.rint(np.stack([u, v], axis=1)).astype(int)


@dataclass(frozen=True)
class WalkthroughLog:
    times: list[str]
    visible_points: np.ndarray  # (frames, elements) visible annotated points of each element's subsample


def _rgb255(colour: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(c * 255) for c in colour)


def outlined_text(image: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int],
                   scale: float = 0.42) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _map_base(projection: MapProjection, display_xy: np.ndarray, elements_xy: list[tuple[np.ndarray, tuple]],
              path_px: np.ndarray, colours: dict) -> np.ndarray:
    size = projection.size
    counts = np.zeros((size, size))
    px = np.clip(projection.to_px(display_xy), 0, size - 1)
    np.add.at(counts, (px[:, 1], px[:, 0]), 1)
    shade = 255 - (np.log1p(counts) / np.log1p(counts).max() * 190)
    base = np.repeat(shade[..., None], 3, axis=2).astype(np.uint8)
    cv2.polylines(base, [path_px.reshape(-1, 1, 2)], False, (170, 190, 230), 1, cv2.LINE_AA)
    for xy, colour in elements_xy:
        for u, v in projection.to_px(xy):
            cv2.circle(base, (int(u), int(v)), 2, _rgb255(colour), -1)
    for row, (label, colour) in enumerate(colours.items()):
        cv2.circle(base, (12, size - 12 - 14 * row), 4, _rgb255(colour), -1)
        cv2.putText(base, label, (22, size - 8 - 14 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1,
                    cv2.LINE_AA)
    return base


def _draw_camera(panel: np.ndarray, projection: MapProjection, centre: np.ndarray, forward: np.ndarray,
                 half_fov: float, path_px: np.ndarray) -> None:
    cv2.polylines(panel, [path_px.reshape(-1, 1, 2)], False, (30, 60, 160), 2, cv2.LINE_AA)
    heading = np.arctan2(forward[1], forward[0])
    tip = projection.to_px(centre[:2])[0]
    for angle in (heading - half_fov, heading + half_fov):
        end = projection.to_px(centre[:2] + 1.2 * np.array([np.cos(angle), np.sin(angle)]))[0]
        cv2.line(panel, tuple(int(v) for v in tip), tuple(int(v) for v in end), (220, 30, 30), 2, cv2.LINE_AA)
    cv2.circle(panel, tuple(int(v) for v in tip), 6, (220, 30, 30), -1)


def _depth_panel(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    coloured = (plt.get_cmap("turbo")(np.clip(depth_m / max_depth_m, 0, 1))[..., :3] * 255).astype(np.uint8)
    coloured[depth_m <= 0] = 0
    return coloured


def render_walkthrough(index: SequenceIndex, times: list[str], cw_turns: int, pixel_step: int,
                       elements: list[FunctionalElement], scan_xyz: np.ndarray, display_xy: np.ndarray,
                       out_path: Path, fps: int = 12, depth_tol: float = 0.05, min_visible: int = 10,
                       points_per_element: int = 300, header_px: int = 32) -> WalkthroughLog:
    """MP4: RGB with ORACLE functional parts | rendered depth | top-down map with the current camera."""
    rng = np.random.default_rng(0)
    element_points = [scan_xyz[rng.permutation(e.indices)[:points_per_element]] for e in elements]
    colours = label_colours(elements)
    centres, forwards = camera_path(index, times)
    first = load_frames(index, times[:1], cw_turns, pixel_step)[0]
    height, width = first.image_hw
    projection = MapProjection.fit(np.vstack([display_xy, centres[:, :2]]), size=height, margin=24)
    path_px = projection.to_px(centres[:, :2])
    base = _map_base(projection, display_xy, [(p[:, :2], colours[e.label]) for p, e in zip(element_points, elements)],
                     path_px, colours)
    half_fov = float(np.arctan2(width / 2, first.intrinsics[0, 0]))
    canvas_hw = (height + header_px, 2 * width + height)
    visible_points = np.zeros((len(times), len(elements)), dtype=int)
    writer = imageio_ffmpeg.write_frames(str(out_path), canvas_hw[::-1], fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", macro_block_size=16,
                                         output_params=["-crf", "28", "-movflags", "+faststart"])
    writer.send(None)
    for k, time in enumerate(times):
        frame = load_frames(index, [time], cw_turns, pixel_step, first_position=k)[0]
        rgb = frame.rgb.copy()
        for j, (points, element) in enumerate(zip(element_points, elements)):
            visible = visible_mask(points, frame, depth_tol)
            visible_points[k, j] = int(visible.sum())
            if visible_points[k, j] < min_visible:
                continue
            uv = np.rint(project(points[visible], frame)[0]).astype(int)
            for u, v in uv:
                cv2.circle(rgb, (int(u), int(v)), 2, _rgb255(colours[element.label]), -1)
            outlined_text(rgb, element.label, (int(uv[:, 0].min()), max(12, int(uv[:, 1].min()) - 6)),
                           _rgb255(colours[element.label]))
        panel = base.copy()
        _draw_camera(panel, projection, centres[k], forwards[k], half_fov, path_px[: k + 1])
        canvas = np.zeros((*canvas_hw, 3), dtype=np.uint8)
        canvas[header_px:, :width] = rgb
        canvas[header_px:, width:2 * width] = _depth_panel(frame.depth_m, max_depth_m=5.0)
        canvas[header_px:, 2 * width:] = panel
        n_seen = int((visible_points[k] >= min_visible).sum())
        header = (f"SceneFun3D 421069 / 42444490   t = {float(time) - float(times[0]):5.1f} s   "
                  f"ORACLE functional parts in view: {n_seen}      RGB | depth 0-5 m | top-down map")
        cv2.putText(canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.send(np.ascontiguousarray(canvas))
    writer.close()
    return WalkthroughLog(list(times), visible_points)


# --------------------------------------------------------------------------- tasks and part sizes


def read_descriptions(path: Path) -> list[tuple[str, list[str]]]:
    return [(d["description"], list(d["annot_id"])) for d in json.loads(path.read_text())["descriptions"]]


def pick_task_examples(descriptions: list[tuple[str, list[str]]], elements: list[FunctionalElement],
                       seen: np.ndarray, n: int) -> list[tuple[str, int]]:
    """(description, element index) pairs: one per functional type first, only parts seen in the video."""
    position = {e.annot_id: j for j, e in enumerate(elements)}
    candidates = [(text, position[ids[0]]) for text, ids in descriptions
                  if ids and ids[0] in position and seen[position[ids[0]]]]
    chosen, used_labels = [], set()
    for text, j in candidates:
        if elements[j].label not in used_labels:
            chosen.append((text, j))
            used_labels.add(elements[j].label)
    chosen += [c for c in candidates if c not in chosen]
    return chosen[:n]


def task_examples_figure(examples: list[tuple[str, int]], elements: list[FunctionalElement], log: WalkthroughLog,
                         index: SequenceIndex, cw_turns: int, scan_xyz: np.ndarray, pixel_step: int = 2,
                         crop_half: int = 120, depth_tol: float = 0.05) -> plt.Figure:
    colours = label_colours(elements)
    fig, axes = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 4.6), squeeze=False)
    for axis, (text, j) in zip(axes[0], examples):
        k = int(np.argmax(log.visible_points[:, j]))
        frame = load_frames(index, [log.times[k]], cw_turns, pixel_step)[0]
        points = scan_xyz[elements[j].indices]
        uv = project(points[visible_mask(points, frame, depth_tol)], frame)[0]
        height, width = frame.image_hw
        u0, v0 = (np.clip(np.median(uv, axis=0).round().astype(int) - crop_half, 0,
                          [width - 2 * crop_half, height - 2 * crop_half]))
        axis.imshow(frame.rgb[v0:v0 + 2 * crop_half, u0:u0 + 2 * crop_half])
        axis.scatter(uv[:, 0] - u0, uv[:, 1] - v0, s=4, color=colours[elements[j].label], alpha=0.7)
        axis.set_title(f"“{textwrap.fill(text, 28)}”\n→ {elements[j].label}", fontsize=9)
        axis.set_axis_off()
    fig.suptitle("SceneFun3D language task → annotated functional part (ORACLE, best view in the video)")
    fig.tight_layout()
    return fig
