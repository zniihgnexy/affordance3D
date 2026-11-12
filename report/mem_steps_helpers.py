"""Figures and tables for notebooks/mem/pipeline_examples.ipynb: one concrete example per pipeline block.

The pipeline lives in affordance3d.mem (pipeline, tokens, search, encoders); this module only draws and scores it.
Anything that touches the laser scan, the rendered depth, the registered poses or the part ids is ground truth:
display or evaluation, never an input.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr

from affordance3d.features import PcaColoring
from affordance3d.frames import Frame
from affordance3d.mem import pipeline as P
from affordance3d.mem import search as S
from affordance3d.mem import tokens as T
from report.mem_demo_helpers import (
    ACTION_COLOUR,
    ScanView,
    _draw_entry,
    contact_error_cm,
    pick_keyframes,
    stand_free_in_scan,
)

MARK_COLOUR, CHOSEN_COLOUR, RIGHT_COLOUR = (255, 60, 60), (40, 220, 90), (60, 140, 255)
MARKS_VERSION = "marks-v2"  # part of the answer cache key: change it when the drawing of the marks changes
BOX_PAD_PX = 8
STATUS_COLOUR = {"real": "#d7f0d2", "oracle": "#ffe2b8", "missing": "#e3e3e3", "model": "#d5e6fb"}


# ---------------------------------------------------------------- overview

OVERVIEW_BOXES = (  # (x, y, text, status)
    (0.0, 3, "1 keyframes\n2 depth\n3 camera pose", "real"),
    (1.5, 3, "4 affordance map\n5 part motion", "oracle"),
    (1.5, 1.9, "14 interaction\ndetection", "missing"),
    (3.0, 3, "observation layer\nper keyframe: image, depth,\npose, time, affordance map", "real"),
    (4.6, 3, "9 part tokens\ntwo features + thin index\n(time, keyframe, pixel, depth)", "real"),
    (6.2, 3, "6 link tokens\nacross keyframes", "real"),
    (4.6, 1.9, "15 event tokens\nbefore / after an action", "missing"),
    (0.0, 0.4, "query in words", "real"),
    (1.5, 0.4, "10a keyframe search\naffordance tokens\n+ plain semantics", "real"),
    (3.0, 0.4, "10b vision-language\nmodel reads the\nmarked keyframes", "model"),
    (4.6, 0.4, "7 contact point\n8 normal, stand point\ncomputed now, not stored", "real"),
    (6.2, 0.4, "11, 12 humanoid\nwalks up and acts", "real"),
)
OVERVIEW_ARROWS = ((0, 1), (1, 3), (3, 4), (4, 5), (2, 6), (7, 8), (8, 9), (9, 10), (10, 11))


def overview_figure() -> plt.Figure:
    """The L2 pipeline as boxes: the write path (top), the read path (bottom) and what each box is today."""
    fig, axis = plt.subplots(figsize=(13.5, 5.2))
    width, height = 1.3, 0.85
    for x, y, text, status in OVERVIEW_BOXES:
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03", fc=STATUS_COLOUR[status], ec="0.35"))
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8)
    for start, end in OVERVIEW_ARROWS:
        (x0, y0, _, _), (x1, y1, _, _) = OVERVIEW_BOXES[start], OVERVIEW_BOXES[end]
        axis.annotate("", xy=(x1 - 0.04, y1 + height / 2), xytext=(x0 + width + 0.04, y0 + height / 2),
                      arrowprops={"arrowstyle": "->", "color": "0.3"})
    axis.text(3.0, 1.5, "every read step looks things up in the memory written above", fontsize=8, style="italic", color="0.3")
    axis.text(-0.05, 4.05, "WRITE  (once per video)", fontsize=10, weight="bold")
    axis.text(-0.05, 1.45, "READ  (once per query)", fontsize=10, weight="bold")
    axis.text(7.62, 3.4, "the memory:\nkeyframe maps\n+ tokens", fontsize=8, va="center")
    names = {"real": "computed from the inputs", "oracle": "ORACLE: taken from the annotations for now",
             "model": "frozen public model, not trained", "missing": "not possible on this video (nobody acts in it)"}
    handles = [plt.Line2D([], [], marker="s", ls="", ms=10, color=STATUS_COLOUR[key], mec="0.35", label=label)
               for key, label in names.items()]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=8, frameon=False)
    axis.set_xlim(-0.1, 8.4)
    axis.set_ylim(0.2, 4.3)
    axis.axis("off")
    return fig


# ---------------------------------------------------------------- blocks 1-3: inputs

def seconds_of(scene: P.Scene, times: Sequence[str]) -> np.ndarray:
    return np.array([float(t) for t in times]) - float(scene.vid.rgb_times[0])


def parts_seen(scene: P.Scene, n: int) -> int:
    """Parts seen in at least MIN_PART_PIXELS pixels of some keyframe when `n` keyframes are used (ground truth)."""
    labels = P.label_maps(scene, P.gt_frames(scene, pick_keyframes(scene, n)))
    most = np.array([[(lab == p).sum() for lab in labels] for p in range(len(scene.elements))]).max(axis=1)
    return int((most >= P.MIN_PART_PIXELS).sum())


def keyframes_figure(scene: P.Scene, keyframes: Sequence[str], frames: Sequence[Frame],
                     seen_by_count: dict[int, int]) -> plt.Figure:
    fig = plt.figure(figsize=(13.5, 7.0))
    grid = fig.add_gridspec(2, 8, height_ratios=[1.7, 1])
    path, bars = fig.add_subplot(grid[0, :5]), fig.add_subplot(grid[0, 5:])
    whole = np.array([P.input_camera(scene, t)[1][:3, 3] for t in scene.vid.rgb_times[::5]])
    chosen = np.array([f.cam_to_world[:3, 3] for f in frames])
    seconds = seconds_of(scene, keyframes)
    path.plot(whole[:, 0], whole[:, 1], color="0.7", lw=1, label=f"camera path ({len(scene.vid.rgb_times)} frames)")
    dots = path.scatter(chosen[:, 0], chosen[:, 1], c=seconds, cmap="viridis", s=38, zorder=3, label=f"{len(keyframes)} keyframes")
    fig.colorbar(dots, ax=path, label="time in the video (s)")
    path.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Where the keyframes were taken (top view, device poses)")
    path.legend(fontsize=8, loc="center right", bbox_to_anchor=(-0.22, 0.5))
    bars.bar([str(n) for n in seen_by_count], list(seen_by_count.values()), color="tab:blue")
    bars.axhline(len(scene.elements), color="0.4", ls="--")
    bars.text(-0.4, len(scene.elements) + 0.4, f"all {len(scene.elements)} annotated parts", fontsize=8, color="0.3")
    for position, value in enumerate(seen_by_count.values()):
        bars.text(position, value - 2.2, str(value), ha="center", color="white", fontsize=9)
    bars.set(xlabel="number of keyframes", ylabel=f"parts seen in ≥ {P.MIN_PART_PIXELS} pixels", title="More keyframes, more parts seen")
    for column, k in enumerate(np.linspace(0, len(frames) - 1, 8).astype(int)):
        axis = fig.add_subplot(grid[1, column])
        axis.imshow(frames[k].rgb[::3, ::3])
        axis.set_title(f"keyframe {k} · {seconds[k]:.0f} s", fontsize=8)
        axis.axis("off")
    fig.tight_layout()
    return fig


def pose_figure(scene: P.Scene, keyframes: Sequence[str], frames: Sequence[Frame], gt: Sequence[Frame]) -> plt.Figure:
    device = np.array([f.cam_to_world[:3, 3] for f in frames])
    truth = np.array([g.cam_to_world[:3, 3] for g in gt])
    error_cm = 100 * np.linalg.norm(device - truth, axis=1)
    fig, (top, bars) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    top.plot(truth[:, 0], truth[:, 1], "o-", color="0.35", ms=3, lw=0.8, label="registered to the scan (ground truth)")
    top.plot(device[:, 0], device[:, 1], "o", color="tab:orange", ms=4, label="device pose (input)")
    for a, b in zip(device, truth):
        top.plot([a[0], b[0]], [a[1], b[1]], color="tab:red", lw=1.2)
    top.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Keyframe camera positions (top view); red = error")
    top.legend(fontsize=8)
    bars.bar(seconds_of(scene, keyframes), error_cm, width=1.8, color="tab:orange")
    bars.axhline(np.median(error_cm), color="0.3", ls="--")
    bars.text(0, np.median(error_cm) + 1.5, f"median {np.median(error_cm):.1f} cm", fontsize=8)
    bars.set(xlabel="time in the video (s)", ylabel="camera position error (cm)",
             title=f"Device pose error per keyframe (largest {error_cm.max():.0f} cm)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- block 4: affordance map

def _boxes(axis: plt.Axes, regions: Sequence[T.Region], numbers: bool = True, colour: str = "white") -> None:
    for region in regions:
        left, top, right, bottom = region.box
        axis.add_patch(plt.Rectangle((left - BOX_PAD_PX, top - BOX_PAD_PX), right - left + 2 * BOX_PAD_PX,
                                     bottom - top + 2 * BOX_PAD_PX, fill=False, ec=colour, lw=1))
        if numbers:
            axis.text(left - BOX_PAD_PX, top - BOX_PAD_PX - 3, str(region.component), color=colour, fontsize=8, va="bottom")


def action_overlay(frame: Frame, actions: np.ndarray) -> np.ndarray:
    overlay = (frame.rgb * 0.5).astype(np.uint8)
    for action, name in enumerate(T.ACTIONS):
        overlay[actions == action] = (np.array(ACTION_COLOUR[name]) * 255).astype(np.uint8)
    return overlay


def affordance_figure(frames: Sequence[Frame], action_maps: Sequence[np.ndarray], regions: Sequence[Sequence[T.Region]],
                      seconds: np.ndarray, n: int = 4) -> plt.Figure:
    """The keyframes with most regions: the affordance map (colour = action type) and its regions (boxes)."""
    order = sorted(range(len(frames)), key=lambda k: -len(regions[k]))[:n]
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 5.7))
    for axis, k in zip(axes, order):
        axis.imshow(action_overlay(frames[k], action_maps[k]))
        _boxes(axis, regions[k])
        axis.set_title(f"keyframe {k} · {seconds[k]:.0f} s · {len(regions[k])} regions", fontsize=9)
        axis.axis("off")
    handles = [plt.Line2D([], [], marker="s", ls="", color=ACTION_COLOUR[a], label=a) for a in T.ACTIONS]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8, frameon=False)
    fig.suptitle("ORACLE affordance map: an action type per pixel (projected annotations, part ids dropped)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def turn_rates(scene: P.Scene, times: Sequence[str]) -> np.ndarray:
    """How fast the camera turns at each time (degrees per second), from the device poses of the frames around it."""
    every = scene.vid.rgb_times
    rates = []
    for t in times:
        at = every.index(t)
        before, after = every[max(at - 1, 0)], every[min(at + 1, len(every) - 1)]
        turn = P.input_camera(scene, before)[1][:3, :3].T @ P.input_camera(scene, after)[1][:3, :3]
        angle = np.degrees(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1.0, 1.0)))
        rates.append(angle / (float(after) - float(before)))
    return np.array(rates)


def alignment_figure(frames: Sequence[Frame], part_labels: Sequence[np.ndarray], rates: np.ndarray,
                     keyframe_ids: Sequence[int], margin: int = 50) -> plt.Figure:
    """Zoom on the labelled parts of each keyframe: the image, and the projected labels (yellow) laid over it."""
    fig, axes = plt.subplots(1, 2 * len(keyframe_ids), figsize=(6.6 * len(keyframe_ids), 4.4), squeeze=False)
    for slot, k in enumerate(keyframe_ids):
        rows, cols = np.nonzero(part_labels[k] >= 0)
        top, bottom = max(rows.min() - margin, 0), rows.max() + margin
        left, right = max(cols.min() - margin, 0), cols.max() + margin
        marked = frames[k].rgb.copy()
        marked[part_labels[k] >= 0] = (255, 255, 0)
        for axis, image, what in ((axes[0, 2 * slot], frames[k].rgb, "image"), (axes[0, 2 * slot + 1], marked, "projected labels")):
            axis.imshow(image[top:bottom, left:right])
            axis.set_title(f"keyframe {k} · camera turning {rates[k]:.0f}°/s · {what}", fontsize=9)
            axis.axis("off")
    fig.tight_layout()
    return fig


def turn_rate_figure(scene: P.Scene, explicit: P.Memory, rates_by_time: dict[str, float]) -> tuple[plt.Figure, float]:
    """Contact error of every v1 entry against the camera turn rate in its evidence keyframe, and their rank correlation."""
    rates = np.array([rates_by_time[entry.keyframe] for entry in explicit.entries])
    errors = np.array([contact_error_cm(scene, entry) for entry in explicit.entries])
    correlation = float(spearmanr(rates, errors).statistic)
    fig, axis = plt.subplots(figsize=(7.0, 3.9))
    axis.scatter(rates, errors, s=24, color="tab:blue")
    axis.axhline(3.0, color="tab:green", ls="--")
    axis.text(rates.max(), 3.25, "3 cm: counts as on the part", ha="right", fontsize=8, color="tab:green")
    axis.set(yscale="log", xlabel="camera turn rate in the evidence keyframe (°/s)", ylabel="contact error (cm)",
             title=f"{len(errors)} v1 entries · rank correlation {correlation:+.2f}")
    fig.tight_layout()
    return fig, correlation


# ---------------------------------------------------------------- block 9: part tokens

def features_figure(frame: Frame, grid: np.ndarray, colouring: PcaColoring, regions: Sequence[T.Region]) -> plt.Figure:
    """One keyframe: its regions, the self-supervised features (PCA colours), and the crops the image-text encoder sees."""
    fig = plt.figure(figsize=(13.5, 5.6))
    layout = fig.add_gridspec(2, 7)
    image, feature = fig.add_subplot(layout[:, 0:2]), fig.add_subplot(layout[:, 2:4])
    image.imshow(frame.rgb)
    colours = colouring.to_rgb(grid.reshape(-1, grid.shape[-1])).reshape(grid.shape[0], grid.shape[1], 3)
    feature.imshow(colours, extent=(0, frame.rgb.shape[1], frame.rgb.shape[0], 0), interpolation="nearest")
    for axis, title, colour in ((image, "keyframe and its affordance regions", "yellow"),
                                (feature, f"self-supervised features, {grid.shape[0]}×{grid.shape[1]} patches (PCA colours)", "white")):
        _boxes(axis, regions, colour=colour)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    for slot, region in enumerate(sorted(regions, key=lambda r: -r.pixels)[:6]):
        axis = fig.add_subplot(layout[slot // 3, 4 + slot % 3])
        x0, y0, x1, y1 = T.context_box(region.box, frame.image_hw)
        axis.imshow(frame.rgb[y0:y1, x0:x1])
        left, top, right, bottom = region.box
        axis.add_patch(plt.Rectangle((left - x0, top - y0), right - left, bottom - top, fill=False, ec="yellow", lw=1.2))
        axis.set_title(f"region {region.component}: {T.ACTIONS[region.action]}, {region.pixels} px", fontsize=8)
        axis.axis("off")
    fig.text(0.79, 0.965, "crops read by the image-text encoder", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def token_table(memory: T.L2Memory, tokens: Sequence[int]) -> pd.DataFrame:
    """What is stored for a token: the thin explicit index and the size of the two features."""
    rows = []
    for index in tokens:
        token, region = memory.tokens[index], memory.tokens[index].region
        rows.append({"token": index, "part": int(memory.part_of[index]), "time": f"{token.time_s:.1f} s",
                     "keyframe": region.keyframe, "action type": T.ACTIONS[region.action], "pixel (u, v)": region.pixel,
                     "depth": f"{region.depth_m:.2f} m", "pixels": region.pixels,
                     "visual feature": f"{token.visual.size} numbers", "semantic feature": f"{token.semantic.size} numbers"})
    return pd.DataFrame(rows).set_index("token")


# ---------------------------------------------------------------- block 6: linking

def truth_parts(memory: T.L2Memory, part_labels: Sequence[np.ndarray], n_parts: int) -> np.ndarray:
    """Ground-truth part under every token's region (evaluation only: the memory never sees part ids)."""
    truth = []
    for token in memory.tokens:
        region = token.region
        ids = part_labels[region.keyframe][memory.components[region.keyframe] == region.component]
        if not (ids >= 0).any():
            raise ValueError(f"the region of keyframe {region.keyframe} at {region.pixel} covers no labelled pixel")
        truth.append(int(np.argmax(np.bincount(ids[ids >= 0], minlength=n_parts))))
    return np.array(truth, dtype=int)


def mixed_regions(memory: T.L2Memory, part_labels: Sequence[np.ndarray]) -> int:
    """How many regions cover pixels of more than one true part (neighbours of one action type that touch)."""
    mixed = 0
    for token in memory.tokens:
        region = token.region
        ids = part_labels[region.keyframe][memory.components[region.keyframe] == region.component]
        mixed += len(np.unique(ids[ids >= 0])) > 1
    return int(mixed)


def linking_scores(memory: T.L2Memory, truth: np.ndarray) -> pd.DataFrame:
    if not len(memory.tokens):
        raise ValueError("the memory has no tokens")
    parts, seen = range(memory.n_parts), np.unique(truth)
    majority = sum(np.bincount(truth[memory.part_of == p]).max() for p in parts)
    whole = [np.bincount(memory.part_of[truth == g]).max() / (truth == g).sum() for g in seen]
    exact = sum(set(memory.tokens_of(p)) == set(np.flatnonzero(truth == truth[memory.tokens_of(p)[0]])) for p in parts)
    return pd.DataFrame([{"tokens": len(truth), "true parts with a token": len(seen), "parts in the memory": memory.n_parts,
                          "tokens whose part is not mixed": f"{majority / len(truth):.0%}",
                          "a true part stays in one piece (mean share)": f"{np.mean(whole):.0%}",
                          "parts linked exactly right": f"{exact} of {len(seen)}"}])


def _crop_axis(axis: plt.Axes, frame: Frame, region: T.Region, colour: str, title: str) -> None:
    x0, y0, x1, y1 = T.context_box(region.box, frame.image_hw)
    axis.imshow(frame.rgb[y0:y1, x0:x1])
    left, top, right, bottom = region.box
    axis.add_patch(plt.Rectangle((left - x0, top - y0), right - left, bottom - top, fill=False, ec="yellow", lw=1))
    axis.set_title(title, fontsize=7.5, color=colour)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set(color=colour, lw=2.5)


def linking_figure(memory: T.L2Memory, frames: Sequence[Frame], truth: np.ndarray, n_parts: int = 4,
                   n_views: int = 7) -> plt.Figure:
    """The parts with most tokens: every token of a part across time. Green = same true part as the majority."""
    if not len(memory.tokens):
        raise ValueError("the memory has no tokens")
    largest = sorted(range(memory.n_parts), key=lambda p: -len(memory.tokens_of(p)))[:n_parts]
    n_views = min(n_views, max(len(memory.tokens_of(p)) for p in largest))
    fig, axes = plt.subplots(n_parts, n_views, figsize=(2.6 * n_views, 2.75 * n_parts), squeeze=False)
    for row, part in enumerate(largest):
        members = memory.tokens_of(part)
        majority = np.bincount(truth[members]).argmax()
        for column in range(n_views):
            axis = axes[row, column]
            if column >= len(members):
                axis.axis("off")
                continue
            token = memory.tokens[members[column]]
            _crop_axis(axis, frames[token.region.keyframe], token.region,
                       "tab:green" if truth[members[column]] == majority else "tab:red",
                       f"part {part} · {token.time_s:.0f} s · {T.ACTIONS[token.region.action]}")
    fig.suptitle("Tokens linked into one part, in time order\n(green frame: same true part; red: a different one)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6)
    return fig


def token_pairs(memory: T.L2Memory, frames: Sequence[Frame], truth: np.ndarray) -> tuple[np.ndarray, ...]:
    """All token pairs: indices of the two tokens, whether they show the same true part, and how far apart they land."""
    positions = np.array([T.region_position(t.region, frames[t.region.keyframe]) for t in memory.tokens])
    first, second = np.triu_indices(len(truth), 1)
    return first, second, truth[first] == truth[second], np.linalg.norm(positions[first] - positions[second], axis=1)


def same_part_gap_cm(memory: T.L2Memory, frames: Sequence[Frame], truth: np.ndarray) -> float:
    """Median distance between two positions computed for the same true part from two keyframes."""
    _, _, same, distance = token_pairs(memory, frames, truth)
    return float(100 * np.median(distance[same]))


def pair_figure(memory: T.L2Memory, frames: Sequence[Frame], truth: np.ndarray) -> plt.Figure:
    """Why linking is hard: the same part seen twice does not land on the same spot, and neighbours look alike."""
    visual = np.array([t.visual for t in memory.tokens])
    times = np.array([t.time_s for t in memory.tokens])
    first, second, same, distance = token_pairs(memory, frames, truth)
    similarity = (visual[first] * visual[second]).sum(axis=1)
    near_other = ~same & (distance < T.LINK_MAX_DISTANCE_M)
    fig, (gap, look) = plt.subplots(1, 2, figsize=(13.5, 4.4))
    gap.scatter(np.abs(times[first] - times[second])[same], 100 * distance[same], s=8, alpha=0.5, color="tab:blue")
    gap.axhline(100 * T.LINK_MAX_DISTANCE_M, color="tab:red", ls="--")
    gap.text(1, 100 * T.LINK_MAX_DISTANCE_M * 1.12, f"link limit {100 * T.LINK_MAX_DISTANCE_M:.0f} cm", color="tab:red", fontsize=8)
    gap.set(yscale="log", xlabel="time between the two keyframes (s)", ylabel="distance between the two positions (cm)",
            title=f"The same true part seen twice ({int(same.sum())} pairs): median {100 * np.median(distance[same]):.0f} cm apart")
    bins = np.linspace(-0.2, 1.0, 31)
    look.hist(similarity[same], bins=bins, alpha=0.6, density=True, color="tab:blue", label="same true part")
    look.hist(similarity[near_other], bins=bins, alpha=0.6, density=True, color="tab:orange",
              label=f"different parts closer than {100 * T.LINK_MAX_DISTANCE_M:.0f} cm")
    look.axvline(T.LINK_MIN_SIMILARITY, color="tab:red", ls="--")
    look.set(xlabel="similarity of the two visual features", ylabel="density", title="Neighbouring parts of one kind look alike")
    look.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- block 10: search, then the vision-language model

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


def search_table(hits: dict[str, Sequence[int | None]], ks: Sequence[int] = (1, 3, 5)) -> pd.DataFrame:
    rows = {name: {f"a right keyframe in the top {k}": f"{share:.0%}" for k, share in S.recall_at(values, ks).items()}
            for name, values in hits.items()}
    return pd.DataFrame(rows).T


def search_figure(frames: Sequence[Frame], memory: T.L2Memory, results: Sequence[tuple[Task, S.KeyframeScores]],
                  top: int = 4) -> plt.Figure:
    """For each task: the best keyframes by plain semantics (left) and with the affordance tokens added (right)."""
    fig, axes = plt.subplots(len(results), 2 * top + 1, figsize=(1.75 * (2 * top + 1), 2.75 * len(results)),
                             gridspec_kw={"width_ratios": [1] * top + [0.12] + [1] * top}, squeeze=False)
    for row, (task, scores) in enumerate(results):
        axes[row, top].axis("off")
        for block, by in enumerate(("semantic", "combined")):
            for place, k in enumerate(scores.ranking(by)[:top]):
                axis = axes[row, block * (top + 1) + place]
                axis.imshow(frames[k].rgb[::2, ::2])
                token = scores.best_token[k]
                if by == "combined" and token != S.NO_TOKEN:
                    left, upper, right, lower = (value / 2 for value in memory.tokens[token].region.box)
                    axis.add_patch(plt.Rectangle((left - 5, upper - 5), right - left + 10, lower - upper + 10, fill=False, ec="yellow", lw=1.2))
                colour = "tab:green" if k in task.keyframes else "tab:red"
                axis.set_title(f"#{place + 1} · keyframe {k}", fontsize=8, color=colour)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set(color=colour, lw=2.5)
        axes[row, 0].set_ylabel(f'"{task.text[:38]}"', fontsize=7.5)
    axes[0, 1].text(0.0, 1.22, "plain semantics: the query against whole keyframes", transform=axes[0, 1].transAxes, fontsize=9)
    axes[0, top + 2].text(0.0, 1.22, "with affordance tokens (yellow box: best region)", transform=axes[0, top + 2].transAxes, fontsize=9)
    fig.suptitle("green frame: the keyframe shows the right part (ground truth); red: it does not", fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.9, bottom=0.01, wspace=0.06, hspace=0.16)
    return fig


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

    def ask(self, key: str, images: Sequence[np.ndarray], prompt: str) -> dict:
        if key not in self.answers:
            self._model = self._model or self._make_model()
            start = time.time()
            answer = self._model.ask(images, prompt)
            self.answers[key] = {"prompt": prompt, "answer": answer, "seconds": round(time.time() - start, 1)}
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            self._cache.write_text(json.dumps(self.answers, indent=1))
        return self.answers[key]


def reading_figure(images: Sequence[np.ndarray], task: str, answer: str, verdict: str) -> plt.Figure:
    fig, axes = plt.subplots(1, len(images), figsize=(4.3 * len(images), 6.3), squeeze=False)
    for axis, image in zip(axes[0], images):
        axis.imshow(image)
        axis.axis("off")
    fig.suptitle(f'task: "{task}"\nmodel answer: "{answer}"  →  {verdict}     (green: the model\'s pick; blue: the right part)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ---------------------------------------------------------------- blocks 7, 8, 13: geometry read at query time

def readout_entry(scene: P.Scene, memory: T.L2Memory, readout: T.Readout, truth_part: int) -> P.MemoryEntry:
    """A read-out in the shape the display and scoring helpers expect.

    `truth_part` (ground truth) is used only to score the contact point and to draw the oracle motion.
    """
    token, element = memory.tokens[readout.token], scene.elements[truth_part]
    seen = tuple(memory.times[memory.tokens[i].region.keyframe] for i in memory.tokens_of(int(memory.part_of[readout.token])))
    return P.MemoryEntry(truth_part, element.annot_id, T.ACTIONS[token.region.action], scene.task_of(element.annot_id),
                         memory.times[token.region.keyframe], readout.pixel, token.region.pixels, seen, readout.contact,
                         readout.normal, readout.stand, scene.motions.get(element.annot_id))


def readout_memory(scene: P.Scene, memory: T.L2Memory, frames: Sequence[Frame], truth: np.ndarray) -> P.Memory:
    """Every part of the L2 memory read out from its evidence token, for the entry figure and the scoreboard."""
    entries = []
    for part in range(memory.n_parts):
        token = memory.evidence(part)
        readout = T.read_geometry(memory, token, frames, scene.up)
        if readout is not None:
            entries.append(readout_entry(scene, memory, readout, int(truth[token])))
    return P.Memory(memory.times, tuple(entries), memory.floor, np.zeros((0, 3)))


def readout_figure(scene: P.Scene, titles: Sequence[str], entries: Sequence[P.MemoryEntry], frames: dict[str, Frame],
                   view: ScanView) -> plt.Figure:
    """Geometry read for each chosen token, drawn on its evidence keyframe: whole frame above, zoom below."""
    fig, axes = plt.subplots(2, len(entries), figsize=(3.5 * len(entries), 8.0), squeeze=False)
    for column, (title, entry) in enumerate(zip(titles, entries)):
        frame = frames[entry.keyframe]
        for axis in axes[:, column]:
            axis.imshow(frame.rgb)
            _draw_entry(axis, scene, entry, frame, stand_free_in_scan(entry, view))
            axis.axis("off")
        u, v = entry.pixel
        axes[1, column].set_xlim(u - 90, u + 90)
        axes[1, column].set_ylim(v + 90, v - 90)
        axes[0, column].set_title(title, fontsize=8)
        axes[1, column].set_title(f"zoom · {entry.action} · evidence keyframe {entry.keyframe} s\n"
                                  f"contact {contact_error_cm(scene, entry):.1f} cm from the part it marks", fontsize=8)
    fig.suptitle("red: contact point   cyan: surface normal   yellow: part motion (ORACLE)   ring: stand point", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.2)
    return fig


def _png_bytes(image: np.ndarray) -> int:
    return len(cv2.imencode(".png", image)[1])


def storage_table(scene: P.Scene, scan_path: Path, keyframes: Sequence[str], frames: Sequence[Frame],
                  action_maps: Sequence[np.ndarray], memory: T.L2Memory, explicit: P.Memory) -> pd.DataFrame:
    """Measured sizes: what the L2 memory keeps, next to the explicit entries of v1 and the laser scan."""
    index_bytes = len(pickle.dumps([(t.time_s, t.region) for t in memory.tokens] + [memory.part_of]))
    poses = np.array([np.concatenate([f.cam_to_world.ravel(), f.intrinsics.ravel()]) for f in frames])
    buffer = io.BytesIO()
    np.save(buffer, poses)
    rows = [("L2 memory", "keyframe images (as in the dataset)", sum(Path(scene.vid.rgb_path(t)).stat().st_size for t in keyframes)),
            ("L2 memory", "depth maps (16-bit PNG, millimetres)", sum(_png_bytes((f.depth_m * 1000).astype(np.uint16)) for f in frames)),
            ("L2 memory", "affordance maps (PNG)", sum(_png_bytes((a + 1).astype(np.uint8)) for a in action_maps)),
            ("L2 memory", "camera poses, intrinsics", buffer.getbuffer().nbytes),
            ("L2 memory", f"part tokens: features of {len(memory.tokens)} tokens (16-bit)", memory.feature_bytes()),
            ("L2 memory", "part tokens: thin index", index_bytes),
            ("for comparison", f"v1 explicit entries ({len(explicit.entries)} parts)", len(pickle.dumps(explicit.entries))),
            ("for comparison", "laser scan of the room (ground truth, never an input)", scan_path.stat().st_size)]
    table = pd.DataFrame(rows, columns=["group", "what", "bytes"])
    table["size"] = [f"{b / 1e6:.2f} MB" if b >= 1e5 else f"{b / 1e3:.1f} KB" for b in table["bytes"]]
    return table.drop(columns="bytes").set_index(["group", "what"])


# ---------------------------------------------------------------- blocks 14, 15: illustration only

def events_sketch() -> plt.Figure:
    """What interaction detection and event tokens would add. A drawing: this video contains no interaction."""
    fig, axis = plt.subplots(figsize=(13.5, 3.4))
    axis.plot([0, 20], [0, 0], color="0.3", lw=1.5)
    for x in range(0, 21, 2):
        axis.plot([x, x], [-0.08, 0.08], color="0.3", lw=1)
        axis.text(x, -0.32, f"{x} s", ha="center", fontsize=8)
    axis.add_patch(plt.Rectangle((8, 0.25), 3.5, 0.5, fc="#f4c7c3", ec="0.3"))
    axis.text(9.75, 0.5, "hand touches the part\n(interval found by block 14)", ha="center", va="center", fontsize=8)
    for x, text, colour in ((7.2, "keyframe before:\npart not hidden by the hand\n→ where the part is", "tab:blue"),
                            (12.3, "keyframe after:\nthe result is visible\n→ new state", "tab:green")):
        axis.plot([x], [0], "o", color=colour, ms=9)
        axis.annotate(text, xy=(x, 0.05), xytext=(x + (-3.6 if x < 9 else 2.3), 1.35), fontsize=8, color=colour, ha="center",
                      arrowprops={"arrowstyle": "->", "color": colour})
    axis.add_patch(FancyBboxPatch((2.4, -1.55), 14.7, 0.8, boxstyle="round,pad=0.05", fc=STATUS_COLOUR["missing"], ec="0.35"))
    axis.text(9.75, -1.15, "event token (block 15): features before + features after  |  explicit: start 8 s, end 11.5 s, "
                           "which part, 2nd time", ha="center", va="center", fontsize=8)
    axis.set(xlim=(-0.5, 20.5), ylim=(-1.8, 2.1), title="ILLUSTRATION ONLY (no data): what blocks 14 and 15 will add when a video shows someone acting")
    axis.axis("off")
    return fig
