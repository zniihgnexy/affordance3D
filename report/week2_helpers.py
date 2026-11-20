"""Figures and tables of the week-2 report (report/2026-11-20_week2_summary.ipynb, builder
notebooks/builders/build_report_2026_11_20_week2.py). They read finished part-vector runs (runs/part_vectors/) and the
keyframes of one scene; English figure labels, as in the other reports."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance3d.frames import Frame
from affordance3d.geometry import project
from affordance3d.mem import action_records as A
from affordance3d.mem import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "part_vectors"
PRED, TRUE = "#D55E00", "#009E73"  # Okabe-Ito vermilion and green, as in the videos
PRESSED = ("key_press", "tip_push", "plug_in", "foot_push")
PULLED = ("hook_pull", "pinch_pull", "unplug")
GROUP_NAMES = {"pressed": "按、插的部件", "rotate": "旋钮", "off_part_rotations": "门窗类（轴在部件外）", "all": "全部"}


def run_json(name: str, file: str = "summary.json") -> dict | list:
    return json.loads((RUNS / name / file).read_text())


def rates_table(name: str, variants: dict[str, str], groups: Sequence[str] = tuple(GROUP_NAMES)) -> pd.DataFrame:
    """Per label group (rows) and variant (columns): '+M / +MA / +MAO' in percent, with the group's size."""
    summary = run_json(name)
    rows = []
    for group in groups:
        first = summary["variants"][next(iter(variants.values()))][group]
        row = {"部件": GROUP_NAMES.get(group, group), "个数": first["n"]}
        for column, key in variants.items():
            rates = summary["variants"][key][group]
            row[column] = "—" if rates["n"] == 0 else f"{rates['m']:.0f} / {rates['ma']:.0f} / {rates['mao']:.0f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("部件")


# ---------------------------------------------------------------- block figures on one scene

def parts_in(label_map: np.ndarray) -> list[int]:
    ids, counts = np.unique(label_map[label_map >= 0], return_counts=True)
    return [int(i) for i, c in zip(ids, counts) if c >= P.MIN_PART_PIXELS]


def keyframe_strip(frames: Sequence[Frame], labels: Sequence[np.ndarray], times: Sequence[str], n: int = 8) -> plt.Figure:
    """`n` of the keyframes, evenly spaced in time, each with the number of annotated parts it shows."""
    picks = np.linspace(0, len(frames) - 1, n).round().astype(int)
    figure, axes = plt.subplots(1, n, figsize=(2.3 * n, 3.4))
    for ax, k in zip(axes, picks):
        ax.imshow(frames[k].rgb)
        ax.set_title(f"#{k + 1}  {float(times[k]) - float(times[0]):.0f} s\n{len(parts_in(labels[k]))} parts", fontsize=9)
        ax.set_axis_off()
    figure.suptitle(f"W1: {n} of the {len(frames)} keyframes (steadiest frames), time from the first keyframe", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure


def masks_figure(scene: P.Scene, frames: Sequence[Frame], labels: Sequence[np.ndarray], picks: Sequence[int],
                 margin: int = 60, least: int = 260) -> plt.Figure:
    """Annotated part masks (ORACLE) of the chosen keyframes, coloured by label, each with its label name; each panel is
    cropped to its parts with a margin."""
    names = sorted({e.label for e in scene.elements})
    colours = {name: plt.cm.tab10(i % 10)[:3] for i, name in enumerate(names)}
    figure, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5.4))
    for ax, k in zip(np.atleast_1d(axes), picks):
        image = frames[k].rgb.astype(float) / 255
        for p in parts_in(labels[k]):
            mask = labels[k] == p
            image[mask] = 0.35 * image[mask] + 0.65 * np.array(colours[scene.elements[p].label])
            v, u = np.argwhere(mask).mean(axis=0)
            ax.text(u, v - 12, scene.elements[p].label, fontsize=7, color="white", ha="center",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "lw": 0})
        ax.imshow(image)
        rows, cols = np.nonzero(labels[k] >= 0)
        if len(rows):
            centre = np.array([(cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2])
            half = max((cols.max() - cols.min()) / 2 + margin, (rows.max() - rows.min()) / 2 + margin, least / 2)
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] + half, centre[1] - half)
        ax.set_title(f"keyframe #{k + 1}: annotated part masks (ORACLE)", fontsize=9)
        ax.set_axis_off()
    figure.tight_layout()
    return figure


def depth_figure(lidar: Frame, rendered: Frame, label_map: np.ndarray, max_m: float = 4.0) -> tuple[plt.Figure, int, int]:
    """RGB | device LiDAR depth | depth rendered from the laser scan; returns also how many parts of the frame have no
    LiDAR depth at any of their pixels, of how many."""
    parts = parts_in(label_map)
    blind = [p for p in parts if not (lidar.depth_m[label_map == p] > 0).any()]
    figure, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    axes[0].imshow(lidar.rgb)
    axes[0].set_title("RGB keyframe", fontsize=10)
    for ax, frame, title in ((axes[1], lidar, "LiDAR depth (device, input)"), (axes[2], rendered, "rendered from laser scan (upper bound)")):
        shown = np.where(frame.depth_m > 0, frame.depth_m, np.nan)
        im = ax.imshow(shown, cmap="turbo", vmin=0, vmax=max_m)
        ax.set_title(title, fontsize=10)
    for p in blind:
        v, u = np.argwhere(label_map == p).mean(axis=0)
        axes[1].plot(u, v, "o", ms=14, mfc="none", mec="white", mew=2)
    for ax in axes:
        ax.set_axis_off()
    figure.colorbar(im, ax=axes[1:], fraction=0.025, label="metres (white: no depth)")
    return figure, len(blind), len(parts)


def label_motion_figure(pairs: Counter) -> plt.Figure:
    """Annotated parts per affordance label, split by motion type (30 validation scenes, annotation files only)."""
    labels = sorted({label for label, _ in pairs}, key=lambda label: pairs[(label, "trans")] + pairs[(label, "rot")])
    trans = np.array([pairs[(label, "trans")] for label in labels])
    rot = np.array([pairs[(label, "rot")] for label in labels])
    figure, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(labels, trans, color="#56B4E9", label="translation (push / pull straight)")
    ax.barh(labels, rot, left=trans, color="#E69F00", label="rotation (turn / swing)")
    ax.set_xlabel("annotated parts")
    ax.set_title("W4: the label alone fixes the motion type except for the two pulled labels", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------- what a robot would do: the contact's path

def _rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle) + axis * (axis @ vector) * (1 - np.cos(angle))


def contact_path(contact: np.ndarray, normal: np.ndarray, label: str, kind: str | None, direction: np.ndarray | None,
                 origin: np.ndarray, angle_deg: float, slide_m: float, steps: int = 24) -> np.ndarray | None:
    """Where the touched point goes when the part is opened: along a pull out of the surface (a press into it), or round
    the axis by `angle_deg`, turning the way that first moves it out of the surface. A knob (axis through the contact)
    gets a small circle. The annotation's signs are unreliable, so the sense is set by this rule for both."""
    if kind is None or direction is None:
        return None
    direction = A.unit(direction)
    if kind == "trans":
        sense = -1.0 if label in PRESSED else 1.0
        if (direction @ normal) * sense < 0:
            direction = -direction
        return contact + np.linspace(0, slide_m, steps)[:, None] * direction
    foot = A.axis_foot(origin, direction, contact)
    radius = contact - foot
    if np.linalg.norm(radius) < 0.02:  # turns the part itself: show a circle of 3 cm
        radius = 0.03 * A.unit(np.cross(direction, normal if abs(direction @ normal) < 0.9 else [0.0, 0.0, 1.0]))
        return foot + np.array([_rotate(radius, direction, a) for a in np.linspace(0, 2 * np.pi, steps)])
    if np.cross(direction, radius) @ normal < 0:
        direction = -direction
    return foot + np.array([_rotate(radius, direction, a) for a in np.radians(np.linspace(0, angle_deg, steps))])


def pick_examples(rows: list[dict], truths: dict[int, A.PartMotion], variant: Sequence[str], per_kind: int = 2) -> list[tuple[str, dict]]:
    """`per_kind` parts of each kind, chosen by a fixed rule (seen in the most keyframes), not by how good their vectors are."""
    kinds = {"door or window": lambda r: truths[r["part"]].hinged, "drawer": lambda r: r["label"] in PULLED and r["true_kind"] == "trans",
             "knob": lambda r: r["label"] == "rotate", "pressed": lambda r: r["label"] in PRESSED}
    mine = [r for r in rows if (r["depth"], r["rule"], r["frames"]) == tuple(variant) and r["vector"] is not None]
    return [(name, row) for name, test in kinds.items() for row in sorted((r for r in mine if test(r)), key=lambda r: -r["seen_in"])[:per_kind]]


def motion_paths_figure(examples: list[tuple[str, dict]], truths: dict[int, A.PartMotion], frames: Sequence[Frame],
                        angle_deg: float = 60, slide_m: float = 0.3, columns: int = 3) -> plt.Figure:
    """For each example part, on its best keyframe: where the touched point goes under the computed motion (vermilion)
    and under the annotated one (green)."""
    n_rows = -(-len(examples) // columns)
    figure, axes = plt.subplots(n_rows, columns, figsize=(4.4 * columns, 4.6 * n_rows), squeeze=False)
    for ax in axes.flat[len(examples):]:
        ax.set_axis_off()
    for ax, (name, row) in zip(axes.flat, examples):
        vector, truth = row["vector"], truths[row["part"]]
        frame = frames[vector["keyframe"]]
        contact, normal = np.array(vector["contact"]), np.array(vector["normal"])
        axis = None if vector["axis"] is None else np.array(vector["axis"])
        paths = [(contact_path(contact, normal, row["label"], vector["kind"], axis, np.array(vector["origin"]), angle_deg, slide_m), PRED),
                 (contact_path(contact, normal, row["label"], truth.kind, truth.direction, truth.origin, angle_deg, slide_m), TRUE)]
        ax.imshow(frame.rgb)
        drawn = [project(contact[None], frame)[0][0]]
        for path, colour in paths:
            if path is None:
                continue
            uv, z, _ = project(path, frame)
            if (z <= 0.05).any():
                continue
            ax.plot(uv[:, 0], uv[:, 1], color=colour, lw=3)
            ax.annotate("", uv[-1], uv[-2], arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2})
            drawn.extend(uv)
        ax.plot(*drawn[0], "o", ms=7, color=PRED, mec="white", mew=1.2)
        uv = np.array(drawn)
        centre, half = (uv.min(0) + uv.max(0)) / 2, max(np.ptp(uv, axis=0).max() / 2 + 60, 110)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] + half, centre[1] - half)
        verdict = "type right" if row["score"]["m"] else "type WRONG"
        verdict += ", axis right" if row["score"]["ma"] else (", axis wrong" if row["score"]["m"] else "")
        ax.set_title(f"{name}: {row['label']}\ncomputed {vector['kind']}, annotated {truth.kind} ({verdict})", fontsize=9)
        ax.set_axis_off()
    figure.legend(handles=[plt.Line2D([], [], color=PRED, lw=3, label="touched point moved by the computed motion"),
                           plt.Line2D([], [], color=TRUE, lw=3, label="moved by the annotated motion")],
                  loc="lower center", ncol=2, fontsize=9, frameon=False)
    figure.suptitle(f"What a robot would do with the memory: open by {angle_deg:.0f} degrees or {slide_m * 100:.0f} cm", fontsize=11)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=2.5)
    return figure


# ---------------------------------------------------------------- the memory's table and the door diagnosis

def part_table_frame(name: str, n: int = 12) -> pd.DataFrame:
    """The first `n` rows of a run's part table (device world frame, metres)."""
    def short(values: list[float] | None) -> str:
        return "—" if values is None else "(" + ", ".join(f"{v:+.2f}" for v in values) + ")"
    rows = [{"部件": r["part"], "标签": r["label"], "任务（标注里的描述）": r["task"][:48], "类型": r["kind"], "接触点 (m)": short(r["contact"]),
             "法线": short(r["normal"]), "方向或转轴": short(r["axis"]), "原点 (m)": short(r["origin"]), "证据帧数": len(r["evidence"])}
            for r in run_json(name, "part_table.json")[:n]]
    return pd.DataFrame(rows).set_index("部件")


def doors_figure(runs: dict[str, str]) -> plt.Figure:
    """The door diagnosis of each run (door_hinges.json, LiDAR): parts whose axis, and axis and origin, come out right
    under each way of placing the hinge."""
    ways = {"rule": "surface edge\n(old rule)", "up_handle": "vertical,\nthrough handle", "up_far_end": "vertical,\nfar end",
            "up_right_end": "vertical, right end\n(side given)"}
    figure, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 3.8), squeeze=False)
    for ax, (title, name) in zip(axes[0], runs.items()):
        counts = run_json(name, "door_hinges.json")["counts"]["lidar"]
        x = np.arange(len(ways))
        ax.bar(x - 0.2, [counts[w]["axis_ok"] for w in ways], 0.4, color="#56B4E9", label="axis right")
        ax.bar(x + 0.2, [counts[w]["axis_and_origin_ok"] for w in ways], 0.4, color="#D55E00", label="axis and origin right")
        ax.axhline(counts["doors"], color="grey", ls="--", lw=1)
        ax.text(len(ways) - 0.5, counts["doors"], f"all {counts['doors']}", va="bottom", ha="right", fontsize=8, color="grey")
        ax.set_xticks(x, list(ways.values()), fontsize=8)
        ax.set_title(f"{title}: {counts['doors']} doors and windows, LiDAR", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure
