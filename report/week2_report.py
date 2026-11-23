"""Charts of the English week-2 report (report/week2_2026-11-16_to_11-22.ipynb, builder
notebooks/builders/build_report_2026_11_20_week2.py): the week's timeline, snapshots of the interactive .rrd memory,
and one chart per result. Every chart reads result files under runs/ that the notebook has already loaded; nothing is
recomputed here except the snapshot projection, which reuses the .rrd exporter's own geometry (report/week2_rrd.py).
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

GREEN, BLUE, GREY, RED, ORANGE, TEAL = "#2e7d32", "#1f5fae", "#8a8a8a", "#c0392b", "#e08e0b", "#00798c"

# ---------------------------------------------------------------- pure helpers (tested)


def view_basis(azim_deg: float, elev_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(towards camera, screen right, screen up) of an orthographic camera at azimuth/elevation, world Z up."""
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    towards = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    return towards, right, np.cross(towards, right)


def ortho_project(points: np.ndarray, azim_deg: float, elev_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Screen coordinates (N, 2) and depth towards the camera (N,) of world points; larger depth is nearer."""
    towards, right, up = view_basis(azim_deg, elev_deg)
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points @ right, points @ up], axis=1), points @ towards


def callout_positions(anchors: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float],
                      pad: float) -> list[tuple[float, float, str]]:
    """Label positions (x, y, horizontal alignment) for tags of the given anchor points: anchors left of the middle get a
    slot in the left margin, the others in the right one; slots are evenly spaced top to bottom in the anchors' own
    vertical order, so leader lines on one side never cross."""
    anchors = np.asarray(anchors, dtype=np.float64)
    middle = (x_range[0] + x_range[1]) / 2
    out: list[tuple[float, float, str]] = [(0.0, 0.0, "left")] * len(anchors)
    for left in (True, False):
        ids = sorted((i for i in range(len(anchors)) if (anchors[i, 0] < middle) == left), key=lambda i: -anchors[i, 1])
        ys = np.linspace(y_range[1], y_range[0], len(ids)) if len(ids) > 1 else [anchors[i, 1] for i in ids]
        x = x_range[0] - pad if left else x_range[1] + pad
        for i, y in zip(ids, ys):
            out[i] = (x, float(y), "right" if left else "left")
    return out


# Related work placed by what the memory stores (x) and how fine it is (y); "update" marks maps updated across visits.
# Sources: week-2 notebook section 5 and the Chinese proposal doc, sections 15 and 16 (checked against arXiv).
STORES = ("text graph / captions", "images (re-read)", "object features", "dense points / voxels", "Gaussian / neural field")
GRAINS = ("object", "part", "part + motion")
WORKS = [  # (name, x, y, updates across visits)
    ("ConceptGraphs", 0, 0, False), ("ReMEmbR", 0.25, 0.15, False), ("OpenFunGraph", 0, 1, False), ("FunGraph", 0.25, 1.15, False),
    ("3D-Mem", 1, 0, False), ("Memory Over Maps", 1.2, 0.15, False), ("Fun3DU", 1, 1, False), ("UniFunc3D", 1.2, 1.15, False),
    ("AffordBot", 1, 1.75, False), ("OpenMask3D", 2, 0, False), ("OK-Robot", 2.2, 0.15, False), ("VLMaps", 3, 0, False),
    ("ConceptFusion", 3.2, 0.15, False), ("DynaMem", 3, 0.3, True), ("LangSplat / LERF", 4, 0, False), ("DGSG-Mind", 3.8, 0.3, True),
    ("GraspSplats / LERF-TOGO", 4, 1, False), ("IAAO (two scans)", 4, 1.75, False),
]
OURS = ("ours: one Gaussian map per room,\ntraining-free, revisit update", 4.05, 2.0)


# ---------------------------------------------------------------- charts


def timeline(steps: Sequence[tuple[str, str]]) -> plt.Figure:
    """One box per step of the week; status done / tested / next."""
    style = {"done": ("#d7f0d2", GREEN), "tested": ("#d5e6fb", BLUE), "next": ("#f3f3f3", "#6b6b6b")}
    width, gap, height = 2.5, 0.35, 1.9
    fig, ax = plt.subplots(figsize=(2.85 * len(steps), 2.8))
    for i, (label, status) in enumerate(steps):
        x = i * (width + gap)
        fill, edge = style[status]
        ax.add_patch(FancyBboxPatch((x, 0), width, height, boxstyle="round,pad=0.05", fc=fill, ec=edge, lw=2,
                                    ls="--" if status == "next" else "-"))
        ax.text(x + width / 2, height / 2, textwrap.fill(label, 22), ha="center", va="center", fontsize=10.5, linespacing=1.2)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + width + gap - 0.04, height / 2), xytext=(x + width + 0.04, height / 2),
                        arrowprops={"arrowstyle": "-|>", "color": "#52514e", "lw": 1.6})
    ax.set_xlim(-0.15, len(steps) * (width + gap) - gap + 0.15)
    ax.set_ylim(-0.15, height + 0.15)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _draw_memory(ax, scan_xyz: np.ndarray, scan_rgb: np.ndarray, glyphs: Sequence[dict], azim: float, elev: float) -> None:
    uv, depth = ortho_project(scan_xyz, azim, elev)
    order = np.argsort(depth)
    ax.scatter(uv[order, 0], uv[order, 1], c=scan_rgb[order] / 255.0, s=0.15, linewidths=0, rasterized=True)
    for g in glyphs:
        puv, _ = ortho_project(g["points"], azim, elev)
        ax.scatter(puv[:, 0], puv[:, 1], color=g["colour"], s=1.2, linewidths=0, alpha=0.8)
        luv, _ = ortho_project(g["line"], azim, elev)
        ax.plot(luv[:-1, 0], luv[:-1, 1], color=g["colour"], lw=2.2, solid_capstyle="round")
        ax.annotate("", xy=luv[-1], xytext=luv[-2], arrowprops={"arrowstyle": "-|>", "color": g["colour"], "lw": 2.2})
    ax.set_aspect("equal")
    ax.axis("off")


def memory_snapshots(scan_xyz: np.ndarray, scan_rgb: np.ndarray, glyphs: Sequence[dict],
                     views: Sequence[tuple[str, float, float]]) -> plt.Figure:
    """The .rrd memory from fixed viewpoints: scan points, each part's points and its arrow at the far end of its slide.
    `glyphs`: dicts with points (part points), line (arrow polyline), colour (0-1 RGB), tag. The first view (top-down)
    is drawn large with its tags as callouts in the side margins; the others below it. Orthographic, x-ray style:
    parts behind walls are shown on top."""
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(2, len(views) - 1, height_ratios=[1.25, 1])
    title, azim, elev = views[0]
    ax = fig.add_subplot(grid[0, :])
    _draw_memory(ax, scan_xyz, scan_rgb, glyphs, azim, elev)
    uv, _ = ortho_project(scan_xyz, azim, elev)
    lo, hi = np.percentile(uv, 1, axis=0), np.percentile(uv, 99, axis=0)
    anchors = np.array([ortho_project(g["line"][:1], azim, elev)[0][0] for g in glyphs])
    for g, anchor, (x, y, ha) in zip(glyphs, anchors, callout_positions(anchors, (lo[0], hi[0]), (lo[1], hi[1]), 0.25)):
        ax.annotate(g["tag"], xy=anchor, xytext=(x, y), fontsize=7.5, color="white", ha=ha, va="center",
                    bbox={"boxstyle": "round,pad=0.25", "fc": g["colour"], "ec": "none"},
                    arrowprops={"arrowstyle": "-", "color": g["colour"], "lw": 0.8})
    ax.set_xlim(lo[0] - 2.6, hi[0] + 2.6)
    ax.set_ylim(lo[1] - 0.4, hi[1] + 0.4)
    ax.set_title(title, fontsize=12)
    for k, (title, azim, elev) in enumerate(views[1:]):
        ax = fig.add_subplot(grid[1, k])
        _draw_memory(ax, scan_xyz, scan_rgb, glyphs, azim, elev)
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def task2_bars(scores: dict[str, float]) -> plt.Figure:
    """SceneFun3D Task 2 AP25 (445 validation tasks) of each arm, with our main arm highlighted."""
    names = list(scores)
    colours = [ORANGE if "ours" in n else (GREY if "upper bound" in n else BLUE) for n in names]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 1.2))
    ax.barh(names, [scores[n] for n in names], color=colours)
    for y, n in enumerate(names):
        ax.text(scores[n] + 0.8, y, f"{scores[n]:.1f}", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("AP25 (%), same 445 tasks, same scorer")
    ax.set_xlim(0, max(scores.values()) * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def stage0_losses(t1: dict) -> plt.Figure:
    """Where the memory arm loses Task 2 (stage bars), SAM 3 misses by label, memory minus raw video by stratum."""
    stage_names = {"hit": "hit", "not_in_keyframes": "not in keyframes", "never_found": "SAM 3 never found",
                   "not_chosen": "found, VLM chose another", "lost_in_mask": "chosen, mask missed"}
    colours = ["#4c9a2a", "#9e9e9e", "#d1495b", "#edae49", TEAL]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1.2]})
    for row, counts in enumerate([t1["stages_memory_arm"], t1["lost_to_annotated"]["stages"]]):
        left, total = 0, sum(counts.values())
        for (stage, label), colour in zip(stage_names.items(), colours):
            width = 100 * counts[stage] / total
            axes[0].barh(row, width, left=left, color=colour, label=label if row == 0 else None)
            if width > 6:
                axes[0].text(left + width / 2, row, f"{counts[stage]}", ha="center", va="center", color="white", fontsize=9)
            left += width
    axes[0].set_yticks([0, 1], ["all 445 tasks", "annotated maps right,\nSAM 3 maps wrong (105)"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("% of tasks")
    axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.58), ncol=3)
    axes[0].set_title("Where Task 2 is lost")
    by_label = t1["predictions"]["T1-P3"]["labels_compared"]
    order = sorted(by_label, key=by_label.get)
    axes[1].barh(order, [100 * by_label[k] for k in order], color="#d1495b")
    axes[1].set_xlabel("% of visible parts SAM 3 never finds")
    axes[1].set_title("SAM 3 misses by label (>= 20 parts)")
    strata = [("all", "all"), ("small", "small parts"), ("large", "large parts"), ("several_of_label", "look-alikes in scene"),
              ("one_of_label", "only one of its label")]
    for i, (key, name) in enumerate(strata):
        s = t1["memory_minus_raw"][key]
        axes[2].errorbar(s["diff"], i, xerr=[[s["diff"] - s["low"]], [s["high"] - s["diff"]]], fmt="o", color=TEAL, capsize=3)
        axes[2].text(s["high"] + 0.8, i, f'{s["diff"]:+.1f} (n={s["tasks"]})', va="center", fontsize=8)
    axes[2].axvline(0, color="grey", lw=0.8)
    axes[2].set_xlim(right=max(t1["memory_minus_raw"][k]["high"] for k, _ in strata) + 9)
    axes[2].set_yticks(range(len(strata)), [n for _, n in strata])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("AP25 hit: memory (3 frames) minus raw video (40 frames), points")
    axes[2].set_title("Where the memory beats the raw video")
    fig.tight_layout()
    return fig


def vector_rules(t3: dict) -> plt.Figure:
    """Part vectors on 30 validation scenes (annotated masks): +M/+MA/+MAO per rule, and hinge axis vs hinge origin."""
    rules = [("c1", "type from label"), ("geometric", "type from geometry"), ("gravity", "label + gravity axis (main)"),
             ("oracle_kind", "type given"), ("oracle_gravity", "type given + gravity axis")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.3), gridspec_kw={"width_ratios": [2.2, 1]})
    width = 0.26
    for j, (metric, name, colour) in enumerate([("m", "+M type", "#9ccbe8"), ("ma", "+MA type + axis", BLUE),
                                                ("mao", "+MAO type + axis + origin", "#0b2e59")]):
        values = [t3["variants"][f"lidar/{rule}/best"]["all"][metric] for rule, _ in rules]
        xs = np.arange(len(rules)) + (j - 1) * width
        axes[0].bar(xs, values, width, color=colour, label=name)
        for x, v in zip(xs, values):
            axes[0].text(x, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    axes[0].set_xticks(range(len(rules)), [textwrap.fill(n, 16) for _, n in rules], fontsize=9)
    axes[0].set_ylabel("% of 471 scored parts")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=9, ncol=3, loc="upper left")
    axes[0].set_title("Given the part: is the motion right? (15° axis, 0.25 m origin)")
    hinge = t3["variants"]["lidar/oracle_gravity/best"]["off_part_rotations"]
    axes[1].bar(["axis direction", "axis + origin"], [hinge["ma"], hinge["mao"]], color=[BLUE, "#0b2e59"])
    for x, v in enumerate([hinge["ma"], hinge["mao"]]):
        axes[1].text(x, v + 1.5, f"{v:.1f}%", ha="center")
    axes[1].set_ylim(0, 100)
    axes[1].set_title(f"Doors and windows ({hinge['n']} hinges), type given")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def openeqa_share(categories: dict[str, dict], focus: Sequence[str]) -> plt.Figure:
    """Share of OpenEQA questions about functional parts per category: keyword-flagged and hand-corrected."""
    names = sorted(categories, key=lambda c: categories[c]["corrected"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ys = np.arange(len(names))
    ax.barh(ys - 0.18, [100 * categories[c]["share"] for c in names], 0.36, color="#bcbcbc", label="keyword-flagged")
    ax.barh(ys + 0.18, [100 * categories[c]["corrected"] for c in names], 0.36, color=TEAL, label="corrected by hand sample")
    ax.set_yticks(ys, [f"{c} *" if c in focus else c for c in names], fontsize=9)
    ax.set_xlabel("% of questions that need a functional part (* = the three focus categories)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def overwrite_bars(ow: dict) -> plt.Figure:
    """Overwrite simulation within one unchanged video: SAM 3 per-frame find rate, and found parts each rule erases."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    found = {"all visible parts": ow["per_frame_found_rate"], "parts found at least once": ow["per_frame_found_rate_ever_found"]}
    axes[0].barh(list(found), [100 * v for v in found.values()], color=BLUE)
    for y, v in enumerate(found.values()):
        axes[0].text(100 * v + 1, y, f"{100 * v:.1f}%", va="center")
    axes[0].set_xlim(0, 100)
    axes[0].set_title("SAM 3 finds a visible part on one keyframe")
    erased = {"newest view decides": ow["erased"]["last_view"]["all"],
              "newest view decides (parts seen 2+ times)": ow["erased"]["last_view"]["shown_twice_or_more"],
              "largest view decides": ow["erased"]["largest_view"]["all"],
              "conditional overwrite (chosen rule)": None}
    for y, (name, v) in enumerate(erased.items()):
        if v is None:
            axes[1].text(0.5, y, "to be measured in stage 2 (needs a geometry check)", va="center", fontsize=9, color=GREY)
        else:
            axes[1].barh(y, 100 * v, color=RED)
            axes[1].text(100 * v + 0.4, y, f"{100 * v:.1f}%", va="center")
    axes[1].set_yticks(range(len(erased)), list(erased))
    axes[1].set_ylim(len(erased) - 0.4, -0.6)
    axes[1].set_xlim(0, 25)
    axes[1].set_title(f"Found parts wrongly erased (of {ow['ever_found']}; nothing changed)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def positioning(works: Sequence[tuple] = WORKS, ours: tuple = OURS) -> plt.Figure:
    """Related work by what the memory stores and how fine it is; filled markers update across visits."""
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for name, x, y, updates in works:
        ax.scatter(x, y, s=70, color=TEAL if updates else "white", edgecolors=TEAL, linewidths=1.6, zorder=3)
        ax.text(x + 0.06, y + 0.06, name, fontsize=8.5)
    ax.scatter(ours[1], ours[2], s=320, marker="*", color=ORANGE, edgecolors="black", zorder=4)
    ax.text(ours[1] - 0.08, ours[2] + 0.12, ours[0], fontsize=9, ha="right", fontweight="bold")
    ax.set_xticks(range(len(STORES)), [textwrap.fill(s, 16) for s in STORES], fontsize=9)
    ax.set_yticks(range(len(GRAINS)), GRAINS)
    ax.set_xlim(-0.4, len(STORES) - 0.4)
    ax.set_ylim(-0.4, len(GRAINS) - 0.25)
    ax.grid(alpha=0.25)
    ax.set_xlabel("what the memory stores")
    ax.set_ylabel("granularity")
    ax.scatter([], [], s=70, color=TEAL, label="updated across visits")
    ax.scatter([], [], s=70, color="white", edgecolors=TEAL, label="static / no memory")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig


def stage2_fits(rows: Sequence[dict]) -> plt.Figure:
    """Gaussian maps fitted so far: held-out PSNR and fitting time, phase-1 vs second-visit videos."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, key, scale, label, line in ((axes[0], "test_psnr_db", 1, "held-out PSNR (dB)", 22),
                                        (axes[1], "fit_s", 1 / 60, "fitting time (min, shared A6000)", None)):
        for y, (role, colour) in enumerate((("primary", BLUE), ("revisit", ORANGE))):
            values = [r[key] * scale for r in rows if r["role"] == role]
            jitter = np.random.default_rng(y).uniform(-0.15, 0.15, len(values))
            ax.scatter(values, y + jitter, color=colour, s=22, alpha=0.8)
            if values:
                ax.plot([np.median(values)] * 2, [y - 0.3, y + 0.3], color="black", lw=2)
        if line is not None:
            ax.axvline(line, color=RED, ls="--", lw=1, label="predicted median bound")
            ax.legend(fontsize=8)
        ax.set_yticks([0, 1], [f"phase-1 video ({sum(r['role'] == 'primary' for r in rows)})",
                               f"second visit ({sum(r['role'] == 'revisit' for r in rows)})"])
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
