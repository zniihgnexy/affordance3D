"""Figures and tables for notebooks/mem/phase1_results.ipynb: where should a memory put its tokens? (phase 1)

The numbers come from the per-scene result files of scripts/run_mem_phase1.py and from the summary written by
scripts/aggregate_mem_phase1.py. Only the illustrations at the end (`token_sets`, `first_pick` and their
figures) read a scene's images: they redo the search of one scene so that a picture shows what the numbers count.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from affordance3d.frames import Frame
from affordance3d.mem import phase1_stats as A
from affordance3d.mem import policies
from affordance3d.mem import search as S
from affordance3d.mem import tokens as T
from affordance3d.mem.keyframes import KEYFRAME_FILE
from affordance3d.mem.pipeline import MIN_PART_PIXELS

POLICY_NAMES = {A.P0: "P0 whole keyframes", A.P1: "P1 uniform grid", A.P2: "P2 random places",
                A.P3: "P3 generic regions", A.P4: "P4 affordance regions"}
POLICY_COLOUR = {A.P4: "#2a78d6", A.P3: "#eb6834", A.P2: "#1baf7a", A.P1: "#eda100", A.P0: "#52514e"}  # one colour per policy, everywhere
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RIGHT_COLOUR, WRONG_COLOUR = "#008300", "#e34948"
READING_NAMES = {A.P4: "P4 affordance regions", A.P3: "P3 generic regions"}  # keys of the model-reading results
GEOMETRY_ROWS = {"single_frame": "one keyframe, read now (true part ids)", "l2_readout": "one keyframe, read now (parts linked by the memory)"}
RUNNER_SEED = 1000  # scripts/run_mem_phase1.py draws P2 and P3 of seed s from default_rng(1000 + s)
BIT, BIT_COLOUR = "whole keyframes + which keyframes hold an affordance region", "#6250d6"
ATTRIBUTION_ARMS = (  # arm in the diagnostics, label, colour, hatched (a baseline given the keyframes of P4)
    ("P0 whole keyframes", "P0 whole keyframes", POLICY_COLOUR[A.P0], False),
    (BIT, "whole keyframes + one bit per keyframe\n(does it hold an affordance region?)", BIT_COLOUR, False),
    ("P2 in the keyframes of P4", "P2 random places, in P4's keyframes", POLICY_COLOUR[A.P2], True),
    ("P3 in the keyframes of P4", "P3 generic regions, in P4's keyframes", POLICY_COLOUR[A.P3], True),
    ("P4 affordance regions", "P4 affordance regions", POLICY_COLOUR[A.P4], False))
ATTRIBUTION_STEPS = (  # paired difference in the diagnostics, label, colour
    (f"{BIT} minus P0 whole keyframes", "one bit per keyframe − P0", BIT_COLOUR),
    (f"P4 affordance regions minus {BIT}", "P4 − one bit per keyframe", BIT_COLOUR),
    ("P4 affordance regions minus P2 in the keyframes of P4", "P4 − P2 in P4's keyframes", POLICY_COLOUR[A.P2]),
    ("P4 affordance regions minus P3 in the keyframes of P4", "P4 − P3 in P4's keyframes", POLICY_COLOUR[A.P3]))
ONE = 1.0  # the budget at which the policies are compared: as many tokens as P4 has
MB = 1e6


# ---------------------------------------------------------------- loading

def load_runs(folder: Path) -> tuple[list[dict], dict]:
    """The scene results that can be scored and the summary made from exactly those scenes."""
    results = A.scored(A.load_results(folder))
    summary_file = folder / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"{summary_file} is missing: run scripts/aggregate_mem_phase1.py first")
    summary = json.loads(summary_file.read_text())
    if summary.get("visits") != [result["visit"] for result in results]:
        raise ValueError(f"the summary covers {summary['scenes']} scenes, not the {len(results)} results that are there now: "
                         "run scripts/aggregate_mem_phase1.py again")
    return results, summary


def load_diagnostics(folder: Path) -> dict:
    """The follow-up check written by scripts/diagnose_mem_phase1.py."""
    file = folder / "diagnostics.json"
    if not file.exists():
        raise FileNotFoundError(f"{file} is missing: run scripts/diagnose_mem_phase1.py (with --features first)")
    return json.loads(file.read_text())


# ---------------------------------------------------------------- tables

def data_table(results: Sequence[dict], summary: dict) -> pd.DataFrame:
    tokens = [r["memory"]["tokens"] for r in results]
    generic = [r["generic_regions"] for r in results if r["generic_regions"] is not None]
    seen = summary["parts_seen"]
    rows = {
        "scenes scored": summary["scenes"],
        "scenes without an answerable task (not scored)": len(summary["scenes_without_tasks"]),
        "tasks: all / answerable (the part is visible in a keyframe)": f"{summary['tasks_all']} / {summary['tasks']}",
        "keyframes per scene": results[0]["n_keyframes"],
        "annotated parts (all scenes run)": summary["parts"],
        "of these, visible in the steady keyframes / in evenly spaced keyframes": f"{seen['steady']} / {seen['uniform']}",
        "P4 tokens per scene: median (smallest to largest)": f"{np.median(tokens):.0f} ({min(tokens)} to {max(tokens)})",
        "generic regions per scene (pool of P3): median": f"{np.median(generic):.0f}" if generic else "not made",
    }
    return pd.DataFrame(rows.items(), columns=["what", "value"]).set_index("what")


def turn_rate_table(data: Path, results: Sequence[dict]) -> pd.DataFrame:
    """How fast the camera turns at the keyframes (device poses): the steady choice against evenly spaced frames."""
    names = {"steady": "steady keyframes (used)", "uniform": "evenly spaced keyframes (for comparison)"}
    rates: dict[str, list[float]] = {kind: [] for kind in names}
    for result in results:
        saved = json.loads((data / result["visit"] / result["video"] / KEYFRAME_FILE).read_text())
        for kind, values in rates.items():
            values.extend(saved["turn_rate_deg_s"][t] for t in saved[kind])
    table = {names[kind]: {"keyframes": len(values), "median turn rate (deg/s)": float(np.median(values)),
                           "90th percentile (deg/s)": float(np.percentile(values, 90)), "fastest (deg/s)": max(values)}
             for kind, values in rates.items()}
    return pd.DataFrame(table).T.round(1)


def rates_table(rows: dict) -> pd.DataFrame:
    """Rates and differences with their intervals, in percent (differences in percentage points)."""
    table = {name: {"value": 100 * row["rate"], "95% interval, low": 100 * row["low"], "95% interval, high": 100 * row["high"]}
             for name, row in rows.items()}
    return pd.DataFrame(table).T.round(1)


def token_table(results: Sequence[dict]) -> pd.DataFrame:
    """Budget 1: how often the best-scoring token lies on the target part, and how large the regions are."""
    rows = {}
    for name in (A.P4, A.P3, A.P2):
        on_part = [A.scene_rates(result, name, ONE, "on_part") for result in results]
        if any(rate is None for rate in on_part):
            continue
        runs = [run for result in results for run in result["search"]
                if run["policy"].startswith(name) and abs(run["budget"] - ONE) < 1e-9]
        rate, low, high = A.bootstrap(on_part)
        rows[POLICY_NAMES[name]] = {"best token is on the target part (%)": round(100 * rate, 1),
                                    "95% interval": f"{100 * low:.1f} to {100 * high:.1f}",
                                    "tokens per scene": round(float(np.mean([run["tokens"] for run in runs])), 1),
                                    "median region area (% of the image)": round(100 * float(np.median([run["area_share"] for run in runs])), 2)}
    return pd.DataFrame(rows).T


def geometry_table(results: Sequence[dict], summary: dict) -> pd.DataFrame:
    """Contact points read from one keyframe against the same read from a fused cloud (true part ids in both)."""
    geometry, rows = summary["geometry"], {}
    for key, row in geometry.items():
        fused = key.startswith("fused")
        voxel = key.split()[1] if fused else None
        read_ms = ([r["geometry"]["fused"][voxel]["read_ms_per_part"] for r in results] if fused
                   else [r["geometry"]["read_ms_per_part"] for r in results] if key == "l2_readout" else [])
        rows[f"fused cloud, {100 * float(voxel):.0f} cm voxels" if fused else GEOMETRY_ROWS[key]] = {
            "parts": row["parts"], "contact read": row["read"], "median error (cm)": round(row["median_cm"], 1),
            "within 3 cm (% of read)": round(100 * row["within_3cm"], 1),
            "within 3 cm (% of all parts)": round(100 * row["within_3cm_of_all"], 1),
            "extra storage per scene (MB)": round(row["bytes_per_scene"] / MB, 1) if fused else 0.0,
            "build time per scene (s)": round(row["build_s"], 1) if fused else 0.0,
            "read time per part (ms)": round(float(np.mean(read_ms)), 1) if read_ms else float("nan"),
        }
    return pd.DataFrame(rows).T


def cost_table(summary: dict) -> pd.DataFrame:
    """What one scene's memory holds and what writing and reading cost in this run."""
    storage, reading = summary["storage_bytes_per_scene"], summary["reading"]
    rows = {
        "keyframe images (MB per scene)": round(storage["keyframe_images"] / MB, 1),
        "depth maps (MB per scene)": round(storage["depth_maps"] / MB, 1),
        "affordance maps (MB per scene)": round(storage["affordance_maps"] / MB, 2),
        "token features (MB per scene)": round(storage["token_features"] / MB, 2),
        "tokens per scene": round(storage["tokens"], 1),
        "writing the memory (s per scene, affordance map given)": round(summary["write_s_per_scene"], 1),
    }
    for name in READING_NAMES.values():
        if name in reading:
            rows[f"model reading, {name} (s per query, median)"] = round(reading[name]["seconds_per_query"], 2)
    return pd.DataFrame(rows.items(), columns=["what", "value"]).set_index("what")


def scene_differences(results: Sequence[dict], other: str, budget: float = ONE, key: str = "hit1") -> np.ndarray:
    """Per scene: the rate of P4 minus the rate of `other` at one budget (whole keyframes have no budget); NaN where not run."""
    differences = []
    for result in results:
        ours = A.scene_rates(result, A.P4, budget, key)
        theirs = A.scene_rates(result, other, None if other == A.P0 else budget, key)
        differences.append(float("nan") if ours is None or theirs is None else float(np.mean(ours) - np.mean(theirs)))
    return np.array(differences)


def typical_scene(results: Sequence[dict], other: str = A.P3, min_tasks: int = 10) -> str:
    """The visit whose P4-minus-`other` difference is the median one among scenes with at least `min_tasks` tasks:
    an example chosen by a rule, not by how good it looks."""
    eligible = [result for result in results if result["tasks"]["answerable"] >= min_tasks] or list(results)
    differences = scene_differences(eligible, other)
    order = [int(i) for i in np.argsort(differences, kind="stable") if not np.isnan(differences[i])]
    if not order:
        raise ValueError(f"no scene has both P4 and {other}")
    return eligible[order[(len(order) - 1) // 2]]["visit"]


# ---------------------------------------------------------------- figures of the numbers

ORACLE_NOTE = "ORACLE affordance map (projected annotations): every number is an upper bound"


def _oracle_note(fig: plt.Figure) -> None:
    fig.text(0.995, 0.005, ORACLE_NOTE, ha="right", va="bottom", fontsize=8, color=MUTED)


def _clean(axis: plt.Axes, grid: str = "x") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    axis.grid(axis=grid, color=GRID, lw=0.8)
    axis.set_axisbelow(True)


def _rate_bars(axis: plt.Axes, rows: dict, keys: Sequence[str], labels: Sequence[str], colours: Sequence[str]) -> None:
    """One thin bar per policy with its 95% interval and the value written next to it."""
    for y, (key, colour) in enumerate(zip(keys, colours)):
        rate, low, high = (100 * rows[key][part] for part in ("rate", "low", "high"))
        axis.barh(y, rate, height=0.5, color=colour)
        axis.plot([low, high], [y, y], color=INK, lw=1.2, solid_capstyle="butt")
        axis.text(max(high, rate) + 1.5, y, f"{rate:.1f}%", va="center", fontsize=9, color=INK)
    axis.set_yticks(range(len(keys)), labels, fontsize=9, color=INK)
    axis.invert_yaxis()
    axis.set_xlim(0, 112)
    axis.set_xticks(range(0, 101, 20))
    _clean(axis)


def _difference_dots(axis: plt.Axes, rows: dict, keys: Sequence[str], labels: Sequence[str], colours: Sequence[str]) -> None:
    """Paired differences in percentage points: a dot, its 95% interval, and a line at zero."""
    ends = []
    for y, (key, colour) in enumerate(zip(keys, colours)):
        rate, low, high = (100 * rows[key][part] for part in ("rate", "low", "high"))
        axis.plot([low, high], [y, y], color=colour, lw=2.5, solid_capstyle="butt")
        axis.plot(rate, y, "o", color=colour, ms=9, mec="white", mew=1.5)
        axis.text(rate, y - 0.28, f"{rate:+.1f}  [{low:+.1f}, {high:+.1f}]", ha="center", va="bottom", fontsize=9, color=INK)
        ends += [low, high]
    axis.axvline(0, color=INK, lw=0.9)
    axis.set_yticks(range(len(keys)), labels, fontsize=9, color=INK)
    axis.set_ylim(len(keys) - 0.5, -0.75)
    reach = max(max(abs(end) for end in ends) * 1.25, 5.0)
    axis.set_xlim(min(-reach * 0.35, min(ends) * 1.25), reach)
    _clean(axis)


def headline_figure(summary: dict, key: str = "first_keyframe_right", what: str = "the first keyframe shows the part") -> plt.Figure:
    """Budget 1: the rate of every policy (left) and the paired differences of P4 against the others (right)."""
    rows = summary[key]
    names = [name for name in (A.P4, A.P3, A.P2, A.P0) if name in rows]
    others = [name for name in names if name != A.P4]
    if A.P4 not in names or not others:
        raise ValueError("the summary needs P4 and at least one other policy at budget 1")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 0.55 * len(names) + 1.3), gridspec_kw={"width_ratios": [1.1, 1]})
    _rate_bars(left, rows, names, [POLICY_NAMES[name] for name in names], [POLICY_COLOUR[name] for name in names])
    left.set_title(f"Share of tasks where {what} (%)", fontsize=10, loc="left", color=INK)
    _difference_dots(right, rows, [f"P4 minus {name}" for name in others], [f"P4 − {name}" for name in others],
                     [POLICY_COLOUR[name] for name in others])
    right.set_title("Paired difference, percentage points (line = 95% interval)", fontsize=10, loc="left", color=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _oracle_note(fig)
    return fig


def reading_figure(summary: dict) -> plt.Figure:
    """The model picks a numbered region in the three best keyframes: share of tasks where it is on the target part."""
    rows = summary["reading"]
    names = [name for name in (A.P4, A.P3) if READING_NAMES[name] in rows]
    paired = "P4 minus P3" in rows
    fig, axes = plt.subplots(1, 2 if paired else 1, figsize=(11.5 if paired else 6.5, 0.55 * len(names) + 1.3),
                             gridspec_kw={"width_ratios": [1.1, 1]} if paired else None, squeeze=False)
    _rate_bars(axes[0, 0], rows, [READING_NAMES[name] for name in names], [POLICY_NAMES[name] for name in names],
               [POLICY_COLOUR[name] for name in names])
    axes[0, 0].set_title("Share of tasks where the model picks the right part (%)", fontsize=10, loc="left", color=INK)
    if paired:
        _difference_dots(axes[0, 1], rows, ["P4 minus P3"], ["P4 − P3"], [POLICY_COLOUR[A.P3]])
        axes[0, 1].set_title("Paired difference, percentage points (line = 95% interval)", fontsize=10, loc="left", color=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _oracle_note(fig)
    return fig


def attribution_figure(diagnostics: dict) -> plt.Figure:
    """Where the gain of the affordance tokens comes from: the arms of the follow-up check (left; hatched = a baseline
    given exactly the keyframes and per-keyframe token counts of P4) and the paired steps between them (right)."""
    rates, steps = diagnostics["first_keyframe_right"], diagnostics["differences"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 0.6 * len(ATTRIBUTION_ARMS) + 1.3), gridspec_kw={"width_ratios": [1.1, 1]})
    _rate_bars(left, rates, [arm[0] for arm in ATTRIBUTION_ARMS], [arm[1] for arm in ATTRIBUTION_ARMS], [arm[2] for arm in ATTRIBUTION_ARMS])
    for bar, arm in zip(left.patches, ATTRIBUTION_ARMS):
        if arm[3]:
            bar.set_hatch("///")
            bar.set_edgecolor("white")
    left.set_title("Share of tasks where the first keyframe shows the part (%)", fontsize=10, loc="left", color=INK)
    _difference_dots(right, steps, [step[0] for step in ATTRIBUTION_STEPS], [step[1] for step in ATTRIBUTION_STEPS],
                     [step[2] for step in ATTRIBUTION_STEPS])
    right.set_title("Paired difference, percentage points (line = 95% interval)", fontsize=10, loc="left", color=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _oracle_note(fig)
    return fig


def sweep_table(diagnostics: dict) -> pd.DataFrame:
    """The rate of every arm under every setting of the two constants of the scoring formula, in percent."""
    names = {"P4": "P4 affordance regions", "P3 same keyframes": "P3 in P4's keyframes", "P2 same keyframes": "P2 in P4's keyframes",
             "P3": "P3 generic regions", "P2": "P2 random places", "P1": "P1 uniform grid 2x1", "shortlist": "one bit per keyframe"}
    table = pd.DataFrame(diagnostics["sweep"]).T
    names = {key: label for key, label in names.items() if key in table.columns}
    table = table.rename(columns=names)[list(names.values())]
    return (100 * table).round(1)


def budget_figure(results: Sequence[dict], key: str = "hit1", what: str = "the first keyframe shows the part",
                  bit: dict | None = None) -> plt.Figure:
    """Rate against memory budget (tokens as a multiple of P4's) for every policy; bands are 95% intervals over scenes.

    Only points measured in every scene are drawn (a grid too fine for the budget cap of some scenes is left out).
    """
    fig, axis = plt.subplots(figsize=(10.5, 4.4))
    whole = [100 * value for value in A.bootstrap([A.scene_rates(result, A.P0, None, key) for result in results])]
    axis.axhspan(whole[1], whole[2], color=POLICY_COLOUR[A.P0], alpha=0.08, lw=0)
    axis.axhline(whole[0], color=POLICY_COLOUR[A.P0], lw=1.5, ls="--", label=f"{POLICY_NAMES[A.P0]} (no tokens)")
    if bit is not None:  # the follow-up check: no tokens, only which keyframes hold an affordance region
        axis.axhline(100 * bit["rate"], color=BIT_COLOUR, lw=1.5, ls=":", label="whole keyframes + one bit per keyframe (no tokens)")
    for name in (A.P4, A.P3, A.P2, A.P1):
        points = [point for point in A.curve(results, name, key) if point["scenes"] == len(results)]
        if not points:
            continue
        budgets = [point["budget"] for point in points]
        axis.fill_between(budgets, [100 * p["low"] for p in points], [100 * p["high"] for p in points],
                          color=POLICY_COLOUR[name], alpha=0.15, lw=0)
        axis.plot(budgets, [100 * p["rate"] for p in points], "-o", color=POLICY_COLOUR[name], lw=2, ms=7, mec="white", mew=1.5,
                  label=POLICY_NAMES[name])
    axis.set_xscale("log", base=2)
    ticks = (0.25, 0.5, 1, 2, 4, 8)
    axis.set_xticks(ticks, [f"{tick:g}×" for tick in ticks])
    axis.minorticks_off()
    axis.set_xlabel("Memory budget: tokens, as a multiple of the number of affordance regions", fontsize=9, color=MUTED)
    axis.set_ylabel("Share of tasks (%)", fontsize=9, color=MUTED)
    axis.set_title(f"Share of tasks where {what}, against memory budget", fontsize=10, loc="left", color=INK)
    _clean(axis, grid="y")
    axis.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _oracle_note(fig)
    return fig


def per_scene_figure(results: Sequence[dict], key: str = "hit1") -> plt.Figure:
    """Scene by scene: P4 minus each other policy at budget 1, sorted; a bar wears the colour of the policy that is ahead."""
    scenes = {name: scene_differences(results, name, key=key) for name in (A.P3, A.P2, A.P0)}
    others = [name for name, values in scenes.items() if not np.isnan(values).all()]
    if not others:
        raise ValueError("no scene has P4 together with another policy at budget 1")
    fig, axes = plt.subplots(1, len(others), figsize=(4.2 * len(others), 3.2), sharey=True, squeeze=False)
    for axis, other in zip(axes[0], others):
        differences = np.sort(100 * scenes[other][~np.isnan(scenes[other])])[::-1]
        colours = [POLICY_COLOUR[A.P4] if value > 0 else POLICY_COLOUR[other] for value in differences]
        axis.bar(range(len(differences)), differences, width=0.7, color=colours)
        axis.axhline(0, color=INK, lw=0.9)
        ahead, behind = int((differences > 0).sum()), int((differences < 0).sum())
        axis.set_title(f"P4 − {other}:  ahead in {ahead} scenes, level in {len(differences) - ahead - behind}, behind in {behind}",
                       fontsize=9, loc="left", color=INK)
        axis.set_xlabel("scenes, sorted", fontsize=9, color=MUTED)
        axis.set_xticks([])
        _clean(axis, grid="y")
    axes[0, 0].set_ylabel("difference (percentage points)", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _oracle_note(fig)
    return fig


# ---------------------------------------------------------------- illustrations on one scene

@dataclass(frozen=True)
class Pick:
    """What one way of placing tokens puts first for a task."""

    keyframe: int  # the keyframe ranked first
    box: policies.Box | None  # the best token of that keyframe; None when there is none (whole keyframes)
    right: bool  # the keyframe shows the part the task names


def generic_pool(regions_file: Path, image_hw: tuple[int, int]) -> list[policies.Candidate]:
    """The segments saved by scripts/make_generic_regions.py as token candidates (the pool P3 draws from)."""
    saved = json.loads(regions_file.read_text())
    return [policies.Candidate(k, (left, top, right, bottom), T.context_box((left, top, right, bottom), image_hw))
            for k, left, top, right, bottom in saved["regions"]]


def token_sets(frames: Sequence[Frame], action_maps: Sequence[np.ndarray], pool: Sequence[policies.Candidate],
               seed: int = 0) -> dict[str, list[policies.Candidate]]:
    """Where each policy puts its tokens at budget 1, drawn with the random numbers the runner uses for `seed`."""
    image_hw = frames[0].image_hw
    regions = [region for k, (actions, frame) in enumerate(zip(action_maps, frames))
               for region in T.find_regions(actions, frame.depth_m, k)[1]]
    afford = policies.affordance_candidates(regions, image_hw)
    rng = np.random.default_rng(RUNNER_SEED + seed)
    random = policies.random_candidates(afford, len(afford), len(frames), image_hw, rng)
    generic = policies.size_matched(pool, afford, len(afford), rng)
    grid = policies.grid_candidates(len(frames), image_hw, policies.grid_shape(len(afford) / len(frames), image_hw))
    return {A.P1: grid, A.P2: random, A.P3: generic, A.P4: afford}


def crops_of(frames: Sequence[Frame], candidates: Sequence[policies.Candidate]) -> list[np.ndarray]:
    return [frames[c.keyframe].rgb[c.crop[1]:c.crop[3], c.crop[0]:c.crop[2]] for c in candidates]


def target_pixels(part_labels: Sequence[np.ndarray], parts: Sequence[int]) -> dict[int, list[tuple[int, int]]]:
    """For each keyframe that shows a target part: the centre pixel (u, v) of every target part in it (ground truth)."""
    centres: dict[int, list[tuple[int, int]]] = {}
    for k, labels in enumerate(part_labels):
        for part in parts:
            rows_cols = np.argwhere(labels == part)
            if len(rows_cols) >= MIN_PART_PIXELS:
                v, u = np.median(rows_cols, axis=0)
                centres.setdefault(k, []).append((int(u), int(v)))
    return centres


def first_pick(query: np.ndarray, keyframe_vectors: np.ndarray, candidates: Sequence[policies.Candidate], vectors: np.ndarray,
               relevant: frozenset[int] | set[int]) -> Pick:
    """The keyframe a query ranks first with these tokens (none = whole keyframes only), scored as the runner scores it."""
    scores = S.score_keyframes(query, keyframe_vectors, vectors, [c.keyframe for c in candidates], [0] * len(candidates))
    keyframe = int(scores.ranking("combined" if candidates else "semantic")[0])
    token = int(scores.best_token[keyframe])
    return Pick(keyframe, None if token == S.NO_TOKEN else candidates[token].box, keyframe in relevant)


def choose_examples(picks: Sequence[dict[str, Pick]], n: int = 3) -> list[int]:
    """Tasks with different outcomes, in this order: affordance tokens right where generic regions are wrong,
    right where random places are wrong, wrong, right."""
    wanted: tuple[Callable[[dict[str, Pick]], bool], ...] = (
        lambda p: p[A.P4].right and not p[A.P3].right, lambda p: p[A.P4].right and not p[A.P2].right,
        lambda p: not p[A.P4].right, lambda p: p[A.P4].right)
    chosen: list[int] = []
    for outcome in wanted:
        match = next((i for i, pick in enumerate(picks) if i not in chosen and outcome(pick)), None)
        if match is not None:
            chosen.append(match)
    return chosen[:n]


def _show(axis: plt.Axes, frame: Frame) -> None:
    axis.imshow(frame.rgb)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(GRID)


def _box(axis: plt.Axes, box: policies.Box, colour: str, width: float = 1.8) -> None:
    left, top, right, bottom = box
    axis.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False, ec="white", lw=width + 1.6))
    axis.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False, ec=colour, lw=width))


def _panel_size(frame: Frame, width: float = 2.9) -> tuple[float, float]:
    height, image_width = frame.image_hw
    return width, width * height / image_width


def policies_figure(frames: Sequence[Frame], sets: dict[str, list[policies.Candidate]], keyframes_shown: Sequence[int]) -> plt.Figure:
    """The same keyframes under every policy: one box per token (a grid cell is drawn as its outline)."""
    names = [name for name in (A.P1, A.P2, A.P3, A.P4) if name in sets]
    width, height = _panel_size(frames[0])
    fig, axes = plt.subplots(len(keyframes_shown), len(names), figsize=(width * len(names), (height + 0.35) * len(keyframes_shown)),
                             squeeze=False)
    for row, keyframe in enumerate(keyframes_shown):
        for axis, name in zip(axes[row], names):
            inside = [c for c in sets[name] if c.keyframe == keyframe]
            _show(axis, frames[keyframe])
            for candidate in inside:
                _box(axis, candidate.box, POLICY_COLOUR[name])
            axis.set_title(f"{POLICY_NAMES[name]}\nkeyframe {keyframe}: {len(inside)} of {len(sets[name])} tokens", fontsize=9, color=INK)
    fig.tight_layout()
    return fig


def picks_figure(frames: Sequence[Frame], texts: Sequence[str], picks: Sequence[dict[str, Pick]],
                 targets: Sequence[dict[int, list[tuple[int, int]]]]) -> plt.Figure:
    """One row per task: the keyframe each policy ranks first, its best token (box) and the target part (ring) if it is there."""
    names = [name for name in (A.P0, A.P2, A.P3, A.P4) if name in picks[0]]
    width, height = _panel_size(frames[0])
    fig = plt.figure(figsize=(width * len(names), (height + 0.75) * len(texts)), layout="constrained")
    for subfigure, text, pick, target in zip(np.atleast_1d(fig.subfigures(len(texts), 1)), texts, picks, targets):
        subfigure.suptitle(f"Task: “{text}”", fontsize=10, color=INK, x=0.01, ha="left")
        for axis, name in zip(np.atleast_1d(subfigure.subplots(1, len(names))), names):
            chosen = pick[name]
            _show(axis, frames[chosen.keyframe])
            if chosen.box is not None:
                _box(axis, chosen.box, POLICY_COLOUR[name], width=2.2)
            for u, v in target.get(chosen.keyframe, []):
                axis.plot(u, v, "o", ms=16, mfc="none", mec="white", mew=3.2)
                axis.plot(u, v, "o", ms=16, mfc="none", mec=RIGHT_COLOUR, mew=1.6)
            verdict, colour = ("RIGHT keyframe", RIGHT_COLOUR) if chosen.right else ("WRONG keyframe", WRONG_COLOUR)
            axis.set_title(f"{POLICY_NAMES[name]}\n{verdict} (keyframe {chosen.keyframe})", fontsize=9, color=INK)
            for spine in axis.spines.values():
                spine.set_color(colour)
                spine.set_linewidth(2.5)
    return fig
