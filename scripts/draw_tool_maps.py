"""The gate figure of the tool-map check (discussion draft, section 24.9): both tools' boxes on two keyframes of three scenes.

    python scripts/draw_tool_maps.py                      # the first three scenes; writes runs/mem_phase1/tool_maps/gate.png
    python scripts/draw_tool_maps.py --visits 421254,421393,421657 --out gate.png

CPU only; it runs after the gate run of scripts/run_tool_maps.py (--gate, both tools). In each scene the two keyframes are
those showing most annotated parts (tool_maps.gate_keyframes, the same rule as --gate). Every panel shows one tool's boxes
with their part name (and SAM 3's score); a solid box has its centre on an annotated part (widened by 8 pixels), a dashed
one does not. The annotated parts are outlined in white: ground truth, drawn only so that the lead can judge the boxes.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the log)

from affordance3d.mem import keyframes  # noqa: E402
from affordance3d.mem import phase1_stats as A  # noqa: E402
from affordance3d.mem import pipeline as P  # noqa: E402
from affordance3d.mem import tool_maps as TM  # noqa: E402

GATE_SCENES = 3  # the first scenes of the phase-1 list, in visit order
TOOL_COLOUR = {"sam3": "#eb6834", "qwen8b": "#2a78d6"}
INK, MUTED = "#0b0b0b", "#52514e"
LABEL_ROOM_PX = 24
PANEL_WIDTH_IN = 3.4


def outline(axis: plt.Axes, labels: np.ndarray) -> None:
    """White outlines of the annotated parts visible in the keyframe."""
    for part in np.unique(labels[labels >= 0]):
        mask = (labels == part).astype(np.uint8)
        if mask.sum() < P.MIN_PART_PIXELS:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = np.vstack([contour[:, 0], contour[:1, 0]])
            axis.plot(points[:, 0], points[:, 1], color="white", lw=1.0)


def draw_boxes(axis: plt.Axes, detections: list[TM.Detection], labels: np.ndarray, colour: str) -> int:
    """The tool's boxes of one keyframe; returns how many have their centre on an annotated part."""
    on_part = TM.centres_on([d.box for d in detections], TM.any_part_mask(labels)) if detections else np.zeros(0, bool)
    for detection, right in zip(detections, on_part):
        left, top, right_edge, bottom = detection.box
        axis.add_patch(Rectangle((left, top), right_edge - left, bottom - top, fill=False, ec=colour, lw=1.8,
                                 ls="-" if right else (0, (3, 2))))
        text = detection.label if detection.score is None else f"{detection.label} {detection.score:.2f}"
        above = top >= LABEL_ROOM_PX  # near the top edge the label goes below the box
        axis.text(left, top - 3 if above else bottom + 3, text, color="white", fontsize=6.5, va="bottom" if above else "top",
                  clip_on=True, bbox={"facecolor": colour, "edgecolor": "none", "pad": 1.0, "alpha": 0.85})
    return int(on_part.sum())


def panel(axis: plt.Axes, rgb: np.ndarray, labels: np.ndarray, record: dict | None, keyframe: int, tool: str, visit: str) -> None:
    axis.imshow(rgb)
    outline(axis, labels)
    axis.set_xticks([])
    axis.set_yticks([])
    if record is None or str(keyframe) not in record["keyframes"]:
        axis.set_title(f"{visit} · keyframe {keyframe}\n{TM.TOOL_NAMES[tool]}: not run", fontsize=8.5, color=MUTED)
        return
    detections = [d for d in TM.tool_detections(record) if d.keyframe == keyframe]
    right = draw_boxes(axis, detections, labels, TOOL_COLOUR[tool])
    seen = len(TM.visible_parts([labels]))
    axis.set_title(f"{visit} · keyframe {keyframe} · {seen} annotated parts\n{TM.TOOL_NAMES[tool]}: {len(detections)} boxes, "
                   f"{right} on a part", fontsize=8.5, color=INK)
    print(f"{visit} keyframe {keyframe} {tool}: {len(detections)} boxes, {right} on a part, {seen} parts visible", flush=True)


def scene_rows(runs: Path, visit: str) -> tuple[list[np.ndarray], list[np.ndarray], list[int], dict]:
    """The two gate keyframes of a scene (images and label maps), their indices, and both tools' maps (None if absent)."""
    result = json.loads((runs / f"{visit}.json").read_text())
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", visit, result["video"])
    steady = keyframes.scene_keyframes(scene, "steady", result["n_keyframes"])
    labels = P.label_maps(scene, P.gt_frames(scene, steady))
    chosen = TM.gate_keyframes(labels)
    images = [P.input_frame(scene, steady[k], np.zeros(P.frame_hw(scene, steady[k], P.PIXEL_STEP), dtype=np.float32)).rgb
              for k in chosen]
    maps = {}
    for tool in TM.TOOLS:
        file = runs / "tool_maps" / tool / f"{visit}.json"
        maps[tool] = json.loads(file.read_text()) if file.exists() else None
    return images, [labels[k] for k in chosen], chosen, maps


def gate_figure(runs: Path, visits: list[str]) -> plt.Figure:
    tools = list(TM.TOOLS)
    scenes = [scene_rows(runs, visit) for visit in visits]
    ratios = [images[0].shape[0] / images[0].shape[1] for images, *_ in scenes]  # upright and lying videos differ
    fig, axes = plt.subplots(len(visits), 2 * len(tools), squeeze=False, gridspec_kw={"height_ratios": ratios},
                             figsize=(PANEL_WIDTH_IN * 2 * len(tools), sum(PANEL_WIDTH_IN * ratio + 0.7 for ratio in ratios) + 0.8))
    for row, (visit, (images, labels, chosen, maps)) in enumerate(zip(visits, scenes)):
        for slot, (rgb, label_map, keyframe) in enumerate(zip(images, labels, chosen)):
            for column, tool in enumerate(tools):
                panel(axes[row, slot * len(tools) + column], rgb, label_map, maps[tool], keyframe, tool, visit)
    fig.suptitle("Tool-built affordance maps at the gate (section 24.9): SAM 3 prompted with 8 part names (orange), "
                 "Qwen3-VL-8B asked for JSON boxes (blue)", fontsize=10.5, color=INK)
    fig.text(0.5, 0.005, "solid box: its centre lies on an annotated part (widened by 8 px); dashed: it does not.   "
             "White outline: annotated parts (ground truth, shown for judging only; never given to a tool).",
             ha="center", va="bottom", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "runs" / "mem_phase1")
    parser.add_argument("--visits", default="", help=f"comma-separated visits (default: the first {GATE_SCENES} scenes)")
    parser.add_argument("--out", type=Path, default=None, help="default: runs/mem_phase1/tool_maps/gate.png")
    args = parser.parse_args()
    visits = [v for v in args.visits.split(",") if v] or [r["visit"] for r in A.scored(A.load_results(args.runs))][:GATE_SCENES]
    out = args.out or args.runs / "tool_maps" / "gate.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    gate_figure(args.runs, visits).savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
