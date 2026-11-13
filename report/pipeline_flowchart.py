"""Pipeline flowchart for stage reports: what each block does, its status, and where the paper's contribution lies.

The drawing is generic (boxes, arrows, contribution tags); `STAGE_1` describes the memory pipeline as it stands
after phase 1. Each stage report updates the description and renders it again:

    python -m report.pipeline_flowchart helm/rounds/R7_figures/pipeline.png            # the pipeline as it stands
    python -m report.pipeline_flowchart helm/rounds/R8_figures/target.png target_r8   # another description, by name
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

STATUS = {  # fill colour and what it means
    "real": ("#d7f0d2", "computed from the inputs"),
    "oracle": ("#ffe2b8", "ORACLE: taken from the annotations for now"),
    "model": ("#d5e6fb", "frozen public model, not trained"),
    "missing": ("#e3e3e3", "not built yet"),
}
CONTRIBUTION_COLOUR, INK, MUTED, EDGE = "#c0392b", "#0b0b0b", "#52514e", "#6b6b6b"
FED_HATCH = "#d9822b"  # hatching on boxes that run on the output of an ORACLE box
BOX_FONT, PANEL_FONT = 8.6, 9.4


@dataclass(frozen=True)
class Box:
    key: str
    x: float  # lower-left corner, in drawing units (about one inch each)
    y: float
    text: str
    status: str
    tag: str = ""  # contribution tag drawn on the box, e.g. "C1"
    fed_by_oracle: bool = False  # computed from the inputs, but on what an ORACLE box produced
    width: float = 1.8
    height: float = 1.15


@dataclass(frozen=True)
class Arrow:
    start: str
    end: str
    dashed: bool = False
    via_y: float | None = None  # route down from the start's bottom to this height, across, then down into the end's top


@dataclass(frozen=True)
class Contribution:
    tag: str
    title: str
    evidence: str
    strength: str  # the helm's judgement of the evidence so far, shown in brackets
    kind: str = "contribution"  # "contribution" (red tag) or "design" (grey tag: a design choice, not claimed)


@dataclass(frozen=True)
class Flowchart:
    title: str
    boxes: tuple[Box, ...]
    arrows: tuple[Arrow, ...]
    contributions: tuple[Contribution, ...]
    lanes: tuple[tuple[float, str], ...] = ()  # (height, label) of the row titles
    notes: tuple[tuple[float, float, str], ...] = ()  # (x, y, text) in small red italics: risks
    hints: tuple[tuple[float, float, str], ...] = ()  # (x, y, text) in small grey italics: how to read the chart
    footer: str = ""
    frame_tag: str = ""  # a contribution that is the whole system: a dashed red frame around every box
    contribution_label: str = "red: proposed contribution (helm's judgement)"
    panel_title: str = ("If this were a paper: where the contribution lies (red), as the helm judges it today; "
                        "[brackets] = the helm's judgement of the evidence, not a result")


def check(chart: Flowchart) -> None:
    """Fail early on a description that cannot be drawn faithfully."""
    keys = [box.key for box in chart.boxes]
    if len(set(keys)) != len(keys):
        raise ValueError("box keys must be unique")
    unknown = {box.status for box in chart.boxes} - STATUS.keys()
    if unknown:
        raise ValueError(f"unknown status {sorted(unknown)}; use one of {sorted(STATUS)}")
    missing = {end for arrow in chart.arrows for end in (arrow.start, arrow.end)} - set(keys)
    if missing:
        raise ValueError(f"arrows refer to unknown boxes {sorted(missing)}")
    tags = {contribution.tag for contribution in chart.contributions} - {""}
    if chart.frame_tag and chart.frame_tag not in tags:
        raise ValueError(f"the frame carries tag {chart.frame_tag} without a contribution line")
    kinds = {contribution.kind for contribution in chart.contributions} - {"contribution", "design"}
    if kinds:
        raise ValueError(f"unknown contribution kind {sorted(kinds)}")
    untagged = {box.tag for box in chart.boxes if box.tag} - tags
    if untagged:
        raise ValueError(f"boxes carry tags without a contribution line: {sorted(untagged)}")


def _draw_box(axis: plt.Axes, box: Box) -> None:
    tagged = bool(box.tag)
    axis.add_patch(FancyBboxPatch((box.x, box.y), box.width, box.height, boxstyle="round,pad=0.04",
                                  fc=STATUS[box.status][0], ec=CONTRIBUTION_COLOUR if tagged else EDGE, lw=2.6 if tagged else 1.0))
    if box.fed_by_oracle:
        axis.add_patch(FancyBboxPatch((box.x, box.y), box.width, box.height, boxstyle="round,pad=0.04", fc="none", ec=FED_HATCH,
                                      lw=0, hatch="////", alpha=0.55))
    axis.text(box.x + box.width / 2, box.y + box.height / 2, box.text, ha="center", va="center", fontsize=BOX_FONT, color=INK,
              bbox={"boxstyle": "round,pad=0.15", "fc": STATUS[box.status][0], "ec": "none", "alpha": 0.9} if box.fed_by_oracle else None)
    if tagged:
        axis.text(box.x + 0.02, box.y + box.height + 0.06, box.tag, ha="left", va="center", fontsize=9, weight="bold", color="white",
                  bbox={"boxstyle": "round,pad=0.25", "fc": CONTRIBUTION_COLOUR, "ec": "none"})


def _draw_arrow(axis: plt.Axes, start: Box, end: Box, arrow: Arrow) -> None:
    style = {"color": "0.3", "lw": 1.2, "ls": (0, (4, 3)) if arrow.dashed else "-"}
    if arrow.via_y is None:
        axis.annotate("", xy=(end.x - 0.05, end.y + end.height / 2), xytext=(start.x + start.width + 0.05, start.y + start.height / 2),
                      arrowprops={"arrowstyle": "->", **style})
        return
    x0, x1 = start.x + start.width / 2, end.x + end.width / 2
    axis.plot([x0, x0, x1], [start.y - 0.05, arrow.via_y, arrow.via_y], **style)
    axis.annotate("", xy=(x1, end.y + end.height + 0.05), xytext=(x1, arrow.via_y), arrowprops={"arrowstyle": "->", **style})


def _draw_panel(axis: plt.Axes, chart: Flowchart, top: float) -> float:
    """The contribution lines under the drawing; returns the height where the panel ends."""
    axis.text(-0.1, top, chart.panel_title, fontsize=10.5, weight="bold", color=INK)
    y = top - 0.4
    for contribution in chart.contributions:
        colour = CONTRIBUTION_COLOUR if contribution.kind == "contribution" else MUTED
        axis.text(0.0, y, contribution.tag or "design", ha="left", va="top", fontsize=9, weight="bold", color="white",
                  bbox={"boxstyle": "round,pad=0.25", "fc": colour, "ec": "none"})
        indent = 0.55 if contribution.tag else 0.95
        axis.text(indent, y, f"{contribution.title}  [{contribution.strength}]", ha="left", va="top", fontsize=PANEL_FONT, weight="bold",
                  color=INK)
        lines = contribution.evidence.count("\n") + 1
        axis.text(indent, y - 0.32, contribution.evidence, ha="left", va="top", fontsize=PANEL_FONT, color=MUTED, linespacing=1.35)
        y -= 0.55 + 0.3 * lines
    return y


def draw(chart: Flowchart) -> plt.Figure:
    """The flowchart: boxes coloured by status, contribution boxes outlined in red, the contribution lines below."""
    check(chart)
    by_key = {box.key: box for box in chart.boxes}
    right = max(box.x + box.width for box in chart.boxes)
    top = max(box.y + box.height for box in chart.boxes)
    bottom = min(box.y for box in chart.boxes)
    panel_lines = sum(c.evidence.count("\n") + 1 for c in chart.contributions)
    panel_height = 1.3 + 0.55 * len(chart.contributions) + 0.3 * panel_lines + 0.3 * (chart.footer.count("\n") + 1)
    width, height = right + 0.5, top - bottom + 1.9 + panel_height + (0 if right >= 14.5 else 0.3)
    fig, axis = plt.subplots(figsize=(width, height))
    if chart.frame_tag:
        left = min(box.x for box in chart.boxes) - 0.12
        axis.add_patch(FancyBboxPatch((left, bottom - 0.15), right - left + 0.12, top - bottom + 0.75, boxstyle="round,pad=0.05",
                                      fc="none", ec=CONTRIBUTION_COLOUR, lw=1.6, ls=(0, (6, 4))))
        axis.text(right - 0.05, top + 0.68, f"{chart.frame_tag}: the whole system", ha="right", va="center", fontsize=9, weight="bold",
                  color="white", bbox={"boxstyle": "round,pad=0.25", "fc": CONTRIBUTION_COLOUR, "ec": "none"})
    for arrow in chart.arrows:
        _draw_arrow(axis, by_key[arrow.start], by_key[arrow.end], arrow)
    for box in chart.boxes:
        _draw_box(axis, box)
    for y, label in chart.lanes:
        axis.text(-0.1, y, label, fontsize=10.5, weight="bold", color=INK)
    for x, y, text in chart.notes:
        axis.text(x, y, text, fontsize=8, style="italic", color=CONTRIBUTION_COLOUR, va="top")
    for x, y, text in chart.hints:
        axis.text(x, y, text, fontsize=8, style="italic", color=MUTED, va="top")
    used = {box.status for box in chart.boxes}  # the legend shows only the colours the chart uses
    handles = [plt.Line2D([], [], marker="s", ls="", ms=10, color=fill, mec=EDGE, label=label)
               for status, (fill, label) in STATUS.items() if status in used]
    if any(box.fed_by_oracle for box in chart.boxes):
        handles.append(FancyBboxPatch((0, 0), 1, 1, fc="white", ec=FED_HATCH, hatch="////", lw=0.8, label="hatched: runs on the ORACLE map"))
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=10, mfc="white", mec=CONTRIBUTION_COLOUR, mew=2.2,
                              label=chart.contribution_label))
    legend_y = bottom - 0.45
    columns = len(handles) if right >= 14.5 else (len(handles) + 1) // 2  # a narrow chart gets the legend on two lines
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.1, legend_y), bbox_transform=axis.transData, ncol=columns,
                fontsize=8.5, frameon=False, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
    end = _draw_panel(axis, chart, legend_y - (0.75 if right >= 14.5 else 1.05))
    if chart.footer:
        axis.text(-0.1, end - 0.05, chart.footer, fontsize=8.8, style="italic", color=INK, va="top", linespacing=1.4)
    axis.text(-0.1, top + 1.15, chart.title, fontsize=12, weight="bold", color=INK)
    axis.set_xlim(-0.3, right + 0.2)
    axis.set_ylim(top + 1.5 - height, top + 1.5)
    axis.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


# ---------------------------------------------------------------- the memory pipeline after phase 1 (2026-11-13)

WRITE_Y, READ_Y, BUS_Y = 4.6, 1.3, 2.85
STAGE_1 = Flowchart(
    title="Functional memory pipeline after phase 1 (2026-11-13): what each block is today, and where a paper's contribution would lie",
    boxes=(
        Box("video", 0.0, WRITE_Y, "Video (tablet)\nRGB + device pose\n+ LiDAR depth", "real"),
        Box("keyframes", 2.1, WRITE_Y, "1  Keyframes\n40 steadiest frames", "real"),
        Box("depth", 4.2, WRITE_Y, "2  Depth per keyframe\nmonocular model,\nscaled by LiDAR", "model"),
        Box("affordance", 6.3, WRITE_Y, "4  Affordance map\nwhere can one act,\nwith which action?", "oracle", "C1"),
        Box("tokens", 8.4, WRITE_Y, "9  Part tokens on the\naffordance regions\n+ flag: keyframe\nholds one", "real", "C1",
            fed_by_oracle=True),
        Box("link", 10.5, WRITE_Y, "6  Link tokens\nacross keyframes", "real", fed_by_oracle=True),
        Box("memory", 12.6, 4.4, "MEMORY\nkeyframes, depth, pose\n+ part tokens\n+ thin index\n(time, keyframe, pixel)", "real",
            width=2.3, height=1.55, fed_by_oracle=True),
        Box("events", 10.5, 3.05, "15  Event tokens\nwhat was done, when", "missing", "C2", height=0.95),
        Box("query", 0.0, READ_Y, "Question in words\n\"open the top drawer\"", "real"),
        Box("search", 2.4, READ_Y, "10a  Find keyframes\nwhole-frame meaning\n+ keyframe flag\n+ part tokens", "real", "C1",
            width=2.1, fed_by_oracle=True),
        Box("reader", 4.9, READ_Y, "10b  Vision-language\nmodel picks one box\non an operable part\nin the top-3 keyframes", "model",
            "C1", width=2.1, fed_by_oracle=True),
        Box("geometry", 7.4, READ_Y, "7-8  Contact, normal,\nstand point computed\nnow from one keyframe\n(design choice)", "real",
            width=2.1, fed_by_oracle=True),
        Box("answer", 9.9, READ_Y, "Answer\nkeyframe + part\n+ where to touch", "real", width=1.9, fed_by_oracle=True),
        Box("humanoid", 12.2, READ_Y, "11-12  Humanoid\nwalks up and acts\n(supplementary demo)", "real", width=1.9, fed_by_oracle=True),
    ),
    arrows=(
        Arrow("video", "keyframes"), Arrow("keyframes", "depth"), Arrow("depth", "affordance"), Arrow("affordance", "tokens"),
        Arrow("tokens", "link"), Arrow("link", "memory"), Arrow("events", "memory", dashed=True),
        Arrow("query", "search"), Arrow("search", "reader"), Arrow("reader", "geometry"), Arrow("geometry", "answer"),
        Arrow("answer", "humanoid"),
        Arrow("memory", "search", dashed=True, via_y=BUS_Y), Arrow("memory", "reader", dashed=True, via_y=BUS_Y),
        Arrow("memory", "geometry", dashed=True, via_y=BUS_Y),
    ),
    contributions=(
        Contribution("C0", "The system: write once from a tablet video, then answer many questions about operable parts, "
                     "with no fused 3D map built or stored",
                     "Not measured yet: no comparison with existing methods; no cost curve over the number of questions; no use case "
                     "named yet. The stored keyframes and depth\n(about 27 MB per scene) are not smaller than a 1 cm point cloud "
                     "(about 23 MB). Against T-FunS3D, which also writes once (from a point cloud), the case needs\nwrite time, "
                     "question time, storage and accuracy on the same input.", "no evidence yet"),
        Contribution("C1", "Index by function: the affordance map flags keyframes and places part tokens; the model then "
                     "chooses among boxes on operable parts",
                     "Phase 1, equal budget: flag only +9.0, tokens only +17.4 (still partly the flag), both +20.5 points over whole "
                     "keyframes; within the same keyframes\n+7.0 over generic tokens, depending on the token weight. End-to-end 54% vs "
                     "10% (generic regions), mostly because the boxes are the annotated parts.\nViews chosen by function per question "
                     "exist (TASA; Fun3DU and UniFunc3D reprocess the scene per question); a flag written once pays off only if many\n"
                     "questions follow (phase 2).", "ORACLE upper bound, not yet evidence for a method"),
        Contribution("C2", "Remember what happened (\"4D\"): event tokens with time",
                     "Not built: SceneFun3D videos show nobody acting. Phase 3 on HD-EPIC; gate G3 decides whether \"4D\" stays "
                     "in the title.", "plan only"),
        Contribution("", "Design choice, not claimed: geometry from one keyframe at question time, no reconstruction",
                     "Naive fusion of the same depth maps is 0.26 cm worse; contacts within 3 cm: 38.5% of 481 true parts, "
                     "31.6% of the 760 parts the memory's own linking makes (different bases).", "supporting", kind="design"),
    ),
    lanes=((WRITE_Y + 1.45, "WRITE  (once per video)"), (READ_Y + 1.35, "READ  (once per question)")),
    notes=((6.3, WRITE_Y - 0.12, ("phase 2 replaces the annotations by a model:\nthe biggest risk (keeping 1/4 of the regions\n"
                                  "leaves +6.5 points over whole keyframes)")),),
    hints=((4.2, BUS_Y + 0.28, "dashed: reading looks things up in the memory"),),
    footer="Still missing for a paper: a vision-language model that finds the part in the same keyframes without this memory "
           "(the plainest baseline; it can refute C1, not prove it);\ncomparison with existing methods on the official metric "
           "and with T-FunS3D on the same input; the cost against the number of questions; a named use case;\nordinary objects; "
           "a second dataset; failure analysis.",
    frame_tag="C0",
)


# ---------------------------------------------------------------- the paper the project lead described (round 8, to confirm)

T_WRITE_Y, T_READ_Y, T_BASE_Y, T_BUS_Y = 6.0, 3.3, 0.6, 4.95
TARGET_R8 = Flowchart(
    title="The paper you described (2026-11-13), as I read it: to be confirmed, not a plan yet",
    boxes=(
        Box("video", 0.0, T_WRITE_Y, "Video (tablet)\nRGB + device pose\n+ LiDAR depth", "real"),
        Box("keyframes", 2.1, T_WRITE_Y, "1  Keyframes", "real"),
        Box("depth", 4.2, T_WRITE_Y, "2  Depth per keyframe", "model"),
        Box("affordance", 6.3, T_WRITE_Y, "4  Build the affordance map\nYOUR NEW METHOD (*)\ntoday: annotations (ORACLE)",
            "missing", "CA", width=2.1),
        Box("index", 8.8, T_WRITE_Y, "Functional index\npart tokens + keyframe flags\n+ action types", "real", "CB", width=2.1),
        Box("memory", 11.3, 5.8, "MEMORY\nwritten once,\nused for every question", "real", width=2.3, height=1.55),
        Box("question", 0.0, T_READ_Y, "Question in words", "real"),
        Box("search", 2.4, T_READ_Y, "Search by function\n(not a VLM reading\nevery frame)", "real", "CB", width=2.1),
        Box("reader", 4.9, T_READ_Y, "Small VLM looks at\na few candidates", "model", width=2.1),
        Box("answer", 7.4, T_READ_Y, "Answer: keyframe, part\n+ where and how to act", "real", width=2.1),
        Box("base_input", 0.0, T_BASE_Y, "Question\n+ the raw video", "real"),
        Box("base_vlm", 2.4, T_BASE_Y, "Traditional vision-language\nmodel reads the frames\nof the video for every question",
            "model", width=2.6),
        Box("base_answer", 5.4, T_BASE_Y, "Answer: frame + box", "real", width=2.1),
    ),
    arrows=(
        Arrow("video", "keyframes"), Arrow("keyframes", "depth"), Arrow("depth", "affordance"), Arrow("affordance", "index"),
        Arrow("index", "memory"),
        Arrow("question", "search"), Arrow("search", "reader"), Arrow("reader", "answer"),
        Arrow("memory", "search", dashed=True, via_y=T_BUS_Y), Arrow("memory", "reader", dashed=True, via_y=T_BUS_Y),
        Arrow("base_input", "base_vlm"), Arrow("base_vlm", "base_answer"),
    ),
    contributions=(
        Contribution("CA", "A new way to build the affordance map from the video (part level, with action types)",
                     "Not built yet. Phase 1 used the annotations, and the gain shrinks fast when parts are missed (keeping 1/4 of "
                     "the regions leaves +6.5 points),\nso the method must find most parts. What would be new has to be checked "
                     "against existing work first.", "to be built"),
        Contribution("CB", "Search by function, instead of a VLM reading the whole video for each question",
                     "Phase 1, with annotations only: the keyframe flag +9.0 points; part tokens +7.0 within the same keyframes "
                     "(depends on the scoring weight);\nencoding the question takes 10-20 ms and ranking under 1 ms.",
                     "ORACLE upper bound only"),
        Contribution("CC", "More accurate and faster than a traditional VLM reading the raw video",
                     "Not measured. Control c (approved, runs first) gives the VLM's numbers now; the fair comparison needs our "
                     "own affordance maps.", "no evidence yet"),
        Contribution("CD", "Especially strong on action-related questions",
                     "Not measured. Which questions count as 'action-related' needs your confirmation; the action types in the "
                     "map were not used by the search in phase 1.", "no evidence yet"),
    ),
    lanes=((T_WRITE_Y + 1.45, "WRITE  (once per video): ours"), (T_READ_Y + 1.35, "READ  (once per question): ours"),
           (T_BASE_Y + 1.35, "BASELINE: a traditional VLM reads the raw video (once per question)")),
    notes=((9.9, T_READ_Y - 0.25, ("CC: compare the two answers on accuracy,\ntime per question, and total cost\n"
                                   "over many questions on one scene")),
           (9.9, T_READ_Y + 1.2, "CD: action-related questions (meaning to confirm)")),
    hints=((4.2, T_BUS_Y + 0.28, "dashed: reading looks things up in the memory"),),
    footer="This is my reading of your words, to be confirmed. (*) marks a point where I inferred something; "
           "the document lists them as questions.",
    contribution_label="red: the contribution you described (my reading)",
    panel_title="The contribution you described, as I read it; [brackets] = the evidence so far, as I judge it",
)

TARGET_R9 = Flowchart(
    title="The paper you described, after the survey and control c (round 9): evidence and novelty of each claim",
    boxes=TARGET_R8.boxes,
    arrows=TARGET_R8.arrows,
    contributions=(
        Contribution("CA", "A new way to build the affordance map from the video (part level, with action types)",
                     "Not built. FunGraph (IROS 2025) trains a part detector with action labels on frames; Functional-SLAM (CoRL 2026) "
                     "builds part nodes online\nand merges them across frames. Simulation: misses cost more than false alarms; with 1/4 "
                     "of the parts left, small ones dropped first, the gain is gone.", "largely taken; weakest claim"),
        Contribution("CB", "Search by function, instead of a VLM reading the whole video for each question",
                     "Control c: 53.1% (annotated maps) vs 28.5% for a VLM reading 40 raw frames; the lead comes mainly from the "
                     "annotated boxes,\nthe memory's own share is not measured yet. FunGraph, KeySG and T-FunS3D already search a "
                     "prebuilt functional index.",
                     "partly taken; ORACLE upper bound only"),
        Contribution("CC", "More accurate and faster than a traditional VLM reading the raw video",
                     "1.2 s and 3 images vs 5.7 s and 41 images per question (writing not counted). 'Write once, query many' "
                     "exists for long video;\nopen on SceneFun3D parts. Aligning with other papers needs the official 3D metric "
                     "and Qwen3-VL-8B.", "idea taken; result on parts open"),
        Contribution("CD", "Especially strong on action-related questions",
                     "Retrieval never used the action key. Gap to the raw-video VLM: knobs +50, sockets +35, door handles +33, "
                     "others +19 to +28, unplug -10 (n = 10);\nno difference between types (p = 0.16). No precedent found for A; "
                     "C partly taken (AffordBot, AFUN); B taken at object level.", "A no support yet; C partly; B taken"),
    ),
    lanes=TARGET_R8.lanes,
    notes=((9.9, T_READ_Y - 0.25, ("CC: compare the two answers on accuracy,\ntime per question, and total cost\n"
                                   "over many questions on one scene")),
           (9.9, T_READ_Y + 1.2, "CD: A and C, with B alongside (confirmed in round 8)")),
    hints=TARGET_R8.hints,
    footer="Novelty from the round-9 survey (method sections or abstracts; no paper read in full). Numbers of our memory use "
           "annotated affordance maps (upper bounds); the raw-video VLM's are not upper bounds (a 4B model).",
    contribution_label="red: the contribution you described",
    panel_title="Each claim after the survey; [brackets] = novelty and evidence as I judge them, not a result",
)

CHARTS = {"stage_1": STAGE_1, "target_r8": TARGET_R8, "target_r9": TARGET_R9}


def main(argv: Sequence[str]) -> None:
    if len(argv) not in (1, 2) or (len(argv) == 2 and argv[1] not in CHARTS):
        raise SystemExit(f"usage: python -m report.pipeline_flowchart <out.png> [{' | '.join(CHARTS)}]")
    out = Path(argv[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(CHARTS[argv[1] if len(argv) == 2 else "stage_1"]).savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
