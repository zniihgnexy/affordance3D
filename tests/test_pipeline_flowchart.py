from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import FancyBboxPatch

from report import pipeline_flowchart as F

SMALL = F.Flowchart(
    title="two boxes",
    boxes=(F.Box("a", 0.0, 1.0, "input", "real"), F.Box("b", 2.0, 1.0, "method", "oracle", "C1")),
    arrows=(F.Arrow("a", "b"),),
    contributions=(F.Contribution("C1", "the idea", "what shows it", "weak"),),
)


def test_draw_gives_one_box_per_block_and_a_tag_per_contribution_box() -> None:
    figure = F.draw(SMALL)

    axis = figure.axes[0]
    boxes = [patch for patch in axis.patches if isinstance(patch, FancyBboxPatch)]
    texts = [text.get_text() for text in axis.texts]
    assert len(boxes) == 2
    assert texts.count("C1") == 2  # on the box and in the contribution lines
    assert boxes[1].get_edgecolor()[:3] != boxes[0].get_edgecolor()[:3]  # the contribution box is outlined differently
    plt.close(figure)


@pytest.mark.parametrize(("broken", "message"), [
    (replace(SMALL, boxes=(SMALL.boxes[0], replace(SMALL.boxes[1], status="done"))), "unknown status"),
    (replace(SMALL, arrows=(F.Arrow("a", "z"),)), "unknown boxes"),
    (replace(SMALL, contributions=()), "without a contribution"),
    (replace(SMALL, boxes=(SMALL.boxes[0], replace(SMALL.boxes[1], key="a"))), "unique"),
])
def test_check_refuses_a_description_that_cannot_be_drawn_faithfully(broken: F.Flowchart, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        F.check(broken)


def test_the_stage_1_chart_is_consistent_and_renders(tmp_path: Path) -> None:
    F.check(F.STAGE_1)
    out = tmp_path / "pipeline.png"

    F.main([str(out)])

    assert out.stat().st_size > 50_000
    placed = {box.tag for box in F.STAGE_1.boxes if box.tag} | {F.STAGE_1.frame_tag}
    claimed = {c.tag for c in F.STAGE_1.contributions if c.kind == "contribution"}
    assert placed == claimed  # every claimed contribution is placed on the chart, and nothing else is marked red
    with pytest.raises(SystemExit):
        F.main([])


def test_a_frame_tag_and_a_design_line_are_drawn_and_checked() -> None:
    chart = replace(SMALL, frame_tag="C0", contributions=(*SMALL.contributions, F.Contribution("C0", "system", "none", "none"),
                                                          F.Contribution("", "a choice", "why", "weak", kind="design")))
    figure = F.draw(chart)

    texts = [text.get_text() for text in figure.axes[0].texts]
    assert "C0: the whole system" in texts and "design" in texts  # the frame tag, and the untagged design line
    plt.close(figure)
    with pytest.raises(ValueError, match="frame"):
        F.check(replace(SMALL, frame_tag="C9"))
    with pytest.raises(ValueError, match="kind"):
        F.check(replace(SMALL, contributions=(F.Contribution("C1", "x", "y", "z", kind="other"),)))


def test_every_named_chart_is_consistent_and_can_be_chosen_by_name(tmp_path: Path) -> None:
    for name, chart in F.CHARTS.items():
        F.check(chart)
        F.main([str(tmp_path / f"{name}.png"), name])
        assert (tmp_path / f"{name}.png").stat().st_size > 50_000
    with pytest.raises(SystemExit):
        F.main([str(tmp_path / "x.png"), "no_such_chart"])
