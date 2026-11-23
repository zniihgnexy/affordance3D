"""report/week2_report.py (charts of the English week-2 report): the snapshot projection, the callout layout and the
related-work placement."""
import numpy as np
import pytest

from report import week2_report as WR


def test_top_down_view_drops_height_and_puts_the_top_nearest():
    uv, depth = WR.ortho_project(np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 3.0]]), azim_deg=0, elev_deg=90)
    assert np.allclose(uv[0], uv[1])  # same x, y: same pixel from above
    assert depth[1] > depth[0]  # the higher point is nearer the camera


def test_side_view_keeps_height_on_the_vertical_axis():
    uv, _ = WR.ortho_project(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 1.0, 0.0]]), azim_deg=0, elev_deg=0)
    assert uv[1, 1] - uv[0, 1] == pytest.approx(2.0)  # height becomes screen up
    assert uv[2, 0] - uv[0, 0] == pytest.approx(1.0)  # +y is screen right when looking from +x


def test_view_basis_is_orthonormal():
    towards, right, up = WR.view_basis(37, 25)
    basis = np.stack([towards, right, up])
    assert np.allclose(basis @ basis.T, np.eye(3))


def test_related_work_sits_on_the_chart_axes():
    for name, x, y, _ in WR.WORKS:
        assert -0.4 <= x <= len(WR.STORES) - 0.4 and -0.4 <= y <= len(WR.GRAINS) - 0.25, name
    assert len({name for name, *_ in WR.WORKS}) == len(WR.WORKS)


def test_callouts_go_to_the_near_margin_in_vertical_order():
    anchors = np.array([[0.0, 5.0], [1.0, 1.0], [9.0, 3.0], [8.0, 7.0], [0.5, 9.0]])
    out = WR.callout_positions(anchors, (0.0, 10.0), (0.0, 10.0), pad=1.0)
    left = [i for i, (x, _, ha) in enumerate(out) if x < 0]
    assert left == [0, 1, 4] and all(out[i][2] == "right" for i in left)
    assert all(out[i][0] == 11.0 and out[i][2] == "left" for i in (2, 3))
    ys = {i: out[i][1] for i in left}
    assert ys[4] > ys[0] > ys[1]  # same top-to-bottom order as the anchors
