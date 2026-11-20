"""The pure helpers of the Rerun export (report/week2_rrd.py): gravity alignment, downsampling, tags, arrow geometry."""
from types import SimpleNamespace

import numpy as np
import pytest

W = pytest.importorskip("report.week2_rrd", reason="needs rerun-sdk (requirements-rerun.txt)")


@pytest.mark.parametrize("up", [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.3, -0.2, 0.93], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
def test_gravity_rotation_turns_up_into_plus_z_without_mirroring(up):
    rotation = W.gravity_rotation(np.array(up))
    assert rotation @ (np.array(up) / np.linalg.norm(up)) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_world_frame_puts_the_floor_at_zero_and_centres_the_scan():
    rng = np.random.default_rng(0)
    up = np.array([0.0, 1.0, 0.0])  # a scan whose gravity is along +Y, floor at y = 5
    xyz = np.column_stack([rng.uniform(2, 4, 2000), rng.uniform(5, 7.5, 2000), rng.uniform(-1, 1, 2000)])
    rotation, offset = W.world_frame(xyz, up, floor_percentile=0.0)
    world = W.to_world(xyz, rotation, offset)
    assert world[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert world[:, 2].max() == pytest.approx(xyz[:, 1].max() - 5, abs=0.01)
    assert np.median(world[:, 0]) == pytest.approx(0.0, abs=1e-9) and np.median(world[:, 1]) == pytest.approx(0.0, abs=1e-9)


def test_downsample_keeps_everything_that_fits_and_a_fixed_sorted_subset_otherwise():
    assert list(W.downsample(5, 10, 0)) == [0, 1, 2, 3, 4]
    picked = W.downsample(1000, 100, 0)
    assert len(picked) == 100 and len(set(picked)) == 100 and list(picked) == sorted(picked)
    assert list(picked) == list(W.downsample(1000, 100, 0))


def test_label_text_and_entity_names():
    assert W.label_text("top oven door", "pull") == "top oven door ▶ pull"
    assert W.slug("Left balcony door's lock") == "left_balcony_door_s_lock"


def test_loop_times_go_out_and_back_once_per_period():
    times = W.loop_times(1.6, 16, 2)
    assert len(times) == 32
    assert times[0] == (0.0, pytest.approx(0.0)) and times[8] == (pytest.approx(0.8), pytest.approx(1.0))
    assert times[16][1] == pytest.approx(0.0) and times[1][0] == pytest.approx(0.1)


def test_arrow_parts_split_a_straight_arrow_and_an_arc():
    strip, origin, vector = W.arrow_parts(np.array([[0.0, 0, 0], [0, 0, 1]]))
    assert strip is None and list(origin) == [0, 0, 0] and list(vector) == [0, 0, 1]
    arc = np.array([[0.0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 3, 0]])
    strip, origin, vector = W.arrow_parts(arc)
    assert strip is arc and list(origin) == [1, 0, 0] and list(vector) == [2, 3, 0]


def test_axis_strip_passes_the_axis_point_nearest_the_contact():
    door = SimpleNamespace(kind="rot", axis=np.array([0.0, 0.0, 2.0]), origin=np.zeros(3), contact=np.array([0.7, 0.0, 1.0]))
    strip = W.axis_strip(door, True, 0.06, 0.4)
    assert strip == pytest.approx(np.array([[0, 0, 0.6], [0, 0, 1.4]]))
    assert W.axis_strip(door, False, 0.06, 0.4) == pytest.approx(np.array([[0, 0, 0.94], [0, 0, 1.06]]))
    assert W.axis_strip(SimpleNamespace(kind="trans", axis=np.ones(3), origin=np.zeros(3), contact=np.zeros(3)), False, 0.06, 0.4) is None
