"""The pure parts of scripts/simulate_overwrite.py: one part's outcome under each rule, and the summary shares."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("open3d")  # the script loads the stage-0 scene code, which imports open3d
_spec = importlib.util.spec_from_file_location(
    "simulate_overwrite", Path(__file__).resolve().parents[1] / "scripts" / "simulate_overwrite.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def test_newest_miss_erases_a_part_found_earlier():
    out = S.part_outcome([(3, 50, True), (9, 80, True), (20, 40, False)])
    assert out == {"shown": 3, "found": 2, "ever": True, "kept_last_view": False, "kept_largest_view": True}


def test_largest_view_ties_take_the_first():
    out = S.part_outcome([(1, 60, False), (2, 60, True)])
    assert out["kept_largest_view"] is False and out["kept_last_view"] is True


def test_part_never_found_is_not_counted_as_erased():
    parts = [S.part_outcome([(0, 10, False), (5, 30, False)]), S.part_outcome([(2, 10, True), (4, 30, False)])]
    summary = S.summarise(parts)
    assert summary["ever_found"] == 1
    assert summary["erased"]["last_view"]["all"] == 1.0
    assert summary["erased"]["largest_view"]["all"] == 1.0
    assert summary["per_frame_found_rate"] == 0.25 and summary["per_frame_found_rate_ever_found"] == 0.5


def test_part_without_sightings_is_rejected():
    with pytest.raises(ValueError):
        S.part_outcome([])
