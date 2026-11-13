import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_mem_controls.py"
spec = importlib.util.spec_from_file_location("aggregate_mem_controls", SCRIPT)
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def _row(hit: bool, frame_right: bool = True) -> dict:
    return {"hit": hit, "hit_old": hit, "frame_right": frame_right, "seconds": 2.0, "images_read": 3}


def _record(p4: list[bool], c1: list[bool], c2: list[bool]) -> dict:
    return {"tasks": [str(i) for i in range(len(p4))],
            "arms": {"p4n": [_row(h) for h in p4], "c1": [_row(h) for h in c1], "c2": [_row(h) for h in c2]}}


def test_summary_compares_every_arm_with_p4_on_the_same_tasks_and_reads_it_by_the_fixed_rule() -> None:
    records = [_record([True, True, True, False], [False, False, True, False], [True, True, False, False]) for _ in range(6)]

    summary = agg.summarise(records)

    assert summary["scenes"] == 6 and summary["tasks"] == 24 and set(summary["arms"]) == {"p4n", "c1", "c2"}
    assert summary["arms"]["p4n"]["right_part"]["rate"] == pytest.approx(0.75)
    assert summary["against_p4"]["c1"]["rate"] == pytest.approx(0.5) and summary["against_p4"]["c1"]["verdict"] == "behind"
    assert summary["against_p4"]["c2"]["rate"] == pytest.approx(0.25)
    assert summary["choosing_keyframes"]["rate"] == pytest.approx(0.25)  # c2 minus c1
    assert "room for phase 2" in summary["against_p4"]["c1"]["reading"]


def test_an_arm_close_to_p4_stops_the_work_and_an_arm_missing_somewhere_is_left_out() -> None:
    records = [_record([True, False], [True, False], [True, False]) for _ in range(5)]
    records[0]["arms"].pop("c2")

    summary = agg.summarise(records)

    assert "c2" not in summary["arms"] and summary["against_p4"]["c1"]["verdict"] == "close"
    assert "stop and discuss" in summary["against_p4"]["c1"]["reading"]
    assert agg.reading("c3", "undetermined") == agg.UNDETERMINED
