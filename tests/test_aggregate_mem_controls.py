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


def test_arm_b_is_the_per_task_average_of_its_five_seeds_and_c_plus_is_set_against_c3() -> None:
    def record() -> dict:
        arms = {"p4n": [_row(True), _row(False)], "c3": [_row(False), _row(False)], "cplus": [_row(True), _row(False)]}
        arms.update({f"b s{seed}": [_row(seed < 2), _row(False)] for seed in agg.B_SEEDS})
        return {"tasks": ["0", "1"], "arms": arms}

    records = [record() for _ in range(5)]

    summary = agg.summarise(records)

    assert summary["arms"]["b"]["right_part"]["rate"] == pytest.approx(0.2)  # 2 of 5 seeds right on one task of two
    assert summary["against_p4"]["b"]["rate"] == pytest.approx(0.3)
    assert summary["our_keyframes_vs_even_frames"]["rate"] == pytest.approx(0.5)
    assert summary["against_p4"]["cplus"]["verdict"] == "close" and "stop and discuss" in summary["against_p4"]["cplus"]["reading"]
    assert "b" not in records[0]["arms"]  # the records themselves are left as they were


def test_arm_b_is_left_out_when_a_seed_is_missing() -> None:
    record = {"tasks": ["0"], "arms": {"p4n": [_row(True)], "b s0": [_row(True)]}}

    assert agg.with_seed_average(record) is record


def test_round9_arms_are_read_by_their_fixed_rules_and_set_against_the_4b_model() -> None:
    def record(p4: list[bool], c4: list[bool], c1s: list[bool], c2s: list[bool]) -> dict:
        return {"visit": "1", "tasks": [str(i) for i in range(len(p4))],
                "arms": {"p4n": [_row(h) for h in p4], "c4": [_row(h) for h in c4],
                         "c1s": [_row(h) for h in c1s], "c2s": [_row(h) for h in c2s]}}

    records = [record([True, True, False, False], [True, True, False, False], [False, False, False, False], [True, False, False, False])
               for _ in range(5)]
    smaller = [record([True, False, False, False], [False, False, False, False], [False] * 4, [False] * 4) for _ in range(5)]

    summary = agg.summarise(records)

    assert summary["against_p4"]["c4"]["verdict"] == "close" and "stop and discuss" in summary["against_p4"]["c4"]["reading"]
    assert summary["choosing_keyframes_same_reading"]["rate"] == pytest.approx(0.25)
    assert summary["choosing_keyframes_same_reading"]["verdict"] == "worth something"
    assert agg.against_smaller(records, smaller)["p4n"]["rate"] == pytest.approx(0.25)
