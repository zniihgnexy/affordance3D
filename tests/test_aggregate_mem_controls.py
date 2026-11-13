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


def _tool_records(arms: dict[str, list[bool]], n: int = 6) -> list[dict]:
    return [{"visit": str(i), "tasks": ["0", "1", "2", "3"], "arms": {arm: [_row(h) for h in hits] for arm, hits in arms.items()}}
            for i in range(n)]


TOOL_ARMS = {"p4n": [True, True, True, False], "c4": [True, True, False, False], "c3": [False] * 4,
             "p4t_sam3": [True, True, False, False], "c4t_sam3": [True, False, False, False], "sam3q": [False] * 4,
             "p4tp_sam3": [True, True, True, False]}
GAIN = {"rate": 0.08, "low": 0.02, "high": 0.12}
TOOL_SUMMARY = {"deciding_tool": {"tool": "sam3", "reason": "tie"}, "tools": {"sam3": {"recall_gain_40_minus_10": GAIN}}}


def test_tool_arms_are_compared_with_p4_but_read_only_by_the_rules_of_section_24_9() -> None:
    summary = agg.summarise(_tool_records(TOOL_ARMS))

    assert {"p4t_sam3", "c4t_sam3", "sam3q", "p4tp_sam3"} <= set(summary["arms"])
    assert summary["against_p4"]["p4t_sam3"]["reading"] == agg.TOOL_READING
    assert summary["against_p4"]["c4"]["reading"] == agg.READINGS[("c4", summary["against_p4"]["c4"]["verdict"])]


def test_no_stop_when_the_tool_memory_beats_its_own_boxes_without_memory_c3_and_sam3_with_the_task_text() -> None:
    readings = agg.tool_map_readings(_tool_records(TOOL_ARMS), TOOL_SUMMARY)

    stop = readings["verdict"]["stop"]
    assert stop["tool_p4_minus_tool_c4"]["rate"] == pytest.approx(0.25) and stop["tool_p4_minus_tool_c4"]["verdict"] == "behind"
    assert not stop["stop"] and stop["reading"] == agg.GO_READING
    assert readings["annotated_p4_minus_annotated_c4"]["rate"] == pytest.approx(0.25)
    assert "not read" in readings["tools"]["qwen8b"]["stop"]["reading"]  # its arms did not run


@pytest.mark.parametrize("change", [{"c4t_sam3": [True, True, False, False]},  # same candidates without memory: close
                                    {"sam3q": [True, True, False, False]},  # SAM 3 with the task text as good
                                    {"c3": [True, True, True, False]}])  # the raw video better
def test_stop_when_the_tool_memory_is_close_to_its_boxes_without_memory_or_no_better_than_c3_or_sam3q(change: dict) -> None:
    stop = agg.tool_map_readings(_tool_records({**TOOL_ARMS, **change}), TOOL_SUMMARY)["verdict"]["stop"]

    assert stop["stop"] and stop["reading"] == agg.STOP_READING


def test_a_wide_interval_on_the_memory_comparison_alone_does_not_stop() -> None:
    records = _tool_records({**TOOL_ARMS, "p4t_sam3": [True] * 4})
    for i, record in enumerate(records):  # the difference is 0 in four scenes and 1 in two: mean 0.33, interval down to 0
        record["arms"]["c4t_sam3"] = [_row(i < 4) for _ in range(4)]

    stop = agg.tool_map_readings(records, TOOL_SUMMARY)["verdict"]["stop"]

    assert stop["tool_p4_minus_tool_c4"]["verdict"] == "undetermined" and not stop["stop"]


def test_the_gap_to_the_annotated_map_is_read_with_the_recall_gain_and_split_into_misses_and_false_alarms() -> None:
    records = _tool_records(TOOL_ARMS)

    readings = agg.tool_map_readings(records, TOOL_SUMMARY)["verdict"]

    assert readings["gap"]["annotated_p4_minus_tool_p4"]["rate"] == pytest.approx(0.25) and readings["gap"]["verdict"] == "large"
    assert readings["split"]["false_alarms"]["rate"] == pytest.approx(0.25) and readings["split"]["misses"]["rate"] == pytest.approx(0.0)
    assert readings["split"]["false_alarms"]["share_of_gap"] == pytest.approx(1.0)
    flat = {**TOOL_SUMMARY, "tools": {"sam3": {"recall_gain_40_minus_10": {"rate": 0.08, "low": -0.01, "high": 0.15}}}}
    assert agg.tool_map_readings(records, flat)["verdict"]["gap"]["verdict"] == "undetermined"
    level = agg.tool_map_readings(_tool_records({**TOOL_ARMS, "p4t_sam3": TOOL_ARMS["p4n"]}), TOOL_SUMMARY)
    assert level["verdict"]["gap"]["verdict"] == "small" and "suffice" in level["verdict"]["gap"]["reading"]


def test_without_a_deciding_tool_the_readings_are_kept_but_no_verdict_is_given() -> None:
    readings = agg.tool_map_readings(_tool_records(TOOL_ARMS), None)

    assert "verdict" not in readings and "no deciding tool" in readings["reading"]
    assert readings["tools"]["sam3"]["gap"]["verdict"] == "undetermined"  # no recall gain known: cannot be "large"
