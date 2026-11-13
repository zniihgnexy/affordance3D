import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_mem_c1.py"
spec = importlib.util.spec_from_file_location("aggregate_mem_c1", SCRIPT)
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def _score(contact: bool = True, kind: bool = True, line_ok: bool = True, **rest: object) -> dict:
    score = dict.fromkeys(("axis_class", "hinge_side", "vertical_axis_class", "vertical_line_ok", "sense", "line_2d_deg", "signed_deg"))
    score.update(contact=contact, label=True, type=kind, line_ok=line_ok, line_deg=5.0 if line_ok else 40.0,
                 joint=contact and kind and line_ok)
    score.update(rest)
    return score


def _records(arms: dict[str, list[dict]], scenes: int = 3) -> list[dict]:
    """The same per-task scores in every scene."""
    n = len(next(iter(arms.values())))
    return [{"visit": str(v), "tasks": ["t"] * n, "arms": {arm: [{"score": s} for s in rows] for arm, rows in arms.items()}}
            for v in range(scenes)]


def test_the_memory_far_ahead_of_the_plain_model_in_direction_makes_c_stand_alone() -> None:
    good = [_score() for _ in range(4)]
    records = _records({"V": [_score(line_ok=False) for _ in range(4)], "G": good, "M": good, "O": good})

    summary = agg.summarise(records)

    readings = summary["readings"]
    joint = readings["M_minus_V_joint_with_annotation"]
    assert joint["rate"] == pytest.approx(1.0) and joint["low"] == pytest.approx(1.0) and "annotation" in readings["M_minus_V_note"]
    assert readings["C_stands_alone"] and not readings["C_merges_into_A"]
    assert readings["G_vs_M"] == "close" and readings["direction_from_geometry"]
    assert summary["scorer_check"]["holds"] and summary["arms"]["V"]["joint"]["rate"] == 0.0


def test_a_plain_model_as_good_as_the_memory_merges_c_into_a() -> None:
    good = [_score() for _ in range(4)]
    bad = [_score(contact=False, line_ok=False) for _ in range(4)]

    readings = agg.summarise(_records({"V": good, "G": bad, "M": good}))["readings"]

    assert readings["C_merges_into_A"] and not readings["C_stands_alone"]
    assert readings["G_vs_M"] == "behind" and not readings["direction_from_geometry"]


def test_the_memory_value_is_m_minus_m4_read_as_in_section_24_3() -> None:
    good = [_score() for _ in range(4)]
    half = [_score(), _score(), _score(contact=False), _score(contact=False)]

    same = agg.summarise(_records({"M": good, "M4": good}))["readings"]
    ahead = agg.summarise(_records({"M": good, "M4": half}))["readings"]

    assert same["memory_value_verdict"] == "close" and same["memory_value_M_minus_M4_joint"]["rate"] == 0.0
    assert ahead["memory_value_verdict"] == "behind" and ahead["memory_value_M_minus_M4_joint"]["rate"] == pytest.approx(0.5)
    assert ahead["memory_value_reading"].startswith("M4 behind M") and "C_stands_alone" not in ahead


def test_direction_is_compared_only_where_both_points_are_on_the_part() -> None:
    m = [_score(), _score(), _score(contact=False), _score(contact=False)]
    v = [_score(line_ok=False), _score(), _score(contact=False, line_ok=False), _score(line_ok=False)]

    direction = agg.summarise(_records({"V": v, "M": m}))["readings"]["M_minus_V_direction_both_on_part_with_annotation"]

    assert direction["n"] == 6 and direction["rate"] == pytest.approx(0.5)  # 2 tasks per scene, M right on both, V on one


def test_fields_that_do_not_apply_are_left_out_and_a_wrong_truth_fails_the_check() -> None:
    o = [_score(axis_class=True, hinge_side=True), _score(sense=True), _score(sense=False)]

    summary = agg.summarise(_records({"O": o}))

    assert summary["arms"]["O"]["axis_class"]["n"] == 3 and summary["arms"]["O"]["sense"]["rate"] == pytest.approx(0.5)
    assert not summary["scorer_check"]["holds"] and summary["scorer_check"]["fields"]["contact"]["rate"] == 1.0
    assert summary["arms"]["O"]["median_line_deg"] == 5.0 and "C_stands_alone" not in summary["readings"]


def test_the_share_of_answers_whose_fields_could_be_read_is_reported() -> None:
    records = _records({"V": [_score(), _score()]})
    records[0]["arms"]["V"][0]["prediction"] = {"point": (1, 2), "label": None, "type": "trans", "arrow": None,
                                                "hinge_side": None, "direction": [0, 0, 1]}

    read = agg.summarise(records)["arms"]["V"]["read"]

    assert read["point"] == pytest.approx(1 / 6) and read["label"] == 0.0 and read["direction"] == pytest.approx(1 / 6)
