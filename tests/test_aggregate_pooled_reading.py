import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_pooled_reading.py"
spec = importlib.util.spec_from_file_location("aggregate_pooled_reading", SCRIPT)
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)

N_SCENES = 20  # enough for the draws at K = 16
SCENES = [str(400000 + i) for i in range(N_SCENES)]


def _rows(hits: list[int], visit: str) -> list[dict]:
    """Records of one arm: a miss is a pick on another scene."""
    return [{"hit": bool(h), "top3_right": True, "a_box_right": True, "frame": f"{visit if h else '999999'}/t1",
             "chosen_scene": visit if h else "999999", "keyframes_shown": [f"{visit}/t1", f"{visit}/t2"], "marks": 6} for h in hits]


def _record(index: int, hits: dict[int, dict[str, list[int]]]) -> dict:
    visit = SCENES[index]
    draws = agg.READER.first_draws(N_SCENES, index)
    n_tasks = len(hits[1]["semantic"])
    return {"visit": visit, "scenes": SCENES, "version": agg.READER.VERSION, "model": "stand-in", "complete": True,
            "tasks": [f"task {t}" for t in range(n_tasks)],
            "draws": {str(k): [SCENES[i] for i in others] for k, others in draws.items()},
            "per_size": {str(k): {arm: _rows(h, visit) for arm, h in by_arm.items()} for k, by_arm in hits.items()}}


def _records(gap_tasks: dict[int, int], n_tasks: int = 25, n_scenes: int = 8,
             jitter: Callable[[int, int], int] = lambda index, k: 0) -> list[dict]:
    """Every scene: semantic right on the first 10 tasks; the function index right on `gap_tasks[k]` more (fewer when
    negative), plus `jitter(scene index, k)`."""
    records = []
    for index in range(n_scenes):
        hits = {}
        for k, extra in gap_tasks.items():
            right = 10 + extra + jitter(index, k)
            hits[k] = {"semantic": [1] * 10 + [0] * (n_tasks - 10), "function index": [1] * right + [0] * (n_tasks - right)}
        records.append(_record(index, hits))
    return records


def _odd(index: int, k: int) -> int:
    return index % 2


def test_a_gap_that_grows_with_the_pool_and_is_large_at_16_is_supported() -> None:
    result = agg.reading(_records({1: 1, 4: 4, 16: 7}, jitter=_odd))

    assert result["slope_per_doubling"]["low"] > 0 and result["gap_at_largest"] >= 0.25
    assert result["supported"] and result["verdict"] == "supported end to end"


def test_a_gap_that_stays_the_same_is_not_supported_however_large() -> None:
    # even scenes gain a task at K = 16, odd scenes at K = 1: no trend overall, either sign in a resample
    records = _records({1: 5, 4: 5, 16: 5}, jitter=lambda index, k: int((index % 2, k) in ((0, 16), (1, 1))))

    result = agg.reading(records)

    assert result["slope_per_doubling"]["rate"] == pytest.approx(0.0, abs=1e-9) and result["slope_per_doubling"]["low"] < 0
    assert result["gap_at_largest"] > 0.05
    assert not result["supported"] and result["verdict"].startswith("not supported end to end either")


def test_a_growing_gap_below_five_points_at_16_is_not_supported() -> None:
    result = agg.reading(_records({1: -1, 4: 0, 16: 1}))  # one task in 25: 4 points

    assert result["slope_per_doubling"]["low"] > 0 and result["gap_at_largest"] == pytest.approx(0.04)
    assert not result["supported"]


def test_size_entry_gives_the_rates_the_paired_gap_and_the_descriptive_shares() -> None:
    entry = agg.size_entry(_records({1: 0, 4: 2, 16: 5}, n_tasks=20, n_scenes=4, jitter=_odd), 16)

    assert entry["semantic"]["hit"]["rate"] == pytest.approx(0.5)
    assert entry["function index minus semantic"]["rate"] == pytest.approx(0.25 + 0.5 / 20)
    assert entry["semantic"]["on_another_scene"] == pytest.approx(0.5) and entry["semantic"]["no_pick"] == 0.0
    assert entry["semantic"]["top3_right"] == 1.0


def test_check_records_keeps_complete_records_lists_the_missing_scenes_and_refuses_other_draws() -> None:
    records = _records({1: 0, 4: 0, 16: 0}, n_scenes=3)
    unfinished = {**_records({1: 0, 4: 0, 16: 0}, n_scenes=4)[3], "complete": False}

    complete, missing = agg.check_records([*records, unfinished])

    assert [r["visit"] for r in complete] == SCENES[:3] and missing == SCENES[3:]
    tampered = {**records[1], "draws": {**records[1]["draws"], "4": list(reversed(records[1]["draws"]["4"]))}}
    with pytest.raises(SystemExit):
        agg.check_records([records[0], tampered])
    with pytest.raises(SystemExit):
        agg.check_records([records[0], {**records[1], "model": "another"}])


def test_k1_check_compares_task_by_task_with_the_single_scene_controls(tmp_path: Path) -> None:
    records = _records({1: 2, 4: 2, 16: 2}, n_tasks=20, n_scenes=2, jitter=_odd)
    for record in records:
        visit = record["visit"]
        p4n = [{"hit": row["hit"], "keyframes_shown": ["t1", "t2"], "marks": 6} for row in record["per_size"]["1"]["function index"]]
        c4 = [{"hit": not row["hit"], "keyframes_shown": ["t1", "t2"], "marks": 5} for row in record["per_size"]["1"]["semantic"]]
        (tmp_path / f"{visit}.json").write_text(json.dumps({"tasks": record["tasks"], "arms": {"p4n": p4n, "c4": c4}}))

    check = agg.k1_check(records, tmp_path)

    ours = check["function index"]
    assert ours["control"] == "p4n" and ours["reference"] == 0.544 and ours["same_outcome"] == 1.0
    assert ours["same_question"] == 1.0 and ours["control_rate"] == pytest.approx(ours["rate"])
    assert check["semantic"]["same_outcome"] == 0.0 and check["semantic"]["same_question"] == 0.0  # other box counts
    assert "same_outcome" not in agg.k1_check(records, tmp_path / "absent")["semantic"]
