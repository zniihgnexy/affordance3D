import importlib.util
from pathlib import Path

import numpy as np
import pytest

from affordance3d.mem import action_records as A

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_mem_c1.py"
spec = importlib.util.spec_from_file_location("run_mem_c1", SCRIPT)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_c1_takes_the_c0_tasks_with_every_motion_and_keeps_their_place_among_the_controls() -> None:
    rows = [{"visit": "1", "task": "a", "has_all_motions": True}, {"visit": "2", "task": "x", "has_all_motions": True},
            {"visit": "1", "task": "b", "has_all_motions": False}, {"visit": "1", "task": "c", "has_all_motions": True}]

    assert [(i, row["task"]) for i, row in run.c1_tasks(rows, "1", ["a", "b", "c"])] == [(0, "a"), (2, "c")]
    with pytest.raises(SystemExit):
        run.c1_tasks(rows, "1", ["a", "c"])


def test_the_answer_key_changes_with_the_question_and_the_model() -> None:
    key = run.answer_key("action", "prompt", ["1.0"], [], "rev")

    assert key == run.answer_key("action", "prompt", ["1.0"], [], "rev") and len(key) == 16
    assert len({key, run.answer_key("action", "prompt", ["1.0"], [], "other"), run.answer_key("M'", "prompt", ["1.0"], [], "rev")}) == 3


def test_a_prediction_is_stored_as_plain_values() -> None:
    prediction = A.Prediction((3, 4), "tip_push", "trans", "push", (1, 0), None, None, np.array([0.123456789, 0.0, -1.0]))

    row = run.prediction_row(prediction)

    assert row == {"point": (3, 4), "label": "tip_push", "type": "trans", "sense": "push", "arrow": (1, 0), "hinge_side": None,
                   "direction": [0.12346, 0.0, -1.0]}
    assert run.prediction_row(None) is None
