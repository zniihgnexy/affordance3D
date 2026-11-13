import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_tool_maps.py"
spec = importlib.util.spec_from_file_location("run_tool_maps", SCRIPT)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_chosen_queries_keep_those_above_the_threshold_or_the_best_few_best_first() -> None:
    scores = np.array([0.2, 0.9, 0.55, 0.5, 0.7])

    assert run.chosen_queries(scores, 0.5, None).tolist() == [1, 4, 2]  # 0.5 itself does not pass, as in SAM 3
    assert run.chosen_queries(scores, None, 2).tolist() == [1, 4]


def test_pixel_box_takes_the_mask_and_falls_back_to_the_predicted_box() -> None:
    mask = np.zeros((16, 20), bool)
    mask[2:5, 3:9] = True

    assert run.pixel_box(mask, np.array([0.0, 0.0, 1.0, 1.0]), (16, 20)) == (3, 2, 9, 5)
    assert run.pixel_box(np.zeros((16, 20), bool), np.array([0.5, 0.25, 1.0, 0.5]), (16, 20)) == (10, 4, 20, 8)


def test_instances_score_as_sam3_and_turn_each_upsampled_mask_into_a_box() -> None:
    torch = pytest.importorskip("torch")
    masks = torch.full((1, 3, 8, 8), -10.0)
    masks[0, 0, :4, :4] = 10.0  # query 0: the top-left quarter
    outputs = SimpleNamespace(pred_logits=torch.tensor([[3.0, -1.0, 2.0]]), presence_logits=torch.tensor([[10.0]]),
                              pred_masks=masks, pred_boxes=torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0, 0, 1, 1], [0.5, 0.5, 1.0, 1.0]]]))

    found = run.instances(outputs, (16, 16), 0.5, None)

    assert [row["box"] for row in found] == [[0, 0, 8, 8], [8, 8, 16, 16]]  # query 2 has an empty mask: its predicted box
    assert found[0]["score"] == pytest.approx(0.9526, abs=1e-3) and found[1]["score"] == pytest.approx(0.8808, abs=1e-3)
    assert len(run.instances(outputs, (16, 16), None, 3)) == 3


class FakeReader:
    def __init__(self, answer: str) -> None:
        self.answer, self.keys = answer, []

    def ask(self, key: str, images: list, prompt: str, **options: object) -> dict:
        self.keys.append(key)
        return {"answer": self.answer, "seconds": 3.2}


def test_qwen_entry_reads_every_box_of_the_answer_and_notes_a_cut_answer() -> None:
    rgb = np.zeros((10, 20, 3), np.uint8)
    whole = FakeReader('```json\n[{"bbox_2d": [0, 0, 500, 500], "label": "knob"}]\n```')
    cut = FakeReader('[{"bbox_2d": [0, 0, 500, 500], "label": "knob"}, {"bbox_2d": [10, 1')

    entry = run.qwen_entry(whole, rgb, "12.5", "rev")

    assert entry["detections"] == [{"box": [0, 0, 10, 5], "score": None, "label": "knob"}]
    assert entry["complete"] and entry["seconds"] == 3.2
    assert not run.qwen_entry(cut, rgb, "12.5", "rev")["complete"]
    assert whole.keys[0] != run.answer_key(run.TM.detection_prompt(), "13.0", "rev")  # one question per keyframe


def test_a_record_made_another_way_is_not_resumed(tmp_path: Path) -> None:
    fresh = {"tool": "sam3", "model": "m@1", "version": run.TM.VERSION, "times": ["1", "2"], "image_hw": [4, 4], "keyframes": {}}
    file = tmp_path / "map.json"
    file.write_text(json.dumps({**fresh, "keyframes": {"0": {"seconds": 1.0, "detections": []}}}))

    assert run.start_record(file, fresh)["keyframes"] == {"0": {"seconds": 1.0, "detections": []}}  # resumed
    file.write_text(json.dumps({**fresh, "model": "m@2"}))
    with pytest.raises(SystemExit):
        run.start_record(file, fresh)
    assert run.start_record(tmp_path / "none.json", fresh) is fresh
