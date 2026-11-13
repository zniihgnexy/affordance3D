import numpy as np
import pytest

from affordance3d.mem import grounding as G


@pytest.mark.parametrize(("answer", "n", "image", "box"), [
    ('{"image": 2, "bbox_2d": [100, 200, 300, 400]}', 3, 2, (100.0, 200.0, 300.0, 400.0)),
    ('```json\n[{"image": 1, "bbox_2d": [300, 400, 100, 200], "label": "knob"}]\n```', 3, 1, (100.0, 200.0, 300.0, 400.0)),
    ('{"image": 5, "bbox_2d": [100, 200, 300, 400]}', 3, None, (100.0, 200.0, 300.0, 400.0)),  # no such image
    ('{"bbox_2d": [10, 20, 30, 40]}', 1, 1, (10.0, 20.0, 30.0, 40.0)),  # one image: it is image 1
    ('{"bbox_2d": [10, 20, 30, 40]}', 3, None, (10.0, 20.0, 30.0, 40.0)),  # several images and none named
    ('{"image": 1, "bbox_2d": [10, 20, 1300, 40]}', 2, 1, None),  # outside the 0-1000 range
    ('{"image": 1, "bbox_2d": [10, 20, 10, 40]}', 2, 1, None),  # zero width
    ("I cannot find it.", 2, None, None),
])
def test_parse_grounding_reads_image_and_box_and_refuses_impossible_ones(answer: str, n: int, image, box) -> None:
    parsed = G.parse_grounding(answer, n)

    assert parsed.image == image and parsed.box == box


def test_to_pixels_scales_relative_boxes_and_keeps_them_inside_the_image() -> None:
    assert G.to_pixels((0.0, 0.0, 1000.0, 1000.0), (480, 640)) == (0, 0, 640, 480)
    assert G.to_pixels((250.0, 500.0, 500.0, 750.0), (480, 640)) == (160, 240, 320, 360)
    assert G.to_pixels((999.0, 999.0, 1000.0, 1000.0), (480, 640))[2:] == (640, 480)


def test_hits_part_uses_the_box_centre_on_the_widened_target() -> None:
    labels = np.full((100, 100), -1)
    labels[40:50, 40:50] = 3  # the target part
    labels[0:10, 0:10] = 7  # another part

    assert G.hits_part((40, 40, 50, 50), labels, [3])
    assert G.hits_part((52, 40, 62, 50), labels, [3])  # centre (57, 45) is 8 px from the part: inside the margin
    assert not G.hits_part((60, 40, 70, 50), labels, [3])  # centre (65, 45) is 16 px away
    assert not G.hits_part(None, labels, [3])
    assert not G.hits_part((0, 0, 10, 10), labels, [3])  # the other part


def test_a_huge_box_can_pass_the_old_rule_but_not_the_new_one() -> None:
    labels = np.full((100, 100), -1)
    labels[10:14, 80:84] = 3
    huge = (0, 0, 100, 100)

    assert G.covers_a_centre(huge, [(82, 12)])  # the phase-1 rule rewards a box that covers everything
    assert not G.hits_part(huge, labels, [3])  # the rule used for every arm now does not


def test_prompts_name_the_task_and_the_number_of_images_and_stay_neutral() -> None:
    task = "Open the top drawer"

    assert task in G.numbered_prompt(task, 7) and "1 to 7" in G.numbered_prompt(task, 7)
    assert "operable" not in G.numbered_prompt(task, 7)  # boxes are called regions, not parts one can operate
    for prompt in (G.numbered_prompt(task, 7), G.grounding_prompt(task, 3), G.frame_prompt(task, 40), G.single_grounding_prompt(task)):
        assert "not the whole object" in prompt  # every arm is told what "the part" means
    assert "1 to 3" in G.grounding_prompt(task, 3) and "bbox_2d" in G.grounding_prompt(task, 3)
    assert "1 to 40" in G.frame_prompt(task, 40) and "bbox_2d" in G.single_grounding_prompt(task)
    assert G.image_labels(2) == ["Image 1:", "Image 2:"] and G.image_labels(1, "Frame") == ["Frame 1:"]


def test_closeness_follows_the_rule_fixed_before_the_controls_ran() -> None:
    from affordance3d.mem import phase1_stats as A

    assert A.closeness((0.20, 0.12, 0.28)) == "behind"  # P4 clearly ahead
    assert A.closeness((0.03, -0.04, 0.10)) == "close"  # the arm does about as well as P4
    assert A.closeness((0.14, 0.02, 0.26)) == "undetermined"  # lower end below 5 points only because the interval is wide


def test_the_answer_cache_passes_options_to_the_model(tmp_path) -> None:
    from report import mem_steps_helpers as H

    seen = {}

    class Model:
        def ask(self, images, prompt, **options):
            seen.update(options)
            return "2"

    reply = H.CachedReader(tmp_path / "answers.json", Model).ask("key", [], "which?", labels=["Image 1:"], max_new_tokens=64)

    assert reply["answer"] == "2" and seen == {"labels": ["Image 1:"], "max_new_tokens": 64}
