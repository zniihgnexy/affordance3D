from pathlib import Path

import numpy as np
import pytest

from affordance3d.mem import encoders as E
from affordance3d.mem import tokens as T
from report import mem_steps_helpers as H
from tests.conftest import TEST_HW, frame_from_depth


def _token(keyframe: int, component: int, box: tuple[int, int, int, int], time_s: float = 0.0) -> T.PartToken:
    region = T.Region(keyframe, component, action=0, box=box, pixel=(box[0], box[1]), pixels=16, depth_m=2.0)
    return T.PartToken(region, time_s, np.array([1.0, 0.0]), np.array([1.0, 0.0]))


def _memory() -> T.L2Memory:
    """Two keyframes; three tokens; the memory has linked tokens 0 and 2 into part 0, token 1 is part 1."""
    components = [np.zeros(TEST_HW, dtype=np.int32) for _ in range(2)]
    components[0][2:6, 2:6] = 1
    components[0][20:24, 30:34] = 2
    components[1][2:6, 2:6] = 1
    tokens = (_token(0, 1, (2, 2, 6, 6)), _token(0, 2, (30, 20, 34, 24)), _token(1, 1, (2, 2, 6, 6), time_s=2.0))
    return T.L2Memory(("10.0", "12.0"), tuple(components), tokens, np.array([0, 1, 0]), floor=0.0)


def test_truth_parts_takes_the_majority_part_id_under_each_region() -> None:
    memory = _memory()
    labels = [np.full(TEST_HW, -1), np.full(TEST_HW, -1)]
    labels[0][2:6, 2:6] = 5
    labels[0][20:24, 30:34] = 7
    labels[1][2:6, 2:5] = 5  # most of the region
    labels[1][2:6, 5] = 7

    truth = H.truth_parts(memory, labels, n_parts=9)

    assert truth.tolist() == [5, 7, 5]
    labels[0][20:24, 30:34] = -1  # a region that covers no labelled pixel cannot be scored
    with pytest.raises(ValueError):
        H.truth_parts(memory, labels, n_parts=9)


def test_mixed_regions_counts_regions_that_cover_two_true_parts() -> None:
    labels = [np.full(TEST_HW, -1), np.full(TEST_HW, -1)]
    labels[0][2:6, 2:6] = 5
    labels[1][2:6, 2:4] = 5
    labels[1][2:6, 4:6] = 7  # the region of keyframe 1 lies on two true parts

    assert H.mixed_regions(_memory(), labels) == 1


def test_same_part_gap_is_the_median_distance_between_positions_of_one_true_part() -> None:
    frames = [frame_from_depth(np.full(TEST_HW, 2.0)), frame_from_depth(np.full(TEST_HW, 2.0), cam_to_world=_shifted(0.1))]

    gap_cm = H.same_part_gap_cm(_memory(), frames, truth=np.array([5, 7, 5]))

    assert gap_cm == pytest.approx(10.0)  # tokens 0 and 2: same pixel and depth, cameras 10 cm apart


def _shifted(metres: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = metres
    return pose


def test_linking_scores_counts_pure_whole_and_exact_parts() -> None:
    perfect = H.linking_scores(_memory(), np.array([5, 7, 5])).iloc[0]
    mixed = H.linking_scores(_memory(), np.array([5, 7, 7])).iloc[0]  # part 0 now mixes two true parts

    assert perfect["parts linked exactly right"] == "2 of 2" and perfect["tokens whose part is not mixed"] == "100%"
    assert mixed["parts linked exactly right"] == "0 of 2" and mixed["tokens whose part is not mixed"] == "67%"


def test_marked_keyframes_numbers_the_regions_across_images_and_returns_their_tokens() -> None:
    frames = [frame_from_depth(np.full(TEST_HW, 2.0)) for _ in range(2)]

    images, tokens = H.marked_keyframes(frames, _memory(), keyframe_ids=[1, 0])
    picked, _ = H.marked_keyframes(frames, _memory(), keyframe_ids=[1, 0], chosen=2, right=[3])

    assert tokens == [2, 0, 1]  # mark 1 is the region of keyframe 1, marks 2 and 3 those of keyframe 0
    assert not frames[0].rgb.any() and not frames[1].rgb.any()  # drawn on copies
    assert (images[0] == H.MARK_COLOUR).all(axis=-1).any() and (images[1] == H.MARK_COLOUR).all(axis=-1).any()
    assert (images[1][0:2, 0:14] == H.MARK_COLOUR).all(axis=-1).any()  # the box of the region at (2, 2)-(6, 6), padded
    assert (picked[1] == H.CHOSEN_COLOUR).all(axis=-1).any() and (picked[1] == H.RIGHT_COLOUR).all(axis=-1).any()
    assert not (images[1] == H.CHOSEN_COLOUR).all(axis=-1).any()  # what the model sees carries no hint


def test_reading_key_changes_with_what_the_model_is_shown() -> None:
    key = H.reading_key("model-a", "which box?", [3, 1], [(2, 2, 6, 6)])

    assert key == H.reading_key("model-a", "which box?", [3, 1], [(2, 2, 6, 6)])
    assert len({key, H.reading_key("model-b", "which box?", [3, 1], [(2, 2, 6, 6)]),
                H.reading_key("model-a", "which one?", [3, 1], [(2, 2, 6, 6)]),
                H.reading_key("model-a", "which box?", [1, 3], [(2, 2, 6, 6)]),
                H.reading_key("model-a", "which box?", [3, 1], [(2, 2, 6, 7)])}) == 5


def test_label_spot_moves_away_from_a_neighbouring_box_and_stays_inside_the_image() -> None:
    box = (20, 20, 30, 30)
    above_is_taken = [(18, 8, 34, 20)]

    free = H.label_spot(box, size=(8, 6), taken=[], image_hw=(48, 64))
    moved = H.label_spot(box, size=(8, 6), taken=above_is_taken, image_hw=(48, 64))
    at_top = H.label_spot((20, 0, 30, 10), size=(8, 6), taken=[], image_hw=(48, 64))

    assert free == (20, 14, 28, 20)  # above the box when nothing is in the way
    assert moved == (20, 30, 28, 36)  # below it when a neighbour sits above
    assert at_top == (20, 10, 28, 16)  # never outside the image


def test_cached_reader_asks_the_model_once_per_question(tmp_path: Path) -> None:
    calls = []

    class Model:
        def ask(self, images: list, prompt: str) -> str:
            calls.append(prompt)
            return "3"

    cache = tmp_path / "answers.json"
    first = H.CachedReader(cache, Model).ask("key", [], "which box?")
    again = H.CachedReader(cache, Model).ask("key", [], "which box?")  # a new reader finds the answer on disk

    assert first["answer"] == again["answer"] == "3" and calls == ["which box?"]


def test_model_source_uses_a_local_copy_only_when_it_is_verified_for_that_revision(tmp_path: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    hub = {"pretrained_model_name_or_path": "org/name", "revision": "abc"}
    (tmp_path / "org" / "name").mkdir(parents=True)
    monkeypatch.setenv(E.MODELS_ENV, str(tmp_path))

    assert E.model_source("org/name", "abc") == hub  # a folder without the marker may be an interrupted download
    (tmp_path / "org" / "name" / E.VERIFIED_MARKER).write_text("abc\n")
    assert E.model_source("org/name", "abc") == {"pretrained_model_name_or_path": str(tmp_path / "org" / "name")}
    assert E.model_source("org/name", "def") == {"pretrained_model_name_or_path": "org/name", "revision": "def"}
    monkeypatch.delenv(E.MODELS_ENV)
    assert E.model_source("org/name", "abc") == hub
