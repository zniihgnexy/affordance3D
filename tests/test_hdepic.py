import dataclasses
import importlib.util
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from affordance3d import hdepic as H

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b0 = load_script("hdepic_b0_events")
b1 = load_script("hdepic_b1_hand_events")


def assoc(*segments: tuple[float, float], masks: tuple[tuple[str, ...], ...] | None = None) -> H.Association:
    tracks = tuple(H.Track(f"t{i}", a, b, masks[i] if masks else ()) for i, (a, b) in enumerate(segments))
    return H.Association("a", "cup", tracks)


def mask(mask_id: str, frame: int, box=(0.0, 0.0, 10.0, 10.0), location=(0.0, 0.0, 0.0)) -> H.Mask:
    return H.Mask(mask_id, frame, box, location, "P02_counter.001")


# --- parsing ---------------------------------------------------------------------------------------------------

def test_times_and_boxes_are_read_from_question_text() -> None:
    text = "Where did I take the object identified by <BBOX 1 2.5 30 40.25> at <TIME 00:01:02.5 video 1> " \
           "from before putting it at <TIME 01:00:00 video 1>?"

    assert H.times_in(text) == [62.5, 3600.0]
    assert H.first_box(text) == (1.0, 2.5, 30.0, 40.25)
    assert H.first_box("no box") is None


def test_query_uses_the_first_box_and_time_and_rounds_to_a_frame() -> None:
    question = {"inputs": {"video 1": {"id": "P02-x"}},
                "question": "object <BBOX 0 0 5 5> seen at <TIME 00:00:01.517 video 1>"}

    query = H.query_of(question)

    assert query == H.Query("P02-x", (0.0, 0.0, 5.0, 5.0), 1.517)
    assert query.frame == 46  # 1.517 s * 30 fps = 45.5 -> 46
    assert H.query_of({"inputs": {"video 1": {"id": "v"}}, "question": "no tags"}) is None


def test_static_duration_and_fixture_verb_are_parsed() -> None:
    assert H.required_static_seconds("remain static for more than 57 seconds?") == 57.0
    assert H.required_static_seconds("no duration") is None
    assert H.fixture_verb("How many times did I close the item indicated by") == "close"
    assert H.fixture_verb("How many times did I open the item") == "open"
    assert H.fixture_verb("Which objects did the person take") is None


def test_itinerary_hops_count_the_moves() -> None:
    assert H.itinerary_hops("from sink to hob") == 1
    assert H.itinerary_hops("from sink to hob, then from hob to sink, then from sink to counter") == 3


# --- matching boxes to annotations -------------------------------------------------------------------------------

def test_swap_axes_exchanges_x_and_y() -> None:
    assert H.swap_axes((1.0, 2.0, 3.0, 4.0)) == (2.0, 1.0, 4.0, 3.0)


def test_box_iou() -> None:
    assert H.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert H.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert H.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert H.box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_resolve_needs_both_the_frame_window_and_the_overlap() -> None:
    query = H.Query("v", (0.0, 0.0, 10.0, 10.0), 10.0)  # frame 300
    owners = {"near": "A", "far": "B", "shifted": "C"}
    masks = [mask("far", 316), mask("shifted", 300, box=(6.0, 0.0, 16.0, 10.0)), mask("loose", 300)]

    assert H.resolve(query, masks, owners).assoc_id is None  # 16 frames away; IoU 0.25; mask in no association
    resolution = H.resolve(query, masks + [mask("near", 285)], owners)
    assert resolution == H.Resolution("A", 1.0, 15, 1)


def test_resolve_prefers_the_highest_overlap_and_counts_rival_associations() -> None:
    query = H.Query("v", (0.0, 0.0, 10.0, 10.0), 10.0)
    masks = [mask("m1", 301, box=(1.0, 0.0, 11.0, 10.0)), mask("m2", 305), mask("m3", 300, box=(0.0, 0.0, 10.0, 12.0))]

    resolution = H.resolve(query, masks, {"m1": "A", "m2": "B", "m3": "C"})

    assert (resolution.assoc_id, resolution.frame_gap, resolution.candidate_assocs) == ("B", 5, 3)


def test_associations_are_read_with_tracks_in_time_order() -> None:
    assoc_info = {"v": {"a1": {"name": "cup", "tracks": [
        {"track_id": "late", "time_segment": [20, 21], "masks": ["m2"]},
        {"track_id": "early", "time_segment": [5, 6], "masks": ["m1"]}]}}}
    mask_info = {"v": {"m1": {"frame_number": 150, "3d_location": [1, 2, 3], "bbox": [0, 0, 1, 1], "fixture": "Null"}}}

    assocs = H.associations_of(assoc_info, "v")

    assert [t.track_id for t in assocs["a1"].tracks] == ["early", "late"]
    assert H.mask_owners(assocs) == {"m1": "a1", "m2": "a1"}
    assert H.masks_of(mask_info, "v") == [H.Mask("m1", 150, (0.0, 0.0, 1.0, 1.0), (1.0, 2.0, 3.0), None)]


# --- rule solvers ------------------------------------------------------------------------------------------------

def test_movement_count_is_the_number_of_tracks() -> None:
    assert H.movement_count(assoc((0, 1), (5, 6), (10, 11))) == 3


def test_retention_filter_checks_gap_and_shift() -> None:
    masks = {"a": mask("a", 0, location=(0, 0, 0)), "b": mask("b", 30, location=(0.1, 0, 0)),
             "c": mask("c", 60, location=(0.15, 0, 0)), "d": mask("d", 90, location=(0.9, 0, 0)),
             "e": H.Mask("e", 120, (0, 0, 1, 1), None, None)}

    assert H.passes_retention_filter(assoc((0, 1), (3, 4), masks=(("a", "b"), ("c", "d"))), masks) is True
    assert H.passes_retention_filter(assoc((0, 1), (2.0, 4), masks=(("a", "b"), ("c", "d"))), masks) is False
    assert H.passes_retention_filter(assoc((0, 1), (3, 4), masks=(("a", "d"), ("c",))), masks) is False  # 0.75 m
    assert H.passes_retention_filter(assoc((0, 1), (3, 4), masks=(("a", "e"), ("e",))), masks) is None
    assert H.passes_retention_filter(assoc((0, 1)), masks) is True


def test_static_intervals_start_after_the_first_move_and_the_last_is_open() -> None:
    assert H.static_intervals(assoc((10, 12), (40, 41))) == [(12, 40), (41, math.inf)]
    assert H.static_intervals(H.Association("a", "cup", ())) == []


def test_qualifying_choices_follow_the_positive_interval() -> None:
    moves = assoc((10, 12), (40, 41), (50, 52))  # static: 12-40 (28 s), 41-50, 52-end
    times = [5.0, 15.0, 13.0, 45.0, 30.0]  # before first move; 25 s left; 27 s left; 5 s left; 10 s left

    assert H.qualifying_choices(moves, times, min_static_s=26) == [2]
    assert H.qualifying_choices(moves, times, min_static_s=20) == [1, 2]
    assert H.qualifying_choices(moves, [11.95], min_static_s=28) == [0]  # within the 0.1 s slack at the start
    assert H.qualifying_choices(moves, [100.0], min_static_s=500) == [0]  # the last interval never ends


# --- narrations --------------------------------------------------------------------------------------------------

def test_narrations_with_verb_match_verbs_that_include_the_word() -> None:
    rows = [{"start_timestamp": 9.0, "verbs": ["pull open"]}, {"start_timestamp": 2.0, "verbs": ["open", "take"]},
            {"start_timestamp": 5.0, "verbs": ["close off"]}, {"start_timestamp": 1.0, "verbs": ["take"]}]

    assert [r["start_timestamp"] for r in H.narrations_with_verb(rows, "open")] == [2.0, 9.0]
    assert [r["start_timestamp"] for r in H.narrations_with_verb(rows, "close")] == [5.0]


def test_pickled_table_loads_allowed_globals_and_refuses_others(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.pkl", tmp_path / "bad.pkl"
    good.write_bytes(pickle.dumps(np.arange(3.0)))  # numpy reconstruction globals, as in the narrations file
    bad.write_bytes(pickle.dumps(Path("x")))

    assert H.load_pickled_table(good).tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(pickle.UnpicklingError):
        H.load_pickled_table(bad)


# --- the B-0 script ----------------------------------------------------------------------------------------------

def synthetic_video() -> dict:
    masks = [mask("m0", 300), mask("m1", 330), mask("m2", 900), mask("m3", 960)]
    assocs = {"A": H.Association("A", "cup", (H.Track("t0", 10.0, 11.0, ("m0", "m1")),
                                               H.Track("t1", 30.0, 32.0, ("m2", "m3"))))}
    narrations = [{"start_timestamp": 3.0, "end_timestamp": 4.0, "narration": "open the drawer", "verbs": ["open"]}]
    return {"masks": masks, "mask_by_id": {m.mask_id: m for m in masks}, "assocs": assocs,
            "owners": H.mask_owners(assocs), "narrations": narrations}


def question(text: str, choices: list[str], correct: int) -> dict:
    return {"inputs": {"video 1": {"id": "P02-x"}}, "question": text, "choices": choices, "correct_idx": correct}


def test_check_question_ties_counts_and_solves_stationary() -> None:
    video = synthetic_video()
    counting = question("How many times did the object <BBOX 0 0 10 10> seen at <TIME 00:00:10.1 video 1> change "
                        "locations in the video?", ["5", "2", "1", "3", "4"], 1)
    stationary = question("After the object <BBOX 0 0 10 10> seen at <TIME 00:00:10 video 1> is first moved, from "
                          "which of the following starting times does the object remain static for more than 15 "
                          "seconds?", ["<TIME 00:00:05 video 1>", "<TIME 00:00:12 video 1>",
                                       "<TIME 00:00:20 video 1>", "<TIME 00:00:31 video 1>",
                                       "<TIME 00:00:40 video 1>"], 1)
    untied = question("How many times did the object <BBOX 50 50 60 60> seen at <TIME 00:00:10 video 1> change "
                      "locations in the video?", ["1", "2", "3", "4", "5"], 0)
    fixture = question("How many times did I open the item indicated by bounding box <BBOX 0 0 1 1> in "
                       "<TIME 00:00:03 video 1>?", ["0", "1", "2", "3", "4"], 1)

    c = b0.check_question(b0.COUNTING, counting, video)
    s = b0.check_question(b0.STATIONARY, stationary, video)
    u = b0.check_question(b0.COUNTING, untied, video)
    f = b0.check_question(b0.FIXTURE_COUNTING, fixture, video)

    assert (c["tied"], c["rule_answer"], c["reproduced"]) == (True, 2, True)
    assert (s["qualifying"], s["reproduced"]) == ([1, 4], False)  # 12 s (18 s left) and 40 s (open end) both qualify
    assert (u["tied"], u["reproduced"]) == (False, False)
    assert (f["tied"], f["narrations"], f["answer_within_listed"]) == (True, 1, True)


def test_swapped_box_axes_tie_a_transposed_box() -> None:
    video = synthetic_video()
    video["masks"] = [dataclasses.replace(m, box=(20.0, 0.0, 30.0, 10.0)) for m in video["masks"]]
    transposed = question("How many times did the object <BBOX 0 20 10 30> seen at <TIME 00:00:10 video 1> change "
                          "locations in the video?", ["1", "2", "3", "4", "5"], 1)

    assert b0.check_question(b0.COUNTING, transposed, video)["tied"] is False
    assert b0.check_question(b0.COUNTING, transposed, video, swap_box_axes=True)["reproduced"] is True


def test_summary_counts_untied_rule_questions_as_wrong_and_applies_both_thresholds() -> None:
    records = [{"prototype": b0.COUNTING, "tied": True, "reproduced": True, "retention_filter": True}] * 9 + \
              [{"prototype": b0.COUNTING, "tied": False, "reproduced": False}] + \
              [{"prototype": b0.FIXTURE_COUNTING, "tied": True, "answer_within_listed": True}] * 10

    summary = b0.summarise(records)

    assert (summary["tied_share"], summary["reproduced_share"], summary["reproduced_among_tied"]) == (0.95, 0.9, 1.0)
    assert summary["verdict"] == "continue"
    worse = records[:8] + [{"prototype": b0.COUNTING, "tied": True, "reproduced": False}] + records[9:]
    assert b0.summarise(worse)["verdict"] == "data failure"


# --- the B-1 script ----------------------------------------------------------------------------------------------

HAND_HEADER = ("tracking_timestamp_us,left_tracking_confidence,tx_left_wrist_device,ty_left_wrist_device,"
               "tz_left_wrist_device,tx_left_palm_device,ty_left_palm_device,tz_left_palm_device,"
               "right_tracking_confidence,tx_right_wrist_device,ty_right_wrist_device,tz_right_wrist_device,"
               "tx_right_palm_device,ty_right_palm_device,tz_right_palm_device")


def test_hand_file_summary_reports_the_device_frame_and_tracked_share(tmp_path: Path) -> None:
    rows = ["0,-1,0,0,0,0,0,0,0.9,1,2,3,1,2,3", "100000,0.8,1,1,1,1,1,1,0.9,1,2,3,1,2,3",
            "200000,0.7,1,1,1,1,1,1,-1,0,0,0,0,0,0"]
    text = "\n".join([HAND_HEADER, *rows]) + "\n"
    with zipfile.ZipFile(tmp_path / "mps_P02-x_vrs.zip", "w") as archive:
        archive.writestr("mps_P02-x_vrs/hand_tracking/wrist_and_palm_poses.csv", text)

    summary = b1.read_hand_files(tmp_path)["mps_P02-x_vrs.zip"]

    assert summary == {"frame": "device", "samples": 3, "step_s": 0.1, "span_s": 0.2,
                       "tracked": {"left": 0.667, "right": 0.667}}
    assert b1.position_frame(["tx_left_wrist_world", "ty_left_palm_world"]) == "world"
    assert b1.position_frame(["tx_left_wrist_world", "ty_left_palm_device"]) == "unknown"


def test_pose_gate_follows_the_plan() -> None:
    assert b1.pose_gate({"status": 403})[0] is False
    assert b1.pose_gate({"status": None, "error": "timeout"})[0] is False
    assert b1.pose_gate({"status": 200, "final_host": "login.microsoftonline.com", "bytes": 10})[1] == "needs a login"
    assert b1.pose_gate({"status": 200, "bytes": None, "content_type": "text/html"})[0] is False
    assert b1.pose_gate({"status": 200, "bytes": 2 * 10**9, "content_type": "application/zip"})[0] is False
    assert b1.pose_gate({"status": 200, "bytes": 5 * 10**8, "content_type": "application/zip"}) == (True, "500.0 MB")


def test_readme_link_pattern_finds_the_intermediate_data_link() -> None:
    readme = "**[Download intermediate data](https://example.org/share/abc?e=1)**"

    assert b1.POSE_LINK.search(readme).group(1) == "https://example.org/share/abc?e=1"
