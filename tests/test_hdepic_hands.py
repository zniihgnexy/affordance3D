import importlib.util
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from affordance3d import hdepic_hands as K

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b1 = load_script("hdepic_b1_hand_events")
fetch = load_script("hdepic_fetch_trajectories")

QUARTER_TURN_Z = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])  # 90 degrees about z, (x, y, z, w)


# --- frames and clocks -------------------------------------------------------------------------------------------

def test_rotate_and_to_world_apply_the_device_to_world_pose() -> None:
    assert np.allclose(K.rotate(QUARTER_TURN_Z, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0])
    assert np.allclose(K.rotate(np.array([0.0, 0.0, 0.0, 2.0]), np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
    world = K.to_world(np.array([[1.0, 0.0, 0.0]]), np.array([[10.0, 0.0, 0.0]]), QUARTER_TURN_Z[None])
    assert np.allclose(world, [[10.0, 1.0, 0.0]])


def test_nearest_poses_take_the_closest_sample_within_the_gap() -> None:
    trajectory = np.array([0, 1000, 2000, 3000])

    index = K.nearest_poses(trajectory, np.array([-500, 400, 1600, 3000, 20_000]), max_gap_us=600)

    assert index.tolist() == [0, 0, 2, 3, -1]


def test_mp4_seconds_count_from_the_first_frame() -> None:
    assert K.mp4_seconds(np.array([5_000_000, 6_500_000]), 5_000_000).tolist() == [0.0, 1.5]


# --- detection ---------------------------------------------------------------------------------------------------

def test_speeds_use_neighbours_and_skip_gaps_and_untracked_samples() -> None:
    seconds = np.array([0.0, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    positions = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.3, 0, 0], [0.3, 0, 0], [0.3, 0, 0]])
    valid = np.array([True, True, True, True, True, True, False])

    speed = K.speeds(seconds, positions, valid)

    assert np.isnan(speed[0]) and np.isnan(speed[-1])
    assert speed[1] == pytest.approx(1.0) and speed[2] == pytest.approx(1.0)
    assert np.isnan(speed[3]) and np.isnan(speed[4])  # 0.3 s gap next to them
    assert np.isnan(speed[5])  # right neighbour untracked


def test_slow_events_give_the_slowest_sample_of_each_long_enough_slow_run() -> None:
    seconds = np.arange(10) / 10
    positions = np.arange(30, dtype=float).reshape(10, 3)
    speed = np.array([np.nan, 0.5, 0.1, 0.05, 0.12, 0.5, 0.1, 0.5, 0.08, 0.02])

    events = K.slow_events(seconds, positions, speed, slow=0.15, min_samples=2)

    # the lone slow sample at 0.6 s is too short to count
    assert [(e.seconds, e.speed) for e in events] == [(0.3, 0.05), (0.9, 0.02)]
    assert events[0].position == (9.0, 10.0, 11.0)


def test_merge_hands_keeps_the_slower_of_two_close_events() -> None:
    events = [K.Event(1.0, (0, 0, 0), 0.10), K.Event(1.2, (0.05, 0, 0), 0.02), K.Event(1.3, (1, 0, 0), 0.05),
              K.Event(3.0, (0, 0, 0), 0.10)]

    merged = K.merge_hands(events)

    assert [(e.seconds, e.speed) for e in merged] == [(1.2, 0.02), (1.3, 0.05), (3.0, 0.10)]


# --- scoring -----------------------------------------------------------------------------------------------------

def test_match_is_one_to_one_within_both_tolerances() -> None:
    truths = [K.Event(10.0, (0, 0, 0)), K.Event(10.5, (0, 0, 0)), K.Event(20.0, (0, 0, 0))]
    detections = [K.Event(10.4, (0.1, 0, 0)), K.Event(10.45, (0.0, 0, 0)), K.Event(20.5, (0.5, 0, 0)),
                  K.Event(30.0, (0, 0, 0))]

    pairs = K.match(detections, truths)

    assert sorted(pairs) == [(0, 0), (1, 1)]  # the far detection near 20 s and the late one find nothing
    assert K.precision_recall(detections, truths) == {"detections": 4, "truths": 3, "matched": 2,
                                                      "precision": 0.5, "recall": 2 / 3}
    assert K.precision_recall([], truths)["precision"] is None


def test_annotated_events_keep_pick_ups_and_put_downs_with_a_frame_and_a_place() -> None:
    priming = {"0": {"start": "", "end": {"frame": 60, "3d_location": [1, 2, 3]}},
               "1": {"start": {"frame": 30, "3d_location": [0, 0, 0]}, "end": {"frame": -1, "3d_location": [0, 0, 0]}},
               "2": {"start": {"frame": 90, "3d_location": []}}}

    events = K.annotated_events(priming)

    assert [(e.seconds, e.kind) for e in events] == [(1.0, "pick-up"), (2.0, "put-down")]


# --- the B-1 script ----------------------------------------------------------------------------------------------

def still_then_moving_world() -> dict:
    """A left palm that moves at 1 m/s, stops at x = 1 for 0.5 s, then moves on; the right hand is untracked."""
    seconds = np.arange(30) / 10
    x = np.concatenate([np.linspace(0, 1, 11), np.full(5, 1.0), 1 + np.linspace(0.1, 1.4, 14)])
    palm = np.stack([x, np.zeros(30), np.zeros(30)], axis=1)
    return {"seconds": seconds, "left_palm": palm, "left_valid": np.ones(30, bool),
            "right_palm": np.zeros((30, 3)), "right_valid": np.zeros(30, bool)}


def test_world_hands_and_detect_find_the_stop() -> None:
    hands = {"us": np.arange(30) * 100_000 + 7_000_000, "left_palm": still_then_moving_world()["left_palm"],
             "left_valid": np.ones(30, bool), "right_palm": np.zeros((30, 3)), "right_valid": np.zeros(30, bool)}
    trajectory = {"us": np.arange(0, 11_000_000, 1000), "t": np.zeros((11_000, 3)),
                  "q": np.tile([0, 0, 0, 1.0], (11_000, 1))}

    world = b1.world_hands(hands, trajectory)
    events = b1.detect(world)

    assert world["seconds"][0] == 0.0 and world["left_valid"].all()
    assert len(events) == 1 and 1.0 <= events[0].seconds <= 1.5 and events[0].position[0] == pytest.approx(1.0)


def test_clock_checks_peak_at_the_true_offset() -> None:
    world = still_then_moving_world()
    truths = [K.Event(1.2, (1.0, 0.0, 0.0))]
    trajectory = {"us": np.arange(0, 4_000_000, 1000), "t": np.stack([np.arange(4000) / 1000, np.zeros(4000),
                                                                       np.zeros(4000)], axis=1),
                  "q": np.tile([0, 0, 0, 1.0], (4000, 1))}  # the device walks along x at 1 m/s
    priming = {"0": {"start": {"3d_location": [3.0, 0.0, 0.0],
                               "prime_stats": {"frame_primed": 30, "dist_to_cam": 2.0}}}}  # 1 s: device at x = 1

    assert b1.palm_object_distances(world, truths, 0.0) == [0.0]
    assert b1.palm_object_distances(world, truths, 1.0)[0] > 0.5
    assert b1.wearer_distance_errors(priming, trajectory, 0, 0.0) == pytest.approx([0.0])
    assert b1.wearer_distance_errors(priming, trajectory, 0, 0.5) == pytest.approx([0.5])
    scan = b1.offset_scan({-0.5: [0.4], 0.0: [0.1, 0.3], 0.5: [0.05]})
    assert (scan["median_at_0"], scan["best_offset_s"], scan["n_at_0"]) == (0.2, 0.5, 2)


def test_pooled_scores_and_verdict_use_both_thresholds() -> None:
    per_video = [{"detections": 10, "truths": 8, "matched": 5}, {"detections": 10, "truths": 2, "matched": 1}]

    result = b1.pooled(per_video)

    assert (result["precision"], result["recall"]) == (0.3, 0.6)
    assert b1.verdict(result) == "fail"
    assert b1.verdict({"precision": 0.5, "recall": 0.6}) == "pass"


def test_stream_seconds_and_hand_table_read_the_mps_files() -> None:
    health = {"RGB Camera Class #1": {"processed": 300}, "Camera Data (SLAM) #1": {"processed": 100},
              "Camera Data (SLAM) #1 - RGB Camera Class #1": {"largest_misalignment_us": 484}}
    header = ("tracking_timestamp_us,left_tracking_confidence,tx_left_wrist_device,ty_left_wrist_device,"
              "tz_left_wrist_device,tx_left_palm_device,ty_left_palm_device,tz_left_palm_device,"
              "right_tracking_confidence,tx_right_wrist_device,ty_right_wrist_device,tz_right_wrist_device,"
              "tx_right_palm_device,ty_right_palm_device,tz_right_palm_device")
    row = "100,0.9,0,0,0,1,1,1,-1,0,0,0,0,0,0"

    table = b1.hand_table(header + "\n" + row + "\n")

    assert b1.stream_seconds(health) == {"rgb_s": 10.0, "slam_s": 10.0, "rgb_slam_misaligned_us": 484}
    assert table["us"].tolist() == [100] and table["left_valid"].tolist() == [True]
    assert table["right_valid"].tolist() == [False] and table["left_palm"].tolist() == [[1.0, 1.0, 1.0]]


# --- the fetch script --------------------------------------------------------------------------------------------

def test_fetch_finds_the_trajectory_member_and_the_video_of_each_archive(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "3.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("3/slam/closed_loop_trajectory.csv", "tracking_timestamp_us\n" + "1\n" * 100)
        archive.writestr("3/slam/semidense_points.csv.gz", b"x" * 10)
    with zipfile.ZipFile(tmp_path / "3.zip") as archive:
        infos = archive.infolist()

    name, compressed, size = fetch.trajectory_member(infos, "3")

    assert name == "3/slam/closed_loop_trajectory.csv" and compressed < size == 222
    assert fetch.trajectory_member(infos, "4") is None
    assert fetch.video_indices({"P02/P02-20240209-184316.vrs": "0"}) == {"0": "P02-20240209-184316"}
