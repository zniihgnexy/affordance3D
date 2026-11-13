"""B-1: can pick-ups and put-downs be found from Aria wrist/palm tracking, without annotations? (draft, 25.2 and 25.5)

    python scripts/hdepic_fetch_trajectories.py     # first: only the device trajectories of the SLAM archives
    python scripts/hdepic_b1_hand_events.py         # writes runs/hdepic/b1_hand_events.json

CPU only. Inputs: the MPS palm positions of the GAZE_HAND zips (device frame, device clock, 10 Hz; read inside the
zips) and the closed-loop device trajectory (device-to-world, same clock). Evaluation only: `priming_info.json`.
- Frame: p_world = T_world_device(t) p_device, with the trajectory sample nearest to each hand sample (<= 10 ms).
- Clock: MP4 time = (device time - device time of the first hand sample) / 1e6. There is one hand sample per SLAM
  camera frame, and the VRS health check shows the RGB frames aligned with the SLAM frames, so MP4 frame 0 (the
  first RGB frame) is taken to be the first SLAM frame. Two checks, descriptive only (the offset is not changed):
  the wearer-to-object distance that the priming annotations give at the priming frame, and the palm-to-object
  distance at annotated pick-ups and put-downs, each over clock offsets of -3 to +3 s.
- Detector (parameters in `affordance3d/hdepic_hands.py`, fixed before scoring): palm speed by central difference;
  in each run of >= 2 samples slower than 0.15 m/s the slowest sample is an event; events of the two hands within
  0.5 s and 15 cm are one. Found: within 1 s and 30 cm of an annotated pick-up or put-down, one to one.
Rule: pass if recall >= 60% and precision >= 50% at 0.15 m/s. The curve over other speeds and a control with the
detections moved 15 s later are descriptive.
"""

import argparse
import csv
import io
import json
import re
import statistics
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d import hdepic_hands as K

HAND_CSV = "hand_tracking/wrist_and_palm_poses.csv"
HEALTH_JSON = "vrs_health_check.json"
POSITION_COLUMN = re.compile(r"^t[xyz]_(left|right)_(wrist|palm)_(\w+)$")
POSE_COLUMNS = ["tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device",
                "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
HANDS = ("left", "right")
MIN_RECALL, MIN_PRECISION = 0.60, 0.50
CURVE_SPEEDS = (0.05, 0.10, 0.15, 0.20, 0.30)
CONTROL_SHIFT_S = 15.0
OFFSETS_S = [step / 10 for step in range(-30, 31)]  # clock offsets tried by the two checks
NEAR_SAMPLE_S = 0.15  # palm-to-object check: the palm sample must be this close to the annotated time


def position_frame(columns: list[str]) -> str:
    """The coordinate frame named by the suffix of the wrist/palm position columns ("device", "world", ...)."""
    frames = {match.group(3) for match in map(POSITION_COLUMN.match, columns) if match}
    return frames.pop() if len(frames) == 1 else "unknown"


def summarise_hand_csv(text: str) -> dict:
    """Frame, sample count, sampling interval and share of tracked samples per hand of one wrist/palm file."""
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    stamps = [int(row["tracking_timestamp_us"]) for row in rows]
    steps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"frame": position_frame(reader.fieldnames or []), "samples": len(rows),
            "step_s": round(statistics.median(steps) / 1e6, 4) if steps else None,
            "span_s": round((stamps[-1] - stamps[0]) / 1e6, 1) if stamps else None,
            "tracked": {hand: round(sum(float(row[f"{hand}_tracking_confidence"]) > 0 for row in rows)
                                    / max(len(rows), 1), 3) for hand in HANDS}}


def read_member(archive: zipfile.ZipFile, suffix: str) -> str | None:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    return archive.read(names[0]).decode() if names else None


def read_hand_files(folder: Path) -> dict[str, dict]:
    """Summary of the wrist/palm file in every MPS zip of a folder, keyed by zip name."""
    result = {}
    for path in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            text = read_member(archive, HAND_CSV)
        result[path.name] = summarise_hand_csv(text) if text else {"frame": "missing"}
    return result


def hand_table(text: str) -> dict[str, np.ndarray]:
    """Device times (us) and, per hand, palm positions (device frame) and whether the hand was tracked."""
    table = pd.read_csv(io.StringIO(text))
    result = {"us": table["tracking_timestamp_us"].to_numpy(np.int64)}
    for hand in HANDS:
        result[f"{hand}_palm"] = table[[f"t{a}_{hand}_palm_device" for a in "xyz"]].to_numpy(float)
        result[f"{hand}_valid"] = table[f"{hand}_tracking_confidence"].to_numpy(float) > 0
    return result


def stream_seconds(health: dict) -> dict:
    """Length of the RGB (30 Hz) and SLAM-camera (10 Hz) streams by frame count, from the VRS health check."""
    rgb, slam = health.get("RGB Camera Class #1", {}), health.get("Camera Data (SLAM) #1", {})
    return {"rgb_s": round(rgb.get("processed", 0) / 30, 2), "slam_s": round(slam.get("processed", 0) / 10, 2),
            "rgb_slam_misaligned_us": health.get("Camera Data (SLAM) #1 - RGB Camera Class #1", {})
            .get("largest_misalignment_us")}


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    """Device times (us), translations and quaternions (x, y, z, w) of a closed-loop trajectory, sorted by time."""
    table = pd.read_csv(path, usecols=POSE_COLUMNS).sort_values("tracking_timestamp_us")
    return {"us": table["tracking_timestamp_us"].to_numpy(np.int64), "t": table[POSE_COLUMNS[1:4]].to_numpy(float),
            "q": table[POSE_COLUMNS[4:]].to_numpy(float)}


def world_hands(hands: dict, trajectory: dict) -> dict[str, np.ndarray]:
    """Palm positions moved into the world frame; samples without a nearby pose become untracked."""
    index = K.nearest_poses(trajectory["us"], hands["us"])
    has_pose = index >= 0
    result = {"seconds": K.mp4_seconds(hands["us"], int(hands["us"][0]))}
    for hand in HANDS:
        world = K.to_world(hands[f"{hand}_palm"], trajectory["t"][index], trajectory["q"][index])
        result[f"{hand}_palm"], result[f"{hand}_valid"] = world, hands[f"{hand}_valid"] & has_pose
    return result


def detect(world: dict, slow: float = K.SLOW_SPEED) -> list[K.Event]:
    """Events of both hands, merged."""
    events = []
    for hand in HANDS:
        speed = K.speeds(world["seconds"], world[f"{hand}_palm"], world[f"{hand}_valid"])
        events += K.slow_events(world["seconds"], world[f"{hand}_palm"], speed, slow)
    return K.merge_hands(events)


def device_positions(trajectory: dict, first_us: int, seconds: np.ndarray) -> np.ndarray:
    """Device positions at MP4 times (NaN where no pose is within 10 ms)."""
    index = K.nearest_poses(trajectory["us"], (np.asarray(seconds) * 1e6).astype(np.int64) + first_us)
    return np.where((index >= 0)[:, None], trajectory["t"][index], np.nan)


def wearer_distance_errors(priming: dict, trajectory: dict, first_us: int, offset_s: float) -> list[float]:
    """|device-to-object distance - annotated dist_to_cam| at each priming frame, with the clock moved by offset_s."""
    rows = [(item["prime_stats"]["frame_primed"] / K.FPS, item["3d_location"], item["prime_stats"]["dist_to_cam"])
            for entry in priming.values() for item in (entry.get("start"), entry.get("end"))
            if isinstance(item, dict) and isinstance(item.get("prime_stats"), dict)
            and item["prime_stats"].get("frame_primed", -1) >= 0 and item["prime_stats"].get("dist_to_cam")]
    if not rows:
        return []
    seconds, objects, given = (np.array([r[i] for r in rows], dtype=float) for i in range(3))
    computed = np.linalg.norm(device_positions(trajectory, first_us, seconds + offset_s) - objects, axis=1)
    return [float(e) for e in np.abs(computed - given) if np.isfinite(e)]


def palm_object_distances(world: dict, truths: list[K.Event], offset_s: float) -> list[float]:
    """Distance from the nearest tracked palm (either hand, within 0.15 s) to the object at each annotated event."""
    result = []
    for truth in truths:
        near = np.abs(world["seconds"] - (truth.seconds + offset_s)) <= NEAR_SAMPLE_S
        gaps = [np.linalg.norm(world[f"{hand}_palm"][near & world[f"{hand}_valid"]] - truth.position, axis=1)
                for hand in HANDS]
        gaps = np.concatenate(gaps)
        if len(gaps):
            result.append(float(gaps.min()))
    return result


def pooled(per_video: list[dict]) -> dict:
    """Precision and recall over all videos from per-video counts."""
    detections, truths, matched = (sum(v[k] for v in per_video) for k in ("detections", "truths", "matched"))
    return {"detections": detections, "truths": truths, "matched": matched,
            "precision": round(matched / detections, 4) if detections else None,
            "recall": round(matched / truths, 4) if truths else None}


def verdict(result: dict) -> str:
    """Pass if both thresholds are met at the operating point."""
    ok = (result["recall"] or 0) >= MIN_RECALL and (result["precision"] or 0) >= MIN_PRECISION
    return "pass" if ok else "fail"


def shifted(events: list[K.Event], seconds: float) -> list[K.Event]:
    return [K.Event(e.seconds + seconds, e.position, e.speed) for e in events]


def offset_scan(values_by_offset: dict[float, list[float]]) -> dict:
    """Median per clock offset, the offset with the smallest median, and the median at offset 0."""
    medians = {offset: float(np.median(v)) for offset, v in values_by_offset.items() if v}
    best = min(medians, key=medians.get)
    return {"median_at_0": round(medians[0.0], 3), "best_offset_s": best, "median_at_best": round(medians[best], 3),
            "n_at_0": len(values_by_offset[0.0]), "medians": {str(k): round(v, 3) for k, v in medians.items()}}


def run_video(video: str, hands_zip: Path, trajectory_csv: Path, priming: dict) -> dict:
    """Detections, scores and alignment checks of one video."""
    with zipfile.ZipFile(hands_zip) as archive:
        hands, health = hand_table(read_member(archive, HAND_CSV)), json.loads(read_member(archive, HEALTH_JSON))
    trajectory = load_trajectory(trajectory_csv)
    world, truths = world_hands(hands, trajectory), K.annotated_events(priming)
    first_us = int(hands["us"][0])
    curve = {str(speed): K.precision_recall(detect(world, speed), truths) for speed in CURVE_SPEEDS}
    detections = detect(world)
    return {"video": video, **K.precision_recall(detections, truths), "curve": curve,
            "control": K.precision_recall(shifted(detections, CONTROL_SHIFT_S), truths),
            "streams": {**stream_seconds(health), "hand_s": round(float(world["seconds"][-1]), 2),
                        "trajectory_s": round((trajectory["us"][-1] - trajectory["us"][0]) / 1e6, 2),
                        "trajectory_starts_before_hands_s": round((first_us - trajectory["us"][0]) / 1e6, 3),
                        "last_annotation_s": round(max((t.seconds for t in truths), default=0.0), 2)},
            "with_pose": round(float(np.mean(K.nearest_poses(trajectory["us"], hands["us"]) >= 0)), 4),
            "wearer_errors": {o: wearer_distance_errors(priming, trajectory, first_us, o) for o in OFFSETS_S},
            "palm_distances": {o: palm_object_distances(world, truths, o) for o in OFFSETS_S}}


def summarise(videos: list[dict]) -> dict:
    """Pooled scores, the verdict, the descriptive curve and control, and the clock checks."""
    result = pooled(videos)
    palm_at_0 = [d for v in videos for d in v["palm_distances"][0.0]]
    return {**result, "thresholds": {"recall": MIN_RECALL, "precision": MIN_PRECISION},
            "operating_point_m_per_s": K.SLOW_SPEED, "verdict": verdict(result),
            "curve": {s: pooled([v["curve"][s] for v in videos]) for s in map(str, CURVE_SPEEDS)},
            "control_shift_15s": pooled([v["control"] for v in videos]),
            "clock_check_wearer_distance": offset_scan({o: [e for v in videos for e in v["wearer_errors"][o]]
                                                        for o in OFFSETS_S}),
            "clock_check_palm_to_object": offset_scan({o: [d for v in videos for d in v["palm_distances"][o]]
                                                      for o in OFFSETS_S}),
            "palm_within_30cm_at_0": round(float(np.mean(np.array(palm_at_0) <= K.MATCH_M)), 4) if palm_at_0 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "hd-epic")
    parser.add_argument("--kitchen", default="P02")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "hdepic" / "b1_hand_events.json")
    args = parser.parse_args()
    frames = sorted({e["frame"] for e in read_hand_files(args.data / "gaze_hand" / args.kitchen).values()})
    slam = args.data / "slam" / args.kitchen
    mapping = json.loads((slam / "vrs_to_multi_slam.json").read_text())
    index_of = {Path(name).stem: index for name, index in mapping.items()}
    priming = json.loads((args.data / "annotations" / "eye-gaze-priming" / "priming_info.json").read_text())
    videos, skipped = [], []
    for hands_zip in sorted((args.data / "gaze_hand" / args.kitchen).glob("*.zip")):
        video = hands_zip.stem.removeprefix("mps_").removesuffix("_vrs")
        trajectory_csv = slam / str(index_of.get(video)) / "slam" / "closed_loop_trajectory.csv"
        if video not in priming or not trajectory_csv.exists():
            skipped.append(video)
            continue
        videos.append(run_video(video, hands_zip, trajectory_csv, priming[video]))
        print(f"{video}: {videos[-1]['detections']} detections, {videos[-1]['truths']} annotated, "
              f"{videos[-1]['matched']} matched", flush=True)
    summary = summarise(videos)
    summary.update({"hand_frames": frames, "videos": len(videos), "skipped": skipped,
                    "priming_videos_without_hands": sorted(v for v in priming if v.startswith(f"{args.kitchen}-")
                                                           and v not in {r["video"] for r in videos})})
    per_video = [{k: v for k, v in r.items() if k not in ("wearer_errors", "palm_distances")} for r in videos]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"kitchen": args.kitchen, "summary": summary, "per_video": per_video}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("clock_check_wearer_distance",
                                                                       "clock_check_palm_to_object")}, indent=1))
    for check in ("clock_check_wearer_distance", "clock_check_palm_to_object"):
        print(check, {k: v for k, v in summary[check].items() if k != "medians"})


if __name__ == "__main__":
    main()
