"""Fetch a SceneFun3D video for the memory pipeline without its full archives.

The scene files (scan, annotations, trajectories) are downloaded whole. Of the frame archives only the
keyframes are fetched: they are chosen first, from the archive directories and the device trajectory alone,
then read out of the remote zips. The choice is saved next to the frames (`keyframes.json`).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from affordance3d import remote_zip
from affordance3d.mem import keyframes as K
from affordance3d.scenefun3d import read_traj_poses
from affordance3d.scenefun3d_download import asset_files, asset_url, content_length, download_scene

N_KEYFRAMES = 40
SCENE_ASSETS = ("laser_scan_5mm", "crop_mask", "annotations", "descriptions", "motions", "hires_poses", "lowres_poses",
                "transform")
FRAME_ARCHIVES = {"hires_wide": ".jpg", "hires_depth": ".png", "hires_wide_intrinsics": ".pincam"}
LIDAR_ARCHIVES = ("lowres_depth", "confidence")
MAX_POSE_GAP_S, MAX_LIDAR_GAP_S = 0.05, 0.02  # as in affordance3d.mem.inputs
KEYFRAME_FILE = "keyframes.json"


def benchmark_video(lists: Path, visit: str) -> str:
    """The one video the benchmark file list assigns to a scene."""
    rows = (line.split(",") for line in (lists / "train_val_set_only_one_video.csv").read_text().split()[1:])
    return dict(rows)[visit]


def stamp(name: str) -> str:
    """`<folder>/<video>_<timestamp>.<ext>` -> timestamp."""
    return Path(name).stem.split("_", 1)[1]


def usable_times(frame_stamps: list[set[str]], gt_pose_times: np.ndarray, arkit_times: np.ndarray,
                 lidar_times: np.ndarray) -> list[str]:
    """Frames the pipeline can use and score: in every frame archive, with a device pose, a LiDAR frame and a ground-truth pose nearby."""
    def near(times: np.ndarray, t: float, gap: float) -> bool:
        return bool(np.min(np.abs(times - t)) <= gap)

    common = set.intersection(*frame_stamps)
    return [t for t in sorted(common, key=float)
            if near(arkit_times, float(t), MAX_POSE_GAP_S) and near(lidar_times, float(t), MAX_LIDAR_GAP_S)
            and near(gt_pose_times, float(t), MAX_POSE_GAP_S)]


def choose_keyframes(times: list[str], rates: np.ndarray, n: int = N_KEYFRAMES) -> dict[str, list[str]]:
    """The two keyframe sets compared in phase 1: evenly spaced, and the steadiest frame of each time window."""
    return {"uniform": [times[i] for i in K.uniform_indices(len(times), n)],
            "steady": [times[i] for i in K.steady_indices(rates, n)]}


def _open(root: Path, visit: str, video: str, asset: str, split: str, arkit_fold: str) -> tuple[remote_zip.RangeReader, list[str]]:
    url = asset_url(asset_files(visit, video, asset)[0], split, arkit_fold)
    reader = remote_zip.RangeReader(content_length(url), partial(remote_zip.http_range, url))
    return reader, remote_zip.member_names(reader)


def fetch_keyframes(root: Path, visit: str, video: str, split: str = "train", arkit_fold: str = "Training",
                    n: int = N_KEYFRAMES, workers: int = 6) -> dict:
    """Download the scene files and only the keyframes of the video; returns the saved keyframe choice."""
    folder = root / visit / video
    saved = folder / KEYFRAME_FILE
    if saved.exists():  # written last, so the scene is complete
        return json.loads(saved.read_text())
    download_scene(root, visit, video, SCENE_ASSETS, split, arkit_fold)
    archives = list(FRAME_ARCHIVES) + list(LIDAR_ARCHIVES)
    with ThreadPoolExecutor(len(archives)) as pool:
        opened = dict(zip(archives, pool.map(lambda a: _open(root, visit, video, a, split, arkit_fold), archives)))
    arkit = read_traj_poses(folder / "lowres_poses.traj")
    arkit_times = np.array([float(t) for t in arkit])
    lidar_times = np.array(sorted(float(stamp(name)) for name in opened["lowres_depth"][1]))
    gt_times = np.array([float(t) for t in read_traj_poses(folder / "hires_poses.traj")])
    times = usable_times([{stamp(name) for name in opened[a][1]} for a in FRAME_ARCHIVES], gt_times, arkit_times, lidar_times)
    rates = K.turn_rates([float(t) for t in times], arkit_times, [pose[:3, :3] for pose in arkit.values()])
    choice = choose_keyframes(times, rates, n)
    choice["turn_rate_deg_s"] = {t: round(float(r), 2) for t, r in zip(times, rates)}
    choice["usable_frames"] = len(times)
    wanted = sorted(set(choice["uniform"]) | set(choice["steady"]), key=float)
    lidar_names = sorted(opened["lowres_depth"][1], key=lambda name: float(stamp(name)))
    lidar_values = np.array([float(stamp(name)) for name in lidar_names])
    nearest = {Path(lidar_names[int(np.argmin(np.abs(lidar_values - float(t))))]).name for t in wanted}
    for asset, suffix in FRAME_ARCHIVES.items():
        names = [f"{asset}/{video}_{t}{suffix}" for t in wanted]
        remote_zip.extract_members(opened[asset][0], names, folder, workers)
    for asset in LIDAR_ARCHIVES:
        remote_zip.extract_members(opened[asset][0], [f"{asset}/{name}" for name in sorted(nearest)], folder, workers)
    saved.write_text(json.dumps(choice, indent=1))
    return choice
