"""Download SceneFun3D scenes for the memory pipeline: scene files in full, only the keyframes of each video.

    python scripts/download_scenefun3d_keyframes.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt

The data servers are slow from the lab server (well under 1 MB/s in total), and a video archive holds about a
thousand frames of which the memory uses forty, so the frames are read out of the remote zips one by one.
Scenes are independent and resumable; the keyframe choice of each video is saved in its folder.
"""

import argparse
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.scenefun3d_partial import fetch_keyframes

LISTS = REPO_ROOT / "third_party" / "scenefun3d" / "benchmark_file_lists"


def rows(path: Path) -> list[list[str]]:
    return [line.split(",") for line in path.read_text().split()[1:]]


def fetch(root: Path, visit: str, videos: dict[str, str], folds: dict[str, str]) -> str:
    video = videos.get(visit)
    if video is None or video not in folds:
        return f"SKIP {visit}: no video or fold in the file lists"
    try:
        choice = fetch_keyframes(root, visit, video, arkit_fold=folds[video])
        return f"OK   {visit} {video}: {choice['usable_frames']} usable frames, {len(choice['steady'])} keyframes"
    except (RuntimeError, OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        # one failing scene must not stop the others; the run is resumable
        return f"FAIL {visit} {video}: {type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=Path, required=True, help="text file with one visit id per line")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    parser.add_argument("--jobs", type=int, default=4, help="scenes fetched at the same time")
    args = parser.parse_args()
    videos = {visit: video for visit, video in rows(LISTS / "train_val_set_only_one_video.csv")}
    folds = {row[0]: row[3] for row in rows(LISTS / "arkitscenes" / "metadata.csv")}
    visits = args.scenes.read_text().split()
    with ThreadPoolExecutor(args.jobs) as pool:
        for line in pool.map(lambda visit: fetch(args.root, visit, videos, folds), visits):
            print(line, flush=True)


if __name__ == "__main__":
    main()
