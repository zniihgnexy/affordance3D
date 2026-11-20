"""Download the other videos of each scene (the "second visit" test, week-2 notebook 7.9 question 3, chosen on 2026-11-20),
keyframes only, into a separate root so that code reading "the" video of a scene under data/scenefun3d is unchanged.

    python scripts/download_revisit_videos.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt

The scene files (scan, crop, annotations, descriptions, motions) are linked from the primary root, not downloaded again;
each video brings its own poses, scan registration and 40 keyframes (affordance3d.scenefun3d_partial.fetch_keyframes).
Resumable; one failing video does not stop the others.
"""

import argparse
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.scenefun3d_partial import fetch_keyframes  # noqa: E402

LISTS = REPO_ROOT / "third_party" / "scenefun3d" / "benchmark_file_lists"
SCENE_FILES = ("{visit}_laser_scan.ply", "{visit}_crop_mask.npy", "{visit}_annotations.json",
               "{visit}_descriptions.json", "{visit}_motions.json")


def rows(path: Path) -> list[list[str]]:
    return [line.split(",") for line in path.read_text().split()[1:]]


def other_videos(visits: list[str], pairs: list[list[str]], primary: dict[str, str]) -> list[tuple[str, str]]:
    """(visit, video) for every video of the given visits except the primary one, in list order."""
    wanted = set(visits)
    return [(visit, video) for visit, video, *_ in pairs if visit in wanted and video != primary.get(visit)]


def link_scene_files(primary_root: Path, root: Path, visit: str) -> None:
    """Link the visit's scene files from the primary root (they are the same for every video of the visit)."""
    (root / visit).mkdir(parents=True, exist_ok=True)
    for pattern in SCENE_FILES:
        name = pattern.format(visit=visit)
        source, link = primary_root / visit / name, root / visit / name
        if source.exists() and not link.exists():
            link.symlink_to(source)


def fetch(primary_root: Path, root: Path, visit: str, video: str, fold: str) -> str:
    try:
        link_scene_files(primary_root, root, visit)
        choice = fetch_keyframes(root, visit, video, arkit_fold=fold)
        return f"OK   {visit} {video}: {choice['usable_frames']} usable frames, {len(choice['steady'])} keyframes"
    except (RuntimeError, OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        return f"FAIL {visit} {video}: {type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=Path, required=True, help="text file with one visit id per line")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "scenefun3d_revisit")
    parser.add_argument("--primary-root", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    parser.add_argument("--jobs", type=int, default=4, help="videos fetched at the same time")
    args = parser.parse_args()
    primary = {visit: video for visit, video in rows(LISTS / "train_val_set_only_one_video.csv")}
    folds = {row[0]: row[3] for row in rows(LISTS / "arkitscenes" / "metadata.csv")}
    todo = other_videos(args.scenes.read_text().split(), rows(LISTS / "train_val_set.csv"), primary)
    print(f"{len(todo)} videos to fetch", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        for line in pool.map(lambda vv: fetch(args.primary_root, args.root, vv[0], vv[1], folds.get(vv[1], "Training")), todo):
            print(line, flush=True)


if __name__ == "__main__":
    main()
