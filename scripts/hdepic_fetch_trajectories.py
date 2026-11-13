"""Fetch only the device trajectory of each HD-EPIC SLAM archive of one kitchen, for B-1 (discussion draft, 25.5).

    python scripts/hdepic_fetch_trajectories.py --list    # sizes only, writes runs/hdepic/b1_trajectories.json
    python scripts/hdepic_fetch_trajectories.py           # then extracts into data/hd-epic/slam/P02/<index>/slam/

The multi-recording SLAM archives (one zip per video, indexed by `vrs_to_multi_slam.json`) also hold point clouds
and observations of several GB. Only the member `<index>/slam/closed_loop_trajectory.csv` (device-to-world pose
in the shared world frame of the kitchen, on the device clock) is read, through HTTP range requests
(`affordance3d.remote_zip`); each member is CRC-checked. The archive directories are read first and nothing is
extracted if the trajectories together are over 1 GB compressed.
"""

import argparse
import json
import sys
import urllib.request
import zipfile
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d import remote_zip
from affordance3d.scenefun3d_download import content_length

BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/SLAM-and-Gaze/{kitchen}/SLAM/multi/"
TRAJECTORY = "slam/closed_loop_trajectory.csv"
MAX_COMPRESSED_BYTES = 10**9


def video_indices(mapping: dict[str, str]) -> dict[str, str]:
    """Video id of every archive index, from `vrs_to_multi_slam.json` ("P02/P02-...-....vrs" -> "0")."""
    return {index: Path(name).stem for name, index in mapping.items()}


def trajectory_member(infos: list, index: str) -> tuple[str, int, int] | None:
    """(name, compressed bytes, bytes) of an archive's trajectory member, if it has one."""
    for info in infos:
        if info.filename == f"{index}/{TRAJECTORY}":
            return info.filename, info.compress_size, info.file_size
    return None


def open_archive(url: str) -> tuple[remote_zip.RangeReader, list]:
    """A range reader of a remote zip with its directory read, and the directory entries."""
    reader = remote_zip.RangeReader(content_length(url), partial(remote_zip.http_range, url))
    remote_zip.member_names(reader)
    with zipfile.ZipFile(reader.clone()) as archive:
        return reader, [info for info in archive.infolist() if not info.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "hd-epic" / "slam")
    parser.add_argument("--kitchen", default="P02")
    parser.add_argument("--list", action="store_true", help="read the archive directories only")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "hdepic" / "b1_trajectories.json")
    args = parser.parse_args()
    base, dest = BASE_URL.format(kitchen=args.kitchen), args.data / args.kitchen
    mapping_path = dest / "vrs_to_multi_slam.json"
    if not mapping_path.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(base + "vrs_to_multi_slam.json", timeout=60) as response:
            mapping_path.write_bytes(response.read())
    videos = video_indices(json.loads(mapping_path.read_text()))
    archives, rows = {}, []
    for index in sorted(videos, key=int):
        reader, infos = open_archive(f"{base}{index}.zip")
        member = trajectory_member(infos, index)
        archives[index] = (reader, member)
        rows.append({"index": index, "video": videos[index], "member": member and member[0],
                     "compressed_bytes": member and member[1], "bytes": member and member[2],
                     "directory_bytes_fetched": reader.fetched_bytes})
    total = sum(row["compressed_bytes"] or 0 for row in rows)
    record = {"kitchen": args.kitchen, "archives": rows, "compressed_total": total,
              "within_limit": total <= MAX_COMPRESSED_BYTES}
    if not args.list and record["within_limit"]:
        for index, (reader, member) in archives.items():
            if member:
                remote_zip.extract_members(reader, [member[0]], dest)
        record["fetched_total"] = sum(reader.fetched_bytes for reader, _ in archives.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    for row in rows:
        print(row)
    print(f"trajectories: {total / 1e6:.1f} MB compressed; within the 1 GB limit: {record['within_limit']}"
          + (f"; fetched {record['fetched_total'] / 1e6:.1f} MB" if "fetched_total" in record else ""))


if __name__ == "__main__":
    main()
