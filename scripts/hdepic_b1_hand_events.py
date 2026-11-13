"""B-1, first step: are the Aria wrist/palm positions usable for finding pick-ups and put-downs? (draft, 25.2)

    python scripts/hdepic_b1_hand_events.py     # reads data/hd-epic/gaze_hand/P02/*.zip, writes runs/hdepic/b1_hand_events.json

CPU only. The plan detects interaction moments from wrist/palm speed and scores them against the annotated
pick-up/put-down frames and 3D locations, which are in world coordinates. This step reads the MPS hand files
(inside the zips, without extracting them) and finds out which frame their positions are in. If they are not in
world coordinates, the device pose is needed; the plan allows it only if a HEAD request shows a download of at
most 1 GB that needs no login. Otherwise B-1 stops here and the result says why (the SLAM archives are not a
substitute). The intermediate-data link is read from the annotations README.
"""

import argparse
import csv
import io
import json
import re
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HAND_CSV = "hand_tracking/wrist_and_palm_poses.csv"
POSITION_COLUMN = re.compile(r"^t[xyz]_(left|right)_(wrist|palm)_(\w+)$")
POSE_LINK = re.compile(r"\[Download intermediate data\]\((https://[^)\s]+)\)")
MAX_POSE_BYTES = 10**9
TIMEOUT_S = 60


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
                                    / max(len(rows), 1), 3) for hand in ("left", "right")}}


def read_hand_files(folder: Path) -> dict[str, dict]:
    """Summary of the wrist/palm file in every MPS zip of a folder, keyed by zip name."""
    result = {}
    for path in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(HAND_CSV)]
            result[path.name] = (summarise_hand_csv(archive.read(members[0]).decode()) if members
                                 else {"frame": "missing"})
    return result


def head(url: str) -> dict:
    """Status, size and final host of a HEAD request (redirects followed); no body is downloaded."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            length = response.headers.get("Content-Length")
            return {"status": response.status, "bytes": int(length) if length else None,
                    "content_type": response.headers.get("Content-Type"),
                    "final_host": urllib.parse.urlsplit(response.url).hostname}
    except urllib.error.HTTPError as error:
        return {"status": error.code, "bytes": None, "content_type": error.headers.get("Content-Type"),
                "final_host": urllib.parse.urlsplit(error.url or url).hostname}
    except urllib.error.URLError as error:
        return {"status": None, "bytes": None, "error": str(error.reason)}


def pose_gate(check: dict) -> tuple[bool, str]:
    """Whether the device pose may be used under the plan: reachable without login and at most 1 GB."""
    if check.get("status") != 200:
        return False, f"not reachable (HTTP status {check.get('status')})"
    if "login" in (check.get("final_host") or ""):
        return False, "needs a login"
    if check.get("bytes") is None or "text/html" in (check.get("content_type") or ""):
        return False, "download size unknown (the link is a web page, not a file)"
    if check["bytes"] > MAX_POSE_BYTES:
        return False, f"{check['bytes'] / 1e9:.2f} GB is over 1 GB"
    return True, f"{check['bytes'] / 1e6:.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "hd-epic")
    parser.add_argument("--kitchen", default="P02")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "hdepic" / "b1_hand_events.json")
    args = parser.parse_args()
    hands = read_hand_files(args.data / "gaze_hand" / args.kitchen)
    frames = sorted({entry["frame"] for entry in hands.values()})
    record: dict = {"kitchen": args.kitchen, "hand_files": hands, "frames": frames}
    if frames == ["world"]:
        record["status"] = "hand positions are in world coordinates: no pose needed (detection not part of this step)"
    else:
        match = POSE_LINK.search((args.data / "annotations" / "README.md").read_text())
        check = head(match.group(1)) if match else {"status": None, "error": "no intermediate-data link in README"}
        usable, reason = pose_gate(check)
        record.update({"pose_needed": True, "pose_check": check, "pose_usable": usable,
                       "status": (f"device pose usable ({reason})" if usable
                                  else f"stopped: hand positions are in the {'/'.join(frames)} frame and the "
                                       f"device pose is {reason}")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    for name, entry in hands.items():
        print(f"{name:40s} {entry}")
    print(record["status"])


if __name__ == "__main__":
    main()
