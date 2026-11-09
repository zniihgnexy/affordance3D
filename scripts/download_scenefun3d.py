"""Download one SceneFun3D scene (official URLs and layout; ranged, resumable, CRC-checked).

Example (from the repository root, environment activated):
    python scripts/download_scenefun3d.py --visit 421069 --video 42444490

Default assets are the E0/E1 set from the quick-validation plan: laser scan, crop mask,
annotations, descriptions, hires RGB, hires depth, hires intrinsics and hires poses.
Check sizes first (each scene is roughly 0.3-1.4 GB compressed) against the local budget.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from affordance3d.scenefun3d_download import DEFAULT_ASSETS, download_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True, help="scene (visit_id)")
    parser.add_argument("--video", required=True, help="video sequence (video_id)")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "scenefun3d")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="server folder: train for train/val scenes")
    parser.add_argument("--assets", nargs="+", default=list(DEFAULT_ASSETS))
    args = parser.parse_args()
    download_scene(args.root, args.visit, args.video, tuple(args.assets), args.split)


if __name__ == "__main__":
    main()
