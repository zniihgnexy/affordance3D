"""Generic regions for phase 1 (policy P3): segment every keyframe of a scene without any notion of affordance.

    python scripts/make_generic_regions.py --visit 421254

Writes runs/mem_phase1/regions/<visit>.json with one [keyframe, left, top, right, bottom] row per segment.
Needs a GPU to be practical (run it from a queue job with AFFORDANCE3D_DEVICE=cuda).
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import imageio.v3 as iio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401

from affordance3d.mem import keyframes
from affordance3d.mem import pipeline as P
from affordance3d.mem.segmenter import RegionSegmenter
from affordance3d.scenefun3d_partial import N_KEYFRAMES, benchmark_video

LISTS = REPO_ROOT / "third_party" / "scenefun3d" / "benchmark_file_lists"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visit", required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--keyframes", default="steady", choices=keyframes.KINDS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N keyframes (for a quick check)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "mem_phase1" / "regions")
    args = parser.parse_args()
    video = args.video or benchmark_video(LISTS, args.visit)
    scene = P.load_scene(REPO_ROOT / "data" / "scenefun3d", args.visit, video)
    times = keyframes.scene_keyframes(scene, args.keyframes, N_KEYFRAMES)[:args.limit]
    segmenter = RegionSegmenter()
    rows, start = [], time.time()
    for k, t in enumerate(times):
        rgb = iio.imread(scene.vid.rgb_path(t))[::P.PIXEL_STEP, ::P.PIXEL_STEP]
        rows += [[k, *box] for box in segmenter.boxes(rgb)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.visit}.json").write_text(json.dumps({"video": video, "keyframes": args.keyframes, "n_keyframes": len(times),
                                                             "regions": rows}))
    print(f"{args.visit}: {len(rows)} regions in {len(times)} keyframes, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
