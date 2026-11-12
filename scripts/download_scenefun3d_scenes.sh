#!/usr/bin/env bash
# Download several SceneFun3D scenes (one video each) with the assets the memory pipeline needs.
#   bash scripts/download_scenefun3d_scenes.sh third_party/scenefun3d/benchmark_file_lists/val_scenes.txt [parallel jobs]
# The video of a scene comes from train_val_set_only_one_video.csv, its ARKitScenes fold from metadata.csv.
# Resumable: scenes that are already complete are checked and skipped by the downloader.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SCENES="$1"
JOBS="${2:-4}"
LISTS="third_party/scenefun3d/benchmark_file_lists"

fetch_one() {
  visit="$1"
  video="$(awk -F, -v v="$visit" '$1 == v {print $2; exit}' "$LISTS/train_val_set_only_one_video.csv")"
  fold="$(awk -F, -v v="$video" '$1 == v {print $4; exit}' "$LISTS/arkitscenes/metadata.csv")"
  if [ -z "$video" ] || [ -z "$fold" ]; then echo "SKIP $visit: no video or fold found"; return 0; fi
  python3 scripts/download_scenefun3d.py --visit "$visit" --video "$video" > /dev/null \
    && python3 scripts/download_scenefun3d.py --visit "$visit" --video "$video" --arkit-fold "$fold" \
         --assets lowres_depth lowres_wide_intrinsics confidence lowres_poses transform motions > /dev/null \
    && echo "OK   $visit $video ($fold)" || echo "FAIL $visit $video ($fold)"
}
export -f fetch_one
export LISTS
grep -v '^\s*$' "$SCENES" | xargs -P "$JOBS" -I{} bash -c 'fetch_one {}'
