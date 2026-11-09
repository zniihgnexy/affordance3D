#!/usr/bin/env bash
# Data for the EPIC-KITCHENS + VISOR section of report/2026-11-09_datasets_report.ipynb (~125 MB).
# One video, P28_101: VISOR sparse frames + masks, VISOR frame mapping, EPIC-KITCHENS-100
# narrations and the EPIC-Fields example reconstruction. Non-commercial licences apply.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/epic_visor"
VISOR="https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7"
mkdir -p "$OUT/P28_101"

fetch() { [ -s "$1" ] || curl -fL --retry 3 -o "$1" "$2"; }
fetch "$OUT/P28_101/visor_annotations.json" "$VISOR/GroundTruth-SparseAnnotations/annotations/train/P28_101.json"
fetch "$OUT/frame_mapping.json" "$VISOR/frame_mapping.json"
fetch "$OUT/EPIC_100_train.csv" \
  "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv"
fetch "$OUT/P28_101/epic_fields.json" \
  "https://raw.githubusercontent.com/epic-kitchens/epic-fields-code/main/example_data/P28_101.json"
if [ ! -d "$OUT/P28_101/frames" ]; then
  fetch "$OUT/P28_101/visor_frames.zip" "$VISOR/GroundTruth-SparseAnnotations/rgb_frames/train/P28/P28_101.zip"
  unzip -tq "$OUT/P28_101/visor_frames.zip"
  unzip -q "$OUT/P28_101/visor_frames.zip" -d "$OUT/P28_101/frames"
  rm "$OUT/P28_101/visor_frames.zip"
fi
echo "done: $OUT"
