#!/usr/bin/env bash
# Put SAM 3 tool maps of the second-visit videos (stage 2) into the shared GPU queue (GPU work never starts outside it).
#   AFFORDANCE3D_QUEUE=<queue folder> bash scripts/remote/queue_tool_maps_revisit.sh <pairs file> [jobs] [priority]
# <pairs file> has one "<visit> <video>" per line (configs/stage2/revisit_videos.txt). The videos are dealt out to [jobs]
# job files (default 2); each needs one card with 12 GB free and may share it. Records go to
# runs/stage2/tool_maps/sam3/<visit>_<video>.json and resume where a stopped run stopped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CARDS="$(cat "$ROOT/scripts/remote/allowed_cards")"  # the only GPUs this project may use (never 2 or 3)
QUEUE="${AFFORDANCE3D_QUEUE:?set AFFORDANCE3D_QUEUE to the queue folder (the one that holds pending/)}"
PAIRS="$1"
JOBS="${2:-2}"
PRIORITY="${3:-100}"
[ -d "$QUEUE/pending" ] || { echo "$QUEUE/pending does not exist"; exit 1; }
for job in $(seq 0 $((JOBS - 1))); do
  pairs="$(grep -v '^\s*$' "$ROOT/$PAIRS" | awk -v n="$JOBS" -v j="$job" 'NR % n == j' | tr ' ' ':' | tr '\n' ' ')"
  [ -n "$pairs" ] || continue
  file="$QUEUE/pending/${PRIORITY}_mem_tools_revisit_${job}.sh"
  cat > "$file" <<JOB
#Q minmb=12000 share=3 cards=$CARDS
set -uo pipefail
cd "$ROOT"
source scripts/remote/env.sh
export AFFORDANCE3D_DEVICE=cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
for pair in $pairs; do
  visit="\${pair%%:*}"; video="\${pair##*:}"
  python scripts/run_tool_maps.py --visit "\$visit" --video "\$video" --data-root data/scenefun3d_revisit --runs runs/stage2 --tool sam3 || echo "FAILED \$pair"
done
JOB
  echo "queued $(basename "$file")"
done
