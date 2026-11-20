#!/usr/bin/env bash
# Put a sharded Gaussian-map batch (scripts/fit_gaussians_batch.py) into the shared GPU queue (never started outside it).
#   AFFORDANCE3D_QUEUE=<queue folder> bash scripts/remote/queue_fit_gaussians_batch.sh <config> [jobs] [priority] [minmb]
# One job file per shard (default 6), one card each; a lower [priority] number runs earlier (default 100, in front of
# the project lead's own jobs, as agreed on 2026-11-20 for stage 2). [minmb] is the free memory a card must have
# (default 8000; the trial's 80-frame video peaked at 3.7 GB). Stopped jobs resume video by video when queued again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CARDS="$(cat "$ROOT/scripts/remote/allowed_cards")"  # the only GPUs this project may use (never 2 or 3)
QUEUE="${AFFORDANCE3D_QUEUE:?set AFFORDANCE3D_QUEUE to the queue folder (the one that holds pending/)}"
CONFIG="$1"
JOBS="${2:-6}"
PRIORITY="${3:-100}"
MINMB="${4:-8000}"
[ -d "$QUEUE/pending" ] || { echo "$QUEUE/pending does not exist"; exit 1; }
[ -f "$ROOT/$CONFIG" ] || { echo "$CONFIG not found under $ROOT"; exit 1; }
name="$(basename "$CONFIG" .yaml)"
for job in $(seq 0 $((JOBS - 1))); do
  file="$QUEUE/pending/${PRIORITY}_mem_gaussians_${name}_${job}.sh"
  cat > "$file" <<JOB
#Q minmb=$MINMB share=2 cards=$CARDS
set -uo pipefail
cd "$ROOT"
source scripts/remote/env.sh
export AFFORDANCE3D_DEVICE=cuda OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/fit_gaussians_batch.py --config "$CONFIG" --shard $job/$JOBS || echo "FAILED shard $job/$JOBS"
JOB
  echo "queued $(basename "$file")"
done
