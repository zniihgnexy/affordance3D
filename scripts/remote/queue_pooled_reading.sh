#!/usr/bin/env bash
# Put the end-to-end pooled reading of phase 2b (discussion draft, section 26.2) into the shared GPU queue (GPU work never
# starts outside the queue).
#   AFFORDANCE3D_QUEUE=<queue folder> bash scripts/remote/queue_pooled_reading.sh <scenes file> [jobs] [priority] [tag] [model]
# <queue folder> is the one that holds pending/. The target scenes are dealt out to [jobs] job files (default 3); each job
# runs scripts/run_pooled_reading.py on its scenes in turn and may share its card. A lower [priority] number runs earlier
# (default 100). [model] is 8b (default; asks for a card with 30 GB free) or 4b (16 GB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CARDS="$(cat "$ROOT/scripts/remote/allowed_cards")"  # the only GPUs this project may use (never 2 or 3)
QUEUE="${AFFORDANCE3D_QUEUE:?set AFFORDANCE3D_QUEUE to the queue folder (the one that holds pending/)}"
SCENES="$1"
JOBS="${2:-3}"
PRIORITY="${3:-100}"
TAG="${4:-mem_pooled_reading}"
MODEL="${5:-8b}"
case "$MODEL" in 4b) MINMB=16000 ;; 8b) MINMB=30000 ;; *) echo "unknown model $MODEL"; exit 1 ;; esac
[ -d "$QUEUE/pending" ] || { echo "$QUEUE/pending does not exist"; exit 1; }

for job in $(seq 0 $((JOBS - 1))); do
  visits="$(grep -v '^\s*$' "$SCENES" | awk -v n="$JOBS" -v j="$job" 'NR % n == j' | tr '\n' ' ')"
  [ -n "$visits" ] || continue
  file="$QUEUE/pending/${PRIORITY}_${TAG}_${job}.sh"
  cat > "$file" <<JOB
#Q minmb=$MINMB share=3 cards=$CARDS
set -uo pipefail
cd "$ROOT"
source scripts/remote/env.sh
export AFFORDANCE3D_DEVICE=cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
for visit in $visits; do
  python scripts/run_pooled_reading.py --visit "\$visit" --model "$MODEL" || echo "FAILED \$visit"
done
JOB
  echo "queued $(basename "$file"): $visits"
done
