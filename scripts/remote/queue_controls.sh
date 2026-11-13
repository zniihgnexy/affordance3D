#!/usr/bin/env bash
# Put the round-7 controls into the shared GPU queue (GPU work never starts outside the queue).
#   AFFORDANCE3D_QUEUE=<queue folder> bash scripts/remote/queue_controls.sh <scenes file> [jobs] [priority] [tag] [arms] [model]
# <queue folder> is the one that holds pending/. The scenes are dealt out to [jobs] job files (default 3); each job needs
# one card with 16 GB free and may share it. A lower [priority] number runs earlier (default 100). [arms] is passed to
# scripts/run_mem_controls.py (default: its first batch). [model] is 4b (default) or 8b; an 8B job asks for a card with
# 30 GB free.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE="${AFFORDANCE3D_QUEUE:?set AFFORDANCE3D_QUEUE to the queue folder (the one that holds pending/)}"
SCENES="$1"
JOBS="${2:-3}"
PRIORITY="${3:-100}"
TAG="${4:-mem_controls}"
ARMS="${5:-p4n,c1,c2,c3}"
MODEL="${6:-4b}"
case "$MODEL" in 4b) MINMB=16000 ;; 8b) MINMB=30000 ;; *) echo "unknown model $MODEL"; exit 1 ;; esac
[ -d "$QUEUE/pending" ] || { echo "$QUEUE/pending does not exist"; exit 1; }

for job in $(seq 0 $((JOBS - 1))); do
  visits="$(grep -v '^\s*$' "$SCENES" | awk -v n="$JOBS" -v j="$job" 'NR % n == j' | tr '\n' ' ')"
  [ -n "$visits" ] || continue
  file="$QUEUE/pending/${PRIORITY}_${TAG}_${job}.sh"
  cat > "$file" <<JOB
#Q minmb=$MINMB share=3
set -uo pipefail
cd "$ROOT"
source scripts/remote/env.sh
export AFFORDANCE3D_DEVICE=cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
for visit in $visits; do
  python scripts/run_mem_controls.py --visit "\$visit" --arms "$ARMS" --model "$MODEL" || echo "FAILED \$visit"
done
JOB
  echo "queued $(basename "$file"): $visits"
done
