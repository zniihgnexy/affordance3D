#!/usr/bin/env bash
# Put phase-1 runs into the shared GPU queue (GPU work never starts outside the queue).
#   AFFORDANCE3D_QUEUE=<queue folder> bash scripts/remote/queue_phase1.sh <scenes file> [jobs] [priority] [tag]
# <queue folder> is the one that holds pending/. The scenes are dealt out to [jobs] job files (default 3);
# each job needs one card with 16 GB free and may share it. A lower [priority] number runs earlier (default 100).
# For every scene a job makes the generic regions, then runs scripts/run_mem_phase1.py with the model-reading step.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE="${AFFORDANCE3D_QUEUE:?set AFFORDANCE3D_QUEUE to the queue folder (the one that holds pending/)}"
SCENES="$1"
JOBS="${2:-3}"
PRIORITY="${3:-100}"
TAG="${4:-mem_phase1}"
[ -d "$QUEUE/pending" ] || { echo "$QUEUE/pending does not exist"; exit 1; }

for job in $(seq 0 $((JOBS - 1))); do
  visits="$(grep -v '^\s*$' "$SCENES" | awk -v n="$JOBS" -v j="$job" 'NR % n == j' | tr '\n' ' ')"
  [ -n "$visits" ] || continue
  file="$QUEUE/pending/${PRIORITY}_${TAG}_${job}.sh"
  cat > "$file" <<JOB
#Q minmb=16000 share=3
set -uo pipefail
cd "$ROOT"
source scripts/remote/env.sh
export AFFORDANCE3D_DEVICE=cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
for visit in $visits; do
  { [ -f "runs/mem_phase1/regions/\$visit.json" ] || python scripts/make_generic_regions.py --visit "\$visit"; } \\
    && python scripts/run_mem_phase1.py --visit "\$visit" --vlm || echo "FAILED \$visit"
done
JOB
  echo "queued $(basename "$file"): $visits"
done
