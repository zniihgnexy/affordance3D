#!/usr/bin/env bash
# Put the tool-map runs of section 24.9 into the shared GPU queue (GPU work never starts outside the queue).
#   AFFORDANCE3D_QUEUE=<queue folder> bash scripts/remote/queue_tool_maps.sh <scenes file> <tool> [jobs] [priority] [tag] [extra]
# <queue folder> is the one that holds pending/. <tool> is sam3, qwen8b or sam3_tasks (see scripts/run_tool_maps.py). The
# scenes are dealt out to [jobs] job files (default 1); each job needs one card and may share it: SAM 3 asks for 12 GB free,
# Qwen3-VL-8B for 30 GB. A lower [priority] number runs earlier (default 100). [extra] goes to scripts/run_tool_maps.py
# (e.g. --gate for the two keyframes of the gate figure). A run that stops resumes where it stopped when queued again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE="${AFFORDANCE3D_QUEUE:?set AFFORDANCE3D_QUEUE to the queue folder (the one that holds pending/)}"
SCENES="$1"
TOOL="$2"
JOBS="${3:-1}"
PRIORITY="${4:-100}"
TAG="${5:-mem_tools_$TOOL}"
EXTRA="${6:-}"
case "$TOOL" in sam3|sam3_tasks) MINMB=12000 ;; qwen8b) MINMB=30000 ;; *) echo "unknown tool $TOOL"; exit 1 ;; esac
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
  python scripts/run_tool_maps.py --visit "\$visit" --tool "$TOOL" $EXTRA || echo "FAILED \$visit"
done
JOB
  echo "queued $(basename "$file"): $visits"
done
