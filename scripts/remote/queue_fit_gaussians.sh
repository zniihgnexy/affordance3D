#!/usr/bin/env bash
# Put a Gaussian-map fit (scripts/fit_gaussians.py) into the shared GPU queue (GPU work never starts outside the queue).
#   AFFORDANCE3D_QUEUE=<queue folder> bash scripts/remote/queue_fit_gaussians.sh <config> [priority] [minmb]
# <queue folder> is the one that holds pending/. One job, one card; a lower [priority] number runs earlier (default 100).
# [minmb] is the free memory the card must have (default 18000). gsplat's CUDA code is built once on the NAS
# (TORCH_EXTENSIONS_DIR, see scripts/remote/env.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE="${AFFORDANCE3D_QUEUE:?set AFFORDANCE3D_QUEUE to the queue folder (the one that holds pending/)}"
CONFIG="$1"
PRIORITY="${2:-100}"
MINMB="${3:-18000}"
[ -d "$QUEUE/pending" ] || { echo "$QUEUE/pending does not exist"; exit 1; }
[ -f "$ROOT/$CONFIG" ] || { echo "$CONFIG not found under $ROOT"; exit 1; }
name="$(basename "$CONFIG" .yaml)"
file="$QUEUE/pending/${PRIORITY}_mem_gaussians_${name}.sh"
cat > "$file" <<JOB
#Q minmb=$MINMB share=2
set -uo pipefail
cd "$ROOT"
source scripts/remote/env.sh
export AFFORDANCE3D_DEVICE=cuda OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/fit_gaussians.py --config "$CONFIG" || echo "FAILED $CONFIG"
JOB
echo "queued $(basename "$file")"
