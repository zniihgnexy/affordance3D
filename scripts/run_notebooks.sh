#!/usr/bin/env bash
# Execute the committed notebooks, in dependency order, into runs/notebook_check/ (the committed
# copies are left untouched). Pass notebook paths to run only some of them.
# On the server, `source scripts/remote/env.sh` first; everything here runs on the CPU.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="runs/notebook_check"
mkdir -p "$OUT"
if [ "$#" -gt 0 ]; then NOTEBOOKS=("$@"); else NOTEBOOKS=(
  notebooks/01_step_by_step_demo.ipynb
  notebooks/02_scenefun3d_e0.ipynb
  report/2026-11-09_datasets_report.ipynb
  notebooks/mem/S1_01_video_inputs_to_actionable_affordance.ipynb
  notebooks/mem/S1_02_humanoid_approach.ipynb
  report/2026-11-10_mem_pipeline_demo.ipynb
  notebooks/mem/pipeline_examples.ipynb  # needs the models from scripts/remote/fetch_models.py; ~15 min extra on a first run
  notebooks/mem/phase1_results.ipynb  # needs the phase-1 results in runs/mem_phase1 (GPU queue jobs, then scripts/aggregate_mem_phase1.py)
); fi
failed=0
for notebook in "${NOTEBOOKS[@]}"; do
  start=$(date +%s)
  # Run from the notebook's own folder so that relative paths behave as in Jupyter.
  if (cd "$(dirname "$notebook")" && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
        --output-dir "$ROOT/$OUT" "$(basename "$notebook")" > "$ROOT/$OUT/$(basename "$notebook" .ipynb).log" 2>&1); then
    echo "PASS  $notebook  ($(( $(date +%s) - start )) s)"
  else
    echo "FAIL  $notebook  (see $OUT/$(basename "$notebook" .ipynb).log)"
    failed=1
  fi
done
exit $failed
