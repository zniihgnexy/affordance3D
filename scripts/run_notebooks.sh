#!/usr/bin/env bash
# Execute notebooks into runs/notebook_check/ (the committed copies are left untouched): by default the current
# week's notebook; pass notebook paths to run others.
# On the server, `source scripts/remote/env.sh` first; everything here runs on the CPU.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="runs/notebook_check"
mkdir -p "$OUT"
# One notebook per week (notebooks/builders/README.md). Week 1 is a merge of saved outputs and is not re-run as a whole:
# to re-run one of its parts, take the part back with `git show 3f2baae:<path> > <path>` and pass that path here.
if [ "$#" -gt 0 ]; then NOTEBOOKS=("$@"); else NOTEBOOKS=(
  report/week2_2026-11-16_to_11-22.ipynb  # needs the part-vector runs, the public-evaluation run and the two week-2 videos
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
