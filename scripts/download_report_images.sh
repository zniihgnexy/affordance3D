#!/usr/bin/env bash
# Official project images shown in the candidate gallery of report/2026-11-09_datasets_report.ipynb.
# They belong to the dataset/simulator projects and are fetched from their pages, not stored in git.
set -euo pipefail

OUT="$(cd "$(dirname "$0")/.." && pwd)/report/assets/external"
mkdir -p "$OUT"
fetch() { [ -s "$OUT/$1" ] || curl -fL --retry 3 -A "Mozilla/5.0" -o "$OUT/$1" "$2"; }

fetch scenefun3d.png https://scenefun3d.github.io/static/images/teaser.png
fetch epic_visor.png https://epic-kitchens.github.io/VISOR/static/img/sparse.jpg
fetch hoi4d.png https://hoi4d.github.io/teaser.png
fetch rio_3rscan.png https://waldjohannau.github.io/RIO/3Rscan.png
fetch umd.png https://users.umiacs.umd.edu/~fermulcm/affordance/part-affordance-dataset/index_files/dataset_teaser_tools.png
fetch contactpose.png https://raw.githubusercontent.com/facebookresearch/ContactPose/main/readme_images/teaser.png
fetch maniskill.jpg https://raw.githubusercontent.com/mani-skill/ManiSkill/main/figures/teaser.jpg
fetch ai2thor.jpg https://user-images.githubusercontent.com/28768645/115940549-d0134a80-a456-11eb-8737-d89ad75a0f14.jpg

# Aria Digital Twin: the dataset page serves its figures from expiring CDN links, so look the image up by its alt text.
if [ ! -s "$OUT/adt.jpg" ]; then
  page="$(curl -fsL -A "Mozilla/5.0" https://www.projectaria.com/datasets/adt/)"
  url="$(printf '%s' "$page" | python3 -I -c '
import html, re, sys
for tag in re.findall(r"<img[^>]+>", sys.stdin.read()):
    if "including raw footage, object annotations" in tag:
        print(html.unescape(re.search(r"src=\"([^\"]+)\"", tag).group(1)))
        break
')"
  [ -n "$url" ] && curl -fL --retry 3 -A "Mozilla/5.0" -o "$OUT/adt.jpg" "$url"
fi
echo "done: $OUT"
