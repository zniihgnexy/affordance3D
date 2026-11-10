#!/usr/bin/env bash
# One-time setup on the A6000 server (run from the repository root in the home clone).
# Creates the NAS folders, links the ignored big folders to them and builds the conda environment.
set -euo pipefail
NAS="${AFFORDANCE3D_NAS:-/mnt/nas/$USER/affordance3D}"
ENV="${AFFORDANCE3D_ENV:-/mnt/nas/$USER/envs/affordance3d}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$NAS"/{data/body_models/smplx,runs,report_assets,third_party,hf_cache,pip_cache,mpl_cache,logs}
chmod 700 "$NAS/data/body_models"  # SMPL-X is licensed to the account holder only
for pair in data:data runs:runs third_party:third_party report/assets:report_assets; do
  link="$ROOT/${pair%%:*}"
  [ -e "$link" ] || ln -s "$NAS/${pair##*:}" "$link"
done

source "$HOME/anaconda3/etc/profile.d/conda.sh"
[ -x "$ENV/bin/python" ] || conda create -y -p "$ENV" -c conda-forge --override-channels python=3.12
conda activate "$ENV"
PIP_CACHE_DIR="$NAS/pip_cache" python -m pip install -r "$ROOT/requirements.txt" -i "${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
python "$ROOT/scripts/check_env.py"
