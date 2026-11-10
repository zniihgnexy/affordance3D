#!/usr/bin/env bash
# Fetch the public data the notebooks use, straight from the original servers (resumable).
# Needs only python3, curl, unzip and git. SMPL-X is not public: copy SMPLX_NEUTRAL.npz into
# data/body_models/smplx/ yourself after registering at smpl-x.is.tue.mpg.de.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 scripts/download_scenefun3d.py --visit 421069 --video 42444490
python3 scripts/download_scenefun3d.py --visit 421069 --video 42444490 \
  --assets lowres_depth lowres_wide_intrinsics confidence lowres_poses transform motions
[ -d third_party/scenefun3d/.git ] || git clone -q https://github.com/SceneFun3D/scenefun3d.git third_party/scenefun3d
git -C third_party/scenefun3d checkout -q e37c166  # the toolkit version the loaders were checked against
bash scripts/download_epic_p28_101.sh
bash scripts/download_report_images.sh
echo "data ready under $ROOT/data"
