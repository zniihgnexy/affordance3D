#!/usr/bin/env bash
# One-time setup for the 3D Gaussian map on the A6000 server (after scripts/remote/setup.sh; run from the repository root).
# The environment's torch is built with CUDA 13.2 and gsplat has no prebuilt wheel for it, so gsplat compiles its CUDA
# code once (on the CPU, for the A6000's sm_86) with the pip CUDA 13.2 compiler. Torch itself is never touched (--no-deps).
# CUDA_HOME is a folder of links on the NAS: the pip layout lacks the unversioned libcudart.so the linker asks for.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source scripts/remote/env.sh
PIP_CACHE_DIR="$AFFORDANCE3D_NAS/pip_cache" python -m pip install --no-deps -r requirements-gaussians.txt \
  -i "${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_CUDA="$(python -c 'import nvidia, os; print(os.path.join(nvidia.__path__[0], "cu13"))')"
mkdir -p "$CUDA_HOME/lib"
ln -sfn "$PIP_CUDA/bin" "$CUDA_HOME/bin"
ln -sfn "$PIP_CUDA/include" "$CUDA_HOME/include"
for f in "$PIP_CUDA"/lib/*; do ln -sfn "$f" "$CUDA_HOME/lib/$(basename "$f")"; done
ln -sfn libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
ln -sfn lib "$CUDA_HOME/lib64"
"$CUDA_HOME/bin/nvcc" --version | tail -1
mkdir -p "$TORCH_EXTENSIONS_DIR"
CUDA_VISIBLE_DEVICES= MAX_JOBS=8 python -c "from gsplat.cuda._backend import _C; assert _C is not None; print('gsplat CUDA code built')"
