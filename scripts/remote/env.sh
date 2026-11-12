#!/usr/bin/env bash
# Source this on the A6000 server before running anything:  source scripts/remote/env.sh
# Layout: code in the home clone (small files only); data, runs, model caches and the conda
# environment on the NAS. GPU work goes through the shared queue only, never started by hand.
export USER="${USER:-$(id -un)}"  # not set when a daemon (the GPU queue) starts the job
export AFFORDANCE3D_NAS="${AFFORDANCE3D_NAS:-/mnt/nas/$USER/affordance3D}"
export AFFORDANCE3D_ENV="${AFFORDANCE3D_ENV:-/mnt/nas/$USER/envs/affordance3d}"
export HF_HOME="$AFFORDANCE3D_NAS/hf_cache"
# Verified local model copies (scripts/remote/fetch_models.py); the code falls back to the hub without it.
export AFFORDANCE3D_MODELS="$AFFORDANCE3D_NAS/models"
export PIP_CACHE_DIR="$AFFORDANCE3D_NAS/pip_cache"
export MPLCONFIGDIR="$AFFORDANCE3D_NAS/mpl_cache"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$AFFORDANCE3D_ENV"
