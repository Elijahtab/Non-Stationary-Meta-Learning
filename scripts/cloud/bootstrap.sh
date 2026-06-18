#!/usr/bin/env bash
# Bootstrap a fresh Linux cloud GPU box for Lifelong-Learning sweeps.
#
# Designed for a "PyTorch 2.x + CUDA" base image (RunPod / Vast.ai), where torch is
# already installed and CUDA-matched. We therefore REUSE the image's torch and install
# only the remaining deps, which avoids the #1 cloud failure mode (a pip-installed torch
# that mismatches the driver/CUDA).
#
# Usage (on the cloud box, after `git clone`):
#   cd Lifelong-Learning
#   bash scripts/cloud/bootstrap.sh
#
# Toggles (env vars):
#   INSTALL_TF=1     also install tensorflow (default 0 — heavy, and not needed for sweeps)
#   CREATE_VENV=1    create a .venv instead of using the image's python (default 0)
#   REPO_URL=...     git URL to clone if not already in a repo (optional)
#
# NOTE: untested end-to-end on a fresh cloud image — validate on first instance and
# adjust. Treat failures here as expected first-run friction, not a bug in the science.
set -euo pipefail

INSTALL_TF="${INSTALL_TF:-0}"
CREATE_VENV="${CREATE_VENV:-0}"

echo "[bootstrap] host: $(uname -a)"
echo "[bootstrap] cores: $(nproc)  | started $(date -u +%FT%TZ)"

# --- system packages (best-effort; skip if no apt / no sudo) -----------------------------
if command -v apt-get >/dev/null 2>&1; then
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update -y || true
    # libgl1/libglib for matplotlib + minigrid rendering on headless boxes
    $SUDO apt-get install -y git build-essential libgl1 libglib2.0-0 || true
fi

# --- python env --------------------------------------------------------------------------
PY=python3
if [ "$CREATE_VENV" = "1" ]; then
    echo "[bootstrap] creating .venv (will NOT inherit image torch)"
    $PY -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY=python
fi
$PY -m pip install --upgrade pip

# --- torch: reuse if present, else install a CUDA build -----------------------------------
if $PY -c "import torch" 2>/dev/null; then
    echo "[bootstrap] reusing preinstalled torch: $($PY -c 'import torch; print(torch.__version__)')"
else
    echo "[bootstrap] torch not found — installing CUDA 12.1 build (adjust for your image)"
    $PY -m pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# --- core deps (exclude torch; tensorflow optional) --------------------------------------
$PY -m pip install \
    "numpy>=1.24" "gymnasium>=0.29" "minigrid>=2.3" "tqdm>=4.66" "pillow>=10.0" \
    "pandas>=2.1" "tensorboard>=2.15" "matplotlib>=3.8" "scipy>=1.11"
if [ "$INSTALL_TF" = "1" ]; then
    echo "[bootstrap] installing tensorflow (heavy; only if you need it)"
    $PY -m pip install "tensorflow>=2.16"
fi

# --- register the package (no deps; we manage deps above) --------------------------------
$PY -m pip install -e . --no-deps || echo "[bootstrap] editable install skipped; will use PYTHONPATH=src"

# --- verify ------------------------------------------------------------------------------
echo "[bootstrap] verifying..."
PYTHONPATH=src $PY - <<'PYCHECK'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import lifelong_learning  # noqa: F401
from lifelong_learning.research.benchmarking import list_frozen_benchmarks
print("benchmarks:", [b["name"] for b in list_frozen_benchmarks()])
print("import OK")
PYCHECK

echo "[bootstrap] running tiny smoke (pilot sweep, 1 seed)..."
PYTHONPATH=src $PY scripts/run_seed_sweep.py --preset pilot --conditions brain_neuromod \
    --seeds 0 --out sweeps/cloud_smoke --device "${SMOKE_DEVICE:-cuda}" || \
    echo "[bootstrap] smoke failed — inspect sweeps/cloud_smoke/logs before launching the real sweep"

echo "[bootstrap] done $(date -u +%FT%TZ). Next: bash scripts/cloud/run_sweep.sh"
