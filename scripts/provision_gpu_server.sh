#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision_gpu_server.sh — reproducible server bring-up for the Hunyuan3D
# 2.1 / 2mv API WITH the texture (paint) stage working.
#
# Tested on Ubuntu 24.04, RTX 2080 Ti 22 GB (cc 7.5). Needs >= 50-60 GB disk.
#
# Key facts baked in (learned the hard way):
#   * bpy only ships cp311 wheels on PyPI -> Python 3.11 via uv.
#   * pymeshlab & bpy both bundle libembree4.so.4 (different versions).
#     bpy's must load FIRST -> run.sh exports LD_LIBRARY_PATH to bpy/lib.
#   * pymeshlab needs system libOpenGL.so.0 or its I/O plugins fail
#     ("Unknown format for load: obj" / "Output Stream Error").
#   * realesrgan pulls opencv-python 5 (a ~500 MB wheel); --no-deps is used
#     because the app already ships cv2 via opencv-python-headless.
#   * basicsr 1.4.2 imports torchvision.transforms.functional_tensor which was
#     removed in torchvision 0.19+ -> sed patch to transforms.functional.
#   * paint pipeline needs ckpt/RealESRGAN_x4plus.pth (manual download).
# ---------------------------------------------------------------------------
set -euo pipefail

: "${VENV_DIR:=$PWD/.venv}"
: "${PY_VER:=3.11}"
: "${TORCH_CUDA_ARCH_LIST:=7.5}"        # adjust to your GPU's compute capability
MIRROR=https://mirrors.aliyun.com/pypi/simple/
CU124=https://download.pytorch.org/whl/cu124

log() { echo "-- $*"; }

# 1) System libs (bpy headless, pymeshlab GL plugins, X11 for blender)
apt-get update -yq >/dev/null
apt-get install -yq libopengl0 libglx-mesa0 libgl1 libegl1 \
  libsm6 libice6 libx11-6 libxrender1 libxext6 libxcb1 >/dev/null

# 2) uv + Python 3.11 (bpy needs cp311)
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install "$PY_VER"

# 3) App venv (uv venvs have no pip; everything uses `uv pip`)
uv venv --python "$PY_VER" "$VENV_DIR"
PY="$VENV_DIR/bin/python"
"$PY" -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

# 4) Core deps: torch cu124 + app requirements.txt
uv pip install --python "$PY" --index-url "$CU124" \
  torch==2.5.1 torchvision==0.20.1
uv pip install --python "$PY" --index-url "$CU124" \
  --no-build-isolation -r requirements.txt

# 5) vendored custom_rasterizer (paint stage) — must be built for THIS gpu
( cd vendor/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer \
    && TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
       UV_CACHE_DIR=/root/.uv-cache uv pip install --python "$PY" \
       --no-build-isolation -e . )

# 6) Paint-only deps (lean; cv2 already provided by opencv-headless)
uv pip install --python "$PY" --index-url "$MIRROR" \
  "bpy==4.2.23" 2>/dev/null || true   # mirror may lack cp311 bpy -> fallback pypi
uv pip install --python "$PY" bpy==4.2.23
uv pip install --python "$PY" --index-url "$MIRROR" \
  "pytorch-lightning>=2.0,<2.6" \
  basicsr==1.4.2 realesrgan==0.3.0 \
  addict yapf future lmdb fast-simplification --no-deps

# 7) basicsr 1.4.2 <-> torchvision 0.20 compat patch
BS="$VENV_DIR/lib/python3.11/site-packages/basicsr/data/degradations.py"
sed -i "s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/" "$BS"

# 8) Real-ESRGAN upscaler weights -> ckpt/ (paint stage, relative to CWD)
mkdir -p ckpt
if [ ! -f ckpt/RealESRGAN_x4plus.pth ]; then
  wget -q -O ckpt/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi

# 9) Runtime env for per-request texture + auto concurrency
grep -q HY3D_MAX_ACTIVE_JOBS .env 2>/dev/null || \
  echo "HY3D_MAX_ACTIVE_JOBS=auto" >> .env
grep -q HY3D_ENABLE_TEXTURE .env 2>/dev/null || \
  echo "HY3D_ENABLE_TEXTURE=0" >> .env

# 10) Sanity
"$PY" -c "import torch, torchvision, bpy, pymeshlab, realesrgan, fast_simplification; print('deps OK')"
"$PY" scripts/first_run_check.py || true

echo "=== provision complete. Launch with: ./run.sh (run.sh sets LD_LIBRARY_PATH for bpy/embree) ==="