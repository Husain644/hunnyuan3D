#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# colab_setup.sh — One-shot Google Colab / T4 provisioning.
#
# Mirrors https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1's quick-start but
# adds: auto-Colab detection, safe PyTorch handling (never touch a working
# CUDA torch), curated deps list, VAE/synth model pre-download, T4 tuning
# (FP16, 256 octree, TORCH_CUDA_ARCH_LIST=7.5).
#
# Behaviour:
#   1. Detect Colab (fail-fast outside Colab unless FORCE=1).
#   2. Install PyTorch ONLY if missing or no CUDA (never clobber existing).
#   3. Clone the official Hunyuan3D-2.1 repo into $SERVICE_ROOT/Hunyuan3D-2.1.
#   4. Install requirements.colab.txt (full, skip torch, skip bpy/open3d/etc).
#   5. Patch a known bpy shim so mesh_utils.py is importable in Colab.
#   6. Pre-load the shape checkpoint (DiT+VAE subfolder) with huggingface_hub.
#   7. Write .colab.env consumed by colab_run.sh and server startup.
#
# Usage (Colab cell):
#   !bash /content/hunyuan3d-api/scripts/colab_setup.sh
#
# Non-Colab (development/debug):
#   FORCE=1 bash scripts/colab_setup.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
SERVICE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="${SERVICE_ROOT}/Hunyuan3D-2.1"
ENV_FILE="${SERVICE_ROOT}/.colab.env"
REQ_FILE="${SERVICE_ROOT}/requirements.colab.txt"
PYTHON="${PYTHON:-python3}"

# ── Flags ─────────────────────────────────────────────────────────────────
FORCE="${FORCE:-0}"
WITH_TEXTURE="${WITH_TEXTURE:-0}"        # set 1 to install texture deps + rasterizer
NO_PRELOAD="${NO_PRELOAD:-0}"           # skip HF model pre-download
DEBUG="${DEBUG:-0}"

# ── Helpers ───────────────────────────────────────────────────────────────
info()  { printf "\n\033[1;34m== %s\033[0m\n" "$*"; }
warn()  { printf "\n\033[1;33m** %s\033[0m\n" "$*"; }
error() { printf "\n\033[1;31m!! %s\033[0m\n" "$*" >&2; exit 1; }

# ── 1. Colab gate ────────────────────────────────────────────────────────
_is_colab() {
    [ "${COLAB_GPU:-}" = "true" ]  && return 0
    [ -n "${COLAB_TPU_ADDR:-}" ]   && return 0
    [ -d "/content" ]              && return 0
    return 1
}

if ! _is_colab && [ "$FORCE" != "1" ]; then
    error "Not a Colab runtime. Set FORCE=1 to run on a bare machine."
fi

info "Colab GPU: ${COLAB_GPU:-<not set>}"
nvidia-smi || warn "nvidia-smi not available; CUDA state unknown"

# ── 2. PyTorch (never clobber) ──────────────────────────────────────────
_ensure_torch() {
    local has_cuda=0
    $PYTHON - <<'PY' 2>/dev/null && has_cuda=1 || has_cuda=0
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
    if [ "$has_cuda" -eq 1 ]; then
        info "PyTorch with CUDA already present; skipping install"
        $PYTHON -c "import torch; print(f'  torch {torch.__version__}  CUDA {torch.version.cuda}')"
        return 0
    fi

    warn "PyTorch missing or no CUDA — installing torch 2.5.1 (CUDA 12.4)"
    $PYTHON -m pip install -q --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124
}
_ensure_torch

# ── 3. Clone Hunyuan3D-2.1 ──────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    info "Hunyuan3D-2.1 repo already at $REPO_DIR; pulling latest"
    git -C "$REPO_DIR" pull --ff-only || true
else
    info "Cloning Tencent-Hunyuan/Hunyuan3D-2.1 (depth 1)"
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO_DIR"
fi

# ── 4. Install deps ──────────────────────────────────────────────────────
info "Installing curated Colab deps from requirements.colab.txt"
$PYTHON -m pip install -q --upgrade pip
$PYTHON -m pip install -q -r "$REQ_FILE"

if [ "$WITH_TEXTURE" = "1" ]; then
    info "Building hy3dpaint/custom_rasterizer (WITH_TEXTURE=1)"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
    RAST_DIR="$REPO_DIR/hy3dpaint/custom_rasterizer"
    if [ -d "$RAST_DIR" ]; then
        $PYTHON "$RAST_DIR/setup.py" build_ext --inplace 2>&1 || \
            warn "Rasterizer build failed (non-fatal without texture)"
    else
        warn "custom_rasterizer dir not found at $RAST_DIR"
    fi
fi

# ── 5. Patch mesh_utils.py bpy shim ─────────────────────────────────────
info "Applying bpy shim to $REPO_DIR/hy3dshape/mesh_utils.py"
MESH_UTILS="$REPO_DIR/hy3dshape/mesh_utils.py"
if [ -f "$MESH_UTILS" ]; then
    $PYTHON - <<'PATCH'
import re, pathlib, sys
p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path("/content/Hunyuan3D-2.1/hy3dshape/mesh_utils.py")
if not p.exists():
    sys.exit(0)
t = p.read_text()
SHIM = '''try:\n    import bpy\nexcept ImportError:\n    class _BpyShim:\n        def __getattr__(self, n): raise NotImplementedError("bpy not available")\n    bpy = _BpyShim()\n'''
if "class _BpyShim" not in t:
    t = re.sub(r"^import bpy\s*$", SHIM, t, count=1, flags=re.MULTILINE)
    p.write_text(t)
    print("  patched")
else:
    print("  already patched")
PATCH
else
    warn "$MESH_UTILS not found; skipping bpy shim"
fi

# ── 6. Pre-download checkpoint (safe to skip) ────────────────────────────
if [ "$NO_PRELOAD" != "1" ]; then
    info "Pre-downloading shape checkpoint (DiT + VAE) ..."
    $PYTHON - <<'DL'
import os
os.environ.setdefault("HF_HOME", os.path.join(
    os.environ.get("SERVICE_ROOT", "/content/hunyuan3d-api"), ".hf_cache"))
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        "tencent/Hunyuan3D-2.1",
        allow_patterns=["hunyuan3d-dit-v2-1/*", "hunyuan3d-vae-v2-1/*"],
        ignore_patterns=["*.md", "*.txt", "*.png"],
    )
    print(f"  cached at {p}")
except Exception as e:
    print(f"  pre-download failed (will lazy-load at runtime): {e}")
DL
else
    info "Skipping checkpoint pre-download (NO_PRELOAD=1)"
fi

# ── 7. Write .colab.env ──────────────────────────────────────────────────
export HF_HOME="${SERVICE_ROOT}/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export HY3D_ENABLE_TEXTURE="${WITH_TEXTURE}"
export HY3D_ENABLE_REMBG=0
export HY3D_MAX_ACTIVE_JOBS=1
export HY3D_STEPS=30
export HY3D_GUIDANCE=5.0
export HY3D_OCTREE=256
export HY3D_VAE_CHUNKS=8000
export HY3D_SHAPE_MODEL=tencent/Hunyuan3D-2.1
export HY3D_SHAPE_SUBFOLDER=hunyuan3d-dit-v2-1
export HY3D_REPO_DIR="$REPO_DIR"
export HY3D_PORT=8080
export HY3D_LOG_LEVEL=info

cat > "$ENV_FILE" <<EOF
# Auto-generated by colab_setup.sh — source in colab_run.sh
export HF_HOME="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export HY3D_ENABLE_TEXTURE="${HY3D_ENABLE_TEXTURE}"
export HY3D_ENABLE_REMBG="${HY3D_ENABLE_REMBG}"
export HY3D_MAX_ACTIVE_JOBS="${HY3D_MAX_ACTIVE_JOBS}"
export HY3D_STEPS="${HY3D_STEPS}"
export HY3D_GUIDANCE="${HY3D_GUIDANCE}"
export HY3D_OCTREE="${HY3D_OCTREE}"
export HY3D_VAE_CHUNKS="${HY3D_VAE_CHUNKS}"
export HY3D_SHAPE_MODEL="${HY3D_SHAPE_MODEL}"
export HY3D_SHAPE_SUBFOLDER="${HY3D_SHAPE_SUBFOLDER}"
export HY3D_REPO_DIR="${HY3D_REPO_DIR}"
export HY3D_PORT="${HY3D_PORT}"
export HY3D_LOG_LEVEL="${HY3D_LOG_LEVEL}"
EOF

info "Written $ENV_FILE"

# ── 8. Quick sanity check ────────────────────────────────────────────────
info "Running smoke imports"
$PYTHON - <<'SMOKE' || warn "Smoke import failed (review errors above)"
import torch, transformers, diffusers, accelerate, trimesh, fastapi
print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"  transformers={transformers.__version__}  diffusers={diffusers.__version__}")
print(f"  trimesh={trimesh.__version__}  fastapi={fastapi.__version__}")
SMOKE

info "Setup complete.  Run: bash scripts/colab_run.sh"
