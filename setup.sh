#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — one-command bootstrap for the Hunyuan3D-2.1 / 2mv Image→3D API.
#
#   git clone <repo>
#   cd <repo>
#   ./setup.sh                 [--start] [--with-texture] [--force] [--skip-models]
#
# Steps (progress printed as [n/8]):
#   [1/8] GPU detection   — NVIDIA GPU, VRAM, driver, CUDA compat (aborts if none)
#   [2/8] Python          — validate version (3.10–3.12)
#   [3/8] venv            — create .venv (idempotent)
#   [4/8] dependencies    — install requirements-gpu.txt (CUDA torch pinned)
#   [5/8] Hunyuan source  — clone Tencent-Hunyuan/Hunyuan3D-2.1 (hy3dshape/...)
#   [6/8] models          — verify/download 2.1 + 2mv checkpoints into models/
#   [7/8] application     — .env, directories, first-run validation
#   [8/8] start           — optional --start (else prints run.sh)
#
# Safe to re-run: existing .venv/.env/models/outputs are preserved; missing
# pieces are repaired. GPU drivers are NEVER auto-installed.
# ---------------------------------------------------------------------------
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SERVICE_ROOT"

# ---------------------------------------------------------------- flags ----
START=0; TEXTURE=0; FORCE=0; SKIP_MODELS=0; AUTO_YES=0
for a in "$@"; do
  case "$a" in
    --start)      START=1 ;;
    --with-texture) TEXTURE=1 ;;
    --force)      FORCE=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --yes|-y)     AUTO_YES=1 ;;
    -h|--help)
      cat <<HELP
usage: ./setup.sh [options]

  --start           start the server when setup finishes
  --with-texture    also build the texture rasterizer (needs >= 21 GB GPU)
  --skip-models     do not download model checkpoints
  --force           continue without an NVIDIA GPU (generation will fail)
  --yes             don't pause before starting

Reproducible one-command setup: GPU gate -> .venv -> deps -> model source
-> checkpoints -> .env -> validation. Idempotent; safe to re-run.
HELP
      exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

info()  { printf "\n\033[1;36m[%s] %s\033[0m\n" "${STEP:- }" "$*"; }
warn()  { printf "\n\033[1;33m** %s\033[0m\n" "$*"; }
step()  { STEP="$1"; info "$2"; }
ok()    { printf "    ok\n"; }
pause() { [ "$AUTO_YES" = 1 ] || { printf "\nPress Enter to continue… "; read -r _; } }
die()   { printf "\n\033[1;31m!! %s\033[0m\n" "$*" >&2; exit 1; }
have()  { command -v "$1" >/dev/null 2>&1; }

# ================================================================== header
cat <<'EOF'
========================================
Image → 3D Setup (Hunyuan3D 2.1 / 2mv)
========================================
EOF

# ================================================================== 1. GPU
step 1 "Detecting GPU"
GPU_NAME=""; VRAM_GB=0; DRIVER=""; CUDA_COMPAT=""; CC=""; GPU_OK=0
if have nvidia-smi; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  VRAM_GB="$(awk -v m="${VRAM_MB:-0}" 'BEGIN{printf "%.1f", m/1024}')" 2>/dev/null || VRAM_GB=0
  CUDA_COMPAT="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \(.*\)$/\1/p' | head -1 || true)"
  [ -n "${GPU_NAME:-}" ] && GPU_OK=1
fi

cat <<EOF
GPU:    ${GPU_NAME:-<none detected>}
VRAM:   ${VRAM_GB} GB
Driver: ${DRIVER:-n/a}
CUDA:   ${CUDA_COMPAT:-n/a} (driver-supported; app uses packaged CUDA libs)
CC:     ${CC:-n/a}
EOF
{ python3 --version 2>/dev/null || python --version 2>/dev/null; } | sed 's/^/Python: /' || true

if [ "$GPU_OK" = 0 ]; then
  if [ "$FORCE" = 1 ]; then
    warn "No NVIDIA GPU detected (--force). The app will install but generation will fail."
  else
    die "No compatible NVIDIA GPU found.
   The Hunyuan3D shape pipelines require a CUDA-capable NVIDIA GPU with
   >= 12 GB VRAM (verified: T4 16GB, RTX 2080 Ti 22GB). Setup refuses to
   continue because the model would only fail later during loading.
   - Install NVIDIA drivers for your GPU, then re-run ./setup.sh
   - Do NOT run with --force unless you know what you are doing."
  fi
fi

MIN_VRAM=12
if [ "$FORCE" = 0 ] && have nvidia-smi && [ "$(printf '%s' "$VRAM_GB" | awk '{print int($1)}')" -lt "$MIN_VRAM" ]; then
  die "GPU has ${VRAM_GB} GB VRAM but the shape model needs >= ${MIN_VRAM} GB.
   Either use a larger GPU or re-run with --force to install anyway."
fi
ok

# ================================================================== 2. Python
step 2 "Validating Python"
PY_BIN="${PYTHON:-}"
if [ -z "$PY_BIN" ]; then
  for c in python3 python; do
    if have "$c"; then PY_BIN="$c"; break; fi
  done
fi
if [ -z "$PY_BIN" ]; then
  die "No Python found (tried: python3, python). Install Python 3.10–3.12 and re-run."
fi
PY_VER="$("$PY_BIN" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" || {
  die "'$PY_BIN' does not launch — install Python 3.10–3.12 and re-run."
}
PY_MAJOR="${PY_VER%%.*}"; PY_MINOR="${PY_VER##*.}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  die "Python $PY_VER detected — the pinned torch 2.5.1 build needs Python 3.10–3.12."
fi
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -gt 12 ]; then
  die "Python $PY_VER detected — torch 2.5.1 (pinned in requirements-gpu.txt) has no "
     "wheels for Python $PY_VER. Install Python 3.10/3.11/3.12 and re-run."
fi
echo "  python $PY_VER ($(command -v "$PY_BIN" || true))"
ok

# ================================================================== 3. venv
step 3 "Creating virtual environment (.venv)"
if [ ! -x .venv/bin/python ]; then
  "$PY_BIN" -m venv .venv
else
  echo "  .venv already exists — reusing"
fi
VENV_PY=.venv/bin/python
VENV_PIP=.venv/bin/pip
if [ "$(uname -s)" = MINGW* ] || [ -x .venv/Scripts/python.exe ]; then
  VENV_PY=.venv/Scripts/python.exe; VENV_PIP=.venv/Scripts/pip.exe
fi
"$VENV_PY" -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null \
  || die "venv Python too old; delete .venv and re-run with a newer interpreter."
"$VENV_PIP" install -q --upgrade pip >/dev/null 2>&1 || warn "pip upgrade skipped"
ok

# ================================================================== 4. deps
step 4 "Installing dependencies (CUDA torch + pinned deps)"
REQ="${REQ_FILE:-requirements-gpu.txt}"
[ -f "$REQ" ] || REQ="requirements.txt"
echo "  using $REQ"
"$VENV_PIP" install -q -r "$REQ" >/dev/null 2>&1 || {
  warn "pip install reported an error — showing detail:"
  "$VENV_PIP" install -r "$REQ" || die "dependency install failed"
}
ok

# ================================================================== 5. source
step 5 "Preparing Hunyuan3D-2.1 source (hy3dshape/hy3dpaint bindings)"
REPO_DIR="${HY3D_REPO_DIR:-$SERVICE_ROOT/vendor/Hunyuan3D-2.1}"
export HY3D_REPO_DIR="$REPO_DIR"
if [ -d "$REPO_DIR/hy3dshape" ] && [ -d "$REPO_DIR/hy3dpaint" ]; then
  echo "  Hunyuan3D source already at $REPO_DIR"
  ( git -C "$REPO_DIR" pull --ff-only 2>/dev/null || true )
else
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO_DIR"
fi
if [ "$TEXTURE" = 1 ]; then
  info "Building custom rasterizer (texture)"
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$( [ -n "$CC" ] && echo "$CC" || echo 7.5 )}"
  export TORCH_CUDA_ARCH_LIST
  RAST="$REPO_DIR/hy3dpaint/custom_rasterizer"
  if [ -d "$RAST" ]; then
    ( cd "$RAST" && "$VENV_PY" setup.py build_ext --inplace ) >/dev/null 2>&1 \
      || warn "rasterizer build failed (texture will be unavailable; shape still works)"
  else
    warn "custom_rasterizer not found — shape-only mode"
  fi
fi
ok

# ================================================================== 6. models
step 6 "Verifying model checkpoints (auto-download if missing)"
if [ "$SKIP_MODELS" = 1 ]; then
  warn "Skipping model download (--skip-models). Model files must exist or generation will fail."
else
  MODELS_DIR="${HY3DGEN_MODELS:-$SERVICE_ROOT/models}"
  export HY3DGEN_MODELS="$MODELS_DIR"
  "$VENV_PY" scripts/model_bootstrap.py --models "$MODELS_DIR"
fi
ok

# ================================================================== 7. app
step 7 "Preparing application (.env, directories, validation)"
# .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  created .env from .env.example"
fi
export HY3D_REPO_DIR="${HY3D_REPO_DIR:-$SERVICE_ROOT/vendor/Hunyuan3D-2.1}"
export HY3DGEN_MODELS="${HY3DGEN_MODELS:-$SERVICE_ROOT/models}"
export HY3D_ENABLE_TEXTURE="${HY3D_ENABLE_TEXTURE:-0}"
export HY3D_ENABLE_REMBG="${HY3D_ENABLE_REMBG:-0}"

# Write machine-specific paths into .env so run.sh (which sources .env) works
# from any shell. Idempotent: existing values are never overwritten.
_inject_env() {  # _inject_env KEY VALUE  (append; never overwrite)
  local k="$1" v="$2"
  if ! grep -q "^${k}=" .env; then
    printf '\n%s=%s\n' "$k" "$v" >> .env
    echo "  added $k to .env"
  fi
}
_set_env() {  # _set_env KEY VALUE  (replace existing value or append)
  local k="$1" v="$2"
  if grep -q "^${k}=" .env; then
    sed -i "s|^${k}=.*|${k}=${v}|" .env
    echo "  updated $k in .env"
  else
    _inject_env "$k" "$v"
  fi
}
_inject_env "HY3D_REPO_DIR" "$HY3D_REPO_DIR"
_inject_env "HY3DGEN_MODELS" "$HY3DGEN_MODELS"
_inject_env "HY3D_ENABLE_TEXTURE" "$HY3D_ENABLE_TEXTURE"
_inject_env "HY3D_ENABLE_REMBG" "$HY3D_ENABLE_REMBG"

# required dirs
mkdir -p outputs/glb outputs/jobs logs models
echo "  directories ready (outputs/, logs/, models/)"

# If the configured HTTP port is already taken (e.g. Jupyter sits on 8080 on
# Vast.ai instances) pick the next free port and record it in .env (updates
# the value in .env, whether default or user-set — the port must be free to
# run.sh successfully).
_port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
CFG_PORT="$(grep -E '^HY3D_PORT=' .env | tail -1 | cut -d= -f2 || echo 8080)"
CFG_PORT="${CFG_PORT:-8080}"
if ! _port_free "$CFG_PORT"; then
  NEW_PORT="$CFG_PORT"
  while [ "$NEW_PORT" -lt 60000 ]; do
    NEW_PORT=$((NEW_PORT + 1))
    if _port_free "$NEW_PORT"; then break; fi
  done
  _set_env "HY3D_PORT" "$NEW_PORT"
  echo "  port $CFG_PORT busy — using $NEW_PORT instead"
fi

# validate imports / adapters / CUDA / checkpoints (lightweight, no generation)
info "First-run validation (no 3D generation)"
"$VENV_PY" scripts/first_run_check.py || {
  warn "validation reported warnings — see above. Fixing what we can:"
  warn "check: .venv/bin/python scripts/first_run_check.py"
}
ok

# ================================================================== 8. start
step 8 "Finishing setup"
echo
cat <<EOF
========================================
SETUP COMPLETE
========================================
Run the service:
  ./run.sh          (or: ./setup.sh --start)

API docs:  http://127.0.0.1:${HY3D_PORT:-8080}/docs
Health:    http://127.0.0.1:${HY3D_PORT:-8080}/health
Public:    http://127.0.0.1:${HY3D_PORT:-8080}/
========================================
EOF
if [ "$START" = 1 ]; then
  pause
  exec ./run.sh
fi
exit 0