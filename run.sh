#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh — start the Hunyuan3D 2.1 / 2mv API server.
#
#   ./run.sh
#
# Loads `.env` (created by setup.sh), shows a summary of the GPU/models it is
# about to serve, then execs uvicorn against the venv interpreter (so fresh
# clones don't need anything installed globally).
# ---------------------------------------------------------------------------
set -euo pipefail
SRV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRV_ROOT"

# Resolve venv python (POSIX first; MINGW/Git Bash Windows fallback).
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  echo "!! .venv not found — run ./setup.sh first" >&2
  exit 1
fi

# Load .env into the environment for this process (and children).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PORT="${HY3D_PORT:-8080}"

echo "== Hunyuan3D 2.1 / 2mv API =="
echo "  texture:   ${HY3D_ENABLE_TEXTURE:-off}"
echo "  rembg:     ${HY3D_ENABLE_REMBG:-off}"
echo "  host:port: ${HY3D_HOST:-0.0.0.0}:${PORT}"
echo "  models:    ${HY3DGEN_MODELS:-models}"

# Show which checkpoints would be served (no generation).
"$PY" - <<'PYEOF' || true
import os, sys
sys.path.insert(0, os.getcwd())
try:
    from app.models_check import all_status
    agg = all_status()
    ready = agg["ready"] or []
    print(f"  ready:     {', '.join(ready) if ready else 'NONE (run scripts/model_bootstrap.py)'}")
    print(f"  missing:   {', '.join(agg['missing']) if agg['missing'] else 'none'}")
    print(f"  cache:     {agg['cache_dir']}")
except Exception as exc:
    print(f"  model status unavailable ({exc})")
PYEOF

exec "$PY" -m uvicorn app.server:app \
    --host "${HY3D_HOST:-0.0.0.0}" \
    --port "$PORT" \
    --log-level "${HY3D_LOG_LEVEL:-info}"