#!/usr/bin/env bash
set -euo pipefail

# Bash-injected HY3D_MODEL_REPO/switches are respected by app/config.py.
echo "== Hunyuan3D 2.1 API =="
echo "  texture:   ${HY3D_ENABLE_TEXTURE:-on}"
echo "  model:     ${HY3D_SHAPE_MODEL:-tencent/Hunyuan3D-Shape-v2-1}"
echo "  output:    ${HY3D_OUTPUT_DIR:-/data/glb}"

python -m uvicorn app.server:app \
    --host "${HY3D_HOST:-0.0.0.0}" \
    --port "${HY3D_PORT:-8080}" \
    --log-level "${HY3D_LOG_LEVEL:-info}"