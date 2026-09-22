"""app/models_check.py — read-only model checkpoint status.

Checks whether the 2.1 / 2mv weight files exist and are plausibly complete in
the local cache (same layout that smart_load_model expects: $HY3DGEN_MODELS /
<repo>/<subfolder>/). Never downloads.

Used by:
  * /health/models endpoint (app/server.py)
  * scripts/first_run_check.py
  * startup banner in run.sh
"""
from __future__ import annotations

import os
from pathlib import Path

MODELS = [
    {
        "id": "2.1",
        "hf_repo": "tencent/Hunyuan3D-2.1",
        "subfolder": "hunyuan3d-dit-v2-1",
        "weights": "model.fp16.ckpt",
        "min_bytes": 7_200_000_000,
    },
    {
        "id": "2mv",
        "hf_repo": "tencent/Hunyuan3D-2mv",
        "subfolder": "hunyuan3d-dit-v2-mv",
        "weights": "model.fp16.safetensors",
        "min_bytes": 4_900_000_000,
    },
]

DEFAULT_CACHE = "models"


def cache_root() -> Path:
    base = os.environ.get("HY3DGEN_MODELS", DEFAULT_CACHE)
    p = Path(base).expanduser()
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def model_dir(m: dict) -> Path:
    return cache_root() / m["hf_repo"] / m["subfolder"]


def model_status(m: dict) -> dict:
    """Return {'id','valid','reason','size_bytes','path'} for one model."""
    d = model_dir(m)
    cfg = d / "config.yaml"
    w = d / m["weights"]
    size = w.stat().st_size if w.exists() else 0
    reasons: list[str] = []
    if not cfg.exists():
        reasons.append("missing config.yaml")
    if not w.exists():
        reasons.append(f"missing {m['weights']}")
    elif size < m["min_bytes"]:
        reasons.append(f"{m['weights']} too small ({size / 2**30:.2f} GiB)")
    return {
        "id": m["id"],
        "valid": not reasons and cfg.exists(),
        "reason": "; ".join(reasons) if reasons else None,
        "size_bytes": size,
        "path": str(d),
    }


def all_status() -> dict:
    """Aggregate status for every supported model."""
    statuses = [model_status(m) for m in MODELS]
    return {
        "models": {s["id"]: s for s in statuses},
        "ready": [s["id"] for s in statuses if s["valid"]],
        "missing": [s["id"] for s in statuses if not s["valid"]],
        "cache_dir": str(cache_root()),
    }


def ready_ids() -> list[str]:
    try:
        return all_status()["ready"]
    except OSError:
        return []