#!/usr/bin/env python3
"""
first_run_check.py — lightweight validation of a fresh install.

No 3D generation is performed. Checks, in order:
  1. Python/torch import + CUDA availability (does not allocate VRAM).
  2. Required third-party modules import cleanly.
  3. vendor source (hy3dshape) is present.
  4. Model checkpoints exist for at least one supported model (2.1 / 2mv).
  5. App adapters import (app.server builds, app.pipeline resolves).

Exits 0 on success, 1 on hard failure (missing model/source), and prints
warnings for non-fatal issues. Designed to be re-run any time.

Usage:
    .venv/bin/python scripts/first_run_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

warnings: list[str] = []
ok_count = 0


def ok(msg: str) -> None:
    global ok_count
    ok_count += 1
    print(f"  [ok]   {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  [warn] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


print("1/4 Python + CUDA")
try:
    import torch  # noqa: F401

    ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        n = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        ok(f"CUDA device: {n} ({vram:.1f} GB)")
    else:
        warn("torch.cuda.is_available() = False — generation will fail; "
             "check NVIDIA driver/container GPU passthrough")
except ImportError as exc:
    fail(f"torch not importable ({exc}). Install via setup.sh: "
         f"{SERVICE_ROOT}/requirements-gpu.txt")


print("2/4 Required modules")
for mod in (
    "fastapi", "uvicorn", "numpy", "PIL", "trimesh",
    "diffusers", "transformers", "timm", "skimage", "scipy",
    "huggingface_hub", "safetensors",
):
    try:
        __import__(mod)
        ok(f"{mod}")
    except ImportError:
        warn(f"{mod} missing — most will break loading; re-run setup.sh")


print("3/4 Hunyuan source (hy3dshape)")
repo_dir = None
try:
    from app.config import SETTINGS
    from app.pipeline import _repo_dirs  # noqa: F401

    dirs = _repo_dirs()
    repo_dir = str(Path(dirs[0]).resolve()) if dirs else None
except Exception as exc:  # noqa: BLE001
    fail(f"could not resolve vendor repo path: {exc}")
if not repo_dir or not Path(repo_dir).exists():
    fail(f"vendor source not found at {repo_dir}. Clone it or re-run setup.sh.")
shape_ok = (Path(repo_dir) / "hy3dshape").is_dir()
paint_ok = (Path(repo_dir) / "hy3dpaint").is_dir()
ok(f"{repo_dir} (hy3dshape={'present' if shape_ok else 'MISSING'}, "
   f"hy3dpaint={'present' if paint_ok else 'missing (texture off)'})")


print("4/4 Checkpoints")
try:
    from app.models_check import MODELS, all_status, cache_root, model_status

    base = cache_root()
    ok(f"model cache: {base}")
    statuses = [model_status(m) for m in MODELS]
    for st in statuses:
        if st["valid"]:
            ok(f"{st['id']}: weights ready ({st['size_bytes'] / 2**30:.2f} GiB)")
        else:
            warn(f"{st['id']}: weights missing/incomplete ({st['reason']}) — "
                 "run scripts/model_bootstrap.py to download")
    agg = all_status()
    if not agg["ready"]:
        fail("no model checkpoints ready; generation will fail. Run: "
             ".venv/bin/python scripts/model_bootstrap.py")
except Exception as exc:  # noqa: BLE001
    warn(f"model check skipped ({exc})")


print("5/5 App adapters")
try:
    import app.server  # noqa: F401
    import app.pipeline  # noqa: F401
    ok("app.server / app.pipeline import OK")
except Exception as exc:  # noqa: BLE001
    print(f"  [warn] app.server import warning: {exc}", file=sys.stderr)
    warnings.append(str(exc))


print()
if warnings:
    print("\n".join(f"  ! {w}" for w in warnings))
    print("(non-fatal checks had warnings)")
print(f"first_run_check: {ok_count} ok, {len(warnings)} warning(s)")
sys.exit(0)