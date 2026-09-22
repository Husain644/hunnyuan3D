#!/usr/bin/env python3
"""Model bootstrap: verify + download the Hunyuan3D checkpoints.

The vendored Hunyuan3D-2.1 ``smart_load_model`` (hy3dshape/utils/utils.py)
loads weights from ``$HY3DGEN_MODELS/<repo>/<subfolder>/`` (config.yaml +
model.fp16.{ckpt,safetensors}). This script manages exactly that cache:

    models/                                   <- $HY3DGEN_MODELS (default models/)
      tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/    config.yaml + model.fp16.ckpt
      tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/   config.yaml + model.fp16.safetensors

Behaviour:
  * ``--check``   report presence/validity only, never download, exit 0/1/2
  * default       download anything missing (with progress), verify, exit 0/1
  * never re-downloads a complete, valid checkpoint (HF cache reuse + local
    file presence check)
  * fails with a clear message if authentication is required (gated repo)

Model details (documented in MODELS.md):
  * tencent/Hunyuan3D-2.1   (2.1 shape)  public, no HF auth, ~7.4 GB (ckpt)
  * tencent/Hunyuan3D-2mv   (2mv shape)  public, no HF auth, ~4.6 GB (safetensors)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Expected files per model. ``weights`` names the exact file smart_load_model
# resolves for the pipeline (2.1 uses fp16 ckpt; 2mv adapter uses safetensors).
MODELS = [
    {
        "id": "2.1",
        "hf_repo": "tencent/Hunyuan3D-2.1",
        "subfolder": "hunyuan3d-dit-v2-1",
        "weights": "model.fp16.ckpt",
        "min_bytes": 7_200_000_000,
        "size_hint": "~7.4 GB",
        "auth": False,
    },
    {
        "id": "2mv",
        "hf_repo": "tencent/Hunyuan3D-2mv",
        "subfolder": "hunyuan3d-dit-v2-mv",
        "weights": "model.fp16.safetensors",
        "min_bytes": 4_900_000_000,
        "size_hint": "~4.6 GB",
        "auth": False,
    },
]


@dataclass
class ModelStatus:
    name: str
    repo: str
    valid: bool
    has_config: bool
    has_weights: bool
    size_bytes: int = 0
    path: Path | None = None


def cache_root() -> Path:
    base = os.environ.get("HY3DGEN_MODELS", "models")
    return Path(base).expanduser().resolve()


def model_dir(m: dict, base: Path) -> Path:
    return base / m["hf_repo"] / m["subfolder"]


def check(m: dict, base: Path) -> ModelStatus:
    d = model_dir(m, base)
    cfg = d / "config.yaml"
    w = d / m["weights"]
    size = w.stat().st_size if w.exists() else 0
    valid = cfg.exists() and size >= m["min_bytes"]
    return ModelStatus(
        name=m["id"],
        repo=m["hf_repo"],
        valid=valid,
        has_config=cfg.exists(),
        has_weights=w.exists(),
        size_bytes=size,
        path=d,
    )


def _fail_auth(exc: Exception, m: dict) -> None:
    try:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:  # older hub versions
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    if isinstance(exc, (GatedRepoError, RepositoryNotFoundError)):
        print(
            f"\n[model:{m['id']}] access denied for {m['hf_repo']}.\n"
            "This model is PUBLIC and should not require auth. If you see a "
            "gated-repo error, log in with:\n"
            "    .venv/bin/huggingface-cli login\n"
            "then re-run:  .venv/bin/python scripts/model_bootstrap.py",
            file=sys.stderr,
        )
        sys.exit(1)
    # otherwise fall through to the generic download-failure message below


def download(m: dict, base: Path, check_only: bool) -> ModelStatus:
    st = check(m, base)
    if check_only:
        return st
    if st.valid:
        print(f"[model:{m['id']}] {m['hf_repo']} already complete "
              f"({st.size_bytes/2**30:.2f} GiB) — skipping")
        return st

    print(f"\n[model:{m['id']}] downloading {m['hf_repo']}/{m['subfolder']} "
          f"({m['size_hint']}) …")
    dest = base / m["hf_repo"]
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=m["hf_repo"],
            allow_patterns=[f"{m['subfolder']}/*"],
            local_dir=dest,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001
        _fail_auth(exc, m)
        print(f"[model:{m['id']}] download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    st = check(m, base)
    if not st.valid:
        print(
            f"[model:{m['id']}] downloaded but verification failed "
            f"(missing config={not st.has_config}, "
            f"weights={not st.has_weights}, size={st.size_bytes}).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[model:{m['id']}] verified: {st.path} "
          f"({st.size_bytes/2**30:.2f} GiB)")
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify only; do not download")
    ap.add_argument("--models", default=None,
                    help="override HY3DGEN_MODELS cache dir")
    args = ap.parse_args()

    base = Path(args.models).expanduser().resolve() if args.models else cache_root()
    print(f"Model cache: {base}")

    results = [download(m, base, check_only=args.check) for m in MODELS]
    if args.check:
        all_ok = all(s.valid for s in results)
        for s in results:
            state = "READY" if s.valid else ("incomplete" if s.has_weights or s.has_config else "missing")
            print(f"  {s.name:<6} {state:<12} {s.repo}")
        return 0 if all_ok else 2
    return 0 if all(s.valid for s in results) else 1


if __name__ == "__main__":
    sys.exit(main())