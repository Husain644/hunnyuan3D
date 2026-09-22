# Model Registry (Hunyuan3D 2.1 / 2mv)

This service serves two checkpoints through a single API (`model=2.1` and
`model=2mv`). Both are the official Tencent-Hunyuan releases, hosted on the
HuggingFace Hub, **public** (no auth required).

| id  | HF repo                  | subfolder          | weight file              | size    | purpose        |
|-----|--------------------------|--------------------|--------------------------|---------|----------------|
| 2.1 | `tencent/Hunyuan3D-2.1`  | `hunyuan3d-dit-v2-1`| `model.fp16.ckpt`        | ~7.4 GB | single-view->3D|
| 2mv | `tencent/Hunyuan3D-2mv`  | `hunyuan3d-dit-v2-mv`| `model.fp16.safetensors` | ~4.6 GB | multi-view->3D |

## Cache layout

The vendored loader (`hy3dshape/utils/utils.py` → `smart_load_model`) reads
weights from `$HY3DGEN_MODELS/<repo>/<subfolder>/`. `setup.sh` sets
`HY3DGEN_MODELS` to `<repo>/models` and downloads both models there:

```
models/
  tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/
    config.yaml
    model.fp16.ckpt
  tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/
    config.yaml
    model.fp16.safetensors
```

This is the canonical layout expected by the code — do not move files to a
flat `models/hunyuan3d-2.1/` directory. Weights are **never committed** to
Git; the whole `models/` dir is gitignored.

## Downloading / verifying

```bash
# one-time (setup.sh does this automatically):
.venv/bin/python scripts/model_bootstrap.py

# check-only, never downloads:
.venv/bin/python scripts/model_bootstrap.py --check
```

Re-running is safe: the script verifies file size against a threshold and
never re-downloads a complete, valid checkpoint. HuggingFace reuses its own
`~/.cache/huggingface` so even interrupted downloads resume.

## What loads when

* `/health/models` — live status of each checkpoint.
* Log line at startup: `models ready: 2.1, 2mv` (or a warning when missing).
* If a model is missing at generation time the job fails with
  "weights missing/incomplete — run scripts/model_bootstrap.py".

## Disk / VRAM notes

* Both weights together need ~12 GiB disk (2.1 ~7.4 + 2mv ~4.6).
* Only the selected model is loaded into VRAM (never both at once).
* 2.1 shape peak VRAM ~8.1 GiB on a 14.6 GB T4; recommend ≥ 12 GB GPU.
* Texture stage (not used by the public endpoint by default) separately
  requires ≥ 21 GB VRAM plus a compiled rasterizer — see `deploy.md`.