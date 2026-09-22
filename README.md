# Hunyuan3D 2.1 / 2mv — Image→3D GLB API

FastAPI service that turns an image (or four multi-view images) into a
**3D `.glb` model** using Hunyuan3D 2.1 (single-view) and Hunyuan3D 2mv
(multi-view). Async job queue, CPU-offload GPU management, tunable for a
14.6 GB T4, verified end-to-end on T4 16 GB and RTX 2080 Ti.

```
image(s) → shape DiT → mesh decode → GLB (with name/description metadata)
```

## One-command setup

Requires a machine with an **NVIDIA GPU (≥ 12 GB VRAM, CUDA capable)** and
Python **3.10–3.12**. No NVIDIA drivers are ever auto-installed.

```bash
git clone git@github.com:Husain644/hunnyuan3D.git hunyuan3d-api
cd hunyuan3d-api

./setup.sh            # detect GPU → venv → deps → clone model source → download weights → validate
# optional:
./setup.sh --start    # also start the server when setup finishes
```

`setup.sh` is **idempotent** — safe to re-run; it never re-downloads valid
checkpoints and never destroys your `.env`, `outputs/`, or existing venv.

### What setup.sh does

1. Detects your GPU (name, VRAM, driver, CUDA compat) and **aborts** with a
   clear message if no compatible card is found (`--force` bypasses, at your
   own risk).
2. Creates `.venv/` and installs pinned deps (`requirements-gpu.txt` →
   CUDA torch).
3. Clones `Tencent-Hunyuan/Hunyuan3D-2.1` into `vendor/` (the reference
   `hy3dshape`/`hy3dpaint` binding source).
4. Verifies / downloads the **2.1** and **2mv** checkpoints into `models/`
   (see [MODELS.md](MODELS.md) for the exact layout). Public weights, no HF
   auth.
5. Generates `.env` from `.env.example` (never overwrites an existing one),
   creates `outputs/`, and runs a lightweight first-run validation.

### Run

```bash
./run.sh                                      # or: ./setup.sh --start
# → http://HOST:8080/   (public page)   /docs  (OpenAPI)
```

### Test

```bash
./test_generation.sh                          # Test A: 2.1, Test B: 2mv
```

Generates synthetic inputs, submits them, polls to completion, and validates
the resulting GLBs (watertight, non-empty) with trimesh.

## Endpoints

Public (RAM-only, serve-once) API — what the web page uses:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/public/generate/upload` | Multipart submit. `model=2.1` (file) or `model=2mv` (view_front/back/left/right). `name`/`description` → embedded GLB metadata. |
| `GET` | `/v1/public/jobs/{id}` | Job status: `status`, `progress`, `stage`, `error`. |
| `GET` | `/v1/public/jobs/{id}/result` | Download GLB (served once, then purged from RAM). |
| `GET` | `/health` | GPU info, active jobs, queue depth. |
| `GET` | `/health/models` | Checkpoint availability per model. |

Disk-persisted API (survives restarts):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/generate` | JSON base64 or multipart submit (2.1). Returns `202` + `{job_id}`. |
| `POST` | `/v1/generate/upload` | Multipart submit (2.1). |
| `GET` | `/v1/jobs/{id}` | Job status. |
| `GET` | `/v1/jobs/{id}/result` | Download GLB. |
| `GET` | `/v1/jobs/{id}/thumbnail` | PNG preview (needs pyrender). |
| `GET` | `/v1/jobs` | List recent jobs. |
| `POST` | `/v1/jobs/{id}/cancel` | Cancel queued job. |

Full OpenAPI docs at `/docs` once running.

### Example

```bash
# 2.1 single view
curl -X POST http://localhost:8080/v1/public/generate/upload \
  -F "model=2.1" -F "name=my chair" -F "file=@chair.png"
# → {"job_id":"...","status":"queued",...}

# 2mv multi view
curl -X POST http://localhost:8080/v1/public/generate/upload \
  -F "model=2mv" -F "view_front=@f.png" -F "view_back=@b.png" \
  -F "view_left=@l.png" -F "view_right=@r.png" -F "name=character"

# poll, then download
curl -s http://localhost:8080/v1/public/jobs/<job_id>
curl -o out.glb http://localhost:8080/v1/public/jobs/<job_id>/result
```

## Configuration (`.env`)

Created from `.env.example` on first setup. Key variables:

| Variable | Default | Effect |
|---|---|---|
| `HY3D_ENABLE_TEXTURE` / `HY3D_ENABLE_REMBG` | `0` | opt-in heavy stages (texture needs ≥ 21 GB + rasterizer build) |
| `HY3D_STEPS` / `HY3D_GUIDANCE` | `30` / `5.0` | shape diffusion settings |
| `HY3D_OCTREE` | `256` | mesh resolution (128–384) |
| `HY3D_VAE_CHUNKS` | `8000` | lower = less peak VRAM in VAE decode |
| `HY3D_MAX_ACTIVE_JOBS` | `1` | keep 1 (shape stage is not reentrant) |
| `HY3D_MAX_UPLOAD_MB` | `20` | public upload size cap (413 over limit) |
| `HY3D_CORS_ORIGINS` | `*` | comma-separated allowed origins |
| `HY3DGEN_MODELS` | `models` | checkpoint cache dir (`setup.sh` sets absolute path) |
| `HY3D_REPO_DIR` | `vendor/Hunyuan3D-2.1` | model source dir (`setup.sh` sets absolute path) |

## Documentation

* **[MODELS.md](MODELS.md)** — both checkpoints, exact cache layout, download/verify.
* **[deploy.md](deploy.md)** — two verified deployments (Lightning T4, Vast.ai 2080 Ti) + both API modes.
* `implement.txt` — the reproducibility spec this setup implements.

## Google Colab (T4)

The Colab path predates the one-command setup and is kept for reference:
`scripts/colab_setup.sh` + `scripts/colab_run.sh` + `scripts/colab_test.py`.

## Docker (optional)

```bash
docker build -t hunyuan3d-api .
docker run --gpus '"device=0"' -p 8080:8080 \
  -v "$(pwd)/outputs:/data" hunyuan3d-api
```

## Notes

- Worker executes blocking CUDA work in a background thread; HTTP stays responsive.
- Jobs never run concurrently on the GPU (`_GPU_LOCK` + `HY3D_MAX_ACTIVE_JOBS=1`).
- Only the selected model is loaded into VRAM (never both at once).
- `models/`, `vendor/`, `.venv/`, `outputs/`, `.env` are gitignored and
  generated on the machine — nothing machine-specific is committed.