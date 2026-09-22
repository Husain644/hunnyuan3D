# Session in  Opencode ->>> opencode -s ses_f5b1b30a1ffeQkkEHTab5xXaPb

  █▀▀█ █▀▀█ █▀▀█ █▀▀▄ █▀▀▀ █▀▀█ █▀▀█ █▀▀█
  █  █ █  █ █▀▀▀ █  █ █    █  █ █  █ █▀▀▀
  ▀▀▀▀ █▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀

  Session   Async Python API for Hunyuan3D 2.1 GLB Export on …
  Continue  opencode -s ses_f5b1b30a1ffeQkkEHTab5xXaPb

# How to convert an image to a 3D .glb model (quick start for next time)

## Overview

Image → 3D GLB conversion runs on a **Lightning.ai Tesla T4 CloudSpace** via an
async FastAPI server. Input images and output GLBs are kept **in memory** — no
disk persistence on the remote. Everything below is done from your local Windows
machine.

## Step 1 — Start opencode in this project

```bash
# from E:\AI (or anywhere)
opencode
# or resume this exact session if you need the history:
opencode -s ses_f5b1b30a1ffeQkkEHTab5xXaPb
```

## Step 2 — Make sure the remote server is running

The CloudSpace sleeps when idle, so the server may be down. Restart it:

```bash
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 \
  ssh s_01m2n3gj3pwc5ga8pfj38nkq1r@ssh.lightning.ai \
  "cd ~/hunyuan3d-api && (nohup ./.venv/bin/python -m app.server > ./outputs/server.log 2>&1 &) && sleep 8 && curl -s http://127.0.0.1:8080/health | head -c 120"
```

Expected: `{"status":"ok","gpu_name":"Tesla T4",...}`. If the SSH key is rejected,
just retry — the CloudSpace may have just restarted.

## Step 3 — Upload an image and get the job id

Plain `curl` from your machine against the public URL:

```bash
curl -X POST "https://8080-01m2n3gj3pwc5ga8pfj38nkq1r.cloudspaces.litng.ai/v1/public/generate/upload" \
  -F "file=@E:\AI\image_for_glb\arm.jpg" -F "enable_texture=false"
```

Returns `{"job_id":"...","status":"queued",...}`. Note the `job_id`.

## Step 4 — Poll until finished (~6–7 min for shape-only)

```bash
curl -s "https://8080-01m2n3gj3pwc5ga8pfj38nkq1r.cloudspaces.litng.ai/v1/public/jobs/<job_id>"
```

`"status":"succeeded","progress":1.0` means it's done. `failed` → check `error`.

## Step 5 — Download the GLB to your local output folder

```bash
curl -o "E:\AI\glb-output\<name>.glb" \
  "https://8080-01m2n3gj3pwc5ga8pfj38nkq1r.cloudspaces.litng.ai/v1/public/jobs/<job_id>/result"
```

The GLB is served once and then purged from the remote's RAM.

## One-command version (steps 3–5)

```bash
ssh -o StrictHostKeyChecking=no s_01m2n3gj3pwc5ga8pfj38nkq1r@ssh.lightning.ai \
  "scp 'E:\AI\image_for_glb\xx.jpg'"  # or upload via curl to localhost then pull the file
```

## Common gotchas

- **Browser upload error "Unexpected non-whitespace character after JSON"** =
  the server was down (CloudSpace slept). Restart it (Step 2) and retry.
- Only **1 job at a time** (`HY3D_MAX_ACTIVE_JOBS=1`); extra uploads wait in the queue.
- `enable_texture=false` = shape-only, ~10 GB, fits the 14.6 GB T4. Texture mode
  needs 21 GB and is disabled on this CloudSpace.
- Job GLBs live in RAM only — if the server restarts mid-job, the job is lost
  and you must re-upload.


# Hunyuan3D 2.1 Async API

FastAPI service that takes an input image, runs Hunyuan3D 2.1's image-to-3D
pipeline (shape DiT → PBR texture), exports **GLB**, stores it on disk, and
exposes **async job status** — tuned to run on a 14.6 GB T4.

## Why this fits a T4

Hunyuan3D 2.1's weights are ~10 GB (DiT) + ~21 GB (Paint / texture). A 14.6 GB
T4 cannot hold both, so the pipeline serializes them:

```text
──── preprocess ──── shape (DiT) ──── offload to CPU ──── texture (Paint) ──── GLB
                     ↑ load 10 GB ↑                          ↑ load ~11 GB ↑
```

Each stage is loaded under a GPU budget lock, runs, then is force-moved back to
CPU (`module.to("cpu")` + `torch.cuda.empty_cache()`) before the next loads.
`enable_model_cpu_offload()` (accelerate hooks) is additionally applied when the
upstream bindings offer it, so intra-stage spill is handled by the framework.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/generate` | Submit a job (multipart file, base64 form field, or JSON body). Returns `202` + `{job_id}`. |
| `GET` | `/v1/jobs/{id}` | Job status: `status`, `progress`, `stage`, `error`, `result_url`. |
| `GET` | `/v1/jobs/{id}/result` | Download the GLB (`model/gltf-binary`). |
| `GET` | `/v1/jobs/{id}/thumbnail` | PNG thumbnail render (pyrender if installed). |
| `GET` | `/v1/jobs` | List recent jobs. |
| `POST` | `/v1/jobs/{id}/cancel` | Cancel a queued job. |
| `GET` | `/health` | GPU name, VRAM, active jobs, queue depth. |

Full OpenAPI docs at `/docs`.

## Quick start

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Clone + build Hunyuan3D 2.1 (reference repo layout must sit beside this
#    service so `hy3dshape`/`hy3dpaint` are importable, or set PYTHONPATH).
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && python setup.py install
cd ../..

# 3. Copy env config and run
cp .env.example .env
python -m app.server            # or: bash run.sh
```

Weights download from HuggingFace on first job from the unified repo
`tencent/Hunyuan3D-2.1` (shape DiT + texture paint).

### Submit a job

```bash
# file upload
curl -X POST http://localhost:8080/v1/generate \
  -F "file=@chair.png" -F "enable_texture=true"

# base64
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64>","octree_resolution":256}'
```

### Poll

```bash
curl -s http://localhost:8080/v1/jobs/<job_id>
# {"status":"running","stage":"texture","progress":0.8,...}

curl -sL http://localhost:8080/v1/jobs/<job_id>/result -o out.glb
```

## Memory / throughput knobs (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `HY3D_ENABLE_TEXTURE` | `1` | `0` → shape-only, ~10 GB peak, ~3× faster |
| `HY3D_STEPS` | `30` | `5` if using the Turbo checkpoint |
| `HY3D_OCTREE` | `256` | lower = smaller mesh, less decode VRAM |
| `HY3D_VAE_CHUNKS` | `8000` | lower = less peak VRAM during VAE decode |
| `HY3D_TEX_RES` / `HY3D_TEX_VIEWS` | `512` / `6` | texture quality vs. time |
| `HY3D_MAX_ACTIVE_JOBS` | `1` | keep `1` (shape stage is not reentrant) |
| `HY3D_VRAM_GB_*` | per-stage | VRAM budgets used by the guard rails |

## Notes

- The worker executes the blocking CUDA work in a background thread (`asyncio.to_thread`);
  HTTP requests stay responsive during a run.
- Job metadata + GLB are persisted under `outputs/` (configurable) and survive restarts.
- v2.0 (`hy3dgen` package) is auto-detected as a fallback if the 2.1 bindings
  aren't installed; the same API and offloading strategy apply.

## Google Colab (T4)

One-click Colab setup defaults to shape-only (~10 GB peak, fits 14.6 GB T4).
Texture is off by default (paint model needs 21 GB, doesn't fit T4).

```python
# Run inside a Colab cell (OR clone + run the scripts)
!git clone https://github.com/you/hunyuan3d-api /content/hunyuan3d-api
!bash /content/hunyuan3d-api/scripts/colab_setup.sh
!bash /content/hunyuan3d-api/scripts/colab_run.sh &
!sleep 3 && python /content/hunyuan3d-api/scripts/colab_test.py
```

The test script spins up the server (if down), hits `/health`, posts a test
image, polls the job to completion, and validates the GLB with trimesh.

## Docker

```bash
docker build -t hunyuan3d-api .
docker run --gpus '"device=0"' -p 8080:8080 \
  -v "$(pwd)/outputs:/data" hunyuan3d-api
```

The image clones Hunyuan3D-2.1 and compiles its custom rasterizer during build.