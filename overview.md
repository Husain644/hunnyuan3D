# Hunyuan3D 2.1 — Project Overview

Image-to-3D model generation service. You give it an image (PNG/JPG), it
produces a **GLB 3D mesh** via Tencent's Hunyuan3D 2.1, tuned to run on a
**14.6 GB Tesla T4** (Lightning.ai CloudSpace). The current deployment uses
**shape-only mode** (no texture). All remote storage is in-memory (RAM) — no
disk persistence of inputs or outputs on the CloudSpace.

---

## 1. How the pipeline works

```
 INPUT IMAGE
    │
    ▼
 ┌────────────────────────────────────────────────────────────┐
 │ preprocess            _square(): center-crop → 1024×1024  │
 │ (rembg if enabled)    background removal (OFF here)        │
 └────────────────────────────────────────────────────────────┘
    │
    ▼  stage = "shape", progress 0.05
 ┌────────────────────────────────────────────────────────────┐
 │ SHAPE  (hy3dshape → Hunyuan3DDiTFlowMatchingPipeline)      │
 │   - Multi-view diffusion (30 steps, CFG 5.0)               │
 │   - Octree → triplane → VAE decode (chunked 8000)         │
 │   - Marching cubes → mesh                                  │
 │   Uses weights: hunyuan3d-dit-v2-1 (7.37 GB, fp16)         │
 └────────────────────────────────────────────────────────────┘
    │                        progress 0.6
    ▼
 ┌────────────────────────────────────────────────────────────┐
 │ TEXTURE (hy3dpaint)  ❌ DISABLED on T4 (needs ~21 GB)      │
 │   would bake PBR texture onto the mesh                     │
 └────────────────────────────────────────────────────────────┘
    │                        progress 0.95 (skipped)
    ▼
 post-process  →  export GLB  →  progress 1.0  →  SUCCEEDED
```

### The T4 memory problem (why stages are serialized)

Hunyuan3D 2.1's weights exceed the T4's memory:

| Stage | Weights | Fits alone on 14.6 GB? |
|---|---|---|
| Shape (DiT + VAE) | ~10 GB | Yes |
| Texture (paint) | ~21 GB | No |

So the pipeline **never keeps two heavy stages on the GPU at once**:

1. Load shape → run on GPU → `model.to("cpu")` + `torch.cuda.empty_cache()`
2. (If texture enabled) load paint → run → offload
3. Peak VRAM ≈ 8.1 GiB during shape — safely under the 14.6 GB ceiling

Developer note: upstream's `enable_model_cpu_offload()` **crashes** the shape
pipeline; the fix is to load on `device="cuda"` directly and skip offload hooks
for the v2.1 backend (see `app/pipeline.py:_build_shape`).

---

## 2. Architecture / moving parts (`app/`)

| File | Responsibility |
|---|---|
| `server.py` | FastAPI app; HTTP endpoints; job worker wiring; public endpoints; a shared `asyncio.Lock` around GPU work |
| `pipeline.py` | The Hunyuan3DPipeline orchestrator: preprocess → shape → texture → export GLB (supports `return_bytes` for in-memory export) |
| `jobs.py` | `JobManager`: in-process async queue (1 worker), tracks status/progress per job |
| `storage.py` | `Storage` (disk) + `MemoryStorage` (RAM only; used by the public path) |
| `gpu_manager.py` | VRAM budgets, per-stage locks, `force_offload()`, CUDA cache control |
| `config.py` | All runtime settings via env / `.env` |
| `schemas.py` | Pydantic request/response models |

### Request flow (end to end)

```
 Client ──POST /v1/public/generate/upload──► server.py
   ▲                                          │  reads image → submit job
   │                                          ▼
   └──GET /v1/public/jobs/{id}◄── JobManager(1 worker, RAM)
        │                        worker: _execute_job_public
        │                        asyncio.to_thread → GPU_LOCK → pipeline.run()
        ▼
 GET /v1/public/jobs/{id}/result  →  GLB bytes (RAM)  →  purged after download
```

- HTTP stays responsive during a run (CUDA runs in a background thread).
- Only **1 active job** (`HY3D_MAX_ACTIVE_JOBS=1`) — extra uploads sit queued.
- Public-pipeline GLBs exist in RAM only; a server restart **loses queued/running
  jobs** — you must re-upload.

### Public page (`app/static/index.html`)

Browser UI at `/` on the public URL: drag & drop an image → upload → animated
progress bar (polls `/v1/public/jobs/{id}` every 5 s) → 3D preview via
`<model-viewer>` → download button. All requests hit the `v1/public/*` endpoints.

---

## 3. Endpoints

### Local API (disk-persisted)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/generate` | JSON submit (base64 image). → `202` + `{job_id}` |
| `POST` | `/v1/generate/upload` | Multipart `file` upload. → `202` + `{job_id}` |
| `GET` | `/v1/jobs/{id}` | Job status: `status`, `progress`, `stage`, `error` |
| `GET` | `/v1/jobs` | List recent jobs |
| `GET` | `/v1/jobs/{id}/result` | Download GLB (`model/gltf-binary`) |
| `GET` | `/v1/jobs/{id}/thumbnail` | PNG preview (pyrender, if available) |
| `POST` | `/v1/jobs/{id}/cancel` | Cancel a queued job |
| `GET` | `/health` | GPU name, VRAM, active jobs, queue depth |

### Public API (RAM-only, no disk)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/public/generate/upload` | Multipart upload → `202` + `{job_id}` |
| `GET` | `/v1/public/jobs/{id}` | Status |
| `GET` | `/v1/public/jobs/{id}/result` | GLB bytes (served once, then purged) |
| `GET` | `/` | The web page (`index.html`) |

Run `GET /docs` (Swagger) on the server for interactive testing.

---

## 4. Configuration (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `HY3D_ENABLE_TEXTURE` | `0` (on T4) | `1` = texture baking (needs ~21 GB, OFF here) |
| `HY3D_ENABLE_REMBG` | `0` (on T4) | `1` = background removal |
| `HY3D_STEPS` | `30` | Shape diffusion steps (`5` for turbo ckpt) |
| `HY3D_GUIDANCE` | `5.0` | CFG guidance scale |
| `HY3D_OCTREE` | `256` | Mesh octree resolution (128–384) |
| `HY3D_VAE_CHUNKS` | `8000` | Lower = less peak VRAM during VAE decode |
| `HY3D_MAX_ACTIVE_JOBS` | `1` | Keep 1 (shape stage not reentrant) |
| `HY3D_PORT` | `8080` | HTTP port |
| `HY3D_REPO_DIR` | — | Path to the Hunyuan3D-2.1 source repo |
| `HY3D_MODEL_REPO` | `tencent/Hunyuan3D-2.1` | HF repo for weights |
| `HY3D_SHAPE_SUBFOLDER` | `hunyuan3d-dit-v2-1` | Where the shape ckpt lives in the repo |
| `HY3D_TEX_RES` / `HY3D_TEX_VIEWS` | `512` / `6` | Texture quality vs. time |

Production `.env` on the CloudSpace sets (shape-only):
`HY3D_ENABLE_TEXTURE=0`, `HY3D_ENABLE_REMBG=0`, `HY3D_STEPS=30`,
`HY3D_GUIDANCE=5.0`, `HY3D_OCTREE=256`, `HY3D_VAE_CHUNKS=8000`,
`HY3D_MAX_ACTIVE_JOBS=1`, `HY3D_PORT=8080`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## 5. How to use it (end to end)

### A. Start the server (CloudSpace is asleep by default)

```bash
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 \
  s_01m2n3gj3pwc5ga8pfj38nkq1r@ssh.lightning.ai \
  "cd ~/hunyuan3d-api && (nohup ./.venv/bin/python -m app.server > ./outputs/server.log 2>&1 &) && sleep 8 && curl -s http://127.0.0.1:8080/health"
```

Expected: `{"status":"ok","gpu_name":"Tesla T4",...}`. If SSH rejects the key,
retry (the CloudSpace just restarted).

### B. Convert an image via the public API (from your local machine)

```bash
# 1. upload → remember the job_id
curl -X POST "https://8080-01m2n3gj3pwc5ga8pfj38nkq1r.cloudspaces.litng.ai/v1/public/generate/upload" \
  -F "file=@E:\AI\image_for_glb\arm.jpg" -F "enable_texture=false"

# 2. poll (takes ~6-7 min)
curl -s "https://8080-01m2n3gj3pwc5ga8pfj38nkq1r.cloudspaces.litng.ai/v1/public/jobs/<job_id>"
#   look for "status":"succeeded","progress":1.0

# 3. download the GLB (served once, then purged)
curl -o "E:\AI\glb-output\arm.glb" \
  "https://8080-01m2n3gj3pwc5ga8pfj38nkq1r.cloudspaces.litng.ai/v1/public/jobs/<job_id>/result"
```

Or use the **browser page** at `https://8080-01m2n3gj3pwc5ga8pfj38nkq1r.cloudspaces.litng.ai/`.

### C. Or use the local (disk) API over SSH

```bash
# put the image on the server
scp "E:\AI\image_for_glb\arm.jpg" s_01m2n3gj3pwc5ga8pfj38nkq1r@ssh.lightning.ai:~/hunyuan3d-api/input.jpg

# submit locally
ssh s_01m2n3gj3pwc5ga8pfj38nkq1r@ssh.lightning.ai \
  'curl -X POST http://127.0.0.1:8080/v1/generate/upload -F "file=@/teamspace/studios/this_studio/hunyuan3d-api/input.jpg" -F "enable_texture=false"'

# poll + download
ssh ... 'curl -s http://127.0.0.1:8080/v1/jobs/<job_id>'
ssh ... 'curl -sL http://127.0.0.1:8080/v1/jobs/<job_id>/result -o /tmp/out.glb'
scp s_...:~/.../tmp/out.glb "E:\AI\glb-output\arm.glb"
```

Local-API GLBs land in `outputs/glb/` on disk and survive restarts.

---

## 6. Timing / performance (measured on T4)

Typical shape-only job (~6.3 min total):

| Phase | Time | Notes |
|---|---|---|
| Model load | ~2-4 min | fp16 ckpt into VRAM (GPU 95-100% util) |
| Diffusion | ~1-2 min | 30 steps @ ~2.07 s/it |
| VAE / volume decode | ~45 s | 1061 chunks @ ~24.6 it/s |
| Surface extraction + GLB export | seconds | marching cubes @ octree 256 |

- **GPU memory curve:** 0.10 → 6.96 GiB (load) → 8.1 GiB (peak) → 179 MiB idle.
- Output sizes: ~2.6–14 MB GLB, 359K–384K verts (mesh quality varies by image).
- Full timing report: `E:\AI\Model3DGeneration\performance.txt`.

---

## 7. Tuning knobs

| Goal | Change |
|---|---|
| Faster result | `HY3D_STEPS=15` (or turbo ckpt `5`), `HY3D_OCTREE=256` |
| Lower VRAM | `HY3D_VAE_CHUNKS=4000`, `HY3D_OCTREE=224` |
| Finer mesh | `HY3D_OCTREE=320` (slower, more VRAM) |
| Textured output (not on T4) | `HY3D_ENABLE_TEXTURE=1` on a GPU with ≥ 24 GB |
| More/larger queue | `HY3D_MAX_ACTIVE_JOBS=1` (don't raise — shape isn't reentrant) |

---

## 8. Known issues / gotchas

- **Browser error "Unexpected non-whitespace character after JSON"** → server was
  down (CloudSpace slept). Restart it and re-upload.
- **Public jobs are RAM-only** → restart mid-job loses everything; re-upload.
- **1 job at a time** on both APIs; later uploads wait in queue.
- **No texture on T4** — paint needs ~21 GB; shape-only GLBs have no material map.
- **`pkill -f 'app.server'` footgun** — the pattern also matches SSH command
  strings; use `pkill -f 'python -m app.server'` or the exact venv python path.
- SSH intermittently fails right after a CloudSpace restart; simply retry.

---

## 9. Deployment notes (Lightning.ai CloudSpace)

- **SSH host:** `s_01m2n3gj3pwc5ga8pfj38nkq1r@ssh.lightning.ai`
- **Project dir:** `/teamspace/studios/this_studio/hunyuan3d-api`
- **Source repo (2.1 bindings):** `/teamspace/studios/this_studio/hy3dgen`
- **Weights cache:** `/teamspace/studios/this_studio/.cache/hy3dgen/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/`
- **Public URL:** `https://8080-01m2n3gj3pwc5ga8pfj38nkq1r.cloudspaces.litng.ai/`
- **Env:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; torch 2.6.0+cu124;
  no PyTorch reinstall allowed on the CloudSpace.