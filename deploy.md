# Deploying Hunyuan3D 2.1 API

> **Starting on a fresh machine?** Use the one-command reproducible setup —
> GPU gate → venv → pinned deps → model source → checkpoints → .env →
> validation — instead of the manual steps below:
> ```bash
> git clone git@github.com:Husain644/hunnyuan3D.git
> cd hunnyuan3D && ./setup.sh --start
> ```
> The rest of this document details the two *verified* deployments and the
> manual config they rely on.

Two production setups have been **verified end-to-end** (image in → GLB out):

| Target | GPU | VRAM | Storage | Works for |
|---|---|---|---|---|
| Lightning.ai CloudSpace | Tesla T4 | 14.6 GB | RAM-only (public) + disk (v1) | shape-only, 1 job |
| Vast.ai instance | RTX 2080 Ti | 22 GB | disk (v1) | shape-only (+ texture w/ reseverve) |

Both run the **same `app/` code** on the `main` branch. The only config
difference is `.env`.

---

## Hardware requirements

- **GPU with ≥ 14 GB VRAM** (T4, RTX 2080 Ti, L4, A10, ...). Verified on T4 and
  2080 Ti (compute capability 7.5 → CUDA ≤ 12.x wheels work).
- CUDA toolkit matching the GPU; the pinned `torch` build from `requirements.txt`
  (e.g. `2.6.0+cu124`) works on Turing/Ampere cards.
- ~10 GB free disk for weights (`~/.cache/hy3dgen/...`), ~7.4 GB download on
  first job.
- Texture stage (`HY3D_ENABLE_TEXTURE=1`) needs a GPU with **≥ 21 GB** and the
  custom rasterizer compiled from `hy3dpaint/custom_rasterizer`.

---

## Prerequisites

1. Install Python 3.11/3.12, CUDA-capable torch, and deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Clone the Hunyuan3D-2.1 source next to this service (the pipeline imports
   `hy3dshape` from it):
   ```bash
   git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
   # note the absolute path; you'll point HY3D_REPO_DIR at it
   ```
3. (Optional, texture only) Compile the rasterizer:
   ```bash
   cd Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && python setup.py install
   ```

## `.env` (copy from `.env.example`)

```dotenv
HY3D_ENABLE_TEXTURE=0                 # 1 = texture (needs ~21 GB GPU)
HY3D_ENABLE_REMBG=0                   # 0 skips background removal (default off)
HY3D_STEPS=30                         # shape diffusion steps
HY3D_OCTREE=256                       # mesh resolution
HY3D_MAX_ACTIVE_JOBS=1                # keep 1 (shape is not reentrant)
HY3D_PORT=8080
HY3D_REPO_DIR=/absolute/path/to/Hunyuan3D-2.1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Run

```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8080
# or: bash run.sh
```

### Local API (disk-persisted GLBs)

```bash
curl -X POST http://localhost:8080/v1/generate/upload \
  -F "file=@chair.png" -F "enable_texture=false"        # → job_id
curl -s http://localhost:8080/v1/jobs/<job_id>           # poll status
curl -o out.glb http://localhost:8080/v1/jobs/<job_id>/result
```

### Public API (RAM-only, zero disk writes, GLB purged after download)

```bash
curl -X POST http://localhost:8080/v1/public/generate/upload \
  -F "file=@chair.png" -F "enable_texture=false"
curl -s http://localhost:8080/v1/public/jobs/<job_id>
curl -o out.glb http://localhost:8080/v1/public/jobs/<job_id>/result  # served once
```

A drag-and-drop web page is served at `/` (uses the public API).

---

## Verified deployment A — Lightning.ai CloudSpace (Tesla T4)

Seed code + venv, then run the same service:
```bash
cd ~/hunyuan3d-api && (nohup ./.venv/bin/python -m app.server > outputs/server.log 2>&1 &)
curl -s http://127.0.0.1:8080/health
```
The CloudSpace's auto-generated public URL exposes port 8080 directly. Memory
facts:
- GPU usable: ~14.6 GB → shape-only peak ~8.1 GiB, texture 21 GB does **not** fit.
- Weights cache: `~/.cache/hy3dgen/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/`.
- A CloudSpace sleep/restart wipes running RAM-only public jobs → re-upload.

## Verified deployment B — Vast.ai (RTX 2080 Ti, supervisor + Caddy)

Managed service that survives crashes, with token-authed public access.

1. Copy the repo, clone Hunyuan3D-2.1, install deps into `/venv/main`:
   ```bash
   git clone ... # this repo → ~/hunyuan3d-api
   git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git ~/hy3dgen
   source /venv/main/bin/activate
   uv pip install -r ~/hunyuan3d-api/requirements.txt scikit-image opencv-python-headless
   ```
2. `.env` as above (`HY3D_REPO_DIR=/root/hy3dgen`, `HY3D_PORT=17070`).
3. Supervisor service — `/opt/supervisor-scripts/hunyuan3d.sh`:
   ```bash
   #!/bin/bash
   source /venv/main/bin/activate
   cd /root/hunyuan3d-api
   export HY3D_REPO_DIR=/root/hy3dgen
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   exec python -m uvicorn app.server:app --host 127.0.0.1 --port 17070 --log-level info
   ```
   `/etc/supervisor/conf.d/hunyuan3d.conf`:
   ```ini
   [program:hunyuan3d]
   command=/opt/supervisor-scripts/hunyuan3d.sh
   autostart=true
   autorestart=unexpected
   stdout_logfile=/dev/stdout
   redirect_stderr=true
   stdbout_logfile_maxbytes=0
   ```
   Load with `supervisorctl reread && supervisorctl update`.
4. Expose behind Caddy auth — add to `/etc/portal.yaml` a free open port
   (`external_port` 10100 → app on `internal_port` 17070), restart caddy.
   Users access `http://<PUBLIC_IP>:<VAST_TCP_PORT_10100>` with their token
   (`Authorization: Bearer $OPEN_BUTTON_TOKEN`); unauthenticated requests get 401.
5. RTX 2080 Ti = cc 7.5, same wheel family as T4; disk GLBs persist under
   `outputs/glb/`.

---

## Tuning

| Goal | Change |
|---|---|
| Faster | `HY3D_STEPS=15`; `HY3D_OCTREE=256` |
| Lower VRAM | `HY3D_VAE_CHUNKS=4000`; `HY3D_OCTREE=224` |
| Finer mesh | `HY3D_OCTREE=320` (slower, more VRAM) |
| Textured GLB | `HY3D_ENABLE_TEXTURE=1` on ≥ 21 GB GPU + compiled rasterizer |

## Known gotchas

- **Do not** call `pkill -f 'app.server'` from an SSH one-liner — the pattern
  matches your own shell. Use `pkill -f 'python -m uvicorn'` or the venv path.
- `enable_model_cpu_offload()` **crashes** the v2.1 shape pipeline — the code
  loads on CUDA fp16 directly instead (fixed in `app/pipeline.py`).
- Browser error "Unexpected non-whitespace character after JSON" = the server is
  down (CloudSpace slept / service died). Restart it and re-upload.
- One job at a time across both APIs; further uploads queue.