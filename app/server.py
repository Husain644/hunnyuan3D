"""FastAPI entry point: async image-to-3D over Hunyuan3D 2.1 (T4-optimised).

Endpoints
---------
* POST /v1/generate   -> {job_id}      multi-part file or base64 JSON
* GET  /v1/jobs/{id}  -> job status (progress, stage, result_url)
* GET  /v1/jobs       -> list
* GET  /v1/jobs/{id}/result -> GLB binary download (if done)
* GET  /v1/jobs/{id}/thumbnail -> PNG preview
* POST /v1/jobs/{id}/cancel
* GET  /v1/health
"""
from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import Settings
from .gpu_manager import GpuAdmission, GpuManager, gpu_info, init_gpu_manager
from .jobs import JobManager
from .pipeline import Hunyuan3DPipeline, PipelineResult, _decode_base64
try:  # optional: Hunyuan3D-2mv (multi-view) adapter
    from .pipeline_mv import Hunyuan3DMVPipeline as _MV_CLS
except Exception:  # noqa: BLE001
    logger = __import__("logging").getLogger("hy3d.server")
    logger.warning("2mv adapter not importable; multi-view model disabled")
    _MV_CLS = None
from .schemas import (
    ErrorOut,
    GenerateRequest,
    HealthOut,
    JobListOut,
    JobOut,
    JobStatus,
)
from .storage import MemoryStorage, PayloadStore, Storage
import app.gpu_manager as gpu_module

logger = logging.getLogger("hy3d.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

GO_APP_PATH = Path(__file__).resolve().parent / "static"
_G_PUB_SWEEP_TTL = 3600.0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start the job workers and the public TTL sweeper."""
    global LOOP
    LOOP = asyncio.get_running_loop()
    JOBS.start(LOOP)
    PUBLIC_JOBS.start(LOOP)
    task = asyncio.create_task(_public_sweeper())

    # Re-queue in-flight jobs that were persisted to disk (crash/restart
    # recovery), so a restart never silently auto-cancels a user's job.
    for _mgr in (JOBS, PUBLIC_JOBS):
        try:
            _mgr.recover()
        except Exception:  # noqa: BLE001
            logger.exception("job recovery failed")

    # Startup banner: which checkpoints are actually available to serve.
    try:
        from .models_check import all_status  # noqa: F401

        agg = all_status()
        if agg["ready"]:
            logger.info("models ready: %s (cache: %s)",
                        ", ".join(agg["ready"]) or "(none)", agg["cache_dir"])
        else:
            logger.warning(
                "NO model checkpoints found in %s. Generation will fail until: "
                ".venv/bin/python scripts/model_bootstrap.py",
                agg["cache_dir"],
            )
    except Exception:  # noqa: BLE001
        logger.warning("could not check model checkpoints at startup")
    yield
    task.cancel()


SETTINGS = Settings()

app = FastAPI(
    title="Hunyuan3D 2.1 API",
    version="0.1.0",
    description="Async image-to-3D (GLB) with T4 CPU-offload orchestration.",
    openapi_url="/openapi.json",
    docs_url="/docs",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
init_gpu_manager(SETTINGS.vram)
GPU: GpuManager = gpu_module.gpu_manager

# One in-process GPU admission controller + a pipeline instance per slot.
# Jobs (both JOBS and PUBLIC_JOBS) must hold a slot (VRAM-aware, auto cap)
# while their GPU phases run; the pool hands each slot its own pipeline so
# concurrent runs never share a mutable model instance.
GPU_ADM = GpuAdmission(jobs_vram_gb=SETTINGS.jobs_vram_gb,
                       hard_cap=SETTINGS.max_active_jobs)
_SLOTS = max(1, GPU_ADM.cap())
_PIPELINE_POOL: deque = deque()
_MV_POOL: deque = deque()
_POOL_LOCK = asyncio.Lock()  # guards round-robin pick/return between tasks

STORAGE = Storage(SETTINGS.output_dir, SETTINGS.job_dir)
PUBLIC_STORAGE = MemoryStorage()
PIPELINE = Hunyuan3DPipeline(SETTINGS, GPU)
try:
    MV_PIPELINE = _MV_CLS(SETTINGS, GPU) if _MV_CLS is not None else None
except Exception:  # noqa: BLE001
    logger.warning("MV pipeline not built (%s)", _MV_CLS)
    MV_PIPELINE = None

_PIPELINE_POOL.extend(Hunyuan3DPipeline(SETTINGS, GPU) for _i in range(_SLOTS))
if MV_PIPELINE is not None:
    _MV_POOL.extend(_MV_CLS(SETTINGS, GPU) for _i in range(_SLOTS))
LOOP: Any = None

SUPPORTED_MODELS = {"2.1", "2mv"}


async def _take_pipeline(mv: bool) -> Any:
    """Round-robin a mutable pipeline instance out of the pool."""
    pool = _MV_POOL if mv else _PIPELINE_POOL
    while True:
        async with _POOL_LOCK:
            if pool:
                return pool.popleft()
        await asyncio.sleep(0.1)


def _return_pipeline(mv: bool, pipe: Any) -> None:
    pool = _MV_POOL if mv else _PIPELINE_POOL
    pool.append(pipe)


# --------------------------------------------------------------------------
# job execution
# --------------------------------------------------------------------------
async def _execute_job(job_id: str, payload: dict) -> dict:
    """Runs the heavy pipeline in a thread (blocking CUDA calls), reporting
    progress back into the job state. ``JOBS`` is resolved at call time."""
    from PIL import Image

    def progress(stage: str, pct: float) -> None:
        job = JOBS.get(job_id)  # JOBS is a module global, bound by call time
        if job is not None:
            job.set_status(JobStatus.RUNNING, stage=stage, progress=pct,
                           message=stage)

    def work(pipe: Any) -> PipelineResult:
        image_src = payload["image"]
        if isinstance(image_src, str):
            img = _decode_base64(image_src)
        elif isinstance(image_src, bytes):
            img = Image.open(io.BytesIO(image_src))
            img.load()
        else:
            raise ValueError("payload image must be base64 str or bytes")
        t0 = time.monotonic()
        res = pipe.run(
            img,
            enable_texture=payload.get("enable_texture"),
            num_inference_steps=payload.get("num_inference_steps"),
            guidance_scale=payload.get("guidance_scale"),
            octree_resolution=payload.get("octree_resolution"),
            tex_resolution=payload.get("tex_resolution"),
            progress=progress,
        )
        logger.info("job %s done in %.1fs (faces=%d)",
                    job_id, time.monotonic() - t0, res.mesh_faces)
        return res

    async with GPU_ADM:
        pipe = await _take_pipeline(mv=False)
        try:
            result: PipelineResult = await asyncio.to_thread(work, pipe)
        finally:
            _return_pipeline(False, pipe)
    dst = STORAGE.store_glb(job_id, result.glb_path)
    return {
        "result_url": f"/v1/jobs/{job_id}/result",
        "glb_path": str(dst),
    }


# One worker slot shared across BOTH queue managers so the public page and the
# internal API can never drive two heavy pipelines into VRAM at the same time.
_GLOBAL_SLOT = asyncio.Semaphore(_SLOTS)

JOBS = JobManager(STORAGE, _execute_job, max_active=_SLOTS, global_slot=_GLOBAL_SLOT,
                  records=PayloadStore(SETTINGS.job_dir / "recovery_internal"))


# --------------------------------------------------------------------------
# public (RAM-only) pipeline: no disk writes, GLB purged after download
# --------------------------------------------------------------------------
async def _execute_job_public(job_id: str, payload: dict) -> dict:
    from PIL import Image

    def progress(stage: str, pct: float) -> None:
        job = PUBLIC_JOBS.get(job_id)
        if job is not None:
            job.set_status(JobStatus.RUNNING, stage=stage, progress=pct,
                           message=stage)

    def _load_img(src):
        if isinstance(src, str):
            return _decode_base64(src)
        img = Image.open(io.BytesIO(src))
        img.load()
        return img

    meta = {}
    if payload.get("name"):
        meta["name"] = payload["name"]
    if payload.get("description"):
        meta["description"] = payload["description"]

    def work(pipe: Any) -> PipelineResult:
        model = payload.get("model", "2.1")
        t0 = time.monotonic()
        if model == "2mv":
            if MV_PIPELINE is None:
                raise RuntimeError("multi-view model not available on this host")
            views = {}
            for tag in ("front", "back", "left", "right"):
                src = payload.get(f"view_{tag}")
                if src:
                    views[tag] = _load_img(src)
            if not views:
                raise ValueError("2mv requires at least one view image")
            res = pipe.run(
                views,
                num_inference_steps=payload.get("num_inference_steps"),
                guidance_scale=payload.get("guidance_scale"),
                octree_resolution=payload.get("octree_resolution"),
                return_bytes=True,
                metadata=meta or None,
                progress=progress,
            )
        else:
            img = _load_img(payload["image"])
            res = pipe.run(
                img,
                enable_texture=payload.get("enable_texture", False),
                num_inference_steps=payload.get("num_inference_steps"),
                guidance_scale=payload.get("guidance_scale"),
                octree_resolution=payload.get("octree_resolution"),
                tex_resolution=payload.get("tex_resolution"),
                return_bytes=True,
                metadata=meta or None,
                progress=progress,
            )
        logger.info("public job %s (%s) done in %.1fs (faces=%d)",
                    job_id, model, time.monotonic() - t0, res.mesh_faces)
        return res

    async with GPU_ADM:
        mv = payload.get("model", "2.1") == "2mv"
        pipe = await _take_pipeline(mv)
        try:
            result: PipelineResult = await asyncio.to_thread(work, pipe)
        finally:
            _return_pipeline(mv, pipe)
    if result.glb_bytes:
        PUBLIC_STORAGE.store_glb_bytes(job_id, result.glb_bytes)
    return {"result_url": f"/v1/public/jobs/{job_id}/result"}


PUBLIC_JOBS = JobManager(PUBLIC_STORAGE, _execute_job_public,
                         max_active=_SLOTS, global_slot=_GLOBAL_SLOT,
                         records=PayloadStore(SETTINGS.job_dir / "recovery"))


async def _public_sweeper() -> None:
    """Periodically drop finished public jobs so RAM stays bounded."""
    import asyncio as _a

    while True:
        await _a.sleep(30)
        try:
            PUBLIC_JOBS.sweep_finished(_G_PUB_SWEEP_TTL)
        except Exception:  # noqa: BLE001
            logger.exception("public sweeper error")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    p = GO_APP_PATH / "index.html"
    if not p.exists():
        return JSONResponse({"detail": "public page not bundled"},
                            status_code=501)
    return FileResponse(p)


@app.post("/v1/public/generate/upload", response_model=JobOut, status_code=202)
async def public_generate_upload(
    file: Optional[UploadFile] = File(None),
    model: str = Form("2.1"),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    view_front: Optional[UploadFile] = File(None),
    view_back: Optional[UploadFile] = File(None),
    view_left: Optional[UploadFile] = File(None),
    view_right: Optional[UploadFile] = File(None),
    enable_texture: Optional[bool] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    octree_resolution: Optional[int] = Form(None),
    tex_resolution: Optional[int] = Form(None),
) -> JobOut:
    model = model or "2.1"
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model '{model}'")
    if model == "2mv" and MV_PIPELINE is None:
        raise HTTPException(status_code=503,
                            detail="multi-view model not available on this host")
    max_bytes = SETTINGS.max_upload_mb * 1024 * 1024

    async def _cap(name: str, data: bytes) -> bytes:
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"'{name}' exceeds the {SETTINGS.max_upload_mb} MB "
                       f"upload limit ({len(data)} bytes)",
            )
        return data

    if model == "2mv":
        views = {
            "front": view_front,
            "back": view_back,
            "left": view_left,
            "right": view_right,
        }
        if all(v is None for v in views.values()):
            raise HTTPException(status_code=400,
                                detail="2mv requires at least one view image")
        payload: dict[str, Any] = {
            "model": model,
            "name": (name or "").strip() or None,
            "description": (description or "").strip() or None,
            **{
                f"view_{tag}": (
                    await _cap(tag, await v.read()) if v is not None else None
                ) for tag, v in views.items()
            },
        }
    else:
        if file is None:
            raise HTTPException(status_code=400,
                                detail="'file' image is required for model 2.1")
        data = await _cap("file", await file.read())
        payload = {
            "model": model,
            "image": data,
            "name": (name or "").strip() or None,
            "description": (description or "").strip() or None,
        }
    opts: dict[str, Any] = {
        "enable_texture": enable_texture if enable_texture is not None else False,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "octree_resolution": octree_resolution,
        "tex_resolution": tex_resolution,
    }
    job_id = str(uuid.uuid4())
    PUBLIC_JOBS.submit(job_id, {**payload, **opts})
    job = PUBLIC_JOBS.get(job_id)
    assert job is not None
    return JobOut(**job.snapshot())


@app.get("/v1/public/jobs/{job_id}", response_model=JobOut)
async def public_job_status(job_id: str) -> JobOut:
    from .storage import MemoryStorage as _MS  # noqa: F401

    job = PUBLIC_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404,
                            detail="public job not found (expired or already served)")
    return JobOut(**job.snapshot())


@app.get("/v1/public/jobs/{job_id}/result")
async def public_job_result(job_id: str) -> Response:
    job = PUBLIC_JOBS.get(job_id)
    if job is None or job.state["status"] != JobStatus.SUCCEEDED.value:
        raise HTTPException(status_code=409, detail="result not ready")
    data = PUBLIC_STORAGE.read_glb(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="result file missing")
    PUBLIC_JOBS.forget(job_id)  # serve once, then purge from RAM
    fname = _download_name(job)
    return Response(
        content=data,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": f'attachment; filename="{fname}.glb"'},
    )


@app.post("/v1/public/jobs/{job_id}/cancel")
async def public_job_cancel(job_id: str) -> JobOut:
    job = PUBLIC_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="public job not found")
    PUBLIC_JOBS.cancel(job_id)
    return JobOut(**job.snapshot())


@app.delete("/v1/public/jobs/{job_id}")
async def public_job_delete(job_id: str) -> JSONResponse:
    job = PUBLIC_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="public job not found")
    status = job.state.get("status")
    deferred = status == "running"
    if deferred:
        PUBLIC_JOBS.cancel(job_id)          # _run marks CANCELLED + forgets
    else:
        if status == "queued":
            PUBLIC_JOBS.cancel(job_id)
        PUBLIC_JOBS.forget(job_id)
    return JSONResponse({"ok": True, "deferred": deferred})


def _download_name(job: Any) -> str:
    """Sanitise the user-supplied object name down to a safe filename token."""
    try:
        raw = (job.payload.get("name") or "").strip()
    except Exception:  # noqa: BLE001
        return "object"
    safe = "".join(c for c in raw if (c.isalnum() or c in "-_ ")).strip()
    safe = " ".join(safe.split())[:60] or "object"
    return f"{safe}_{job.job_id[:8]}"


# --------------------------------------------------------------------------
# live system stats
# --------------------------------------------------------------------------
_CPU_STATE: list[int] = []
_GPU_UTIL_CACHE: list[float] = []
_GPU_UTIL_T = 0.0


def _read_stat() -> list[int]:
    try:
        with open("/proc/stat") as fh:
            parts = fh.readline().split()[1:]
        return [int(x) for x in parts]
    except Exception:  # noqa: BLE001 (non-Linux)
        return []


def _cpu_percent() -> Optional[float]:
    """CPU utilisation over a ~200 ms /proc/stat delta."""
    global _CPU_STATE
    now = _read_stat()
    if not now:
        return None
    if _CPU_STATE:
        idle = now[3] - _CPU_STATE[3]
        tot = sum(now) - sum(_CPU_STATE)
        _CPU_STATE = now
        if tot <= 0:
            return None
        return round(max(0.0, min(100.0, 100.0 * (1.0 - idle / tot))), 1)
    _CPU_STATE = now
    return 0.0


async def _gpu_util() -> Optional[float]:
    global _GPU_UTIL_CACHE, _GPU_UTIL_T
    now = time.time()
    if _GPU_UTIL_CACHE and now - _GPU_UTIL_T < 1.0:
        return _GPU_UTIL_CACHE[0]
    try:
        out = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            ),
        )
        val = float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        val = None
    _GPU_UTIL_CACHE = [val]
    _GPU_UTIL_T = now
    return val


def _disk_free_gb() -> Optional[float]:
    try:
        return round(shutil.disk_usage(str(SETTINGS.output_dir)).free / 1e9, 1)
    except Exception:  # noqa: BLE001
        return None


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    info = gpu_info()
    if SETTINGS.max_active_jobs is None:
        admission = "auto"
    else:
        admission = f"fixed:{SETTINGS.max_active_jobs}"
    return HealthOut(
        status="ok" if info.available else "degraded",
        gpu_name=info.name,
        vram_total_gb=info.total_gb if info.total_gb else None,
        vram_used_gb=info.used_gb,
        active_jobs=JOBS.active + PUBLIC_JOBS.active,
        max_active_jobs=GPU_ADM.cap(),
        active_cap=GPU_ADM.cap(),
        admission=admission,
        queue_depth=JOBS.queue_depth + PUBLIC_JOBS.queue_depth,
        texture_enabled=SETTINGS.enable_texture,
        model_repo=SETTINGS.model_repo,
        gpu_util=await _gpu_util(),
        cpu_percent=await asyncio.get_running_loop().run_in_executor(
            None, _cpu_percent
        ),
        disk_free_gb=_disk_free_gb(),
    )


@app.post("/v1/reset", include_in_schema=False)
async def reset_server() -> dict:
    """In-page "reset": clear leftover/stuck jobs and free GPU memory without
    restarting the process."""
    cleared = JOBS.reset() + PUBLIC_JOBS.reset()
    released = 0
    async with _POOL_LOCK:
        for pool in (_PIPELINE_POOL, _MV_POOL):
            for _i in range(len(pool)):
                pipe = pool.popleft()
                unload = getattr(pipe, "_unload_all", None)
                if callable(unload):
                    try:
                        unload()
                        released += 1
                    except Exception:  # noqa: BLE001
                        logger.exception("reset: pipeline unload failed")
                pool.append(pipe)
    GpuManager.empty_cache()
    info = gpu_info()
    logger.info("Reset: cleared %d job(s), unloaded %d pipeline(s)",
                cleared, released)
    return {
        "cleared": cleared,
        "pipelines_unloaded": released,
        "vram_used_gb": info.used_gb if info.available else None,
        "vram_free_gb": info.free_gb if info.available else None,
    }


@app.get("/health/models")
async def health_models() -> dict:
    """Checkpoint availability for each supported model (never downloads)."""
    from .models_check import all_status

    agg = all_status()
    ready = sorted(agg["ready"])
    return {
        "status": "ok" if ready else "degraded",
        "supported": sorted(SUPPORTED_MODELS),
        "ready": ready,
        "missing": sorted(agg["missing"]),
        "cache_dir": agg["cache_dir"],
        "models": {
            mid: {
                "ready": st["valid"],
                "size_bytes": st["size_bytes"],
                "note": st["reason"] or "ready",
            }
            for mid, st in agg["models"].items()
        },
    }


@app.post("/v1/generate", response_model=JobOut, status_code=202)
async def generate(body: GenerateRequest) -> JobOut:
    """JSON submission: ``{image: <base64>, ...opts}``. Returns 202 + job id."""
    return _submit_job(body.image, _opts_from(body))


@app.post("/v1/generate/upload", response_model=JobOut, status_code=202)
async def generate_upload(
    file: UploadFile = File(...),
    enable_texture: Optional[bool] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    octree_resolution: Optional[int] = Form(None),
    tex_resolution: Optional[int] = Form(None),
) -> JobOut:
    """Multipart submission: ``file`` (image) + optional overrides."""
    data = await file.read()
    opts: dict[str, Any] = {
        "enable_texture": enable_texture if enable_texture is not None else SETTINGS.enable_texture,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "octree_resolution": octree_resolution,
        "tex_resolution": tex_resolution,
    }
    return _submit_job(data, opts)


def _opts_from(body: GenerateRequest) -> dict[str, Any]:
    return {
        "enable_texture": body.enable_texture if body.enable_texture is not None else SETTINGS.enable_texture,
        "num_inference_steps": body.num_inference_steps,
        "guidance_scale": body.guidance_scale,
        "octree_resolution": body.octree_resolution,
        "tex_resolution": body.tex_resolution,
    }


def _submit_job(image_src: Any, opts: dict[str, Any]) -> JobOut:
    job_id = str(uuid.uuid4())
    payload = {"image": image_src, **opts}
    JOBS.submit(job_id, payload)
    job = JOBS.get(job_id)
    assert job is not None
    return JobOut(**job.snapshot())


@app.get("/v1/jobs", response_model=JobListOut)
async def list_jobs(limit: int = 100) -> JobListOut:
    states = JOBS.list_states(limit=min(max(limit, 1), 500))
    return JobListOut(jobs=[JobOut(**s) for s in reversed(states)], total=len(states))


@app.get("/v1/jobs/{job_id}", response_model=JobOut)
async def job_status(job_id: str) -> JobOut:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut(**job.snapshot())


@app.get(
    "/v1/jobs/{job_id}/result",
    responses={200: {"content": {"model/gltf-binary": {}}}},
)
async def download_result(job_id: str) -> Response:
    job = JOBS.get(job_id)
    if job is None or job.state["status"] != JobStatus.SUCCEEDED.value:
        raise HTTPException(status_code=409, detail="result not ready")
    data = STORAGE.read_glb(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="result file missing")
    return Response(
        content=data,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.glb"'},
    )


@app.get("/v1/jobs/{job_id}/thumbnail")
async def thumbnail(job_id: str) -> Response:
    job = JOBS.get(job_id)
    if job is None or job.state["status"] != JobStatus.SUCCEEDED.value:
        raise HTTPException(status_code=409, detail="result not ready")
    data = STORAGE.read_glb(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="result file missing")
    try:
        import io as _io

        import trimesh

        scene = trimesh.load(_io.BytesIO(data), file_type="glb", force="mesh")
        png = scene.scene_to_bytes() if hasattr(scene, "scene_to_bytes") else None
        if png is None:
            # fallback: render via pyrender if available
            try:
                import pyrender  # type: ignore

                mesh = trimesh.load(_io.BytesIO(data), file_type="glb", force="mesh")
                r = pyrender.OffscreenRenderer(512, 512)
                scene = pyrender.Scene.from_trimesh_scene(mesh.scene())
                color = r.render(scene)[0]
                from PIL import Image

                buf = _io.BytesIO()
                Image.fromarray(color).save(buf, format="PNG")
                png = buf.getvalue()
            except Exception:
                png = None
        if png:
            return Response(content=png, media_type="image/png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("thumbnail render failed: %s", exc)
    raise HTTPException(status_code=415, detail="thumbnail unavailable")


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> JobOut:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    JOBS.cancel(job_id)
    return JobOut(**job.snapshot())


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host=SETTINGS.host,
        port=SETTINGS.port,
        log_level=SETTINGS.log_level,
    )


if __name__ == "__main__":
    main()