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
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .config import Settings
from .gpu_manager import GpuManager, gpu_info, init_gpu_manager
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
from .storage import MemoryStorage, Storage
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
    yield
    task.cancel()


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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SETTINGS = Settings()
init_gpu_manager(SETTINGS.vram)
GPU: GpuManager = gpu_module.gpu_manager

# One in-process GPU lock: both the disk (JOBS) and RAM-only (PUBLIC_JOBS)
# pipelines serialize their CUDA work through it, so jobs never overlap on GPU.
_GPU_LOCK = asyncio.Lock()

STORAGE = Storage(SETTINGS.output_dir, SETTINGS.job_dir)
PUBLIC_STORAGE = MemoryStorage()
PIPELINE = Hunyuan3DPipeline(SETTINGS, GPU)
try:
    MV_PIPELINE = _MV_CLS(SETTINGS, GPU) if _MV_CLS is not None else None
except Exception:  # noqa: BLE001
    logger.warning("MV pipeline not built (%s)", _MV_CLS)
    MV_PIPELINE = None
LOOP: Any = None

SUPPORTED_MODELS = {"2.1", "2mv"}


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

    def work() -> PipelineResult:
        image_src = payload["image"]
        if isinstance(image_src, str):
            img = _decode_base64(image_src)
        elif isinstance(image_src, bytes):
            img = Image.open(io.BytesIO(image_src))
            img.load()
        else:
            raise ValueError("payload image must be base64 str or bytes")
        return PIPELINE.run(
            img,
            enable_texture=payload.get("enable_texture"),
            num_inference_steps=payload.get("num_inference_steps"),
            guidance_scale=payload.get("guidance_scale"),
            octree_resolution=payload.get("octree_resolution"),
            tex_resolution=payload.get("tex_resolution"),
            progress=progress,
        )

    import asyncio

    async with _GPU_LOCK:
        result: PipelineResult = await asyncio.to_thread(work)
    dst = STORAGE.store_glb(job_id, result.glb_path)
    return {
        "result_url": f"/v1/jobs/{job_id}/result",
        "glb_path": str(dst),
    }


JOBS = JobManager(STORAGE, _execute_job, max_active=SETTINGS.max_active_jobs)


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

    def work() -> PipelineResult:
        model = payload.get("model", "2.1")
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
            return MV_PIPELINE.run(
                views,
                num_inference_steps=payload.get("num_inference_steps"),
                guidance_scale=payload.get("guidance_scale"),
                octree_resolution=payload.get("octree_resolution"),
                return_bytes=True,
                metadata=meta or None,
                progress=progress,
            )
        img = _load_img(payload["image"])
        return PIPELINE.run(
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

    async with _GPU_LOCK:
        result: PipelineResult = await asyncio.to_thread(work)
    if result.glb_bytes:
        PUBLIC_STORAGE.store_glb_bytes(job_id, result.glb_bytes)
    return {"result_url": f"/v1/public/jobs/{job_id}/result"}


PUBLIC_JOBS = JobManager(PUBLIC_STORAGE, _execute_job_public,
                         max_active=SETTINGS.max_active_jobs)


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
                f"view_{tag}": (await v.read()) if v is not None else None
                for tag, v in views.items()
            },
        }
    else:
        if file is None:
            raise HTTPException(status_code=400,
                                detail="'file' image is required for model 2.1")
        data = await file.read()
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


def _download_name(job: Any) -> str:
    """Sanitise the user-supplied object name down to a safe filename token."""
    try:
        raw = (job.payload.get("name") or "").strip()
    except Exception:  # noqa: BLE001
        return "object"
    safe = "".join(c for c in raw if (c.isalnum() or c in "-_ ")).strip()
    safe = " ".join(safe.split())[:60] or "object"
    return f"{safe}_{job.job_id[:8]}"


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    info = gpu_info()
    return HealthOut(
        status="ok" if info.available else "degraded",
        gpu_name=info.name,
        vram_total_gb=info.total_gb if info.total_gb else None,
        vram_used_gb=info.used_gb,
        active_jobs=JOBS.active,
        max_active_jobs=JOBS.max_active,
        queue_depth=JOBS.queue_depth,
        texture_enabled=SETTINGS.enable_texture,
        model_repo=SETTINGS.model_repo,
    )


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