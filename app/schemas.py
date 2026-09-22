"""Pydantic schemas for the Hunyuan3D 2.1 async API."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerateRequest(BaseModel):
    """Body for the JSON form of generation (base64 image)."""

    image: str = Field(
        ..., description="Base64-encoded PNG/JPEG. May be data-URL prefixed."
    )
    enable_texture: Optional[bool] = Field(
        None, description="Override server texture default."
    )
    num_inference_steps: Optional[int] = Field(
        None, ge=1, le=200, description="Shape diffusion steps."
    )
    guidance_scale: Optional[float] = Field(
        None, ge=0.0, le=30.0, description="CFG guidance scale."
    )
    octree_resolution: Optional[int] = Field(
        None, ge=128, le=512, description="Mesh octree resolution."
    )
    tex_resolution: Optional[int] = Field(
        None, ge=128, le=1024, description="Texture map resolution."
    )
    overwrite: Optional[bool] = Field(False, description="Force rerun if done before.")


class JobOut(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    stage: Optional[str] = None
    message: Optional[str] = None
    created_at: float
    updated_at: float
    error: Optional[str] = None
    result_url: Optional[str] = None
    result_mime: Optional[str] = None


class JobListOut(BaseModel):
    jobs: list[JobOut]
    total: int


class HealthOut(BaseModel):
    status: Literal["ok", "degraded", "busy"]
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_used_gb: float = 0.0
    active_jobs: int
    max_active_jobs: int
    active_cap: int = 1
    admission: str = "auto"
    queue_depth: int
    texture_enabled: bool
    model_repo: str
    gpu_util: Optional[float] = None
    cpu_percent: Optional[float] = None
    disk_free_gb: Optional[float] = None


class ErrorOut(BaseModel):
    detail: str