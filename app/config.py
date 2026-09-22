"""Runtime configuration for the Hunyuan3D 2.1 async service.

All values can be overridden via environment variables or a dotenv file.
The defaults target a 14.6 GB (T4) GPU with sequential CPU offloading.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _on_colab() -> bool:
    """Detect a Google Colab runtime.

    Checks the well-known env vars / paths, PLUS the authoritative test:
    importing `google.colab` (which only exists in a Colab kernel).
    """
    if os.environ.get("COLAB_GPU") or os.environ.get("COLAB_TPU_ADDR"):
        return True
    if Path("/content").is_dir() and not os.environ.get("JUPYTERHUB_BASE_URL"):
        return True
    try:
        # flake8: noqa
        import google.colab  # type: ignore # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


IS_COLAB = _on_colab()


# Hard ceiling for the T4 16 GB card (14.6 GB usable after reserved).
# Stage-level budgets are kept lower so that the *second* half of the
# pipeline (texture) can still fit after the first half is offloaded.
T4_USABLE_GB = 14.6


def _active_cap_setup() -> Optional[int]:
    """Parse HY3D_MAX_ACTIVE_JOBS: numeric = fixed cap, "auto"/empty = None
    (resolved at runtime from measured VRAM by the GpuAdmission controller)."""
    raw = _env("HY3D_MAX_ACTIVE_JOBS", "auto").strip().lower()
    if raw in {"auto", "", "0", "none"}:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _bank_setup() -> dict[str, float]:
    """Reserve VRAM for each pipeline stage. Sum of the listed stage does
    NOT have to fit on the GPU at once -- the pipeline serializes them."""
    return {
        # Shape DiT (10 GB fused) + VAE decode (1 GB) on a 14.6 GB card
        "shape": float(_env("HY3D_VRAM_GB_SHAPE", "11.5")),
        # Paint: delight (1-2 GB) + multiview (8 GB) + baking buffer
        "tex": float(_env("HY3D_VRAM_GB_TEX", "11.0")),
        "rembg": float(_env("HY3D_VRAM_GB_REMBG", "0.5")),
        # Safety headroom below the T4 ceiling
        "reserve": float(_env("HY3D_VRAM_GB_RESERVE", "1.2")),
    }


def _model_setup() -> dict[str, str]:
    """HuggingFace repo and shape subfolder for the v2.1 weights.

    The official `tencent/Hunyuan3D-2.1` repo bundles shape (DiT+VAE) and
    texture (paint) under the *same* HF repo; the shape checkpoint lives in
    the `hunyuan3d-dit-v2-1` subfolder.
    """
    return {
        "repo": _env("HY3D_MODEL_REPO", "tencent/Hunyuan3D-2.1"),
        "shape": _env("HY3D_SHAPE_MODEL", "tencent/Hunyuan3D-2.1"),
        "shape_subfolder": _env("HY3D_SHAPE_SUBFOLDER", "hunyuan3d-dit-v2-1"),
        # v2.1 ships texture under the shape checkpoint's Paint config
        "paint": _env("HY3D_PAINT_MODEL", "tencent/Hunyuan3D-2.1"),
        "dev_id": _env("HY3D_CPU_DEVICE", "CPU"),
    }


def _path(key: str, default: str) -> Path:
    p = Path(_env(key, default)).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Settings:
    host: str = _env("HY3D_HOST", "0.0.0.0")
    port: int = int(_env("HY3D_PORT", "8080"))
    log_level: str = _env("HY3D_LOG_LEVEL", "info")
    debug: bool = _env("HY3D_DEBUG", "0").lower() not in {"0", "false", "no"}

    # Texture/rembg default OFF: both load extra multi-GB weights and need a
    # large GPU (texture >= 21 GB + compiled rasterizer). Opt in explicitly
    # with HY3D_ENABLE_TEXTURE=1 / HY3D_ENABLE_REMBG=1 in .env.
    enable_texture: bool = _env("HY3D_ENABLE_TEXTURE", "0").lower() not in {"0", "false", "no"}
    enable_rembg: bool = _env("HY3D_ENABLE_REMBG", "0").lower() not in {"0", "false", "no"}

    # Upload cap (MiB); FastAPI rejects larger public uploads with HTTP 413.
    max_upload_mb: int = int(_env("HY3D_MAX_UPLOAD_MB", "20"))

    # Comma-separated allowed CORS origins. "*" disables the origin check.
    @property
    def cors_origins(self) -> list[str]:
        raw = _env("HY3D_CORS_ORIGINS", "*")
        return [o.strip() for o in raw.split(",") if o.strip()] or ["*"]

    # Diffusion / mesh extraction knobs (see hy3dshape.pipelines)
    num_inference_steps: int = int(_env("HY3D_STEPS", "30"))
    guidance_scale: float = float(_env("HY3D_GUIDANCE", "5.0"))
    octree_resolution: int = int(_env("HY3D_OCTREE", "256"))
    mc_algo: str = _env("HY3D_MC_ALGO", "mc")
    num_chunks: int = int(_env("HY3D_VAE_CHUNKS", "8000"))
    tex_resolution: int = int(_env("HY3D_TEX_RES", "512"))
    tex_num_views: int = int(_env("HY3D_TEX_VIEWS", "6"))

    # Concurrency: "auto" (default) picks the GPU slot count at runtime from
    # measured VRAM (floor(total_vram_gb / jobs_vram_gb)). A numeric env value
    # pins a fixed cap (min 1). Never exceeds what one checkpoint needs.
    max_active_jobs: Optional[int] = field(
        default_factory=lambda: _active_cap_setup()
    )
    jobs_vram_gb: float = float(_env("HY3D_JOBS_VRAM_GB", "9.0"))

    # Storage
    output_dir: Path = field(
        default_factory=lambda: _path("HY3D_OUTPUT_DIR", "outputs/glb")
    )
    job_dir: Path = field(default_factory=lambda: _path("HY3D_JOB_DIR", "outputs/jobs"))
    cache_dir: Path = field(
        default_factory=lambda: _path("HF_HOME", "~/.cache/huggingface")
    )

    # Model weights
    model_repo: str = _model_setup()["repo"]
    shape_model: str = _model_setup()["shape"]
    shape_subfolder: str = _model_setup()["shape_subfolder"]
    paint_model: str = _model_setup()["paint"]
    cpu_device: str = _model_setup()["dev_id"]

    # Diffusion workers: used for a few extra low-cost passes only.
    worker_threads: int = int(_env("HY3D_WORKERS", "2"))

    @property
    def vram(self) -> dict[str, float]:
        return _bank_setup()

    @property
    def outputs(self) -> dict[str, Path]:
        return {"glb": self.output_dir, "jobs": self.job_dir}


settings = Settings()