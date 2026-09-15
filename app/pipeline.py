"""Hunyuan3D 2.1 image-to-3D pipeline with T4-friendly sequential offload.

Works with the reference layout from the Hunyuan3D-2.1 repo:

    Hy3DGen/
      hy3dshape/pipelines.py            -> Hunyuan3DDiTFlowMatchingPipeline
      hy3dpaint/textureGenPipeline.py   -> Hunyuan3DPaintPipeline, config
      hy3dpaint/custom_rasterizer/      -> compiled C++ rasterizer

Every heavy component is loaded, run, then moved back to CPU before the next
component is loaded, so peak VRAM for v2.1 (shape 10 GB + paint 21 GB) stays
inside the 14.6 GB usable capacity of a T4. Falls back to the v2.0 ``hy3dgen``
uniform API if only that package is installed.
"""
from __future__ import annotations

import importlib
import io
import logging
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from .config import Settings
from .gpu_manager import GpuManager

logger = logging.getLogger("hy3d.pipeline")


@dataclass
class PipelineResult:
    glb_path: Path
    mesh_faces: int = 0
    texture_written: bool = False
    elapsed_s: float = 0.0


class Hunyuan3DError(RuntimeError):
    """Fatal pipeline failure surfaced to the job layer."""


# --------------------------------------------------------------------------
# Version / package detection
# --------------------------------------------------------------------------
def _repo_dirs() -> list[Path]:
    """Locate hy3dshape/ + hy3dpaint/ (v2.1 repo).

    Searches the working directory, the service parent, and every import
    path (PYTHONPATH / venv site-packages) for a directory that contains
    both sibling packages.
    """
    hints: list[Path] = []
    hints.append(Path.cwd())
    env = sys.modules.get(__name__).__package__ or ""
    if env:
        hints.append(Path(__file__).resolve().parents[3])
    for p in list(sys.path):
        if p and Path(p).exists():
            hints.append(Path(p))
    out: list[Path] = []
    seen: set[Path] = set()
    for root in hints:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        sh = root / "hy3dshape"
        pt = root / "hy3dpaint"
        if sh.is_dir() and pt.is_dir():
            out.append(root)
    return out


@dataclass
class _Backend:
    version: str = "2.1"
    sys_path_root: Optional[Path] = None
    shape_cls: Any = None
    paint_cls: Any = None
    texgen_cfg: Any = None


BACKEND = _Backend()


def _init_backend() -> _Backend:
    """Resolve and import the Hunyuan3D python bindings (2.1 first, 2.0 fallback)."""
    if BACKEND.shape_cls is not None:
        return BACKEND

    roots = _repo_dirs()
    for root in roots:
        sys.path.insert(0, str(root))
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402
            from hy3dpaint.textureGenPipeline import (  # noqa: E402
                Hunyuan3DPaintPipeline,
                Hunyuan3DPaintConfig,
            )

            BACKEND.version = "2.1"
            BACKEND.sys_path_root = root
            BACKEND.shape_cls = Hunyuan3DDiTFlowMatchingPipeline
            BACKEND.paint_cls = Hunyuan3DPaintPipeline
            BACKEND.texgen_cfg = Hunyuan3DPaintConfig
            logger.info("Using Hunyuan3D v2.1 bindings from %s", root)
            return BACKEND
        except Exception as exc:  # noqa: BLE001
            logger.debug("v2.1 import failed (%s); trying v2.0", exc)
            sys.path = [p for p in sys.path if str(root) not in str(p)]

    # v2.0 fallback (hy3dgen monolithic package)
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402
        from hy3dgen.texgen import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig  # noqa: E402

        BACKEND.version = "2.0"
        BACKEND.shape_cls = Hunyuan3DDiTFlowMatchingPipeline
        BACKEND.paint_cls = Hunyuan3DPaintPipeline
        BACKEND.texgen_cfg = Hunyuan3DTexGenConfig
        logger.info("Using Hunyuan3D v2.0 (hy3dgen) bindings")
        return BACKEND
    except Exception as exc:  # noqa: BLE001
        raise Hunyuan3DError(
            "Neither Hunyuan3D-2.1 (hy3dshape/hy3dpaint) nor v2.0 (hy3dgen) "
            "bindings are importable. Clone Tencent-Hunyuan/Hunyuan3D-2.1 next "
            "to this service and build hy3dpaint/custom_rasterizer.\n"
            + "".join(traceback.format_exception_only(type(exc), exc))
        ) from exc


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def _decode_base64(data: str) -> Image.Image:
    try:
        import base64

        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data, validate=True)
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception as exc:  # noqa: BLE001
        raise Hunyuan3DError(f"Invalid base64 image payload: {exc}") from exc


def _square(image: Image.Image, size: int = 1024) -> Image.Image:
    """Center-crop to square + downscale, as the shape model expects."""
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if side > size:
        image = image.resize((size, size), Image.LANCZOS)
    return image.convert("RGB")


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
class Hunyuan3DPipeline:
    """Stateful orchestrator of the two-stage shape + texture pipeline.

    Threading model: a single worker executes the pipeline under the stage
    locks from :class:`GpuManager`. External callers only invoke ``run`` via
    the job worker, so no internal reentrancy protection is needed beyond the
    GPU stage locks.
    """

    def __init__(self, settings: Settings, gpu: GpuManager) -> None:
        self.s = settings
        self.gpu = gpu
        self._load_lock = threading.Lock()

        # Lazily-built stage pipelines, kept resident only while a job runs.
        self._rembg: Any = None
        self._shape: Any = None
        self._paint: Any = None
        self._backend: Optional[_Backend] = None

        self.stats = {"jobs": 0, "shape_runs": 0, "tex_runs": 0}

    @property
    def backend(self) -> _Backend:
        """Resolve Hunyuan3D bindings lazily so the server can boot in a
        degraded state on machines that don't have the model installed."""
        if self._backend is None:
            self._backend = _init_backend()
        return self._backend

    # -- stage construction -------------------------------------------------
    def _build_rembg(self) -> Any:
        if self._rembg is not None:
            return self._rembg
        # Candidate locations across v2.0 (hy3dgen) and v2.1 (hy3dshape) repos.
        for mod_path in ("hy3dgen.rembg", "hy3dshape.rembg", "hy3dshape.rembg_util",
                         "hy3dshape.removeBackground"):
            try:
                mod = importlib.import_module(mod_path)
                remover_cls = getattr(mod, "BackgroundRemover", None)
                if remover_cls is None:
                    continue
                self._rembg = remover_cls()
                if remove_torch := _torch():
                    if remove_torch.cuda.is_available():
                        try:
                            self._rembg.to("cuda")
                        except Exception:  # noqa: BLE001
                            pass
                return self._rembg
            except Exception:  # noqa: BLE001
                continue
        raise Hunyuan3DError(
            "Background removal requested but no BackgroundRemover found; "
            "set HY3D_ENABLE_REMBG=0 or install rembg + a Hunyuan3D binding."
        )

    def _build_shape(self) -> Any:
        if self._shape is not None:
            return self._shape
        logger.info("Loading shape pipeline %s ...", self.s.shape_model)
        self._shape = self.backend.shape_cls.from_pretrained(
            self.s.shape_model,
            cache_dir=str(self.s.cache_dir),
        )
        return self._shape

    def _build_paint(self) -> Any:
        if self._paint is not None:
            return self._paint
        cfg_cls = self.backend.texgen_cfg
        cfg = cfg_cls(
            max_num_view=self.s.tex_num_views,
            resolution=self.s.tex_resolution,
        )
        logger.info("Loading paint pipeline (tex res=%s) ...", self.s.tex_resolution)
        self._paint = self.backend.paint_cls(cfg)
        return self._paint

    # -- lifecycle ----------------------------------------------------------
    def _load_stage(self, stage: str, builder) -> Any:
        """Load one stage under its GPU budget lock, freeing prior stages."""
        for other in ("shape", "tex"):
            if other != stage and getattr(self, f"_{other}", None) is not None:
                model = getattr(self, f"_{other}")
                self.gpu.force_offload(model)
                setattr(self, f"_{other}", None)
        with self.gpu.stage_lock(stage):
            self.gpu.empty_cache()
            obj = builder()
            self.gpu.mark_resident(stage, type(obj).__name__)
            self.gpu.empty_cache()
            return obj

    def _unload_all(self) -> None:
        for stage in ("shape", "tex"):
            model = getattr(self, f"_{stage}", None)
            if model is not None:
                self.gpu.force_offload(model)
                setattr(self, f"_{stage}", None)
        if self._rembg is not None:
            self.gpu.force_offload(self._rembg)
            self._rembg = None
        self.gpu.empty_cache()

    # -- preprocessing ------------------------------------------------------
    def _preprocess(self, image: Image.Image) -> Image.Image:
        image = _square(image)
        if self.s.enable_rembg:
            try:
                rembg = self._load_stage("rembg", self._build_rembg)
                image = image.convert("RGBA")
                image = rembg(image)
            except Hunyuan3DError as exc:
                logger.warning("%s; falling back to raw image", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Background removal failed (%s); using raw image", exc)
        return _square(image)

    # -- main entry ---------------------------------------------------------
    def run(
        self,
        image: Union[Image.Image, str, np.ndarray],
        *,
        enable_texture: Optional[bool] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        octree_resolution: Optional[int] = None,
        tex_resolution: Optional[int] = None,
        progress: Optional[callable] = None,  # fn(stage: str, pct: float)
    ) -> PipelineResult:
        use_texture = self.s.enable_texture if enable_texture is None else enable_texture
        steps = num_inference_steps if num_inference_steps is not None else self.s.num_inference_steps
        guidance = guidance_scale if guidance_scale is not None else self.s.guidance_scale
        octree = octree_resolution if octree_resolution is not None else self.s.octree_resolution
        tex_res = tex_resolution if tex_resolution is not None else self.s.tex_resolution
        chunks = self.gpu.recommended_chunk()

        def report(stage: str, pct: float) -> None:
            if progress:
                progress(stage, pct)

        try:
            import trimesh  # noqa: F401  (required for GLB export path)

            _ = trimesh  # bound for the vertices/faces fallback below
        except ImportError:  # pragma: no cover
            raise Hunyuan3DError("trimesh is required for GLB export.")

        report("preprocess", 0.02)
        if isinstance(image, str):
            image = _decode_base64(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = self._preprocess(image)

        # ---- Stage 1: shape ---------------------------------------------
        report("shape", 0.05)
        shape_pipe = self._load_stage("shape", self._build_shape)
        try:
            offload = getattr(shape_pipe, "enable_model_cpu_offload", None)
            if callable(offload):
                offload()
                logger.debug("shape pipeline: model cpu offload enabled")
            self.gpu.mark_resident("shape", type(shape_pipe).__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning("enable_model_cpu_offload on shape failed: %s", exc)

        mesh = None
        try:
            out = shape_pipe(
                image=image,
                num_inference_steps=steps,
                guidance_scale=guidance,
                octree_resolution=octree,
                mc_algo=self.s.mc_algo,
                num_chunks=chunks,
            )
            report("shape", 0.6)
            mesh = out[0] if isinstance(out, (list, tuple)) else out
            self.stats["shape_runs"] += 1
        except Exception as exc:  # noqa: BLE001
            raise Hunyuan3DError(f"Shape stage failed: {exc}") from exc
        finally:
            # Free the DiT before pulling paint into VRAM.
            self.gpu.force_offload(shape_pipe)
            self._shape = None
            self.gpu.empty_cache()

        if mesh is None:
            raise Hunyuan3DError("Shape stage returned no mesh.")

        if not hasattr(mesh, "export"):
            # v2.x may return a raw trimesh-lite wrapper
            logger.warning("Mesh object has no .export; has=%s", dir(mesh)[:5])

        # ---- Stage 2: texture (optional) ---------------------------------
        textured = False
        if use_texture:
            report("texture", 0.6)
            paint_pipe = self._load_stage("tex", self._build_paint)
            try:
                offload = getattr(paint_pipe, "enable_model_cpu_offload", None)
                if callable(offload):
                    offload()
                # Sub-models w/ independent pipelines (v2.1) may each offload.
                for m in getattr(paint_pipe, "models", {}).values():
                    sub = getattr(m, "pipeline", None)
                    off = getattr(sub, "enable_model_cpu_offload", None)
                    if callable(off):
                        try:
                            off()
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("enable_model_cpu_offload on paint failed: %s", exc)

            try:
                textured_mesh = paint_pipe(
                    mesh,
                    image_path=None,
                    image=image,          # v2.1 convenience; harmless if unused
                    max_num_view=self.s.tex_num_views,
                    resolution=tex_res,
                )
                report("texture", 0.95)
                textured = True
                self.stats["tex_runs"] += 1
            except TypeError:
                # v2.0 signature: paint_pipe(mesh, image='...')
                try:
                    textured_mesh = paint_pipe(mesh, image=image)
                    textured = True
                    self.stats["tex_runs"] += 1
                except Exception as exc:  # noqa: BLE001
                    raise Hunyuan3DError(f"Texture stage failed: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                raise Hunyuan3DError(f"Texture stage failed: {exc}") from exc
            finally:
                self.gpu.force_offload(paint_pipe)
                self._paint = None
                self.gpu.empty_cache()
            mesh = textured_mesh
        else:
            report("texture", 0.95)  # skipped

        # ---- post-process + export ---------------------------------------
        report("export", 0.96)
        mesh = self._postprocess(mesh)

        job_suffix = _rand_suffix()
        out_path = self.s.output_dir / f"mesh_{job_suffix}.glb"
        if hasattr(mesh, "export"):
            try:
                mesh.export(str(out_path))
            except TypeError:
                mesh.export(str(out_path), include_normals=textured or True)
        else:
            # v2.1 shape may return a Surface object with vertices/faces.
            verts, faces = getattr(mesh, "vertices", None), getattr(mesh, "faces", None)
            if verts is None or faces is None:
                raise Hunyuan3DError("Pipeline returned a mesh without export() or vertices/faces.")
            md = trimesh.Trimesh(vertices=verts, faces=faces)
            md.export(str(out_path))

        report("export", 1.0)
        self.stats["jobs"] += 1
        self._unload_all()
        return PipelineResult(
            glb_path=out_path,
            mesh_faces=int(getattr(mesh, "faces", None).shape[0]) if getattr(mesh, "faces", None) is not None else 0,
            texture_written=textured,
        )

    # -- mesh cleanup -------------------------------------------------------
    def _postprocess(self, mesh: Any) -> Any:
        """Apply light mesh hygiene. Kept import-lenient: if pymeshlab or the
        hy3dgen postprocessors are missing, the raw mesh is returned."""
        if self.s.enable_texture is False or not hasattr(mesh, "export"):
            return mesh
        try:
            from hy3dgen.shapegen import (  # type: ignore  # noqa: F401
                DegenerateFaceRemover,
                FloaterRemover,
            )

            def _apply(remover):
                try:
                    return remover()(mesh)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("postprocess %s skipped: %s", type(remover).__name__, exc)
                    return mesh

            mesh = _apply(FloaterRemover)
            mesh = _apply(DegenerateFaceRemover)
        except Exception:  # noqa: BLE001
            pass
        return mesh


def _torch():
    try:
        import torch  # type: ignore

        return torch
    except ImportError:  # pragma: no cover
        return None


def _rand_suffix() -> str:
    import time

    return f"{int(time.time()*1000):x}{threading.get_ident():x}"