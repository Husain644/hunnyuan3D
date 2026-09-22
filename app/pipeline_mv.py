"""Hunyuan3D-2mv adapter: 1-4 view (front/back/left/right) shape generation.

The Hunyuan3D-2mv checkpoint (``tencent/Hunyuan3D-2mv``,
``hunyuan3d-dit-v2-mv`` subfolder) is trained with a multi-view DINO
conditioner (``DinoImageEncoderMV`` + ``MVImageProcessorV2``). The v2.1
source repo at ``HY3D_REPO_DIR`` ships *identical* class names under the
``hy3dshape.*`` namespace, while the 2mv ``config.yaml`` references the
older ``hy3dgen.shapegen.*`` package name. This adapter bridges the gap:

* installs a ``hy3dgen.shapegen.*`` module alias pointing at the
  installed ``hy3dshape.*`` classes, so ``smart_load_model`` can
  instantiate the 2mv config unchanged;
* loads the 2mv weights via the same v2.1
  ``Hunyuan3DDiTFlowMatchingPipeline`` class (which already implements
  view-idx ordering: front=0, left=1, back=2, right=3);
* feeds the views dict straight into the pipeline.

The 2.1 (single-view) pipeline in :mod:`.pipeline` is left untouched --
this is an independent, separately-constructed adapter.
"""
from __future__ import annotations

import io
import logging
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .pipeline import (
    Hunyuan3DError,
    PipelineResult,
    _decode_base64,
    _rand_suffix,
    _repo_dirs,
    _square,
    _torch,
)

logger = logging.getLogger("hy3d.mv")


# --------------------------------------------------------------------------
# namespace shim: hy3dgen.shapegen.* -> hy3dshape.*
# --------------------------------------------------------------------------
def _install_hy3dgen_alias(backend_root: Optional[Path]) -> bool:
    """Register ``hy3dgen.shapegen.*`` modules that re-export the v2.1
    ``hy3dshape`` classes so the 2mv ``config.yaml`` targets resolve.

    Returns True on success. No-op (False) if the hy3dshape package is not
    importable from the resolved backend repo.
    """
    alias = "hy3dgen.shapegen"
    if alias in sys.modules:
        return True
    if backend_root is None:
        shape_dir = None
    else:
        shape_dir = backend_root / "hy3dshape"
    for d in (shape_dir,):
        if d is not None and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    try:
        import hy3dshape  # noqa: F401
        import hy3dshape.models
        import hy3dshape.preprocessors
        import hy3dshape.schedulers
        import hy3dshape.pipelines
        import hy3dshape.utils.utils
    except Exception as exc:  # noqa: BLE001
        logger.warning("hy3dshape alias install failed: %s", exc)
        return False

    import types

    hy3dgen_pkg = types.ModuleType("hy3dgen")
    hy3dgen_pkg.__path__ = []  # namespace package
    sys.modules["hy3dgen"] = hy3dgen_pkg

    shapegen_pkg = types.ModuleType(alias)
    shapegen_pkg.__path__ = []
    sys.modules[alias] = shapegen_pkg

    # Re-export every submodule referenced by the 2mv config under the
    # hy3dgen.shapegen namespace.
    sys.modules[f"{alias}.models"] = sys.modules.get("hy3dshape.models")
    sys.modules[f"{alias}.preprocessors"] = sys.modules.get("hy3dshape.preprocessors")
    sys.modules[f"{alias}.schedulers"] = sys.modules.get("hy3dshape.schedulers")
    sys.modules[f"{alias}.pipelines"] = sys.modules.get("hy3dshape.pipelines")
    sys.modules[f"{alias}.utils.utils"] = sys.modules.get("hy3dshape.utils.utils")
    sys.modules[f"{alias}"] = shapegen_pkg
    logger.info("Installed hy3dgen.shapegen.* alias -> hy3dshape.*")
    return True


@dataclass
class _MVBackend:
    root: Optional[Path] = None
    shape_cls: Any = None


_BACKEND: _MVBackend = _MVBackend()


def _init_mv_backend() -> _MVBackend:
    """Locate the Hunyuan3D-2.1 repo and pull the 2mv weights from it."""
    if _BACKEND.shape_cls is not None:
        return _BACKEND
    roots = _repo_dirs()
    last_exc = None
    for root in roots:
        try:
            shape_dir = root / "hy3dshape"
            paint_dir = root / "hy3dpaint"
            for d in (shape_dir, paint_dir):
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

            _BACKEND.root = root
            _BACKEND.shape_cls = Hunyuan3DDiTFlowMatchingPipeline
            logger.info("MV backend from %s", root)
            return _BACKEND
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    if last_exc is None:
        last_exc = RuntimeError("No Hunyuan3D-2.1 repo found (HY3D_REPO_DIR)")
    raise Hunyuan3DError(
        "2mv backend requires the Hunyuan3D-2.1 source (hy3dshape/).\n"
        + "".join(traceback.format_exception_only(type(last_exc), last_exc))
    )


def _compose_views(
    views: Dict[str, Union[Image.Image, str, np.ndarray]]
) -> Dict[str, Image.Image]:
    """Normalize the front/back/left/right inputs to PIL images."""
    out: Dict[str, Image.Image] = {}
    for tag in ("front", "back", "left", "right"):
        v = views.get(tag)
        if v is None:
            continue
        if isinstance(v, str):
            img = _decode_base64(v) if "," in v or v.startswith("data:") else Image.open(v)
            img = img.convert("RGB")
        elif isinstance(v, np.ndarray):
            img = Image.fromarray(v)
        elif isinstance(v, Image.Image):
            img = v
        else:
            raise Hunyuan3DError(f"Unsupported {tag} view type {type(v)!r}")
        out[tag] = img
    if not out:
        raise Hunyuan3DError("2mv requires at least one view image (front recommended).")
    return out


class Hunyuan3DMVPipeline:
    """Multi-view (1-4 view) shape generation using the 2mv checkpoint.

    Texture is optional and runs through the same v2.1 paint pipeline;
    when disabled the adapter produces the untextured mesh GLB.
    """

    MV_REPO = "tencent/Hunyuan3D-2mv"
    MV_SUBFOLDER = "hunyuan3d-dit-v2-mv"
    view_order = ("front", "left", "back", "right")

    def __init__(self, settings, gpu, *, repo: Optional[str] = None,
                 subfolder: Optional[str] = None) -> None:
        self.s = settings
        self.gpu = gpu
        self.mv_repo = repo or self.MV_REPO
        self.mv_subfolder = subfolder or self.MV_SUBFOLDER
        self._load_lock = threading.Lock()
        self._shape: Any = None
        self.stats = {"jobs": 0, "shape_runs": 0}

    @property
    def backend(self) -> _MVBackend:
        return _init_mv_backend()

    def _build_shape(self) -> Any:
        if self._shape is not None:
            return self._shape
        torch = _torch()
        if torch is None or not torch.cuda.is_available():
            raise Hunyuan3DError("2mv shape stage requires CUDA.")
        if not _install_hy3dgen_alias(self.backend.root):
            raise Hunyuan3DError("Could not bridge hy3dgen.shapegen -> hy3dshape.")
        logger.info(
            "Loading 2mv shape pipeline (%s/%s) on cuda fp16 ...",
            self.mv_repo, self.mv_subfolder,
        )
        self._shape = self.backend.shape_cls.from_pretrained(
            self.mv_repo,
            subfolder=self.mv_subfolder,
            device="cuda",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        return self._shape

    def run(
        self,
        views: Dict[str, Union[Image.Image, str, np.ndarray]],
        *,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        octree_resolution: Optional[int] = None,
        num_chunks: Optional[int] = None,
        return_bytes: bool = False,
        progress=None,
        metadata: Optional[dict] = None,  # -> GLB JSON "extras"
    ) -> PipelineResult:
        import trimesh  # noqa: F401

        steps = num_inference_steps or self.s.num_inference_steps
        guidance = guidance_scale or self.s.guidance_scale
        octree = octree_resolution or self.s.octree_resolution
        chunks = num_chunks or self.gpu.recommended_chunk()

        def report(stage, pct):
            if progress:
                progress(stage, pct)

        views = _compose_views(views)
        # Align key order to the fixed view-idx (front=0,left=1,back=2,right=3).
        view_dict = {k: views[k] for k in self.view_order if k in views}
        report("preprocess", 0.02)
        view_dict = {k: _square(v) for k, v in view_dict.items()}

        report("shape", 0.05)
        shape_pipe = self._shape or self._build_shape()
        self.gpu.mark_resident("shape", type(shape_pipe).__name__)
        mesh = None
        try:
            out = shape_pipe(
                image=view_dict,
                num_inference_steps=steps,
                guidance_scale=guidance,
                octree_resolution=octree,
                num_chunks=chunks,
            )
            report("shape", 0.75)
            # forward returns [] for empty / single trimesh
            mesh = out[0] if isinstance(out, list) and out else out
            if isinstance(mesh, (list, tuple)):
                mesh = mesh[0]
            self.stats["shape_runs"] += 1
        except Exception as exc:  # noqa: BLE001
            raise Hunyuan3DError(f"2mv shape stage failed: {exc}") from exc
        finally:
            self.gpu.force_offload(shape_pipe)
            self._shape = None
            self.gpu.empty_cache()

        if mesh is None:
            raise Hunyuan3DError("2mv shape stage returned no mesh.")

        report("export", 0.95)
        mesh = self._postprocess(mesh)
        if metadata is not None:
            try:
                if hasattr(mesh, "metadata"):
                    extras = dict(getattr(mesh, "metadata", {}).get("extras", {}) or {})
                    extras.update(metadata)
                    mesh.metadata["extras"] = extras
            except Exception:  # noqa: BLE001
                logger.debug("metadata attach failed", exc_info=True)
        out_path = self.s.output_dir / f"mesh_{_rand_suffix()}.glb"
        glb_bytes: Optional[bytes] = None
        if hasattr(mesh, "export"):
            if return_bytes:
                buf = io.BytesIO()
                try:
                    mesh.export(file_obj=buf, file_type="glb")
                except TypeError:
                    mesh.export(buf)
                glb_bytes = buf.getvalue()
            else:
                try:
                    mesh.export(str(out_path))
                except TypeError:
                    mesh.export(str(out_path))
        else:
            verts, faces = getattr(mesh, "vertices", None), getattr(mesh, "faces", None)
            if verts is None or faces is None:
                raise Hunyuan3DError("2mv pipeline returned a mesh without export()/verts/faces.")
            md = trimesh.Trimesh(vertices=verts, faces=faces)
            if return_bytes:
                glb_bytes = md.export(file_obj=io.BytesIO(), file_type="glb")
            else:
                md.export(str(out_path))

        report("export", 1.0)
        self.stats["jobs"] += 1
        self.gpu.empty_cache()
        return PipelineResult(
            glb_path=out_path,
            mesh_faces=int(getattr(mesh, "faces", None).shape[0]) if getattr(mesh, "faces", None) is not None else 0,
            texture_written=False,
            glb_bytes=glb_bytes,
        )

    def _postprocess(self, mesh: Any) -> Any:
        return mesh